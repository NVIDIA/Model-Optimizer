# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
from _test_utils.torch.diffusers_models import get_tiny_dit, get_tiny_flux, get_tiny_unet
from safetensors import safe_open

import modelopt.torch.quantization as mtq
from modelopt.torch.export.diffusers_utils import generate_diffusion_dummy_inputs
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint


def _load_config(config_path):
    with open(config_path) as file:
        return json.load(file)


def _calib_with_dummy_inputs(m):
    param = next(m.parameters())
    dummy_inputs = generate_diffusion_dummy_inputs(m, param.device, param.dtype)
    assert dummy_inputs is not None
    m(**dummy_inputs)


@pytest.mark.parametrize("model_factory", [get_tiny_unet, get_tiny_dit, get_tiny_flux])
@pytest.mark.parametrize(
    ("config_id", "quant_cfg"),
    [
        ("int8", mtq.INT8_DEFAULT_CFG),
        ("int8_smoothquant", mtq.INT8_SMOOTHQUANT_CFG),
        ("fp8", mtq.FP8_DEFAULT_CFG),
    ],
)
def test_export_diffusers_real_quantized(tmp_path, model_factory, config_id, quant_cfg):
    model = model_factory()
    export_dir = tmp_path / f"export_{type(model).__name__}_{config_id}_real_quant"

    def _calib_fn(m):
        param = next(m.parameters())
        dummy_inputs = generate_diffusion_dummy_inputs(m, param.device, param.dtype)
        assert dummy_inputs is not None
        m(**dummy_inputs)

    mtq.quantize(model, quant_cfg, forward_loop=_calib_fn)

    export_hf_checkpoint(model, export_dir=export_dir)

    config_path = export_dir / "config.json"
    assert config_path.exists()

    config_data = _load_config(config_path)
    assert "quantization_config" in config_data


def test_export_diffusers_real_quantized_fp4(tmp_path):
    """FP4 export test using get_tiny_dit (the only tiny model with FP4-compatible weight shapes)."""
    model = get_tiny_dit()
    export_dir = tmp_path / "export_DiTTransformer2DModel_fp4_real_quant"

    def _calib_fn(m):
        param = next(m.parameters())
        dummy_inputs = generate_diffusion_dummy_inputs(m, param.device, param.dtype)
        assert dummy_inputs is not None
        m(**dummy_inputs)

    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=_calib_fn)

    export_hf_checkpoint(model, export_dir=export_dir)

    config_path = export_dir / "config.json"
    assert config_path.exists()

    config_data = _load_config(config_path)
    assert "quantization_config" in config_data


def test_export_diffusers_sharded_default_no_header_metadata(tmp_path):
    """A default (non-opt-in) sharded FP8 export succeeds and writes no header metadata.

    Regression test for the FLUX FP8 export crash (NotImplementedError on sharded
    safetensors). A tiny max_shard_size forces the tiny model to split into multiple
    shards (+ index.json), reproducing the large-model path. With the header quant
    metadata off by default, post-processing is a no-op: the export must succeed and
    leave a clean safetensors header (the ComfyUI metadata is opt-in).
    """
    model = get_tiny_flux()
    export_dir = tmp_path / "export_flux_fp8_sharded_default"

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=_calib_with_dummy_inputs)

    # Tiny shard size forces sharding even for this tiny model.
    export_hf_checkpoint(model, export_dir=export_dir, max_shard_size="1KB")

    assert list(export_dir.glob("*.safetensors.index.json")), (
        "expected a sharded export (index.json) with a tiny max_shard_size"
    )
    shard_files = sorted(export_dir.glob("*.safetensors"))
    assert len(shard_files) >= 2, "expected the model to split across multiple shards"

    for shard in shard_files:
        with safe_open(str(shard), framework="pt") as f:
            md = f.metadata() or {}
        assert "quantization_config" not in md, f"unexpected header metadata in {shard.name}"
        assert "_quantization_metadata" not in md, f"unexpected header metadata in {shard.name}"
