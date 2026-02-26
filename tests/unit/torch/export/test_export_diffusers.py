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

pytest.importorskip("diffusers")

import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.diffusers_utils import generate_diffusion_dummy_inputs
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint


def _load_config(config_path):
    with open(config_path) as file:
        return json.load(file)


@pytest.mark.parametrize("model_factory", [get_tiny_unet, get_tiny_dit, get_tiny_flux])
def test_export_diffusers_models_non_quantized(tmp_path, model_factory):
    model = model_factory()
    export_dir = tmp_path / f"export_{type(model).__name__}"

    export_hf_checkpoint(model, export_dir=export_dir)

    config_path = export_dir / "config.json"
    assert config_path.exists()

    config_data = _load_config(config_path)
    assert "quantization_config" not in config_data


def test_export_diffusers_unet_quantized_matches_llm_config(tmp_path, monkeypatch):
    model = get_tiny_unet()
    export_dir = tmp_path / "export_unet_quant"

    import modelopt.torch.export.unified_export_hf as unified_export_hf

    monkeypatch.setattr(unified_export_hf, "has_quantized_modules", lambda *_: True)

    fuse_calls = {"count": 0}
    process_calls = {"count": 0}

    def _fuse_stub(*_args, **_kwargs):
        fuse_calls["count"] += 1

    def _process_stub(*_args, **_kwargs):
        process_calls["count"] += 1

    monkeypatch.setattr(unified_export_hf, "_fuse_qkv_linears_diffusion", _fuse_stub)
    monkeypatch.setattr(unified_export_hf, "_process_quantized_modules", _process_stub)

    dummy_quant_config = {
        "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
        "producer": {"name": "modelopt", "version": "0.0"},
    }
    monkeypatch.setattr(
        unified_export_hf, "get_quant_config", lambda *_args, **_kwargs: dummy_quant_config
    )

    export_hf_checkpoint(model, export_dir=export_dir)

    assert fuse_calls["count"] == 1
    assert process_calls["count"] == 1

    config_path = export_dir / "config.json"
    assert config_path.exists()

    config_data = _load_config(config_path)
    assert "quantization_config" in config_data
    assert config_data["quantization_config"] == convert_hf_quant_config_format(dummy_quant_config)


@pytest.mark.parametrize("model_factory", [get_tiny_unet, get_tiny_dit, get_tiny_flux])
def test_export_diffusers_real_quantized(tmp_path, model_factory):
    model = model_factory()
    export_dir = tmp_path / f"export_{type(model).__name__}_real_quant"

    def _calib_fn(m):
        param = next(m.parameters())
        dummy_inputs = generate_diffusion_dummy_inputs(m, param.device, param.dtype)
        assert dummy_inputs is not None
        m(**dummy_inputs)

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=_calib_fn)

    export_hf_checkpoint(model, export_dir=export_dir)

    config_path = export_dir / "config.json"
    assert config_path.exists()

    config_data = _load_config(config_path)
    assert "quantization_config" in config_data
