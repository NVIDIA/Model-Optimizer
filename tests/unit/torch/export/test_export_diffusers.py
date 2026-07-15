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

import copy
import json

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.diffusers_models import (
    get_tiny_dit,
    get_tiny_flux,
    get_tiny_flux2,
    get_tiny_unet,
)

pytest.importorskip("diffusers")

from safetensors import safe_open
from safetensors.torch import save_file

import modelopt.torch.export.unified_export_hf as unified_export_hf
import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.diffusers_utils import (
    generate_diffusion_dummy_inputs,
    hide_quantizers_from_state_dict,
)
from modelopt.torch.export.unified_export_hf import _postprocess_safetensors, export_hf_checkpoint


def _load_config(config_path):
    with open(config_path) as file:
        return json.load(file)


def _write_sharded_checkpoint(export_dir, shards):
    """Write ``shards`` (list of state-dict chunks) as sharded safetensors + index.json.

    Mimics the layout produced by ``save_pretrained`` when a component is split across
    multiple files because it exceeds ``max_shard_size``.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    total = len(shards)
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        filename = f"diffusion_pytorch_model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(export_dir / filename))
        for key, tensor in shard.items():
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(export_dir / "diffusion_pytorch_model.safetensors.index.json", "w") as file:
        json.dump(index, file)


def _read_safetensors_metadata(path):
    with safe_open(str(path), framework="pt") as file:
        return dict(file.metadata() or {})


@pytest.mark.parametrize(
    "model_factory", [get_tiny_unet, get_tiny_dit, get_tiny_flux, get_tiny_flux2]
)
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


def test_flux2_dummy_inputs_shape():
    """Verify Flux2-specific dummy input shapes: 4-col RoPE ids, no pooled_projections, guidance."""
    model = get_tiny_flux2()
    cfg = model.config
    inputs = generate_diffusion_dummy_inputs(model, torch.device("cpu"), torch.float32)

    assert inputs is not None, "generate_diffusion_dummy_inputs returned None for Flux2"

    # hidden_states: (batch, seq_len, in_channels)
    assert inputs["hidden_states"].shape == (1, 16, cfg.in_channels)

    # encoder_hidden_states: (batch, text_seq_len, joint_attention_dim)
    assert inputs["encoder_hidden_states"].shape == (1, 8, cfg.joint_attention_dim)

    # RoPE ids must have 4 columns (not 3 like Flux1)
    rope_ndim = len(cfg.axes_dims_rope)
    assert rope_ndim == 4
    assert inputs["img_ids"].shape == (16, rope_ndim)
    assert inputs["txt_ids"].shape == (8, rope_ndim)

    # Flux2 must NOT have pooled_projections (unlike Flux1)
    assert "pooled_projections" not in inputs

    # guidance_embeds defaults to True for Flux2
    assert "guidance" in inputs


def test_svdquant_diffusers_export_promotes_clean_keys():
    """Fast CPU check of the diffusers SVDQuant export promotion.

    SVDQuant calibration stores the low-rank factors on ``weight_quantizer`` and the
    smoothing scale on ``input_quantizer``; the diffusers export promotes both to
    clean module-level keys (``svdquant_lora_a/b``, ``pre_quant_scale``) and hides the
    quantizers, so the saved state dict carries no live quantizer tensors. The full
    NVFP4 end-to-end coverage lives in the GPU test
    ``tests/examples/diffusers/test_export_diffusers_hf_ckpt.py`` (``qwen_nvfp4_svdquant``).
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))

    quant_config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    quant_config["algorithm"] = {"method": "svdquant", "lowrank": 8}
    mtq.quantize(model, quant_config, lambda m: m(torch.randn(8, 64)))

    # Calibration populated the quantizer-owned SVDQuant tensors.
    linear = model[0]
    assert linear.weight_quantizer.svdquant_lora_a is not None
    assert linear.weight_quantizer.svdquant_lora_b is not None
    assert getattr(linear.input_quantizer, "_pre_quant_scale", None) is not None

    # Export promotes them to clean module-level keys and hides the quantizers.
    unified_export_hf._promote_quantizer_tensors_to_module(model)
    with hide_quantizers_from_state_dict(model):
        keys = set(model.state_dict().keys())

    assert any(k.endswith(".svdquant_lora_a") for k in keys)
    assert any(k.endswith(".svdquant_lora_b") for k in keys)
    assert any(k.endswith(".pre_quant_scale") for k in keys)
    assert not any("weight_quantizer" in k or "input_quantizer" in k for k in keys), (
        "live quantizer state leaked into the exported state dict"
    )

    # The promotion is undone after export, leaving the live module unchanged.
    unified_export_hf._remove_promoted_quantizer_tensors(model)
    keys_after = set(model.state_dict().keys())
    assert not any(
        k.endswith((".svdquant_lora_a", ".svdquant_lora_b", ".pre_quant_scale")) for k in keys_after
    )


@pytest.mark.parametrize(
    "opt_in_kwargs",
    [
        {"enable_layerwise_quant_metadata": True},
        {"merged_base_safetensor_path": "/tmp/base.safetensors"},
    ],
)
def test_postprocess_sharded_opt_in_raises(tmp_path, opt_in_kwargs):
    """Opting into ComfyUI post-processing on a sharded checkpoint is unsupported.

    Documents the existing limitation (out of scope for this fix). The bug fix is the
    default no-op path (see ``test_postprocess_default_is_noop``); only an explicit
    opt-in reaches this guard.
    """
    export_dir = tmp_path / "sharded_opt_in"
    _write_sharded_checkpoint(
        export_dir,
        [
            {"layer_a.weight": torch.zeros(4, 4), "layer_a.weight_scale": torch.ones(1)},
            {"layer_b.weight": torch.zeros(4, 4), "layer_b.weight_scale": torch.ones(1)},
        ],
    )

    with pytest.raises(NotImplementedError, match="sharded safetensors"):
        _postprocess_safetensors(
            export_dir,
            hf_quant_config={"quant_algo": "FP8"},
            **opt_in_kwargs,
        )


def test_postprocess_single_file_metadata_when_opted_in(tmp_path):
    """With the opt-in flag, a non-sharded export injects quant config + per-layer metadata."""
    export_dir = tmp_path / "single_file"
    export_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"layer_a.weight": torch.zeros(4, 4), "layer_a.weight_scale": torch.ones(1)},
        str(export_dir / "diffusion_pytorch_model.safetensors"),
    )

    _postprocess_safetensors(
        export_dir,
        hf_quant_config={"quant_algo": "FP8"},
        enable_layerwise_quant_metadata=True,
    )

    metadata = _read_safetensors_metadata(export_dir / "diffusion_pytorch_model.safetensors")
    assert "quantization_config" in metadata
    assert json.loads(metadata["_quantization_metadata"])["layers"] == {
        "layer_a": {"format": "fp8"}
    }


def test_postprocess_default_is_noop(tmp_path):
    """By default (no opt-in) nothing is written to the safetensors header.

    The header quant metadata is a single-file deployment (e.g. ComfyUI) feature, so a
    plain export must leave the checkpoint untouched. This no-op default is also what
    keeps a default *sharded* export from reaching the unsupported-sharded path that
    caused the original FP8 FLUX crash.
    """
    export_dir = tmp_path / "default_noop"
    _write_sharded_checkpoint(
        export_dir,
        [
            {"layer_a.weight": torch.zeros(4, 4), "layer_a.weight_scale": torch.ones(1)},
            {"layer_b.weight": torch.zeros(4, 4), "layer_b.weight_scale": torch.ones(1)},
        ],
    )

    # No opt-in kwargs: must not raise (even though sharded) and must inject nothing.
    _postprocess_safetensors(export_dir, hf_quant_config={"quant_algo": "FP8"})

    for shard in sorted(export_dir.glob("*.safetensors")):
        metadata = _read_safetensors_metadata(shard)
        assert "quantization_config" not in metadata
        assert "_quantization_metadata" not in metadata
