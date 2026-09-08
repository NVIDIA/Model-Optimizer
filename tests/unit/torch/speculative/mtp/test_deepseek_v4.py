# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Focused codec and checkpoint-discovery coverage for native DSV4 MTPs."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

from modelopt.torch.speculative.mtp import deepseek_v4
from modelopt.torch.speculative.mtp.deepseek_v4 import (
    _DIRECT_STATE_KEYS,
    _MTP_PREFIX,
    DeepSeekV4MTPAdapter,
    NativeMTPBoostModel,
    _checkpoint_layout,
    _dsv4_hadamard_fallback,
    _expected_mtp_keys,
    decode_dsv4_fp8_weight,
    decode_dsv4_mxfp4_weight,
    encode_dsv4_fp8_weight,
    encode_dsv4_mxfp4_weight,
)


def test_dsv4_hadamard_fallback_matches_normalized_sylvester_transform():
    """The dependency-free vendor fallback preserves its rotation convention."""
    values = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)

    transformed = _dsv4_hadamard_fallback(values)

    assert torch.equal(
        transformed,
        torch.tensor([[5, -1, -2, 0]], dtype=torch.bfloat16),
    )


def test_dsv4_adapter_leaves_nonzero_fsdp_ranks_on_meta(monkeypatch):
    """CPU-efficient FSDP2 loads checkpoint tensors only on rank zero."""

    class FakeModel:
        endpoints_frozen = False

        def freeze_target_endpoints(self):
            self.endpoints_frozen = True

    model = FakeModel()
    captured: dict[str, object] = {}
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(deepseek_v4, "_checkpoint_layout", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        DeepSeekV4MTPAdapter,
        "_transformers_config",
        classmethod(lambda _cls, _layout: (object(), 61)),
    )

    def build(_cls, _config, _layer_index, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(DeepSeekV4MTPAdapter, "_build_empty_model", classmethod(build))
    monkeypatch.setattr(
        DeepSeekV4MTPAdapter,
        "_load_model",
        classmethod(lambda *_args: pytest.fail("nonzero rank read checkpoint tensors")),
    )

    result = DeepSeekV4MTPAdapter.create(
        "unused",
        dtype=torch.bfloat16,
        device=None,
        rollout_steps=2,
        hsm_mode="uniform_layer_sample",
    )

    assert result is model
    assert captured["device"] == "meta"
    assert model.endpoints_frozen


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="requires PyTorch's UE8M0 dtype")
def test_dsv4_codecs_round_trip_and_preserve_mxfp4_midpoint_ties():
    """The export codecs produce decodable native layouts and lower-tie E2M1 codes."""
    fp8_master = torch.linspace(-3, 3, 128 * 128, dtype=torch.bfloat16).reshape(128, 128)
    fp8_weight, fp8_scale = encode_dsv4_fp8_weight(fp8_master)
    fp8_decoded = decode_dsv4_fp8_weight(fp8_weight, fp8_scale)
    assert fp8_weight.dtype == torch.float8_e4m3fn
    assert fp8_decoded.shape == fp8_master.shape
    assert torch.isfinite(fp8_decoded).all()

    # The trailing 6 fixes the block exponent at zero.  Exact values at each
    # E2M1 midpoint must retain the lower representable value.
    mxfp4_master = torch.tensor(
        [[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] + [6.0] * 24],
        dtype=torch.bfloat16,
    )
    packed, scale = encode_dsv4_mxfp4_weight(mxfp4_master)
    decoded = decode_dsv4_mxfp4_weight(packed, scale)
    assert packed.view(torch.uint8)[0, :4].tolist() == [0x10, 0x32, 0x54, 0x76]
    assert torch.equal(
        decoded[0, :8], torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.bfloat16)
    )


def test_dsv4_adapter_requires_exact_mtp_coverage(tmp_path):
    """Auto detection is narrow, while strict loading rejects missing MTP tensors."""
    config = {
        "model_type": "deepseek_v4",
        "n_routed_experts": 1,
        "num_nextn_predict_layers": 1,
    }
    expected = _expected_mtp_keys(config)
    weight_map = dict.fromkeys(expected, "model.safetensors")
    weight_map.update({"embed.weight": "model.safetensors", "head.weight": "model.safetensors"})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    assert DeepSeekV4MTPAdapter.supports(tmp_path)
    assert _checkpoint_layout(tmp_path, strict=True).embedding_key == "embed.weight"

    weight_map.pop(next(iter(expected)))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="coverage is not exact"):
        _checkpoint_layout(tmp_path, strict=True)


def _values(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    """Small deterministic BF16 master tensor for a synthetic checkpoint."""
    return (
        torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).add_(offset).div_(64)
    ).to(torch.bfloat16)


def _register_parameter(root: nn.Module, name: str, value: torch.Tensor) -> None:
    """Register a parameter at a dotted state-dict path on a lightweight fake MTP model."""
    module = root
    for component in name.split(".")[:-1]:
        if not hasattr(module, component):
            module.add_module(component, nn.Module())
        module = getattr(module, component)
    module.register_parameter(name.rsplit(".", 1)[-1], nn.Parameter(torch.empty_like(value)))


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="requires PyTorch's UE8M0 dtype")
def test_synthetic_dsv4_mtp_load_and_native_export_round_trip(tmp_path):
    """Strictly load BF16 masters, then regenerate the source FP8/MXFP4 MTP shard."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    source_tensors: dict[str, torch.Tensor] = {}
    expected_state: dict[str, torch.Tensor] = {}
    config = {
        "model_type": "deepseek_v4",
        "n_routed_experts": 1,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False,
    }
    expected_mtp_keys = _expected_mtp_keys(config)

    embedding = _values((4, 128), 1)
    lm_head = _values((4, 128), 2)
    source_tensors["embed.weight"] = embedding
    source_tensors["head.weight"] = lm_head
    expected_state["embedding.weight"] = embedding
    expected_state["lm_head.weight"] = lm_head

    fp8_weight_suffixes = {
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "e_proj.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight",
        "h_proj.weight",
    }
    for offset, (source_suffix, state_name) in enumerate(_DIRECT_STATE_KEYS.items(), start=3):
        source_name = _MTP_PREFIX + source_suffix
        if source_suffix in fp8_weight_suffixes:
            master = _values((128, 128), offset)
            quantized, scale = encode_dsv4_fp8_weight(master)
            source_tensors[source_name] = quantized
            source_tensors[source_name.replace(".weight", ".scale")] = scale
            expected_state[state_name] = decode_dsv4_fp8_weight(quantized, scale)
        else:
            if source_suffix.endswith(".weight"):
                shape = (2, 128) if source_suffix == "ffn.gate.weight" else (128,)
            elif source_suffix.endswith("_fn"):
                shape = (2, 8)
            elif source_suffix.endswith("_scale"):
                shape = (1,)
            else:
                shape = (2,)
            dtype = (
                torch.float32
                if "hc_" in source_suffix or "bias" in source_suffix
                else torch.bfloat16
            )
            source = _values(shape, offset).to(dtype)
            source_tensors[source_name] = source
            expected_state[state_name] = source.to(torch.bfloat16)

    expert_masters = {}
    for offset, projection in enumerate(("w1", "w2", "w3"), start=100):
        master = _values((16, 32), offset)
        quantized, scale = encode_dsv4_mxfp4_weight(master)
        source_name = f"{_MTP_PREFIX}ffn.experts.0.{projection}.weight"
        source_tensors[source_name] = quantized
        source_tensors[source_name.replace(".weight", ".scale")] = scale
        expert_masters[projection] = decode_dsv4_mxfp4_weight(quantized, scale)
    expected_state["mtp.decoder.mlp.experts.gate_up_proj"] = torch.cat(
        (expert_masters["w1"], expert_masters["w3"]), dim=0
    ).unsqueeze(0)
    expected_state["mtp.decoder.mlp.experts.down_proj"] = expert_masters["w2"].unsqueeze(0)

    assert {key for key in source_tensors if key.startswith(_MTP_PREFIX)} == expected_mtp_keys
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source_tensors, "model.safetensors")})
    )
    safetensors_torch.save_file(source_tensors, str(tmp_path / "model.safetensors"))

    layout = _checkpoint_layout(tmp_path, strict=True)
    model = NativeMTPBoostModel.__new__(NativeMTPBoostModel)
    nn.Module.__init__(model)
    for name, value in expected_state.items():
        _register_parameter(model, name, value)

    DeepSeekV4MTPAdapter._load_model(model, layout)
    loaded_state = model.state_dict()
    for name, expected in expected_state.items():
        assert torch.equal(loaded_state[name], expected)
    assert not model.embedding.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert model.mtp.e_proj.weight.requires_grad

    output_dir = tmp_path / "native-export"
    DeepSeekV4MTPAdapter.export(model, tmp_path, output_dir, state_dict=loaded_state)
    with safetensors_torch.safe_open(
        str(output_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as exported:
        fp8_weight = exported.get_tensor(f"{_MTP_PREFIX}e_proj.weight")
        fp8_scale = exported.get_tensor(f"{_MTP_PREFIX}e_proj.scale")
        expert_weight = exported.get_tensor(f"{_MTP_PREFIX}ffn.experts.0.w1.weight")
        expert_scale = exported.get_tensor(f"{_MTP_PREFIX}ffn.experts.0.w1.scale")
    assert fp8_weight.dtype == torch.float8_e4m3fn
    assert expert_weight.dtype == torch.int8
    assert torch.isfinite(decode_dsv4_fp8_weight(fp8_weight, fp8_scale)).all()
    assert torch.isfinite(decode_dsv4_mxfp4_weight(expert_weight, expert_scale)).all()
