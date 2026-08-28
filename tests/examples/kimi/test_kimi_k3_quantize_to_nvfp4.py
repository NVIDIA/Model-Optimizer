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

"""Integration tests for the calibration-free Kimi-K3 checkpoint converter."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.quantization.qtensor import FP8QTensor, MXFP4QTensor, MXFP8QTensor

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "examples" / "kimi" / "kimi_k3" / "quantize_to_nvfp4.py"
)
_SPEC = importlib.util.spec_from_file_location("kimi_k3_quantize_to_nvfp4", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
k3_cast = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(k3_cast)


def test_published_recipe_resolves_to_streaming_conversion_settings():
    settings = k3_cast._conversion_settings_from_recipe(k3_cast._PUBLISHED_RECIPE)

    assert settings == {
        "cast_mxfp4_to_nvfp4": True,
        "attn_fp8": False,
        "attn_mxfp8": False,
        "attn_fp8_pb": True,
        "input_scale": 1.0,
    }


def test_recipe_rejects_unknown_enabled_quantizer():
    quantize = copy.deepcopy(k3_cast.load_recipe(k3_cast._PUBLISHED_RECIPE).quantize.model_dump())
    quantize["quant_cfg"].append(
        {
            "quantizer_name": "*block_sparse_moe.shared_experts*weight_quantizer",
            "enable": True,
            "cfg": {},
        }
    )

    with pytest.raises(ValueError, match="does not support"):
        k3_cast._conversion_settings_from_quantize_config(quantize)


def test_recipe_rejects_algorithm_drift():
    quantize = copy.deepcopy(k3_cast.load_recipe(k3_cast._PUBLISHED_RECIPE).quantize.model_dump())
    quantize["algorithm"]["method"] = "smoothquant"

    with pytest.raises(ValueError, match="calibration-free max algorithm"):
        k3_cast._conversion_settings_from_quantize_config(quantize)


def test_recipe_accepts_additional_algorithm_defaults():
    quantize = copy.deepcopy(k3_cast.load_recipe(k3_cast._PUBLISHED_RECIPE).quantize.model_dump())
    quantize["algorithm"]["future_default"] = False
    quantize["algorithm"]["layerwise"]["future_default"] = None

    settings = k3_cast._conversion_settings_from_quantize_config(quantize)

    assert settings["cast_mxfp4_to_nvfp4"] is True


def test_recipe_rejects_explicit_input_scale(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--source_ckpt",
            str(tmp_path / "source"),
            "--output_ckpt",
            str(tmp_path / "output"),
            "--recipe",
            k3_cast._PUBLISHED_RECIPE,
            "--input_scale",
            "1.0",
        ],
    )

    with pytest.raises(SystemExit):
        k3_cast.main()

    assert "--recipe cannot be combined" in capsys.readouterr().err


def test_rank0_rendezvous_rejects_mismatched_configuration(tmp_path):
    ready_path = tmp_path / "ready.json"
    fingerprint = {"source_ckpt": "/models/Kimi-K3", "shards": ["model-1.safetensors"]}
    ready_path.write_text(
        json.dumps({"run_id": "run-1", "world_size": 4, "fingerprint": fingerprint})
    )

    assert not k3_cast._rank0_ready(
        ready_path, "other-run", world_size=4, rank=1, fingerprint=fingerprint
    )
    assert k3_cast._rank0_ready(ready_path, "run-1", world_size=4, rank=1, fingerprint=fingerprint)
    with pytest.raises(ValueError, match="rank 1 has --world_size 2"):
        k3_cast._rank0_ready(ready_path, "run-1", world_size=2, rank=1, fingerprint=fingerprint)
    with pytest.raises(ValueError, match=r"fingerprint differs in: \['source_ckpt'\]"):
        k3_cast._rank0_ready(
            ready_path,
            "run-1",
            world_size=4,
            rank=1,
            fingerprint={**fingerprint, "source_ckpt": "/other/Kimi-K3"},
        )


def test_rank_report_rejects_mismatched_fingerprint(tmp_path):
    report_path = tmp_path / "rank-00001.json"
    fingerprint = {"cast_mxfp4_to_nvfp4": True}
    report_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "rank": 1,
                "fingerprint": {"cast_mxfp4_to_nvfp4": False},
            }
        )
    )

    with pytest.raises(ValueError, match="rank 1 report conversion fingerprint"):
        k3_cast._rank_report_ready(report_path, "run-1", rank=1, fingerprint=fingerprint)


def test_module_name_aliases_strip_language_model_prefix():
    assert k3_cast._module_name_aliases("language_model.lm_head") == [
        "language_model.lm_head",
        "lm_head",
    ]


def _mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qtensor, scale = MXFP4QTensor.quantize(weight, block_size=32)
    expected_scale_shape = (*weight.shape[:-1], weight.shape[-1] // 32)
    return qtensor._quantized_data, scale.reshape(expected_scale_shape)


def _write_source_checkpoint(tmp_path: Path) -> tuple[Path, str, dict[str, torch.Tensor]]:
    source = tmp_path / "source"
    source.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    expert_prefix = "language_model.model.layers.1.block_sparse_moe.experts.0"

    torch.manual_seed(7)
    w1 = torch.randn(4, 64, dtype=torch.bfloat16) * 0.25
    w2 = torch.randn(4, 64, dtype=torch.bfloat16) * 2.0
    w3 = torch.randn(4, 64, dtype=torch.bfloat16) * 8.0
    q_proj = torch.randn(16, 64, dtype=torch.bfloat16)
    b_proj = torch.randn(8, 64, dtype=torch.bfloat16)
    f_a_proj = torch.randn(8, 64, dtype=torch.bfloat16)
    f_b_proj = torch.randn(16, 8, dtype=torch.bfloat16)
    passthrough = torch.randn(4, 4, dtype=torch.bfloat16)

    state: dict[str, torch.Tensor] = {
        "language_model.model.layers.1.self_attn.q_proj.weight": q_proj,
        "language_model.model.layers.1.self_attn.b_proj.weight": b_proj,
        "language_model.model.layers.1.self_attn.f_a_proj.weight": f_a_proj,
        "language_model.model.layers.1.self_attn.f_b_proj.weight": f_b_proj,
        "language_model.model.layers.1.input_layernorm.weight": passthrough,
    }
    for proj, weight in (("w1", w1), ("w2", w2), ("w3", w3)):
        packed, scale = _mxfp4(weight)
        base = f"{expert_prefix}.{proj}"
        state[base + ".weight_packed"] = packed
        state[base + ".weight_scale"] = scale

    save_file(state, str(source / shard_name))
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": dict.fromkeys(state, shard_name),
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["KimiK3ForConditionalGeneration"],
                "text_config": {
                    "quantization_config": {
                        "quant_method": "compressed-tensors",
                        "format": "mxfp4-pack-quantized",
                    }
                },
            }
        )
    )
    (source / "tokenizer_config.json").write_text("{}")
    return source, shard_name, state


def test_convert_shard_casts_experts_and_quantizes_attention(tmp_path):
    source, shard_name, source_state = _write_source_checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    report = k3_cast.convert_shard(
        source / shard_name,
        output / shard_name,
        device="cpu",
        cast=True,
        attn_fp8=True,
        input_scale_value=1.0,
    )

    expert = "language_model.model.layers.1.block_sparse_moe.experts.0"
    q_proj = "language_model.model.layers.1.self_attn.q_proj"
    b_proj = "language_model.model.layers.1.self_attn.b_proj"
    f_a_proj = "language_model.model.layers.1.self_attn.f_a_proj"
    with safe_open(output / shard_name, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for proj in ("w1", "w2", "w3"):
            base = f"{expert}.{proj}"
            assert base + ".weight_packed" not in keys
            assert f.get_tensor(base + ".weight").dtype == torch.uint8
            assert f.get_tensor(base + ".weight_scale").dtype == torch.float8_e4m3fn
            assert f.get_tensor(base + ".weight_scale_2").shape == torch.Size([])
            assert f.get_tensor(base + ".input_scale").item() == 1.0

        # Fused GEMM1 requires gate/up (w1/w3) to share one global scale.
        assert torch.equal(
            f.get_tensor(expert + ".w1.weight_scale_2"),
            f.get_tensor(expert + ".w3.weight_scale_2"),
        )

        assert f.get_tensor(q_proj + ".weight").dtype == torch.float8_e4m3fn
        assert f.get_tensor(q_proj + ".weight_scale").shape == torch.Size([])
        assert f.get_tensor(q_proj + ".input_scale").item() == 1.0

        # vLLM packs b/f_a with q/k/v/g, so the entire fused linear is FP8.
        for base in (b_proj, f_a_proj):
            assert f.get_tensor(base + ".weight").dtype == torch.float8_e4m3fn
            assert f.get_tensor(base + ".weight_scale").shape == torch.Size([])
            assert f.get_tensor(base + ".input_scale").item() == 1.0
        assert torch.equal(
            f.get_tensor("language_model.model.layers.1.input_layernorm.weight"),
            source_state["language_model.model.layers.1.input_layernorm.weight"],
        )

    assert report["stats"]["experts_converted"] == 3
    assert report["stats"]["attn_fp8_converted"] == 3
    assert report["stats"]["cast_blocks_total"] == 24
    assert report["stats"]["cast_blocks_lossless"] == 24
    assert report["banks"] == ["language_model.model.layers.1.block_sparse_moe.experts"]
    assert report["attn_modules"] == [b_proj, f_a_proj, q_proj]


def test_convert_shard_requantizes_w1_w3_with_shared_scale(tmp_path):
    source, shard_name, _ = _write_source_checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    report = k3_cast.convert_shard(
        source / shard_name,
        output / shard_name,
        device="cpu",
        cast=False,
        attn_fp8=False,
        input_scale_value=1.0,
    )

    expert = "language_model.model.layers.1.block_sparse_moe.experts.0"
    with safe_open(output / shard_name, framework="pt", device="cpu") as f:
        assert torch.equal(
            f.get_tensor(expert + ".w1.weight_scale_2"),
            f.get_tensor(expert + ".w3.weight_scale_2"),
        )
        assert not torch.equal(
            f.get_tensor(expert + ".w1.weight_scale_2"),
            f.get_tensor(expert + ".w2.weight_scale_2"),
        )

    assert report["stats"]["experts_converted"] == 3
    assert "cast_blocks_total" not in report["stats"]


def test_convert_shard_quantizes_attention_to_mxfp8_without_input_scale(tmp_path):
    source, shard_name, source_state = _write_source_checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    report = k3_cast.convert_shard(
        source / shard_name,
        output / shard_name,
        device="cpu",
        cast=True,
        attn_fp8=False,
        input_scale_value=1.0,
        attn_mxfp8=True,
    )

    base = "language_model.model.layers.1.self_attn.q_proj"
    original = source_state[base + ".weight"]
    with safe_open(output / shard_name, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        quantized = f.get_tensor(base + ".weight")
        scale = f.get_tensor(base + ".weight_scale")
        assert quantized.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.uint8
        assert scale.shape == (original.shape[0], original.shape[1] // 32)
        assert base + ".input_scale" not in keys

        restored = MXFP8QTensor(original.shape, original.dtype, quantized).dequantize(scale=scale)
        assert torch.allclose(restored, original, rtol=0.05, atol=0.05)

    assert report["stats"]["attn_mxfp8_converted"] == 3
    config = k3_cast._build_hf_quant_config(
        report["banks"], report["attn_modules"], attn_fp8=False, attn_mxfp8=True
    )
    assert config["quantization"]["quantized_layers"][base] == {"quant_algo": "MXFP8"}
    assert config["quantization"]["quantized_layers"]["layers.1.self_attn.in_proj_qkvgfab"] == {
        "quant_algo": "MXFP8"
    }


def test_convert_shard_quantizes_attention_to_block_fp8(tmp_path):
    source, shard_name, source_state = _write_source_checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    report = k3_cast.convert_shard(
        source / shard_name,
        output / shard_name,
        device="cpu",
        cast=True,
        attn_fp8=False,
        input_scale_value=1.0,
        attn_fp8_pb=True,
    )

    bases = [
        "language_model.model.layers.1.self_attn.q_proj",
        "language_model.model.layers.1.self_attn.b_proj",
        "language_model.model.layers.1.self_attn.f_a_proj",
        "language_model.model.layers.1.self_attn.f_b_proj",
    ]
    with safe_open(output / shard_name, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for base in bases:
            original = source_state[base + ".weight"]
            quantized = f.get_tensor(base + ".weight")
            scale = f.get_tensor(base + ".weight_scale")
            assert quantized.dtype == torch.float8_e4m3fn
            assert scale.dtype == torch.float32
            assert scale.shape == (
                (original.shape[0] + 127) // 128,
                1,
                (original.shape[1] + 127) // 128,
                1,
            )
            assert base + ".input_scale" not in keys

            restored = FP8QTensor(original.shape, original.dtype, quantized).dequantize(
                scale=scale.squeeze(1).squeeze(-1),
                block_sizes={-2: 128, -1: 128},
            )
            assert torch.allclose(restored, original, rtol=0.05, atol=0.05)

    assert report["stats"]["attn_fp8_pb_converted"] == 4
    config = k3_cast._build_hf_quant_config(
        report["banks"],
        report["attn_modules"],
        attn_fp8=False,
        attn_fp8_pb=True,
    )
    quantization = config["quantization"]
    assert quantization["quantized_layers"][bases[0]] == {"quant_algo": "FP8_PB_WO"}
    assert quantization["quantized_layers"]["layers.1.self_attn.in_proj_qkvgfab"] == {
        "quant_algo": "FP8_PB_WO"
    }
    assert "*self_attn.f_b_proj*" not in quantization["exclude_modules"]


def test_manifest_and_index_replace_source_mxfp4_schema(tmp_path):
    source, shard_name, _ = _write_source_checkpoint(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    report = k3_cast.convert_shard(
        source / shard_name,
        output / shard_name,
        device="cpu",
        cast=True,
        attn_fp8=True,
        input_scale_value=1.0,
    )

    hf_quant_config = k3_cast._build_hf_quant_config(
        report["banks"], report["attn_modules"], attn_fp8=True
    )
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    k3_cast._write_index_and_manifest(
        output,
        source_index,
        [report],
        hf_quant_config,
        attn_fp8=True,
    )
    k3_cast._rewrite_config_json(source / "config.json", output, hf_quant_config)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    expert = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    assert expert + ".weight_packed" not in weight_map
    assert weight_map[expert + ".weight"] == shard_name
    assert weight_map[expert + ".weight_scale"] == shard_name
    assert weight_map[expert + ".weight_scale_2"] == shard_name
    assert weight_map[expert + ".input_scale"] == shard_name
    assert index["metadata"]["total_size"] == report["tensor_bytes"]

    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config["text_config"]
    quant = config["quantization_config"]
    assert quant["quant_method"] == "modelopt_mixed"
    assert quant["quant_algo"] == "MIXED_PRECISION"
    assert len(quant["config_groups"]) == 2
    assert quant["quantized_layers"]["language_model.model.layers.1.block_sparse_moe.experts"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert quant["quantized_layers"]["language_model.model.layers.1.self_attn.q_proj"] == {
        "quant_algo": "FP8"
    }
    assert quant["quantized_layers"]["layers.1.block_sparse_moe.experts"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert quant["quantized_layers"]["layers.1.mlp.experts"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert quant["quantized_layers"]["layers.1.self_attn.q_proj"] == {"quant_algo": "FP8"}
    assert quant["quantized_layers"]["layers.1.self_attn.b_proj"] == {"quant_algo": "FP8"}
    assert quant["quantized_layers"]["layers.1.self_attn.f_a_proj"] == {"quant_algo": "FP8"}
    assert quant["quantized_layers"]["layers.1.self_attn.in_proj_qkvgfab"] == {"quant_algo": "FP8"}
