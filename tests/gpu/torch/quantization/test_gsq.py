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

import copy
import json

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_qwen3_5_vl_offline
from safetensors.torch import load_file

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from examples.hf_ptq.hf_gsq_rco import _reload_packed_for_validation
from modelopt.torch.export.quant_utils import preprocess_linear_fusion
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.utils.shared_input import SHARED_PATTERNS, find_shared_input_groups


def _fp8_snapshot(model):
    result = {}
    for name, module in model.named_modules():
        q = getattr(module, "weight_quantizer", None)
        if not isinstance(module, torch.nn.Linear) or q is None or not q.is_enabled:
            continue
        scale = q.amax.float().reshape(1) / 448
        stored = module.weight.detach().clone()
        result[name] = (
            stored,
            q(stored).detach(),
            (stored.float() / scale).to(torch.float8_e4m3fn),
            scale,
        )
    return result


def _check_fp8_export(exported, expected):
    rounding = {}
    for name, (stored, qdq, codes, scale) in expected.items():
        torch.testing.assert_close(
            exported[name + ".weight"].view(torch.uint8), codes.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(exported[name + ".weight_scale"], scale, rtol=0, atol=0)
        native = (codes.float() * scale).to(stored.dtype)
        rounding[name] = {
            "stored_mismatches": int((native != stored).sum()),
            "stored_max_abs": float((native.float() - stored.float()).abs().max()),
            "qdq_mismatches": int((native != qdq).sum()),
            "qdq_max_abs": float((native.float() - qdq.float()).abs().max()),
            "qdq_max_bf16_ulp": int(
                (native.view(torch.int16).int() - qdq.view(torch.int16).int()).abs().max()
            ),
        }
    return rounding


@pytest.mark.parametrize("algorithm", ["autoquant_max", "gsq_max", "gsq_gptq"])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_gdn_shared_projection_weights_survive_native_export(
    tmp_path, record_property, algorithm, nvfp4
):
    """GDN's fused qkv/z deployment grid must already match calibration and learning."""
    model = (
        get_tiny_qwen3_5_vl_offline(num_hidden_layers=1, layer_types=["linear_attention"])
        .cuda()
        .bfloat16()
        .eval()
    )
    model.config.architectures = [type(model).__name__]
    attention = model.model.language_model.layers[0].linear_attn
    with torch.no_grad():
        attention.in_proj_qkv.weight.mul_(3.7)
        attention.in_proj_z.weight.mul_(0.13)
    inputs = torch.randint(5, model.config.text_config.vocab_size, (1, 16), device="cuda")
    numeric = (
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
        if nvfp4
        else {"num_bits": (4, 3), "axis": None}
    )
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*in_proj_qkv.weight_quantizer",
                "cfg": numeric,
            },
            {
                "quantizer_name": "*in_proj_z.weight_quantizer",
                "cfg": numeric,
            },
        ],
        "algorithm": (
            "max"
            if algorithm == "autoquant_max"
            else {"method": "gsq", "initializer": algorithm.removeprefix("gsq_"), "steps": 2}
        ),
    }
    if algorithm == "autoquant_max":
        # Ordinary controls export the same hardened bank used during AutoQuant scoring.
        search = {
            "quantization_formats": [cfg],
            "data_loader": [inputs],
            "forward_step": lambda m, x: m(x, use_cache=False),
            "loss_func": lambda output, x: output.logits.float().square().mean(),
            "num_calib_steps": 1,
            "num_score_steps": 1,
            "disabled_layers": ["*visual*", "lm_head", "*in_proj_a", "*in_proj_b"],
            "checkpoint": str(tmp_path / "candidates.pt"),
            "method": {
                "method": "gradient",
                "data_fingerprint": "tiny-gdn-fixed-data",
                "preserve_candidate_weights": True,
            },
            "verbose": False,
        }
        _, bank = mtq.auto_quantize(
            copy.deepcopy(model), **search, constraints={"effective_bits": 16}
        )
        minimum = (
            16
            * sum(min(s["costs"]) for s in bank["candidate_stats"].values())
            / bank["cost_denominator"]
        )
        model, _ = mtq.auto_quantize(model, **search, constraints={"effective_bits": minimum})
    else:
        mtq.quantize(model, cfg, lambda m: m(inputs, use_cache=False))
    with torch.no_grad():
        expected_logits = model(inputs, use_cache=False).logits
        expected = {
            name: module.weight_quantizer(module.weight).detach().clone()
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and module.weight_quantizer.is_enabled
        }
        fp8 = _fp8_snapshot(model) if not nvfp4 else None
    if algorithm != "autoquant_max":
        assert model.model.language_model.layers[0].gsq["linear_attn"]["steps"] == 2
    export_hf_checkpoint(model, export_dir=tmp_path / "export")
    del model
    exported = load_file(str(tmp_path / "export/model.safetensors"), device="cuda")
    packed_only = _reload_packed_for_validation(tmp_path / "export", multimodal=True)
    for name, weight in expected.items():
        native = packed_only.get_submodule(name).weight
        if nvfp4:
            torch.testing.assert_close(native, weight, rtol=0, atol=0)
        else:
            stored, _, codes, scale = fp8[name]
            torch.testing.assert_close(
                native, (codes.float() * scale).to(stored.dtype), rtol=0, atol=0
            )
            if algorithm != "autoquant_max":
                torch.testing.assert_close(native, stored, rtol=0, atol=0)
    if fp8 is not None:
        record_property("native_weight_rounding", json.dumps(_check_fp8_export(exported, fp8)))
    with torch.no_grad():
        packed_logits = packed_only(inputs, use_cache=False).logits
        record_property(
            "packed_reload_max_abs", float((packed_logits - expected_logits).abs().max())
        )
        torch.testing.assert_close(packed_logits, expected_logits, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("initializer", ["max", "gptq"])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_gsq_region_reconstruction_and_restore(tmp_path, record_property, initializer, nvfp4):
    torch.manual_seed(17)
    model = get_tiny_llama().cuda().bfloat16().eval()
    model.config.architectures = ["LlamaForCausalLM"]
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight[0, :16].zero_()
    pristine = copy.deepcopy(model)
    inputs = torch.randint(0, model.config.vocab_size, (1, 16), device="cuda")
    numeric = (
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
        if nvfp4
        else {"num_bits": (4, 3), "axis": None}
    )
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": numeric},
            {"quantizer_name": "*lm_head*", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Embedding", "enable": False},
        ]
    }
    samples = {}
    handles = []
    for name in ("self_attn", "mlp"):

        def capture(m, args, kwargs, output, name=name):
            samples[name] = (
                args,
                kwargs,
                output if isinstance(output, torch.Tensor) else output[0],
            )

        handles.append(
            getattr(pristine.model.layers[0], name).register_forward_hook(capture, with_kwargs=True)
        )
    with torch.no_grad():
        pristine(inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    baseline = copy.deepcopy(pristine)
    cfg["algorithm"] = {"method": initializer, "layerwise": {"enable": True}}
    mtq.quantize(baseline, cfg, lambda m: m(inputs, use_cache=False))
    for _, members in find_shared_input_groups(baseline, SHARED_PATTERNS):
        preprocess_linear_fusion(members)
    cfg["algorithm"] = {"method": "gsq", "initializer": initializer, "steps": 2}
    mtq.quantize(model, cfg, lambda m: m(inputs, use_cache=False))
    with torch.no_grad():
        for name, (args, kwargs, target) in samples.items():
            errors = []
            for candidate in (baseline, model):
                output = getattr(candidate.model.layers[0], name)(*args, **kwargs)
                output = output if isinstance(output, torch.Tensor) else output[0]
                errors.append((output.float() - target.float()).square().mean())
            assert errors[1] <= errors[0] * 1.01 + 1e-8
    result = model(inputs).logits
    assert torch.isfinite(result).all()
    for layer in model.model.layers:
        assert layer.gsq["self_attn"]["best_loss"] <= layer.gsq["self_attn"]["initial_loss"]
        assert layer.gsq["mlp"]["best_loss"] <= layer.gsq["mlp"]["initial_loss"]
        assert not layer.self_attn.q_proj.input_quantizer.is_enabled
    assert torch.equal(model.lm_head.weight, pristine.lm_head.weight)
    assert torch.equal(model.model.embed_tokens.weight, pristine.model.embed_tokens.weight)
    mto.save(model, tmp_path / "model.pt")
    restored = mto.restore(pristine, tmp_path / "model.pt")
    torch.testing.assert_close(restored(inputs).logits, result, rtol=0, atol=0)

    projection = restored.model.layers[0].self_attn.q_proj
    weight = projection.weight.detach().clone()
    shape, dtype = weight.shape, weight.dtype
    qdq = projection.weight_quantizer(weight).detach()
    fp8 = _fp8_snapshot(restored) if not nvfp4 else None
    export_hf_checkpoint(restored, export_dir=tmp_path / "export")
    exported = load_file(str(tmp_path / "export" / "model.safetensors"), device="cuda")
    prefix = "model.layers.0.self_attn.q_proj."
    if nvfp4:
        packed = NVFP4QTensor(shape, dtype, exported[prefix + "weight"])
        decoded = packed.dequantize(
            dtype=dtype,
            scale=exported[prefix + "weight_scale"],
            double_scale=exported[prefix + "weight_scale_2"],
            block_sizes={-1: 16},
        )
    else:
        decoded = (
            exported[prefix + "weight"].float() * exported[prefix + "weight_scale"].float()
        ).to(dtype)
    if nvfp4:
        torch.testing.assert_close(decoded, qdq, rtol=0, atol=0)
    else:
        record_property("native_weight_rounding", json.dumps(_check_fp8_export(exported, fp8)))
    assert torch.count_nonzero(decoded[0, :16]) == 0


@pytest.mark.parametrize("initializer", ["max", "gptq"])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_auto_quantize_rco_candidates_survive_native_export(
    tmp_path, record_property, initializer, nvfp4
):
    """Ordinary controls and RCO share hard candidates that native packing preserves exactly."""
    torch.manual_seed(43)
    original = get_tiny_llama().cuda().bfloat16().eval()
    original.config.architectures = ["LlamaForCausalLM"]
    with torch.no_grad():
        attention = original.model.layers[0].self_attn
        attention.q_proj.weight.mul_(0.13)
        attention.v_proj.weight.mul_(3.7)
        attention.q_proj.weight[0, :16].zero_()
    inputs = torch.randint(0, original.config.vocab_size, (1, 16), device="cuda")
    numeric = (
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}}
        if nvfp4
        else {"num_bits": (4, 3), "axis": None}
    )
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": numeric},
            {"quantizer_name": "*lm_head*", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Embedding", "enable": False},
        ],
        "algorithm": {"method": initializer, "layerwise": {"enable": True}},
    }
    source = tmp_path / "candidates.pt"
    kwargs = {
        "quantization_formats": [config],
        "data_loader": [inputs],
        "forward_step": lambda m, x: m(x, use_cache=False),
        "num_calib_steps": 1,
        "num_score_steps": 1,
        "disabled_layers": ["lm_head"],
        "verbose": False,
    }
    baseline, bank = mtq.auto_quantize(
        copy.deepcopy(original),
        **kwargs,
        constraints={"effective_bits": 16},
        checkpoint=str(source),
        method={
            "method": "gradient",
            "data_fingerprint": "tiny-native-fixed-data",
            "preserve_candidate_weights": True,
        },
        loss_func=lambda output, x: torch.nn.functional.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
            x[:, 1:].reshape(-1),
        ),
    )
    torch.testing.assert_close(baseline(inputs).logits, original(inputs).logits, rtol=0, atol=0)
    minimum = (
        16
        * sum(min(s["costs"]) for s in bank["candidate_stats"].values())
        / bank["cost_denominator"]
    )
    immutable = source.read_bytes()
    model, state = mtq.auto_quantize(
        copy.deepcopy(original),
        **kwargs,
        constraints={"effective_bits": minimum},
        method={
            "method": "rco",
            "steps": 1,
            "data_fingerprint": "tiny-native-fixed-data",
            "candidate_checkpoint": str(source),
        },
    )
    assert source.read_bytes() == immutable
    assert state["best"]["is_satisfied"]
    assert torch.equal(model.lm_head.weight, original.lm_head.weight)
    assert torch.equal(model.model.embed_tokens.weight, original.model.embed_tokens.weight)
    before = model(inputs).logits.detach()
    mto.save(model, tmp_path / "selected.pt")
    model = mto.restore(copy.deepcopy(original), tmp_path / "selected.pt")
    torch.testing.assert_close(model(inputs).logits, before, rtol=0, atol=0)
    expected = {
        name: (module.weight_quantizer(module.weight).detach().clone(), module.weight.dtype)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module.weight_quantizer.is_enabled
    }
    assert "model.layers.0.self_attn.q_proj" in expected
    fp8 = _fp8_snapshot(model) if not nvfp4 else None
    export_hf_checkpoint(model, export_dir=tmp_path / "export")
    exported = load_file(str(tmp_path / "export" / "model.safetensors"), device="cuda")
    for name, (qdq, dtype) in expected.items():
        if nvfp4:
            packed = NVFP4QTensor(qdq.shape, dtype, exported[name + ".weight"])
            decoded = packed.dequantize(
                dtype=dtype,
                scale=exported[name + ".weight_scale"],
                double_scale=exported[name + ".weight_scale_2"],
                block_sizes={-1: 16},
            )
        else:
            decoded = (
                exported[name + ".weight"].float() * exported[name + ".weight_scale"].float()
            ).to(dtype)
        if nvfp4:
            torch.testing.assert_close(decoded, qdq, rtol=0, atol=0)
    if fp8 is not None:
        record_property("native_weight_rounding", json.dumps(_check_fp8_export(exported, fp8)))
    assert torch.count_nonzero(expected["model.layers.0.self_attn.q_proj"][0][0, :16]) == 0
    del model, original, state, bank
    source.unlink()
    (tmp_path / "selected.pt").unlink()
    packed_only = _reload_packed_for_validation(tmp_path / "export")
    with torch.no_grad():
        packed_logits = packed_only(inputs).logits
        record_property("packed_reload_max_abs", float((packed_logits - before).abs().max()))
        torch.testing.assert_close(packed_logits, before, rtol=0.02, atol=0.02)
