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

"""Native small-prefix GPTQ/Bypass coverage using revision-pinned model configs."""

import inspect
import json

import pytest
import torch
import torch.distributed as dist
from _test_utils.automodel_families import build_model, cases

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from examples.local_quantization_distillation.repair import quantize_student
from modelopt.torch.distill.plugins.automodel import (
    AutoModelLocalDistillationAdapter,
    prepare_packed_batch,
)
from modelopt.torch.distill.plugins.automodel.families import native_family
from modelopt.torch.quantization.nn import TensorQuantizer

_RECORDS = [
    record
    for record in cases()
    if record.get("native_supported", True) and record.get("minimum_pattern_prefix")
]
_FORMATS = {
    "iq2_s": {"num_bits": "iq2_s", "backend": "ggml"},
    "iq1_m": {"num_bits": "iq1_m", "backend": "ggml"},
    "iq2_xs": {"num_bits": "iq2_xs", "backend": "ggml"},
    "nvfp4": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
}

_GDN_MODELS = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-2.4T-A95B",
    "Qwen/Qwen3.8-Flash-Next",
)


@pytest.fixture
def native_eager(request):
    """Test actual native kernels without compiling duplicate Torch wrappers."""
    pytest.importorskip("nemo_automodel")
    if dist.is_initialized() and dist.get_world_size() > 1:
        pytest.skip("Independent family probes run separately from the multi-rank ownership suite")
    parameters = request.node.callspec.params
    model_id = parameters.get("model_id", parameters.get("record", {}).get("model_id", ""))
    # Native QSA requires its real FlexAttention kernel; ordinary Torch helper
    # wrappers need no compiler for a numerical compatibility check.
    with torch.compiler.set_stance("default" if "Flash-Next" in model_id else "force_eager"):
        yield torch.device("cuda", torch.cuda.current_device())


def _pair(record, device):
    teacher = build_model(record, device).requires_grad_(False)
    student = build_model(record, device)
    # Native PLE/Engram modules own communication handles. Rebuild instead of
    # deepcopying those handles, matching the public teacher/student loading path.
    student.load_state_dict(teacher.state_dict())
    return teacher, student


def _batch(record, device, model=None):
    length = 256 if "DeepSeek-V4-Pro" in record["model_id"] else 32
    tokens = torch.randint(3, 110, (2, length), device=device)
    documents = torch.ones_like(tokens)
    documents[:, length // 2 :] = 2
    batch = {
        "input_ids": tokens,
        "attention_mask": documents,
        "_packed_seq_ids": documents,
        "position_ids": torch.arange(length // 2, device=device).repeat(2, 2),
        "valid_mask": documents.ne(0),
    }
    if model is not None and native_family(model).name in (
        "muse_spark",
        "nemotron_omni",
        "mistral_large4",
    ):
        batch = {key: value[:1].clone() for key, value in batch.items()}
        batch["input_ids"][:, -1] = 0
        batch["attention_mask"][:, -1] = 0
        batch["_packed_seq_ids"][:, -1] = 0
        batch["position_ids"][:, -1] = 0
        batch["valid_mask"][:, -1] = False
        if native_family(model).name == "muse_spark":
            batch["input_ids"][0, 2] = model.config.patch_token_id
            batch["pixel_values"] = torch.randn(4, 24, device=device, dtype=torch.bfloat16)
            batch["image_grid_thw"] = torch.tensor([[1, 2, 2]], device=device)
        elif native_family(model).name == "nemotron_omni":
            batch["input_ids"][0, 2:6] = model.img_context_token_id
            batch["pixel_values"] = torch.randn(1, 3, 8, 8, device=device, dtype=torch.bfloat16)
    return batch


def _logits(model, batch):
    chunks = []
    for microbatch in AutoModelLocalDistillationAdapter().microbatches(batch, 1):
        inputs = {
            key: value
            for key, value in prepare_packed_batch(model, microbatch).items()
            if key != "valid_mask"
        }
        if "use_cache" in inspect.signature(model.forward).parameters:
            inputs["use_cache"] = False
        with torch.no_grad():
            output = model(**inputs)
            chunks.append(output if isinstance(output, torch.Tensor) else output.logits)
    return torch.cat(chunks, dim=0)


def _reload(record, result, batch, device, tmp_path):
    path = tmp_path / "repaired.pt"
    mto.save(result.student, path)
    restored = mto.restore(build_model(record, device), path)
    torch.testing.assert_close(
        _logits(restored, batch), _logits(result.student, batch), rtol=0, atol=0
    )


@pytest.mark.parametrize("record", _RECORDS, ids=[record["model_id"] for record in _RECORDS])
@pytest.mark.parametrize("format_name", _FORMATS)
@pytest.mark.timeout(180)
def test_native_model_family_gptq_bypass_roundtrip(record, format_name, native_eager, tmp_path):
    """Exercise the native distinct-layer prefix, format GPTQ and exact reload."""
    teacher, student = _pair(record, native_eager)
    batch = _batch(record, native_eager, teacher)
    adapter = AutoModelLocalDistillationAdapter()
    family = native_family(student)
    assert family is not None
    patterns = family.mlp_weight_patterns(student)
    assert patterns
    settings = {
        "quantization": {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}]
            + [{"quantizer_name": pattern, "cfg": _FORMATS[format_name]} for pattern in patterns],
            "algorithm": {
                "method": "gptq",
                "block_size": 256,
                "perc_damp": 0.01,
                "fused": format_name == "nvfp4",
                "importance_weighted": True,
                "layerwise": {"enable": True},
            },
        }
    }
    if format_name == "nvfp4":
        activation_patterns = dict.fromkeys(
            pattern.split(".*weight_quantizer")[0] + ".*input_quantizer" for pattern in patterns
        )
        settings["quantization"]["quant_cfg"].extend(
            {"quantizer_name": pattern, "cfg": _FORMATS[format_name]}
            for pattern in activation_patterns
        )
    check_prefix_resume = family.name == "mistral_large4" and format_name == "iq2_s"
    if check_prefix_resume:
        checkpoint = tmp_path / "ptq"
        settings["quantization"]["algorithm"]["layerwise"]["checkpoint_dir"] = str(checkpoint)
    quantize_student(student, [batch], adapter, settings)
    if check_prefix_resume:
        manifest = checkpoint / "manifest.json"
        state = json.loads(manifest.read_text())
        state["last_completed_layer"] = 4
        manifest.write_text(json.dumps(state))
        assert not list(checkpoint.glob("*/next_inputs.pt"))
        resumed = build_model(record, native_eager)
        resumed.load_state_dict(teacher.state_dict())
        quantize_student(resumed, [batch], AutoModelLocalDistillationAdapter(), settings)
        torch.testing.assert_close(_logits(resumed, batch), _logits(student, batch), rtol=0, atol=0)
        del resumed
    if family.name == "minimax_m3_vl":
        vision_quantizers = [
            q
            for name, q in student.named_modules()
            if "vision_tower" in name
            and name.endswith("weight_quantizer")
            and isinstance(q, TensorQuantizer)
        ]
        assert vision_quantizers
        for q in vision_quantizers:
            q.set_from_attribute_config(
                {"enable": True, "num_bits": 8, "axis": 0, "block_sizes": None, "backend": None}
            )
        selection = mtd.LocalDistillationConfig()
        resolved = adapter.resolve_units(student, teacher, selection)
        assert {unit.student_path for unit in resolved} == set(family.units(student, selection))
    routing_handles, teacher_routes = [], {}
    if family.name == "deepseek_v4":
        for name, gate in teacher.named_modules():
            if name.endswith(".mlp.gate") and hasattr(gate, "_pending_input_ids"):
                routing_handles.append(
                    gate.register_forward_hook(
                        lambda module, args, output, name=name: teacher_routes.__setitem__(
                            name, output[1].detach().clone()
                        )
                    )
                )

                def check_route(module, args, output, name=name):
                    torch.testing.assert_close(output[1], teacher_routes[name], rtol=0, atol=0)

                routing_handles.append(
                    student.get_submodule(name).register_forward_hook(check_route)
                )
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        {
            "updates": 2,
            "global_batch_size": batch["input_ids"].shape[0],
            "micro_batch_size": 1,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
        validation_dataloader=[batch],
        adapter=adapter,
    )
    for handle in routing_handles:
        handle.remove()
    assert len(result.metrics["units"]) == record.get(
        "expected_repair_units", record["minimum_pattern_prefix"]
    )
    assert all(
        torch.isfinite(torch.tensor(unit["validation_mse"]))
        for unit in result.metrics["units"].values()
    )
    assert all(
        unit["gradient_norm"] > 0
        for update in result.metrics["history"]
        for unit in update["units"].values()
    )
    if family.name in ("muse_spark", "nemotron_omni", "mistral_large4"):
        expected_tokens = int(batch["valid_mask"].sum())
        for update in result.metrics["history"]:
            assert all(unit["valid_tokens"] == expected_tokens for unit in update["units"].values())
        coverage = [
            counts
            for unit in result.metrics["units"].values()
            for counts in unit.get("expert_coverage", {}).values()
        ]
        assert coverage
        for counts in coverage:
            assert sum(counts["routed_assignments"]) == 2 * expected_tokens * 2
    _reload(record, result, batch, native_eager, tmp_path)


@pytest.mark.parametrize("model_id", _GDN_MODELS)
@pytest.mark.timeout(180)
def test_native_qwen_gdn_w4a4_calibration_gradients_and_reload(model_id, native_eager, tmp_path):
    """Protect fused-conv calibration, packed layout and every weight's scale gradient."""
    record = next(record for record in _RECORDS if record["model_id"] == model_id)
    teacher, student = _pair(record, native_eager)
    batch = _batch(record, native_eager)
    adapter = AutoModelLocalDistillationAdapter()
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*linear_attn*weight_quantizer",
                    "cfg": {"num_bits": 4, "axis": 0},
                },
                {
                    "quantizer_name": "*linear_attn*input_quantizer",
                    "cfg": {"num_bits": 4, "axis": None},
                },
            ],
            "algorithm": "max",
        },
        lambda model: [
            adapter.teacher_microbatch(model, [], micro, lambda unit, record: None)
            for micro in adapter.microbatches(batch, 1)
        ],
    )
    quantizers = {
        name: quantizer
        for name, quantizer in student.named_modules()
        if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled
    }
    assert quantizers and all(quantizer.amax is not None for quantizer in quantizers.values())
    frozen = {
        name: quantizer.amax.clone()
        for name, quantizer in quantizers.items()
        if "input_quantizer" in name
    }
    seen, handles = set(), []
    expected = {name for name in quantizers if "weight_quantizer" in name}
    for name in expected:

        def observe(module, inputs, output, name=name):
            if torch.is_grad_enabled() and output.requires_grad:
                output.register_hook(lambda gradient, name=name: seen.add(name))

        handles.append(quantizers[name].register_forward_hook(observe))
    try:
        result = mtd.local_distill(
            student,
            teacher,
            [batch, batch],
            {
                "unit_types": ["gdn"],
                "updates": 2,
                "global_batch_size": 2,
                "micro_batch_size": 1,
                "scheduler": "constant",
                "warmup_fraction": 0,
            },
            adapter=adapter,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert seen == expected
    for name, value in frozen.items():
        torch.testing.assert_close(quantizers[name].amax, value, rtol=0, atol=0)
    _reload(record, result, batch, native_eager, tmp_path)


@pytest.mark.parametrize("model_id", ["moonshotai/Kimi-K3"])
def test_kimi_first_moe_selective_prefix_and_reload(model_id, native_eager, tmp_path):
    """Selective source loading preserves real prefix inputs and complete MoE math."""
    import json

    from nemo_automodel.components.models.kimi_k3.config import KimiK3Config
    from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
    from safetensors.torch import save_file

    from examples.local_quantization_distillation.kimi_k3.source import (
        capture_inputs,
        load_moe,
        prefix_model,
    )

    record = next(record for record in _RECORDS if record["model_id"] == model_id)
    teacher = build_model(record, native_eager).eval().requires_grad_(False)
    adapter = KimiK3StateDictAdapter(
        teacher.config, teacher.model.moe_config, teacher.backend, dtype=torch.bfloat16
    )
    exported = adapter.to_hf(teacher.state_dict())
    save_file(
        {key: value.contiguous() for key, value in exported.items()}, tmp_path / "tiny.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(exported, "tiny.safetensors")})
    )
    KimiK3Config(text_config=teacher.config.to_dict()).save_pretrained(tmp_path)
    batch = _batch(record, native_eager)
    expected = capture_inputs(teacher, batch)
    prefix = prefix_model(tmp_path)
    actual = capture_inputs(prefix, batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = load_moe(tmp_path, None)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(actual), teacher.model.layers["1"].mlp(expected), rtol=0, atol=0
        )
