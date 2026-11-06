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

"""Actual HF family topology and tiny random-weight PTQ/Bypass contracts."""

import copy
import math
import sys
from contextlib import nullcontext

import pytest
import torch
from _test_utils.transformers_families import build_model, cases, hf_torch_kernels, release_cases
from transformers import DynamicCache

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.local_units import RepairUnit
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.distill.plugins.transformers.families import transformers_family
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer

_RECORDS = cases() + [
    record
    for record in release_cases()
    if record["family"] in ("deepseek_v41", "step5", "kimi_k31", "mimo_v2")
]


@pytest.mark.parametrize("record", _RECORDS, ids=lambda record: record["family"])
def test_hf_family_semantic_boundaries(record):
    """Wrong hybrid aliases or text roots must not select recurrence as attention."""
    model = build_model(record, "meta")
    family = transformers_family(model)
    assert family.name != "generic"
    layers = family.decoder_layers(model)
    assert len(layers) == record["minimum_pattern_prefix"]
    if record["family"] == "qwen3_omni_moe":
        assert family.decoder_layers(model.model) is layers
    names = {id(module): name for name, module in model.named_modules()}
    expected_by_role = {}
    for role in ("mlp", "moe", "attention", "gdn", "mamba", "decoder"):
        selection = mtd.LocalDistillationConfig(unit_types=(role,))
        paths = family.unit_paths(model, selection, quantized_only=False)
        expected = []
        for candidate in layers:
            kind = getattr(candidate, "block_type", None)
            if role == "decoder":
                expected.append(names[id(candidate)])
            elif "mixer" in candidate._modules:
                actual_role = {"full_attention": "attention", "linear_attention": "mamba"}.get(
                    kind, kind
                )
                if role == actual_role:
                    expected.append(names[id(candidate.mixer)])
            elif role in ("mlp", "moe"):
                child = next(
                    (
                        candidate._modules[name]
                        for name in ("mlp", "block_sparse_moe", "ffn", "moe")
                        if name in candidate._modules
                    ),
                    None,
                )
                if child is not None:
                    is_moe = type(child).__name__.lower().endswith(("moe", "sparsemoeblock"))
                    if role == "mlp" or is_moe:
                        expected.append(names[id(child)])
                if role == "mlp" and "share_expert" in candidate._modules:
                    expected.append(names[id(candidate.share_expert)])
            elif role in ("attention", "gdn"):
                linear = candidate._modules.get("linear_attn")
                attention = next(
                    (
                        candidate._modules[name]
                        for name in ("self_attn", "attn", "attention")
                        if name in candidate._modules
                    ),
                    None,
                )
                is_linear = (
                    kind == "linear_attention"
                    or getattr(candidate, "is_linear_attn", False)
                    or getattr(candidate, "is_kda", False)
                )
                child = linear if role == "gdn" else None
                if attention is not None and (
                    (role == "gdn" and is_linear) or (role == "attention" and not is_linear)
                ):
                    child = attention
                if child is not None:
                    expected.append(names[id(child)])
        assert paths == expected, (record["family"], role)
        expected_by_role[role] = expected
    combined = ("gdn", "attention", "mlp", "moe", "mamba")
    ordered = list(
        dict.fromkeys(
            path
            for block in layers
            for role in combined
            for path in expected_by_role[role]
            if path.startswith(names[id(block)] + ".")
        )
    )
    assert (
        family.unit_paths(
            model, mtd.LocalDistillationConfig(unit_types=combined), quantized_only=False
        )
        == ordered
    )
    selection = mtd.LocalDistillationConfig()
    paths = family.unit_paths(model, selection, quantized_only=False)
    patterns = family.weight_patterns(model, selection)
    mtq.quantize(
        model,
        {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}]
            + [
                {"quantizer_name": pattern, "cfg": {"num_bits": 4, "axis": 0}}
                for pattern in patterns
            ],
            "algorithm": None,
        },
    )
    assert family.unit_paths(model, selection) == paths
    for name, quantizer in model.named_modules():
        if isinstance(quantizer, TensorQuantizer) and name.endswith("weight_quantizer"):
            if any(part in ("gate", "router") for part in name.split(".")):
                assert not quantizer.is_enabled, name
    assert not any(name.startswith("nemo_automodel") for name in sys.modules)


def _batch(device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(3, 110, (1, 32), generator=generator, device=device)
    docs = torch.ones_like(ids)
    docs[:, 16:31] = 2
    docs[:, 31:] = 0
    return {
        "input_ids": ids,
        "attention_mask": docs,
        "_packed_seq_ids": docs,
        "position_ids": torch.arange(16, device=device).repeat(1, 2),
        "valid_mask": docs.ne(0),
    }


@pytest.mark.parametrize("record", _RECORDS, ids=lambda record: record["family"])
@pytest.mark.timeout(240)
def test_hf_family_ptq_bypass_finite_gradients_and_heldout(record):
    """Every real family runs two updates with masked packed teacher captures."""
    _repair(record)


@pytest.mark.parametrize(
    ("family", "roles"),
    [
        ("qwen3_next", ("attention", "gdn")),
        ("glm5_next", ("attention", "gdn")),
        ("kimi_linear", ("attention", "gdn")),
        ("nemotron_h", ("attention", "mamba")),
        ("mistral_large4", ("attention",)),
        ("deepseek_v4", ("attention",)),
        ("glm5_next", ("decoder",)),
        ("deepseek_v4", ("decoder",)),
        ("qwen4_exp", ("decoder",)),
    ],
)
@pytest.mark.timeout(240)
def test_hf_family_hybrid_and_structured_unit_replay(family, roles):
    """Hybrid math and arbitrary-rank decoder boundaries use actual HF forwards."""
    _repair(next(record for record in _RECORDS if record["family"] == family), roles)


@pytest.mark.parametrize("active_window", [True, False])
def test_hf_qwen3_window_packing_matches_separate_documents(active_window):
    """A document mask must preserve active native sliding-window semantics."""
    record = copy.deepcopy(next(r for r in _RECORDS if r["family"] == "qwen3"))
    depth = record["config"]["num_hidden_layers"]
    record["config"].update(
        use_sliding_window=True,
        sliding_window=8,
        max_window_layers=0 if active_window else depth,
        layer_types=["sliding_attention" if active_window else "full_attention"] * depth,
    )
    torch.manual_seed(17)
    teacher = build_model(record, "cpu").requires_grad_(False)
    adapter = TransformersLocalDistillationAdapter()
    path = "model.layers.1"
    unit = RepairUnit(path, path, path)
    batch = _batch("cpu", 19)
    if not active_window:
        teacher.config._attn_implementation = "sdpa"
        bias = torch.zeros(1, 1, 32, 32)
        bias[:, :, 0, :] = -torch.inf
        bias[:, :, 2, 0] = 2.0
        batch["attention_mask"] = bias
        prepared = adapter.prepare_batch(teacher, batch)["attention_mask"]
        assert torch.isneginf(prepared[:, :, 0, :]).all()
        assert prepared[0, 0, 2, 0] == 2.0
    packed, separate = [], []
    adapter.teacher_microbatch(
        teacher, [unit], batch, lambda _, record: packed.append(record.target)
    )
    for start, end in ((0, 16), (16, 31)):
        document = {
            "input_ids": batch["input_ids"][:, start:end],
            "position_ids": batch["position_ids"][:, start:end],
        }
        if not active_window:
            # Independently apply ordinary causal attention within this document.
            future = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            document["attention_mask"] = bias[:, :, start:end, start:end].masked_fill(
                future, -torch.inf
            )
            document["valid_mask"] = torch.ones(1, end - start, dtype=torch.bool)
        adapter.teacher_microbatch(
            teacher, [unit], document, lambda _, record: separate.append(record.target)
        )
    actual = torch.cat(packed, dim=1)[:, :31]
    expected = torch.cat(separate, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    assert len(packed) == (2 if active_window else 1)
    if not active_window:
        backend = teacher.config._attn_implementation
        try:
            teacher.config._attn_implementation = "eager"
            causal = torch.ones(1, 1, 8, 8, dtype=torch.bool).tril()
            ordinary = {
                "input_ids": batch["input_ids"][:, :8],
                "position_ids": batch["position_ids"][:, :8],
                "valid_mask": torch.ones(1, 8, dtype=torch.bool),
            }
            boolean_outputs, additive_outputs = [], []
            adapter.teacher_microbatch(
                teacher,
                [unit],
                {**ordinary, "attention_mask": causal},
                lambda _, record: boolean_outputs.append(record.target),
            )
            dtype = next(teacher.parameters()).dtype
            additive = torch.zeros_like(causal, dtype=dtype).masked_fill(
                ~causal, torch.finfo(dtype).min
            )
            adapter.teacher_microbatch(
                teacher,
                [unit],
                {**ordinary, "attention_mask": additive},
                lambda _, record: additive_outputs.append(record.target),
            )
            assert len(boolean_outputs) == len(additive_outputs) == 1
            torch.testing.assert_close(boolean_outputs[0], additive_outputs[0], rtol=0, atol=0)
        finally:
            teacher.config._attn_implementation = backend


def _omni_record():
    record = copy.deepcopy(next(r for r in _RECORDS if r["family"] == "qwen3_omni_moe"))
    record.update(omni_wrapper=True, architectures=["Qwen3OmniMoeForConditionalGeneration"])
    return record


def test_hf_omni_wrapper_forward_routes_actual_thinker():
    """The generation-only outer wrapper still captures its actual text boundary."""
    teacher = build_model(_omni_record(), "cpu").requires_grad_(False)
    path = "thinker.model.language_model.layers.0.mlp"
    unit = RepairUnit(path, path, path)
    captured, hooks = [], []
    handles = [
        teacher.register_forward_pre_hook(lambda *_: hooks.append("before")),
        teacher.register_forward_hook(lambda *_: hooks.append("after")),
    ]
    had_forward_override = "forward" in teacher.__dict__
    try:
        TransformersLocalDistillationAdapter().teacher_microbatch(
            teacher,
            [unit],
            {"input_ids": torch.ones(1, 8, dtype=torch.long)},
            lambda _, record: captured.append(record.target),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert hooks == ["before", "after"]
    assert ("forward" in teacher.__dict__) == had_forward_override
    assert len(captured) == 1
    assert captured[0].shape == (1, 8, 256)
    assert torch.isfinite(captured[0]).all()


def test_hf_generation_cache_rejected_before_teacher_forward():
    """Independent repair documents must not mutate a caller's generation cache."""
    teacher = build_model(next(r for r in _RECORDS if r["family"] == "llama"), "cpu")
    cache = DynamicCache(config=teacher.config)
    with pytest.raises(ValueError):
        TransformersLocalDistillationAdapter().teacher_microbatch(
            teacher,
            [],
            {"input_ids": torch.ones(1, 8, dtype=torch.long), "past_key_values": cache},
            lambda *_: None,
        )
    assert cache.get_seq_length() == 0


@pytest.mark.timeout(240)
def test_hf_omni_wrapper_ptq_bypass():
    result = _repair(_omni_record())
    assert all(
        path.startswith("thinker.model.language_model.layers.") for path in result.metrics["units"]
    )


@pytest.mark.parametrize("family", ["qwen3_next", "glm5_next"])
@pytest.mark.timeout(600)
def test_hf_family_native_recurrent_cuda_smoke(family):
    """Check actual optional GDN/KDA dispatch separately from portable Torch math."""
    pytest.importorskip("fla", reason="Native recurrent CUDA smoke requires optional FLA")
    _repair(
        next(record for record in _RECORDS if record["family"] == family), ("gdn",), portable=False
    )


def _integer_codes(model):
    codes = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, QuantModule):
                for index, (weight, quantizer) in enumerate(module.iter_weights_for_calibration()):
                    if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled:
                        threshold = quantizer._get_amax(weight).float()
                        values = quantizer(weight).float()
                        codes[name, index] = (
                            (values * 7 / threshold.clamp_min(1e-24)).round().to(torch.int8)
                        )
    return codes


def _repair(record, roles=("mlp", "moe"), *, portable=True):
    torch.manual_seed(17)
    teacher = build_model(record, "cuda").requires_grad_(False)
    with hf_torch_kernels(teacher) if portable else nullcontext():
        return _repair_model(teacher, roles)


def _repair_model(teacher, roles):
    student = copy.deepcopy(teacher)
    adapter = TransformersLocalDistillationAdapter()
    family = transformers_family(student)
    batch, heldout = _batch("cuda", 19), _batch("cuda", 29)
    selection = mtd.LocalDistillationConfig(unit_types=roles)
    paths = family.unit_paths(student, selection, quantized_only=False)
    assert paths
    mtq.quantize(
        student,
        {
            "quant_cfg": [{"quantizer_name": "*", "enable": False}]
            + [
                {"quantizer_name": pattern, "cfg": {"num_bits": 4, "axis": 0}}
                for pattern in family.weight_patterns(student, selection)
            ],
            "algorithm": "max",
        },
        lambda model: adapter.teacher_microbatch(model, [], batch, lambda *_: None),
    )
    original_codes = _integer_codes(student)
    assert original_codes
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        {
            "updates": 2,
            "unit_types": roles,
            "global_batch_size": 1,
            "micro_batch_size": 1,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
        validation_dataloader=[heldout],
        adapter=adapter,
    )
    assert set(result.metrics["units"]) == set(paths)
    assert len(result.metrics["history"]) == 2
    for update in result.metrics["history"]:
        assert all(
            math.isfinite(row["gradient_norm"])
            and row["gradient_norm"] > 0
            and row["valid_tokens"] == 31
            for row in update["units"].values()
        )
    assert all(math.isfinite(row["validation_mse"]) for row in result.metrics["units"].values())
    quantizers = [q for q in student.modules() if isinstance(q, TensorQuantizer) and q.is_enabled]
    assert quantizers
    assert all(not hasattr(q, "_local_repair_state") for q in quantizers)
    repaired_codes = _integer_codes(student)
    assert original_codes.keys() == repaired_codes.keys()
    for name, codes in original_codes.items():
        torch.testing.assert_close(codes, repaired_codes[name], rtol=0, atol=0)
    assert all(not any(parameter.requires_grad for parameter in q.parameters()) for q in quantizers)
    assert not any(name.startswith("nemo_automodel") for name in sys.modules)
    return result
