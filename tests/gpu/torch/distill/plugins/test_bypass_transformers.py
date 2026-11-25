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

"""Real stable-5.23 text-model Bypass, selection, packing and original-weight prefixes."""

import copy
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", minversion="5.23")

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill._bypass_capture import BypassUnit
from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import register_hf_attentions_on_the_fly


@pytest.fixture
def native_cp_models():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    teacher = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    adapter = TransformersBypassAdapter()
    adapter.student = copy.deepcopy(teacher)
    return teacher, adapter


@pytest.mark.parametrize("validity", ["default", "all", "partial", "zero", "override"])
def test_native_cp_preserves_effective_loss_validity(native_cp_models, validity):
    teacher, adapter = native_cp_models
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
    if validity != "default":
        batch["loss_mask"] = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1]] if validity == "partial" else [[0] * 7]
        )
        if validity == "all":
            batch["loss_mask"].fill_(1)
        elif validity == "override":
            batch["valid_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.bool)
    if validity in ("partial", "zero"):
        with pytest.raises(ValueError):
            adapter._cp_inputs(teacher, batch)
    else:
        tokens, positions = adapter._cp_inputs(teacher, batch)
        torch.testing.assert_close(tokens, batch["input_ids"])
        torch.testing.assert_close(positions, torch.arange(7).unsqueeze(0))


@pytest.mark.parametrize("side", ["teacher", "student"])
@pytest.mark.parametrize("state", ["active_p", "disabled_p", "calibration_p", "kv_only"])
def test_native_cp_refuses_alternate_p_attention(native_cp_models, side, state):
    teacher, adapter = native_cp_models
    model = teacher if side == "teacher" else adapter.student
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
    original_attention_type = type(model.model.layers[0].self_attn)
    with torch.no_grad():
        expected = model(**batch).logits
    register_hf_attentions_on_the_fly(model)
    attention = QuantModuleRegistry.convert(model.model.layers[0].self_attn)
    for name in ("q", "k", "v", "p"):
        getattr(attention, f"{name}_bmm_quantizer").disable()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, expected, rtol=0, atol=0)
    assert original_attention_type in type(attention).__mro__
    quantizer = attention.p_bmm_quantizer
    quantizer.num_bits = (4, 3)
    quantizer.block_sizes = None
    quantizer.amax = torch.tensor(3.0)
    if state not in ("disabled_p", "kv_only"):
        quantizer.enable()
        assert attention._p_qdq_mode() == "fp8"
        if state == "calibration_p":
            quantizer.disable_quant()
            quantizer.enable_calib()
    elif state == "kv_only":
        for name in ("k", "v"):
            kv = getattr(attention, f"{name}_bmm_quantizer")
            kv.num_bits = (4, 3)
            kv.block_sizes = None
            kv.amax = torch.tensor(3.0)
            kv.enable()
    if state == "active_p":
        assert quantizer.is_enabled and quantizer._if_quant
        with pytest.raises(ValueError):
            adapter._cp_inputs(teacher, batch)
    else:
        tokens, positions = adapter._cp_inputs(teacher, batch)
        torch.testing.assert_close(tokens, batch["input_ids"])
        torch.testing.assert_close(positions, torch.arange(7).unsqueeze(0))


@pytest.mark.parametrize("side", ["teacher", "student"])
@pytest.mark.parametrize(
    ("family", "attention"),
    [
        ("llama", "default"),
        ("llama", "dynamic"),
        ("llama", "longrope"),
        ("qwen3", "default"),
        ("qwen3", "sliding"),
        ("qwen3_moe", "default"),
        ("qwen3_moe", "sliding"),
    ],
)
def test_native_cp_refuses_document_dependent_attention(family, attention, side):
    config_cls, model_cls = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "qwen3_moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    }[family]
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 4,
        "use_cache": False,
    }
    if family == "qwen3_moe":
        kwargs.update(moe_intermediate_size=16, num_experts=2, num_experts_per_tok=1)
    policy = {}
    if attention == "dynamic":
        policy["rope_parameters"] = {"rope_type": "dynamic", "factor": 2.0}
    elif attention == "longrope":
        policy["rope_parameters"] = {
            "rope_type": "longrope",
            "factor": 2.0,
            "original_max_position_embeddings": 4,
            "short_factor": [1.0] * 4,
            "long_factor": [2.0] * 4,
        }
    elif attention == "sliding":
        # Let HF realize the layer policy, including Qwen3's max_window_layers rule.
        policy.update(use_sliding_window=True, sliding_window=2)
        if family == "qwen3":
            policy["max_window_layers"] = 0
    teacher_config = config_cls(**kwargs, **(policy if side == "teacher" else {}))
    student_config = config_cls(**kwargs, **(policy if side == "student" else {}))
    for config in (teacher_config, student_config):
        config._attn_implementation = "sdpa"
    teacher = model_cls(teacher_config).to(torch.bfloat16).eval()
    adapter = TransformersBypassAdapter()
    adapter.student = model_cls(student_config).to(torch.bfloat16).eval()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
    if attention != "default":
        with pytest.raises(ValueError):
            adapter._cp_inputs(teacher, batch)
    else:
        tokens, positions = adapter._cp_inputs(teacher, batch)
        torch.testing.assert_close(tokens, batch["input_ids"])
        torch.testing.assert_close(positions, torch.arange(8).unsqueeze(0))


def test_teacher_replay_caches_model_scans_without_changing_targets():
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_next", rope_parameters={})
            self.mlp = torch.nn.Linear(1, 4)
            self.lm_head = torch.nn.Linear(4, 7)
            self.scans = 0
            self.batches = []

        def named_modules(self, *args, **kwargs):
            self.scans += 1
            return super().named_modules(*args, **kwargs)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, **kwargs):
            self.batches.append(tuple(input_ids.shape))
            return self.lm_head(self.mlp(input_ids.float().unsqueeze(-1)))

    teacher = Teacher()
    student = copy.deepcopy(teacher)
    adapter = TransformersBypassAdapter()
    adapter.student = student
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2, 2]]),
    }
    adapter.validate_batch(batch, mtd.BypassConfig(sequence_length=4))
    units = [BypassUnit("mlp", "mlp", "mlp")]
    targets = []
    for _ in range(2):
        current = []
        adapter.teacher_microbatch(
            teacher, units, batch, lambda _unit, record: current.append(record.target)
        )
        targets.append(torch.cat(current))
        if len(targets) == 1:
            scans = teacher.scans, student.scans
    torch.testing.assert_close(targets[0], targets[1], rtol=0, atol=0)
    assert (teacher.scans, student.scans) == scans
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "mlp.input_quantizer", "cfg": {"num_bits": 8}},
            ],
            "algorithm": "max",
        },
        lambda model: model(input_ids=batch["input_ids"]),
    )
    scans = teacher.scans, student.scans
    for enabled in (True, False, True):
        if enabled:
            student.mlp.input_quantizer.enable()
        else:
            student.mlp.input_quantizer.disable()
        teacher.batches.clear()
        current = []
        adapter.teacher_microbatch(
            teacher, units, batch, lambda _unit, record: current.append(record.target)
        )
        assert teacher.batches == ([(1, 2), (1, 2)] if enabled else [(2, 2)])
        torch.testing.assert_close(torch.cat(current), targets[0], rtol=0, atol=0)
        assert (teacher.scans, student.scans) == scans
    student.mlp.input_quantizer.disable()
    for name in (
        "q_bmm_quantizer",
        "k_bmm_quantizer",
        "v_bmm_quantizer",
        "p_bmm_quantizer",
        "future_activation",
    ):
        student.mlp.add_module(
            name, TensorQuantizer(QuantizerAttributeConfig(type="dynamic", enable=False))
        )
    for name in (
        "q_bmm_quantizer",
        "k_bmm_quantizer",
        "v_bmm_quantizer",
        "p_bmm_quantizer",
        "future_activation",
    ):
        quantizer = getattr(student.mlp, name)
        quantizer.enable()
        teacher.batches.clear()
        current = []
        adapter.teacher_microbatch(
            teacher, units, batch, lambda _unit, record: current.append(record.target)
        )
        assert teacher.batches == [(1, 2), (1, 2)], name
        torch.testing.assert_close(torch.cat(current), targets[0], rtol=0, atol=0)
        assert (teacher.scans, student.scans) == scans
        quantizer.disable()
    teacher.lm_head = torch.nn.Linear(4, 9)
    widths = []
    handle = teacher.lm_head.register_forward_hook(
        lambda _m, _a, output: widths.append(output.shape[-1])
    )
    try:
        adapter.teacher_microbatch(teacher, units, batch, lambda *_: None)
    finally:
        handle.remove()
    assert widths and set(widths) == {1}
    packed_4d = {
        **batch,
        "attention_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool).tril_(),
    }
    adapter.validate_batch(packed_4d, mtd.BypassConfig(sequence_length=4))
    current = []
    adapter.teacher_microbatch(
        teacher, units, packed_4d, lambda _unit, record: current.append(record.target)
    )
    torch.testing.assert_close(torch.cat(current), targets[0], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["prepare", "validate", "documents"])
def test_local_teacher_preparation_preserves_exception_identity(failure):
    class BrokenBatch(dict):
        def items(self):
            raise original

    class BrokenValidation(dict):
        def get(self, key, default=None):
            raise original

    class BrokenModel(torch.nn.Module):
        @property
        def config(self):
            raise original

    original = OSError("caller preparation failure")
    adapter = TransformersBypassAdapter()
    model = torch.nn.Linear(1, 1) if failure != "documents" else BrokenModel()
    batch = {"input_ids": torch.ones(1, 1, dtype=torch.long)}
    with pytest.raises(OSError) as caught:
        if failure == "prepare":
            adapter._document_teacher(model, [], BrokenBatch(batch), lambda *_: None)
        elif failure == "validate":
            adapter.teacher_microbatch(model, [], BrokenValidation(batch), lambda *_: None)
        else:
            adapter.teacher_microbatch(model, [], batch, lambda *_: None)
    assert caught.value is original
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    assert {"prepare": "items", "validate": "get", "documents": "config"}[failure] in frames


def _quantize(model, batch, bits="iq1_m", *, patterns=None):
    return mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[
                    {"quantizer_name": pattern, "cfg": {"num_bits": bits, "backend": "ggml"}}
                    for pattern in (patterns or ["*mlp*weight_quantizer*"])
                ],
                {"quantizer_name": "*mlp.gate.weight_quantizer*", "enable": False},
                {"quantizer_name": "*shared_expert_gate*", "enable": False},
            ],
            "algorithm": "max",
        },
        lambda m: m(**batch),
    )


def _moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        use_cache=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return Qwen3MoeForCausalLM(config).eval()


@pytest.mark.parametrize("bits", ["iq1_m", "iq2_s"])
def test_transformers_bypass_selection_and_prefix(bits, tmp_path):
    torch.manual_seed(41)
    teacher = _moe()
    batch = {
        "input_ids": torch.randint(3, 32, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    batches = [batch, {**batch, "input_ids": batch["input_ids"].roll(1, -1)}]
    student = _quantize(copy.deepcopy(teacher), batch, bits)
    initial = copy.deepcopy(student)
    teacher_before = copy.deepcopy(teacher.state_dict())
    suffix_before = copy.deepcopy(student.model.layers[1].state_dict())
    config = {
        "num_steps": 2,
        "layer_indices": [0],
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "sequence_length": 8,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }
    assert mtd.bypass(student, teacher, batches, config) is student
    assert all(
        torch.equal(v, suffix_before[n]) for n, v in student.model.layers[1].state_dict().items()
    )
    assert all(torch.equal(v, teacher_before[n]) for n, v in teacher.state_dict().items())
    prefix_teacher, prefix_student = copy.deepcopy(teacher), initial
    for model in (prefix_teacher, prefix_student):
        model.model.layers = nn.ModuleList(list(model.model.layers)[:1])
        model.config.num_hidden_layers = 1
    mtd.bypass(prefix_student, prefix_teacher, batches, {**config, "layer_indices": None})
    for (name, actual), (other, expected) in zip(
        student.model.layers[0].state_dict().items(),
        prefix_student.model.layers[0].state_dict().items(),
    ):
        assert name == other
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    full = _quantize(copy.deepcopy(teacher), batch, bits)
    before = full.model.layers[0].mlp.gate_proj.weight.detach().clone()
    mtd.bypass(full, teacher, [batch], {**config, "num_steps": 1, "layer_indices": None})
    assert not torch.equal(full.model.layers[0].mlp.gate_proj.weight, before)
    assert all(
        pinned_packed_weight(q, bits) is not None
        for q in full.model.layers[1].mlp.experts.gate_up_proj_weight_quantizers
    )

    if bits == "iq2_s":
        with torch.no_grad():
            expected = full(**batch).logits
        mto.save(full, tmp_path / "student.pt")
        restored = mto.restore(_moe(), tmp_path / "student.pt")
        with torch.no_grad():
            torch.testing.assert_close(restored(**batch).logits, expected, rtol=0, atol=0)
        for actual, reference in zip(
            restored.model.layers[1].mlp.experts.gate_up_proj_weight_quantizers,
            full.model.layers[1].mlp.experts.gate_up_proj_weight_quantizers,
        ):
            assert torch.equal(
                pinned_packed_weight(actual, bits), pinned_packed_weight(reference, bits)
            )


@pytest.mark.parametrize(
    ("family", "loss_key"),
    [
        ("qwen3_next", "valid_mask"),
        ("qwen3_vl_moe", "loss_mask"),
        ("deepseek_v3", "loss_mask"),
        ("deepseek_v32", "loss_mask"),
        ("kimi_k25", "loss_mask"),
        ("minimax_m3_vl", "loss_mask"),
        ("nemotron_h", "loss_mask"),
        ("llama", "loss_mask"),
        ("qwen3", "loss_mask"),
        ("qwen3_moe", "loss_mask"),
        ("qwen3_5_moe", "loss_mask"),
    ],
)
def test_transformers_packed_replay_matches_independent_documents(family, loss_key):
    from _test_utils.transformers_families import build_model, cases, hf_torch_kernels

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    torch.manual_seed(43)
    teacher = build_model(cases([family])[0], "cpu")
    ids = torch.randint(3, 100, (2, 8))
    docs = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 3, 3, 3]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 0, 1, 2, 0, 1, 2]])
    if family == "qwen3_vl_moe":
        positions = positions.unsqueeze(0).expand(3, -1, -1).clone()
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "_packed_seq_ids": docs,
        "position_ids": positions,
        loss_key: torch.arange(8).expand(2, -1).remainder(2).bool(),
    }
    adapter = TransformersBypassAdapter()
    with hf_torch_kernels(teacher):
        student = _quantize(
            copy.deepcopy(teacher),
            {k: v for k, v in batch.items() if k not in ("_packed_seq_ids", loss_key)},
            "iq2_s",
            patterns=adapter.weight_patterns(teacher) if family == "nemotron_h" else None,
        )
        units = adapter.resolve_units(student, teacher, mtd.BypassConfig())
        captured, separate = [], []
        replay_shapes = []
        handle = teacher.register_forward_pre_hook(
            lambda _m, _a, kwargs: replay_shapes.append(tuple(kwargs["input_ids"].shape)),
            with_kwargs=True,
        )
        try:
            adapter.teacher_microbatch(teacher, units, batch, lambda u, r: captured.append((u, r)))
        finally:
            handle.remove()
        assert any(shape[0] > 1 for shape in replay_shapes)
        for row in range(2):
            for document in docs[row].unique_consecutive():
                mask = docs[row] == document
                pos = (
                    positions[:, row : row + 1, mask]
                    if positions.ndim == 3
                    else positions[row : row + 1, mask]
                )
                item = {
                    "input_ids": ids[row : row + 1, mask],
                    "attention_mask": torch.ones_like(ids[row : row + 1, mask]),
                    "position_ids": pos,
                    loss_key: batch[loss_key][row : row + 1, mask],
                }
                adapter.teacher_microbatch(
                    teacher, units, item, lambda u, r: separate.append((u, r))
                )
        for unit in units:

            def targets(records):
                values = []
                for owner, record in records:
                    if owner.name == unit.name:
                        mask = record.valid_mask
                        values.append(
                            record.target.flatten(0, 1) if mask is None else record.target[mask]
                        )
                return torch.cat(values)

            torch.testing.assert_close(targets(captured), targets(separate), rtol=0.01, atol=0.01)
        mtd.bypass(
            student,
            teacher,
            [batch],
            {"num_steps": 1, "global_batch_size": 2, "sequence_length": 8},
        )
    assert all(torch.isfinite(p).all() for p in student.parameters())


def test_modelspec_drives_nemotron_ptq_and_bypass():
    from _test_utils.transformers_families import build_model, cases, hf_torch_kernels

    from modelopt.torch.distill.plugins.transformers import TransformersBypassAdapter

    torch.manual_seed(47)
    teacher = build_model(cases(["nemotron_h"])[0], "cpu")
    batch = {
        "input_ids": torch.randint(3, 100, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    adapter = TransformersBypassAdapter()
    patterns = adapter.weight_patterns(teacher)
    assert patterns
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    *[
                        {"quantizer_name": path, "cfg": {"num_bits": "iq2_s", "backend": "ggml"}}
                        for path in patterns
                    ],
                ],
                "algorithm": "max",
            },
            lambda m: m(**batch),
        )
        original = {n: p.detach().clone() for n, p in student.named_parameters()}
        mtd.bypass(
            student,
            teacher,
            [batch],
            {"num_steps": 1, "global_batch_size": 1, "sequence_length": 8},
        )
    changed = [n for n, p in student.named_parameters() if not torch.equal(p, original[n])]
    assert changed
    assert all("mixer" in name for name in changed)
    assert all(torch.isfinite(p).all() for p in student.parameters())
    teacher_ref = weakref.ref(teacher)
    del teacher
    gc.collect()
    assert teacher_ref() is None
    assert not adapter._providers


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4"])
def test_shared_iq2_routed_iq1_and_frozen_down_through_public_bypass(frozen_format, monkeypatch):
    from _test_utils.transformers_families import build_model, cases, hf_torch_kernels

    from modelopt.torch.distill._bypass_optimizer import Lion

    if frozen_format == "nvfp4" and not torch.cuda.is_available():
        pytest.skip("Native NVFP4 requires an authorized CUDA test allocation")
    device = "cuda" if frozen_format == "nvfp4" else "cpu"
    torch.manual_seed(53)
    teacher = build_model(cases(["qwen3_next"])[0], device)
    batch = {
        "input_ids": torch.randint(3, 100, (1, 8), device=device),
        "attention_mask": torch.ones(1, 8, device=device, dtype=torch.long),
    }
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    with hf_torch_kernels(teacher):
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*mlp.experts.*weight_quantizer*",
                        "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                    },
                    {
                        "quantizer_name": "*mlp.shared_expert.*weight_quantizer*",
                        "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                    },
                    {
                        "quantizer_name": "*mlp.experts.down_proj_weight_quantizers*",
                        "cfg": frozen_cfg,
                    },
                ],
                "algorithm": "max",
            },
            lambda m: m(**batch),
        )
        frozen = {
            n: v.detach().clone()
            for n, v in student.state_dict().items()
            if ".experts.down_proj" in n or ".mlp.gate." in n or "shared_expert_gate" in n
        }
        observed = set()
        original_step = Lion.step

        def record_gradients(optimizer, *args, **kwargs):
            for module in student.modules():
                state = getattr(module, "_local_repair_state", None)
                if state is not None and any(
                    p.grad is not None and p.grad.count_nonzero() for p in state.parameters()
                ):
                    observed.add(module.num_bits)
            return original_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(Lion, "step", record_gradients)
        mtd.bypass(
            student,
            teacher,
            [batch],
            {"num_steps": 1, "global_batch_size": 1, "sequence_length": 8},
        )
    assert observed == {"iq1_m", "iq2_s"}
    after = student.state_dict()
    assert frozen
    for name, value in frozen.items():
        torch.testing.assert_close(after[name], value, rtol=0, atol=0)


def test_teacher_replay_preserves_shared_input_and_output_embedding():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_next", rope_parameters={})
            self.embed_tokens = nn.Embedding(8, 4)
            self.mlp = nn.Linear(4, 4)
            self.lm_head = self.embed_tokens

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, **kwargs):
            hidden = self.mlp(self.embed_tokens(input_ids))
            return torch.nn.functional.linear(hidden, self.lm_head.weight)

    teacher = Teacher()
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = teacher.mlp(teacher.embed_tokens(tokens)).detach()
    captured = []
    adapter = TransformersBypassAdapter()
    adapter.student = copy.deepcopy(teacher)
    adapter.teacher_microbatch(
        teacher,
        [BypassUnit("mlp", "mlp", "mlp")],
        {"input_ids": tokens},
        lambda _unit, record: captured.append(record.target),
    )
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert "forward" not in teacher.embed_tokens.__dict__


def test_weight_patterns_keep_nested_router_frozen():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_moe")
    model.mlp = nn.Module()
    model.mlp.register_parameter("fused_weight", nn.Parameter(torch.randn(4, 4)))
    model.mlp.gate = nn.Linear(4, 4)
    model.mlp.experts = nn.Linear(4, 4)
    patterns = TransformersBypassAdapter().weight_patterns(model, {"modules": ["mlp"]})
    assert patterns
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[{"quantizer_name": pattern, "cfg": {"num_bits": 4}} for pattern in patterns],
            ],
            "algorithm": None,
        },
    )
    assert model.mlp.experts.weight_quantizer.is_enabled
    assert not model.mlp.gate.weight_quantizer.is_enabled
