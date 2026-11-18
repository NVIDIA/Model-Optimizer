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

"""Offline end-to-end public Bypass flows, including real native save/restore."""

import copy
import json
import random
import statistics
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import pytest
import torch
from _test_utils.bypass import BypassNetwork as _Network
from _test_utils.bypass import assert_exact_state, committed_checkpoint
from _test_utils.bypass import bypass_problem as _problem
from torch import nn

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill._bypass_checkpoint import _rng_state
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields
from modelopt.torch.quantization.utils.core_utils import enable_weight_access_and_writeback


@pytest.mark.parametrize("validation_failure", [False, True])
def test_bypass_streams_independent_units_and_finalizes(tmp_path, validation_failure):
    student, teacher, batches = _problem()
    teacher_state = copy.deepcopy(teacher.state_dict())
    initial = (student(batches[0]) - teacher(batches[0])).square().mean().item()
    baseline = copy.deepcopy(student)
    original_codes = {
        name: IQ_FORMAT_REGISTRY[bits]
        .pack(student.get_submodule(name).weight, student.get_submodule(name).weight_quantizer)
        .clone()
        for name, bits in [("mlp", "iq1_m"), ("moe", "iq2_s")]
    }
    teacher_inputs, student_inputs = [], []
    handles = [
        teacher.moe.register_forward_pre_hook(
            lambda m, args: teacher_inputs.append(args[0].detach().clone())
        ),
        student.moe.register_forward_pre_hook(
            lambda m, args: student_inputs.append(args[0].detach().clone())
        ),
    ]
    config = {
        "num_steps": 8,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "sequence_length": 3,
        "scheduler": "constant",
        "warmup_fraction": 0,
    }

    def validation_batches():
        yield batches[0]
        if validation_failure:
            raise RuntimeError("deliberate validation failure")

    try:
        if validation_failure:
            with pytest.raises(RuntimeError, match="deliberate validation failure"):
                mtd.bypass(
                    student, teacher, batches, config, validation_dataloader=validation_batches()
                )
            result = student
        else:
            result = mtd.bypass(
                student, teacher, batches, config, validation_dataloader=validation_batches()
            )
    finally:
        for handle in handles:
            handle.remove()
    assert result is student
    assert len(teacher_inputs) == len(student_inputs) == 18
    for actual, expected in zip(student_inputs, teacher_inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert student.training and teacher.training and not teacher.head.training
    assert all(torch.equal(teacher_state[k], v) for k, v in teacher.state_dict().items())
    torch.testing.assert_close(student.head.weight, baseline.head.weight, rtol=0, atol=0)
    assert (student(batches[0]) - teacher(batches[0])).square().mean() < initial
    for name, bits in [("mlp", "iq1_m"), ("moe", "iq2_s")]:
        q = student.get_submodule(name).weight_quantizer
        fmt = IQ_FORMAT_REGISTRY[bits]
        payload = pinned_packed_weight(q, bits)
        global_scale, local, _ = decode_scale_fields(original_codes[name], fmt)
        assert torch.equal(
            replace_scale_fields(payload, fmt, global_scale, local), original_codes[name]
        )
    config["micro_batch_size"] = 2
    mtd.bypass(baseline, teacher, batches, mtd.BypassConfig(**config))
    torch.testing.assert_close(student(batches[0]), baseline(batches[0]), rtol=0, atol=0)
    assert not any("repair" in name or "teacher" in name for name, _ in student.named_modules())
    mto.save(student, tmp_path / "student.pt")
    plain = _Network(
        OrderedDict(
            [
                ("mlp", nn.Linear(256, 256, bias=False)),
                ("activation", nn.SiLU()),
                ("moe", nn.Linear(256, 256, bias=False)),
                ("head", nn.Linear(256, 4, bias=False)),
            ]
        )
    )
    restored = mto.restore(plain, tmp_path / "student.pt")
    torch.testing.assert_close(restored(batches[0]), student(batches[0]), rtol=0, atol=0)


def test_bypass_refuses_aliases_and_cleans_failed_replay():
    student, teacher, batches = _problem()
    original = copy.deepcopy(student.state_dict())
    with pytest.raises(ValueError):
        mtd.bypass(student, teacher, batches, {"num_steps": 0, "modules": ["", "mlp"]})
    with pytest.raises(ValueError, match="sequence_length"):
        mtd.bypass(
            student,
            teacher,
            [{"hidden_states": batches[0]}],
            {"num_steps": 1, "global_batch_size": 2, "sequence_length": 4},
        )
    shared = copy.deepcopy(student)
    shared.moe.input_quantizer = shared.mlp.input_quantizer
    shared.mlp.input_quantizer._if_calib = True
    with pytest.raises(ValueError):
        mtd.bypass(shared, teacher, batches, {"num_steps": 0})
    assert shared.mlp.input_quantizer._if_calib
    teacher.mlp.weight = nn.Parameter(student.mlp.weight.detach())
    with pytest.raises(ValueError):
        mtd.bypass(student, teacher, batches, {"num_steps": 0, "modules": ["mlp"]})
    assert all(torch.equal(v, original[n]) for n, v in student.state_dict().items())
    student, teacher, batches = _problem()
    hook_counts = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in teacher.modules()]
    flags = [(p, p.requires_grad) for p in [*student.parameters(), *teacher.parameters()]]

    def fail(module, args):
        raise RuntimeError("deliberate replay failure")

    handle = student.moe.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="deliberate replay"):
            mtd.bypass(
                student,
                teacher,
                batches,
                {"num_steps": 1, "global_batch_size": 2, "sequence_length": 3},
            )
    finally:
        handle.remove()
    assert all(p.requires_grad == flag for p, flag in flags)
    assert all(
        len(m._forward_hooks) == after and len(m._forward_pre_hooks) == before
        for m, after, before in hook_counts
    )
    assert not any("repair" in n for n, _ in student.named_modules())


@pytest.mark.parametrize(
    "mask_kind",
    [
        "valid_mask",
        "loss_mask",
        "attention_mask",
        "attention_with_metadata",
        "additive_attention",
        "weighted_valid",
        "weighted_loss",
        "integer_valid",
        "integer_loss",
        "integer_attention",
        "attention4d",
        "explicit_attention4d",
    ],
)
def test_bypass_normalizes_valid_elements_over_the_whole_update(monkeypatch, mask_kind):
    from modelopt.torch.distill._bypass_optimizer import Lion
    from modelopt.torch.quantization.repair import IQRepairPolicy

    student, teacher, batches = _problem()
    reference = copy.deepcopy(student)
    x = batches[0]
    valid = torch.tensor([[True, True, True], [True, False, False]])
    attention4d = torch.ones(3, 3, dtype=torch.bool).tril()[None, None] & valid[:, None, None, :]
    if mask_kind in ("additive_attention", "weighted_valid", "weighted_loss", "attention4d"):
        key, mask = {
            "additive_attention": (
                "attention_mask",
                torch.zeros_like(valid, dtype=torch.float32).masked_fill_(~valid, -torch.inf),
            ),
            "weighted_valid": ("valid_mask", valid.float() * 0.5),
            "weighted_loss": ("loss_mask", valid.float() * 0.5),
            "attention4d": ("attention_mask", attention4d),
        }[mask_kind]
        before = copy.deepcopy(student.state_dict())
        with pytest.raises(ValueError):
            mtd.bypass(
                student,
                teacher,
                [{"hidden_states": x, key: mask}],
                {"num_steps": 1, "global_batch_size": 2, "sequence_length": 3, "modules": ["mlp"]},
            )
        assert all(torch.equal(value, before[name]) for name, value in student.state_dict().items())
        return
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(reference.mlp)
        # Match projection shapes so this isolates normalization from BLAS batch-shape roundoff.
        prediction = torch.cat([reference.mlp(sample) for sample in x.split(1)])
        target = torch.cat([teacher.mlp(sample) for sample in x.split(1)]).detach()
        torch.nn.functional.mse_loss(prediction[valid], target[valid]).backward()
        expected = [p.grad.detach().clone() for p in fields]
    finally:
        policy.close()
    observed = []
    original_step = Lion.step

    def record_actual_gradients(optimizer, *args, **kwargs):
        observed.extend(
            p.grad.detach().clone() for g in optimizer.param_groups for p in g["params"]
        )
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(Lion, "step", record_actual_gradients)
    key = mask_kind if mask_kind in ("valid_mask", "loss_mask") else "attention_mask"
    batch = {"hidden_states": x, key: valid}
    if mask_kind.startswith("integer_"):
        key = {
            "integer_valid": "valid_mask",
            "integer_loss": "loss_mask",
            "integer_attention": "attention_mask",
        }[mask_kind]
        batch = {"hidden_states": x, key: valid.long()}
    if mask_kind == "explicit_attention4d":
        batch.update(valid_mask=valid, attention_mask=attention4d)
    if mask_kind == "loss_mask":
        batch["attention_mask"] = torch.ones_like(valid)
    if mask_kind == "attention_with_metadata":
        batch.update(_packed_seq_ids=torch.tensor([[0, 0, 1], [0, 1, 1]]), padding_mask=~valid)
    mtd.bypass(
        student,
        teacher,
        [batch],
        {
            "modules": ["mlp"],
            "num_steps": 1,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "sequence_length": 3,
            "max_grad_norm": 1e9,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
    )
    assert len(observed) == len(expected)
    for actual, wanted in zip(observed, expected):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize(
    ("alias", "offload"),
    [
        ("parameter", False),
        ("storage_view", False),
        ("module", False),
        ("quantizer", False),
        ("router_weight", False),
        ("router_quantizer", False),
        ("parameter", True),
        ("storage_view", True),
        ("router_weight", True),
        ("teacher_storage", False),
        ("teacher_storage", "teacher"),
        ("teacher_storage", "student"),
        ("teacher_storage", "both"),
        ("teacher_storage", "preload_teacher"),
        ("teacher_storage", "preload_student"),
        ("buffer", False),
        ("buffer", True),
    ],
)
def test_bypass_refuses_selected_state_shared_with_excluded_owners(alias, offload):
    student, teacher, batches = _problem()
    selected = "mlp"
    if alias.startswith("router_"):

        class MoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(256, 1, bias=False)
                self.experts = nn.ModuleList([nn.Linear(256, 256, bias=False)])

            def forward(self, x):
                return self.experts[0](x) * self.gate(x).sigmoid()

        teacher = nn.Sequential(OrderedDict([("moe", MoE())]))
        student = mtq.quantize(
            copy.deepcopy(teacher),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                    },
                ],
                "algorithm": "max",
            },
            lambda model: model(batches[0]),
        )
        expert, router = student.moe.experts[0], student.moe.gate
        if alias == "router_weight":
            router.weight = nn.Parameter(expert.weight[:1].detach())
        else:
            router.weight_quantizer = expert.weight_quantizer
        selected = "moe"
    elif alias == "buffer":
        control = torch.ones(4)
        student.mlp.register_buffer("shared_control", control)
        student.moe.register_buffer("shared_control", control)
    elif alias == "teacher_storage":
        teacher.mlp.weight = nn.Parameter(student.mlp.weight.detach())
    elif alias == "parameter":
        student.moe.weight = student.mlp.weight
    elif alias == "storage_view":
        student.moe.weight = nn.Parameter(student.mlp.weight.detach().T)
    elif alias == "module":
        student.moe = student.mlp
    else:
        student.moe.weight_quantizer = student.mlp.weight_quantizer
    before = copy.deepcopy(student.state_dict())
    teacher_before = copy.deepcopy(teacher.state_dict())
    if offload in (True, "student", "both", "preload_student"):
        pytest.importorskip("accelerate").cpu_offload(
            student,
            execution_device="cpu",
            offload_buffers=True,
            preload_module_classes=[type(student).__name__] if offload == "preload_student" else [],
        )
    if offload in ("teacher", "both", "preload_teacher"):
        pytest.importorskip("accelerate").cpu_offload(
            teacher,
            execution_device="cpu",
            offload_buffers=True,
            preload_module_classes=[type(teacher).__name__] if offload == "preload_teacher" else [],
        )
    parameters = dict(student.named_parameters(remove_duplicate=False))
    flags = {name: parameter.requires_grad for name, parameter in parameters.items()}
    with pytest.raises(ValueError):
        mtd.bypass(student, teacher, batches, {"num_steps": 0, "modules": [selected]})
    assert all(
        parameter is parameters[name]
        for name, parameter in student.named_parameters(remove_duplicate=False)
    )
    assert {
        name: parameter.requires_grad
        for name, parameter in student.named_parameters(remove_duplicate=False)
    } == flags
    with mtq.utils.enable_weight_access_and_writeback(student, student, writeback=False):
        assert all(torch.equal(value, before[name]) for name, value in student.state_dict().items())
    with mtq.utils.enable_weight_access_and_writeback(teacher, teacher, writeback=False):
        assert all(
            torch.equal(value, teacher_before[name]) for name, value in teacher.state_dict().items()
        )
    assert not any("_local_repair_state" in name for name in student.state_dict())


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("teacher_offload", [False, True], ids=["resident", "cpu-offload"])
def test_public_bypass_preserves_caller_state_and_seeds_initial_reader(fail, teacher_offload):
    initial, teacher, batches = _problem()
    teacher.head.weight.requires_grad_(False)
    if teacher_offload:
        pytest.importorskip("accelerate").cpu_offload(teacher, execution_device="cpu")
    teacher_flags = {
        name: parameter.requires_grad for name, parameter in teacher.named_parameters()
    }
    teacher_modes = {module: module.training for module in teacher.modules()}

    class RandomBatches:
        def __init__(self):
            self.draws = []

        def draw(self):
            values = (random.random(), np.random.random(), torch.rand(()).item())
            self.draws.append(values)
            return sum(values) * 0.01

        def __iter__(self):
            self.draw()
            return self.read()

        def read(self):
            for index, batch in enumerate(batches):
                if fail and index == 1:
                    raise OSError("reader failure")
                yield batch + self.draw()

    previous = None
    for caller_seed in (101, 203):
        random.seed(caller_seed)
        np.random.seed(caller_seed)
        torch.manual_seed(caller_seed)
        caller = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        data, student = RandomBatches(), copy.deepcopy(initial)
        recipe = {
            "num_steps": 2,
            "global_batch_size": 2,
            "sequence_length": 3,
            "scheduler": "constant",
            "warmup_fraction": 0,
        }
        if fail:
            with pytest.raises(OSError):
                mtd.bypass(student, teacher, data, recipe)
        else:
            mtd.bypass(student, teacher, data, recipe)
        assert {
            name: parameter.requires_grad for name, parameter in teacher.named_parameters()
        } == teacher_flags
        assert all(module.training == mode for module, mode in teacher_modes.items())
        assert random.getstate() == caller[0]
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == caller[1][0]
        np.testing.assert_array_equal(actual_numpy[1], caller[1][1])
        assert actual_numpy[2:] == caller[1][2:]
        assert torch.equal(torch.get_rng_state(), caller[2])
        assert all(
            torch.equal(torch.cuda.get_rng_state(i), state) for i, state in enumerate(caller[3])
        )
        if previous is not None:
            draws, state = previous
            assert data.draws == draws
            assert all(
                torch.equal(value, state[name]) for name, value in student.state_dict().items()
            )
        previous = (data.draws, copy.deepcopy(student.state_dict()))


def test_bypass_normalizes_valid_elements_over_the_whole_update(monkeypatch):
    from modelopt.torch.distill._bypass_optimizer import Lion
    from modelopt.torch.quantization.repair import IQRepairPolicy

    student, teacher, batches = _problem()
    reference = copy.deepcopy(student)
    x = batches[0]
    valid = torch.tensor([[True, True, True], [True, False, False]])
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(reference.mlp)
        # Match projection shapes so this isolates normalization from BLAS batch-shape roundoff.
        prediction = torch.cat([reference.mlp(sample) for sample in x.split(1)])
        target = torch.cat([teacher.mlp(sample) for sample in x.split(1)]).detach()
        torch.nn.functional.mse_loss(prediction[valid], target[valid]).backward()
        expected = [p.grad.detach().clone() for p in fields]
    finally:
        policy.close()
    observed = []
    original_step = Lion.step

    def record_actual_gradients(optimizer, *args, **kwargs):
        observed.extend(
            p.grad.detach().clone() for g in optimizer.param_groups for p in g["params"]
        )
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(Lion, "step", record_actual_gradients)
    mtd.bypass(
        student,
        teacher,
        [{"hidden_states": x, "valid_mask": valid}],
        {
            "modules": ["mlp"],
            "num_steps": 1,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "sequence_length": 3,
            "max_grad_norm": 1e9,
            "scheduler": "constant",
            "warmup_fraction": 0,
        },
    )
    assert len(observed) == len(expected)
    for actual, wanted in zip(observed, expected):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-10)


@pytest.mark.timeout(180)
def test_native_valid_token_mse_bounds_saved_state():
    pytest.importorskip("triton")
    from modelopt.torch.kernels.quantization.repair import masked_mse_sum

    if not torch.cuda.is_available():
        pytest.skip("Fused MSE requires CUDA")
    torch.manual_seed(113)
    for empty in (False, True):
        prediction = (
            torch.randn(1, 32768, 512, device="cuda", dtype=torch.bfloat16)[..., ::2]
            .detach()
            .requires_grad_(True)
        )
        target = torch.randn_like(prediction)
        valid = torch.rand(1, 32768, device="cuda") > 0.2
        if empty:
            valid.zero_()
        reference = prediction.detach().float().requires_grad_(True)
        oracle = torch.where(valid[..., None], reference - target.float(), 0).square().sum()
        expected = torch.autograd.grad(oracle, reference)[0].to(prediction.dtype)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda x: saved.append((x.shape, x.dtype)) or x, lambda x: x
        ):
            actual = masked_mse_sum(prediction, target, valid, 256)
        torch.testing.assert_close(actual, oracle, rtol=1e-6, atol=0)
        observed = torch.autograd.grad(actual, prediction)[0]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert (prediction.shape, torch.float32) not in saved


@pytest.mark.parametrize("offload", ["cpu", "disk"])
def test_public_bypass_offloaded_nested_boundary_matches_resident(offload, tmp_path):
    accelerate = pytest.importorskip("accelerate")
    initial, target, batches = _problem()
    initial.mlp, target.mlp = nn.Sequential(initial.mlp), nn.Sequential(target.mlp)
    config = {
        "num_steps": 2,
        "global_batch_size": 2,
        "micro_batch_size": 1,
        "sequence_length": 3,
        "modules": ["mlp"],
        "checkpoint_interval": 1,
    }
    reference = copy.deepcopy(initial)
    mtd.bypass(
        reference,
        copy.deepcopy(target),
        batches,
        {**config, "checkpoint_dir": str(tmp_path / "reference")},
    )

    class BufferOnlyComposite(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.25))
            self.child = nn.Identity()
            self.child.register_buffer("offset", torch.tensor(0.125))

        def forward(self, hidden_states):
            return hidden_states * self.scale + self.child.offset

    def fresh_pair(label, changed=None, *, preload=False):
        student, teacher = copy.deepcopy(initial), copy.deepcopy(target)
        if changed is not None:
            root = teacher if changed == "teacher" else student
            with torch.no_grad():
                root.head.weight.add_(0.001)  # Frozen and outside selected mlp.
        for role, root in (("student", student), ("teacher", teacher)):
            if preload == "buffers":
                root.add_module("buffer_only", BufferOnlyComposite())
            elif preload:
                root.register_parameter("root_weight", nn.Parameter(torch.ones(())))
            preload_class = BufferOnlyComposite if preload == "buffers" else type(root)
            access = {
                "execution_device": "cpu",
                "offload_buffers": True,
                "preload_module_classes": [preload_class.__name__] if preload else None,
            }
            if offload == "cpu":
                accelerate.cpu_offload(root, **access)
            else:
                accelerate.disk_offload(root, tmp_path / f"{label}-{role}", **access)
            assert not hasattr(root.mlp, "_hf_hook")
        return student, teacher

    def caller_state(*models):
        return [
            (
                {name: (p.device, p.requires_grad) for name, p in model.named_parameters()},
                {name: b.device for name, b in model.named_buffers()},
                {name: m.training for name, m in model.named_modules()},
            )
            for model in models
        ]

    def assert_caller_state(models, expected):
        for current, previous in zip(caller_state(*models), expected, strict=True):
            assert current[0] == previous[0]  # Parameter devices and requires_grad.
            assert current[2] == previous[2]  # Per-module caller modes.
            assert all(current[1][name] == device for name, device in previous[1].items())
            for name in current[1].keys() - previous[1].keys():
                # Finalization adds real pinned IQ buffers to the managed layer.
                assert "_ggml_pinned_" in name and current[1][name].type == "meta"

    class Interrupted(list):
        def __iter__(self):
            for index, batch in enumerate(list.__iter__(self)):
                if index == 1:
                    raise OSError("Simulated offloaded data-reader interruption")
                yield batch

    student, teacher = fresh_pair("interrupted")
    before = caller_state(student, teacher)
    caller_rng = _rng_state([])
    with pytest.raises(OSError, match="offloaded data-reader"):
        mtd.bypass(
            student,
            teacher,
            Interrupted(batches),
            {**config, "checkpoint_dir": str(tmp_path / "resumed")},
        )
    assert_caller_state((student, teacher), before)
    assert_exact_state(_rng_state([]), caller_rng)
    generation, saved = committed_checkpoint(tmp_path / "resumed")
    assert saved["step"] == 1
    for role in ("student", "teacher"):
        refused, changed_target = fresh_pair(f"refused-{role}", changed=role)
        before = caller_state(refused, changed_target)
        # Forward reads real backing state while returning all owners to meta.
        outputs = [root(batches[0]).detach().clone() for root in (refused, changed_target)]
        with pytest.raises(ValueError):
            mtd.bypass(
                refused,
                changed_target,
                batches,
                {**config, "resume_from": str(generation)},
            )
        assert_caller_state((refused, changed_target), before)
        assert_exact_state(_rng_state([]), caller_rng)
        for root, output in zip((refused, changed_target), outputs, strict=True):
            assert_exact_state(root(batches[0]), output)

    resumed, teacher = fresh_pair("resumed")
    before = caller_state(resumed, teacher)
    assert (
        mtd.bypass(
            resumed,
            teacher,
            batches,
            {
                **config,
                "resume_from": str(tmp_path / "resumed"),
                "checkpoint_dir": str(tmp_path / "resumed"),
            },
        )
        is resumed
    )
    assert_caller_state((resumed, teacher), before)
    assert_exact_state(_rng_state([]), caller_rng)
    _, actual = committed_checkpoint(tmp_path / "resumed")
    _, expected = committed_checkpoint(tmp_path / "reference")
    assert actual["step"] == expected["step"] == 2
    for key in ("fields", "optimizers", "rng_state", "data_position"):
        assert_exact_state(actual[key], expected[key])
    assert not hasattr(resumed.mlp[0].weight_quantizer, "_local_repair_state")
    # Read one real owner at a time, including the final pinned codes and frozen
    # moe/head outside the selected boundary; never onload either whole model.
    for actual_root, expected_root in ((resumed, reference), (teacher, target)):
        for path in ("mlp.0", "moe", "head"):
            owner = actual_root.get_submodule(path)
            with enable_weight_access_and_writeback(owner, actual_root, writeback=False):
                assert_exact_state(
                    owner.state_dict(), expected_root.get_submodule(path).state_dict()
                )
    assert_exact_state(resumed(batches[0]), reference(batches[0]))
    assert_exact_state(teacher(batches[0]), target(batches[0]))

    # Composite preload hooks must refuse before onloading descendants, whether
    # their independent state consists of parameters or only persistent buffers.
    class PreflightCursor(list):
        started = False

        def __iter__(self):
            self.started = True
            return super().__iter__()

    for preload in (True, "buffers"):
        composite, composite_teacher = fresh_pair(f"preload-{preload}", preload=preload)
        if preload == "buffers":
            for root in (composite, composite_teacher):
                owner = root.buffer_only
                assert not list(owner.parameters())
                assert owner.scale.is_meta and owner.child.offset.is_meta
        before = caller_state(composite, composite_teacher)
        outputs = [root(batches[0]).detach().clone() for root in (composite, composite_teacher)]
        data = PreflightCursor(batches)
        with pytest.raises(ValueError):
            mtd.bypass(
                composite,
                composite_teacher,
                data,
                {**config, "checkpoint_dir": str(tmp_path / f"preload-{preload}-checkpoint")},
            )
        assert not data.started  # Refuse during identity, before priming/training.
        assert_caller_state((composite, composite_teacher), before)
        assert_exact_state(_rng_state([]), caller_rng)
        for root, output in zip((composite, composite_teacher), outputs, strict=True):
            assert_exact_state(root(batches[0]), output)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("first", ["distill", "quantization"])
def test_distill_quantization_import_order_keeps_trainer_plugin(first):
    pytest.importorskip("transformers")
    code = "\n".join(
        [
            f"import modelopt.torch.{first}",
            "import sys",
            "import modelopt.torch.distill as mtd",
            "import modelopt.torch.quantization as mtq",
            "assert callable(mtd.bypass)",
            "assert 'modelopt.torch.quantization.plugins.transformers_trainer' in sys.modules",
            "assert hasattr(mtq.plugins, 'QATTrainer')",
            "assert hasattr(mtq.plugins, 'QADTrainer')",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=180)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory and kernel qualification")
def test_bypass_cuda_mse_bounds_scratch_and_preserves_masked_vjp(monkeypatch, capsys):
    from modelopt.torch.distill._bypass_capture import _MSE_CHUNK_ELEMENTS, mse_statistics

    prediction = torch.ones((1, 32768, 4096), device="cuda", dtype=torch.bfloat16)
    prediction.requires_grad_(True)
    target = torch.zeros_like(prediction)
    mask = torch.arange(32768, device="cuda").remainder(2).eq(0).unsqueeze(0)
    metrics = {"device": torch.cuda.get_device_name(), "shape": list(prediction.shape)}
    for elements in (1 << 18, _MSE_CHUNK_ELEMENTS["cuda"]):
        monkeypatch.setitem(_MSE_CHUNK_ELEMENTS, "cuda", elements)
        times, peaks = [], []
        for repeat in range(4):
            prediction.grad = None
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss, count = mse_statistics(prediction, target, mask)
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() - baseline
            if repeat:
                times.append(elapsed)
                peaks.append(peak)
        assert loss.item() == count.item() == prediction.numel() // 2
        assert torch.all(prediction.grad[:, ::2] == 2)
        assert torch.all(prediction.grad[:, 1::2] == 0)
        # One narrow gradient plus chunk-sized FP32 operands; prohibit full FP32 copies.
        budget = prediction.numel() * prediction.element_size() + 16 * elements + (16 << 20)
        assert max(peaks) <= budget
        metrics[str(elements)] = {
            "median_forward_backward_seconds": statistics.median(times),
            "extra_peak_allocated_bytes": max(peaks),
            "extra_budget_bytes": budget,
        }
    with capsys.disabled():
        print("Bypass CUDA MSE metrics: " + json.dumps(metrics, sort_keys=True), flush=True)
