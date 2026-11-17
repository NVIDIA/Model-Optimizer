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
from collections import OrderedDict

import pytest
import torch
from torch import nn

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields


class _Network(nn.Sequential):
    def forward(self, hidden_states, attention_mask=None):
        return super().forward(hidden_states)


def _problem():
    torch.manual_seed(37)
    student = _Network(
        OrderedDict(
            [
                ("mlp", nn.Linear(256, 256, bias=False)),
                ("activation", nn.SiLU()),
                ("moe", nn.Linear(256, 256, bias=False)),
                ("head", nn.Linear(256, 4, bias=False)),
            ]
        )
    )
    batches = [torch.randn(2, 3, 256) for _ in range(8)]
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "mlp.weight_quantizer",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
                {
                    "quantizer_name": "moe.weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(batches[0]),
    )
    teacher = _Network(
        OrderedDict(
            [
                ("mlp", nn.Linear(256, 256, bias=False)),
                ("activation", nn.SiLU()),
                ("moe", nn.Linear(256, 256, bias=False)),
                ("head", nn.Linear(256, 4, bias=False)),
            ]
        )
    )
    with torch.no_grad():
        for name in ("mlp", "moe"):
            layer = student.get_submodule(name)
            teacher.get_submodule(name).weight.copy_(layer.weight_quantizer(layer.weight) * 1.02)
        teacher.head.weight.copy_(student.head.weight)
    teacher.head.eval()
    return student, teacher, batches


def test_bypass_streams_independent_units_and_finalizes(tmp_path):
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
    try:
        result = mtd.bypass(student, teacher, batches, config)
    finally:
        for handle in handles:
            handle.remove()
    assert result is student
    assert len(teacher_inputs) == len(student_inputs) == 16
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
        prediction = reference.mlp(x)
        target = teacher.mlp(x).detach()
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
