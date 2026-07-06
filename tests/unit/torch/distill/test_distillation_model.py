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

"""Unit tests for the ``DistillationModel`` class contract and the distill DM registries.

These tests exercise ``modelopt/torch/distill/distillation_model.py`` directly with tiny,
hand-weighted ``nn.Linear`` stacks so every loss value can be computed exactly by hand.
End-to-end ``mtd.convert``/``mtd.export`` flows on vision models are covered by
``test_distill.py``; this file targets the wrapper's per-layer capture-hook machinery,
``compute_kd_loss`` aggregation semantics, mode propagation, and error paths instead.
"""

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss

import modelopt.torch.distill as mtd
from modelopt.torch.distill.distillation_model import DistillationModel
from modelopt.torch.distill.loss_balancers import STUDENT_LOSS_KEY
from modelopt.torch.distill.registry import DistillationDMRegistry, LayerwiseDistillationDMRegistry

# Fixed input: batch of one, two features. All expected activations below derive from it.
INPUT = torch.tensor([[1.0, 1.0]])

# Hand-picked weights (bias-free) making every activation exact in fp32:
#   student: fc1 = identity -> [1, 1];    fc2 = [1, 1] -> 2
#   teacher: fc1 = 2*identity -> [2, 2];  fc2 = [1, 1] -> 4
STUDENT_FC1_W = [[1.0, 0.0], [0.0, 1.0]]
TEACHER_FC1_W = [[2.0, 0.0], [0.0, 2.0]]
SHARED_FC2_W = [[1.0, 1.0]]

STUDENT_FC1_OUT = torch.tensor([[1.0, 1.0]])
TEACHER_FC1_OUT = torch.tensor([[2.0, 2.0]])
STUDENT_OUT = torch.tensor([[2.0]])
TEACHER_OUT = torch.tensor([[4.0]])

# MSE(fc1) = mean((1-2)^2, (1-2)^2) = 1.0 ; MSE(fc2/output) = (2-4)^2 = 4.0
FC1_MSE = 1.0
FC2_MSE = 4.0


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class ScaledL1Loss(Loss):
    """Loss accepting an extra kwarg, to verify ``**loss_fn_kwargs`` forwarding."""

    def forward(self, pred, target, scale=1.0):
        return scale * (pred - target).abs().mean()


class BiasedMSELoss(Loss):
    """Loss with a learnable parameter, to verify ``loss_modules`` collection."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        return F.mse_loss(pred + self.bias, target)


class ParamBalancer(mtd.DistillationLossBalancer):
    """Balancer with a parameter — must be rejected by config validation."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, loss):
        return sum(loss.values())


def _assign(linear: nn.Linear, rows: list[list[float]]) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(rows))


def _make_models() -> tuple[StudentNet, TeacherNet]:
    student, teacher = StudentNet(), TeacherNet()
    _assign(student.fc1, STUDENT_FC1_W)
    _assign(student.fc2, SHARED_FC2_W)
    _assign(teacher.fc1, TEACHER_FC1_W)
    _assign(teacher.fc2, SHARED_FC2_W)
    return student, teacher


def _convert(student, teacher, criterion, loss_balancer=None, **extra) -> DistillationModel:
    config = {
        "teacher_model": teacher,
        "criterion": criterion,
        "loss_balancer": loss_balancer,
        **extra,
    }
    return mtd.convert(student, mode=[("kd_loss", config)])


def _output_only_model(**extra) -> DistillationModel:
    """Model distilling only final outputs, no balancer needed."""
    student, teacher = _make_models()
    return _convert(student, teacher, nn.MSELoss(), **extra)


def _two_pair_model(student_weights=(0.25, 0.25)) -> DistillationModel:
    """Model with per-layer (fc1, fc2) MSE criteria and a static balancer."""
    student, teacher = _make_models()
    criterion = {
        ("fc1", "fc1"): nn.MSELoss(),
        ("fc2", "fc2"): nn.MSELoss(),
    }
    with pytest.warns(UserWarning, match="do not sum to 1"):
        balancer = mtd.StaticLossBalancer(list(student_weights))
    return _convert(student, teacher, criterion, loss_balancer=balancer)


def _layer_pairs(model: DistillationModel) -> list[tuple[nn.Module, nn.Module]]:
    return list(model._layers_to_loss.keys())


# -------------------------------------------------------------------------------------------------
# Construction via the mode/convert path
# -------------------------------------------------------------------------------------------------


def test_convert_wraps_student_and_preserves_output():
    student, teacher = _make_models()
    reference_out = student(INPUT)

    model = _convert(student, teacher, nn.MSELoss())

    assert isinstance(model, DistillationModel)
    assert isinstance(model, StudentNet)  # dynamic class subclasses the original
    assert model is student  # conversion happens in-place
    assert model.teacher_model is teacher
    assert model.loss_balancer is None
    assert len(model.loss_modules) == 0  # MSELoss has no parameters

    # Forward returns the student's output, unchanged by the wrapping.
    out = model(INPUT)
    assert torch.equal(out, reference_out)
    assert torch.equal(out, STUDENT_OUT)


def test_teacher_model_like_class_is_instantiated():
    torch.manual_seed(0)
    student, _ = _make_models()
    model = _convert(student, TeacherNet, nn.MSELoss())

    assert isinstance(model.teacher_model, TeacherNet)
    assert model.teacher_model is not TeacherNet


def test_teacher_frozen_student_trainable():
    model = _output_only_model()

    assert all(not p.requires_grad for p in model.teacher_model.parameters())
    student_params = [model.fc1.weight, model.fc2.weight]
    assert all(p.requires_grad for p in student_params)


def test_loss_modules_collects_only_parameterized_losses():
    student, teacher = _make_models()
    loss_fn = BiasedMSELoss()
    model = _convert(student, teacher, {("fc1", "fc1"): loss_fn})

    assert len(model.loss_modules) == 1
    assert model.loss_modules[0] is loss_fn
    # The loss parameter must be reachable from the wrapper (e.g. for optimizers).
    assert any(p is loss_fn.bias for p in model.parameters())
    assert any(k.startswith("_loss_modules") for k in model.state_dict())


# -------------------------------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------------------------------


def test_registries_are_distinct_with_expected_prefixes():
    assert DistillationDMRegistry is not LayerwiseDistillationDMRegistry
    assert DistillationDMRegistry.prefix == "Distill"
    assert LayerwiseDistillationDMRegistry.prefix == "LayerwiseDistill"


def test_convert_registers_student_class_in_distill_registry():
    model = _output_only_model()

    assert StudentNet in DistillationDMRegistry
    dm_cls = DistillationDMRegistry[StudentNet]
    assert type(model) is dm_cls
    assert issubclass(dm_cls, DistillationModel)
    assert issubclass(dm_cls, StudentNet)
    assert dm_cls.__name__ == "DistillStudentNet"
    # The kd_loss convert path registers under this string key.
    assert "student_class" in DistillationDMRegistry
    # The layerwise registry is independent and untouched by the kd_loss mode.
    assert StudentNet not in LayerwiseDistillationDMRegistry


def test_registry_lookup_of_unregistered_class_raises():
    class NeverRegistered(nn.Module):
        def forward(self, x):
            return x

    assert NeverRegistered not in DistillationDMRegistry
    assert DistillationDMRegistry.get(NeverRegistered) is None
    with pytest.raises(KeyError):
        DistillationDMRegistry[NeverRegistered]


# -------------------------------------------------------------------------------------------------
# Intermediate-output capture hooks
# -------------------------------------------------------------------------------------------------


def test_hooks_registered_per_layer_pair():
    model = _two_pair_model()

    # One student + one teacher hook per pair.
    assert len(model._hook_handles) == 4
    for student_layer, teacher_layer in _layer_pairs(model):
        assert len(student_layer._forward_hooks) == 1
        assert len(teacher_layer._forward_hooks) == 1
        # Capture slots initialized empty.
        assert student_layer._intermediate_output is None
        assert teacher_layer._intermediate_output is None


def test_forward_captures_intermediate_outputs():
    model = _two_pair_model()
    model(INPUT)

    (s_fc1, t_fc1), (s_fc2, t_fc2) = _layer_pairs(model)
    assert torch.equal(s_fc1._intermediate_output, STUDENT_FC1_OUT)
    assert torch.equal(t_fc1._intermediate_output, TEACHER_FC1_OUT)
    assert torch.equal(s_fc2._intermediate_output, STUDENT_OUT)
    assert torch.equal(t_fc2._intermediate_output, TEACHER_OUT)

    # Student activations stay in the autograd graph; teacher ran under no_grad.
    assert s_fc1._intermediate_output.grad_fn is not None
    assert not t_fc1._intermediate_output.requires_grad


def test_second_forward_overwrites_captured_outputs():
    model = _two_pair_model()
    model(INPUT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate-capture warnings are expected here
        model(2 * INPUT)

    (s_fc1, t_fc1), _ = _layer_pairs(model)
    assert torch.equal(s_fc1._intermediate_output, 2 * STUDENT_FC1_OUT)
    assert torch.equal(t_fc1._intermediate_output, 2 * TEACHER_FC1_OUT)


def test_duplicate_capture_warns_only_for_teacher_in_eval():
    model = _output_only_model()
    model.eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(INPUT)
        model(INPUT)

    messages = [str(w.message) for w in caught]
    # Teacher hook warns whenever an output is still stored; the student hook
    # only warns in training mode (eval double-forward is a legitimate pattern).
    assert sum("Teacher's Module" in m for m in messages) == 1
    assert not any("Student's Module" in m for m in messages)


def test_compute_kd_loss_between_forwards_avoids_warnings():
    model = _two_pair_model()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            model(INPUT)
            model.compute_kd_loss(student_loss=torch.tensor(2.0))

    assert not any("intermediate output" in str(w.message) for w in caught)


# -------------------------------------------------------------------------------------------------
# compute_kd_loss aggregation
# -------------------------------------------------------------------------------------------------


def test_output_only_kd_loss_exact_value():
    model = _output_only_model()
    model(INPUT)

    loss = model.compute_kd_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(FC2_MSE)  # (2 - 4)^2
    assert loss.requires_grad


def test_two_pair_weighted_sum_exact_value():
    model = _two_pair_model(student_weights=(0.25, 0.25))
    model(INPUT)

    total = model.compute_kd_loss(student_loss=torch.tensor(2.0))
    # StaticLossBalancer: 0.25 * MSE(fc1) + 0.25 * MSE(fc2) + (1 - 0.5) * student_loss
    #                   = 0.25 * 1.0     + 0.25 * 4.0      + 0.5 * 2.0 = 2.25
    assert total.item() == pytest.approx(0.25 * FC1_MSE + 0.25 * FC2_MSE + 0.5 * 2.0)


def test_skip_balancer_returns_loss_dict_with_expected_keys_and_values():
    model = _two_pair_model()
    model(INPUT)

    loss_dict = model.compute_kd_loss(student_loss=torch.tensor(2.0), skip_balancer=True)
    assert isinstance(loss_dict, dict)
    # Keys are "<LossClassName>_<pair index>" in criterion order, plus the student key.
    assert list(loss_dict.keys()) == [STUDENT_LOSS_KEY, "MSELoss_0", "MSELoss_1"]
    assert loss_dict[STUDENT_LOSS_KEY].item() == pytest.approx(2.0)
    assert loss_dict["MSELoss_0"].item() == pytest.approx(FC1_MSE)
    assert loss_dict["MSELoss_1"].item() == pytest.approx(FC2_MSE)


def test_compute_kd_loss_clears_captured_outputs():
    model = _two_pair_model()
    model(INPUT)
    model.compute_kd_loss(student_loss=torch.tensor(0.0))

    for student_layer, teacher_layer in _layer_pairs(model):
        assert student_layer._intermediate_output is None
        assert teacher_layer._intermediate_output is None


def test_loss_reduction_fn_applied_to_each_pair_loss():
    student, teacher = _make_models()
    model = _convert(student, teacher, {("fc1", "fc1"): nn.MSELoss(reduction="none")})
    model(INPUT)

    # Unreduced elementwise loss is [[1., 1.]]; the reduction fn sums it to 2.0.
    loss = model.compute_kd_loss(loss_reduction_fn=lambda t: t.sum())
    assert loss.item() == pytest.approx(2.0)


def test_loss_fn_kwargs_forwarded_to_criterion():
    student, teacher = _make_models()
    model = _convert(student, teacher, {("fc2", "fc2"): ScaledL1Loss()})
    model(INPUT)

    # |2 - 4| * scale = 2 * 3 = 6
    loss = model.compute_kd_loss(scale=3.0)
    assert loss.item() == pytest.approx(6.0)


def test_student_loss_without_balancer_raises():
    model = _output_only_model()
    model(INPUT)

    with pytest.raises(AssertionError, match="without using Loss Balancer"):
        model.compute_kd_loss(student_loss=torch.tensor(1.0))


def test_backward_flows_to_student_only():
    model = _output_only_model()
    model(INPUT)

    loss = model.compute_kd_loss()
    loss.backward()

    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None
    assert torch.any(model.fc2.weight.grad != 0)
    assert all(p.grad is None for p in model.teacher_model.parameters())


# -------------------------------------------------------------------------------------------------
# train/eval mode propagation
# -------------------------------------------------------------------------------------------------


def test_teacher_forced_to_eval_during_forward():
    model = _output_only_model()

    # NOTE: `.train()` on the wrapper inadvertently propagates to the teacher child module
    # (acknowledged in the source); the forward pass re-forces eval before the teacher runs.
    model.train()
    assert model.training
    assert model.teacher_model.training

    model(INPUT)
    assert not model.teacher_model.training
    assert model.training  # student side unaffected

    # requires_grad stays off for the teacher regardless of mode flips.
    assert all(not p.requires_grad for p in model.teacher_model.parameters())


def test_mode_switch_clears_captured_outputs():
    model = _two_pair_model()
    model.train()
    model(INPUT)
    assert all(
        s._intermediate_output is not None and t._intermediate_output is not None
        for s, t in _layer_pairs(model)
    )

    model.eval()  # train -> eval clears stale captures
    assert all(
        s._intermediate_output is None and t._intermediate_output is None
        for s, t in _layer_pairs(model)
    )

    model(INPUT)
    model.train()  # eval -> train clears as well
    assert all(
        s._intermediate_output is None and t._intermediate_output is None
        for s, t in _layer_pairs(model)
    )


def test_same_mode_train_call_preserves_captured_outputs():
    model = _two_pair_model()
    model.train()
    model(INPUT)

    model.train()  # no mode change -> captures must survive
    assert all(
        s._intermediate_output is not None and t._intermediate_output is not None
        for s, t in _layer_pairs(model)
    )


# -------------------------------------------------------------------------------------------------
# Forward-restriction and hiding context managers
# -------------------------------------------------------------------------------------------------


def test_only_teacher_forward_returns_teacher_output():
    model = _two_pair_model()

    with model.only_teacher_forward():
        out = model(INPUT)

    assert torch.equal(out, TEACHER_OUT)  # teacher's output, not the student's
    for student_layer, teacher_layer in _layer_pairs(model):
        assert student_layer._intermediate_output is None  # student never ran
        assert teacher_layer._intermediate_output is not None
    assert not model._only_teacher_fwd  # flag restored on exit


def test_only_student_forward_skips_teacher():
    model = _two_pair_model()

    with model.only_student_forward():
        out = model(INPUT)

    assert torch.equal(out, STUDENT_OUT)
    for student_layer, teacher_layer in _layer_pairs(model):
        assert student_layer._intermediate_output is not None
        assert teacher_layer._intermediate_output is None  # teacher never ran
    assert not model._only_student_fwd


def test_hide_teacher_model_context_manager():
    model = _output_only_model()
    teacher = model.teacher_model

    with model.hide_teacher_model():
        assert model.teacher_model is not teacher
        assert type(model.teacher_model) is nn.Module  # bare placeholder
    assert model.teacher_model is teacher  # restored

    with model.hide_teacher_model(enable=False):
        assert model.teacher_model is teacher  # no-op when disabled


def test_hide_loss_modules_context_manager():
    student, teacher = _make_models()
    model = _convert(student, teacher, {("fc1", "fc1"): BiasedMSELoss()})
    loss_modules = model.loss_modules
    assert len(loss_modules) == 1

    with model.hide_loss_modules():
        assert len(model.loss_modules) == 0
    assert model.loss_modules is loss_modules


# -------------------------------------------------------------------------------------------------
# State dict exposure and restore
# -------------------------------------------------------------------------------------------------


def test_minimal_state_dict_matches_plain_student_keys():
    model = _output_only_model()  # expose_minimal_state_dict defaults to True

    assert set(model.state_dict()) == set(StudentNet().state_dict())


def test_full_state_dict_roundtrip_restores_student_values():
    model = _output_only_model(expose_minimal_state_dict=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(k.startswith("_teacher_model") for k in state)

    with torch.no_grad():
        model.fc1.weight.fill_(123.0)  # perturb, then restore
    model.load_state_dict(state)
    assert torch.equal(model.fc1.weight, torch.tensor(STUDENT_FC1_W))


def test_load_plain_student_checkpoint_updates_weights():
    model = _output_only_model()

    donor = StudentNet()
    _assign(donor.fc1, [[5.0, 6.0], [7.0, 8.0]])
    _assign(donor.fc2, [[9.0, 10.0]])
    model.load_state_dict(donor.state_dict())

    assert torch.equal(model.fc1.weight, donor.fc1.weight)
    assert torch.equal(model.fc2.weight, donor.fc2.weight)
    # Teacher untouched by the student-only checkpoint.
    assert torch.equal(model.teacher_model.fc1.weight, torch.tensor(TEACHER_FC1_W))


# -------------------------------------------------------------------------------------------------
# Export
# -------------------------------------------------------------------------------------------------


def test_export_removes_hooks_and_restores_original_class():
    model = _two_pair_model()
    teacher = model.teacher_model
    pairs = _layer_pairs(model)
    reference_out = model(INPUT)
    model.compute_kd_loss(student_loss=torch.tensor(0.0))  # clear captures

    exported = model.export()

    assert type(exported) is StudentNet  # dynamic class fully unwound
    assert not hasattr(exported, "_teacher_model")
    assert len(exported._forward_hooks) == 0

    # All capture hooks removed from both sides of every pair.
    for student_layer, teacher_layer in pairs:
        assert len(student_layer._forward_hooks) == 0
        assert len(teacher_layer._forward_hooks) == 0

    # Forward still works, produces identical output, and no longer captures anything.
    out = exported(INPUT)
    assert torch.equal(out, reference_out)
    for student_layer, _ in pairs:
        assert student_layer._intermediate_output is None
    teacher(INPUT)
    assert teacher.fc1._intermediate_output is None


# -------------------------------------------------------------------------------------------------
# Error paths
# -------------------------------------------------------------------------------------------------


def test_unknown_student_layer_name_raises():
    student, teacher = _make_models()
    with pytest.raises(AttributeError, match="bogus"):
        _convert(student, teacher, {("bogus", "fc1"): nn.MSELoss()})


def test_unknown_teacher_layer_name_raises():
    student, teacher = _make_models()
    with pytest.raises(AttributeError, match="bogus"):
        _convert(student, teacher, {("fc1", "bogus"): nn.MSELoss()})


def test_missing_criterion_raises():
    student, teacher = _make_models()
    with pytest.raises(AssertionError, match="criterion"):
        mtd.convert(student, mode=[("kd_loss", {"teacher_model": teacher})])


def test_missing_teacher_model_raises():
    student, _ = _make_models()
    with pytest.raises(AssertionError, match="teacher_model"):
        mtd.convert(student, mode=[("kd_loss", {"criterion": nn.MSELoss()})])


def test_multiple_pairs_without_balancer_raises():
    student, teacher = _make_models()
    criterion = {
        ("fc1", "fc1"): nn.MSELoss(),
        ("fc2", "fc2"): nn.MSELoss(),
    }
    with pytest.raises(ValueError, match="multiple layer-loss pairs"):
        _convert(student, teacher, criterion, loss_balancer=None)


def test_parameterized_loss_balancer_rejected():
    student, teacher = _make_models()
    with pytest.raises(AssertionError, match="cannot have parameters"):
        _convert(student, teacher, nn.MSELoss(), loss_balancer=ParamBalancer())
