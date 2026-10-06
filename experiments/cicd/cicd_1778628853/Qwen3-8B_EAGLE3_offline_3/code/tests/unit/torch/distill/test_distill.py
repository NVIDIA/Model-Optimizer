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

import inspect
import warnings

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.vision_models import get_tiny_mobilenet_and_input
from torchvision.models import alexnet

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto


def get_input_tensor():
    """Dummy input tensor."""
    return torch.rand(2, 3, 112, 112)


def tiny_mobilenet():
    return get_tiny_mobilenet_and_input()[0]


def tiny_alexnet():
    return alexnet(num_classes=10)  # same num classes as tiny_mobilenet


@pytest.fixture
def distillation_model():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.LogitsDistillationLoss(),
        "loss_balancer": mtd.StaticLossBalancer(),
    }
    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    return distillation_model


def test_distillation_model_loss_params():
    teacher, student = tiny_alexnet(), tiny_mobilenet()
    kd_loss = mtd.MGDLoss(32, 48)

    student_params = student.parameters()
    teacher_params = teacher.parameters()
    loss_params = kd_loss.parameters()

    def count_params(params):
        return sum(p.numel() for p in params)

    individual_count = (
        count_params(student_params) + count_params(teacher_params) + count_params(loss_params)
    )

    config = {
        "teacher_model": teacher,
        "criterion": kd_loss,
    }
    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    assert count_params(distillation_model.parameters()) == individual_count

    distillation_model.to(torch.float16)

    assert next(kd_loss.parameters()).dtype == torch.float16


def test_distillation_model_no_balancer():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.LogitsDistillationLoss(),
        "loss_balancer": None,
    }
    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    distillation_model(get_input_tensor())
    loss = distillation_model.compute_kd_loss()
    assert isinstance(loss, torch.Tensor) and loss.numel() == 1


def test_distillation_model_multiloss_balancer():
    student = tiny_alexnet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": {
            ("avgpool", "avgpool"): torch.nn.MSELoss(),
            ("classifier", "classifier"): mtd.LogitsDistillationLoss(),
        },
        "loss_balancer": mtd.StaticLossBalancer([0.33, 0.33]),
    }
    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    output = distillation_model(get_input_tensor())
    distillation_model.compute_kd_loss(student_loss=output.mean())


def test_distillation_model_mft():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.MFTLoss(threshold=0.2),
        "loss_balancer": None,
    }

    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    input_tensor = get_input_tensor()
    labels = torch.randint(0, 10, (input_tensor.size(0),))  # Dummy labels for MFT
    distillation_model(input_tensor)
    loss = distillation_model.compute_kd_loss(labels=labels)
    assert isinstance(loss, torch.Tensor) and loss.numel() == 1


def test_distillation_mode_default_config():
    student = tiny_mobilenet()
    with pytest.raises(AssertionError):
        mtd.convert(student, mode="kd_loss")


def test_distillation_mode_config_types(distillation_model):
    manager = mto.ModeloptStateManager(distillation_model)
    cfg = manager._state[-1][1]["config"]
    assert callable(cfg["teacher_model"]) and not isinstance(cfg["teacher_model"], nn.Module)
    assert isinstance(cfg["criterion"], dict)


def test_distillation_save_restore(distillation_model, tmp_path):
    mto.save(distillation_model, tmp_path / "ckpt.pt")

    new_student = tiny_mobilenet()
    distillation_model_new = mto.restore(new_student, tmp_path / "ckpt.pt")

    # Ensure state is not actually restored
    manager = mto.ModeloptStateManager(distillation_model_new)
    assert not manager.has_state
    assert isinstance(distillation_model_new, type(new_student))

    # Subsequent convert should behave normally
    config = {
        "teacher_model": distillation_model.teacher_model,
        "criterion": mtd.LogitsDistillationLoss(),
    }
    distillation_model_newer = mtd.convert(new_student, mode=[("kd_loss", config)])
    manager = mto.ModeloptStateManager(distillation_model_newer)
    assert manager.has_state
    assert isinstance(distillation_model_newer, mtd.DistillationModel)


def test_distillation_export(distillation_model, tmp_path):
    model_exported = mtd.export(distillation_model)
    assert not hasattr(model_exported, "_teacher_model")
    assert hasattr(model_exported, mto.ModeloptStateManager._state_key)

    mto.save(model_exported, tmp_path / "ckpt.pt")
    new_student = tiny_mobilenet()
    new_student_restored = mto.restore(new_student, tmp_path / "ckpt.pt")
    assert isinstance(new_student_restored, new_student.__class__)


def test_logits_distillation(distillation_model):
    optimizer = torch.optim.AdamW(distillation_model.parameters())

    optimizer.zero_grad()
    output = distillation_model(get_input_tensor())
    total_loss = distillation_model.compute_kd_loss(student_loss=output.mean())
    total_loss.backward()
    optimizer.step()


def test_minimal_state_dict_mode():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.MGDLoss(32, 48),
        "loss_balancer": None,
        "expose_minimal_state_dict": True,
    }

    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    state_dict = distillation_model.state_dict()
    assert not any(key.startswith("_teacher_model") for key in state_dict)
    assert any(key.startswith("_loss_modules") for key in state_dict)

    distillation_model.load_state_dict(state_dict)


def test_full_state_dict_mode():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.MGDLoss(32, 48),
        "loss_balancer": None,
        "expose_minimal_state_dict": False,
    }

    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    state_dict = distillation_model.state_dict()
    assert any(key.startswith("_teacher_model") for key in state_dict)
    assert any(key.startswith("_loss_modules") for key in state_dict)

    distillation_model.load_state_dict(state_dict)


def test_load_student_only_state():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_alexnet,
        "criterion": mtd.MGDLoss(32, 48),
        "loss_balancer": None,
        "expose_minimal_state_dict": False,
    }

    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    state_dict = tiny_mobilenet().state_dict()
    distillation_model.load_state_dict(state_dict)


def test_multiple_modelopt_states():
    student = tiny_mobilenet().train()
    teacher = tiny_alexnet()
    mto.ModeloptStateManager(teacher, init_state=True)

    config = {
        "teacher_model": teacher,
        "criterion": mtd.MGDLoss(32, 48),
        "loss_balancer": None,
    }
    distillation_model = mtd.convert(student, mode=[("kd_loss", config)])

    assert not mto.ModeloptStateManager.is_converted(teacher)

    mtd.export(distillation_model)

    assert mto.ModeloptStateManager.is_converted(teacher)


def test_dynamic_module_inheritance(distillation_model):
    assert isinstance(distillation_model, mto.dynamic.DynamicModule)
    assert isinstance(distillation_model, distillation_model.original_cls)


def test_forward_signature_replacement(distillation_model):
    student_cls = distillation_model.original_cls
    assert inspect.signature(distillation_model.__class__.forward) == inspect.signature(
        student_cls.forward
    )


def test_duplicate_fwd_hook_call(distillation_model):
    distillation_model.train()

    with warnings.catch_warnings(record=True) as w:
        distillation_model(get_input_tensor())
        distillation_model(get_input_tensor())
        distillation_model(get_input_tensor())
        assert len(w) == 2  # one for student and one for teacher


def test_teacher_fwd_only(distillation_model):
    with distillation_model.only_teacher_forward():
        distillation_model(get_input_tensor())

    assert distillation_model.teacher_model._intermediate_output is not None
    assert distillation_model._intermediate_output is None


def test_student_fwd_only(distillation_model):
    with distillation_model.only_student_forward():
        distillation_model(get_input_tensor())

    assert distillation_model.teacher_model._intermediate_output is None
    assert distillation_model._intermediate_output is not None


def test_train_eval_mode_switch(distillation_model):
    distillation_model.train()
    assert distillation_model._intermediate_output is None
    distillation_model(get_input_tensor())
    assert distillation_model._intermediate_output is not None
    distillation_model.eval()
    assert distillation_model._intermediate_output is None
    distillation_model(get_input_tensor())
    assert distillation_model._intermediate_output is not None
    distillation_model.train()
    assert distillation_model._intermediate_output is None
