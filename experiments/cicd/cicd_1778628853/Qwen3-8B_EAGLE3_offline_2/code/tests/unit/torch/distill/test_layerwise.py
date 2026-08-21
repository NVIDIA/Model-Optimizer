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

import warnings

import pytest
import torch
from _test_utils.torch.vision_models import get_tiny_mobilenet_and_input

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto


def get_input_tensor():
    """Dummy input tensor."""
    return torch.rand(2, 3, 112, 112)


def tiny_mobilenet():
    return get_tiny_mobilenet_and_input()[0]


@pytest.fixture
def layerwise_distillation_model():
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.2", "features.2"): torch.nn.MSELoss(),
        },
        "loss_balancer": mtd.StaticLossBalancer(),
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    return layerwise_model


def test_layerwise_hooks_registration(layerwise_distillation_model):
    """Test that layerwise-specific hooks are registered correctly."""
    # Check that student layers have _teacher_layer attribute
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert hasattr(student_layer, "_teacher_layer")
        assert student_layer._teacher_layer[0] is teacher_layer
        assert hasattr(student_layer, "_intermediate_output")

        # Check that teacher layers have both input and output capture attributes
        assert hasattr(teacher_layer, "_intermediate_input")
        assert hasattr(teacher_layer, "_intermediate_output")


def test_layerwise_forward_pass(layerwise_distillation_model):
    """Test that forward pass works and captures both teacher inputs and outputs."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    layerwise_distillation_model(input_tensor)

    # Check that teacher intermediate inputs and outputs are captured
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is None
        assert teacher_layer._intermediate_output is not None
        assert student_layer._intermediate_output is not None


def test_layerwise_input_injection(layerwise_distillation_model):
    """Test that teacher inputs are injected into student layers during layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    # Perform forward pass
    layerwise_distillation_model(input_tensor)

    # Verify that teacher inputs were captured (they should be reset after injection)
    # After forward, teacher inputs should have been consumed by student layers
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        # After full forward pass, teacher_layer._intermediate_input should be None
        # because it gets consumed by the student layerwise hook
        assert teacher_layer._intermediate_input is None


def test_layerwise_loss_computation(layerwise_distillation_model):
    """Test that loss computation works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    output = layerwise_distillation_model(input_tensor)
    loss = layerwise_distillation_model.compute_kd_loss(student_loss=output.mean())

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert loss.requires_grad


def test_layerwise_only_student_forward(layerwise_distillation_model):
    """Test that only_student_forward context manager works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    # When using only_student_forward, teacher inputs should not be captured
    with warnings.catch_warnings(record=True) as w:
        with layerwise_distillation_model.only_student_forward():
            layerwise_distillation_model(input_tensor)

        # Should get warning about missing teacher input
        warning_messages = [str(warning.message) for warning in w]
        assert any("has no intermediate input stored" in msg for msg in warning_messages)

    # Verify teacher didn't run
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is None
        assert teacher_layer._intermediate_output is None
        assert student_layer._intermediate_output is not None


def test_layerwise_only_teacher_forward(layerwise_distillation_model):
    """Test that only_teacher_forward context manager works with layerwise distillation."""
    layerwise_distillation_model.train()
    input_tensor = get_input_tensor()

    with layerwise_distillation_model.only_teacher_forward():
        layerwise_distillation_model(input_tensor)

    # Verify teacher ran and student didn't
    for student_layer, teacher_layer in layerwise_distillation_model._layers_to_loss:
        assert teacher_layer._intermediate_input is not None
        assert teacher_layer._intermediate_output is not None
        assert student_layer._intermediate_output is None


def test_layerwise_export(layerwise_distillation_model):
    """Test that export correctly cleans up layerwise-specific attributes."""
    # Check that _teacher_layer exists before export
    for student_layer, _ in layerwise_distillation_model._layers_to_loss:
        assert hasattr(student_layer, "_teacher_layer")

    # Export the model
    exported_model = mtd.export(layerwise_distillation_model)

    # Check that _teacher_layer is removed after export
    for student_layer in exported_model.modules():
        assert not hasattr(student_layer, "_teacher_layer")

    assert not hasattr(exported_model, "_teacher_model")
    assert not isinstance(exported_model, mtd.LayerwiseDistillationModel)


def test_layerwise_save_restore(layerwise_distillation_model, tmp_path):
    """Test that save/restore works correctly with layerwise distillation."""
    mto.save(layerwise_distillation_model, tmp_path / "ckpt.pt")

    new_student = tiny_mobilenet()
    restored_model = mto.restore(new_student, tmp_path / "ckpt.pt")

    # Ensure state is not actually restored (expected behavior from test_distill.py)
    manager = mto.ModeloptStateManager(restored_model)
    assert not manager.has_state
    assert isinstance(restored_model, type(new_student))


def test_layerwise_multiloss():
    """Test layerwise distillation with multiple loss functions."""
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.1", "features.1"): torch.nn.MSELoss(),
            ("features.3", "features.3"): torch.nn.MSELoss(),
        },
        "loss_balancer": mtd.StaticLossBalancer([0.5, 0.5]),
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    # Verify hooks are registered for all layers
    assert len(layerwise_model._layers_to_loss) == 2

    # Test forward pass
    output = layerwise_model(get_input_tensor())
    loss = layerwise_model.compute_kd_loss(student_loss=output.mean())

    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1


def test_layerwise_gradient_flow():
    """Test that gradients flow correctly through layerwise distillation."""
    student = tiny_mobilenet().train()
    config = {
        "teacher_model": tiny_mobilenet(),
        "criterion": {
            ("features.2", "features.2"): torch.nn.MSELoss(),
        },
        "loss_balancer": None,
    }
    layerwise_model = mtd.convert(student, mode=[("layerwise_kd", config)])

    # Save param snapshots by module
    param_snapshots = {
        name: p.clone() for name, p in layerwise_model.named_parameters() if p.requires_grad
    }

    # Forward and backward
    optimizer = torch.optim.SGD(layerwise_model.parameters(), lr=0.5)
    optimizer.zero_grad()
    layerwise_model(get_input_tensor())
    loss = layerwise_model.compute_kd_loss()
    loss.backward()
    optimizer.step()

    # Check: parameters in only the target layer(s) are changed
    updated_any = False
    for name, param in layerwise_model.named_parameters():
        if not param.requires_grad:
            continue
        changed = not torch.allclose(param, param_snapshots[name])
        if "features.2" in name:
            assert changed, f"'{name}' parameters did not change!"
            updated_any = True
        else:
            assert not changed, f"Parameters in unrelated layer '{name}' changed!"
    assert updated_any, (
        "No parameters were updated in 'features.2' or related layers during training"
    )
