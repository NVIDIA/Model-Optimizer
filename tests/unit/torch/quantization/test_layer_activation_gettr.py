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

"""Tests for LayerActivationGettr."""

import pytest
import torch
from torch import nn

from modelopt.torch.quantization.model_calib import LayerActivationGettr


class SimpleModel(nn.Module):
    """Simple model with two linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test_patch_and_unpatch_layer():
    """Test that _patch_layer adds expected attributes and that unpatching removes them."""
    model = SimpleModel()
    layer = model.layer1

    LayerActivationGettr._patch_and_initialize_layer(layer)

    assert hasattr(layer, "_original_forward")
    assert hasattr(layer, "inputs")
    assert hasattr(layer, "outputs")
    assert hasattr(layer, "_stop_after_collection")
    assert layer.inputs == []
    assert layer.outputs == []

    LayerActivationGettr._unpatch_and_cleanup_layer(layer)

    assert not hasattr(layer, "_original_forward")
    assert not hasattr(layer, "inputs")
    assert not hasattr(layer, "outputs")
    assert not hasattr(layer, "_stop_after_collection")


@pytest.mark.parametrize("stop_after_collection", [False])
def test_patch(stop_after_collection):
    """Test that patched layer collects inputs and outputs."""
    model = SimpleModel()
    layer = model.layer1

    LayerActivationGettr._patch_and_initialize_layer(
        layer, stop_after_collection=stop_after_collection
    )

    x = torch.ones(2, 4)
    output = layer(x)

    assert len(layer.inputs) == 1
    assert len(layer.outputs) == 1
    assert torch.equal(layer.inputs[0][0][0], x)
    assert torch.equal(layer.outputs[0], output)

    LayerActivationGettr._unpatch_and_cleanup_layer(layer)


def test_get_input_activations():
    """Test get_input_activations collects inputs and unpatches."""
    model = SimpleModel()
    getter = LayerActivationGettr(model)

    def forward_loop(m):
        m(torch.randn(2, 4))
        m(torch.randn(2, 4))

    inputs = getter.get_input_activations(model.layer1, forward_loop)

    # Should collect all inputs from forward_loop
    assert len(inputs) == 2
    # Layer should be unpatched
    assert not hasattr(model.layer1, "_original_forward")


def test_get_output_activations():
    """Test get_output_activations runs inputs and collects outputs."""
    model = SimpleModel()
    getter = LayerActivationGettr(model)

    # Create test inputs
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)
    inputs = [((x1,), {}), ((x2,), {})]

    outputs = getter.get_output_activations(model.layer1, inputs)

    assert len(outputs) == 2
    # Verify outputs match direct forward
    assert torch.equal(outputs[0], model.layer1(x1))
    assert torch.equal(outputs[1], model.layer1(x2))
    # Layer should be unpatched
    assert not hasattr(model.layer1, "_original_forward")
