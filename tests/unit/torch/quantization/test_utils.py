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

import pytest
import torch

from modelopt.torch.quantization.utils import (
    convert_quantization_axis_to_reduce_axis,
    reduce_block_amax,
)
from modelopt.torch.quantization.utils.activation_collector import LayerActivationCollector


@pytest.mark.parametrize(
    ("block_sizes", "test_input", "expected_scales"),
    [
        (
            {-1: 2, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([5.0, 7.0], dtype=torch.bfloat16),
        ),
        (
            {-1: 4, -2: 2},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([7.0, 15.0], dtype=torch.bfloat16),
        ),
        (
            {-2: 4},
            torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                dtype=torch.bfloat16,
            ),
            torch.tensor([[12.0, 13.0, 14.0, 15.0]], dtype=torch.bfloat16),
        ),
    ],
)
def test_reduce_block_amax(block_sizes, test_input, expected_scales):
    scales = reduce_block_amax(test_input, block_sizes)

    torch.allclose(scales, expected_scales)


@pytest.mark.parametrize(
    ("shape", "quant_axis", "expected_reduce_axis"),
    [
        ((2, 3, 4), None, None),  # Per-tensor quantization (None axis)
        ((2, 3, 4), 0, [1, 2]),  # Single axis cases
        ((2, 3, 4), -2, [0, 2]),  # Negative indices
        ((2, 3, 4, 5), [0, -1], [1, 2]),  # Multiple axes
        ((2, 3, 4), (0, 2), [1]),  # Tuple instead of list
    ],
)
def test_convert_quantization_axis_to_reduce_axis(shape, quant_axis, expected_reduce_axis):
    """Test converting quantization axes to reduction axes."""
    # Create a tensor with the specified shape
    input_tensor = torch.randn(shape)

    # Convert quantization axis to reduce axis
    result = convert_quantization_axis_to_reduce_axis(input_tensor, quant_axis)

    # Check if the result matches expected
    assert result == expected_reduce_axis, (
        f"For shape {shape} and quant_axis {quant_axis}, expected {expected_reduce_axis} but got {result}"
    )

    # Additional sanity check: if we sum-reduce along the returned axes,
    # we should get a tensor with the same shape as if we kept the quantization axes
    if result is not None and len(result) > 0:
        # Create a ones tensor for easy shape verification
        ones = torch.ones(shape)

        # Reduce sum across the returned axes
        reduced = ones
        for axis in sorted(
            result, reverse=True
        ):  # Reduce from highest dim to avoid changing indices
            reduced = reduced.sum(dim=axis, keepdim=True)

        # Build expected shape after reduction
        expected_shape = list(shape)
        for axis in sorted(result, reverse=True):
            expected_shape[axis] = 1

        assert reduced.shape == tuple(expected_shape), (
            f"Reduction result shape {reduced.shape} doesn't match expected {tuple(expected_shape)}"
        )


def test_layer_activation_collector_support_api(monkeypatch):
    class _SupportedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    class _UnsupportedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    supported = _SupportedModel()
    unsupported = _UnsupportedModel()

    def _supports_layers(model):
        return hasattr(model, "layers")

    def _discover_layers(model):
        return model.layers

    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(_supports_layers, _discover_layers)

    assert LayerActivationCollector.is_supported(supported)
    assert LayerActivationCollector.get_decoder_layers(supported) is supported.layers
    assert not LayerActivationCollector.is_supported(unsupported)
    assert LayerActivationCollector.get_decoder_layers(unsupported) is None


def test_layer_activation_collector_decoder_discoverer_resolution_order(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    calls = {"first": 0, "second": 0}

    def _supported(_model):
        return True

    def _first_discoverer(_model):
        calls["first"] += 1

    def _second_discoverer(model):
        calls["second"] += 1
        return model.layers

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(_supported, _first_discoverer), (_supported, _second_discoverer)],
    )

    model = _Model()
    resolved = LayerActivationCollector.get_decoder_layers(model)
    assert resolved is model.layers
    assert calls["first"] == 1
    assert calls["second"] == 1


def test_layer_activation_collector_decoder_discoverer_no_match(monkeypatch):
    class _Model(torch.nn.Module):
        pass

    def _unsupported(_model):
        return False

    def _discoverer(_model):
        return torch.nn.ModuleList([torch.nn.Identity()])

    monkeypatch.setattr(
        LayerActivationCollector, "_decoder_layer_support", [(_unsupported, _discoverer)]
    )

    model = _Model()
    assert LayerActivationCollector.get_decoder_layers(model) is None
    assert not LayerActivationCollector.is_supported(model)


def test_layer_activation_collector_decoder_discoverer_dedup(monkeypatch):
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def _supported(model):
        return hasattr(model, "layers")

    def _discoverer(model):
        return model.layers

    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(_supported, _discoverer)
    LayerActivationCollector.register_decoder_layer_support(_supported, _discoverer)

    assert len(LayerActivationCollector._decoder_layer_support) == 1


def test_layer_activation_collector_skip_forward_captures_correct_inputs(monkeypatch):
    """The skip/run/capture strategy produces the same inputs as a plain forward."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale

        def forward(self, hidden_states):
            return hidden_states * self.scale

    class _ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_ToyLayer(2.0), _ToyLayer(0.5), _ToyLayer(3.0)])

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )

    model = _ToyDecoder()
    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]

    def _forward_loop(m):
        for b in batches:
            m(b)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        inp0 = collector.get_input_activations(model.layers[0], _forward_loop)
        inp1 = collector.get_input_activations(model.layers[1], _forward_loop)
        inp2 = collector.get_input_activations(model.layers[2], _forward_loop)
    finally:
        collector._unpatch_all_layers()

    expected_0 = batches
    expected_1 = [model.layers[0](b) for b in batches]
    expected_2 = [model.layers[1](b) for b in expected_1]

    for (args, _kw), exp in zip(inp0, expected_0):
        assert torch.allclose(args[0], exp)
    for (args, _kw), exp in zip(inp1, expected_1):
        assert torch.allclose(args[0], exp)
    for (args, _kw), exp in zip(inp2, expected_2):
        assert torch.allclose(args[0], exp)


def test_layer_activation_collector_run_uses_cached_inputs_not_parent(monkeypatch):
    """Verify that the 'run' layer uses cached inputs, not garbage from skip layers."""

    call_log = []

    class _ToyLayer(torch.nn.Module):
        def __init__(self, name, bias):
            super().__init__()
            self.layer_name = name
            self.bias = bias

        def forward(self, hidden_states):
            call_log.append(self.layer_name)
            return hidden_states + self.bias

    class _ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [_ToyLayer("L0", 1.0), _ToyLayer("L1", 2.0), _ToyLayer("L2", 3.0)]
            )

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )

    model = _ToyDecoder()
    batches = [torch.tensor([[10.0]])]

    def _forward_loop(m):
        for b in batches:
            m(b)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        collector.get_input_activations(model.layers[0], _forward_loop)
        call_log.clear()
        collector.get_input_activations(model.layers[1], _forward_loop)
        call_log_for_layer1 = list(call_log)
        call_log.clear()
        inp2 = collector.get_input_activations(model.layers[2], _forward_loop)
    finally:
        collector._unpatch_all_layers()

    assert "L0" in call_log_for_layer1
    assert "L1" not in call_log_for_layer1

    assert torch.allclose(inp2[0][0][0], torch.tensor([[10.0 + 1.0 + 2.0]]))
