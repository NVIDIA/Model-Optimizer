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

"""Unit tests for sequential_calibrate and LayerActivationCollector."""

from collections import deque

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization.model_calib import sequential_calibrate
from modelopt.torch.quantization.utils.activation_collector import LayerActivationCollector


class _DecoderBlock(nn.Module):
    """Minimal transformer decoder block."""

    def __init__(self, dim=16):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(x)
        return x


class _SimpleTransformerModel(nn.Module):
    """model.layers (ModuleList) -- the simplest pattern recognised by get_decoder_layers."""

    def __init__(self, n_layers=3, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])
        self.embed = nn.Embedding(32, dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _FlatMLP(nn.Module):
    """No decoder-layer structure -- should be rejected by sequential_calibrate."""

    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class _SimpleTwoLayerModel(nn.Module):
    """Minimal model with explicit layers for activation-collection tests."""

    def __init__(self, dim=16):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _make_model_and_data(n_layers=3, dim=16, n_batches=2, batch_size=4):
    torch.manual_seed(42)
    model = _SimpleTransformerModel(n_layers=n_layers, dim=dim)
    tokens = [torch.randint(0, 32, (batch_size, 8)) for _ in range(n_batches)]
    return model, tokens


# LayerActivationCollector tests


def _register_test_discoverer(monkeypatch):
    """Register a simple discoverer that finds model.layers on any model."""
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


def test_collector_collects_correct_number_of_inputs(monkeypatch):
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    data = [torch.randn(2, 8) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector._patch_all_layers()
    try:
        inputs = collector.get_input_activations(model.layers[0], forward_loop)
        assert len(inputs) == 3
    finally:
        collector._unpatch_all_layers()


def test_collector_activations_match_expected(monkeypatch):
    """First layer should receive the raw input data."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    data = [torch.randn(2, 8)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector._patch_all_layers()
    try:
        inputs = collector.get_input_activations(model.layers[0], forward_loop)
        args, kwargs = inputs[0]
        assert torch.allclose(args[0], data[0])
    finally:
        collector._unpatch_all_layers()


def test_collector_second_layer_receives_transformed_input(monkeypatch):
    """Second layer should receive first layer's output, not raw input."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    x = torch.randn(2, 8)

    def forward_loop(m):
        m(x)

    expected = model.layers[0](x)

    collector._patch_all_layers()
    try:
        collector.get_input_activations(model.layers[0], forward_loop)
        inputs = collector.get_input_activations(model.layers[1], forward_loop)
        args, _ = inputs[0]
        assert torch.allclose(args[0], expected)
    finally:
        collector._unpatch_all_layers()


def test_collector_forward_is_restored_after_collection(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)

    def forward_loop(m):
        m(torch.randn(2, 8))

    collector._patch_all_layers()
    collector.get_input_activations(model.layers[0], forward_loop)
    collector._unpatch_all_layers()

    assert not hasattr(model, "_original_forward")
    assert not hasattr(model.layers[0], "_seq_calib")
    assert not hasattr(model.layers[0], "_original_forward")


def test_collector_cleanup_on_forward_loop_error(monkeypatch):
    """Patching should be cleaned up even if forward_loop raises."""
    _register_test_discoverer(monkeypatch)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)

    def bad_forward_loop(m):
        raise RuntimeError("intentional error")

    collector._patch_all_layers()
    try:
        with pytest.raises(RuntimeError, match="intentional error"):
            collector.get_input_activations(model.layers[0], bad_forward_loop)
    finally:
        collector._unpatch_all_layers()

    assert not hasattr(model, "_original_forward")
    assert not hasattr(model.layers[0], "_seq_calib")


# sequential_calibrate tests
def test_seq_calib_raises_on_none_forward_loop(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=2)
    with pytest.raises(ValueError, match="forward_loop must not be None"):
        sequential_calibrate(
            model,
            forward_loop=None,
            calib_func=lambda *a, **kw: None,
        )


def test_seq_calib_raises_on_unrecognized_model():
    model = _FlatMLP()
    with pytest.raises(ValueError, match="Could not find transformer layers"):
        sequential_calibrate(
            model,
            forward_loop=lambda m: m(torch.randn(2, 16)),
            calib_func=lambda *a, **kw: None,
        )


def test_seq_calib_empty_forward_loop_raises(monkeypatch):
    """If forward_loop feeds no data, sequential_calibrate raises RuntimeError."""
    _register_test_discoverer(monkeypatch)
    model = _SimpleTransformerModel(n_layers=2, dim=16)

    with pytest.raises(RuntimeError, match="collected no inputs during forward_loop"):
        sequential_calibrate(
            model,
            forward_loop=lambda m: None,
            calib_func=lambda *a, **kw: None,
        )


# ---------------------------------------------------------------------------
# Skip / run / capture path verification tests
# ---------------------------------------------------------------------------


class _TupleReturningBlock(nn.Module):
    """Decoder layer that returns a tuple, mimicking HuggingFace decoder layers."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return (self.linear(x), None)


class _TupleUnpackingModel(nn.Module):
    """Parent model that unpacks layer outputs as tuples.

    This would crash with a naive skip that returns a bare tensor.
    """

    def __init__(self, n_layers=4, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_TupleReturningBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


class _InterLayerNormModel(nn.Module):
    """Model with LayerNorm between decoder layers (not inside them)."""

    def __init__(self, n_layers=4, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_TupleReturningBlock(dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = norm(x)
            x, _ = layer(x)
        return x


def test_skip_output_preserves_tuple_structure(monkeypatch):
    """Skip layers must return a tuple when the real layer returns a tuple.

    Without this, the parent's ``x, _ = layer(x)`` unpacking would crash.
    """
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in model.layers:
            inputs = collector.get_input_activations(layer, forward_loop)
            assert len(inputs) == len(data)
    finally:
        collector._unpatch_all_layers()


def test_skip_output_preserves_shape_with_inter_layer_norm(monkeypatch):
    """Skip outputs must have correct shape for un-patched LayerNorm between layers."""
    _register_test_discoverer(monkeypatch)
    model = _InterLayerNormModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in model.layers:
            inputs = collector.get_input_activations(layer, forward_loop)
            assert len(inputs) == len(data)
    finally:
        collector._unpatch_all_layers()


def test_run_layer_populates_output_meta(monkeypatch):
    """After a layer executes in 'run' mode, its output_meta must be set."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=3, dim=16)
    data = [torch.randn(2, 16)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        # Layer 0 starts as capture — no output_meta yet
        collector.get_input_activations(model.layers[0], forward_loop)
        assert model.layers[0]._seq_calib.output_meta is None

        # Calibrating layer 1 puts layer 0 into run, which sets output_meta
        collector.get_input_activations(model.layers[1], forward_loop)
        meta = model.layers[0]._seq_calib.output_meta
        assert meta is not None
        assert meta[0] == "tuple", "Tuple-returning layer should produce tuple metadata"
    finally:
        collector._unpatch_all_layers()


def test_run_layer_consumes_cached_inputs(monkeypatch):
    """The run layer must pop all cached inputs during the forward loop."""
    _register_test_discoverer(monkeypatch)
    n_batches = 4
    model = _TupleUnpackingModel(n_layers=3, dim=16)
    data = [torch.randn(2, 16) for _ in range(n_batches)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        collector.get_input_activations(model.layers[0], forward_loop)
        collector.get_input_activations(model.layers[1], forward_loop)

        # Before calibrating layer 2, layer 1 transitions to run.
        # Its cached_inputs should be populated from collected_inputs.
        collector._set_layer_states(2)
        assert len(model.layers[1]._seq_calib.cached_inputs) == n_batches

        # After the forward loop, all cached inputs should be consumed
        forward_loop(model)
        assert len(model.layers[1]._seq_calib.cached_inputs) == 0
    finally:
        collector._unpatch_all_layers()


def test_set_layer_states_transitions(monkeypatch):
    """Unit test for _set_layer_states: verify mode assignments at each index.

    Simulates the state a real forward loop would leave behind by manually
    populating collected_inputs before each call.
    """
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    fake_inp = ((torch.zeros(1, 16),), {})

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:

        def modes():
            return [model.layers[i]._seq_calib.mode for i in range(5)]

        collector._set_layer_states(0)
        assert modes() == ["capture", "original", "original", "original", "original"]

        model.layers[0]._seq_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(1)
        assert modes() == ["run", "capture", "original", "original", "original"]

        model.layers[1]._seq_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(2)
        assert modes() == ["skip", "run", "capture", "original", "original"]

        model.layers[2]._seq_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(3)
        assert modes() == ["skip", "skip", "run", "capture", "original"]

        model.layers[3]._seq_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(4)
        assert modes() == ["skip", "skip", "skip", "run", "capture"]
    finally:
        collector._unpatch_all_layers()


def test_set_layer_states_raises_on_empty_collected_inputs(monkeypatch):
    """_set_layer_states must raise if the previous layer has no collected inputs."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=2, dim=16)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        # layer 0 was never in capture mode, so collected_inputs is empty
        with pytest.raises(RuntimeError, match="no collected inputs to replay"):
            collector._set_layer_states(1)
    finally:
        collector._unpatch_all_layers()


def test_run_asserts_on_empty_cached_inputs(monkeypatch):
    """A layer in 'run' mode with no cached inputs must raise AssertionError."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=2, dim=16)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        model.layers[0]._seq_calib.mode = "run"
        model.layers[0]._seq_calib.cached_inputs = deque()

        with pytest.raises(AssertionError, match="no cached inputs to replay"):
            model(torch.randn(2, 16))
    finally:
        collector._unpatch_all_layers()


def test_cleanup_removes_seq_calib_attr(monkeypatch):
    """After unpatch, no layer should have the _seq_calib attribute."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=3, dim=16)
    data = [torch.randn(2, 16)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    for layer in model.layers:
        collector.get_input_activations(layer, forward_loop)
    collector._unpatch_all_layers()

    for i, layer in enumerate(model.layers):
        assert not hasattr(layer, "_seq_calib"), f"Layer {i} still has _seq_calib after cleanup"
        assert not hasattr(layer, "_original_forward"), (
            f"Layer {i} still has _original_forward after cleanup"
        )
    assert not hasattr(model, "_original_forward")


def test_skip_output_meta_not_shared_across_heterogeneous_layers(monkeypatch):
    """Each layer stores its own output_meta, supporting heterogeneous architectures."""

    class _SmallBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return (self.linear(x), None, torch.zeros(1))

    class _BigBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return (self.linear(x),)

    class _HeterogeneousModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_SmallBlock(), _BigBlock(), _SmallBlock()])

        def forward(self, x):
            for layer in self.layers:
                out = layer(x)
                x = out[0]
            return x

    _register_test_discoverer(monkeypatch)
    model = _HeterogeneousModel()
    data = [torch.randn(2, 8)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in model.layers:
            collector.get_input_activations(layer, forward_loop)

        # After full calibration, layers 0 and 1 have been through 'run' and have output_meta
        meta_0 = model.layers[0]._seq_calib.output_meta
        meta_1 = model.layers[1]._seq_calib.output_meta
        assert meta_0 is not None
        assert meta_1 is not None
        # SmallBlock returns 3-element tuple, BigBlock returns 1-element tuple
        assert len(meta_0[1]) == 3
        assert len(meta_1[1]) == 1
    finally:
        collector._unpatch_all_layers()
