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

"""Integration tests for sequential_calibrate and LayerActivationCollector."""

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
    """model.layers (ModuleList) — the simplest pattern recognised by get_decoder_layers."""

    def __init__(self, n_layers=3, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])
        self.embed = nn.Embedding(32, dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _TupleReturningBlock(nn.Module):
    """Decoder layer that returns a tuple, mimicking HuggingFace decoder layers."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return (self.linear(x), None)


class _TupleUnpackingModel(nn.Module):
    """Parent model that unpacks layer outputs as tuples."""

    def __init__(self, n_layers=4, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_TupleReturningBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


def _make_model_and_data(n_layers=3, dim=16, n_batches=2, batch_size=4):
    torch.manual_seed(42)
    model = _SimpleTransformerModel(n_layers=n_layers, dim=dim)
    tokens = [torch.randint(0, 32, (batch_size, 8)) for _ in range(n_batches)]
    return model, tokens


def _run_forward(model, data):
    for batch in data:
        model(batch)


def _register_test_discoverer(monkeypatch):
    """Register a simple discoverer that finds model.layers on any model."""
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


def test_seq_calib_func_called_per_layer(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=4)
    call_count = [0]

    def counting_calib(layer, forward_loop, **kwargs):
        call_count[0] += 1

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=counting_calib,
    )

    assert call_count[0] == 4


def test_seq_calib_func_receives_correct_layer(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=3)
    called_layers = []

    def track_layers(layer, forward_loop, **kwargs):
        called_layers.append(layer)

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=track_layers,
    )

    for i, layer in enumerate(model.layers):
        assert called_layers[i] is layer


def test_seq_calib_kwargs_forwarded(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=2)
    received_kwargs = []

    def capture_kwargs(layer, forward_loop, **kwargs):
        received_kwargs.append(kwargs)

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=capture_kwargs,
        alpha=0.5,
        method="max",
    )

    assert len(received_kwargs) == 2
    for kw in received_kwargs:
        assert kw["alpha"] == 0.5
        assert kw["method"] == "max"


def test_seq_calib_layer_forward_loop_runs_all_batches(monkeypatch):
    """The per-layer forward loop passed to calib_func should replay all batches."""
    _register_test_discoverer(monkeypatch)
    n_batches = 5
    model, data = _make_model_and_data(n_layers=2, n_batches=n_batches)
    batch_counts = []

    def count_batches(layer, forward_loop, **kwargs):
        counter = {"n": 0}
        orig_forward = layer.forward

        def counting_forward(*args, **kw):
            counter["n"] += 1
            return orig_forward(*args, **kw)

        layer.forward = counting_forward
        forward_loop(layer)
        layer.forward = orig_forward
        batch_counts.append(counter["n"])

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=count_batches,
    )

    for count in batch_counts:
        assert count == n_batches


def test_seq_calib_does_not_alter_weights(monkeypatch):
    """sequential_calibrate itself should not modify model weights."""
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=3)
    weights_before = {n: p.clone() for n, p in model.named_parameters()}

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=lambda layer, forward_loop, **kw: None,
    )

    for n, p in model.named_parameters():
        assert torch.equal(p, weights_before[n]), f"Weight {n} was modified"


def test_seq_calib_activations_update_across_layers(monkeypatch):
    """Subsequent layers should see activations transformed by prior layers."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=2, dim=16)
    tokens = [torch.randint(0, 32, (2, 4))]

    layer_inputs_record = {}

    def record_inputs(layer, forward_loop, **kwargs):
        activations = []
        orig_forward = layer.forward

        def capture_forward(*args, **kw):
            activations.append(args[0].clone())
            return orig_forward(*args, **kw)

        layer.forward = capture_forward
        forward_loop(layer)
        layer.forward = orig_forward

        layer_idx = list(model.layers).index(layer)
        layer_inputs_record[layer_idx] = activations

    sequential_calibrate(
        model,
        forward_loop=lambda m: [m(t) for t in tokens],
        calib_func=record_inputs,
    )

    assert not torch.allclose(layer_inputs_record[0][0], layer_inputs_record[1][0]), (
        "Layer 1 should receive different activations than layer 0"
    )


def test_mode_transitions_across_calibration_steps(monkeypatch):
    """Verify layer modes after each sequential calibration step.

    After get_input_activations(layers[i]) returns, the current layer is reset
    to 'original'.  Layers further back are left in 'run' (just calibrated) or
    'skip' (fully done), reflecting the state the next forward loop will see.
    """
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:

        def modes():
            return [model.layers[i]._seq_calib.mode for i in range(5)]

        collector.get_input_activations(model.layers[0], forward_loop)
        assert modes() == ["original", "original", "original", "original", "original"]

        collector.get_input_activations(model.layers[1], forward_loop)
        assert modes() == ["run", "original", "original", "original", "original"]

        collector.get_input_activations(model.layers[2], forward_loop)
        assert modes() == ["skip", "run", "original", "original", "original"]

        collector.get_input_activations(model.layers[3], forward_loop)
        assert modes() == ["skip", "skip", "run", "original", "original"]

        collector.get_input_activations(model.layers[4], forward_loop)
        assert modes() == ["skip", "skip", "skip", "run", "original"]
    finally:
        collector._unpatch_all_layers()


def test_run_layer_reflects_weight_updates(monkeypatch):
    """After calib_func modifies weights, the next layer should see updated activations."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    dim = 8

    class _ScaleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * self.weight

    class _TwoScaleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_ScaleLayer(), _ScaleLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = _TwoScaleModel()
    x = torch.randn(2, dim)

    activations_before_weight_update = model.layers[0](x).clone()

    def forward_loop(m):
        m(x)

    def weight_doubling_calib(layer, layer_forward_loop, **kwargs):
        with torch.no_grad():
            layer.weight.mul_(2.0)
        layer_forward_loop(layer)

    sequential_calibrate(
        model,
        forward_loop=forward_loop,
        calib_func=weight_doubling_calib,
    )

    # Layer 0's weight was doubled by calib_func.  When collecting inputs
    # for layer 1, the run-mode replay of layer 0 should use the updated
    # weight, so layer 1 should have received 2x the original activations.
    expected = activations_before_weight_update * 2.0
    # Verify by running model.layers[0] with its updated weights
    actual = model.layers[0](x)
    assert torch.allclose(actual, expected)
