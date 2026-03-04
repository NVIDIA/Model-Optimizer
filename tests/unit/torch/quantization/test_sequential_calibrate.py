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

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization.model_calib import sequential_calibrate
from modelopt.torch.quantization.utils import LayerActivationCollector


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


def _run_forward(model, data):
    for batch in data:
        model(batch)


# LayerActivationCollector tests


def test_collector_collects_correct_number_of_inputs():
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    data = [torch.randn(2, 8) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    inputs = collector.get_input_activations(model.layers[0], forward_loop)
    assert len(inputs) == 3


def test_collector_activations_match_expected():
    """First layer should receive the raw input data."""
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    data = [torch.randn(2, 8)]

    def forward_loop(m):
        for d in data:
            m(d)

    inputs = collector.get_input_activations(model.layers[0], forward_loop)
    args, kwargs = inputs[0]
    assert torch.allclose(args[0], data[0])


def test_collector_second_layer_receives_transformed_input():
    """Second layer should receive first layer's output, not raw input."""
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)
    x = torch.randn(2, 8)

    def forward_loop(m):
        m(x)

    expected = model.layers[0](x)
    inputs = collector.get_input_activations(model.layers[1], forward_loop)
    args, _ = inputs[0]
    assert torch.allclose(args[0], expected)


def test_collector_forward_is_restored_after_collection():
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)

    def forward_loop(m):
        m(torch.randn(2, 8))

    collector.get_input_activations(model.layers[0], forward_loop)

    assert not hasattr(model, "_original_forward")
    assert not hasattr(model.layers[0], "inputs")
    assert not hasattr(model.layers[0], "_original_forward")


def test_collector_cleanup_on_forward_loop_error():
    """Patching should be cleaned up even if forward_loop raises."""
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model)

    def bad_forward_loop(m):
        raise RuntimeError("intentional error")

    with pytest.raises(RuntimeError, match="intentional error"):
        collector.get_input_activations(model.layers[0], bad_forward_loop)

    assert not hasattr(model, "_original_forward")
    assert not hasattr(model.layers[0], "inputs")


# sequential_calibrate tests


def test_seq_calib_raises_on_none_forward_loop():
    model, _ = _make_model_and_data(n_layers=2)
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


def test_seq_calib_func_called_per_layer():
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


def test_seq_calib_func_receives_correct_layer():
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


def test_seq_calib_kwargs_forwarded():
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


def test_seq_calib_layer_forward_loop_runs_all_batches():
    """The per-layer forward loop passed to calib_func should replay all batches."""
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


def test_seq_calib_does_not_alter_weights():
    """sequential_calibrate itself should not modify model weights."""
    model, data = _make_model_and_data(n_layers=3)
    weights_before = {n: p.clone() for n, p in model.named_parameters()}

    sequential_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=lambda layer, forward_loop, **kw: None,
    )

    for n, p in model.named_parameters():
        assert torch.equal(p, weights_before[n]), f"Weight {n} was modified"


def test_seq_calib_activations_update_across_layers():
    """Subsequent layers should see activations transformed by prior layers."""
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


def test_seq_calib_empty_forward_loop():
    """If forward_loop feeds no data, calib_func still gets called with an empty replay."""
    model = _SimpleTransformerModel(n_layers=2, dim=16)
    replay_counts = []

    def check_empty_replay(layer, forward_loop, **kwargs):
        counter = {"n": 0}
        orig_forward = layer.forward

        def counting_forward(*args, **kw):
            counter["n"] += 1
            return orig_forward(*args, **kw)

        layer.forward = counting_forward
        forward_loop(layer)
        layer.forward = orig_forward
        replay_counts.append(counter["n"])

    sequential_calibrate(
        model,
        forward_loop=lambda m: None,
        calib_func=check_empty_replay,
    )

    for count in replay_counts:
        assert count == 0
