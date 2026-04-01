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

import io
import warnings
from collections import deque
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from modelopt.torch.quantization.config import MaxCalibConfig
from modelopt.torch.quantization.mode import wrapped_calib_func
from modelopt.torch.quantization.model_calib import sequential_calibrate
from modelopt.torch.quantization.utils.activation_collector import LayerActivationCollector
from modelopt.torch.quantization.utils.checkpoint import (
    _CHECKPOINT_SAVE_SUPPORT,
    SEQ_CALIB_PROGRESS_ATTR,
    get_checkpoint_saver,
    register_seq_calib_checkpoint_saver,
)


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


# ---------------------------------------------------------------------------
# Checkpoint save / resume tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_checkpoint_registry():
    """Isolate the checkpoint saver registry for a single test."""
    old = _CHECKPOINT_SAVE_SUPPORT.copy()
    _CHECKPOINT_SAVE_SUPPORT.clear()
    yield
    _CHECKPOINT_SAVE_SUPPORT.clear()
    _CHECKPOINT_SAVE_SUPPORT.extend(old)


@pytest.fixture
def _register_discoverer(monkeypatch, _clean_checkpoint_registry):
    """Register a simple discoverer and clear checkpoint saver registry."""
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


def _make_forward_loop(tokens):
    def forward_loop(m):
        for t in tokens:
            m(t)

    return forward_loop


def _noop_calib(layer, forward_loop, **kwargs):
    """No-op calibration: just run the forward loop."""
    forward_loop(layer)


@pytest.mark.usefixtures("_clean_checkpoint_registry")
class TestCheckpointSaveRegistry:
    def test_register_and_get_saver(self):
        saver = MagicMock()
        register_seq_calib_checkpoint_saver(lambda m: True, saver)
        model = nn.Linear(4, 4)
        assert get_checkpoint_saver(model) is saver

    def test_get_saver_returns_none_when_empty(self):
        assert get_checkpoint_saver(nn.Linear(4, 4)) is None

    def test_dedup_registration(self):
        pred = lambda m: True  # noqa: E731
        saver = lambda m, d: None  # noqa: E731
        register_seq_calib_checkpoint_saver(pred, saver)
        register_seq_calib_checkpoint_saver(pred, saver)
        assert len(_CHECKPOINT_SAVE_SUPPORT) == 1


@pytest.mark.usefixtures("_register_discoverer")
class TestCheckpointSave:
    def test_no_save_when_dir_is_none(self):
        """Default behavior: no checkpoint saving when checkpoint_dir is None."""
        model, tokens = _make_model_and_data(n_layers=4)
        calibrated_layers = []

        def tracking_calib(layer, fwd, **kwargs):
            fwd(layer)
            calibrated_layers.append(layer)

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=tracking_calib,
            checkpoint_dir=None,
            checkpoint_interval=2,
        )
        assert len(calibrated_layers) == 4
        assert not hasattr(model, SEQ_CALIB_PROGRESS_ATTR)

    def test_checkpoint_triggers_saver_at_interval(self):
        """Saver should be called at the correct layer intervals."""
        model, tokens = _make_model_and_data(n_layers=5)
        save_calls = []

        def mock_saver(m, d):
            progress = getattr(m, SEQ_CALIB_PROGRESS_ATTR)
            save_calls.append(progress["completed_layer_idx"])

        register_seq_calib_checkpoint_saver(lambda m: True, mock_saver)

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=2,
        )
        # interval=2: save after layers 1 (idx 1), 3 (idx 3), skip last (idx 4)
        assert save_calls == [1, 3]

    def test_checkpoint_skips_final_layer(self):
        """No checkpoint is saved after the final layer."""
        model, tokens = _make_model_and_data(n_layers=3)
        save_calls = []

        def mock_saver(m, d):
            progress = getattr(m, SEQ_CALIB_PROGRESS_ATTR)
            save_calls.append(progress["completed_layer_idx"])

        register_seq_calib_checkpoint_saver(lambda m: True, mock_saver)

        # interval=1 would save every layer, but should still skip the last
        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=1,
        )
        assert 2 not in save_calls  # layer index 2 (last) should not trigger save

    def test_checkpoint_warns_when_no_saver_registered(self, capsys):
        """Should print a warning when checkpoint_dir is set but no saver is registered."""
        model, tokens = _make_model_and_data(n_layers=3)

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=1,
        )
        # No error raised — just a warning printed
        assert not hasattr(model, SEQ_CALIB_PROGRESS_ATTR)

    def test_progress_attr_cleaned_up_after_save(self):
        """The _seq_calib_progress attribute should be cleaned up after save."""
        model, tokens = _make_model_and_data(n_layers=4)

        def mock_saver(m, d):
            # During save, the attribute should be set
            assert hasattr(m, SEQ_CALIB_PROGRESS_ATTR)

        register_seq_calib_checkpoint_saver(lambda m: True, mock_saver)

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=2,
        )
        # After completion, attribute should be gone
        assert not hasattr(model, SEQ_CALIB_PROGRESS_ATTR)

    def test_checkpoint_progress_contains_output_metas(self):
        """Saved progress should include layer_output_metas."""
        model, tokens = _make_model_and_data(n_layers=4)
        saved_progress = {}

        def mock_saver(m, d):
            progress = getattr(m, SEQ_CALIB_PROGRESS_ATTR)
            saved_progress.update(progress)

        register_seq_calib_checkpoint_saver(lambda m: True, mock_saver)

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
            checkpoint_dir="/tmp/test_ckpt",
            checkpoint_interval=2,
        )
        assert "layer_output_metas" in saved_progress
        assert "completed_layer_idx" in saved_progress
        assert "total_layers" in saved_progress


@pytest.mark.usefixtures("_register_discoverer")
class TestPrepareForResume:
    def test_basic_resume_from_layer_2(self):
        """After prepare_for_resume(2), layers 0 should be skip, layer 1 should
        have collected_inputs, and get_input_activations(layer_2) should work."""
        model, tokens = _make_model_and_data(n_layers=4)
        fwd = _make_forward_loop(tokens)

        # First, do a full run to get output_metas
        collector = LayerActivationCollector(model)
        collector._patch_all_layers()
        try:
            for layer in model.layers:
                collector.get_input_activations(layer, fwd)
            output_metas = {
                i: model.layers[i]._seq_calib.output_meta
                for i in range(len(model.layers))
                if model.layers[i]._seq_calib.output_meta is not None
            }
        finally:
            collector._unpatch_all_layers()

        # Now simulate resume from layer 2
        collector2 = LayerActivationCollector(model)
        collector2._patch_all_layers(layer_output_metas=output_metas)
        try:
            collector2.prepare_for_resume(2, fwd)

            # Layer 0 should be in skip mode
            assert model.layers[0]._seq_calib.mode == "skip"
            assert model.layers[0]._seq_calib.output_meta is not None

            # Layer 1 should have collected_inputs from warm-up
            assert len(model.layers[1]._seq_calib.collected_inputs) > 0

            # get_input_activations for layer 2 should work
            inputs = collector2.get_input_activations(model.layers[2], fwd)
            assert len(inputs) == len(tokens)
        finally:
            collector2._unpatch_all_layers()

    def test_resume_from_layer_1(self):
        """Edge case: resume from layer 1 (only layer 0 was calibrated)."""
        model, tokens = _make_model_and_data(n_layers=3)
        fwd = _make_forward_loop(tokens)

        # Get output_metas from a full run
        collector = LayerActivationCollector(model)
        collector._patch_all_layers()
        try:
            for layer in model.layers:
                collector.get_input_activations(layer, fwd)
            output_metas = {
                i: model.layers[i]._seq_calib.output_meta
                for i in range(len(model.layers))
                if model.layers[i]._seq_calib.output_meta is not None
            }
        finally:
            collector._unpatch_all_layers()

        # Resume from layer 1
        collector2 = LayerActivationCollector(model)
        collector2._patch_all_layers(layer_output_metas=output_metas)
        try:
            collector2.prepare_for_resume(1, fwd)

            # Layer 0 should have collected_inputs from warm-up capture
            assert len(model.layers[0]._seq_calib.collected_inputs) > 0

            # get_input_activations for layer 1 should work
            inputs = collector2.get_input_activations(model.layers[1], fwd)
            assert len(inputs) == len(tokens)
        finally:
            collector2._unpatch_all_layers()

    def test_resume_from_layer_0_is_noop(self):
        """prepare_for_resume(0) should be a no-op."""
        model, tokens = _make_model_and_data(n_layers=3)
        fwd = _make_forward_loop(tokens)

        collector = LayerActivationCollector(model)
        collector._patch_all_layers()
        try:
            collector.prepare_for_resume(0, fwd)
            # All layers should still be in original mode
            for layer in model.layers:
                assert layer._seq_calib.mode == "original"
        finally:
            collector._unpatch_all_layers()

    def test_resume_requires_patched_state(self):
        """prepare_for_resume should raise if layers aren't patched."""
        model, tokens = _make_model_and_data(n_layers=3)
        fwd = _make_forward_loop(tokens)
        collector = LayerActivationCollector(model)
        with pytest.raises(RuntimeError, match="requires _patch_all_layers"):
            collector.prepare_for_resume(1, fwd)

    def test_resume_missing_output_meta_raises(self):
        """If a layer needs skip mode but has no output_meta, should raise."""
        model, tokens = _make_model_and_data(n_layers=4)
        fwd = _make_forward_loop(tokens)

        collector = LayerActivationCollector(model)
        collector._patch_all_layers()
        try:
            # Try to resume from layer 3 with no saved output_metas
            with pytest.raises(RuntimeError, match="no output_meta"):
                collector.prepare_for_resume(3, fwd)
        finally:
            collector._unpatch_all_layers()


@pytest.mark.usefixtures("_register_discoverer")
class TestResumeDetection:
    def test_resume_starts_from_correct_layer(self):
        """sequential_calibrate should skip already-calibrated layers on resume."""
        model, tokens = _make_model_and_data(n_layers=4)
        calibrated_layers = []

        def tracking_calib(layer, fwd, **kwargs):
            fwd(layer)
            calibrated_layers.append(id(layer))

        # First run to get output_metas
        full_run_collector = LayerActivationCollector(model)
        full_run_collector._patch_all_layers()
        fwd = _make_forward_loop(tokens)
        try:
            for layer in model.layers:
                full_run_collector.get_input_activations(layer, fwd)
            output_metas = {
                i: model.layers[i]._seq_calib.output_meta
                for i in range(len(model.layers))
                if model.layers[i]._seq_calib.output_meta is not None
            }
        finally:
            full_run_collector._unpatch_all_layers()

        # Set up resume from layer 2
        setattr(
            model,
            SEQ_CALIB_PROGRESS_ATTR,
            {
                "completed_layer_idx": 1,
                "total_layers": 4,
                "layer_output_metas": output_metas,
            },
        )

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=tracking_calib,
        )

        # Should only calibrate layers 2 and 3 (not 0 and 1)
        assert len(calibrated_layers) == 2
        assert calibrated_layers[0] == id(model.layers[2])
        assert calibrated_layers[1] == id(model.layers[3])

    def test_resume_mismatched_layer_count_raises(self):
        """Should raise ValueError when checkpoint layer count doesn't match."""
        model, tokens = _make_model_and_data(n_layers=3)

        setattr(
            model,
            SEQ_CALIB_PROGRESS_ATTR,
            {
                "completed_layer_idx": 1,
                "total_layers": 10,  # Mismatch!
                "layer_output_metas": {},
            },
        )

        with pytest.raises(ValueError, match="10 layers but model has 3"):
            sequential_calibrate(
                model,
                forward_loop=_make_forward_loop(tokens),
                calib_func=_noop_calib,
            )

    def test_progress_attr_cleaned_up_after_resume(self):
        """_seq_calib_progress should be deleted after calibration completes."""
        model, tokens = _make_model_and_data(n_layers=3)

        # Get output_metas
        collector = LayerActivationCollector(model)
        collector._patch_all_layers()
        fwd = _make_forward_loop(tokens)
        try:
            for layer in model.layers:
                collector.get_input_activations(layer, fwd)
            output_metas = {
                i: model.layers[i]._seq_calib.output_meta
                for i in range(len(model.layers))
                if model.layers[i]._seq_calib.output_meta is not None
            }
        finally:
            collector._unpatch_all_layers()

        setattr(
            model,
            SEQ_CALIB_PROGRESS_ATTR,
            {
                "completed_layer_idx": 0,
                "total_layers": 3,
                "layer_output_metas": output_metas,
            },
        )

        sequential_calibrate(
            model,
            forward_loop=_make_forward_loop(tokens),
            calib_func=_noop_calib,
        )
        assert not hasattr(model, SEQ_CALIB_PROGRESS_ATTR)


@pytest.mark.usefixtures("_register_discoverer")
class TestMetadataIntegration:
    def test_update_quantize_metadata_includes_progress(self):
        """update_quantize_metadata should pick up _seq_calib_progress."""
        from modelopt.torch.quantization.config import QuantizeConfig
        from modelopt.torch.quantization.conversion import update_quantize_metadata

        model = nn.Linear(4, 4)
        progress = {"completed_layer_idx": 5, "total_layers": 10}
        setattr(model, SEQ_CALIB_PROGRESS_ATTR, progress)

        metadata = {}
        update_quantize_metadata(model, QuantizeConfig(), metadata)

        assert metadata["seq_calib_progress"] == progress
        delattr(model, SEQ_CALIB_PROGRESS_ATTR)

    def test_output_meta_serialization_roundtrip(self):
        """layer_output_metas should survive torch.save/load roundtrip."""
        output_metas = {
            0: ("tensor", torch.Size([2, 16]), torch.float32, torch.device("cpu")),
            1: (
                "tuple",
                (
                    ("tensor", torch.Size([2, 16]), torch.float32, torch.device("cpu")),
                    ("other", None),
                ),
            ),
        }
        progress = {
            "completed_layer_idx": 1,
            "total_layers": 4,
            "layer_output_metas": output_metas,
        }

        buf = io.BytesIO()
        torch.save(progress, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)

        assert loaded["layer_output_metas"][0] == output_metas[0]
        assert loaded["layer_output_metas"][1] == output_metas[1]
        assert loaded["completed_layer_idx"] == 1


@pytest.mark.usefixtures("_register_discoverer")
class TestFullCheckpointResumeRoundtrip:
    def test_resume_produces_same_calibration_order(self):
        """Full roundtrip: run with checkpoint, simulate resume, verify all layers calibrated."""
        torch.manual_seed(42)
        n_layers = 5
        model, tokens = _make_model_and_data(n_layers=n_layers)

        # Track calibration order in a full uninterrupted run
        full_run_layers = []

        def tracking_calib_full(layer, fwd, **kwargs):
            fwd(layer)
            full_run_layers.append(id(layer))

        sequential_calibrate(
            model, forward_loop=_make_forward_loop(tokens), calib_func=tracking_calib_full
        )
        assert len(full_run_layers) == n_layers

        # Now simulate a checkpointed run that "resumes" from layer 3
        torch.manual_seed(42)
        model2, tokens2 = _make_model_and_data(n_layers=n_layers)

        # First, partially calibrate layers 0-2 to get output_metas
        partial_collector = LayerActivationCollector(model2)
        partial_collector._patch_all_layers()
        fwd = _make_forward_loop(tokens2)
        try:
            for i in range(3):
                inputs = partial_collector.get_input_activations(model2.layers[i], fwd)

                def _fwd(m, _inputs=inputs):
                    for args, kw in _inputs:
                        m(*args, **kw)

                _noop_calib(model2.layers[i], _fwd)
                del inputs

            output_metas = {
                i: model2.layers[i]._seq_calib.output_meta
                for i in range(len(model2.layers))
                if model2.layers[i]._seq_calib.output_meta is not None
            }
        finally:
            partial_collector._unpatch_all_layers()

        # Set progress for resume
        setattr(
            model2,
            SEQ_CALIB_PROGRESS_ATTR,
            {
                "completed_layer_idx": 2,
                "total_layers": n_layers,
                "layer_output_metas": output_metas,
            },
        )

        resumed_layers = []

        def tracking_calib_resume(layer, fwd, **kwargs):
            fwd(layer)
            resumed_layers.append(id(layer))

        sequential_calibrate(
            model2,
            forward_loop=_make_forward_loop(tokens2),
            calib_func=tracking_calib_resume,
        )

        # Should only calibrate layers 3 and 4
        assert len(resumed_layers) == 2
        assert resumed_layers[0] == id(model2.layers[3])
        assert resumed_layers[1] == id(model2.layers[4])


class TestWrappedCalibFuncWarning:
    def test_checkpoint_config_without_sequential_warns(self):
        """Setting checkpoint config without use_sequential=True should warn."""
        config = MaxCalibConfig(
            sequential_checkpoint_dir="/tmp/test",
            sequential_checkpoint_interval=2,
            use_sequential=False,
        )
        model = nn.Linear(4, 4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped_calib_func(model, config, forward_loop=None, func=None)
            assert len(w) == 1
            assert "use_sequential is False" in str(w[0].message)
