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

"""Unit tests for layerwise_calibrate and LayerActivationCollector."""

import copy
import hashlib
import json
import sys
from collections import deque

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization._layerwise_checkpoint import _assignment_checkpoint
from modelopt.torch.quantization._selected_calibration import _SelectedLayerCalibrator
from modelopt.torch.quantization._source_estimation import _estimate_source_activations
from modelopt.torch.quantization.model_calib import layerwise_calibrate
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.layerwise_calib import (
    LayerActivationCollector,
    _OutsideQuantizerCalibrator,
    _SkipLayer,
)
from modelopt.torch.utils.dataset_utils import create_forward_loop


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


class _TransformerWithLMHead(_SimpleTransformerModel):
    def __init__(self, n_layers=3, dim=16):
        super().__init__(n_layers=n_layers, dim=dim)
        self.lm_head = nn.Linear(dim, 32, bias=False)

    def forward(self, x, **kwargs):
        return self.lm_head(super().forward(x, **kwargs))


class _FlatMLP(nn.Module):
    """No decoder-layer structure -- should be rejected by layerwise_calibrate."""

    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class _TrackingQuantizer(TensorQuantizer):
    """Deterministic quantizer used to distinguish QDQ and FP activations."""

    def __init__(self):
        super().__init__(amax=3.0)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x / 2 if self.is_enabled else x


class _QuantizedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = _TrackingQuantizer()

    def forward(self, x):
        return self.quantizer(x)


class _QuantizedTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = _TrackingQuantizer()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.quantizer(x)


class _ModelWithQuantizedTail(nn.Module):
    def __init__(self, with_tail=True):
        super().__init__()
        self.layers = nn.ModuleList([_QuantizedLayer()])
        self.tail = _QuantizedTail() if with_tail else nn.Identity()

    def forward(self, x):
        return self.tail(self.layers[0](x))


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


@pytest.mark.parametrize("limit", [None, 192, 384])
def test_collector_collects_correct_number_of_inputs(monkeypatch, limit):
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTwoLayerModel(dim=8)
    collector = LayerActivationCollector(model, max_activation_bytes=limit)
    data = [torch.randn(2, 8) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector._patch_all_layers()
    try:
        inputs = collector.get_input_activations(model.layers[0], forward_loop)
        assert len(inputs) == 3
        if limit == 192:
            with pytest.raises(ValueError):
                collector.get_input_activations(model.layers[1], forward_loop)
        else:
            following = collector.get_input_activations(model.layers[1], forward_loop)
            for (args, _), (next_args, _) in zip(inputs, following):
                torch.testing.assert_close(next_args[0], model.layers[0]._original_forward(args[0]))
            if limit is not None:
                assert collector.peak_activation_bytes == limit
    finally:
        collector._unpatch_all_layers()


def test_layerwise_hybrid_cache_does_not_change_next_layer_inputs(monkeypatch):
    # GLM's native recurrent cache requires a newer optional Transformers version.
    glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    config = glm.Glm5NextTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        linear_head_dim=16,
        linear_num_heads=2,
        pad_token_id=0,
        hc_mult=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        index_head_dim=16,
        index_n_heads=2,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(20261023)
    model = glm.Glm5NextTextModel(config).eval()
    _register_test_discoverer(monkeypatch)
    inputs = torch.randint(1, 64, (1, 16))
    expected = {}
    handles = [
        layer.register_forward_pre_hook(lambda m, args: expected.update({m: args[0].clone()}))
        for layer in model.layers
    ]
    try:
        with torch.no_grad():
            model(inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    def calibrate_layer(layer, forward_loop):
        handle = layer.register_forward_pre_hook(
            lambda m, args: torch.testing.assert_close(args[0], expected[m], rtol=0, atol=0)
        )
        try:
            forward_loop(layer)
            forward_loop(layer)
        finally:
            handle.remove()

    layerwise_calibrate(model, lambda m: m(inputs), calibrate_layer)
    assert model.config.use_cache is True


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
    assert not hasattr(model.layers[0], "_layerwise_calib")
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
    assert not hasattr(model.layers[0], "_layerwise_calib")


# layerwise_calibrate tests
def test_layerwise_calib_raises_on_none_forward_loop(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=2)
    with pytest.raises(ValueError, match="forward_loop must not be None"):
        layerwise_calibrate(
            model,
            forward_loop=None,
            calib_func=lambda *a, **kw: None,
        )


def test_layerwise_calib_raises_on_unrecognized_model():
    model = _FlatMLP()
    with pytest.raises(ValueError, match="Could not find transformer layers"):
        layerwise_calibrate(
            model,
            forward_loop=lambda m: m(torch.randn(2, 16)),
            calib_func=lambda *a, **kw: None,
        )


def test_layerwise_calib_empty_forward_loop_raises(monkeypatch):
    """If forward_loop feeds no data, layerwise_calibrate raises RuntimeError."""
    _register_test_discoverer(monkeypatch)
    model = _SimpleTransformerModel(n_layers=2, dim=16)

    with pytest.raises(RuntimeError, match="collected no inputs during forward_loop"):
        layerwise_calibrate(
            model,
            forward_loop=lambda m: None,
            calib_func=lambda *a, **kw: None,
        )


@pytest.mark.parametrize("raises", [False, True])
def test_outside_calibrator_hides_and_restores_layer_aliases(raises):
    model = _ModelWithQuantizedTail()
    original = model.layers[0]
    model.layers_alias = model.layers
    model.layer_alias = original

    def calib_func(target, _forward_loop):
        assert isinstance(target.layers[0], _SkipLayer)
        assert target.layers_alias[0] is target.layers[0]
        assert target.layer_alias is target.layers[0]
        assert original not in target.modules()
        if raises:
            raise RuntimeError("injected")

    calibrator = _OutsideQuantizerCalibrator(
        model,
        [original],
        lambda target: target(torch.tensor([2.0])),
        calib_func,
        {},
        qdq_from_prev=True,
    )
    assert {(id(parent), name) for parent, name, _ in calibrator.transformer_layer_slots} == {
        (id(model), "layer_alias"),
        (id(model.layers), "0"),
    }

    if raises:
        with pytest.raises(RuntimeError, match="injected"):
            calibrator.calibrate()
    else:
        calibrator.calibrate()

    assert model.layers[0] is original
    assert model.layers_alias[0] is original
    assert model.layer_alias is original


@pytest.mark.parametrize(
    ("qdq_from_prev", "expected_tail_input"),
    [(True, 1.0), (False, 2.0)],
)
def test_layerwise_calibrates_only_outside_quantizers_with_full_model_forward(
    monkeypatch, qdq_from_prev, expected_tail_input
):
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: list(m.layers))],
    )
    model = _ModelWithQuantizedTail()
    decoder_quantizer = model.layers[0].quantizer
    calibrated_quantizers = []
    calibrated_targets = []
    decoder_amax_before_extra_pass = []

    def calib_func(target, target_forward_loop):
        calibrated_targets.append(target)
        if target is model:
            decoder_amax_before_extra_pass.append(decoder_quantizer._amax.clone())
        calibrated_quantizers.append(
            {id(module) for module in target.modules() if isinstance(module, TensorQuantizer)}
        )
        target_forward_loop(target)

    layerwise_calibrate(
        model,
        lambda m: m(torch.tensor([2.0])),
        calib_func,
        get_qdq_activations_from_prev_layer=qdq_from_prev,
    )

    assert calibrated_targets == [model.layers[0], model]
    assert calibrated_quantizers == [{id(decoder_quantizer)}, {id(model.tail.quantizer)}]
    torch.testing.assert_close(model.tail.inputs[-1], torch.tensor([expected_tail_input]))
    torch.testing.assert_close(decoder_quantizer._amax, decoder_amax_before_extra_pass[0])
    assert decoder_quantizer.calls == (2 if qdq_from_prev else 1)
    assert decoder_quantizer.is_enabled


def test_layerwise_skips_full_model_pass_without_outside_quantizer(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model = _ModelWithQuantizedTail(with_tail=False)
    calibrated_targets = []

    def calib_func(target, target_forward_loop):
        calibrated_targets.append(target)
        target_forward_loop(target)

    layerwise_calibrate(model, lambda m: m(torch.tensor([2.0])), calib_func)

    assert calibrated_targets == [model.layers[0]]


@pytest.mark.parametrize(
    ("offloaded", "with_tail", "warns"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
    ],
)
def test_layerwise_offload_warning_gating(monkeypatch, offloaded, with_tail, warns):
    _register_test_discoverer(monkeypatch)
    model = _ModelWithQuantizedTail(with_tail=with_tail)
    warnings = []
    monkeypatch.setattr(
        "modelopt.torch.quantization.utils.layerwise_calib.has_accelerate_offload",
        lambda target: target is model and offloaded,
    )
    monkeypatch.setattr(
        "modelopt.torch.quantization.utils.layerwise_calib.warn_rank_0", warnings.append
    )

    layerwise_calibrate(model, lambda m: m(torch.tensor([2.0])), lambda *_args: None)

    assert bool(warnings) is warns
    if warns:
        assert "CPU- or disk-offloaded" in warnings[0]


@pytest.mark.parametrize(
    ("skip_forward_without_activation_calib", "expected_forward_calls"),
    [(None, 2), (False, 2), (True, 1)],
)
def test_layerwise_max_outside_calibration_uses_configured_forward_behavior(
    monkeypatch, skip_forward_without_activation_calib, expected_forward_calls
):
    _register_test_discoverer(monkeypatch)
    config = copy.deepcopy(mtq.INT8_WEIGHT_ONLY_CFG)
    config["quant_cfg"].append({"quantizer_name": "*lm_head*weight_quantizer", "enable": True})
    algorithm = {"method": "max", "layerwise": {"enable": True}}
    if skip_forward_without_activation_calib is not None:
        algorithm["skip_forward_without_activation_calib"] = skip_forward_without_activation_calib
    config["algorithm"] = algorithm
    model = _TransformerWithLMHead(n_layers=1, dim=16)
    forward_calls = 0

    def forward_loop(target):
        nonlocal forward_calls
        forward_calls += 1
        target(torch.randint(0, 32, (2, 8)))

    mtq.quantize(model, config, forward_loop=forward_loop)

    assert forward_calls == expected_forward_calls


def test_layerwise_export_rejects_enabled_outside_quantizer(monkeypatch, tmp_path):
    _register_test_discoverer(monkeypatch)
    model = _ModelWithQuantizedTail()

    with pytest.raises(ValueError, match="outside transformer layers"):
        layerwise_calibrate(
            model,
            lambda m: m(torch.tensor([2.0])),
            lambda *_args, **_kwargs: None,
            export_dir=str(tmp_path / "export"),
        )

    assert not (tmp_path / "export").exists()


def test_layerwise_export_rejects_weight_only_outside_quantizer(monkeypatch, tmp_path):
    _register_test_discoverer(monkeypatch)
    config = copy.deepcopy(mtq.INT8_WEIGHT_ONLY_CFG)
    config["quant_cfg"].append({"quantizer_name": "*lm_head*weight_quantizer", "enable": True})
    algorithm = {
        "method": "max",
        "layerwise": {"enable": True, "export_dir": str(tmp_path / "export")},
    }
    config["algorithm"] = algorithm

    with pytest.raises(ValueError, match="outside transformer layers"):
        mtq.quantize(
            _TransformerWithLMHead(n_layers=1, dim=16),
            config,
            forward_loop=lambda model: model(torch.randint(0, 32, (2, 8))),
        )

    assert not (tmp_path / "export").exists()


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


def test_inter_layer_op_raises_descriptive_error(monkeypatch):
    """Real-device inter-layer ops on meta skip placeholders raise an actionable error."""
    _register_test_discoverer(monkeypatch)
    model = _InterLayerNormModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16) for _ in range(3)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        with pytest.raises(RuntimeError, match="non-layerwise calibration"):
            for layer in model.layers:
                collector.get_input_activations(layer, forward_loop)
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
        assert model.layers[0]._layerwise_calib.output_meta is None

        # Calibrating layer 1 puts layer 0 into run, which sets output_meta
        collector.get_input_activations(model.layers[1], forward_loop)
        meta = model.layers[0]._layerwise_calib.output_meta
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
        assert len(model.layers[1]._layerwise_calib.cached_inputs) == n_batches

        # After the forward loop, all cached inputs should be consumed
        forward_loop(model)
        assert len(model.layers[1]._layerwise_calib.cached_inputs) == 0
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
            return [model.layers[i]._layerwise_calib.mode for i in range(5)]

        collector._set_layer_states(0)
        assert modes() == ["capture", "original", "original", "original", "original"]

        model.layers[0]._layerwise_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(1)
        assert modes() == ["run", "capture", "original", "original", "original"]

        model.layers[1]._layerwise_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(2)
        assert modes() == ["skip", "run", "capture", "original", "original"]

        model.layers[2]._layerwise_calib.collected_inputs = [fake_inp]
        collector._set_layer_states(3)
        assert modes() == ["skip", "skip", "run", "capture", "original"]

        model.layers[3]._layerwise_calib.collected_inputs = [fake_inp]
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


class _OutOfMemoryOnceBlock(_DecoderBlock):
    """Decoder block whose next forward runs out of memory when ``oom`` is set."""

    oom = False

    def forward(self, x, **kwargs):
        if self.oom:
            self.oom = False
            raise torch.cuda.OutOfMemoryError("out of memory")
        return super().forward(x, **kwargs)


class _InputIdsModel(_SimpleTransformerModel):
    """Takes ``input_ids`` as the calibration forward loop passes it."""

    def __init__(self):
        super().__init__(n_layers=3, dim=16)
        self.layers[0] = _OutOfMemoryOnceBlock(16)

    def forward(self, input_ids, **kwargs):
        return super().forward(input_ids)


def test_out_of_memory_in_a_replayed_layer_is_not_split(monkeypatch):
    """A batch split on OOM would replay the next cached batch in its place, so the OOM must stop."""
    _register_test_discoverer(monkeypatch)
    model = _InputIdsModel()
    batches = [{"input_ids": torch.randint(0, 32, (4, 8))} for _ in range(3)]
    forward_loop = create_forward_loop(dataloader=batches)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        collector.get_input_activations(model.layers[0], forward_loop)
        model.layers[0].oom = True  # layer 0 replays its cached batches while layer 1 is captured
        with pytest.raises(RuntimeError) as error:
            collector.get_input_activations(model.layers[1], forward_loop)
        assert not isinstance(error.value, torch.cuda.OutOfMemoryError)
    finally:
        collector._unpatch_all_layers()


def test_run_asserts_on_empty_cached_inputs(monkeypatch):
    """A layer in 'run' mode with no cached inputs must raise AssertionError."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=2, dim=16)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        model.layers[0]._layerwise_calib.mode = "run"
        model.layers[0]._layerwise_calib.cached_inputs = deque()

        with pytest.raises(AssertionError, match="no cached inputs to replay"):
            model(torch.randn(2, 16))
    finally:
        collector._unpatch_all_layers()


def test_cleanup_removes_layerwise_calib_attr(monkeypatch):
    """After unpatch, no layer should have the _layerwise_calib attribute."""
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
        assert not hasattr(layer, "_layerwise_calib"), (
            f"Layer {i} still has _layerwise_calib after cleanup"
        )
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

    originals = list(model.layers)
    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in originals:
            collector.get_input_activations(layer, forward_loop)

        # After full calibration, layers 0 and 1 have been through 'run' and have output_meta.
        # Access via originals since skip-position entries are now _SkipLayer dummies.
        meta_0 = originals[0]._layerwise_calib.output_meta
        meta_1 = originals[1]._layerwise_calib.output_meta
        assert meta_0 is not None
        assert meta_1 is not None
        # SmallBlock returns 3-element tuple, BigBlock returns 1-element tuple
        assert len(meta_0[1]) == 3
        assert len(meta_1[1]) == 1
    finally:
        collector._unpatch_all_layers()


# ---------------------------------------------------------------------------
# _SkipLayer swap / restore tests
# ---------------------------------------------------------------------------


def test_skip_layers_replaced_with_dummy(monkeypatch):
    """After calibrating enough layers, skip-position entries must be _SkipLayer with no params."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16) for _ in range(2)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:
        for layer in list(model.layers):
            collector.get_input_activations(layer, forward_loop)

        # Layers 0..2 should be dummies (swapped when calibrating layers 2..4)
        for i in range(3):
            assert isinstance(model.layers[i], _SkipLayer), f"Layer {i} should be _SkipLayer"
            assert list(model.layers[i].parameters()) == [], (
                f"Layer {i} dummy should have no params"
            )
        # Layers 3 (run) and 4 (original) remain real
        for i in range(3, 5):
            assert not isinstance(model.layers[i], _SkipLayer), f"Layer {i} should still be real"
    finally:
        collector._unpatch_all_layers()


def test_cleanup_restores_original_layers(monkeypatch):
    """After _unpatch_all_layers, all ModuleList entries must be the original modules."""
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    originals = list(model.layers)
    data = [torch.randn(2, 16)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    for layer in originals:
        collector.get_input_activations(layer, forward_loop)
    collector._unpatch_all_layers()

    for i, orig in enumerate(originals):
        assert model.layers[i] is orig, f"Layer {i} not restored to original after cleanup"
        assert not hasattr(orig, "_layerwise_calib"), f"Layer {i} still has _layerwise_calib"


def _int8_cfg_with_algorithm(algorithm: dict) -> dict:
    """Build an INT8 quant config with the given ``algorithm`` block.

    Starts from a real shipped INT8 config (so include/exclude rules match
    production PTQ) and overrides only the ``algorithm`` entry. Layerwise
    calibration runs only when ``algorithm`` sets ``layerwise={"enable": True}``;
    a plain ``{"method": ...}`` yields the standard non-layerwise (sequential) path.
    """
    cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    cfg["algorithm"] = algorithm
    return cfg


def test_layerwise_calibrate_uses_global_layer_tqdm(monkeypatch):
    _register_test_discoverer(monkeypatch)

    class _FakeTqdm:
        instances = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.postfixes = []
            self.updates = []
            self.closed = False
            _FakeTqdm.instances.append(self)

        def set_postfix_str(self, status, refresh=True):
            self.postfixes.append((status, refresh))

        def update(self, n=1):
            self.updates.append(n)

        def close(self):
            self.closed = True

    monkeypatch.setattr("modelopt.torch.quantization.model_calib.tqdm", _FakeTqdm)

    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=3, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    def forward_loop(m):
        for batch in calib_data:
            m(batch)

    def calib_func(layer, layer_forward_loop):
        layer_forward_loop(layer)

    layerwise_calibrate(model, forward_loop, calib_func)

    assert len(_FakeTqdm.instances) == 1
    pbar = _FakeTqdm.instances[0]
    assert pbar.kwargs["total"] == 3
    assert pbar.kwargs["initial"] == 0
    assert pbar.kwargs["desc"] == "Layerwise calibration"
    assert pbar.kwargs["dynamic_ncols"] is True
    assert pbar.updates == [1, 1, 1]
    assert pbar.closed
    assert any(status.startswith("Calibrating layer 1/3") for status, _ in pbar.postfixes)
    assert any(status.startswith("Calibrating layer 3/3") for status, _ in pbar.postfixes)


def _awq_layerwise_config() -> dict:
    """INT4 weight-only AWQ config sized for the _DecoderBlock test model."""
    cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    # Resize AWQ block to fit dim=16 hidden.
    for entry in cfg["quant_cfg"]:
        if entry.get("quantizer_name") == "*weight_quantizer":
            entry.setdefault("cfg", {})["block_sizes"] = {-1: 8, "type": "static"}
    cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 0.5, "layerwise": {"enable": True}}
    return cfg


def _svdquant_layerwise_config() -> dict:
    """SVDQuant config sized for the _DecoderBlock test model."""
    cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    for entry in cfg["quant_cfg"]:
        if entry.get("quantizer_name") == "*weight_quantizer":
            entry.setdefault("cfg", {})["block_sizes"] = {-1: 8, "type": "static"}
    cfg["algorithm"] = {"method": "svdquant", "lowrank": 4, "layerwise": {"enable": True}}
    return cfg


def test_mtq_quantize_layerwise_e2e_max(monkeypatch):
    """End-to-end: mtq.quantize with layerwise enabled produces populated amax values.

    ``max`` is the representative algorithm for the layerwise happy path because
    every other algorithm seeds amax via max_calibrate first — if max works, the
    shared skip/run/capture machinery is sound. Other algorithms are covered by
    the dispatch-only test below to avoid hardware requirements (e.g. gptq needs
    CUDA) or unnecessary duplication.
    """
    _register_test_discoverer(monkeypatch)
    config = _int8_cfg_with_algorithm({"method": "max", "layerwise": {"enable": True}})

    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=3, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    def forward_loop(m):
        for batch in calib_data:
            m(batch)

    model = mtq.quantize(model, config, forward_loop=forward_loop)

    for i, layer in enumerate(model.layers):
        assert not isinstance(layer, _SkipLayer), f"layer {i} left as _SkipLayer"
        assert not hasattr(layer, "_layerwise_calib"), f"layer {i} leaked _layerwise_calib"

    amax_count = sum(
        1
        for layer in model.layers
        for module in layer.modules()
        if (
            isinstance(module, TensorQuantizer)
            and module.is_enabled
            and getattr(module, "_amax", None) is not None
        )
    )
    assert amax_count > 0, "no TensorQuantizer in decoder layers had _amax populated"

    with torch.no_grad():
        model(calib_data[0])


def test_mtq_quantize_layerwise_calibrates_lm_head(monkeypatch):
    _register_test_discoverer(monkeypatch)
    config = _int8_cfg_with_algorithm(
        {
            "method": "max",
            "layerwise": {"enable": True, "get_qdq_activations_from_prev_layer": True},
        }
    )
    config["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": True})
    model = _TransformerWithLMHead(n_layers=2, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8))]

    mtq.quantize(model, config, forward_loop=lambda m: [m(batch) for batch in calib_data])

    assert model.lm_head.input_quantizer._amax is not None
    with torch.no_grad():
        model(calib_data[0])


@pytest.mark.parametrize(
    "algorithm",
    ["gptq", "awq_lite", "smoothquant", "mse"],
)
def test_mtq_quantize_layerwise_dispatches_for_algorithm(monkeypatch, algorithm):
    """Every layerwise-supporting algorithm must route through layerwise_calibrate.

    Stubs layerwise_calibrate to a spy so the dispatch contract is checked without
    running the algorithm's full calibration — lets ``gptq`` (CUDA-only at runtime)
    and other expensive algorithms participate in CPU unit tests.
    """
    spy: dict = {}

    def stub(model, forward_loop, calib_func, **kwargs):
        spy["calib_func"] = calib_func
        spy["kwargs"] = kwargs

    monkeypatch.setattr("modelopt.torch.quantization.mode.layerwise_calibrate", stub)

    if algorithm == "awq_lite":
        config = _awq_layerwise_config()
    else:
        config = _int8_cfg_with_algorithm({"method": algorithm, "layerwise": {"enable": True}})

    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=2, dim=16)
    mtq.quantize(
        model,
        config,
        forward_loop=lambda m: m(torch.randint(0, 32, (2, 8))),
    )

    assert "calib_func" in spy, f"{algorithm} did not dispatch through layerwise_calibrate"
    assert callable(spy["calib_func"])


def test_mtq_quantize_layerwise_raises_for_unsupported_algorithm():
    """Modes with ``_supports_layerwise = False`` must raise a clear ValueError."""
    config = _svdquant_layerwise_config()
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=2, dim=16)
    with pytest.raises(ValueError, match=r"does not support layerwise.enable=True"):
        mtq.quantize(
            model,
            config,
            forward_loop=lambda m: m(torch.randint(0, 32, (2, 8))),
        )


def _collect_amax(model):
    return {
        name: q._amax.clone().detach()
        for name, q in model.named_modules()
        if isinstance(q, TensorQuantizer) and q.is_enabled and getattr(q, "_amax", None) is not None
    }


def _assert_amax_close(actual, expected, label):
    assert set(actual) == set(expected), (
        f"Different quantizers populated for {label}: "
        f"missing={set(expected) - set(actual)}, extra={set(actual) - set(expected)}"
    )
    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=1e-5,
            atol=1e-6,
            msg=f"{label}: amax mismatch at {name}",
        )


def test_layerwise_no_qdq_matches_sequential_amax(monkeypatch):
    """Layerwise + ``get_qdq_activations_from_prev_layer=False`` must produce the
    same per-quantizer amax as the non-layerwise (sequential) max-calibration
    flow. Both paths feed every layer full-precision activations.
    """
    _register_test_discoverer(monkeypatch)

    torch.manual_seed(0)
    model_seq = _SimpleTransformerModel(n_layers=3, dim=16)
    model_lw = copy.deepcopy(model_seq)
    calib_data = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    def fwd(m):
        for batch in calib_data:
            m(batch)

    mtq.quantize(model_seq, _int8_cfg_with_algorithm({"method": "max"}), forward_loop=fwd)
    mtq.quantize(
        model_lw,
        _int8_cfg_with_algorithm(
            {
                "method": "max",
                "layerwise": {"enable": True, "get_qdq_activations_from_prev_layer": False},
            }
        ),
        forward_loop=fwd,
    )

    seq_amax = _collect_amax(model_seq)
    assert seq_amax, "sequential calibration populated no amax values"
    _assert_amax_close(_collect_amax(model_lw), seq_amax, "layerwise vs sequential")


@pytest.mark.parametrize(
    ("qdq_from_prev", "estimate", "recover"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        pytest.param(
            False,
            False,
            True,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
        pytest.param(
            True,
            False,
            True,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
    ],
)
def test_selected_layerwise_mixed_algorithms_match_group_calibration(
    monkeypatch, tmp_path, qdq_from_prev, estimate, recover
):
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(42)
    source = _SimpleTransformerModel(n_layers=3).eval()
    data = [torch.randint(0, 32, (2, 8)) for _ in range(2)]
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *[
                {"quantizer_name": pattern + ".weight_quantizer", "cfg": {"num_bits": 3, "axis": 0}}
                for pattern in ("layers.*.attn", "layers.*.ffn.*")
            ],
            *[
                {
                    "quantizer_name": pattern + ".input_quantizer",
                    "cfg": {"num_bits": 4, "axis": None},
                }
                for pattern in ("layers.*.attn", "layers.*.ffn.*")
            ],
        ],
        "algorithm": None,
    }
    layerwise = {"enable": True, "get_qdq_activations_from_prev_layer": qdq_from_prev or estimate}
    algorithms = {
        "attn": {"method": "max", "layerwise": layerwise},
        "ffn": [
            {"method": "max"},
            {
                "method": "mse",
                "step_size": 0.25,
                "start_multiplier": 0.5,
                "stop_multiplier": 1.0,
                "layerwise": layerwise,
            },
        ],
    }
    groups = [
        ([f"layers.{i}.{name}"], algorithm)
        for i in range(3)
        for name, algorithm in algorithms.items()
    ]
    original_groups = copy.deepcopy(groups)

    def replay(batches):
        def forward_loop(model):
            for batch in batches:
                model(batch)

        return forward_loop

    reference = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
    inputs = [reference.embed(batch).detach() for batch in data]
    for layer in reference.layers:
        # Capture the source inputs once, then calibrate each group through the public API.
        captured = {name: [] for name in algorithms}
        handles = [
            getattr(layer, name).register_forward_pre_hook(
                lambda m, args, name=name: captured[name].append(args[0].detach().clone())
            )
            for name in algorithms
        ]
        with mtq.set_quantizer_by_cfg_context(layer, [{"quantizer_name": "*", "enable": False}]):
            source_outputs = [layer(batch).detach() for batch in inputs]
        for handle in handles:
            handle.remove()
        for name, algorithm in algorithms.items():
            chain = copy.deepcopy(algorithm if isinstance(algorithm, list) else [algorithm])
            for step in chain:
                step["layerwise"] = {"enable": False}
            mtq.calibrate(getattr(layer, name), chain, replay(captured[name]))
        inputs = [layer(batch).detach() for batch in inputs] if qdq_from_prev else source_outputs

    for order_index, order in enumerate((groups, list(reversed(groups)))):
        actual = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
        flags = {q: q.is_enabled for q in actual.modules() if isinstance(q, TensorQuantizer)}
        if estimate:
            result = _estimate_source_activations(
                actual, order, replay(data), max_activation_bytes=2**20
            )
            assert result["peak_activation_bytes"] <= 2**20
        else:
            callback = _SelectedLayerCalibrator(actual, order)
            if recover:
                options = dict(
                    callback.layerwise_config,
                    save_every=2 if qdq_from_prev else 1,
                    calib_mutates_weights=False,
                )
                directory = tmp_path / str(order_index)
                binding = {
                    "assignment_digest": hashlib.sha256(
                        json.dumps(groups, sort_keys=True).encode()
                    ).hexdigest(),
                    "provenance": {
                        "source": "fixture-source",
                        "inputs": "fixture-data",
                        "code": "fixture-code",
                    },
                    "num_layers": 3,
                    "max_bytes": 2**20,
                    "save_every": options["save_every"],
                }
                last_layer = actual.layers[2]

                def interrupt(layer, forward_loop):
                    if layer is last_layer:
                        raise RuntimeError("test interruption")
                    callback(layer, forward_loop)

                with _assignment_checkpoint(directory, **binding) as checkpoint:
                    publish = checkpoint.store._publish

                    def fail_publication(destination, payload):
                        if not qdq_from_prev and len(payload["layers"]) == 2:
                            raise RuntimeError("test interruption")
                        publish(destination, payload)

                    monkeypatch.setattr(checkpoint.store, "_publish", fail_publication)
                    with pytest.raises(RuntimeError, match="test interruption"):
                        layerwise_calibrate(
                            actual, replay(data), interrupt, _checkpoint_state=checkpoint, **options
                        )
                committed = 2 if qdq_from_prev else 1
                prefix = {
                    p: (p.stat().st_ino, p.read_bytes())
                    for d in sorted(directory.glob("layer_*"))[:committed]
                    for p in d.iterdir()
                    if p.name != "next_inputs.pt"
                }
                with (
                    pytest.raises(ValueError),
                    _assignment_checkpoint(directory, **dict(binding, assignment_digest="0" * 64)),
                ):
                    pytest.fail("Accepted another assignment")
                boundary_inputs = directory / f"layer_{committed - 1:04d}" / "next_inputs.pt"
                saved_inputs = boundary_inputs.read_bytes()
                boundary_inputs.write_bytes(saved_inputs + b"corrupt")
                with pytest.raises(ValueError), _assignment_checkpoint(directory, **binding):
                    pytest.fail("Accepted a corrupted activation boundary")
                boundary_inputs.write_bytes(saved_inputs)
                actual = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
                callback = _SelectedLayerCalibrator(actual, order)
                with _assignment_checkpoint(directory, **binding) as checkpoint:
                    assert checkpoint.start_layer == committed
                    layerwise_calibrate(
                        actual, replay(data), callback, _checkpoint_state=checkpoint, **options
                    )
                assert all(
                    (p.stat().st_ino, p.read_bytes()) == state for p, state in prefix.items()
                )
                assert not list(directory.glob("layer_*/next_inputs.pt"))
                assert len(list(directory.glob(".boundary-????????"))) == 1

                def unexpected(*args):
                    pytest.fail("Completed recovery repeated calibration or activation capture")

                actual = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
                flags = {
                    q: q.is_enabled for q in actual.modules() if isinstance(q, TensorQuantizer)
                }
                with _assignment_checkpoint(directory, **binding) as checkpoint:
                    assert checkpoint.start_layer == 3
                    layerwise_calibrate(
                        actual, unexpected, unexpected, _checkpoint_state=checkpoint, **options
                    )
                if qdq_from_prev and order_index == 0:
                    # A storage failure must leave a bounded, recoverable prefix.
                    limited = tmp_path / "limited"
                    limits = dict(binding, max_bytes=65536, calib_mutates_weights=True)
                    settings = dict(options, calib_mutates_weights=True)
                    actual = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
                    callback = _SelectedLayerCalibrator(actual, order)
                    with (
                        _assignment_checkpoint(limited, **limits) as checkpoint,
                        pytest.raises((ValueError, RuntimeError)),
                    ):
                        layerwise_calibrate(
                            actual,
                            replay(data),
                            callback,
                            _checkpoint_state=checkpoint,
                            **settings,
                        )
                    assert sum(p.stat().st_size for p in limited.rglob("*") if p.is_file()) <= 65536
                    actual = mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))
                    flags = {
                        q: q.is_enabled for q in actual.modules() if isinstance(q, TensorQuantizer)
                    }
                    callback = _SelectedLayerCalibrator(actual, order)
                    with _assignment_checkpoint(
                        limited, **dict(limits, max_bytes=2**20)
                    ) as checkpoint:
                        layerwise_calibrate(
                            actual,
                            replay(data),
                            callback,
                            _checkpoint_state=checkpoint,
                            **settings,
                        )
            else:
                layerwise_calibrate(actual, replay(data), callback, **callback.layerwise_config)
        assert flags == {q: q.is_enabled for q in flags}
        assert actual.state_dict().keys() == reference.state_dict().keys()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(actual(data[0]), reference(data[0]), rtol=0, atol=0)
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(
                parameter, dict(source.named_parameters())[name], rtol=0, atol=0
            )
    assert groups == original_groups


@pytest.mark.parametrize(
    "invalid", ["activation_budget", "shared_weights", "training", "meta", "uncovered"]
)
def test_source_estimation_refuses_layout_or_budget_before_calibration(monkeypatch, invalid):
    _register_test_discoverer(monkeypatch)
    model = mtq.quantize(_SimpleTransformerModel(n_layers=2).eval(), _int8_cfg_with_algorithm(None))
    groups = [([f"layers.{i}"], {"method": "max", "layerwise": {"enable": True}}) for i in range(2)]
    if invalid == "shared_weights":
        model.layers[1].attn.weight = model.layers[0].attn.weight
    elif invalid == "training":
        model.train()
    elif invalid == "meta":
        model.to("meta")
    elif invalid == "uncovered":
        groups.pop()
    before = copy.deepcopy(model.state_dict())
    flags = {q: q.is_enabled for q in model.modules() if isinstance(q, TensorQuantizer)}
    with pytest.raises(ValueError):
        _estimate_source_activations(
            model,
            groups,
            lambda m: m(torch.zeros(2, 8, dtype=torch.long)),
            max_activation_bytes=1 if invalid == "activation_budget" else 2**20,
        )
    assert flags == {q: q.is_enabled for q in flags}
    assert not any(hasattr(m, "_layerwise_calib") for m in model.modules())
    assert before.keys() == model.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    if invalid == "activation_budget":
        result = _estimate_source_activations(
            model,
            groups,
            lambda m: m(torch.zeros(2, 8, dtype=torch.long)),
            max_activation_bytes=2**20,
        )
        assert 1 < result["peak_activation_bytes"] <= 2**20


@pytest.mark.parametrize("invalid", ["overlap", "cross_layer", "traversal", "mutating"])
def test_selected_layerwise_rejects_invalid_groups_before_calibration(monkeypatch, invalid):
    _register_test_discoverer(monkeypatch)
    model = mtq.quantize(_SimpleTransformerModel(n_layers=2), _int8_cfg_with_algorithm(None))
    algorithm = {"method": "max", "layerwise": {"enable": True}}
    groups = [(["layers.0.attn"], algorithm)]
    if invalid == "overlap":
        groups.append((["layers.0"], algorithm))
    elif invalid == "cross_layer":
        groups[0][0].append("layers.1.attn")
    elif invalid == "traversal":
        groups.append(
            (
                ["layers.1.attn"],
                {
                    "method": "max",
                    "layerwise": {
                        "enable": True,
                        "get_qdq_activations_from_prev_layer": False,
                    },
                },
            )
        )
    else:
        groups[0] = (["layers.0.attn"], {"method": "gptq", "layerwise": {"enable": True}})
    before = copy.deepcopy(model.state_dict())
    flags = {q: q.is_enabled for q in model.modules() if isinstance(q, TensorQuantizer)}
    with pytest.raises(ValueError):
        _SelectedLayerCalibrator(model, groups)
    assert flags == {q: q.is_enabled for q in flags}
    assert before.keys() == model.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_layerwise_no_qdq_captures_inputs_before_calib_func_mutates_weights(monkeypatch):
    """A destructive ``calib_func`` (zeros weights) must not affect what is
    captured for downstream layers under ``qdq_from_prev=False`` — otherwise
    weight-mutating algorithms (GPTQ/AWQ/SmoothQuant) silently propagate
    updates forward and break the "identical to non-layerwise pass" contract.
    """
    _register_test_discoverer(monkeypatch)
    calib_data = [torch.randint(0, 32, (2, 8))]

    def run_and_capture(calib_func):
        torch.manual_seed(0)
        model = _SimpleTransformerModel(n_layers=3, dim=16)
        captured: dict[int, torch.Tensor] = {}
        real = LayerActivationCollector.cache_outputs_for_next_layer_calib

        def spy(self, layer, fwd):
            result = real(self, layer, fwd)
            captured[self._layer_to_idx[layer] + 1] = result[0][0][0].clone().detach()
            return result

        with monkeypatch.context() as m:
            m.setattr(LayerActivationCollector, "cache_outputs_for_next_layer_calib", spy)
            layerwise_calibrate(
                model,
                forward_loop=lambda mm: [mm(b) for b in calib_data],
                calib_func=calib_func,
                get_qdq_activations_from_prev_layer=False,
            )
        return captured

    def identity(layer, fwd, **_):
        fwd(layer)

    def destructive(layer, fwd, **_):
        fwd(layer)
        for sub in layer.modules():
            if isinstance(sub, nn.Linear):
                sub.weight.data.zero_()

    benign = run_and_capture(identity)
    mutated = run_and_capture(destructive)

    for i in (1, 2):
        torch.testing.assert_close(mutated[i], benign[i], rtol=1e-5, atol=1e-6)


def _layer_dir_names(checkpoint_dir):
    return sorted(p.name for p in checkpoint_dir.iterdir() if p.name.startswith("layer_"))


def test_layerwise_save_every_writes_next_inputs_only_at_window_boundaries(monkeypatch, tmp_path):
    """With save_every=2 on a 4-layer model, every layer dir is still written
    (so resume can replay skip layers), but ``next_inputs.pt`` — the large
    activation cache — appears only at window boundaries.
    """
    _register_test_discoverer(monkeypatch)

    config = _int8_cfg_with_algorithm(
        {
            "method": "max",
            "layerwise": {
                "enable": True,
                "checkpoint_dir": str(tmp_path),
                "save_every": 2,
            },
        }
    )
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=4, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8))]
    mtq.quantize(model, config, forward_loop=lambda m: [m(b) for b in calib_data])

    # Window boundaries are layer_idx 1 and 3 -> all 4 layer dirs exist (window-save).
    assert _layer_dir_names(tmp_path) == [
        "layer_0000",
        "layer_0001",
        "layer_0002",
        "layer_0003",
    ]
    # next_inputs.pt is only at the boundary layers (the resume restart points).
    assert not (tmp_path / "layer_0000" / "next_inputs.pt").exists()
    assert (tmp_path / "layer_0001" / "next_inputs.pt").exists()
    assert not (tmp_path / "layer_0002" / "next_inputs.pt").exists()
    # Last layer never has next_inputs.pt (no subsequent layer).
    assert not (tmp_path / "layer_0003" / "next_inputs.pt").exists()


@pytest.mark.parametrize(
    ("scenario", "n_layers", "save_every", "calib_mutates_weights", "rewind_to"),
    [
        # Pins the quantizer_buffers.pt restore path (no weights.pt on disk).
        ("non_mutating", 3, 1, False, 0),
        # Pins the per-call snapshot fix: each save() captures the
        # just-calibrated layer's state before the next-layer capture forward
        # swaps it to _SkipLayer.
        ("save_every", 4, 2, True, 1),
    ],
)
def test_layerwise_checkpoint_resume_matches_one_shot_amax(
    monkeypatch, tmp_path, scenario, n_layers, save_every, calib_mutates_weights, rewind_to
):
    """Full run → rewind manifest → fresh resume reproduces one-shot ``_amax``.

    Single test covering both checkpoint optimizations. For the
    non-mutating calibration case also asserts the on-disk shape (no
    ``weights.pt``, ``quantizer_buffers.pt`` present per layer).
    """
    _register_test_discoverer(monkeypatch)

    calib_data = [torch.randint(0, 32, (2, 8))]
    forward_loop = lambda m: [m(b) for b in calib_data]  # noqa: E731

    def build_cfg(ckpt_dir):
        return _int8_cfg_with_algorithm(
            {
                "method": "max",
                "layerwise": {
                    "enable": True,
                    "checkpoint_dir": str(ckpt_dir),
                    "save_every": save_every,
                    "calib_mutates_weights": calib_mutates_weights,
                },
            }
        )

    baseline_dir = tmp_path / "baseline"
    torch.manual_seed(0)
    baseline_model = _SimpleTransformerModel(n_layers=n_layers, dim=16)
    mtq.quantize(baseline_model, build_cfg(baseline_dir), forward_loop=forward_loop)
    baseline_amax = _collect_amax(baseline_model)
    assert baseline_amax

    resume_dir = tmp_path / "resume"
    torch.manual_seed(0)
    setup_model = _SimpleTransformerModel(n_layers=n_layers, dim=16)
    mtq.quantize(setup_model, build_cfg(resume_dir), forward_loop=forward_loop)

    if scenario == "non_mutating":
        for name in _layer_dir_names(resume_dir):
            d = resume_dir / name
            assert not (d / "weights.pt").exists()
            assert (d / "quantizer_buffers.pt").exists()

    (resume_dir / "manifest.json").write_text(
        json.dumps(
            {
                "last_completed_layer": rewind_to,
                "num_layers": n_layers,
                "save_every": save_every,
                "calib_mutates_weights": calib_mutates_weights,
            }
        )
    )

    torch.manual_seed(0)
    resumed_model = _SimpleTransformerModel(n_layers=n_layers, dim=16)
    mtq.quantize(resumed_model, build_cfg(resume_dir), forward_loop=forward_loop)

    _assert_amax_close(_collect_amax(resumed_model), baseline_amax, f"{scenario} resume")


def test_layerwise_save_every_mid_window_crash_recovers_at_prev_boundary(monkeypatch, tmp_path):
    """A crash inside a window must not advance ``last_completed_layer``; resume
    re-calibrates the unfinished window from the previous boundary.

    Monkeypatches ``torch.save`` to raise on the second window's first per-layer
    write (layer 2), then asserts the on-disk manifest still points at layer 1
    (the previous boundary).
    """
    _register_test_discoverer(monkeypatch)

    cfg = _int8_cfg_with_algorithm(
        {
            "method": "max",
            "layerwise": {
                "enable": True,
                "checkpoint_dir": str(tmp_path),
                "save_every": 2,
            },
        }
    )
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=4, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8))]

    real_torch_save = torch.save
    state = {"crashed": False}

    def crashing_torch_save(obj, path, *args, **kwargs):
        # Crash on the first per-layer file write for layer 2 (mid-window).
        if not state["crashed"] and "layer_0002" in str(path):
            state["crashed"] = True
            raise RuntimeError("simulated crash during layer 2 save")
        return real_torch_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(
        "modelopt.torch.quantization.utils.layerwise_calib.torch.save",
        crashing_torch_save,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        mtq.quantize(model, cfg, forward_loop=lambda m: [m(b) for b in calib_data])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["last_completed_layer"] == 1, f"manifest leaked mid-window state: {manifest}"


def test_layerwise_checkpoint_mismatch_save_every_raises(monkeypatch, tmp_path):
    """Resuming with a different ``save_every`` than the checkpoint was produced
    with must raise — the on-disk window layout assumes a fixed value.
    """
    _register_test_discoverer(monkeypatch)

    cfg_first = _int8_cfg_with_algorithm(
        {
            "method": "max",
            "layerwise": {
                "enable": True,
                "checkpoint_dir": str(tmp_path),
                "save_every": 2,
            },
        }
    )
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=4, dim=16)
    calib_data = [torch.randint(0, 32, (2, 8))]
    mtq.quantize(model, cfg_first, forward_loop=lambda m: [m(b) for b in calib_data])

    # Rewind manifest so the second run sees an in-progress resume,
    # then change save_every to trigger the mismatch check.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "last_completed_layer": 1,
                "num_layers": 4,
                "save_every": 2,
                "calib_mutates_weights": True,
            }
        )
    )
    cfg_mismatched = _int8_cfg_with_algorithm(
        {
            "method": "max",
            "layerwise": {
                "enable": True,
                "checkpoint_dir": str(tmp_path),
                "save_every": 4,
            },
        }
    )
    torch.manual_seed(0)
    fresh_model = _SimpleTransformerModel(n_layers=4, dim=16)
    with pytest.raises(ValueError, match="save_every mismatch"):
        mtq.quantize(fresh_model, cfg_mismatched, forward_loop=lambda m: [m(b) for b in calib_data])
