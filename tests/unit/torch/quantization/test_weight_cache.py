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

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import qtensor as qtensor_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import (
    SequentialQuantizer,
    StaticBlockScaleQuantizer,
    TensorQuantizer,
    register_quant_backend,
    unregister_quant_backend,
)
from modelopt.torch.quantization.qtensor import FP8QTensor, base_qtensor
from modelopt.torch.quantization.utils import (
    disable_weight_cache,
    enable_weight_cache,
    invalidate_weight_cache,
    native_weight,
    weight_cache_scope,
)
from modelopt.torch.quantization.utils.native_weight import (
    dequantize_pinned_native,
    get_pinned_native_weight,
    pin_native_weight,
)

BACKEND_CALLS = []


def _ste_round_backend(inputs, quantizer):
    BACKEND_CALLS.append(quantizer)
    return inputs + (torch.round(inputs * 8) / 8 - inputs).detach()


@pytest.fixture
def ste_round_backend():
    register_quant_backend("ste_round", _ste_round_backend)
    BACKEND_CALLS.clear()
    yield
    unregister_quant_backend("ste_round")


def _quantized_mlp(**weight_cfg):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": 0, **weight_cfg},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    return mtq.quantize(model, cfg, lambda m: m(torch.randn(4, 16)))


def _accumulate(model, micro_batches):
    """Forward/backward every micro-batch, recomputing activations as Megatron does."""
    model.zero_grad()
    outputs = [checkpoint(model, x, use_reentrant=False) for x in micro_batches]
    for out in outputs:
        out.square().sum().backward()
    return [out.detach() for out in outputs], [p.grad.clone() for p in model.parameters()]


# The block size of 12 pads both weights, making a fresh quantizer input on every forward.
@pytest.mark.parametrize(
    "weight_cfg", [{}, {"backend": "ste_round"}, {"axis": None, "block_sizes": {-1: 12}}]
)
def test_cached_step_matches_uncached(ste_round_backend, weight_cfg):
    model = _quantized_mlp(**weight_cfg)
    micro_batches = [torch.randn(4, 16) for _ in range(3)]
    ref_outputs, ref_grads = _accumulate(model, micro_batches)

    assert enable_weight_cache(model) == 2
    BACKEND_CALLS.clear()
    with weight_cache_scope(model) as stats:
        outputs, grads = _accumulate(model, micro_batches)

    for out, ref in zip(outputs + grads, ref_outputs + ref_grads):
        assert torch.equal(out, ref)
    # Each weight is fake-quantized once, then reused by 3 forwards and 3 recomputations.
    assert (stats.misses, stats.hits) == (2, 10)
    if "backend" in weight_cfg:
        assert len(BACKEND_CALLS) == 2
    assert stats.retained_bytes == 0 and stats.peak_bytes > 0


def test_weight_and_quantizer_changes_recompute():
    model = _quantized_mlp()
    enable_weight_cache(model)
    linear, x = model[0], torch.randn(4, 16)
    with weight_cache_scope(model) as stats:
        model(x)
        with torch.no_grad():
            linear.weight.mul_(0.5)  # an in-place update changes the weight's version
        # A copy is not enrolled, so it gives the uncached result.
        assert torch.equal(model(x), copy.deepcopy(model)(x))
        linear.weight_quantizer.amax = linear.weight_quantizer.amax * 0.5
        invalidate_weight_cache(model)
        assert torch.equal(model(x), copy.deepcopy(model)(x))
    assert (stats.misses, stats.invalidations) == (5, 3)


def test_pinned_native_weight_reuses_the_step_cache(monkeypatch):
    model = _quantized_mlp(num_bits=(4, 3))
    linear = model[0]
    native, scales = FP8QTensor.quantize(linear.weight.detach(), axis=0)
    linear.weight_quantizer._set_buffer("_scale", scales)
    pin_native_weight(linear.weight_quantizer, native, linear.weight)
    batches = [torch.randn(4, 16) for _ in range(3)]
    expected_outputs, expected_gradients = _accumulate(model, batches)
    decoded = []
    dequantize = FP8QTensor.dequantize

    def observed(self, *args, **kwargs):
        decoded.append(1)
        return dequantize(self, *args, **kwargs)

    monkeypatch.setattr(FP8QTensor, "dequantize", observed)
    enable_weight_cache(model)
    with weight_cache_scope(model) as stats:
        outputs, gradients = _accumulate(model, batches)
    for actual, expected in zip(outputs + gradients, expected_outputs + expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(decoded) == 1
    assert (stats.misses, stats.hits) == (2, 10)


def test_native_codec_identity_survives_custom_names_and_checkpoint_reload(
    monkeypatch, tmp_path, request
):
    monkeypatch.setattr(
        base_qtensor, "_NATIVE_TENSOR_CLASSES", dict(base_qtensor._NATIVE_TENSOR_CLASSES)
    )
    builtin = qtensor_module.FP8QTensor
    local_codecs = []

    def restore_safe_globals():
        keep = [item for item in torch.serialization.get_safe_globals() if item not in local_codecs]
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(keep)

    request.addfinalizer(restore_safe_globals)

    class FP8QTensor(builtin):
        pass

    local_codecs.append(FP8QTensor)

    def first_provider():
        class PluginTensor(builtin):
            pass

        local_codecs.append(PluginTensor)
        return PluginTensor

    def second_provider():
        class PluginTensor(builtin):
            pass

        local_codecs.append(PluginTensor)
        return PluginTensor

    first, second = first_provider(), second_provider()
    assert base_qtensor.get_native_tensor_class("FP8QTensor") is builtin
    for index, codec in enumerate((builtin, FP8QTensor, first, second)):
        model = _quantized_mlp(num_bits=(4, 3))
        linear = model[0]
        native, scales = codec.quantize(linear.weight.detach(), axis=0)
        linear.weight_quantizer._set_buffer("_scale", scales)
        pin_native_weight(linear.weight_quantizer, native, linear.weight)
        identity = f"{codec.__module__}.{codec.__qualname__}"
        assert linear.weight_quantizer._native_weight_meta["class"] == identity
        if codec is builtin:
            # Existing checkpoints used the reserved built-in short name.
            linear.weight_quantizer._native_weight_meta["class"] = "FP8QTensor"
        batches = [torch.randn(2, 16, requires_grad=True)]
        expected, gradients = _accumulate(model, batches)
        inputs_gradient = batches[0].grad.clone()
        path = tmp_path / f"native-{index}.pt"
        mto.save(model, path)
        restored = mto.restore(nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)), path)
        payload = get_pinned_native_weight(restored[0].weight_quantizer, restored[0].weight)
        assert type(payload) is codec
        assert torch.equal(
            payload._quantized_data.view(torch.uint8), native._quantized_data.view(torch.uint8)
        )
        restored_inputs = [batches[0].detach().clone().requires_grad_(True)]
        actual, actual_gradients = _accumulate(restored, restored_inputs)
        for value, wanted in zip(actual + actual_gradients, expected + gradients):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        torch.testing.assert_close(restored_inputs[0].grad, inputs_gradient, rtol=0, atol=0)
    reloaded = first_provider()
    assert reloaded is not first
    assert (
        base_qtensor.get_native_tensor_class(f"{first.__module__}.{first.__qualname__}") is reloaded
    )
    quantizer, source = restored[0].weight_quantizer, restored[0].weight
    for identity, reason in (
        ("PluginTensor", "ambiguous"),
        ("unimported.provider.Codec", "not registered"),
    ):
        quantizer._native_weight_meta["class"] = identity
        with pytest.raises(ValueError, match=reason):
            get_pinned_native_weight(quantizer, source)


def test_weight_cache_contract(ste_round_backend):
    model = _quantized_mlp()
    with pytest.raises(ValueError, match="enable_weight_cache"), weight_cache_scope(model):
        pass
    for weight_cfg in [{}, {"backend": "ste_round"}]:
        with pytest.raises(ValueError, match="pass_through_bwd"):
            enable_weight_cache(_quantized_mlp(pass_through_bwd=False, **weight_cfg))
    # Frozen LSQ scales have no gradient, but LSQ's backward still clips and rescales.
    for tied_amax in [False, True]:
        lsq = _quantized_mlp()
        quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=4, block_sizes={-1: 16}))
        quantizer.amax = torch.ones(32)
        lsq[0].weight_quantizer = StaticBlockScaleQuantizer.from_tensor_quantizer(quantizer)
        lsq[0].weight_quantizer.enable_lsq(False, learnable_amax=[], tied_amax=tied_amax)
        with pytest.raises(ValueError, match="LSQ"):
            enable_weight_cache(lsq)
    learned = _quantized_mlp()
    learned[0].weight_quantizer.register_parameter("scale", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="learned scales"):
        enable_weight_cache(learned)
    composed = _quantized_mlp()
    composed[0].weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
    with pytest.raises(ValueError, match="only TensorQuantizer"):
        enable_weight_cache(composed)

    enable_weight_cache(model)
    with pytest.raises(RuntimeError, match="boom"), weight_cache_scope(model):
        with pytest.raises(RuntimeError, match="inside weight_cache_scope"):
            enable_weight_cache(model)
        model(torch.randn(4, 16))
        with pytest.raises(RuntimeError, match="nested"), weight_cache_scope(model):
            pass
        raise RuntimeError("boom")
    # A failed step closes its scope, so the next one starts empty.
    with weight_cache_scope(model) as stats:
        model(torch.randn(4, 16))
    assert stats.misses == 2 and stats.hits == 0


def test_enrollment_by_chunk():
    chunks = [_quantized_mlp(), _quantized_mlp()]
    with pytest.raises(ValueError, match="more than one"):
        enable_weight_cache([chunks[0], chunks[0][0]])
    # Like Megatron virtual pipeline chunks: each chunk owns its quantizers and can be scoped alone.
    assert enable_weight_cache(chunks, select=lambda name, q: name.startswith("0.")) == 2
    with weight_cache_scope(chunks[1]) as stats:
        chunks[1](torch.randn(4, 16))
    assert stats.misses == 1
    with pytest.raises(ValueError, match="already enrolled"):
        enable_weight_cache(nn.Sequential(chunks[1]))

    disable_weight_cache(chunks[1])
    with pytest.raises(ValueError, match="enable_weight_cache"), weight_cache_scope(chunks[1]):
        pass
    with weight_cache_scope(chunks[0]) as stats:
        chunks[0](torch.randn(4, 16))
    assert stats.misses == 1


@pytest.mark.parametrize("transform", ["pre_scale", "bias"])
def test_pinned_native_weights_honor_input_transforms(transform):
    model = _quantized_mlp(num_bits=(4, 3))
    quantizer, source = model[0].weight_quantizer, model[0].weight
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    if transform == "pre_scale":
        quantizer.pre_quant_scale = torch.full((16,), 2.0)
    else:
        quantizer.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=(4, 3), axis=0, bias={-1: None, "type": "static"})
        )
        quantizer.bias_value = torch.tensor(0.125)
    reference = copy.deepcopy(quantizer)
    del reference._native_weight_meta
    torch.testing.assert_close(quantizer(source), reference(source), rtol=0, atol=0)
    with pytest.raises(ValueError):
        pin_native_weight(quantizer, native, source)


@pytest.mark.parametrize("policy", ["clipped", "lsq", "learned"])
def test_pinned_native_weights_preserve_backward_policy(policy):
    model = _quantized_mlp(num_bits=(4, 3))
    quantizer = model[0].weight_quantizer
    source = model[0].weight.detach().clone().requires_grad_()
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    if policy == "clipped":
        quantizer._pass_through_bwd = False
    elif policy == "lsq":
        quantizer._lsq = True
    else:
        quantizer.register_parameter("learned_scale", nn.Parameter(torch.ones(())))
    reference = copy.deepcopy(quantizer)
    del reference._native_weight_meta
    expected = source.detach().clone().requires_grad_()
    actual_output, expected_output = quantizer(source), reference(expected)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    actual_output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(source.grad, expected.grad, rtol=0, atol=0)


def test_pinned_native_scales_are_live_codec_fields():

    model = _quantized_mlp(num_bits=(4, 3))
    q, source = model[0].weight_quantizer, model[0].weight
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    q._set_buffer("_scale", scale)
    pin_native_weight(q, native, source)
    original = q(source).detach().clone()
    q._scale.mul_(0.5)
    q._amax.mul_(2)
    pinned = get_pinned_native_weight(q, source)
    assert pinned is not None
    expected = dequantize_pinned_native(q, pinned, source.dtype)
    torch.testing.assert_close(q(source), expected, rtol=0, atol=0)
    torch.testing.assert_close(q(source), original * 0.5, rtol=0, atol=0)
    # Update the logical master and re-pin after changing native scales.
    updated = expected.detach().clone().requires_grad_()
    pin_native_weight(q, pinned, updated)
    torch.testing.assert_close(q(updated), updated, rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["uncalibrated", "constant", "dynamic"])
@pytest.mark.parametrize("cached", [False, True])
def test_native_pin_miss_preserves_threshold_policy(policy, cached):
    model = _quantized_mlp(num_bits=(4, 3))
    linear = model[0]
    quantizer, source = linear.weight_quantizer, linear.weight
    del quantizer._amax
    if policy == "constant":
        quantizer.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=(4, 3), axis=0, use_constant_amax=True)
        )
    elif policy == "dynamic":
        quantizer.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=(4, 3), axis=0, type="dynamic")
        )
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    reference = copy.deepcopy(quantizer)
    del reference._native_weight_meta
    with torch.no_grad():
        source.add_(0.125)
    assert get_pinned_native_weight(quantizer, source) is None
    if cached:
        enable_weight_cache(linear)
    with weight_cache_scope(linear) if cached else nullcontext():
        if policy == "uncalibrated":
            with pytest.raises(ValueError, match="calibrate thresholds"):
                quantizer(source)
        else:
            torch.testing.assert_close(quantizer(source), reference(source), rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["fingerprint", "install"])
def test_native_pin_replacement_failure_preserves_previous_payload(monkeypatch, phase):
    model = _quantized_mlp(num_bits=(4, 3))
    quantizer, source = model[0].weight_quantizer, model[0].weight
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    other = source.detach().roll(1, -1)
    replacement, _ = FP8QTensor.quantize(other, axis=0)
    assert not torch.equal(replacement._quantized_data, native._quantized_data)
    original = native_weight._source_keys if phase == "fingerprint" else quantizer._set_buffer
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == (1 if phase == "fingerprint" else 2):
            raise RuntimeError("interrupted pin replacement")
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(
            native_weight if phase == "fingerprint" else quantizer,
            "_source_keys" if phase == "fingerprint" else "_set_buffer",
            fail,
        )
        with pytest.raises(RuntimeError, match="interrupted pin"):
            pin_native_weight(quantizer, replacement, other)
    retained = get_pinned_native_weight(quantizer, source)
    assert retained is not None
    assert torch.equal(retained._quantized_data, native._quantized_data)


@pytest.mark.parametrize("assign", [False, True])
def test_native_pin_warm_validation_detects_loaded_fingerprints(assign):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=0))
    source = torch.randn(4, 16)
    native, scale = FP8QTensor.quantize(source, axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    assert get_pinned_native_weight(quantizer, source) is not None
    replacement = copy.deepcopy(quantizer)
    other, other_scale = FP8QTensor.quantize(source * 2, axis=0)
    replacement._set_buffer("_scale", other_scale)
    pin_native_weight(replacement, other, source * 2)
    quantizer.load_state_dict(replacement.state_dict(), assign=assign)
    assert get_pinned_native_weight(quantizer, source) is None


def test_uncached_native_pin_keeps_higher_order_straight_through_gradients():
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=0))
    source = torch.randn(4, 16, requires_grad=True)
    native, scale = FP8QTensor.quantize(source.detach(), axis=0)
    quantizer._set_buffer("_scale", scale)
    pin_native_weight(quantizer, native, source)
    (gradient,) = torch.autograd.grad(quantizer(source).square().sum(), source, create_graph=True)
    (second,) = torch.autograd.grad(gradient.sum(), source)
    torch.testing.assert_close(second, torch.full_like(source, 2), rtol=0, atol=0)
