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


"""Native replay precision, saved tensors and observable quantizer behavior."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.repair_formats as repair_formats
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.repair import IQRepairPolicy
from modelopt.torch.quantization.repair_formats import IQScaleState


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
def test_materialized_iq_replay_saves_int8_levels_and_matches_native_vjp(dtype, name):
    torch.manual_seed(104)
    model = nn.Linear(512, 2, bias=False).to(dtype)
    mtq.quantize(model, _config([_iq("weight_quantizer", name)]))
    quantizer = model.weight_quantizer
    processed_weight = (
        quantizer._process_for_blockquant(model.weight)
        if quantizer.is_static_block_quant
        else model.weight
    )
    state = IQScaleState(
        quantizer, processed_weight, model.weight.shape, field_prefix="weight_quantizer"
    )
    if dtype == torch.float16:
        state.global_base.fill_(2**-24)
    reference = copy.deepcopy(state)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append((value.numel(), value.dtype)) or value, lambda value: value
    ):
        # TensorQuantizer can expose an equivalent block view of the logical weight.
        actual = state(model._parameters["weight"].reshape(4, 256))
    fmt = IQ_FORMAT_REGISTRY[name]
    levels, _ = fmt.decode_int8(reference.packed, torch.tensor(reference.weight_shape))
    steps = reference.projected_steps()
    expected = (
        (levels.reshape(-1, steps.shape[-1], fmt.int8_group_size).float() * steps.unsqueeze(-1))
        .reshape(actual.shape)
        .to(dtype)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tuple(state.parameters()), gradient)
    expected_gradients = torch.autograd.grad(expected, tuple(reference.parameters()), gradient)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert (model.weight.numel(), torch.int8) in saved
    assert (model.weight.numel(), torch.float32) not in saved

    # An initial update favoring a smaller scale must not prevent a later increase.
    with torch.no_grad():
        state.global_base.fill_(2**-5)
        state.global_base[0] = 0
    direction = levels.float().reshape(-1, fmt.block_size)
    direction = direction / direction.square().sum(-1, keepdim=True)
    direction = (direction * state.global_sign[:, None]).reshape(actual.shape)
    weight = model._parameters["weight"]
    before = state(weight).detach().reshape(-1, fmt.block_size)
    optimizer = torch.optim.SGD([state.global_delta], lr=1024)
    for sign in (1, -1):
        state.zero_grad(set_to_none=True)
        (sign * (state(weight).float().reshape_as(direction) * direction).sum()).backward()
        optimizer.step()
        state.project_()
        updated = state(weight).detach().reshape(-1, fmt.block_size)
        if sign == 1:
            assert updated[0].count_nonzero() == 0
            assert updated[1:].norm() < before[1:].norm()
    assert torch.isfinite(updated).all()
    assert updated[0].count_nonzero() > 0

    with torch.no_grad():
        state.global_delta.fill_(1000)
        state.local_delta.fill_(1000)
    global_scale, local_scale = state.scales()
    assert torch.isfinite(global_scale).all()
    assert torch.isfinite(local_scale).all()
    native = state.native_payload()
    if dtype == torch.float16:
        decoded = fmt.dequantize(native, torch.tensor(state.weight_shape), dtype=dtype)
        assert torch.isinf(decoded).any()
        quantizer = model.weight_quantizer
        raw_before = weight.detach().clone()
        flags = quantizer._enable_pre_quant_scale, copy.deepcopy(quantizer._rotate)
        pinned_before = pinned_packed_weight(quantizer, name)
        with pytest.raises(ValueError):
            state.finalize_(quantizer, weight, native=native)
        torch.testing.assert_close(weight, raw_before, rtol=0, atol=0)
        assert (quantizer._enable_pre_quant_scale, quantizer._rotate) == flags
        assert pinned_packed_weight(quantizer, name) is pinned_before
    with torch.no_grad():
        state.global_delta[0] = float("nan")
    with pytest.raises(ValueError):
        state.native_payload()

    for width in (512, 384):
        model = nn.Linear(width, 2, bias=False).to(dtype)
        entry = _iq("weight_quantizer", name)
        entry["cfg"]["block_sizes"] = {-1: 256}
        mtq.quantize(model, _config([entry]))
        quantizer = model.weight_quantizer
        quantizer.pre_quant_scale = torch.full((width,), 1.25, dtype=dtype)
        weight = model._parameters["weight"]
        before = quantizer(weight).detach()
        processed_weight = quantizer._process_for_blockquant(weight * quantizer.pre_quant_scale)
        finite_state = IQScaleState(
            quantizer, processed_weight, weight.shape, field_prefix="weight_quantizer"
        )
        with torch.no_grad():
            finite_state.global_delta.fill_(0.25)
            finite_state.local_delta.fill_(1)
        native = finite_state.native_payload()
        assert not torch.equal(native, finite_state.packed)
        decoded = fmt.dequantize(native, torch.tensor((2, 512)), dtype=dtype)
        if width < 512:
            assert decoded[:, width:].count_nonzero()
        expected = decoded[:, :width]
        assert not torch.equal(expected, before)
        assert not torch.equal(expected, weight)

        # The caller supplies processed coordinates; baking must clear one-time transforms.
        quantizer.set_from_attribute_config({"rotate": {"enable": True}})
        assert quantizer.pre_quant_scale is not None and quantizer.rotate_is_enabled
        pin_packed_weight(quantizer, name, finite_state.packed, source=processed_weight)
        finite_state.finalize_(quantizer, weight, native=native)
        torch.testing.assert_close(weight, expected, rtol=0, atol=0)
        pinned = pinned_packed_weight(quantizer, name)
        assert pinned is not None and torch.equal(pinned, native)
        assert not quantizer._enable_pre_quant_scale
        assert not quantizer.rotate_is_enabled
        source = torch.nn.functional.pad(weight, (0, 512 - width)).reshape(-1, 256)
        assert torch.equal(fmt.pack(source, quantizer).reshape_as(native), native)
        torch.testing.assert_close(quantizer(weight), expected, rtol=0, atol=0)


@pytest.mark.parametrize("behavior", ["disabled", "hooked"])
def test_packed_replay_gate_preserves_weight_quantizer_behavior(monkeypatch, behavior):

    model = nn.Linear(256, 2)
    mtq.quantize(model, _config([_iq("weight_quantizer", "iq1_s")]))
    policy = IQRepairPolicy()
    handle = None
    try:
        policy.prepare(model)
        quantizer = model.weight_quantizer
        if behavior == "disabled":
            quantizer.disable()
        else:
            handle = quantizer.register_forward_hook(lambda _module, _args, output: output * 2)

        # Qualify CPU inputs solely to reach the unavailable CUDA optimization's gate.
        monkeypatch.setattr(repair_formats, "_packed_projection_qualified", lambda *args: True)

        def unavailable(*args, **kwargs):
            raise AssertionError("Packed projection bypassed observable quantizer behavior")

        monkeypatch.setattr(repair_formats.IQScaleState, "linear", unavailable)
        inputs = torch.randn(3, 256, requires_grad=True)
        expected = F.linear(inputs, quantizer(model._parameters["weight"]), model.bias)
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), inputs)[0],
            torch.autograd.grad(expected.sum(), inputs)[0],
            rtol=0,
            atol=0,
        )
    finally:
        if handle is not None:
            handle.remove()
        policy.close()


@pytest.mark.parametrize(
    "unsupported", ["offset-shape", "offset-dtype", "offset-stride", "bias-shape", "bias-dtype"]
)
def test_grouped_replay_refuses_unsupported_layout_before_cuda_dispatch(monkeypatch, unsupported):
    model = nn.ModuleList([nn.Linear(256, 2, bias=False) for _ in range(4)])
    mtq.quantize(model, _config([_iq("*weight_quantizer", "iq1_s")]))
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        quantizers = [module.weight_quantizer for module in model]
        inputs = torch.randn(3, 256)
        offsets = torch.tensor([1, 2, 3, 3], dtype=torch.int32)
        bias = torch.zeros(4, 2)
        if unsupported == "offset-shape":
            offsets = offsets[:-1]
        elif unsupported == "offset-dtype":
            offsets = offsets.float()
        elif unsupported == "offset-stride":
            offsets = torch.stack([offsets, offsets], dim=-1)[:, 0]
        elif unsupported == "bias-shape":
            bias = bias[:-1]
        else:
            bias = bias.double()
        # Exercise metadata refusal on CPU; no optimized CUDA implementation is replaced.
        monkeypatch.setattr(repair_formats, "_packed_projection_qualified", lambda *args: True)
        assert repair_formats.repair_grouped_linear(quantizers, inputs, offsets, bias) is None
    finally:
        policy.close()
