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

"""Native IQ field gradients, frozen downstream projections, and ordinary persistence."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight, pinned_packed_weight
from modelopt.torch.quantization.ggml.scale_fields import decode_scale_fields, replace_scale_fields


def _config(entries):
    return {"quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries], "algorithm": "max"}


def _iq(path, name):
    return {"quantizer_name": path, "cfg": {"num_bits": name, "backend": "ggml"}}


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.timeout(180)
def test_packed_iq_linear_matches_native_vjp(name, input_grad):
    pytest.importorskip("triton")
    from modelopt.torch.kernels.quantization.iq_gemm import iq_linear
    from modelopt.torch.kernels.quantization.repair import apply_block_scales, decode_iq_levels
    from modelopt.torch.quantization.repair import IQRepairPolicy

    if not torch.cuda.is_available():
        pytest.skip("Packed IQ projection math requires CUDA")
    torch.manual_seed(103)
    shape = (13, 512)
    model = nn.Linear(shape[1], shape[0], bias=False).cuda().bfloat16()
    mtq.quantize(
        model,
        _config([_iq("*weight_quantizer", name)]),
        lambda m: m(torch.randn(1, shape[1], device="cuda", dtype=torch.bfloat16)),
    )
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        state = model.weight_quantizer._local_repair_state
        fmt = IQ_FORMAT_REGISTRY[name]
        native = fmt.dequantize(state.native_payload(), torch.tensor(shape), dtype=torch.bfloat16)
        levels, _ = fmt.decode_int8(state.packed, torch.tensor(shape))
        fused_levels = decode_iq_levels(state.packed, name, shape)
        torch.testing.assert_close(fused_levels, levels.reshape(shape), rtol=0, atol=0)
        for tokens in (7, 0):
            x = (
                torch.randn(tokens, shape[1] * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
                .detach()
                .requires_grad_(input_grad)
            )
            bias = torch.randn(shape[0], device="cuda", dtype=x.dtype, requires_grad=True)
            steps = state.projected_steps().detach().requires_grad_(True)
            saved = []

            def observe(value):
                saved.append((value.shape, value.dtype))
                return value

            with torch.autograd.graph.saved_tensors_hooks(observe, lambda value: value):
                actual = iq_linear(x, state.packed, steps, name, shape, bias)
            reference_weight = native.detach().requires_grad_(True)
            reference_x = x.detach().requires_grad_(input_grad)
            reference_bias = bias.detach().requires_grad_(True)
            expected = F.linear(reference_x, reference_weight, reference_bias)
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015)
            upstream = torch.randn_like(expected)
            actual_grad = torch.autograd.grad(
                actual, [steps, bias, *([x] if input_grad else [])], upstream
            )
            reference_grad = torch.autograd.grad(
                expected,
                [reference_weight, reference_bias, *([reference_x] if input_grad else [])],
                upstream,
            )
            group = fmt.int8_group_size
            groups = (-1, steps.shape[-1], group)
            expected_scale = (
                (reference_grad[0].reshape(groups).float() * levels.reshape(groups).float())
                .sum(-1, keepdim=True)
                .reshape_as(steps)
            )
            torch.testing.assert_close(actual_grad[0], expected_scale, rtol=5e-4, atol=2e-3)
            fused_steps = steps.detach().requires_grad_(True)
            fused_weight = apply_block_scales(fused_levels, fused_steps, x.dtype)
            torch.testing.assert_close(fused_weight, native, rtol=0, atol=0)
            fused_gradient = torch.autograd.grad(fused_weight, fused_steps, reference_grad[0])[0]
            torch.testing.assert_close(fused_gradient, expected_scale, rtol=0, atol=0)
            torch.testing.assert_close(actual_grad[1], reference_grad[1], rtol=0, atol=0)
            if input_grad:
                torch.testing.assert_close(actual_grad[2], reference_grad[2], rtol=0.01, atol=0.015)
            assert (torch.Size(shape), x.dtype) not in saved
    finally:
        policy.close()


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
def test_iq_fields_finalize_and_reload(name, tmp_path):
    from modelopt.torch.quantization.repair import IQRepairPolicy

    torch.manual_seed(17)
    model = nn.Linear(256, 4, bias=False)
    x = torch.randn(3, 256)
    mtq.quantize(model, _config([_iq("weight_quantizer", name)]), lambda m: m(x))
    fmt = IQ_FORMAT_REGISTRY[name]
    if name == "iq1_s":
        # Native FP16 scales include negative zero, whose sign must survive a
        # zero-update projection even though floating-point equality cannot see it.
        native = fmt.pack(model.weight, model.weight_quantizer)
        global_scale, local, _ = decode_scale_fields(native, fmt)
        global_scale[0] = -0.0
        native = replace_scale_fields(native, fmt, global_scale, local)
        with torch.no_grad():
            model.weight.copy_(
                fmt.dequantize(native, torch.tensor(model.weight.shape), dtype=model.weight.dtype)
            )
        pin_packed_weight(model.weight_quantizer, name, native, source=model.weight)
    before = model(x).detach()
    original = fmt.pack(model.weight, model.weight_quantizer).clone()
    if name == "iq1_s":
        assert torch.signbit(decode_scale_fields(original, fmt)[0][0])
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        assert torch.equal(model.weight_quantizer._local_repair_state.native_payload(), original)
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        optimizer = torch.optim.SGD(fields, lr=0.01)
        model(x).square().mean().backward()
        assert model.weight.grad is None
        assert any(p.grad is not None and p.grad.count_nonzero() for p in fields)
        optimizer.step()
        policy.project_()
        changed = model(x).detach()
        assert not torch.equal(changed, before)
        policy.finalize_()
    finally:
        policy.close()
    payload = pinned_packed_weight(model.weight_quantizer, name)
    global_scale, local, _ = decode_scale_fields(original, fmt)
    assert torch.equal(replace_scale_fields(payload, fmt, global_scale, local), original)
    torch.testing.assert_close(model(x), changed, rtol=0, atol=0)
    mto.save(model, tmp_path / "student.pt")
    restored = mto.restore(nn.Linear(256, 4, bias=False), tmp_path / "student.pt")
    torch.testing.assert_close(restored(x), changed, rtol=0, atol=0)
    assert torch.equal(pinned_packed_weight(restored.weight_quantizer, name), payload)
    assert all("repair" not in name for name, _ in restored.named_modules())


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 256, bias=False)
        self.up = nn.Linear(256, 256, bias=False)
        self.down = nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _MixedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(256, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared = _Expert()

    def forward(self, x):
        scores = self.router(x).softmax(-1)
        return self.shared(x) + sum(
            expert(x) * scores[:, i : i + 1] for i, expert in enumerate(self.experts)
        )


@pytest.mark.parametrize("frozen_format", ["fp8", "nvfp4", "nvfp4_compressed"])
def test_iq_scales_backpropagate_through_frozen_projection(frozen_format):
    from modelopt.torch.quantization.repair import IQRepairPolicy

    if frozen_format.startswith("nvfp4") and not torch.cuda.is_available():
        pytest.skip("Real NVFP4 forward/backward requires a CUDA allocation")
    device = "cuda" if frozen_format.startswith("nvfp4") else "cpu"
    torch.manual_seed(23)
    model = _MixedMoE().to(device)
    x = torch.randn(2, 256, device=device)
    frozen_cfg = (
        {"num_bits": (4, 3), "axis": None}
        if frozen_format == "fp8"
        else {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    )
    mtq.quantize(
        model,
        _config(
            [
                _iq("shared.*weight_quantizer", "iq2_s"),
                _iq("router.weight_quantizer", "iq2_s"),
                _iq("experts.*.gate.weight_quantizer", "iq1_m"),
                _iq("experts.*.up.weight_quantizer", "iq1_m"),
                _iq("experts.1.up.weight_quantizer", "iq2_xxs"),
                {"quantizer_name": "experts.*.down.weight_quantizer", "cfg": frozen_cfg},
            ]
        ),
        lambda m: m(x),
    )
    if frozen_format == "nvfp4_compressed":
        mtq.compress(
            model, mtq.CompressConfig(compress={"default": False, "experts.*.down*": True})
        )
    frozen = {
        n: v.detach().clone()
        for n, v in model.state_dict().items()
        if "router" in n or ("experts" in n and ".down." in n)
    }
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        expected = model.shared(x)
        scores = model.router(x).detach().softmax(-1)
        for i, expert in enumerate(model.experts):
            # The oracle uses the actual quantized frozen down weight and ordinary differentiable linear.
            weight = expert.down.weight_quantizer(expert.down.weight).detach()
            hidden = F.silu(expert.gate(x)) * expert.up(x)
            expected = expected + F.linear(hidden, weight) * scores[:, i : i + 1]
        expected.square().sum().backward()
        gradients = [p.grad.detach().clone() for p in fields]
        for p in fields:
            p.grad = None
        model(x).square().sum().backward()
        for parameter, reference in zip(fields, gradients):
            torch.testing.assert_close(parameter.grad, reference, rtol=2e-5, atol=2e-7)
        assert any(p.grad.count_nonzero() for p in fields)
        assert all(p.grad is None for expert in model.experts for p in expert.down.parameters())
        optimizer = torch.optim.Adam(fields, lr=0.001)
        optimizer.step()
        assert set(optimizer.state) == set(fields)
        policy.finalize_()
    finally:
        policy.close()
    after = model.state_dict()
    for name, value in frozen.items():
        torch.testing.assert_close(after[name], value, rtol=0, atol=0)


def test_iq_policy_refuses_aliases_and_restores_on_failure():
    from modelopt.torch.quantization.repair import IQRepairPolicy, plan_iq_repair

    model = nn.Sequential(nn.Linear(256, 2, bias=False))
    x = torch.randn(1, 256)
    mtq.quantize(model, _config([_iq("*weight_quantizer", "iq1_m")]), lambda m: m(x))
    alias = nn.Sequential(model[0], model[0])
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        plan_iq_repair([alias])
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    other = copy.deepcopy(model)
    other[0].weight = nn.Parameter(model[0].weight.detach())
    with pytest.raises(ValueError):
        plan_iq_repair([model[0], other[0]])
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        raise RuntimeError("replay failed")
    except RuntimeError:
        policy.close()
    assert model[0].weight.requires_grad
    torch.testing.assert_close(model(x), F.linear(x, model[0].weight_quantizer(model[0].weight)))
    assert all("repair" not in n for n, _ in model.named_modules())


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.timeout(180)
def test_grouped_iq_projection_matches_uneven_native_experts(name):
    pytest.importorskip("triton")
    from modelopt.torch.kernels.quantization.iq_grouped_gemm import iq_grouped_linear
    from modelopt.torch.quantization.repair import IQRepairPolicy

    if not torch.cuda.is_available():
        pytest.skip("Native grouped IQ projections require CUDA")
    torch.manual_seed(107)
    model = nn.ModuleList([nn.Linear(512, 13, bias=False).cuda().bfloat16() for _ in range(3)])
    mtq.quantize(model, _config([_iq("*weight_quantizer", name)]), None)
    policy = IQRepairPolicy()
    try:
        policy.prepare(model)
        states = [m.weight_quantizer._local_repair_state for m in model]
        packeds = [s.packed for s in states]
        pointers = torch.tensor([p.data_ptr() for p in packeds], device="cuda", dtype=torch.int64)
        for counts in ((5, 0, 3), (0, 0, 0)):
            ends = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0)
            # An unowned EP tail must stay zero in both autograd directions.
            x = torch.randn(
                sum(counts) + 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            steps = torch.stack([s.projected_steps() for s in states]).detach().requires_grad_(True)
            bias = torch.randn(3, 13, device="cuda", dtype=x.dtype, requires_grad=True)
            actual = iq_grouped_linear(
                x, packeds, steps, ends, name, (13, 512), bias, packed_pointers=pointers
            )
            reference_x = x.detach().requires_grad_(True)
            reference_steps = steps.detach().requires_grad_(True)
            reference_bias = bias.detach().requires_grad_(True)
            pieces, start = [], 0
            for index, count in enumerate(counts):
                levels, _ = states[index].format.decode_int8(
                    packeds[index], torch.tensor((13, 512))
                )
                weight = (
                    (
                        levels.reshape(
                            -1, steps.shape[-1], states[index].format.int8_group_size
                        ).float()
                        * reference_steps[index].unsqueeze(-1)
                    )
                    .reshape(13, 512)
                    .to(x.dtype)
                )
                pieces.append(
                    F.linear(reference_x[start : start + count], weight, reference_bias[index])
                )
                start += count
            expected = torch.cat([*pieces, reference_x[start:, :13] * 0])
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015)
            upstream = torch.randn_like(actual)
            observed = torch.autograd.grad(actual, [x, steps, bias], upstream)
            oracle = torch.autograd.grad(
                expected, [reference_x, reference_steps, reference_bias], upstream
            )
            for a, b in zip(observed, oracle, strict=True):
                torch.testing.assert_close(a, b, rtol=0.01, atol=0.015)
    finally:
        policy.close()


@pytest.mark.parametrize("name", ["iq1_m", "iq2_xxs"])
@pytest.mark.timeout(180)
def test_automatic_packed_projection_matches_native_fields(name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Automatic packed qualification is specific to GB300")
    from modelopt.torch.quantization.repair import IQRepairPolicy

    torch.manual_seed(127)
    model = nn.Linear(4096, 2048, bias=True).cuda().bfloat16()
    mtq.quantize(model, _config([_iq("*weight_quantizer", name)]), None)
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(model)
        state = model.weight_quantizer._local_repair_state
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        actual = model(x)
        assert state.projection_counts == {"packed_dense": 1}
        expected = F.linear(x, state(model.weight), model.bias)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015)
        upstream = torch.randn_like(expected) / expected.numel()
        observed = torch.autograd.grad(actual, [x, *fields], upstream)
        oracle = torch.autograd.grad(expected, [x, *fields], upstream)
        for a, b in zip(observed, oracle, strict=True):
            error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            assert error < 5e-4, error.item()
        # The long-context path must keep the faster materialized GEMM.
        with torch.no_grad():
            model(torch.zeros(32768, 4096, device="cuda", dtype=torch.bfloat16))
        assert state.projection_counts == {"packed_dense": 1, "materialized": 2}
        assert not any(
            t.shape == model.weight.shape and t.is_floating_point() for t in state.buffers()
        )
    finally:
        policy.close()


@pytest.mark.parametrize("name", ["iq1_m", "iq2_xxs"])
@pytest.mark.timeout(180)
def test_automatic_grouped_projection_preserves_native_fields(name):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Automatic packed qualification is specific to GB300")
    from transformers.integrations.moe import _grouped_linear

    from modelopt.torch.quantization.repair import IQRepairPolicy
    from modelopt.torch.quantization.repair_formats import repair_grouped_linear

    torch.manual_seed(131)
    bank = nn.ModuleList([nn.Linear(4096, 2048, bias=False).cuda().bfloat16() for _ in range(4)])
    mtq.quantize(bank, _config([_iq("*weight_quantizer", name)]), None)
    policy = IQRepairPolicy()
    try:
        fields = policy.prepare(bank)
        quantizers = [m.weight_quantizer for m in bank]
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ends = torch.tensor([5, 5, 8, 8], device="cuda", dtype=torch.int32)
        actual = repair_grouped_linear(quantizers, x, ends)
        assert actual is not None
        expected = _grouped_linear(
            x, torch.stack([q(m.weight) for q, m in zip(quantizers, bank, strict=True)]), ends
        )
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015)
        upstream = torch.randn_like(expected) / expected.numel()
        observed = torch.autograd.grad(actual, [x, *fields], upstream)
        oracle = torch.autograd.grad(expected, [x, *fields], upstream)
        for a, b in zip(observed, oracle, strict=True):
            error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20)
            assert error < 5e-4, error.item()
        first = quantizers[0]._local_repair_state
        assert first.projection_counts["packed_grouped"] == 1
        pointers = first._grouped_pointers[1]
        with torch.no_grad():
            repair_grouped_linear(quantizers, x, ends)
        assert first._grouped_pointers[1] is pointers
        assert pointers.numel() == 4 and not pointers.requires_grad
    finally:
        policy.close()
