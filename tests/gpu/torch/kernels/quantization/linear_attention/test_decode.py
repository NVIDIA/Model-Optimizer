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

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.linear_attention.decode import _fp8_qdq, fused_recurrence
from modelopt.torch.quantization.linear_attention.config import LinearAttentionDecodeConfig
from modelopt.torch.quantization.linear_attention.decode import (
    _encode,
    recurrent_decode,
    recurrent_decode_reference,
)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("state_format", [None, "fp8_e4m3", "int8"])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("read_stored", [False, True])
def test_fused_recurrence_outputs_state_and_gradients(kda, state_format, replay, read_stored):
    state_qdq = state_format is not None
    state_format = state_format or "fp8_e4m3"
    torch.manual_seed(193)
    shape = (37, 2, 16)
    q, k = [
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).requires_grad_() for _ in range(2)
    ]
    v = torch.randn(37, 2, 11, device="cuda", requires_grad=True)
    g = (-torch.rand(shape if kda else shape[:-1], device="cuda") * 0.03).requires_grad_()
    beta = (torch.rand(shape[:-1], device="cuda") * 0.4).requires_grad_()
    initial = (torch.randn(2, 16, 11, device="cuda") * 0.1).requires_grad_()
    cfg = LinearAttentionDecodeConfig(
        mode="replay" if replay else "token",
        quantize_initial=False,
        readout="stored" if read_stored else "working",
        replay={"window": 5} if replay else None,
    )
    encoded_key = _encode(k, replay, 16).values
    actual = fused_recurrence(
        q,
        encoded_key,
        v,
        g,
        beta,
        initial,
        state_qdq=state_qdq,
        state_format=state_format,
        block_v=16,
        replay=replay,
        factor_qdq=replay,
        window=5,
        read_stored=read_stored,
    )
    expected, carry = recurrent_decode_reference(
        q,
        k,
        v,
        g,
        beta,
        config=cfg,
        initial_state=initial,
        state_qdq=state_qdq,
        state_format=state_format,
        block_v=16,
    )
    final = carry.reconstruct()
    targets = (q, k, v, g, beta, initial)
    probes = [torch.randn_like(x) for x in actual[:2]]
    gradients = [
        torch.autograd.grad(sum((a * p).sum() for a, p in zip(result, probes)), targets)
        for result in (actual[:2], (expected, final))
    ]
    for a, e in zip((*actual[:2], *gradients[0]), (expected, final, *gradients[1])):
        torch.testing.assert_close(a, e, rtol=1e-3, atol=8e-6)


@pytest.mark.parametrize("mode", ["token", "replay"])
@pytest.mark.parametrize("interval", [3, 8])
@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_fused_carry_handoff_and_auxiliary_gradients(mode, interval, state_format):
    torch.manual_seed(217)
    q, k = [
        F.normalize(torch.randn(23, 2, 16, device="cuda"), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(23, 2, 19, device="cuda", requires_grad=True)
    g = (-torch.rand_like(q) * 0.05).requires_grad_()
    beta = (torch.rand(23, 2, device="cuda") * 0.4).requires_grad_()
    initial = (torch.randn(2, 16, 19, device="cuda") * 0.2).requires_grad_()
    args = (q, k, v, g, beta)
    reference = LinearAttentionDecodeConfig(
        mode=mode, decay_log_step=0.01, replay={"window": 5} if mode == "replay" else None
    )
    candidate = reference.model_copy(update={"implementation": "triton"})
    expected, expected_carry = recurrent_decode_reference(
        *args,
        config=reference,
        state_qdq=True,
        state_format=state_format,
        block_v=16,
        initial_state=initial,
    )
    carry = None
    outputs = []
    start = 0
    for end in [0, 3, 11, 12, 23]:
        output, carry = recurrent_decode(
            *(x[start:end] for x in args),
            config=candidate,
            state_qdq=True,
            state_format=state_format,
            block_v=16,
            initial_state=initial if carry is None else None,
            carry=carry,
            checkpoint_interval=interval,
        )
        outputs.append(output)
        start = end
    actual = torch.cat(outputs)
    actual_state, expected_state = carry.reconstruct(), expected_carry.reconstruct()
    assert carry.cursor == expected_carry.cursor
    assert carry.position == 23 and carry.started
    torch.testing.assert_close(
        carry.anchor.scales, expected_carry.anchor.scales, rtol=5e-5, atol=2e-8
    )
    for a, e in zip(carry.entries, expected_carry.entries):
        torch.testing.assert_close(a.update.scales, e.update.scales, rtol=5e-5, atol=2e-8)
        torch.testing.assert_close(a.key.values, e.key.values, rtol=0, atol=0)
    probes = [torch.randn_like(actual), torch.randn_like(actual_state)]
    gradients = [
        torch.autograd.grad(sum((a * p).sum() for a, p in zip(result, probes)), (*args, initial))
        for result in ((actual, actual_state), (expected, expected_state))
    ]
    for a, e in zip(
        (actual, actual_state, *gradients[0]), (expected, expected_state, *gradients[1])
    ):
        torch.testing.assert_close(a, e, rtol=1e-3, atol=1e-5)


@triton.jit
def _codec_kernel(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):  # noqa: N803
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0)
    decoded, scale = _fp8_qdq(values)
    tl.store(Y + offsets, decoded, offsets < N)
    tl.store(S, scale)


@pytest.mark.parametrize("scale", [0.0, 2.0**-12, 1.0, 2.0**12, 0.00314159, 731.173])
def test_software_codec_matches_float8_boundaries(scale):
    # Exhaust all finite positive E4M3 values and their rounding ties, plus signs.
    representable = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    ties = (representable[:-1] + representable[1:]) / 2
    probe = torch.cat(
        (
            representable,
            ties,
            torch.nextafter(ties, torch.full_like(ties, float("inf"))),
            torch.nextafter(ties, torch.zeros_like(ties)),
        )
    )
    probe = torch.cat((probe, -probe)) * scale
    values = probe.cuda()
    actual = torch.empty_like(values)
    actual_scale = torch.empty((), device="cuda")
    _codec_kernel[(1,)](
        values, actual, actual_scale, len(values), triton.next_power_of_2(len(values))
    )
    expected = _encode(values, True, len(values))
    torch.testing.assert_close(actual, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected.scales[0], rtol=0, atol=0)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("mode", ["token", "replay"])
@pytest.mark.parametrize("horizon", ["near_unit", "mixed"])
@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_wide_long_trajectory(kda, mode, horizon, state_format):
    torch.manual_seed(329)
    q, k = [
        F.normalize(torch.randn(257, 2, 128, device="cuda"), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(257, 2, 70, device="cuda", requires_grad=True)
    shape = q.shape if kda else q.shape[:-1]
    g = -torch.rand(shape, device="cuda") * (1e-4 if horizon == "near_unit" else 3)
    if horizon == "mixed":
        g[::7] = -30
    g.requires_grad_()
    beta = (torch.rand(257, 2, device="cuda") * 0.6).requires_grad_()
    initial = (torch.randn(2, 128, 70, device="cuda") * 0.1).requires_grad_()
    cfg = LinearAttentionDecodeConfig(mode=mode, replay={"window": 8} if mode == "replay" else None)
    args = q, k, v, g, beta
    expected, ec = recurrent_decode_reference(
        *args,
        config=cfg,
        state_qdq=True,
        state_format=state_format,
        block_v=64,
        initial_state=initial,
    )
    actual, ac = recurrent_decode(
        *args,
        config=cfg.model_copy(update={"implementation": "triton"}),
        state_qdq=True,
        state_format=state_format,
        block_v=64,
        initial_state=initial,
    )
    probes = [torch.randn_like(actual), torch.randn_like(ac.reconstruct())]
    gradients = [
        torch.autograd.grad(sum((a * p).sum() for a, p in zip(result, probes)), (*args, initial))
        for result in ((actual, ac.reconstruct()), (expected, ec.reconstruct()))
    ]
    # Fixed reduction and scale arithmetic must also agree over long trajectories.
    for a, e in zip(
        (actual, ac.reconstruct(), *gradients[0]), (expected, ec.reconstruct(), *gradients[1])
    ):
        relative = (a - e).norm() / e.norm().clamp_min(1e-8)
        assert relative < 1e-4, relative.item()
        assert torch.isfinite(a).all()


@pytest.mark.parametrize("block_v", [32, 128])
@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_full_key_width_and_value_block_tail(block_v, kda, state_format):
    torch.manual_seed(418)
    q, k = [
        F.normalize(torch.randn(9, 2, 128, device="cuda"), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(9, 2, 133, device="cuda", requires_grad=True)
    g = (-torch.rand(q.shape if kda else q.shape[:-1], device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(9, 2, device="cuda", requires_grad=True)
    initial = torch.randn(2, 128, 133, device="cuda", requires_grad=True)
    args = q, k, v, g, beta
    cfg = LinearAttentionDecodeConfig(mode="replay", replay={"window": 3})
    results = []
    for implementation in ("torch", "triton"):
        output, carry = recurrent_decode(
            *args,
            config=cfg.model_copy(update={"implementation": implementation}),
            state_qdq=True,
            state_format=state_format,
            block_v=block_v,
            initial_state=initial,
        )
        final = carry.reconstruct()
        gradients = torch.autograd.grad(
            output.square().sum() + final.square().sum(), (*args, initial)
        )
        results.append((output, final, *gradients))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-5)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize(
    ("mode", "block_v", "readout", "factor_qdq"),
    [
        ("token", 32, "stored", False),
        ("replay", 64, "working", False),
        ("replay", 128, "stored", True),
    ],
)
@pytest.mark.timeout(180)
def test_hadamard_fused_split_carry_and_gradients(kda, mode, block_v, readout, factor_qdq):
    torch.manual_seed(917)
    q, k = [
        F.normalize(torch.randn(37, 2, 17, device="cuda"), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(37, 2, 96, device="cuda", requires_grad=True)
    g = (-torch.rand(q.shape if kda else q.shape[:-1], device="cuda") * 0.03).requires_grad_()
    beta = (torch.rand(37, 2, device="cuda") * 0.4).requires_grad_()
    initial = (torch.randn(2, 17, 96, device="cuda") * 0.1).requires_grad_()
    args = (q, k, v, g, beta)
    cfg = LinearAttentionDecodeConfig(
        mode=mode,
        readout=readout,
        state_codec="int8_hadamard32",
        decay_log_step=1 / 256,
        replay={"window": 8, "factor_qdq": factor_qdq} if mode == "replay" else None,
    )
    shared = {"state_qdq": True, "state_format": "int8", "block_v": block_v}
    expected, ref_carry = recurrent_decode_reference(
        *args, config=cfg, initial_state=initial, **shared
    )
    carry, start, pieces = None, 0, []
    for end in (0, 2, 8, 8, 19, 37):
        output, carry = recurrent_decode(
            *(x[start:end] for x in args),
            config=cfg.model_copy(update={"implementation": "triton"}),
            initial_state=initial if carry is None else None,
            carry=carry,
            **shared,
        )
        pieces.append(output)
        start = end
    actual = torch.cat(pieces)
    assert carry.value_basis == "hadamard32" and carry.cursor == ref_carry.cursor
    assert carry.anchor.scales.shape == (2, 17, 3) and carry.anchor.scales.dtype == torch.float16
    torch.testing.assert_close(carry.anchor.scales, ref_carry.anchor.scales, rtol=0, atol=0)
    actual_tensors = (
        actual,
        carry.reconstruct(),
        carry.anchor.values,
        *(e.update.values for e in carry.entries),
    )
    expected_tensors = (
        expected,
        ref_carry.reconstruct(),
        ref_carry.anchor.values,
        *(e.update.values for e in ref_carry.entries),
    )
    probes = [torch.randn_like(x) for x in expected_tensors]
    gradients = [
        torch.autograd.grad(sum((x * p).sum() for x, p in zip(result, probes)), (*args, initial))
        for result in (actual_tensors, expected_tensors)
    ]
    for a, e in zip((*actual_tensors, *gradients[0]), (*expected_tensors, *gradients[1])):
        torch.testing.assert_close(a, e, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.timeout(180)
def test_hadamard_long_trajectory(kda):
    torch.manual_seed(991)
    q, k = [
        F.normalize(torch.randn(257, 1, 128, device="cuda"), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(257, 1, 64, device="cuda", requires_grad=True)
    g = (-torch.rand(q.shape if kda else q.shape[:-1], device="cuda") * 1e-4).requires_grad_()
    beta = (torch.rand(257, 1, device="cuda") * 0.1).requires_grad_()
    initial = (torch.randn(1, 128, 64, device="cuda") * 0.1).requires_grad_()
    args = (q, k, v, g, beta)
    results = []
    for implementation in ("torch", "triton"):
        cfg = LinearAttentionDecodeConfig(
            mode="replay",
            implementation=implementation,
            state_codec="int8_hadamard32",
            replay={"window": 8, "factor_qdq": False},
        )
        out, carry = recurrent_decode(
            *args, config=cfg, initial_state=initial, state_qdq=True, state_format="int8"
        )
        final = carry.reconstruct()
        grads = torch.autograd.grad(out.square().sum() + final.square().sum(), (*args, initial))
        results.append((out, final, *grads))
    for a, e in zip(*results):
        assert torch.isfinite(a).all() and torch.isfinite(e).all()
        assert (a - e).norm() / e.norm().clamp_min(1e-8) < 1e-4
