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

"""Native two-precision GDN parity and an independent triangular-solve adjoint."""

import pytest
import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig, gdn_state_qat
from modelopt.torch.quantization.linear_attention._two_precision import (
    two_precision_decode,
    validate_two_precision,
)
from modelopt.torch.quantization.linear_attention.utils import forward_value
from modelopt.torch.quantization.nn import TensorQuantizer

native = pytest.importorskip("emulation", reason="Requires pinned upstream ReplaySSM emulation")


@pytest.fixture(params=["fp8", "nvfp4"])
def read_quantizer(request):
    nvfp4 = request.param == "nvfp4"
    return TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1) if nvfp4 else (4, 3),
            type="dynamic",
            pass_through_bwd=True,
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)} if nvfp4 else {-1: 128},
        )
    ).cuda()


def _oracle(q, k, v, g, beta, state, quantizer):
    # Canonical [H,K,V] state, unlike the production value-first window adjoint.
    def rounded(x, dtype=torch.bfloat16):
        return forward_value(x, x.to(dtype))

    def read_copy(x):
        rounded = quantizer(x.detach().transpose(-1, -2).contiguous()).transpose(-1, -2)
        return forward_value(x, rounded)

    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * q.shape[-1] ** -0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    beta = rounded(beta)
    outputs = []
    for start in range(0, len(q), 16):
        stop = min(start + 16, len(q))
        local_g = g[start:stop].cumsum(0)
        keys = k[start:stop].transpose(0, 1)
        read = read_copy(state)
        updates = []
        for i in range(stop - start):
            t = start + i
            decay = local_g[i].exp()[:, None, None]
            weights = (local_g[i] - local_g[:i]).exp().T
            base = rounded(state if t == 0 or i == 15 else read)
            working = decay * base
            if i == 15:
                # Solve all raw-input residuals at once, rather than sequential replay.
                all_keys = rounded(keys)
                gram = all_keys @ all_keys.transpose(-1, -2)
                gates = local_g.T
                retentions = (gates[:, :, None] - gates[:, None, :]).exp()
                b = beta[start:stop].T
                lower = torch.tril(gram * retentions * b[:, :, None], diagonal=-1)
                lower = lower + torch.eye(16, device=q.device)
                residual = rounded(
                    v[start:stop].transpose(0, 1)
                    - gates.exp()[..., None] * (all_keys @ rounded(state))
                )
                u = torch.linalg.solve_triangular(lower, b[..., None] * residual, upper=False)
                weighted = u * (gates[:, -1:] - gates).exp()[..., None]
                output = torch.einsum("hk,hkv->hv", q[t], working)
                output = output + torch.einsum("hlk,hk,hlv->hv", keys, q[t], weighted)
                state = decay * state + all_keys.transpose(-1, -2) @ rounded(weighted)
            else:
                if i:
                    working = working + keys[:, :i].transpose(-1, -2) @ (
                        torch.stack(updates, 1) * weights[..., None]
                    )
                delta = beta[t, :, None] * (v[t] - torch.einsum("hk,hkv->hv", k[t], working))
                working = working + k[t, :, :, None] * delta[:, None, :]
                output = torch.einsum("hk,hkv->hv", q[t], working)
                updates.append(rounded(delta, torch.float16))
            outputs.append(output)
    return torch.stack(outputs)


def test_native_two_precision_forward_and_window_backward(read_quantizer):
    torch.manual_seed(41)
    q, k = [
        torch.randn(33, 2, 128, device="cuda", dtype=torch.bfloat16).float().requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(33, 2, 64, device="cuda", dtype=torch.bfloat16).float().requires_grad_()
    a = (torch.randn(33, 2, device="cuda") * 0.1 - 3).bfloat16().float()
    b = torch.randn_like(a).bfloat16().float()
    g = (-torch.nn.functional.softplus(a)).requires_grad_()
    beta = b.sigmoid().requires_grad_()
    state = (torch.randn(2, 128, 64, device="cuda") * 0.1).requires_grad_()
    params = (q, k, v, g, beta, state)
    rate, bias = torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda")
    mode = validate_two_precision(None, read_quantizer)
    actual = two_precision_decode(*params, (a, b, rate, bias), mode)
    runner = native.GDNReplay(state.detach().transpose(-1, -2)[None].contiguous(), 2, mode)
    exact = native.GDNReplay(state.detach().transpose(-1, -2)[None].contiguous(), 2, "s32_rs")
    expected = []
    for i in range(33):
        packed = torch.cat([x[i].flatten() for x in (q, k, v)])[None].bfloat16()
        inputs = (packed, a[i : i + 1], b[i : i + 1], rate, bias)
        expected.append(runner.step(*inputs)[0].clone())
        exact_output = exact.step(*inputs)[0].clone()
        # Quantized reads must never contaminate the FP32 recurrence checkpoint.
        torch.testing.assert_close(runner.state_hi, exact.state_hi, rtol=0, atol=0)
        if i in (0, 15, 31):
            torch.testing.assert_close(expected[-1], exact_output, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    reference = _oracle(*params, read_quantizer)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-6)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, params, upstream)
    expected_gradients = torch.autograd.grad(reference, params, upstream)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        torch.testing.assert_close(gradient, expected_gradient, rtol=5e-3, atol=2e-5)


def test_two_precision_public_api_padding_and_baseline(read_quantizer):
    q, k, v = [
        torch.randn(1, 19, 1, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(3)
    ]
    raw_g = torch.full((1, 19, 1), -3.0, device="cuda", dtype=torch.bfloat16)
    raw_beta = torch.zeros_like(raw_g)
    rate = torch.zeros(1, device="cuda")
    state = torch.randn(1, 1, 128, 128, device="cuda", requires_grad=True)
    quantizer = read_quantizer
    kwargs = {
        "policy": LinearAttentionConfig(
            backend="serving", precision="replayssm_two_precision", replay_window=16
        ),
        "read_quantizer": quantizer,
        "initial_state": state,
        "prefill_lengths": [0],
        "sequence_lengths": [17],
        "use_qk_l2norm_in_kernel": True,
        "gate_inputs": (raw_g, raw_beta, rate, rate),
    }
    args = q, k, v, -torch.nn.functional.softplus(raw_g.float()), raw_beta.float().sigmoid()
    output, final = gdn_state_qat(*args, **kwargs)
    assert final is None and torch.count_nonzero(output[:, 17:]) == 0
    gradients = torch.autograd.grad(output.float().square().sum(), (q, k, v, state))
    for gradient in gradients[:3]:
        assert torch.count_nonzero(gradient[:, 17:]) == 0
    assert gradients[-1].norm() > 0
    with pytest.raises(ValueError, match="explicit prefill lengths"):
        gdn_state_qat(*args, **{**kwargs, "prefill_lengths": None})
    with pytest.raises(ValueError, match="K=128"):
        gdn_state_qat(q.repeat_interleave(2, -1), k.repeat_interleave(2, -1), *args[2:], **kwargs)
    quantizer.disable()
    baseline, _ = gdn_state_qat(*args, **kwargs)
    torch.testing.assert_close(output[:, 0], baseline[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 15], baseline[:, 15], rtol=0, atol=0)
    assert not torch.equal(output[:, 1:15], baseline[:, 1:15])


def test_two_precision_packed_matches_separate_sequences(read_quantizer):
    torch.manual_seed(42)
    inputs = [
        torch.randn(1, 160, 1, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(3)
    ]
    a = torch.full((1, 160, 1), -3.0, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    rate = torch.zeros(1, device="cuda")
    quantizer = read_quantizer
    policy = LinearAttentionConfig(
        backend="serving", precision="replayssm_two_precision", replay_window=16
    )

    def forward(xs, start, stop, prefixes, lengths, boundaries=None):
        raw_g, raw_beta = a[:, start:stop], b[:, start:stop]
        return gdn_state_qat(
            *(x[:, start:stop] for x in xs),
            -torch.nn.functional.softplus(raw_g.float()),
            raw_beta.float().sigmoid(),
            policy=policy,
            read_quantizer=quantizer,
            prefill_lengths=prefixes,
            sequence_lengths=lengths,
            cu_seqlens=boundaries,
            use_qk_l2norm_in_kernel=True,
            gate_inputs=(raw_g, raw_beta, rate, rate),
        )[0]

    # Different prefix lengths, partial replay windows, and padding in both segments.
    bounds = torch.tensor([0, 80, 160], device="cuda", dtype=torch.int32)
    packed = forward(inputs, 0, 160, [62, 58], [79, 75], bounds)
    separate = torch.cat(
        [forward(inputs, 0, 80, [62], [79]), forward(inputs, 80, 160, [58], [75])], 1
    )
    torch.testing.assert_close(packed, separate, rtol=0, atol=0)
    upstream = torch.randn_like(packed)
    grads = torch.autograd.grad(packed, inputs, upstream, retain_graph=True)
    reference_grads = torch.autograd.grad(separate, inputs, upstream)
    for actual, expected in zip(grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual[:, 79:80].count_nonzero() and not actual[:, 155:].count_nonzero()
    isolated = torch.autograd.grad(packed[:, 80:].float().square().sum(), inputs)
    assert all(not gradient[:, :80].count_nonzero() for gradient in isolated)
    with torch.no_grad():
        changed = [x.clone() for x in inputs]
        for x in changed:
            x[:, :80].neg_()
        result = forward(changed, 0, 160, [62, 58], [79, 75], bounds)
        torch.testing.assert_close(result[:, 80:], packed[:, 80:], rtol=0, atol=0)
