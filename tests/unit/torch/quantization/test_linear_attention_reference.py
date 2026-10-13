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
from _test_utils.torch.quantization.linear_attention_reference import (
    chunk_gdn_reference,
    recurrent_delta_rule_reference,
    state_fp8_qdq_reference,
)


def inputs(length=7, kda=False):
    torch.manual_seed(81)
    q, k = [
        F.normalize(torch.randn(1, length, 1, 3, dtype=torch.float64), dim=-1) for _ in range(2)
    ]
    v = torch.randn(1, length, 2, 4, dtype=torch.float64)
    g = -torch.rand(1, length, 2, *([3] if kda else []), dtype=torch.float64)
    beta = torch.rand(1, length, 2, dtype=torch.float64)
    return [x.requires_grad_() for x in (q, k, v, g, beta)]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("state_v_first", [False, True])
def test_chunk_recurrence_output_state_and_gradients(packed, state_v_first):
    args = inputs()
    state = torch.randn(2 if packed else 1, 2, 3, 4, dtype=torch.float64)
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
    state.requires_grad_()
    kwargs = {
        "initial_state": state,
        "state_v_first": state_v_first,
        "cu_seqlens": torch.tensor([0, 2, 7]) if packed else None,
    }
    actual = chunk_gdn_reference(*args, chunk_size=3, **kwargs)
    expected = recurrent_delta_rule_reference(*args, **kwargs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)
    probe = [torch.randn_like(x) for x in actual]
    grads = [
        torch.autograd.grad(sum((x * p).sum() for x, p in zip(result, probe)), (*args, state))
        for result in (actual, expected)
    ]
    for a, e in zip(*grads):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kda", [False, True])
def test_recurrence_smooth_gradcheck(kda):
    args = inputs(length=2, kda=kda)
    state = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda *x: recurrent_delta_rule_reference(*x[:-1], initial_state=x[-1]),
        (*args, state),
        fast_mode=True,
    )


def test_kda_one_step_and_scalar_decay_equivalence():
    q, k, v, g, beta = inputs(length=1, kda=True)
    state = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    out, final = recurrent_delta_rule_reference(q, k, v, g, beta, initial_state=state)
    keys = k[:, 0].expand(-1, 2, -1)
    decayed = state * g[:, 0].exp().unsqueeze(-1)
    residual = v[:, 0] - torch.einsum("bhk,bhkv->bhv", keys, decayed)
    expected = decayed + torch.einsum("bhk,bhv->bhkv", keys, beta[:, 0, :, None] * residual)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(
        out[:, 0], torch.einsum("bhk,bhkv->bhv", q[:, 0].expand(-1, 2, -1), expected) / 3**0.5
    )
    scalar = g[..., 0]
    gdn = recurrent_delta_rule_reference(q, k, v, scalar, beta, initial_state=state)
    kda = recurrent_delta_rule_reference(
        q, k, v, scalar.unsqueeze(-1).expand_as(g), beta, initial_state=state
    )
    for a, e in zip(kda, gdn):
        torch.testing.assert_close(a, e)


def test_state_qdq_granularity_zero_tail_and_identity_ste():
    torch.manual_seed(14)
    state = torch.randn(2, 2, 3, 37, requires_grad=True)
    with torch.no_grad():
        state[0].zero_()
        state[1, ..., :16] *= 10
    quantized = state_fp8_qdq_reference(state, block_v=16)
    assert torch.isfinite(quantized).all()
    assert torch.equal(quantized[0], state[0])
    assert not torch.equal(quantized, state)
    assert not torch.equal(quantized, state_fp8_qdq_reference(state, block_v=64))
    probe = torch.randn_like(state)
    (grad,) = torch.autograd.grad((quantized * probe).sum(), state)
    torch.testing.assert_close(grad, probe, rtol=0, atol=0)


def test_packed_sequences_reset_the_state():
    args = inputs(length=7, kda=True)
    initial = torch.randn(2, 2, 3, 4, dtype=torch.float64)
    packed = recurrent_delta_rule_reference(
        *args, initial_state=initial, cu_seqlens=torch.tensor([0, 2, 7])
    )
    separate = [
        recurrent_delta_rule_reference(
            *(x[:, lo:hi] for x in args), initial_state=initial[n : n + 1]
        )
        for n, (lo, hi) in enumerate(((0, 2), (2, 7)))
    ]
    torch.testing.assert_close(packed[0], torch.cat([x[0] for x in separate], dim=1))
    torch.testing.assert_close(packed[1], torch.cat([x[1] for x in separate]))
