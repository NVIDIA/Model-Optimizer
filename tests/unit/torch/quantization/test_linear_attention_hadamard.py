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

import torch
import torch.nn.functional as F

from modelopt.torch.quantization.linear_attention import (
    LinearAttentionDecodeConfig,
    recurrent_decode_reference,
)


def _rotate(value):
    # Independent dense Sylvester matrix; production uses a butterfly transform.
    matrix = value.new_tensor([[(-1) ** (i & j).bit_count() for j in range(32)] for i in range(32)])
    return (value.reshape(*value.shape[:-1], -1, 32) @ (matrix / 32**0.5)).reshape_as(value)


def _qdq(value):
    with torch.no_grad():
        groups = value.float().unflatten(-1, (-1, 32))
        scale = (groups.abs().amax(-1, keepdim=True) / 127).clamp_min(6e-8)
        normalized = groups / scale
        codes = torch.where(normalized >= 0, (normalized + 0.5).floor(), (normalized - 0.5).ceil())
        rounded = (codes.clamp(-127, 127) * scale.half().float()).flatten(-2).to(value.dtype)
    return (value - value.detach()) + rounded


def _inputs(length=11):
    torch.manual_seed(321)
    q, k = [F.normalize(torch.randn(length, 2, 4, dtype=torch.float64), dim=-1) for _ in range(2)]
    v = torch.randn(length, 2, 64, dtype=torch.float64)
    g = -torch.rand(k.shape, dtype=torch.float64) * 0.05
    beta = torch.rand(length, 2, dtype=torch.float64) * 0.4
    initial = torch.randn(2, 4, 64, dtype=torch.float64) * 0.1
    return tuple(x.requires_grad_() for x in (q, k, v, g, beta)), initial.requires_grad_()


def _oracle(args, initial):
    q, k, v, g, beta = args
    state = _rotate(initial)
    state = _qdq(state)
    outputs = []
    for t in range(len(q)):
        decay = g[t].exp().reshape(2, -1, 1)
        decayed = state * decay
        correction = beta[t, :, None] * (_rotate(v[t]) - torch.einsum("hk,hkv->hv", k[t], decayed))
        working = decayed + k[t, :, :, None] * correction[:, None, :]
        state = _qdq(working) if (t + 1) % 3 == 0 else working
        outputs.append(_rotate(torch.einsum("hk,hkv->hv", q[t], state)) / q.shape[-1] ** 0.5)
    return torch.stack(outputs), _rotate(state)


def _check_with_grads(actual, expected, args, initial, tolerance=2e-10):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=tolerance, atol=tolerance)
    gradients = [
        torch.autograd.grad(
            sum(x.square().sum() for x in result), (*args, initial), retain_graph=True
        )
        for result in (actual, expected)
    ]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=tolerance, atol=tolerance)


def test_hadamard_replay_matches_dense_oracle_and_split_carry():
    args, initial = _inputs()
    cfg = LinearAttentionDecodeConfig(
        mode="replay",
        state_codec="int8_hadamard32",
        replay={"window": 3, "factor_qdq": False},
    )
    kwargs = {"config": cfg, "state_format": "int8", "state_qdq": True}
    output, carry = recurrent_decode_reference(*args, initial_state=initial, **kwargs)
    _check_with_grads((output, carry.reconstruct()), _oracle(args, initial), args, initial)
    assert carry.value_basis == "hadamard32" and carry.anchor.block_v == 32
    assert carry.anchor.scales.shape == (2, 4, 2)
    first, split_carry = recurrent_decode_reference(
        *(x[:2] for x in args), initial_state=initial, **kwargs
    )
    second, split_carry = recurrent_decode_reference(
        *(x[2:] for x in args), carry=split_carry, **kwargs
    )
    _check_with_grads(
        (torch.cat((first, second)), split_carry.reconstruct()),
        (output, carry.reconstruct()),
        args,
        initial,
        0,
    )
