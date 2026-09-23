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

import itertools

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    matmul_gdn,
    matmul_kda,
)
from modelopt.torch.quantization.linear_attention.config import LinearAttentionDecodeConfig
from modelopt.torch.quantization.linear_attention.decode import recurrent_decode_reference
from modelopt.torch.quantization.linear_attention.reference import recurrent_delta_rule_reference


def _inputs(kda=True, length=73):
    torch.manual_seed(193)
    q, k = [
        F.normalize(torch.randn(length, 2, 16, dtype=torch.float64), dim=-1).requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(length, 2, 11, dtype=torch.float64, requires_grad=True)
    g = (
        -torch.rand((length, 2, 16) if kda else (length, 2), dtype=torch.float64) * 0.03
    ).requires_grad_()
    beta = (torch.rand(length, 2, dtype=torch.float64) * 0.4).requires_grad_()
    state = (torch.randn(2, 16, 11, dtype=torch.float64) * 0.1).requires_grad_()
    return (q, k, v, g, beta), state


def _values_and_grads(output, state, args, initial):
    loss = output.square().sum() + state.square().sum()
    return output, state, *torch.autograd.grad(loss, (*args, initial), retain_graph=True)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("mode", ["token", "replay"])
def test_unquantized_decode_recurrence_and_gradients(kda, mode):
    args, state = _inputs(kda)
    cfg = LinearAttentionDecodeConfig(
        mode=mode, replay={"window": 8, "factor_qdq": False} if mode == "replay" else None
    )
    output, carry = recurrent_decode_reference(*args, config=cfg, initial_state=state)
    exact, final = recurrent_delta_rule_reference(
        *(x.unsqueeze(0) for x in args), initial_state=state.unsqueeze(0)
    )
    for a, b in zip(
        _values_and_grads(output, carry.reconstruct(), args, state),
        _values_and_grads(exact[0], final[0], args, state),
    ):
        torch.testing.assert_close(a, b, rtol=2e-11, atol=1e-12)
    assert carry.position == 73
    assert carry.cursor == (1 if mode == "replay" else 0)


@pytest.mark.parametrize("mode", ["token", "replay", "reencode"])
@pytest.mark.parametrize("readout", ["working", "stored"])
def test_quantized_carry_continuation_metadata_and_gradients(mode, readout):
    args, state = _inputs()
    cfg = LinearAttentionDecodeConfig(
        mode="token" if mode == "token" else "replay",
        readout=readout,
        replay={"window": 8, "encoding": "reencode" if mode == "reencode" else "once"}
        if mode != "token"
        else None,
    )
    expected, final = recurrent_decode_reference(
        *args, config=cfg, initial_state=state, state_qdq=True, block_v=16
    )
    chunks = []
    carry = None
    lo = 0
    for hi in [0, 5, 16, 16, 37, 73]:
        result, carry = recurrent_decode_reference(
            *(x[lo:hi] for x in args),
            config=cfg,
            carry=carry,
            initial_state=state if carry is None else None,
            state_qdq=True,
            block_v=16,
        )
        chunks.append(result)
        lo = hi
    actual = torch.cat(chunks)
    for a, b in zip(
        _values_and_grads(actual, carry.reconstruct(), args, state),
        _values_and_grads(expected, final.reconstruct(), args, state),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert carry.position == 73
    assert carry.anchor.format == "fp8_e4m3"
    assert carry.anchor.scales.shape == (2, 1)
    assert not carry.anchor.scales.requires_grad
    for entry in carry.entries:
        assert entry.key.scales.shape == (2, 1)
        assert entry.update.values.requires_grad
        assert not entry.update.scales.requires_grad


def test_initial_write_and_readout_order_are_explicit():
    args, state = _inputs(length=1)
    outputs = []
    for readout in ["working", "stored"]:
        output, carry = recurrent_decode_reference(
            *args,
            config=LinearAttentionDecodeConfig(readout=readout),
            initial_state=state,
            state_qdq=True,
            block_v=16,
        )
        outputs.append(output)
    assert not torch.equal(*outputs)
    empty, untouched = recurrent_decode_reference(
        *(x[:0] for x in args),
        config=LinearAttentionDecodeConfig(),
        initial_state=state,
        state_qdq=True,
    )
    assert empty.shape == (0, 2, 11)
    assert untouched.anchor.values is state
    assert not untouched.started


def test_grid_gate_ste_keeps_gate_gradients_and_changes_trajectory():
    args, state = _inputs()
    output, carry = recurrent_decode_reference(
        *args, config=LinearAttentionDecodeConfig(decay_log_step=0.02), initial_state=state
    )
    grad = torch.autograd.grad(output.square().sum() + carry.reconstruct().square().sum(), args[3])[
        0
    ]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    exact, _ = recurrent_decode_reference(
        *args, config=LinearAttentionDecodeConfig(), initial_state=state
    )
    assert not torch.equal(output, exact)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("prefixes", [[0, 0, 0, 0], [0, 3, 65, 0], [0, 5, 68, 0]])
def test_explicit_packed_phase_with_empty_entries(kda, prefixes):
    args, one_state = _inputs(kda)
    args = tuple(x.unsqueeze(0) for x in args)
    initial = torch.stack([one_state.detach().clone() for _ in range(4)]).requires_grad_()
    boundaries = [0, 0, 5, 73, 73]
    function = matmul_kda if kda else matmul_gdn
    actual = function(
        *args,
        policy=LinearAttentionConfig(backend="matmul", decode={}),
        prefill_lengths=prefixes,
        cu_seqlens=torch.tensor(boundaries),
        initial_state=initial,
        output_final_state=True,
    )
    outputs, states = [], []
    for n, (lo, hi) in enumerate(itertools.pairwise(boundaries)):
        if lo == hi:
            outputs.append(args[2][0, lo:hi])
            states.append(initial[n])
            continue
        o, s = recurrent_delta_rule_reference(
            *(x[:, lo:hi] for x in args), initial_state=initial[n : n + 1]
        )
        outputs.append(o[0])
        states.append(s[0])
    expected = torch.cat(outputs).unsqueeze(0), torch.stack(states)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-11, atol=1e-12)
    gradients = [
        torch.autograd.grad(
            sum(x.square().sum() for x in result), (*args, initial), retain_graph=True
        )
        for result in (actual, expected)
    ]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=2e-10, atol=1e-11)


@pytest.mark.parametrize("mode", ["token", "replay"])
def test_carry_rejects_changed_policy_and_shape(mode):
    args, state = _inputs(length=4)
    cfg = LinearAttentionDecodeConfig(mode=mode, replay={"window": 8} if mode == "replay" else None)
    _, carry = recurrent_decode_reference(*args, config=cfg, initial_state=state, state_qdq=True)
    for changed in [
        cfg.model_copy(update={"readout": "working"}),
        cfg.model_copy(update={"decay_log_step": 0.01}),
    ]:
        with pytest.raises(ValueError, match="policy or state shape"):
            recurrent_decode_reference(*args, config=changed, carry=carry, state_qdq=True)
    with pytest.raises(ValueError, match="policy or state shape"):
        recurrent_decode_reference(*args, config=cfg, carry=carry, state_qdq=True, block_v=16)
    with pytest.raises(ValueError, match="either carry or initial"):
        recurrent_decode_reference(
            *args, config=cfg, carry=carry, initial_state=state, state_qdq=True
        )


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("all_empty", [False, True])
def test_grouped_heads_transposed_state_and_empty_batch(kda, all_empty):
    args, state = _inputs(kda, length=0 if all_empty else 9)
    q, k, v, g, beta = [x.unsqueeze(0).repeat(2, *([1] * x.ndim)) for x in args]
    q, k = q[:, :, :1], k[:, :, :1]
    initial = state.unsqueeze(0).repeat(2, 1, 1, 1).transpose(-1, -2).detach().requires_grad_()
    function = matmul_kda if kda else matmul_gdn
    policy = LinearAttentionConfig(
        backend="matmul", decode={"mode": "replay", "replay": {"window": 3}}
    )
    shared = {
        "policy": policy,
        "state_qdq": True,
        "output_final_state": True,
    }
    actual = function(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial,
        state_v_first=True,
        prefill_lengths=[0, 0] if all_empty else [2, 5],
        **shared,
    )
    expected = function(
        q.repeat_interleave(2, dim=2),
        k.repeat_interleave(2, dim=2),
        v,
        g,
        beta,
        initial_state=initial.transpose(-1, -2),
        prefill_lengths=[0, 0] if all_empty else [2, 5],
        **shared,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1].transpose(-1, -2), expected[1], rtol=0, atol=0)
    gradients = [
        torch.autograd.grad(result[0].sum() + result[1].sum(), initial, retain_graph=True)[0]
        for result in (actual, expected)
    ]
    torch.testing.assert_close(*gradients, rtol=0, atol=0)
    if all_empty:
        torch.testing.assert_close(actual[1], initial, rtol=0, atol=0)
