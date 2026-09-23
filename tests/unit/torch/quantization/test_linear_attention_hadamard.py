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

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionDecodeConfig,
    matmul_gdn,
    matmul_kda,
    recurrent_decode_reference,
)
from modelopt.torch.quantization.linear_attention.decode import _encode, _hadamard32
from modelopt.torch.quantization.linear_attention.matmul import LinearAttentionMatmulSites
from modelopt.torch.quantization.nn import TensorQuantizer


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


def _inputs(kda, length=11):
    torch.manual_seed(321)
    q, k = [F.normalize(torch.randn(length, 2, 4, dtype=torch.float64), dim=-1) for _ in range(2)]
    v = torch.randn(length, 2, 64, dtype=torch.float64)
    g = -torch.rand(k.shape if kda else k.shape[:-1], dtype=torch.float64) * 0.05
    beta = torch.rand(length, 2, dtype=torch.float64) * 0.4
    initial = torch.randn(2, 4, 64, dtype=torch.float64) * 0.1
    return tuple(x.requires_grad_() for x in (q, k, v, g, beta)), initial.requires_grad_()


def _oracle(args, initial, mode, readout, quantize=True):
    q, k, v, g, beta = args
    state = _rotate(initial)
    if quantize:
        state = _qdq(state)
    outputs = []
    for t in range(len(q)):
        decay = g[t].exp().reshape(2, -1, 1)
        decayed = state * decay
        correction = beta[t, :, None] * (_rotate(v[t]) - torch.einsum("hk,hkv->hv", k[t], decayed))
        working = decayed + k[t, :, :, None] * correction[:, None, :]
        state = _qdq(working) if quantize and (mode == "token" or (t + 1) % 3 == 0) else working
        read = state if readout == "stored" else working
        outputs.append(_rotate(torch.einsum("hk,hkv->hv", q[t], read)) / q.shape[-1] ** 0.5)
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


def test_hadamard_codec_basis_scales_ties_and_identity_gradient():
    raw = torch.zeros(1, 2, 64, dtype=torch.float64)
    raw[0, 0, :7] = torch.tensor([254, -254, 1, -1, 3, -3, 0])
    raw[0, 1, 32:] = torch.linspace(-0.123, 0.123, 32)
    raw.requires_grad_()
    torch.testing.assert_close(_hadamard32(raw), _rotate(raw), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(_hadamard32(_hadamard32(raw)), raw, rtol=1e-14, atol=1e-13)
    encoded = _encode(raw, True, 64, state=True, state_format="int8", state_codec="int8_hadamard32")
    torch.testing.assert_close(encoded.values, _qdq(raw), rtol=0, atol=0)
    torch.testing.assert_close(
        encoded.values[0, 0, :7], raw.new_tensor([254, -254, 2, -2, 4, -4, 0])
    )
    assert encoded.scales.shape == (1, 2, 2)
    assert encoded.scales.dtype == torch.float16 and not encoded.scales.requires_grad
    assert encoded.scales[0, 0, 0] == 2
    assert encoded.scales[0, 0, 1] == torch.tensor(6e-8, dtype=torch.float16)
    probe = torch.randn_like(raw)
    (gradient,) = torch.autograd.grad((encoded.values * probe).sum(), raw)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize(
    ("mode", "readout"), [("token", "stored"), ("replay", "working"), ("replay", "stored")]
)
def test_hadamard_recurrence_matches_dense_oracle_and_split_carry(kda, mode, readout):
    args, initial = _inputs(kda)
    cfg = LinearAttentionDecodeConfig(
        mode=mode,
        readout=readout,
        state_codec="int8_hadamard32",
        replay={"window": 3, "factor_qdq": False} if mode == "replay" else None,
    )
    kwargs = {"config": cfg, "state_format": "int8", "state_qdq": True}
    output, carry = recurrent_decode_reference(*args, initial_state=initial, **kwargs)
    _check_with_grads(
        (output, carry.reconstruct()), _oracle(args, initial, mode, readout), args, initial
    )
    assert carry.value_basis == "hadamard32" and carry.anchor.block_v == 32
    assert carry.anchor.scales.shape == (2, 4, 2)
    pieces, split_carry, start = [], None, 0
    for end in (0, 2, 3, 3, 7, 11):
        piece, split_carry = recurrent_decode_reference(
            *(x[start:end] for x in args),
            carry=split_carry,
            initial_state=initial if split_carry is None else None,
            **kwargs,
        )
        if end == 0:
            assert split_carry.reconstruct() is initial and not split_carry.started
        pieces.append(piece)
        start = end
    _check_with_grads(
        (torch.cat(pieces), split_carry.reconstruct()),
        (output, carry.reconstruct()),
        args,
        initial,
        0,
    )
    with pytest.raises(ValueError, match="Carry policy"):
        recurrent_decode_reference(
            *args,
            carry=carry,
            **{**kwargs, "config": cfg.model_copy(update={"state_codec": "tile"})},
        )


@pytest.mark.parametrize("kda", [False, True])
def test_hadamard_disabled_qdq_is_a_change_of_basis(kda):
    args, initial = _inputs(kda)
    cfg = LinearAttentionDecodeConfig(mode="replay", replay={"window": 3, "factor_qdq": False})
    results = []
    for codec in ("tile", "int8_hadamard32"):
        out, carry = recurrent_decode_reference(
            *args, initial_state=initial, config=cfg.model_copy(update={"state_codec": codec})
        )
        results.append((out, carry.reconstruct()))
    _check_with_grads(*results, args, initial)


@pytest.mark.parametrize(("kda", "prefix"), [(False, 0), (False, 3), (True, 3), (True, 11)])
def test_hadamard_exact_prefill_handoff_and_state_layout(kda, prefix):
    args, initial = _inputs(kda)
    function = matmul_kda if kda else matmul_gdn
    policy = LinearAttentionConfig(backend="matmul", decode={"state_codec": "int8_hadamard32"})
    actual = function(
        *(x.unsqueeze(0) for x in args),
        sites=LinearAttentionMatmulSites(),
        policy=policy,
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        prefill_lengths=[prefix],
        state_qdq=True,
        state_format="int8",
        initial_state=initial.transpose(-1, -2).unsqueeze(0),
        state_v_first=True,
        output_final_state=True,
    )
    prefix_out, state = recurrent_decode_reference(
        *(x[:prefix] for x in args), initial_state=initial, config=LinearAttentionDecodeConfig()
    )
    suffix, carry = recurrent_decode_reference(
        *(x[prefix:] for x in args),
        initial_state=state.reconstruct(),
        config=policy.decode,
        state_qdq=True,
        state_format="int8",
    )
    expected = (torch.cat((prefix_out, suffix)), carry.reconstruct())
    _check_with_grads((actual[0][0], actual[1][0].transpose(-1, -2)), expected, args, initial)


def test_hadamard_rejects_unsupported_codec_combinations():
    args, initial = _inputs(False)
    cfg = LinearAttentionDecodeConfig(state_codec="int8_hadamard32")
    for kwargs, message in [
        ({"state_qdq": True}, "requires INT8"),
        ({"state_format": "int8", "block_v": 16}, "block_v >= 32"),
    ]:
        with pytest.raises(ValueError, match=message):
            recurrent_decode_reference(*args, config=cfg, initial_state=initial, **kwargs)
    with pytest.raises(ValueError, match="Dv divisible"):
        recurrent_decode_reference(args[0], args[1], args[2][..., :33], *args[3:], config=cfg)
    with pytest.raises(ValueError, match="decode handoff"):
        LinearAttentionDecodeConfig(state_codec="int8_hadamard32", prefill_state_qdq=True)
    with pytest.raises(ValueError, match="block_v >= 32"):
        LinearAttentionConfig(backend="matmul", state={"block_v": 16}, decode=cfg)
