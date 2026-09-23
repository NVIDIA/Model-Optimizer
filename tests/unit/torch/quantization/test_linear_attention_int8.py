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
    LinearAttentionDecodeConfig,
    recurrent_decode_reference,
    state_qdq_reference,
)
from modelopt.torch.quantization.linear_attention.decode import _encode
from modelopt.torch.quantization.nn import TensorQuantizer

INT8 = {"num_bits": 8, "unsigned": False, "narrow_range": True, "type": "dynamic", "axis": (0, 1)}


@pytest.mark.parametrize("block_v", [16, 32, 64, 128])
def test_int8_state_codec_matches_modelopt_and_identity_gradient(block_v):
    # Exact scale two exercises both signs, endpoints and nearest-even half ties.
    state = torch.zeros(2, 3, 5, block_v + 3)
    state[..., 0, :9] = torch.tensor([-127, -126.5, -1.5, -0.5, 0, 0.5, 1.5, 126.5, 127]) * 2
    state[0, :, 0, block_v:] = torch.tensor([-254, 1, 254])
    state.requires_grad_()
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**INT8))
    expected = torch.cat(
        [quantizer(tile.flatten(-2)).reshape_as(tile) for tile in state.split(block_v, -1)], -1
    )
    encoded = _encode(state, True, block_v, state=True, state_format="int8")
    reference = state_qdq_reference(state, block_v, "int8")
    torch.testing.assert_close(encoded.values, expected, rtol=0, atol=0)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    assert encoded.format == "int8"
    assert torch.equal(encoded.scales[1, :, -1], torch.ones(3))
    assert not encoded.scales.requires_grad
    probe = torch.randn_like(state)
    for result in (encoded.values, reference):
        (gradient,) = torch.autograd.grad((result * probe).sum(), state, retain_graph=True)
        torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


def _inputs(kda):
    torch.manual_seed(913)
    q, k = [F.normalize(torch.randn(23, 2, 16), dim=-1).requires_grad_() for _ in range(2)]
    v = torch.randn(23, 2, 19, requires_grad=True)
    g = (-torch.rand(q.shape if kda else q.shape[:-1]) * 0.03).requires_grad_()
    beta = (torch.rand(23, 2) * 0.4).requires_grad_()
    state = (torch.randn(2, 16, 19) * 0.1).requires_grad_()
    return (q, k, v, g, beta), state


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("mode", ["token", "replay"])
def test_int8_decode_continuation_and_format_rejection(kda, mode):
    args, initial = _inputs(kda)
    cfg = LinearAttentionDecodeConfig(mode=mode, replay={"window": 5} if mode == "replay" else None)
    kwargs = {"config": cfg, "state_qdq": True, "state_format": "int8", "block_v": 16}
    expected, final = recurrent_decode_reference(*args, initial_state=initial, **kwargs)
    pieces, carry, start = [], None, 0
    for end in (0, 3, 10, 10, 23):
        output, carry = recurrent_decode_reference(
            *(x[start:end] for x in args),
            carry=carry,
            initial_state=initial if carry is None else None,
            **kwargs,
        )
        pieces.append(output)
        start = end
    actual = torch.cat(pieces)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(carry.reconstruct(), final.reconstruct(), rtol=0, atol=0)
    probes = (torch.randn_like(expected), torch.randn_like(initial))
    grads = [
        torch.autograd.grad((o * probes[0]).sum() + (s * probes[1]).sum(), (*args, initial))
        for o, s in ((actual, carry.reconstruct()), (expected, final.reconstruct()))
    ]
    for actual_grad, expected_grad in zip(*grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert carry.anchor.format == "int8"
    assert all(e.key.format == e.update.format == "fp8_e4m3" for e in carry.entries)
    with pytest.raises(ValueError, match="Carry policy"):
        recurrent_decode_reference(
            *(x[:0] for x in args), carry=carry, **{**kwargs, "state_format": "fp8_e4m3"}
        )
