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

from modelopt.recipe import load_recipe
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionDecodeConfig,
    matmul_gdn,
    matmul_kda,
    recurrent_decode,
)
from modelopt.torch.quantization.linear_attention.decode import _encode
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_state_qdq_matches_tensor_quantizer(state_format):
    torch.manual_seed(762)
    # A partial value tile checks grouping and padding against TensorQuantizer.
    value = torch.randn(2, 3, 5, 19, requires_grad=True)
    cfg = {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1)}
    if state_format == "int8":
        cfg.update(num_bits=8, unsigned=False, narrow_range=True)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**cfg))
    expected = torch.cat(
        [quantizer(tile.flatten(-2)).reshape_as(tile) for tile in value.split(16, -1)], -1
    )
    encoded = _encode(
        value, True, 16, state=True, state_format=state_format, state_quantizer=quantizer
    )
    torch.testing.assert_close(encoded.values, expected, rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((encoded.values * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)
    assert encoded.scales.shape == (2, 3, 2)
    assert not encoded.scales.requires_grad


def _inputs(kda=True, length=11):
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


@pytest.mark.parametrize(("kda", "block_size"), [(False, 32), (True, 64)])
def test_block_state_quantizer_prefill_decode_and_gradients(kda, block_size):
    args, initial = _inputs(kda, length=5)
    # Partial value groups exercise TensorQuantizer padding as well as per-key scales.
    recipe = load_recipe(
        "general/ptq/linear_attention_state_int8_block32_dynamic"
    ).quantize.model_dump()
    cfg = recipe["quant_cfg"][2 if kda else 1]["cfg"]
    cfg["block_sizes"] = {-1: block_size}
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**cfg))
    policy = LinearAttentionConfig(**recipe["linear_attention"][0]["cfg"])
    policy.decode.precision = "full"  # Double-precision recurrence/gradient oracle.
    policy.state.block_v = 16
    if kda:
        policy.decode.prefill_state_qdq = True
        policy.decode.readout = "stored"
    function = matmul_kda if kda else matmul_gdn
    output, final = function(
        *(x.unsqueeze(0) for x in args),
        policy=policy,
        state_quantizer=quantizer,
        prefill_lengths=[3],
        initial_state=initial.unsqueeze(0),
        output_final_state=True,
    )
    # GDN exercises the serving-aligned recipe; KDA retains prefix-QDQ/stored-read coverage.
    state = quantizer(initial) if policy.decode.prefill_state_qdq else initial
    expected = []
    q, k, v, g, beta = args
    for t in range(len(q)):
        if t == 3:
            state = quantizer(state)
        decay = g[t].exp().unsqueeze(-1) if kda else g[t].exp()[:, None, None]
        decayed = state * decay
        residual = v[t] - (k[t].unsqueeze(-1) * decayed).sum(-2)
        state = decayed + k[t].unsqueeze(-1) * (beta[t].unsqueeze(-1) * residual).unsqueeze(-2)
        stored = quantizer(state) if t >= 3 else state
        read = stored if t >= 3 and policy.decode.readout == "stored" else state
        expected.append((q[t].unsqueeze(-1) * read).sum(-2) / q.shape[-1] ** 0.5)
        state = stored
        if t == 2 and policy.decode.prefill_state_qdq:
            state = quantizer(state)
    for actual, expected in zip(
        _values_and_grads(output[0], final[0], args, initial),
        _values_and_grads(torch.stack(expected), state, args, initial),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_serving_precision_requires_native_schedule():
    with pytest.raises(ValueError, match="working readout"):
        LinearAttentionDecodeConfig(precision="vllm_0_15")


def test_grid_gate_ste_keeps_gate_gradients_and_changes_trajectory():
    args, state = _inputs()
    output, carry = recurrent_decode(
        *args, config=LinearAttentionDecodeConfig(decay_log_step=0.02), initial_state=state
    )
    grad = torch.autograd.grad(output.square().sum() + carry.reconstruct().square().sum(), args[3])[
        0
    ]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    exact, _ = recurrent_decode(*args, config=LinearAttentionDecodeConfig(), initial_state=state)
    assert not torch.equal(output, exact)
