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

"""GPU tests for the vendored chunked GatedDeltaNet kernel with in-kernel FP8 state QDQ."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

fla = pytest.importorskip("fla.ops.gated_delta_rule")
vendored = pytest.importorskip(
    "modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule",
    reason="the vendored kernel needs flash-linear-attention >= 0.5.1 and Triton",
)
if not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)):
    pytest.skip(
        "Native E4M3 needs a CUDA device with compute capability >= 8.9", allow_module_level=True
    )

state_qdq_chunk_gated_delta_rule = vendored.chunk_gated_delta_rule

CHUNK = 64


def make_inputs(batch=2, seq_len=4 * CHUNK, heads=2, k_dim=64, v_dim=128, dtype=torch.float32):
    torch.manual_seed(0)
    kw = {"device": "cuda", "dtype": dtype}
    q = F.normalize(torch.randn(batch, seq_len, heads, k_dim, **kw), dim=-1)
    k = F.normalize(torch.randn(batch, seq_len, heads, k_dim, **kw), dim=-1)
    v = torch.randn(batch, seq_len, heads, v_dim, **kw)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, **kw))
    beta = torch.rand(batch, seq_len, heads, **kw)
    return q, k, v, g, beta


def per_head_fp8_qdq(state):
    """FP8 E4M3 quant-dequant of a ``[N, H, K, V]`` state, one scale per (N, H), kernel arithmetic."""
    amax = state.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (state / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale


def test_state_qdq_off_matches_fla():
    q, k, v, g, beta = make_inputs()
    expected, expected_state = fla.chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    out, state = state_qdq_chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, state_qdq=0
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_state_qdq_matches_per_chunk_reference(with_initial_state):
    """Quantizing inside the kernel (one scale per head, ``state_qdq_block_v=V``) equals quantizing
    the state between per-chunk kernel calls, including a provided initial state."""
    q, k, v, g, beta = make_inputs()
    initial_state = torch.randn(2, 2, 64, 128, device="cuda") if with_initial_state else None
    outputs, state = [], initial_state
    for s in range(0, q.shape[1], CHUNK):
        if state is not None:
            state = per_head_fp8_qdq(state)
        o, state = fla.chunk_gated_delta_rule(
            q[:, s : s + CHUNK],
            k[:, s : s + CHUNK],
            v[:, s : s + CHUNK],
            g[:, s : s + CHUNK],
            beta[:, s : s + CHUNK],
            initial_state=state,
            output_final_state=True,
        )
        outputs.append(o)
    expected, expected_state = torch.cat(outputs, dim=1), per_head_fp8_qdq(state)

    out, final_state = state_qdq_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        state_qdq=1,
        state_qdq_block_v=128,
    )
    unquantized, _ = fla.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=initial_state)

    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=2e-3, atol=2e-3)
    assert (out - unquantized).abs().max() > (out - expected).abs().max(), (
        "FP8 state quantization must move the output away from the unquantized kernel"
    )


def test_state_qdq_block_v_sets_granularity():
    """The default 64-column tile gives two scales per 128-wide head; a 128 tile gives one."""
    q, k, v, g, beta = make_inputs()
    default, _ = state_qdq_chunk_gated_delta_rule(q, k, v, g, beta, state_qdq=1)
    per_head, _ = state_qdq_chunk_gated_delta_rule(
        q, k, v, g, beta, state_qdq=1, state_qdq_block_v=128
    )
    assert torch.isfinite(default).all()
    assert not torch.equal(default, per_head)
    for bad in (48, 256):
        with pytest.raises(ValueError, match="power of two"):
            state_qdq_chunk_gated_delta_rule(q, k, v, g, beta, state_qdq=1, state_qdq_block_v=bad)


def test_state_qdq_backward_is_straight_through():
    """Training runs through the quantized kernel; gradients reach every input and the state."""
    q, k, v, g, beta = (x.clone().requires_grad_() for x in make_inputs(dtype=torch.bfloat16))
    initial_state = torch.randn(2, 2, 64, 128, device="cuda", requires_grad=True)
    out, final_state = state_qdq_chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True, state_qdq=1
    )
    (out.float().square().sum() + final_state.square().sum()).backward()
    for tensor in (q, k, v, g, beta, initial_state):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_state_qdq_varlen():
    """Packed sequences quantize each sequence's own state."""
    q, k, v, g, beta = make_inputs(batch=1, seq_len=6 * CHUNK)
    cu_seqlens = torch.tensor([0, 2 * CHUNK, 6 * CHUNK], device="cuda")
    out, final_state = state_qdq_chunk_gated_delta_rule(
        q, k, v, g, beta, cu_seqlens=cu_seqlens, output_final_state=True, state_qdq=1
    )
    first, first_state = state_qdq_chunk_gated_delta_rule(
        *(x[:, : 2 * CHUNK] for x in (q, k, v, g, beta)), output_final_state=True, state_qdq=1
    )
    torch.testing.assert_close(out[:, : 2 * CHUNK], first, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(final_state[:1], first_state, rtol=1e-4, atol=1e-4)


def test_w_quantizer_fake_quantizes_the_state_matmul_operand():
    """``w_quantizer`` is applied once to the WY tensor ``w`` before its matmul with the state; a
    ModelOpt TensorQuantizer with a dynamic per-token FP8 scale works as the callable."""
    w_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), type="dynamic", axis=(0, 1, 2))
    ).cuda()
    q, k, v, g, beta = (x.clone().requires_grad_() for x in make_inputs())
    state_only, _ = state_qdq_chunk_gated_delta_rule(q, k, v, g, beta, state_qdq=1)
    out, final_state = state_qdq_chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, state_qdq=1, w_quantizer=w_quantizer
    )
    assert torch.isfinite(out).all() and not torch.equal(out, state_only)
    (out.square().sum() + final_state.square().sum()).backward()
    for tensor in (q, k, v, g, beta):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    with pytest.raises(TypeError, match="w_quantizer"):
        state_qdq_chunk_gated_delta_rule(q, k, v, g, beta, w_quantizer=1)
