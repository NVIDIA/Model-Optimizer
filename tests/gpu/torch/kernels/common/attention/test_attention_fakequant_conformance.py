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

"""Conformance tests: the on-write V bake reproduces an INDEPENDENT canonical NVFP4 oracle.

The prior V tests (test_decode_attention.py) compare the bake against itself (incremental vs
one-shot) or against attention *outputs*, so a shared scale/rounding bug in the bake would pass.
Here the reference uses ``NVFP4QTensor.quantize/dequantize`` on transposed V, so its last-axis
block-16 contract becomes V's key axis. It never calls the attention kernel's V-QDQ helper. The
tests check that complete groups stay QDQ while the partial group remains pristine for
in-kernel QDQ on read.
"""

import pytest
import torch

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention.decode_attention import fake_quant_v_onwrite

_NVFP4_BLOCK_SIZE = 16

# NVFP4's E4M3 block scale uses ``tl.float8e4nv``, which requires SM89+ (Ada/Hopper/Blackwell).
_HAS_SM89 = (
    TRITON_KERNEL_AVAILABLE
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9)
)
pytestmark = pytest.mark.skipif(not _HAS_SM89, reason="NVFP4 V bake uses tl.float8e4nv (sm_89+)")


def _nvfp4_v_oracle(v: torch.Tensor, global_scale: float = 1.0) -> torch.Tensor:
    """Canonical NVFP4 QDQ of ``[keys, head_dim]`` with block-16 along keys."""
    seq_len = v.shape[0]
    padded_len = ((seq_len + _NVFP4_BLOCK_SIZE - 1) // _NVFP4_BLOCK_SIZE) * _NVFP4_BLOCK_SIZE
    padded = torch.zeros(padded_len, v.shape[1], device=v.device, dtype=v.dtype)
    padded[:seq_len] = v
    transposed = padded.T.contiguous()
    q, scale, double_scale = NVFP4QTensor.quantize(
        transposed,
        _NVFP4_BLOCK_SIZE,
        weights_scaling_factor_2=torch.tensor(global_scale, device=v.device),
        try_tensorrt=False,
    )
    dequant = q.dequantize(
        dtype=v.dtype,
        scale=scale,
        double_scale=double_scale,
        block_sizes={-1: _NVFP4_BLOCK_SIZE},
    )
    return dequant.T[:seq_len].contiguous()


def _make_paged_v(v: torch.Tensor, page_size: int):
    """Scatter ``v`` [seq, head_dim] into a single-request paged cache [num_blocks, page, 1, D]."""
    seq, head_dim = v.shape
    num_blocks = (seq + page_size - 1) // page_size
    cache = torch.zeros(num_blocks, page_size, 1, head_dim, device=v.device, dtype=v.dtype)
    block_table = torch.arange(num_blocks, device=v.device, dtype=torch.int32).view(1, num_blocks)
    for blk in range(num_blocks):
        start, end = blk * page_size, min((blk + 1) * page_size, seq)
        cache[blk, : end - start, 0] = v[start:end]
    return cache, block_table


def _gather_v(cache: torch.Tensor, block_table: torch.Tensor, seq: int) -> torch.Tensor:
    """Reconstruct ``[seq, head_dim]`` from the paged cache via the block table (reverse of scatter)."""
    page_size, head_dim = cache.shape[1], cache.shape[3]
    out = torch.empty(seq, head_dim, device=cache.device, dtype=cache.dtype)
    for blk in range(block_table.shape[1]):
        start, end = blk * page_size, min((blk + 1) * page_size, seq)
        if start >= seq:
            break
        out[start:end] = cache[int(block_table[0, blk]), : end - start, 0]
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
@pytest.mark.parametrize("seq_len", [15, 16, 17, 127, 128, 129, 255, 256, 257])
def test_baked_v_matches_independent_oracle(seq_len, page_size, dtype):
    """Persistent groups match canonical QDQ and the partial group remains pristine."""
    torch.manual_seed(seq_len * 100 + page_size)
    head_dim = 128
    v = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    v[:, 0] = 0.017578125  # a block where QDQ(QDQ(V)) differs from QDQ(V)
    cache, block_table = _make_paged_v(v, page_size)
    raw = cache.clone()

    baked_len = (seq_len // _NVFP4_BLOCK_SIZE) * _NVFP4_BLOCK_SIZE
    v_lo = torch.zeros(1, device="cuda", dtype=torch.int32)
    v_hi = torch.tensor([baked_len], device="cuda", dtype=torch.int32)
    fake_quant_v_onwrite(cache, block_table, v_lo, v_hi, page_size=page_size)  # global scale 1.0

    if baked_len > 0:
        baked = _gather_v(cache, block_table, baked_len)
        oracle = _nvfp4_v_oracle(v[:baked_len], global_scale=1.0)
        torch.testing.assert_close(baked, oracle, rtol=0, atol=0)

    if baked_len < seq_len:
        trailing_baked = _gather_v(cache, block_table, seq_len)[baked_len:]
        trailing_raw = _gather_v(raw, block_table, seq_len)[baked_len:]
        torch.testing.assert_close(trailing_baked, trailing_raw, rtol=0, atol=0)


def test_bake_is_non_trivial():
    """Sanity: the bake actually quantizes (baked != raw), so a no-op bake cannot pass silently."""
    torch.manual_seed(7)
    v = torch.randn(_NVFP4_BLOCK_SIZE, 128, device="cuda", dtype=torch.float16)
    cache, block_table = _make_paged_v(v, page_size=16)
    raw = cache.clone()
    fake_quant_v_onwrite(
        cache,
        block_table,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([_NVFP4_BLOCK_SIZE], device="cuda", dtype=torch.int32),
        page_size=16,
    )
    assert not torch.equal(cache, raw)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_baked_v_matches_calibrated_global_scale(dtype):
    """A calibrated per-tensor scale uses the same numerical contract as the oracle."""
    torch.manual_seed(19)
    v = torch.randn(16, 128, device="cuda", dtype=dtype)
    cache, block_table = _make_paged_v(v, page_size=16)
    global_scale = 0.125
    fake_quant_v_onwrite(
        cache,
        block_table,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([16], device="cuda", dtype=torch.int32),
        page_size=16,
        v_qdq_scale=global_scale,
    )
    torch.testing.assert_close(
        _gather_v(cache, block_table, 16),
        _nvfp4_v_oracle(v, global_scale=global_scale),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_oracle_detects_double_qdq(dtype):
    """The conformance fixture must fail if a completed group is quantized twice."""
    raw = torch.full((16, 128), 0.017578125, device="cuda", dtype=dtype)
    once = _nvfp4_v_oracle(raw)
    twice = _nvfp4_v_oracle(once)
    assert once[0, 0] == 0.015625
    assert twice[0, 0] == 0.01171875
    assert not torch.equal(once, twice)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
