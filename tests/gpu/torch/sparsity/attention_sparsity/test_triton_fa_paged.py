# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU tests for paged KV cache mode of the Triton flash attention kernel."""

import pytest
import torch
from conftest import make_qkv, make_varlen_meta

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels import attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scatter_to_paged_cache(k, v, b_start_loc, b_seq_len, num_kv_heads, head_dim, page_size):
    """Scatter contiguous K/V into a paged KV cache + block table.

    Args:
        k: [total_kv, num_kv_heads, head_dim] contiguous keys
        v: [total_kv, num_kv_heads, head_dim] contiguous values
        b_start_loc: [batch] start offsets
        b_seq_len: [batch] sequence lengths
        num_kv_heads: number of KV heads
        head_dim: head dimension
        page_size: tokens per page

    Returns:
        k_cache: [num_blocks, page_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, page_size, num_kv_heads, head_dim]
        block_table: [batch, max_blocks_per_seq]
    """
    batch = b_seq_len.shape[0]
    device = k.device
    dtype = k.dtype

    # Calculate blocks needed per sequence
    blocks_per_seq = []
    for b in range(batch):
        slen = int(b_seq_len[b].item())
        blocks_per_seq.append((slen + page_size - 1) // page_size)

    max_blocks = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq)

    k_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    block_table = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)

    global_block = 0
    for b in range(batch):
        start = int(b_start_loc[b].item())
        slen = int(b_seq_len[b].item())
        for blk in range(blocks_per_seq[b]):
            block_table[b, blk] = global_block
            tok_start = blk * page_size
            tok_end = min(tok_start + page_size, slen)
            n_toks = tok_end - tok_start
            k_cache[global_block, :n_toks] = k[start + tok_start : start + tok_end]
            v_cache[global_block, :n_toks] = v[start + tok_start : start + tok_end]
            global_block += 1

    return k_cache, v_cache, block_table


# ---------------------------------------------------------------------------
# Paged KV cache tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestPagedKV:
    """Paged KV cache mode tests — verify paged output matches contiguous."""

    def test_paged_matches_contiguous(self):
        """Paged mode produces same output as contiguous mode with identical data."""
        batch = 2
        seq_len = 128
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total = batch * seq_len

        torch.manual_seed(42)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)

        # Contiguous reference
        out_contig = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale)

        # Build paged cache from the same K/V
        locs_k, lens_k = locs, lens
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs_k, lens_k, num_kv_heads, head_dim, page_size
        )

        # Paged mode
        out_paged = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            b_start_loc_k=locs_k,
            b_seq_len_k=lens_k,
            max_input_len_k=seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        torch.testing.assert_close(out_paged, out_contig, rtol=1e-2, atol=1e-2)

    def test_paged_no_nan(self):
        """Paged mode output is finite."""
        batch = 2
        seq_len = 256
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total = batch * seq_len

        torch.manual_seed(55)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)

        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs, lens, num_kv_heads, head_dim, page_size
        )

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            b_seq_len_k=lens,
            max_input_len_k=seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        assert not torch.isnan(out).any(), "NaN in paged output"
        assert not torch.isinf(out).any(), "Inf in paged output"

    def test_paged_variable_length(self):
        """Paged mode works with variable-length sequences."""
        seq_lens = [64, 128]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total = sum(seq_lens)

        torch.manual_seed(77)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta(seq_lens)

        # Contiguous reference
        out_contig = attention(q, k, v, locs, lens, max(seq_lens), softmax_scale=scale)

        # Paged
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs, lens, num_kv_heads, head_dim, page_size
        )

        out_paged = attention(
            q,
            k,
            v,
            locs,
            lens,
            max(seq_lens),
            softmax_scale=scale,
            b_seq_len_k=lens,
            max_input_len_k=max(seq_lens),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        torch.testing.assert_close(out_paged, out_contig, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("page_size", [16, 32, 64])
    def test_paged_different_page_sizes(self, page_size):
        """Paged mode works with different page sizes."""
        batch = 2
        seq_len = 128
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)
        total = batch * seq_len

        torch.manual_seed(88)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)

        out_contig = attention(q, k, v, locs, lens, seq_len, softmax_scale=scale)

        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs, lens, num_kv_heads, head_dim, page_size
        )

        out_paged = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            b_seq_len_k=lens,
            max_input_len_k=seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        torch.testing.assert_close(out_paged, out_contig, rtol=1e-2, atol=1e-2)

    def test_paged_with_sparsity(self):
        """Paged mode works with N:M sparsity enabled."""
        batch = 2
        seq_len = 256
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total = batch * seq_len

        torch.manual_seed(99)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)

        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs, lens, num_kv_heads, head_dim, page_size
        )

        out_paged_sparse = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            softmax_scale=scale,
            b_seq_len_k=lens,
            max_input_len_k=seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
            sparsity_n=2,
            sparsity_m=4,
        )

        assert not torch.isnan(out_paged_sparse).any(), "NaN in paged + sparse output"
        assert not torch.isinf(out_paged_sparse).any(), "Inf in paged + sparse output"
        assert out_paged_sparse.shape == q.shape

    def test_paged_decode(self):
        """Paged mode works for decode (single Q token, long KV context)."""
        batch = 2
        seq_lens_k = [64, 128]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total_kv = sum(seq_lens_k)

        torch.manual_seed(33)
        q_flat = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
        k_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v_flat = torch.randn(total_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        b_start_loc_q = torch.arange(batch, device="cuda", dtype=torch.int32)
        b_seq_len_q = torch.ones(batch, device="cuda", dtype=torch.int32)
        cumsum = [0]
        for sl in seq_lens_k:
            cumsum.append(cumsum[-1] + sl)
        b_start_loc_k = torch.tensor(cumsum[:-1], device="cuda", dtype=torch.int32)
        b_seq_len_k = torch.tensor(seq_lens_k, device="cuda", dtype=torch.int32)

        # Build paged cache
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k_flat, v_flat, b_start_loc_k, b_seq_len_k, num_kv_heads, head_dim, page_size
        )

        out = attention(
            q_flat,
            k_flat,
            v_flat,
            b_start_loc_q,
            b_seq_len_q,
            1,
            is_causal=False,
            softmax_scale=scale,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max(seq_lens_k),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        assert out.shape == q_flat.shape
        assert not torch.isnan(out).any(), "NaN in paged decode output"
