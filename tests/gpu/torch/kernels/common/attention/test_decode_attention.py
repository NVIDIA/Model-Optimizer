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

"""GPU tests for the paged decode attention kernel (with optional skip-softmax)."""

import pytest
import torch

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention.decode_attention import attention_decode


def _paged_cache(k, v, seq_lens, page_size):
    """[B, KVH, S, D] K/V -> paged [num_blocks, page_size, KVH, D] + block_table."""
    batch, num_kv, _seq, head_dim = k.shape
    blocks = [(int(seq_lens[b].item()) + page_size - 1) // page_size for b in range(batch)]
    k_cache = torch.zeros(sum(blocks), page_size, num_kv, head_dim, device=k.device, dtype=k.dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(batch, max(blocks), device=k.device, dtype=torch.int32)
    g = 0
    for b in range(batch):
        sl = int(seq_lens[b].item())
        kb, vb = k[b].transpose(0, 1), v[b].transpose(0, 1)  # [S, KVH, D]
        for blk in range(blocks[b]):
            block_table[b, blk] = g
            ts, te = blk * page_size, min((blk + 1) * page_size, sl)
            k_cache[g, : te - ts] = kb[ts:te]
            v_cache[g, : te - ts] = vb[ts:te]
            g += 1
    return k_cache, v_cache, block_table


def _dense_decode(q, k, v, scale):
    """Reference decode attention. q [B,H,D]; k,v [B,KVH,S,D] (GQA)."""
    num_q, num_kv = q.shape[1], k.shape[1]
    kk = k.repeat_interleave(num_q // num_kv, dim=1)
    vv = v.repeat_interleave(num_q // num_kv, dim=1)
    scores = torch.matmul(q.unsqueeze(2), kk.transpose(-2, -1)).squeeze(2) * scale  # [B,H,S]
    p = scores.float().softmax(dim=-1).to(v.dtype)
    return torch.matmul(p.unsqueeze(2), vv).squeeze(2)  # [B,H,D]


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestDecodeAttention:
    def _inputs(self, batch, num_q_heads, num_kv_heads, seq_k, head_dim, seed=0):
        torch.manual_seed(seed)
        q = torch.randn(batch, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, num_kv_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, num_kv_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        seq_lens = torch.full((batch,), seq_k, device="cuda", dtype=torch.int32)
        return q, k, v, seq_lens

    @pytest.mark.parametrize(("num_q_heads", "num_kv_heads"), [(4, 4), (8, 2)])
    def test_matches_dense_no_skip(self, num_q_heads, num_kv_heads):
        """Without skipping, the kernel computes exact dense decode attention."""
        batch, seq_k, head_dim, page_size = 3, 500, 64, 16
        scale = 1.0 / (head_dim**0.5)
        q, k, v, seq_lens = self._inputs(batch, num_q_heads, num_kv_heads, seq_k, head_dim)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q, k_cache, v_cache, block_table, seq_lens, softmax_scale=scale, page_size=page_size
        )
        torch.testing.assert_close(out, _dense_decode(q, k, v, scale), rtol=5e-3, atol=5e-3)

    def test_tiny_threshold_matches_dense(self):
        """A near-zero lambda skips almost nothing, so output stays close to dense."""
        batch, seq_k, head_dim, page_size = 2, 384, 64, 16
        scale = 1.0 / (head_dim**0.5)
        q, k, v, seq_lens = self._inputs(batch, 4, 2, seq_k, head_dim, seed=1)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            skip_softmax_threshold=2**-20,
            page_size=page_size,
        )
        torch.testing.assert_close(out, _dense_decode(q, k, v, scale), rtol=1e-2, atol=1e-2)

    def test_sink_skips_most_tiles(self):
        """A dominant sink at position 0 makes distant tiles negligible -> skipped."""
        batch, seq_k, head_dim, page_size = 1, 2048, 64, 16
        scale = 1.0 / (head_dim**0.5)
        q = torch.ones(batch, 4, head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros(batch, 4, seq_k, head_dim, device="cuda", dtype=torch.float16)
        k[:, :, 0] = 20.0  # sink dominates every query
        v = torch.randn(batch, 4, seq_k, head_dim, device="cuda", dtype=torch.float16)
        seq_lens = torch.full((batch,), seq_k, device="cuda", dtype=torch.int32)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            skip_softmax_threshold=0.1,
            page_size=page_size,
            num_kv_splits=1,  # single split => global prefix max => maximal skipping
            measure_sparsity=True,
        )
        # Sink => the vast majority of the 2048/128 = 16 tiles/head are skippable.
        total, skipped = out._sparsity_total, out._sparsity_skipped
        assert total == 4 * (seq_k // 128), (total, skipped)
        assert skipped / total > 0.8, (skipped, total)
        # Output still tracks the (sink-dominated) dense result.
        torch.testing.assert_close(out, _dense_decode(q, k, v, scale), rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8, 16])
    def test_split_k_matches_dense(self, num_kv_splits):
        """Split-K combine is numerically exact regardless of the split count."""
        batch, seq_k, head_dim, page_size = 2, 1000, 64, 16
        scale = 1.0 / (head_dim**0.5)
        q, k, v, seq_lens = self._inputs(batch, 8, 2, seq_k, head_dim, seed=3)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            page_size=page_size,
            num_kv_splits=num_kv_splits,
        )
        torch.testing.assert_close(out, _dense_decode(q, k, v, scale), rtol=5e-3, atol=5e-3)

    @pytest.mark.parametrize("num_kv_splits", [2, 4])
    def test_split_k_with_tiny_threshold_matches_dense(self, num_kv_splits):
        """Skip + split-K compose: a near-zero lambda still matches dense."""
        batch, seq_k, head_dim, page_size = 2, 768, 64, 16
        scale = 1.0 / (head_dim**0.5)
        q, k, v, seq_lens = self._inputs(batch, 4, 2, seq_k, head_dim, seed=4)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            softmax_scale=scale,
            skip_softmax_threshold=2**-20,
            page_size=page_size,
            num_kv_splits=num_kv_splits,
        )
        torch.testing.assert_close(out, _dense_decode(q, k, v, scale), rtol=1e-2, atol=1e-2)

    def test_varlen_lengths(self):
        """Per-request KV lengths (non-uniform, non-page-aligned) are handled."""
        batch, num_q_heads, num_kv_heads, head_dim, page_size = 3, 4, 2, 64, 16
        scale = 1.0 / (head_dim**0.5)
        seq_k = 600
        q, k, v, _ = self._inputs(batch, num_q_heads, num_kv_heads, seq_k, head_dim, seed=2)
        seq_lens = torch.tensor([130, 511, 600], device="cuda", dtype=torch.int32)
        k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens, page_size)

        out = attention_decode(
            q, k_cache, v_cache, block_table, seq_lens, softmax_scale=scale, page_size=page_size
        )
        # Reference must honor each request's own KV length.
        for b in range(batch):
            sl = int(seq_lens[b].item())
            ref = _dense_decode(q[b : b + 1], k[b : b + 1, :, :sl], v[b : b + 1, :, :sl], scale)
            torch.testing.assert_close(out[b : b + 1], ref, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
