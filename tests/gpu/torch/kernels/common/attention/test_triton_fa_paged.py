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

import inspect
import math

import pytest
import torch
from conftest import make_qkv, make_varlen_meta

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention, triton_fa
    from modelopt.torch.kernels.quantization.attention.bmm2_qdq import fake_quant_v_onwrite

NATIVE_E4M3_AVAILABLE = TRITON_KERNEL_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
requires_native_e4m3 = pytest.mark.skipif(
    not NATIVE_E4M3_AVAILABLE, reason="Native E4M3 requires compute capability >= 8.9"
)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need triton")
@pytest.mark.parametrize("loader", ["_load_paged_k_tile", "_load_paged_v_tile"])
def test_paged_loaders_widen_page_ids_before_pointer_math(loader):
    source = inspect.getsource(getattr(triton_fa, loader).fn)
    assert "page_global = page_global.to(tl.int64)" in source


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


def _suffix_causal_reference(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale):
    """Reference for chunked prefill where Q is the latest suffix of KV."""
    out = torch.empty_like(q)
    q_start = 0
    kv_start = 0
    head_repeat = num_heads // num_kv_heads
    for q_len, kv_len in zip(q_lens, kv_lens):
        q_b = q[q_start : q_start + q_len].float()
        k_b = k[kv_start : kv_start + kv_len].float().repeat_interleave(head_repeat, dim=1)
        v_b = v[kv_start : kv_start + kv_len].float().repeat_interleave(head_repeat, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_b, k_b) * scale
        prefix_len = kv_len - q_len
        q_pos = torch.arange(q_len, device=q.device) + prefix_len
        kv_pos = torch.arange(kv_len, device=q.device)
        scores = scores.masked_fill(kv_pos[None, :] > q_pos[:, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref = torch.einsum("hqk,khd->qhd", probs, v_b)
        out[q_start : q_start + q_len] = ref.to(q.dtype)
        q_start += q_len
        kv_start += kv_len
    return out


# ---------------------------------------------------------------------------
# Paged KV cache tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestPagedKV:
    """Paged KV cache mode tests — verify paged output matches contiguous."""

    @pytest.mark.parametrize(
        ("page_size", "v_qdq_scale", "match"),
        [(8, 1.0, "page_size"), (16, 0.0, "v_qdq_scale")],
    )
    def test_v_onwrite_validates_page_size_and_scale(self, page_size, v_qdq_scale, match):
        cache = torch.empty(1, 16, 1, 1, device="cuda")
        zeros = torch.zeros(1, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match=match):
            fake_quant_v_onwrite(
                cache,
                zeros.view(1, 1),
                zeros,
                zeros,
                max_new_tokens=1,
                page_size=page_size,
                v_qdq_scale=v_qdq_scale,
            )

    @pytest.mark.parametrize(
        ("seq_len", "seed", "attention_kwargs"),
        [
            pytest.param(128, 42, {}, id="dense"),
            pytest.param(
                256,
                99,
                {"sparsity_n": 2, "sparsity_m": 4, "dense_recent_tokens": 64},
                id="sparse-2-4",
            ),
        ],
    )
    def test_paged_matches_contiguous(self, seq_len, seed, attention_kwargs):
        """Paged dense and sparse prefill match their contiguous counterparts."""
        batch = 2
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)
        total = batch * seq_len

        torch.manual_seed(seed)
        q, k, v = make_qkv(total, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len] * batch)

        # Contiguous reference
        out_contig = attention(
            q, k, v, locs, lens, seq_len, softmax_scale=scale, **attention_kwargs
        )

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
            **attention_kwargs,
        )

        torch.testing.assert_close(out_paged, out_contig, rtol=1e-2, atol=1e-2)

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

    def test_paged_mixed_prefill_decode_sparse_nm_keeps_decode_dense(self):
        """Decode rows stay dense even when batched with sparse prefill rows."""
        q_lens = [64, 1]
        kv_lens = [64, 128]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(35)
        q = torch.randn(sum(q_lens), num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        locs_q, lens_q = make_varlen_meta(q_lens)
        locs_k, lens_k = make_varlen_meta(kv_lens)
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs_k, lens_k, num_kv_heads, head_dim, page_size
        )

        common_kwargs = {
            "is_causal": True,
            "softmax_scale": scale,
            "b_start_loc_k": locs_k,
            "b_seq_len_k": lens_k,
            "max_input_len_k": max(kv_lens),
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "page_size": page_size,
        }
        out_dense = attention(q, k, v, locs_q, lens_q, max(q_lens), **common_kwargs)
        out_sparse = attention(
            q,
            k,
            v,
            locs_q,
            lens_q,
            max(q_lens),
            **common_kwargs,
            sparsity_n=2,
            sparsity_m=4,
            dense_recent_tokens=64,
        )

        decode_start = q_lens[0]
        torch.testing.assert_close(
            out_sparse[decode_start:], out_dense[decode_start:], rtol=1e-2, atol=1e-2
        )

    def test_paged_chunked_prefill_matches_suffix_causal_reference(self):
        """Paged causal mode offsets suffix Q positions into the longer KV span."""
        q_lens = [7, 13]
        kv_lens = [31, 48]
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 16
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(44)
        q = torch.randn(sum(q_lens), num_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

        locs_q, lens_q = make_varlen_meta(q_lens)
        locs_k, lens_k = make_varlen_meta(kv_lens)
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, locs_k, lens_k, num_kv_heads, head_dim, page_size
        )

        out = attention(
            q,
            k,
            v,
            locs_q,
            lens_q,
            max(q_lens),
            is_causal=True,
            softmax_scale=scale,
            b_start_loc_k=locs_k,
            b_seq_len_k=lens_k,
            max_input_len_k=max(kv_lens),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )
        ref = _suffix_causal_reference(q, k, v, q_lens, kv_lens, num_heads, num_kv_heads, scale)

        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    @requires_native_e4m3
    def test_v_cache_finalizes_complete_groups_once(self):
        """Eager prefill and fixed-grid decode bake groups once and leave tails raw."""
        seq_len, head_dim, page_size = 49, 32, 16
        k = torch.zeros(seq_len, 1, head_dim, device="cuda", dtype=torch.float16)
        v = torch.full_like(k, 0.017578125)  # once: 0.015625; twice: 0.01171875
        locs, lens = make_varlen_meta([seq_len])
        _, raw, block_table = _scatter_to_paged_cache(k, v, locs, lens, 1, head_dim, page_size)
        baked = raw.clone()
        fake_quant_v_onwrite(
            baked,
            block_table,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([32], device="cuda", dtype=torch.int32),
            max_new_tokens=32,
            page_size=page_size,
        )
        after_prefill = baked.clone()
        fake_quant_v_onwrite(
            baked,
            block_table,
            torch.tensor([32], device="cuda", dtype=torch.int32),
            torch.tensor([48], device="cuda", dtype=torch.int32),
            max_new_tokens=1,
            page_size=page_size,
        )
        torch.testing.assert_close(baked[:2], after_prefill[:2], rtol=0, atol=0)
        assert torch.all(baked[:3] == 0.015625)
        torch.testing.assert_close(baked[3, 0], raw[3, 0], rtol=0, atol=0)

    @requires_native_e4m3
    def test_v_cache_matches_independent_signed_key_axis_oracle(self):
        page_size, num_kv_heads, head_dim = 8, 2, 4
        logical = (
            torch.arange(16 * num_kv_heads * head_dim, device="cuda", dtype=torch.float16)
            .reshape(16, num_kv_heads, head_dim)
            .sub_(61)
            .div_(13)
        )
        cache = torch.full(
            (3, page_size, num_kv_heads, head_dim),
            99.0,
            device="cuda",
            dtype=logical.dtype,
        )
        block_table = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
        cache[2], cache[0] = logical[:page_size], logical[page_size:]
        unused_page = cache[1].clone()

        fake_quant_v_onwrite(
            cache,
            block_table,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.tensor([16], device="cuda", dtype=torch.int32),
            max_new_tokens=16,
            page_size=page_size,
        )
        key_last = logical.permute(1, 2, 0).contiguous()
        q, scale, double_scale = NVFP4QTensor.quantize(
            key_last,
            16,
            weights_scaling_factor_2=torch.tensor(1.0, device="cuda"),
            try_tensorrt=False,
        )
        expected = q.dequantize(
            dtype=logical.dtype,
            scale=scale,
            double_scale=double_scale,
            block_sizes={-1: 16},
        ).permute(2, 0, 1)
        actual = torch.cat((cache[2], cache[0]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache[1], unused_page, rtol=0, atol=0)

    @requires_native_e4m3
    def test_baked_prefix_and_raw_tail_match_full_onread(self):
        """The paged Triton chunked-prefill path handles a baked prefix and raw tail."""
        q_len, seq_len, num_heads, head_dim, page_size = 7, 17, 2, 32, 16
        q = torch.zeros(q_len, num_heads, head_dim, device="cuda", dtype=torch.float32)
        k = torch.zeros(seq_len, 1, head_dim, device="cuda", dtype=torch.float16)
        v = torch.full_like(k, 0.017578125)
        q_locs, q_lens = make_varlen_meta([q_len])
        locs, lens = make_varlen_meta([seq_len])
        k_cache, raw, block_table = _scatter_to_paged_cache(
            k, v, locs, lens, 1, head_dim, page_size
        )
        common = {
            "is_causal": False,
            "b_seq_len_k": lens,
            "max_input_len_k": seq_len,
            "k_cache": k_cache,
            "block_table": block_table,
            "page_size": page_size,
            "p_qdq": "nvfp4",
            "v_qdq": "nvfp4",
        }
        out_onread = attention(q, k, v, q_locs, q_lens, q_len, v_cache=raw, **common)
        baked = raw.clone()
        fake_quant_v_onwrite(
            baked,
            block_table,
            locs,
            torch.tensor([16], device="cuda", dtype=torch.int32),
            max_new_tokens=16,
        )
        out_baked = attention(
            q,
            k,
            v,
            q_locs,
            q_lens,
            q_len,
            v_cache=baked,
            v_cache_quantized=True,
            **common,
        )
        torch.testing.assert_close(out_baked, out_onread, rtol=1e-4, atol=1e-5)

    @requires_native_e4m3
    def test_p_qdq_uses_cache_dtype_with_fp32_dummy_tensors(self):
        seq_len, head_dim, page_size = 128, 16, 16
        scale = head_dim**-0.5
        boundary_p = 0.1248
        q = torch.zeros(1, 1, head_dim, device="cuda", dtype=torch.float32)
        q[..., 0] = -math.log2(boundary_p) / (scale * triton_fa.LOG2E)
        k = torch.zeros(seq_len, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        k[1:, 0, 0] = -1.0
        v = torch.zeros_like(k)
        v[1:16, 0, 0] = 1.0
        q_locs, q_lens = make_varlen_meta([1])
        kv_locs, kv_lens = make_varlen_meta([seq_len])
        k_cache, v_cache, block_table = _scatter_to_paged_cache(
            k, v, kv_locs, kv_lens, 1, head_dim, page_size
        )
        common = {
            "is_causal": False,
            "softmax_scale": scale,
            "b_start_loc_k": kv_locs,
            "b_seq_len_k": kv_lens,
            "max_input_len_k": seq_len,
            "p_qdq": "nvfp4",
        }
        contiguous = attention(q, k, v, q_locs, q_lens, 1, **common)
        dummy = torch.empty(0, 1, head_dim, device="cuda", dtype=torch.float32)
        paged = attention(
            q,
            dummy,
            dummy,
            q_locs,
            q_lens,
            1,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
            **common,
        )

        torch.testing.assert_close(paged, contiguous, rtol=2e-3, atol=5e-4)
