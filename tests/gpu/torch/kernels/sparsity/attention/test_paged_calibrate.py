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

"""Paged-cache calibration kernel tests and the calibration/serving tile contract."""

import pytest
import torch
from conftest import make_qkv, make_varlen_meta

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention import attention
    from modelopt.torch.kernels.sparsity.attention.calibrate import attention_calibrate

pytestmark = pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")

TRIALS = [1e-3, 1e-2, 1e-1, 3e-1]


def _pack_paged(k, v, page_size, *, layout="NHD", shuffle=True, num_spare_blocks=7):
    """Pack one sequence's contiguous K/V into a (optionally shuffled) paged cache."""
    seq = k.shape[0]
    num_blocks = (seq + page_size - 1) // page_size
    order = torch.randperm(num_blocks) if shuffle else torch.arange(num_blocks)
    shape = (num_blocks + num_spare_blocks, page_size, k.shape[1], k.shape[2])
    k_cache = k.new_zeros(shape)
    if layout == "HND":
        k_cache = k.new_zeros(shape[0], shape[2], shape[1], shape[3]).permute(0, 2, 1, 3)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(1, num_blocks, device=k.device, dtype=torch.int32)
    for i in range(num_blocks):
        page = int(order[i]) + num_spare_blocks  # keep low pages unused
        ts, te = i * page_size, min((i + 1) * page_size, seq)
        k_cache[page, : te - ts] = k[ts:te]
        v_cache[page, : te - ts] = v[ts:te]
        block_table[0, i] = page
    return k_cache, v_cache, block_table


class TestPagedCalibrate:
    @pytest.mark.parametrize("layout", ["NHD", "HND"])
    @pytest.mark.parametrize("seq_len", [256, 300, 512])  # 300: non-128-aligned padding
    def test_paged_matches_contiguous_prefill(self, seq_len, layout):
        """Paged and contiguous calibration agree exactly on counters and output."""
        torch.manual_seed(0)
        num_heads, num_kv_heads, head_dim, page_size = 8, 2, 64, 16
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16)
        locs, lens = make_varlen_meta([seq_len])

        out_ref, counters_ref = attention_calibrate(
            q, k, v, locs, lens, seq_len, is_causal=True, threshold_trials=TRIALS
        )

        k_cache, v_cache, block_table = _pack_paged(k, v, page_size, layout=layout)
        k_dummy = torch.empty(0, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        out_paged, counters_paged = attention_calibrate(
            q,
            k_dummy,
            k_dummy,
            locs,
            lens,
            seq_len,
            is_causal=True,
            threshold_trials=TRIALS,
            b_seq_len_k=lens,
            max_input_len_k=seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        assert torch.equal(counters_ref, counters_paged)
        torch.testing.assert_close(out_paged, out_ref, rtol=1e-3, atol=1e-3)

    def test_paged_decode_measures_full_cache(self):
        """A one-row decode query measures every KV tile of the paged cache."""
        torch.manual_seed(1)
        num_heads, num_kv_heads, head_dim, page_size = 8, 2, 64, 16
        ctx = 384
        q, k, v = make_qkv(ctx, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16)
        k_cache, v_cache, block_table = _pack_paged(k, v, page_size)
        k_dummy = torch.empty(0, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        locs = torch.zeros(1, device="cuda", dtype=torch.int32)

        _, counters = attention_calibrate(
            q[:1],
            k_dummy,
            k_dummy,
            locs,
            torch.ones(1, device="cuda", dtype=torch.int32),
            1,
            is_causal=False,
            threshold_trials=TRIALS,
            b_seq_len_k=torch.tensor([ctx], device="cuda", dtype=torch.int32),
            max_input_len_k=ctx,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )

        num_kv_tiles = -(-ctx // 128)
        assert counters[:, 0].tolist() == [num_heads * num_kv_tiles] * len(TRIALS)

    def test_high_block_id_pointer_arithmetic(self):
        """Block IDs whose int32 byte offsets would wrap still read correctly."""
        num_kv_heads, head_dim, page_size = 2, 64, 16
        block_elems = page_size * num_kv_heads * head_dim
        # Smallest block ID whose element offset exceeds int32. V aliases the K
        # cache storage (same values on both operands), halving the allocation.
        high_block = (2**31) // block_elems + 1
        bytes_needed = (high_block + 1) * block_elems * 2  # one shared K/V cache, bf16
        free, _ = torch.cuda.mem_get_info()
        if free < bytes_needed + (2 << 30):
            pytest.skip(f"needs ~{bytes_needed / 2**30:.1f} GiB free GPU memory")

        torch.manual_seed(2)
        num_heads = 4
        q, k, _ = make_qkv(page_size, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16)
        k_cache = torch.zeros(
            high_block + 1, page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        v_cache = k_cache  # alias: V reads the same storage (and the same values)
        k_cache[high_block] = k
        block_table = torch.tensor([[high_block]], device="cuda", dtype=torch.int32)
        locs, lens = make_varlen_meta([page_size])

        out_ref, counters_ref = attention_calibrate(
            q, k, k, locs, lens, page_size, is_causal=True, threshold_trials=TRIALS
        )
        k_dummy = torch.empty(0, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        out_paged, counters_paged = attention_calibrate(
            q,
            k_dummy,
            k_dummy,
            locs,
            lens,
            page_size,
            is_causal=True,
            threshold_trials=TRIALS,
            b_seq_len_k=lens,
            max_input_len_k=page_size,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            page_size=page_size,
        )
        del k_cache, v_cache

        assert torch.equal(counters_ref, counters_paged)
        torch.testing.assert_close(out_paged, out_ref, rtol=1e-3, atol=1e-3)


class TestCalibrationServingTileContract:
    """Active skip launches and calibration must count identically (same tiles)."""

    def _contrasty_qkv(self, seq_len, num_heads, num_kv_heads, head_dim):
        """K with a dominant head-of-sequence so later tiles are skippable."""
        torch.manual_seed(3)
        q, k, v = make_qkv(seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16)
        k = k * 0.05
        k[:32] = k[:32] * 600.0  # first tile dominates the running max by >> log2(threshold)
        return q, k, v

    @pytest.mark.parametrize("threshold", [1e-3, 1e-2])
    def test_serve_skip_counts_equal_calibrate_counts(self, threshold):
        seq_len, num_heads, num_kv_heads, head_dim = 512, 8, 2, 64
        q, k, v = self._contrasty_qkv(seq_len, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len])
        scale = 1.0 / (head_dim**0.5)

        _, counters = attention_calibrate(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=True,
            softmax_scale=scale,
            threshold_trials=[threshold],
        )

        out = attention(
            q,
            k,
            v,
            locs,
            lens,
            seq_len,
            is_causal=True,
            softmax_scale=scale,
            skip_softmax_threshold=threshold,
            measure_sparsity=True,
        )

        calib_total, calib_skipped = int(counters[0, 0]), int(counters[0, 1])
        assert calib_skipped > 0, "test data must produce skippable tiles"
        # Same 128x128 tile geometry and same prefix-max criterion => the serve
        # kernel must skip exactly the tiles calibration predicted.
        assert out._sparsity_total == calib_total
        assert out._sparsity_skipped == calib_skipped

    @pytest.mark.parametrize("qdq_kw", [{"p_qdq": "nvfp4"}, {"v_qdq": "nvfp4", "v_qdq_amax": 1.0}])
    def test_skip_rejects_pv_qdq(self, qdq_kw):
        """Active skip rejects P/V QDQ: quantized operands break the calibrated contract."""
        seq_len, num_heads, num_kv_heads, head_dim = 256, 4, 2, 64
        q, k, v = self._contrasty_qkv(seq_len, num_heads, num_kv_heads, head_dim)
        locs, lens = make_varlen_meta([seq_len])

        with pytest.raises(ValueError, match="cannot be combined with attention quantization"):
            attention(
                q,
                k,
                v,
                locs,
                lens,
                seq_len,
                is_causal=True,
                skip_softmax_threshold=1e-2,
                **qdq_kw,
            )
