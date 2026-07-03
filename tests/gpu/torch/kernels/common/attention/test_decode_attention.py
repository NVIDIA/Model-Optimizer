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

"""GPU tests for the minimal paged split-K decode kernel."""

import pytest
import torch

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention.decode_attention import attention_decode
    from modelopt.torch.kernels.quantization.attention.v_qdq import fake_quant_v_onwrite

NATIVE_E4M3_AVAILABLE = TRITON_KERNEL_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
requires_native_e4m3 = pytest.mark.skipif(
    not NATIVE_E4M3_AVAILABLE, reason="Native E4M3 requires compute capability >= 8.9"
)


def _paged_cache(k, v, seq_lens, page_size=16):
    batch, num_kv_heads, _, head_dim = k.shape
    blocks = [(int(length) + page_size - 1) // page_size for length in seq_lens]
    k_cache = torch.zeros(
        sum(blocks), page_size, num_kv_heads, head_dim, device=k.device, dtype=k.dtype
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(batch, max(blocks), device=k.device, dtype=torch.int32)
    physical = 0
    for batch_idx, length in enumerate(seq_lens):
        for logical in range(blocks[batch_idx]):
            block_table[batch_idx, logical] = physical
            start = logical * page_size
            stop = min(start + page_size, int(length))
            k_cache[physical, : stop - start] = k[batch_idx, :, start:stop].transpose(0, 1)
            v_cache[physical, : stop - start] = v[batch_idx, :, start:stop].transpose(0, 1)
            physical += 1
    return k_cache, v_cache, block_table


def _dense_decode(q, k, v, seq_lens, scale):
    output = torch.empty_like(q)
    group_size = q.shape[1] // k.shape[1]
    for batch_idx, length in enumerate(seq_lens):
        for head_idx in range(q.shape[1]):
            kv_head = head_idx // group_size
            scores = (
                torch.matmul(k[batch_idx, kv_head, :length].float(), q[batch_idx, head_idx].float())
                * scale
            )
            output[batch_idx, head_idx] = torch.matmul(
                torch.softmax(scores, dim=0), v[batch_idx, kv_head, :length].float()
            ).to(q.dtype)
    return output


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + Triton")
@pytest.mark.parametrize("num_kv_splits", [1, 8, 32])
def test_split_k_varlen_gqa_matches_dense(num_kv_splits):
    torch.manual_seed(13)
    batch, num_q_heads, num_kv_heads, seq_len, head_dim = 2, 8, 2, 511, 64
    seq_lens = (130, seq_len)
    q = torch.randn(batch, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table = _paged_cache(k, v, seq_lens)
    seq_lens_device = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    scale = head_dim**-0.5

    output = attention_decode(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens_device,
        softmax_scale=scale,
        num_kv_splits=num_kv_splits,
    )

    torch.testing.assert_close(
        output, _dense_decode(q, k, v, seq_lens, scale), rtol=5e-3, atol=5e-3
    )


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + Triton")
@requires_native_e4m3
def test_baked_v_prefix_and_pristine_tail_match_full_onread():
    seq_len, num_q_heads, num_kv_heads, head_dim = 17, 4, 1, 64
    q = torch.zeros(1, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.full_like(k, 0.017578125)
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    k_cache, raw_v_cache, block_table = _paged_cache(k, v, (seq_len,))
    common = {
        "p_qdq": "nvfp4",
        "v_qdq": "nvfp4",
        "num_kv_splits": 1,
    }

    onread = attention_decode(q, k_cache, raw_v_cache, block_table, seq_lens, **common)
    baked_v_cache = raw_v_cache.clone()
    fake_quant_v_onwrite(
        baked_v_cache,
        block_table,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([16], device="cuda", dtype=torch.int32),
    )
    baked_v_before = baked_v_cache.clone()
    baked = attention_decode(
        q,
        k_cache,
        baked_v_cache,
        block_table,
        seq_lens,
        v_cache_quantized=True,
        **common,
    )

    torch.testing.assert_close(baked, onread, rtol=0, atol=0)
    torch.testing.assert_close(baked_v_cache, baked_v_before, rtol=0, atol=0)
