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

# Adapted from triton_unified_attention.py from
# https://github.com/vllm-project/vllm/blob/v0.15.0/vllm/v1/attention/ops/triton_unified_attention.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Triton attention for prefill and decode with paged KV cache.

Supports variable sequence lengths, causal masking, GQA, and sliding window.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


# 2:4 structured sparsity helpers (from vLLM flash_attn_triton_sparse24)
@triton.jit
def _sparse24_noabs_ops(x0, x1, x2, x3):
    """Compute 2:4 sparsity mask: for every 4 values, determine which 2 are largest."""
    (a1, a2, a3, a4, a5, a6) = (
        x0 > x1,
        x0 > x2,
        x0 > x3,
        x1 > x2,
        x1 > x3,
        x2 > x3,
    )
    # Use (x == 0) instead of ~x to avoid interpreter bug with __invert__ on bool tensors
    na1 = a1 == 0
    na2 = a2 == 0
    na3 = a3 == 0
    na4 = a4 == 0
    na5 = a5 == 0
    na6 = a6 == 0
    m0 = a2 & a3 | a1 & a2 | a1 & a3
    m1 = na1 & a5 | a4 & a5 | na1 & a4
    m2 = na2 & na4 | na2 & a6 | na4 & a6
    m3 = na3 & na5 | na3 & na6 | na5 & na6
    return x0, x1, x2, x3, m0, m1, m2, m3


@triton.jit
def _apply_sparse24_to_qk_tile(
    qk,
    M: tl.constexpr,
    N: tl.constexpr,
    MASK_VAL: tl.constexpr,
):
    """Apply 2:4 sparsity to attention score tile [M, N]: keep top 2 of every 4 along N."""
    reshaped = tl.reshape(qk, (M, N // 4, 4))
    cols = tl.arange(0, 4)[None, None, :]
    x0 = tl.sum(tl.where(cols == 0, reshaped, 0.0), axis=2)
    x1 = tl.sum(tl.where(cols == 1, reshaped, 0.0), axis=2)
    x2 = tl.sum(tl.where(cols == 2, reshaped, 0.0), axis=2)
    x3 = tl.sum(tl.where(cols == 3, reshaped, 0.0), axis=2)
    _, _, _, _, m0, m1, m2, m3 = _sparse24_noabs_ops(x0, x1, x2, x3)
    s0 = tl.where(m0, x0, MASK_VAL)
    s1 = tl.where(m1, x1, MASK_VAL)
    s2 = tl.where(m2, x2, MASK_VAL)
    s3 = tl.where(m3, x3, MASK_VAL)
    sparse_reshaped = tl.full((M, N // 4, 4), 0.0, dtype=qk.dtype)
    sparse_reshaped = tl.where((cols == 0), tl.expand_dims(s0, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 1), tl.expand_dims(s1, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 2), tl.expand_dims(s2, 2), sparse_reshaped)
    sparse_reshaped = tl.where((cols == 3), tl.expand_dims(s3, 2), sparse_reshaped)
    sparse_qk = tl.reshape(sparse_reshaped, (M, N))
    return sparse_qk


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    APPLY_SPARSE24: tl.constexpr,
    SKIP_DIAGONAL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    if CAUSAL:
        # Causal: only attend up to the query position
        max_seq_prefix_len = (
            context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
        )
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        # Non-causal (cross-attention): attend to all K/V positions
        max_seq_prefix_len = seq_len
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if CAUSAL and SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if CAUSAL:
            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = seq_offset[None, :] <= query_abs_pos
            if SLIDING_WINDOW > 0:
                seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
        else:
            seq_mask = tile_mask[None, :]

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))

        if APPLY_SPARSE24:
            if CAUSAL and SKIP_DIAGONAL_BLOCKS:
                tile_key_start = j * TILE_SIZE
                tile_key_end = tile_key_start + TILE_SIZE
                query_abs_start = context_len + q_block_local_idx * BLOCK_Q
                query_abs_end = query_abs_start + BLOCK_Q
                is_diagonal = (tile_key_start < query_abs_end) & (tile_key_end > query_abs_start)
                if not is_diagonal:
                    S = _apply_sparse24_to_qk_tile(S, BLOCK_M, TILE_SIZE, float("-inf"))
            else:
                S = _apply_sparse24_to_qk_tile(S, BLOCK_M, TILE_SIZE, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if CAUSAL and SLIDING_WINDOW > 0:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where((context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0)
        acc += tl.dot(P.to(V.dtype), V)

    acc = acc / L[:, None]
    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if CAUSAL:
        max_seq_prefix_len = (
            context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
        )
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = seq_len
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if CAUSAL and SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        if CAUSAL:
            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = seq_offset[None, :] <= query_abs_pos
            if SLIDING_WINDOW > 0:
                seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
        else:
            seq_mask = tile_mask[None, :]

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if CAUSAL and SLIDING_WINDOW > 0:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where((context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    block_table_stride: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size. Must be power of 2."""
    if sliding_window == 1024 and head_size in (128, 256):
        return 32
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    apply_sparse24: bool = False,
    skip_diagonal_blocks: bool = True,
) -> None:
    """Unified attention over paged KV cache (prefill and decode).

    Args:
        q: [num_tokens, num_query_heads, head_size]
        k: [num_blocks, block_size, num_kv_heads, head_size] (paged K cache)
        v: [num_blocks, block_size, num_kv_heads, head_size] (paged V cache)
        out: [num_tokens, num_query_heads, head_size]
        cu_seqlens_q: [num_seqs + 1] cumulative query token counts
        max_seqlen_q: max query length
        seqused_k: [num_seqs] total sequence length per batch (context + query)
        max_seqlen_k: max sequence length
        softmax_scale: attention scale (e.g. 1/sqrt(head_size))
        causal: True for causal self-attention, False for cross-attention
        window_size: (q_window, k_window), -1 means disabled; only used when causal=True
        block_table: [num_seqs, max_blocks_per_seq]
        seq_threshold_3D: if set with 3D buffers, use 3D kernel when num_seqs <= this
        num_par_softmax_segments: number of segments for 3D kernel
        softmax_segm_output, softmax_segm_max, softmax_segm_expsum: 3D kernel buffers
        apply_sparse24: If True, apply 2:4 structured sparsity to attention scores.
            Only applied during prefill (max_seqlen_q > 1); automatically disabled
            during decode. The 3D kernel path also ignores this flag (a warning is
            emitted). TILE_SIZE must be divisible by 4.
        skip_diagonal_blocks: If True, keep diagonal tiles dense (local attention
            preserved) when sparse24 is active.
    """
    block_size = v.shape[1]
    num_seqs = seqused_k.shape[0]
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    head_size_padded = max(triton.next_power_of_2(head_size), 16)
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    if apply_sparse24:
        assert TILE_SIZE_PREFILL % 4 == 0, (
            f"sparse24 requires TILE_SIZE divisible by 4, got TILE_SIZE_PREFILL={TILE_SIZE_PREFILL}"
        )

    use_3d = (
        seq_threshold_3D is not None
        and num_par_softmax_segments is not None
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
        and max_seqlen_q <= 1
        and num_seqs <= seq_threshold_3D
    )

    # Sparse24 is only meaningful during prefill (max_seqlen_q > 1).
    # During decode (max_seqlen_q <= 1), disable it regardless of the caller's flag.
    # The 3D kernel (decode-only) therefore never sees sparse24 enabled.
    is_prefill = max_seqlen_q > 1
    effective_sparse24 = apply_sparse24 and is_prefill

    if not use_3d:
        kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            scale=softmax_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            SLIDING_WINDOW=sliding_window_val,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            APPLY_SPARSE24=effective_sparse24,
            SKIP_DIAGONAL_BLOCKS=skip_diagonal_blocks,
            CAUSAL=causal,
        )
    else:
        kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            scale=softmax_scale,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            BLOCK_SIZE=block_size,
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            SLIDING_WINDOW=sliding_window_val,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            CAUSAL=causal,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
    apply_sparse24: bool = False,
    skip_diagonal_blocks: bool = True,
    b_start_loc_k: torch.Tensor | None = None,
    b_seq_len_k: torch.Tensor | None = None,
    max_input_len_k: int | None = None,
) -> None:
    """Prefill attention over contiguous Q/K/V (packed format).

    Converts contiguous tensors to paged format and calls unified_attention.
    For causal self-attention, Q and K/V share the same sequence lengths
    (``b_seq_len``). For cross-attention (``is_causal=False``), K/V may have
    different lengths specified via ``b_seq_len_k``.

    Note:
        When ``apply_sparse24=True``, TILE_SIZE must be divisible by 4 (the 2:4
        sparsity reshape requires ``N // 4``). Current tile sizes (16, 32) satisfy
        this. Causal-masked elements participate in the 2:4 top-2 selection, which
        may waste sparsity slots near the diagonal.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3 or o.dim() != 3:
        raise ValueError(
            "q, k, v, o must be rank-3 [total_tokens, num_heads, head_dim]; "
            f"got q.dim()={q.dim()}, k.dim()={k.dim()}, v.dim()={v.dim()}, o.dim()={o.dim()}."
        )
    head_dim = q.shape[2]
    if k.shape[2] != head_dim or v.shape[2] != head_dim or o.shape[2] != head_dim:
        raise ValueError(
            "q, k, v, o must have same head_dim (shape[2]); "
            f"got {q.shape[2]}, {k.shape[2]}, {v.shape[2]}, {o.shape[2]}."
        )
    if o.shape[0] != q.shape[0] or o.shape[1] != q.shape[1]:
        raise ValueError(f"o must match q shape; got o={o.shape}, q={q.shape}.")
    num_kv_heads = k.shape[1]
    if num_kv_heads <= 0:
        raise ValueError(f"k.shape[1] (num_kv_heads) must be positive; got {num_kv_heads}.")
    if q.shape[1] % num_kv_heads != 0:
        raise ValueError(
            f"num_heads (q.shape[1]) must be divisible by num_kv_heads (k.shape[1]); "
            f"got {q.shape[1]} and {num_kv_heads}."
        )

    # For causal self-attention, Q and K/V share lengths.
    # For cross-attention, K/V lengths come from separate parameters.
    if b_seq_len_k is None:
        # Self-attention: Q and K/V have same total tokens and lengths
        total_q = q.shape[0]
        if k.shape[0] != total_q or v.shape[0] != total_q:
            raise ValueError(
                "For causal self-attention, q, k, v must have same shape[0]; "
                f"got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}. "
                "For cross-attention, pass b_seq_len_k and b_start_loc_k."
            )
        b_seq_len_k = b_seq_len
        b_start_loc_k = b_start_loc
        max_input_len_k = max_input_len

    batch = b_seq_len.shape[0]
    if b_start_loc_k is None:
        b_start_loc_k = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
        b_start_loc_k[1:] = torch.cumsum(b_seq_len_k.to(torch.int64), dim=0)
        b_start_loc_k = b_start_loc_k[:batch]
    if max_input_len_k is None:
        max_input_len_k = int(b_seq_len_k.max().item())

    device = q.device
    dtype = q.dtype
    block_size = ((max_input_len_k + 31) // 32) * 32

    k_cache = torch.zeros((batch, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.zeros((batch, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    for i in range(batch):
        start = int(b_start_loc_k[i].item())
        length = int(b_seq_len_k[i].item())
        if length > 0:
            k_cache[i, :length, :, :] = k[start : start + length]
            v_cache[i, :length, :, :] = v[start : start + length]

    block_table = torch.arange(batch, device=device, dtype=torch.int32).unsqueeze(1)

    cu_seqlens_q = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = torch.cumsum(b_seq_len.to(torch.int64), dim=0)
    seqused_k = b_seq_len_k.to(torch.int32)

    scale = 1.0 / (head_dim**0.5) if softmax_scale is None else softmax_scale
    sw_q = sliding_window_q if sliding_window_q is not None else -1
    sw_k = sliding_window_k if sliding_window_k is not None else -1
    window_size = (sw_q, sw_k)

    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_input_len,
        seqused_k=seqused_k,
        max_seqlen_k=block_size,
        softmax_scale=scale,
        causal=is_causal,
        window_size=window_size,
        block_table=block_table,
        apply_sparse24=apply_sparse24,
        skip_diagonal_blocks=skip_diagonal_blocks,
    )


__all__ = ["context_attention_fwd", "unified_attention"]
