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

import math

import pytest
import torch

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.kernels.common.attention.decode_attention import attention_decode
    from modelopt.torch.kernels.quantization.attention.bmm2_qdq import fake_quant_v_onwrite

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


def _nvfp4_qdq_reference(x, global_scale=1.0 / (6.0 * 448.0)):
    blocks = x.reshape(-1, 16)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = (block_amax / (6.0 * global_scale)).clamp(max=448.0)
    scales = scales.to(torch.float8_e4m3fn).float() * global_scale
    scale_safe = torch.where(scales == 0, 1.0, scales)
    levels = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    scaled = blocks.abs() / scale_safe
    quantized = levels[(scaled[..., None] - levels).abs().argmin(dim=-1)] * scale_safe
    quantized = torch.copysign(quantized, blocks)
    return torch.where(scales == 0, 0.0, quantized).reshape_as(x)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + Triton")
@pytest.mark.parametrize("num_kv_splits", [1, 32])
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
@pytest.mark.parametrize(
    ("cache_dtype", "num_q_heads", "head_dim", "value", "v_qdq_amax", "v_qdq_scale"),
    [
        pytest.param(torch.float16, 4, 64, 0.017578125, None, 1.0, id="fp16-default-gqa"),
        pytest.param(
            torch.bfloat16,
            1,
            16,
            0.019,
            1.0,
            1.0 / (6.0 * 448.0),
            id="bf16-custom-amax-carrier",
        ),
    ],
)
def test_baked_v_prefix_and_pristine_tail_match_full_onread(
    cache_dtype, num_q_heads, head_dim, value, v_qdq_amax, v_qdq_scale
):
    """Baked prefixes and pristine tails share the cache carrier at either V scale."""
    seq_len, num_kv_heads = 17, 1
    q = torch.zeros(1, num_q_heads, head_dim, device="cuda", dtype=torch.float32)
    k = torch.zeros(1, num_kv_heads, seq_len, head_dim, device="cuda", dtype=cache_dtype)
    v = torch.full_like(k, value)
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    k_cache, raw_v_cache, block_table = _paged_cache(k, v, (seq_len,))
    common = {
        "p_qdq": "nvfp4",
        "num_kv_splits": 1,
        "v_qdq": "nvfp4",
        "v_qdq_amax": v_qdq_amax,
    }

    onread = attention_decode(q, k_cache, raw_v_cache, block_table, seq_lens, **common)
    baked_v_cache = raw_v_cache.clone()
    fake_quant_v_onwrite(
        baked_v_cache,
        block_table,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.tensor([16], device="cuda", dtype=torch.int32),
        max_new_tokens=16,
        v_qdq_scale=v_qdq_scale,
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


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + Triton")
@requires_native_e4m3
def test_p_quantizes_model_dtype_input_but_accumulates_fp32():
    seq_len, head_dim = 128, 16
    scale = head_dim**-0.5
    q = torch.zeros(1, 1, head_dim, device="cuda", dtype=torch.float32)
    boundary_p = 0.04163
    q[..., 0] = -math.log2(boundary_p) / (scale * 1.4426950408889634)
    k = torch.zeros(1, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k[:, :, 1:, 0] = -1.0
    v = torch.zeros_like(k)
    v[:, :, 1:16, 0] = 1.0
    k_cache, v_cache, block_table = _paged_cache(k, v, (seq_len,))
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

    output = attention_decode(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=1,
        p_qdq="nvfp4",
    )
    scores = torch.matmul(k[0, 0].float(), q[0, 0]) * (scale * 1.4426950408889634)
    p = torch.exp2(scores - scores.max())
    p_qdq = _nvfp4_qdq_reference(p.to(torch.bfloat16).float())
    reference = (p_qdq[:, None] * v[0, 0].float()).sum(0) / p.sum()

    torch.testing.assert_close(output[0, 0], reference, rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + Triton")
@requires_native_e4m3
def test_p_qdq_matches_fixed_split_local_oracle():
    """The production 32-split schedule quantizes split-local unnormalized P."""
    seq_len, head_dim, num_splits = 4096, 16, 32
    scale = head_dim**-0.5
    q = torch.zeros(1, 1, head_dim, device="cuda", dtype=torch.float32)
    q[..., 0] = 1.0
    k = torch.zeros(1, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k[..., 0] = torch.linspace(-8.0, 8.0, seq_len, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(19)
    v = torch.randn_like(k)
    k_cache, v_cache, block_table = _paged_cache(k, v, (seq_len,))
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

    output = attention_decode(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        softmax_scale=scale,
        num_kv_splits=num_splits,
        p_qdq="nvfp4",
    )

    scores = torch.matmul(k[0, 0].float(), q[0, 0]) * (scale * 1.4426950408889634)
    split_scores = scores.reshape(num_splits, -1)
    split_max = split_scores.amax(dim=1)
    p = torch.exp2(split_scores - split_max[:, None])
    p_qdq = _nvfp4_qdq_reference(p.to(torch.bfloat16).float())
    split_acc = torch.einsum("sk,skd->sd", p_qdq, v[0, 0].float().reshape(num_splits, -1, head_dim))
    running_max = torch.tensor(-float("inf"), device="cuda")
    running_sum = torch.tensor(0.0, device="cuda")
    acc = torch.zeros(head_dim, device="cuda")
    for split_idx in range(num_splits):
        new_max = torch.maximum(running_max, split_max[split_idx])
        correction = torch.exp2(running_max - new_max)
        split_correction = torch.exp2(split_max[split_idx] - new_max)
        acc = acc * correction + split_acc[split_idx] * split_correction
        running_sum = running_sum * correction + p[split_idx].sum() * split_correction
        running_max = new_max
    reference = acc / running_sum

    torch.testing.assert_close(output[0, 0], reference, rtol=5e-5, atol=5e-5)
