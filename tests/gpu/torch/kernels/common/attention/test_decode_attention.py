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


def _softmax_exp2(x, softmax_mode):
    if softmax_mode == "mixed_fp16":
        return torch.exp2(x.half()).float()
    assert softmax_mode == "fp32"
    return torch.exp2(x)


def _combine_split_states(split_max, split_sum, split_acc, softmax_mode):
    running_max = torch.tensor(-float("inf"), device=split_max.device)
    running_sum = torch.tensor(0.0, device=split_max.device)
    acc = torch.zeros_like(split_acc[0])
    for split_idx in range(split_max.shape[0]):
        new_max = torch.maximum(running_max, split_max[split_idx])
        correction = _softmax_exp2(running_max - new_max, softmax_mode)
        split_correction = _softmax_exp2(split_max[split_idx] - new_max, softmax_mode)
        acc = acc * correction + split_acc[split_idx] * split_correction
        running_sum = running_sum * correction + split_sum[split_idx] * split_correction
        running_max = new_max
    return acc / running_sum


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
@pytest.mark.parametrize("softmax_mode", ["fp32", "mixed_fp16"])
def test_p_qdq_matches_fixed_split_local_oracle(softmax_mode):
    """The production 32-split schedule quantizes split-local unnormalized P."""
    seq_len, head_dim, num_splits = 8192, 16, 32
    block_n = 128
    scale = head_dim**-0.5
    q = torch.zeros(1, 1, head_dim, device="cuda", dtype=torch.float32)
    q[..., 0] = 1.0 / (scale * 1.4426950408889634)
    k = torch.zeros(1, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    token_idx = torch.arange(seq_len, device="cuda")
    token_split = token_idx // (2 * block_n)
    tile_phase = (token_idx // block_n) % 2
    score_pattern = -0.25 * ((token_idx % block_n) % 3) + 0.25 * tile_phase + 0.25 * token_split
    k[0, 0, :, 0] = score_pattern.to(torch.bfloat16)
    v = torch.zeros_like(k)
    # Channel 0 isolates the split-local correction; channel 1 nearly cancels only
    # under the intended FP32 combine, making combine precision observable.
    v[0, 0, :, 0] = torch.where(tile_phase == 0, 256.0, -215.0).to(torch.bfloat16)
    v[0, 0, :, 1] = torch.where(token_split == num_splits - 1, -5.25, 1.0).to(torch.bfloat16)
    k_cache, v_cache, block_table = _paged_cache(k, v, (seq_len,))
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)

    common = {
        "softmax_scale": scale,
        "num_kv_splits": num_splits,
        "p_qdq": "nvfp4",
    }
    output = attention_decode(
        q, k_cache, v_cache, block_table, seq_lens, softmax_mode=softmax_mode, **common
    )
    if softmax_mode == "mixed_fp16":
        fp32_output = attention_decode(
            q, k_cache, v_cache, block_table, seq_lens, softmax_mode="fp32", **common
        )
        assert not torch.equal(output.to(k.dtype), fp32_output.to(k.dtype))

    scores = torch.matmul(k[0, 0].float(), q[0, 0]) * (scale * 1.4426950408889634)
    split_scores = scores.reshape(num_splits, 2, block_n)
    split_values = v[0, 0].float().reshape(num_splits, 2, block_n, head_dim)
    split_max = []
    split_sum = []
    split_acc = []
    for split_idx in range(num_splits):
        running_max = torch.tensor(-float("inf"), device="cuda")
        running_sum = torch.tensor(0.0, device="cuda")
        local_acc = torch.zeros(head_dim, device="cuda")
        for tile_idx in range(2):
            tile_scores = split_scores[split_idx, tile_idx]
            new_max = torch.maximum(running_max, tile_scores.amax())
            p = _softmax_exp2(tile_scores - new_max, softmax_mode)
            correction = _softmax_exp2(running_max - new_max, softmax_mode)
            running_sum = running_sum * correction + p.sum()
            local_acc *= correction
            p_qdq = _nvfp4_qdq_reference(p.to(torch.bfloat16).float())
            local_acc += (p_qdq[:, None] * split_values[split_idx, tile_idx]).sum(dim=0)
            running_max = new_max
        split_max.append(running_max)
        split_sum.append(running_sum)
        split_acc.append(local_acc)
    split_max = torch.stack(split_max)
    split_sum = torch.stack(split_sum)
    split_acc = torch.stack(split_acc)
    reference = _combine_split_states(split_max, split_sum, split_acc, "fp32")
    if softmax_mode == "mixed_fp16":
        mixed_combine_reference = _combine_split_states(
            split_max, split_sum, split_acc, "mixed_fp16"
        )
        assert reference.to(k.dtype)[1] != mixed_combine_reference.to(k.dtype)[1]

    torch.testing.assert_close(output[0, 0], reference, rtol=5e-5, atol=5e-5)
