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

"""NVFP4 QDQ helpers for attention BMM1 operands."""

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.quantization.common.nvfp4_quant import nvfp4_scalar_qdq

__all__ = ["fake_quant_k_onwrite"]


@triton.jit
def _fake_quant_k_onwrite_kernel(
    K_cache,
    Slot_mapping,
    num_tokens,
    stride_kc_block,
    stride_kc_pos,
    stride_kc_head,
    page_size,
    global_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(Slot_mapping + token, mask=token < num_tokens, other=-1).to(tl.int64)
    valid = (token < num_tokens) & (slot >= 0)
    block = tl.maximum(slot, 0) // page_size
    offset = tl.maximum(slot, 0) % page_size
    ptrs = K_cache + block * stride_kc_block + offset * stride_kc_pos + head * stride_kc_head + dims
    values = tl.load(ptrs, mask=valid & (dims < HEAD_DIM), other=0.0).to(tl.float32)
    grouped = tl.reshape(values, (BLOCK_D // 16, 16))
    block_amax = tl.expand_dims(tl.max(tl.abs(grouped), axis=1), 1)
    quantized = tl.reshape(nvfp4_scalar_qdq(grouped, block_amax, global_scale, 16), (BLOCK_D,))
    tl.store(ptrs, quantized.to(K_cache.dtype.element_ty), mask=valid & (dims < HEAD_DIM))


def fake_quant_k_onwrite(
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_qdq_scale: float = 1.0,
) -> None:
    """NVFP4 fake-quantize newly written paged-cache K rows in place.

    K uses block-16 groups along head dimension, the contraction dimension of
    BMM1. ``slot_mapping`` is vLLM's flattened physical cache slot for each new
    token; negative slots are ignored.
    """
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [num_blocks, page_size, num_kv_heads, head_dim]")
    if slot_mapping.ndim != 1:
        raise ValueError("slot_mapping must be one-dimensional")
    if not (math.isfinite(k_qdq_scale) and k_qdq_scale > 0):
        raise ValueError(f"k_qdq_scale must be finite and positive, got {k_qdq_scale}")
    if slot_mapping.numel() == 0:
        return

    num_kv_heads, head_dim = k_cache.shape[2:]
    if head_dim % 16:
        raise ValueError(f"head_dim={head_dim} cannot be grouped into block-16 NVFP4")
    block_d = triton.next_power_of_2(head_dim)

    with torch.cuda.device(k_cache.device):
        _fake_quant_k_onwrite_kernel[(slot_mapping.numel(), num_kv_heads)](
            k_cache,
            slot_mapping,
            slot_mapping.numel(),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.shape[1],
            k_qdq_scale,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
