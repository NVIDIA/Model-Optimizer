# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped dynamic-NVFP4 fake quantization for separate expert weights.

The N-quantizer TEGroupedLinear path stores one weight tensor per expert. Calling
each TensorQuantizer independently launches an amax reduction and an NVFP4
quantization kernel for every expert. This module preserves those per-expert
semantics while reducing the forward path to one or two launches total:

1. compute one global amax per expert when no calibrated amax is supplied; and
2. quantize every expert's last-dimension blocks with dynamic E4M3 scales.

Inputs remain separate tensors. A compact device pointer table avoids stacking
or copying the expert weights before quantization.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_kernel import _torch_dtype_to_tl
from .nvfp4_quant import fp4_round_magnitude, fp8_quantize_scale

__all__ = ["grouped_nvfp4_fakequant"]


@triton.jit
def _grouped_absmax_kernel(
    input_ptrs,
    amax_ptr,
    elements_per_expert,
    chunks_per_expert,
    DTYPE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(axis=0)
    expert_idx = program_idx // chunks_per_expert
    chunk_idx = program_idx % chunks_per_expert

    input_ptr = tl.load(input_ptrs + expert_idx).to(tl.pointer_type(DTYPE))
    offsets = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = offsets < elements_per_expert
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    chunk_amax = tl.max(tl.abs(values), axis=0)
    tl.atomic_max(amax_ptr + expert_idx, chunk_amax)


@triton.jit
def _grouped_nvfp4_fakequant_kernel(
    input_ptrs,
    output_ptr,
    amax_ptr,
    elements_per_expert,
    tiles_per_expert,
    block_size: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    program_idx = tl.program_id(axis=0)
    expert_idx = program_idx // tiles_per_expert
    tile_idx = program_idx % tiles_per_expert

    input_ptr = tl.load(input_ptrs + expert_idx).to(tl.pointer_type(DTYPE))
    tile_size: tl.constexpr = block_size * BLOCKS_PER_PROGRAM
    offsets = tile_idx * tile_size + tl.arange(0, tile_size)
    mask = offsets < elements_per_expert

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values_by_block = tl.reshape(values, (BLOCKS_PER_PROGRAM, block_size))
    block_amax = tl.max(tl.abs(values_by_block), axis=1, keep_dims=True)

    global_amax = tl.load(amax_ptr + expert_idx).to(tl.float32)
    global_scale = global_amax / (6.0 * 448.0)
    global_scale = tl.where(global_scale > 0.0, global_scale, 1e-12)

    block_scale = fp8_quantize_scale(block_amax, global_scale)
    # Match fp4_fake_quant_block: very small or zero blocks use unit scale.
    block_scale = tl.where(block_scale >= 1e-5, block_scale, 1.0)
    scaled = tl.abs(values_by_block) / tl.broadcast_to(
        block_scale, (BLOCKS_PER_PROGRAM, block_size)
    )
    quantized = fp4_round_magnitude(scaled)
    quantized = quantized * tl.broadcast_to(block_scale, (BLOCKS_PER_PROGRAM, block_size))
    quantized = tl.where(values_by_block >= 0, quantized, -quantized)

    output_offsets = expert_idx * elements_per_expert + offsets
    tl.store(
        output_ptr + output_offsets,
        tl.reshape(quantized, (tile_size,)).to(DTYPE),
        mask=mask,
    )


def _validate_weights(weights: list[torch.Tensor], block_size: int) -> None:
    if not weights:
        raise ValueError("grouped_nvfp4_fakequant requires at least one expert weight")
    reference = weights[0]
    if not reference.is_cuda:
        raise ValueError("grouped_nvfp4_fakequant requires CUDA tensors")
    if reference.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported weight dtype: {reference.dtype}")
    if reference.shape[-1] % block_size != 0:
        raise ValueError(
            f"weight last dimension ({reference.shape[-1]}) must be divisible by {block_size}"
        )
    if not reference.is_contiguous():
        raise ValueError("expert weights must be contiguous")
    for weight in weights[1:]:
        if weight.shape != reference.shape:
            raise ValueError("all expert weights must have the same shape")
        if weight.dtype != reference.dtype:
            raise ValueError("all expert weights must have the same dtype")
        if weight.device != reference.device:
            raise ValueError("all expert weights must be on the same device")
        if not weight.is_contiguous():
            raise ValueError("expert weights must be contiguous")


def grouped_nvfp4_fakequant(
    weights: list[torch.Tensor],
    block_size: int = 16,
    expert_amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Fake-quantize N expert weights with grouped dynamic NVFP4 kernels.

    The output tensors are views into one contiguous allocation, but each view
    has the same shape, dtype, and device as its corresponding input tensor. A
    supplied per-expert amax reduces execution to the quantization launch only;
    otherwise, the grouped amax reduction runs first.
    """
    _validate_weights(weights, block_size)

    reference = weights[0]
    num_experts = len(weights)
    elements_per_expert = reference.numel()
    input_ptrs = torch.tensor(
        [weight.data_ptr() for weight in weights],
        dtype=torch.int64,
        device=reference.device,
    )
    has_expert_amax = expert_amax is not None
    if has_expert_amax:
        if expert_amax.numel() != num_experts:
            raise ValueError(
                f"expert_amax must contain {num_experts} values, got {expert_amax.numel()}"
            )
        expert_amax = (
            expert_amax.to(device=reference.device, dtype=torch.float32).reshape(-1).contiguous()
        )
    else:
        expert_amax = torch.zeros(num_experts, dtype=torch.float32, device=reference.device)
    output = torch.empty(
        (num_experts, *reference.shape),
        dtype=reference.dtype,
        device=reference.device,
    )

    chunk_size = 2048
    chunks_per_expert = triton.cdiv(elements_per_expert, chunk_size)
    # Match fp4_fake_quant_block's 16x64 tile footprint: 1024 values, or 64
    # consecutive NVFP4 blocks, per Triton program.
    blocks_per_program = 64
    tile_size = block_size * blocks_per_program
    tiles_per_expert = triton.cdiv(elements_per_expert, tile_size)
    quant_grid = (num_experts * tiles_per_expert,)

    with torch.cuda.device(reference.device):
        if not has_expert_amax:
            amax_grid = (num_experts * chunks_per_expert,)
            _grouped_absmax_kernel[amax_grid](
                input_ptrs,
                expert_amax,
                elements_per_expert,
                chunks_per_expert,
                DTYPE=_torch_dtype_to_tl(reference.dtype),
                CHUNK_SIZE=chunk_size,
            )
        _grouped_nvfp4_fakequant_kernel[quant_grid](
            input_ptrs,
            output,
            expert_amax,
            elements_per_expert,
            tiles_per_expert,
            block_size=block_size,
            DTYPE=_torch_dtype_to_tl(reference.dtype),
            BLOCKS_PER_PROGRAM=blocks_per_program,
        )

    return tuple(output.unbind(0))
