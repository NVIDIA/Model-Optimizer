# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused per-expert axis-0 fake-quant Triton kernels for TEGroupedLinear.

Replaces the stack-then-quantize-then-unbind pattern in modelopt's TEGrouped
plugin (`te_grouped_quantized_linear_fn`) with a single Triton launch that
processes N expert weights in place, with no contiguous-tensor staging.

Design — tensor of pointers
---------------------------

The N expert weights live as separate Parameters (one per expert), so they're
NOT contiguous in HBM. To avoid a `torch.stack` memcopy (the cost AC5
characterized on OMNIML-5064), we feed the kernel a `[N]` int64 tensor of
expert base pointers. Each Triton program reads its expert's pointer first,
then strides through a block of elements at that address.

Grid: (N, num_blocks_per_expert).
Program 0 of axis 0 → expert 0, program 1 → expert 1, etc.

See OMNIML-5072 AC5 (Option B follow-up) for the motivation.

VALIDATION STATUS (2026-06-11): kernel implemented, numerical fidelity NOT
yet validated against modelopt's reference `fake_quant_impl`, and bench
performance NOT yet measured. See VALIDATION_TODO.md in this directory.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

__all__ = ["grouped_axis0_fakequant", "grouped_axis0_fakequant_backward"]


def _torch_dtype_to_tl(dtype: torch.dtype):
    """Map a torch dtype to its Triton-language equivalent."""
    return {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
    }[dtype]


@triton.jit
def _grouped_axis0_fakequant_fwd_kernel(
    weight_ptrs_buf,    # int64 [N]  — N expert base pointers (cast from .data_ptr())
    output_ptrs_buf,    # int64 [N]  — N output base pointers
    amax_vec_ptr,       # [N, 1, 1] (or anything with N as the leading dim)
    elements_per_expert,
    num_bits,
    narrow_range: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # Per-expert base pointers (loaded once per program).
    w_int = tl.load(weight_ptrs_buf + expert_idx)
    out_int = tl.load(output_ptrs_buf + expert_idx)
    w_ptr = w_int.to(tl.pointer_type(DTYPE))
    out_ptr = out_int.to(tl.pointer_type(DTYPE))

    # Per-expert amax → quant scale.
    # amax is stored as fp32; convert to working precision.
    amax = tl.load(amax_vec_ptr + expert_idx).to(tl.float32)
    # qmax = 2^(num_bits-1) - 1 when narrow_range else 2^(num_bits-1)
    # For num_bits=8 narrow_range=True (modelopt default): qmax=127
    qmax = ((1 << (num_bits - 1)) - 1) if narrow_range else (1 << (num_bits - 1))
    qmin = -qmax if narrow_range else -qmax  # signed symmetric
    scale = amax / qmax

    # Block of elements within this expert.
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_expert

    x = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Fake-quant: round(clip(x / scale)) * scale.
    # Use scale guards to avoid div-by-zero before _amax is calibrated (early
    # batches may carry an _amax of 0; matches modelopt's fake_tensor_quant
    # behavior of passing through unchanged).
    safe_scale = tl.where(scale > 0.0, scale, 1.0)
    q = x / safe_scale
    q = tl.maximum(tl.minimum(q, qmax), qmin)
    # Round-half-to-even (banker's), matching cuda_ext.fake_tensor_quant exactly.
    # libdevice.rint is CUDA's __rint* builtin. Imported via the same path that
    # modelopt's nvfp4_quant.py uses (triton.language.extra.cuda.libdevice).
    q_rounded = libdevice.rint(q)
    out = tl.where(scale > 0.0, q_rounded * scale, x)

    tl.store(out_ptr + offsets, out.to(DTYPE), mask=mask)


@triton.jit
def _grouped_axis0_fakequant_bwd_kernel(
    weight_ptrs_buf,    # int64 [N]  — same buffer as fwd
    grad_out_ptrs_buf,  # int64 [N]  — upstream grad pointers (per expert)
    grad_in_ptrs_buf,   # int64 [N]  — output: downstream grad pointers
    amax_vec_ptr,       # [N, ...]   — same buffer as fwd
    elements_per_expert,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Clip-aware STE backward.

    For each expert i:
        grad_in[i] = grad_out[i] if |w[i]| <= amax[i] else 0
    matches modelopt's `_fake_tensor_quant_backward` semantics.
    """
    expert_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    w_ptr = tl.load(weight_ptrs_buf + expert_idx).to(tl.pointer_type(DTYPE))
    grad_out_ptr = tl.load(grad_out_ptrs_buf + expert_idx).to(tl.pointer_type(DTYPE))
    grad_in_ptr = tl.load(grad_in_ptrs_buf + expert_idx).to(tl.pointer_type(DTYPE))

    amax = tl.load(amax_vec_ptr + expert_idx).to(tl.float32)

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements_per_expert

    # Stay in DTYPE (bf16/fp16) throughout — eliminates fp32 round-trip seen in
    # the Btriton2 baseline that capped bwd bandwidth at ~4.2 TB/s vs cuda_ext's
    # ~8 TB/s on B300. amax cast to DTYPE once; comparison done in low precision
    # (amax values are O(1)-O(10), well within bf16 range).
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    amax_dt = amax.to(DTYPE)

    # Clip-aware STE: pass through gradient where |w| <= amax, else zero.
    pass_through = tl.abs(w) <= amax_dt
    grad_in = tl.where(pass_through, g, 0.0)

    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def _build_ptr_buf(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pack a list of tensors' .data_ptr() into a single int64 tensor on the same device."""
    return torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def grouped_axis0_fakequant(
    weights: list[torch.Tensor],
    amax_vec: torch.Tensor,
    num_bits: int = 8,
    narrow_range: bool = True,
) -> list[torch.Tensor]:
    """Apply per-expert axis-0 fake-quant in a single Triton launch.

    Args:
        weights: List of N expert weight tensors. Each must have the same shape
            `[out, in]` and same dtype.
        amax_vec: Per-expert amax buffer of shape `[N, 1, 1]` (or any shape where
            element `i` is expert `i`'s amax). dtype should be float32 for
            numerical headroom; the kernel casts to fp32 internally.
        num_bits: integer bit-width for the fake-quant.
        narrow_range: if True, output range is [-qmax, +qmax]; else [-qmax, +qmax-1].
            modelopt's default is True.

    Returns:
        List of N quantized weight tensors, each the same shape and dtype as
        the corresponding input.
    """
    assert len(weights) >= 1, "grouped_axis0_fakequant requires at least one expert"
    N = len(weights)
    shape0 = weights[0].shape
    dtype0 = weights[0].dtype
    device0 = weights[0].device
    elements_per_expert = weights[0].numel()
    for w in weights[1:]:
        assert w.shape == shape0, "all expert weights must share the same shape"
        assert w.dtype == dtype0, "all expert weights must share the same dtype"
        assert w.device == device0, "all expert weights must share the same device"

    outputs = [torch.empty_like(w) for w in weights]

    weight_ptrs = _build_ptr_buf(weights)
    output_ptrs = _build_ptr_buf(outputs)

    # BLOCK_SIZE=2048 was empirically best in the Btriton2 sweep — larger blocks
    # (16384 + num_warps=8) regressed both fwd and bwd, likely from worse warp
    # occupancy and load coalescing on B300.
    BLOCK_SIZE = 2048
    num_blocks_per_expert = triton.cdiv(elements_per_expert, BLOCK_SIZE)
    grid = (N, num_blocks_per_expert)

    with torch.cuda.device(device0):
        _grouped_axis0_fakequant_fwd_kernel[grid](
            weight_ptrs,
            output_ptrs,
            amax_vec,
            elements_per_expert,
            num_bits,
            narrow_range=narrow_range,
            DTYPE=_torch_dtype_to_tl(dtype0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return outputs


def grouped_axis0_fakequant_backward(
    weights: list[torch.Tensor],
    grad_outputs: list[torch.Tensor],
    amax_vec: torch.Tensor,
) -> list[torch.Tensor]:
    """Apply per-expert clip-aware STE backward in a single Triton launch.

    Matches modelopt's `_fake_tensor_quant_backward` semantics — gradient
    passes through where `|w[i]| <= amax[i]`, else zero.

    Args:
        weights: List of N expert weight tensors (the original fwd inputs).
        grad_outputs: List of N upstream gradients, one per expert.
        amax_vec: Per-expert amax buffer (same shape as in fwd).

    Returns:
        List of N downstream gradients, one per expert.
    """
    N = len(weights)
    assert len(grad_outputs) == N
    shape0 = weights[0].shape
    dtype0 = weights[0].dtype
    device0 = weights[0].device
    elements_per_expert = weights[0].numel()

    grad_inputs = [torch.empty_like(w) for w in weights]

    weight_ptrs = _build_ptr_buf(weights)
    grad_out_ptrs = _build_ptr_buf(grad_outputs)
    grad_in_ptrs = _build_ptr_buf(grad_inputs)

    BLOCK_SIZE = 2048
    num_blocks_per_expert = triton.cdiv(elements_per_expert, BLOCK_SIZE)
    grid = (N, num_blocks_per_expert)

    with torch.cuda.device(device0):
        _grouped_axis0_fakequant_bwd_kernel[grid](
            weight_ptrs,
            grad_out_ptrs,
            grad_in_ptrs,
            amax_vec,
            elements_per_expert,
            DTYPE=_torch_dtype_to_tl(dtype0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return grad_inputs
