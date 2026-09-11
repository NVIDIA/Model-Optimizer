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

"""Fused Triton kernel for the NVFP4 weight-MSE FP8 scale sweep.

Replaces the Python sweep in :class:`NVFP4MSECalibrator` with a single kernel that,
for each NVFP4 block, evaluates either all 126 valid FP8 E4M3 scale candidates or a
bounded range of code offsets and emits the per-block ``best_amax`` directly.

Exhaustive candidates retain ``valid_fp8_e4m3_value / 448``; bounded candidates use
``valid_fp8_e4m3_value / normalization_max`` (see :func:`fp8_scale_candidates`),
where the normalization max is 448 normally and 256 in 4/6 mode. Their FP8 round-trip
is the identity, so the kernel can use
``scale = candidate * global_amax / 6.0`` without an explicit FP8 cast — making it
runnable on any CUDA GPU with Triton (no ``tl.float8e4nv`` requirement).

Tile shape (``BLOCKS_PER_PROGRAM``) and ``num_warps`` are autotuned per ``N_BLOCKS``.
"""

import torch
import triton
import triton.language as tl

from ..common.nvfp4_quant import fp4_round_magnitude
from ._fp8_scale_candidates import fp8_scale_candidates, fp8_scale_codes
from .fp4_kernel import compute_fp4_scales

__all__ = [
    "fp8_scale_candidates",
    "nvfp4_fp8_scale_sweep",
    "nvfp4_fp8_scale_sweep_hessian",
]


# Selected from a (BLOCKS_PER_PROGRAM, num_warps) sweep on B300:
#   BPP=16,nw=2: 6.06 ms   BPP=32,nw=4: 6.06 ms   BPP=64,nw=8: 5.08 ms
# The smaller-tile entries cover cases where N_BLOCKS is small enough that BPP=64
# would underfill the SMs.
_FP8_SWEEP_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 16}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=8),
]


@triton.autotune(configs=_FP8_SWEEP_AUTOTUNE_CONFIGS, key=["N_BLOCKS"])
@triton.jit
def _fp8_scale_sweep_kernel(
    x_ptr,  # [N_BLOCKS * BLOCK_SIZE], any float dtype (loaded as fp32)
    candidates_ptr,  # [NUM_CANDIDATES] fp32
    base_codes_ptr,  # [N_BLOCKS] uint8; used only for bounded search
    global_amax_ptr,  # scalar fp32
    best_amax_ptr,  # [N_BLOCKS] fp32 output
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MIN_OFFSET: tl.constexpr,
    BOUNDED: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKS_PER_PROGRAM
    block_idx = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_idx < N_BLOCKS

    # Load weights for this tile and pre-compute their absolute values once.
    # The squared error is sign-invariant since FP4 quant preserves sign:
    #   (w - w_q)^2 = (|w| - |w_q|)^2 = (|w| - q_mag * scale)^2
    # so we never need ``w`` itself again, dropping a tl.where + negation per element.
    elem_offs = block_idx[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    elem_mask = block_mask[:, None]
    w_abs = tl.abs(tl.load(x_ptr + elem_offs, mask=elem_mask, other=0.0).to(tl.float32))

    global_amax = tl.load(global_amax_ptr).to(tl.float32)

    best_loss = tl.full([BLOCKS_PER_PROGRAM], float("inf"), dtype=tl.float32)
    best_idx = tl.zeros([BLOCKS_PER_PROGRAM], dtype=tl.int32)

    # Loop over the exhaustive or bounded FP8 candidates (compile-time unrolled).
    # Scales are guaranteed positive and finite (constructed from a positive candidate
    # times nonneg global_amax), so the degenerate-scale guard from nvfp4_scalar_quant is
    # unnecessary apart from the global_amax == 0 case handled below.
    for k in tl.static_range(NUM_CANDIDATES):
        if BOUNDED:
            code = tl.load(base_codes_ptr + block_idx, mask=block_mask, other=1).to(tl.int32)
            code = tl.maximum(1, tl.minimum(126, code + MIN_OFFSET + k))
            c = tl.load(candidates_ptr + code - 1).to(tl.float32)
            scale = (c * global_amax / 6.0)[:, None]
        else:
            c = tl.load(candidates_ptr + k).to(tl.float32)
            scale = c * global_amax / 6.0
        # Avoid divide-by-zero when global_amax == 0; in that case w_abs is also zero
        # (global_amax = max|w|), so the loss is zero for every candidate either way.
        scale_safe = tl.where(scale == 0.0, 1.0, scale)
        q_mag = fp4_round_magnitude(w_abs / scale_safe)
        diff = w_abs - q_mag * scale_safe
        loss = tl.sum(diff * diff, axis=1)  # [BLOCKS_PER_PROGRAM]
        is_better = loss < best_loss
        best_loss = tl.where(is_better, loss, best_loss)
        best_idx = tl.where(is_better, k, best_idx)

    # Map each block's winning candidate index back to its amax = global_amax * c[best].
    if BOUNDED:
        code = tl.load(base_codes_ptr + block_idx, mask=block_mask, other=1).to(tl.int32)
        code = tl.maximum(1, tl.minimum(126, code + MIN_OFFSET + best_idx))
        best_c = tl.load(candidates_ptr + code - 1, mask=block_mask, other=0.0).to(tl.float32)
    else:
        best_c = tl.load(candidates_ptr + best_idx, mask=block_mask, other=0.0).to(tl.float32)
    best_amax = global_amax * best_c
    tl.store(best_amax_ptr + block_idx, best_amax, mask=block_mask)


def _prepare_block_sweep(x: torch.Tensor, block_size: int):
    """Validate a block-sweep weight tensor; return ``(n_blocks, x_flat, best_amax)``.

    Shared by both FP8 scale-sweep entry points so their input contracts stay in sync.
    """
    if not x.is_cuda:
        raise ValueError("nvfp4 FP8 scale sweep requires a CUDA tensor.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}.")
    if x.numel() % block_size != 0:
        raise ValueError(f"x.numel() ({x.numel()}) is not divisible by block_size ({block_size}).")
    n_blocks = x.numel() // block_size
    x_flat = x.contiguous().view(-1)
    best_amax = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    return n_blocks, x_flat, best_amax


def nvfp4_fp8_scale_sweep(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    offset_range: tuple[int, int] | None = None,
    initial_amax: torch.Tensor | None = None,
    fp8_max_for_normalization: float = 448.0,
) -> torch.Tensor:
    """Find the per-block FP8 scale that minimizes NVFP4 quantization MSE.

    Equivalent to :class:`NVFP4MSECalibrator`'s exhaustive or bounded sweep, but fused
    into a single Triton kernel: every block's weight elements are loaded once, all
    candidates are evaluated in registers, and the running argmin is kept inline.

    Args:
        x: Weight tensor on CUDA. Total element count must be divisible by
            ``block_size``; layout is treated as a flat ``[N_BLOCKS, BLOCK_SIZE]``.
        global_amax: Scalar FP32 global amax (``= reduce_amax(per_block_amax)``).
        block_size: NVFP4 block size (typically 16).
        offset_range: Inclusive E4M3 code-offset range, or None for exhaustive search.
        initial_amax: Max-calibrated per-block amax used to derive the starting codes.
            Computed from ``x`` when omitted in bounded mode.
        fp8_max_for_normalization: FP8 normalization max (448, or 256 for 4/6 mode).

    Returns:
        ``best_amax`` of shape ``[N_BLOCKS]``, fp32, on the same device as ``x``.
    """
    n_blocks, x_flat, best_amax = _prepare_block_sweep(x, block_size)
    candidates = fp8_scale_candidates(
        x.device, fp8_max_for_normalization if offset_range is not None else 448.0
    ).to(dtype=torch.float32)
    global_amax_f32 = global_amax.detach().to(device=x.device, dtype=torch.float32).reshape(1)
    if offset_range is None:
        base_codes = torch.empty(1, dtype=torch.uint8, device=x.device)
        min_offset = 0
        num_candidates = int(candidates.numel())
    else:
        if initial_amax is None:
            initial_amax = x_flat.view(n_blocks, block_size).float().abs().amax(dim=-1)
        if initial_amax.numel() != n_blocks:
            raise ValueError(
                f"initial_amax.numel() ({initial_amax.numel()}) must equal n_blocks ({n_blocks})."
            )
        base_codes = fp8_scale_codes(
            initial_amax.reshape(-1), global_amax_f32, fp8_max_for_normalization
        )
        min_offset, max_offset = offset_range
        num_candidates = max_offset - min_offset + 1

    grid = lambda meta: (triton.cdiv(n_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    with torch.cuda.device(x.device):
        _fp8_scale_sweep_kernel[grid](
            x_flat,
            candidates,
            base_codes,
            global_amax_f32,
            best_amax,
            n_blocks,
            BLOCK_SIZE=block_size,
            NUM_CANDIDATES=num_candidates,
            MIN_OFFSET=min_offset,
            BOUNDED=offset_range is not None,
        )
    return best_amax


# Each program sweeps a tile of output rows of one cin-block, so the block's [BS, BS] Hessian
# loads once and dwᵀ H dw runs as a [ROWS, BS] x [BS, BS] tl.dot on tensor cores.
# ROWS_PER_PROGRAM=32 / num_warps=4 was fastest in a shape sweep; hard-coded (not autotuned)
# since there is no second config to tune over.
_HESSIAN_ROWS_PER_PROGRAM = 32
_HESSIAN_NUM_WARPS = 4


@triton.jit
def _fp8_scale_sweep_hessian_kernel(
    x_ptr,  # [COUT * N_CIN_BLOCKS * BLOCK_SIZE], any float dtype (loaded as fp32)
    hessian_ptr,  # [N_CIN_BLOCKS * BLOCK_SIZE * BLOCK_SIZE] fp32
    candidate_scales_ptr,  # [NUM_CANDIDATES] fp32: per-candidate FP8-quantized block scale
    candidate_amaxes_ptr,  # [NUM_CANDIDATES] fp32: per-candidate block amax (kernel output value)
    candidates_ptr,  # [126] normalized finite positive E4M3 values; bounded search only
    base_codes_ptr,  # [COUT * N_CIN_BLOCKS] uint8; bounded search only
    global_amax_ptr,  # scalar fp32; bounded search only
    best_amax_ptr,  # [COUT * N_CIN_BLOCKS] fp32 output
    COUT,
    N_CIN_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MIN_OFFSET: tl.constexpr,
    BOUNDED: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cin_block = pid % N_CIN_BLOCKS
    rows = (pid // N_CIN_BLOCKS) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    row_mask = rows < COUT
    # Block layout is row-major over (cout, cin // block_size): block = row*N_CIN + n.
    block_idx = rows * N_CIN_BLOCKS + cin_block

    # Signed residual: the Hessian cross-terms mean the sign does not cancel (unlike plain MSE).
    elem = tl.arange(0, BLOCK_SIZE)
    w = tl.load(
        x_ptr + block_idx[:, None] * BLOCK_SIZE + elem[None, :], mask=row_mask[:, None], other=0.0
    ).to(tl.float32)  # [ROWS, BS]
    w_abs = tl.abs(w)
    w_sign = tl.where(w >= 0, 1.0, -1.0)

    idx = tl.arange(0, BLOCK_SIZE)
    hessian = tl.load(
        hessian_ptr
        + cin_block * (BLOCK_SIZE * BLOCK_SIZE)
        + idx[:, None] * BLOCK_SIZE
        + idx[None, :]
    ).to(tl.float32)  # [BS, BS]

    best_loss = tl.full([ROWS_PER_PROGRAM], float("inf"), dtype=tl.float32)
    best_idx = tl.zeros([ROWS_PER_PROGRAM], dtype=tl.int32)
    global_amax = tl.load(global_amax_ptr).to(tl.float32)

    # Non-unrolled loop: unrolling 126 tl.dot bodies explodes compile time for no runtime gain.
    for k in tl.range(NUM_CANDIDATES):
        if BOUNDED:
            code = tl.load(base_codes_ptr + block_idx, mask=row_mask, other=1).to(tl.int32)
            code = tl.maximum(1, tl.minimum(126, code + MIN_OFFSET + k))
            candidate = tl.load(candidates_ptr + code - 1).to(tl.float32)
            scale = (candidate * global_amax / 6.0)[:, None]
        else:
            scale = tl.load(candidate_scales_ptr + k).to(tl.float32)
        scale_safe = tl.where(scale == 0.0, 1.0, scale)  # scale == 0 only if global_amax == 0
        q_mag = fp4_round_magnitude(w_abs / scale_safe)
        dw = w_sign * (w_abs - q_mag * scale_safe)  # = w - quant(w), [ROWS, BS]
        # dwᵀ H dw per row (H symmetric); allow_tf32=False keeps it true fp32 vs the reference.
        hdw = tl.dot(dw, hessian, allow_tf32=False)  # [ROWS, BS]
        loss = tl.sum(hdw * dw, axis=1)  # [ROWS]
        is_better = loss < best_loss
        best_loss = tl.where(is_better, loss, best_loss)
        best_idx = tl.where(is_better, k, best_idx)

    if BOUNDED:
        code = tl.load(base_codes_ptr + block_idx, mask=row_mask, other=1).to(tl.int32)
        code = tl.maximum(1, tl.minimum(126, code + MIN_OFFSET + best_idx))
        best_amax = (
            tl.load(candidates_ptr + code - 1, mask=row_mask, other=0.0).to(tl.float32)
            * global_amax
        )
    else:
        best_amax = tl.load(candidate_amaxes_ptr + best_idx, mask=row_mask, other=0.0).to(
            tl.float32
        )
    tl.store(best_amax_ptr + block_idx, best_amax, mask=row_mask)


def nvfp4_fp8_scale_sweep_hessian(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    hessian: torch.Tensor,
    block_size: int = 16,
    offset_range: tuple[int, int] | None = None,
    initial_amax: torch.Tensor | None = None,
    fp8_max_for_normalization: float = 448.0,
) -> torch.Tensor:
    """Find the per-block FP8 scale minimizing the Hessian-weighted NVFP4 quant error.

    Hessian-weighted counterpart of :func:`nvfp4_fp8_scale_sweep`: for each NVFP4 block
    it minimizes ``dwᵀ H dw`` (``dw = w - quant(w)``) over the exhaustive or bounded
    FP8 E4M3 candidates, where ``H`` is the per-cin-block local Hessian shared across all
    output rows. Used by :class:`NVFP4MSECalibrator` for ``local_hessian`` calibration.

    Args:
        x: Weight tensor on CUDA in the blocked ``[N_BLOCKS, block_size]`` layout, row-major
            over ``(cout, cin // block_size)`` so flat block ``b`` has cin-block
            ``b % (cin // block_size)``.
        global_amax: Scalar FP32 global amax (``= reduce_amax(per_block_amax)``).
        hessian: Per-cin-block Hessian of shape ``[cin // block_size, block_size, block_size]``,
            fp32 (typically normalized by sample count).
        block_size: NVFP4 block size (typically 16).
        offset_range: Inclusive E4M3 code-offset range, or None for exhaustive search.
        initial_amax: Max-calibrated per-block amax used to derive the starting codes.
            Computed from ``x`` when omitted in bounded mode.
        fp8_max_for_normalization: FP8 normalization max (448, or 256 for 4/6 mode).

    Returns:
        ``best_amax`` of shape ``[N_BLOCKS]``, fp32, on the same device as ``x``.
    """
    n_blocks, x_flat, best_amax = _prepare_block_sweep(x, block_size)
    if hessian.dim() != 3 or hessian.shape[1] != block_size or hessian.shape[2] != block_size:
        raise ValueError(
            f"hessian must have shape [n_cin_blocks, {block_size}, {block_size}], "
            f"got {tuple(hessian.shape)}."
        )
    n_cin_blocks = hessian.shape[0]
    if n_blocks % n_cin_blocks != 0:
        raise ValueError(
            f"n_blocks ({n_blocks}) is not divisible by n_cin_blocks ({n_cin_blocks})."
        )

    cout = n_blocks // n_cin_blocks
    grid = (triton.cdiv(cout, _HESSIAN_ROWS_PER_PROGRAM) * n_cin_blocks,)
    with torch.cuda.device(x.device):
        global_amax_f32 = global_amax.detach().to(device=x.device, dtype=torch.float32).reshape(())
        # Candidate scales via the reference ``compute_fp4_scales`` (the exact fake-quant path)
        # keep the kernel's residual bit-identical to the reference sweep.
        candidate_amaxes = fp8_scale_candidates(x.device).to(dtype=torch.float32) * global_amax_f32
        candidate_scales = compute_fp4_scales(
            candidate_amaxes, global_amax_f32, quantize_block_scales=True
        ).to(dtype=torch.float32)
        if offset_range is None:
            base_codes = torch.empty(1, dtype=torch.uint8, device=x.device)
            min_offset = 0
            num_candidates = int(candidate_amaxes.numel())
        else:
            if initial_amax is None:
                initial_amax = x_flat.view(n_blocks, block_size).float().abs().amax(dim=-1)
            if initial_amax.numel() != n_blocks:
                raise ValueError(
                    f"initial_amax.numel() ({initial_amax.numel()}) must equal "
                    f"n_blocks ({n_blocks})."
                )
            base_codes = fp8_scale_codes(
                initial_amax.reshape(-1), global_amax_f32, fp8_max_for_normalization
            )
            min_offset, max_offset = offset_range
            num_candidates = max_offset - min_offset + 1
        hessian_flat = hessian.contiguous().to(device=x.device, dtype=torch.float32).view(-1)
        _fp8_scale_sweep_hessian_kernel[grid](
            x_flat,
            hessian_flat,
            candidate_scales,
            candidate_amaxes,
            fp8_scale_candidates(
                x.device, fp8_max_for_normalization if offset_range is not None else 448.0
            ).to(dtype=torch.float32),
            base_codes,
            global_amax_f32,
            best_amax,
            cout,
            n_cin_blocks,
            BLOCK_SIZE=block_size,
            NUM_CANDIDATES=num_candidates,
            MIN_OFFSET=min_offset,
            BOUNDED=offset_range is not None,
            ROWS_PER_PROGRAM=_HESSIAN_ROWS_PER_PROGRAM,
            num_warps=_HESSIAN_NUM_WARPS,
        )
    return best_amax
