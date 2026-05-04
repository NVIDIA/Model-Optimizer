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

Replaces the 126-iteration Python sweep in :class:`NVFP4MSECalibrator` with a single
kernel that, for each NVFP4 block, evaluates all 126 valid FP8 E4M3 scale candidates
and emits the per-block ``best_amax`` directly.

The 126 candidates are constructed as ``valid_fp8_e4m3_value / 448`` (see
:func:`fp8_scale_candidates`). For these specific candidates, the FP8 round-trip on
the per-block scale is the identity, so the kernel can use
``scale = candidate * global_amax / 6.0`` without an explicit FP8 cast — making it
runnable on any CUDA GPU with Triton (no ``tl.float8e4nv`` requirement).

Tile shape (``BLOCKS_PER_PROGRAM``) and ``num_warps`` are autotuned per ``N_BLOCKS``.
"""

import torch
import triton
import triton.language as tl

from .nvfp4_quant import fp4_round_magnitude

__all__ = ["fp8_scale_candidates", "nvfp4_fp8_scale_sweep"]


def fp8_scale_candidates(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the 126 valid finite positive FP8 E4M3 scale candidates / 448."""
    uint8_values = torch.arange(0, 128, dtype=torch.uint8, device=device)
    fp8_values = uint8_values.view(torch.float8_e4m3fn).float()
    valid_mask = torch.isfinite(fp8_values) & (fp8_values > 0)
    return fp8_values[valid_mask] / 448.0


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
    global_amax_ptr,  # scalar fp32
    best_amax_ptr,  # [N_BLOCKS] fp32 output
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
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

    # Loop over the 126 FP8 candidates (compile-time unrolled).
    # Scales are guaranteed positive and finite (constructed from a positive candidate
    # times nonneg global_amax), so the degenerate-scale guard from nvfp4_scalar_quant is
    # unnecessary apart from the global_amax == 0 case handled below.
    for k in tl.static_range(NUM_CANDIDATES):
        c = tl.load(candidates_ptr + k).to(tl.float32)
        scale = c * global_amax / 6.0
        # Avoid divide-by-zero when global_amax == 0; the resulting err == w_abs² is
        # the same for every candidate, so any best_idx is fine.
        scale_safe = tl.where(scale == 0.0, 1.0, scale)
        q_mag = fp4_round_magnitude(w_abs / scale_safe)
        diff = w_abs - q_mag * scale
        loss = tl.sum(diff * diff, axis=1)  # [BLOCKS_PER_PROGRAM]
        is_better = loss < best_loss
        best_loss = tl.where(is_better, loss, best_loss)
        best_idx = tl.where(is_better, k, best_idx)

    # Map each block's winning candidate index back to its amax = global_amax * c[best].
    best_c = tl.load(candidates_ptr + best_idx, mask=block_mask, other=0.0).to(tl.float32)
    best_amax = global_amax * best_c
    tl.store(best_amax_ptr + block_idx, best_amax, mask=block_mask)


def nvfp4_fp8_scale_sweep(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block_size: int = 16,
    candidates: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find the per-block FP8 scale that minimizes NVFP4 quantization MSE.

    Equivalent to the 126-step sweep in :class:`NVFP4MSECalibrator`, but fused into
    a single Triton kernel: every block's weight elements are loaded once, all 126
    candidates are evaluated in registers, and the running argmin is kept inline.

    Args:
        x: Weight tensor on CUDA. Total element count must be divisible by
            ``block_size``; layout is treated as a flat ``[N_BLOCKS, BLOCK_SIZE]``.
        global_amax: Scalar FP32 global amax (``= reduce_amax(per_block_amax)``).
        block_size: NVFP4 block size (typically 16).
        candidates: Optional precomputed candidate tensor of shape ``[126]`` (must
            be the FP8 E4M3 valid values divided by 448). Built lazily if omitted.

    Returns:
        ``best_amax`` of shape ``[N_BLOCKS]``, fp32, on the same device as ``x``.
    """
    assert x.is_cuda, "nvfp4_fp8_scale_sweep requires a CUDA tensor"
    if x.numel() % block_size != 0:
        raise ValueError(f"x.numel() ({x.numel()}) is not divisible by block_size ({block_size}).")

    if candidates is None:
        candidates = fp8_scale_candidates(x.device)
    candidates = candidates.contiguous().to(device=x.device, dtype=torch.float32)

    n_blocks = x.numel() // block_size
    x_flat = x.contiguous().view(-1)
    global_amax_f32 = global_amax.detach().to(device=x.device, dtype=torch.float32).reshape(1)
    best_amax = torch.empty(n_blocks, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(n_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    with torch.cuda.device(x.device):
        _fp8_scale_sweep_kernel[grid](
            x_flat,
            candidates,
            global_amax_f32,
            best_amax,
            n_blocks,
            BLOCK_SIZE=block_size,
            NUM_CANDIDATES=int(candidates.numel()),
        )
    return best_amax
