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

"""Diagnostic Triton kernel for Decoupled Scale Search (DSS) on the NVFP4 weight-MSE sweep.

DSS (SOAR, arXiv:2605.12245) decouples the FP4 *quantization* scale ``s_q`` (high precision)
from the stored *dequantization* scale ``s_d`` (FP8 E4M3-constrained)::

    for s_d in nearby_E4M3_scales:  # stored, FP8-constrained
        for s_q in nearby_real_scales:  # free, high-precision
            q = round_fp4(w / s_q)
            w_hat = q * s_d

This kernel does NOT change calibration. It only *measures* how much DSS could reduce the
per-block reconstruction loss versus the coupled (``s_q == s_d``) sweep that
:class:`NVFP4MSECalibrator` already runs. For each NVFP4 block it sweeps every FP8 ``s_d``
candidate against a multiplicative ``s_q = beta * s_d`` grid and reports, per block:

  * ``coupled_loss`` — best loss over ``s_d`` at ``beta == 1`` (reproduces today's sweep),
  * ``dss_loss``     — best loss over the full ``(s_d, beta)`` grid,
  * ``best_beta``    — the ``beta`` at the DSS optimum.

Because ``beta == 1`` is in the grid, ``dss_loss <= coupled_loss`` by construction; the
diagnostic question is whether ``coupled_loss - dss_loss`` is ever materially positive.
"""

import torch
import triton
import triton.language as tl

from ._fp8_scale_candidates import fp8_scale_candidates
from .nvfp4_quant import fp4_round_magnitude

__all__ = ["default_dss_betas", "nvfp4_dss_diag_sweep"]


def default_dss_betas(device: torch.device | str = "cpu") -> torch.Tensor:
    """SOAR's quant-scale grid: ``s_q = beta * s_d`` for ``beta`` in ``[0.50, 1.50]`` step 0.01.

    Built as ``arange(50, 151) / 100`` so the coupled reference ``beta == 1.0`` is exactly
    representable (``100 / 100``), which :func:`nvfp4_dss_diag_sweep` requires.
    """
    return torch.arange(50, 151, dtype=torch.float32, device=device) / 100.0


# Mirror nvfp4_fp8_sweep.py's tile/warp shape. Both sweep loops are runtime ``tl.range`` (not
# unrolled): a 126x101 ``static_range`` unroll overruns the NVPTX compiler. The candidate and
# beta tables are read from global memory per iteration instead.
_DSS_DIAG_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 16}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=8),
]


@triton.autotune(configs=_DSS_DIAG_AUTOTUNE_CONFIGS, key=["N_BLOCKS"])
@triton.jit
def _dss_diag_sweep_kernel(
    x_ptr,  # [N_BLOCKS * BLOCK_SIZE], any float dtype (loaded as fp32)
    candidates_ptr,  # [NUM_CANDIDATES] fp32 (FP8 E4M3 values / 448)
    betas_ptr,  # [NUM_BETAS] fp32 quant-scale multipliers
    global_amax_ptr,  # scalar fp32
    coupled_loss_ptr,  # [N_BLOCKS] fp32 output (best loss at beta == 1)
    dss_loss_ptr,  # [N_BLOCKS] fp32 output (best loss over full grid)
    best_beta_ptr,  # [N_BLOCKS] fp32 output (beta at the dss optimum)
    N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES,
    NUM_BETAS,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCKS_PER_PROGRAM
    block_idx = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_idx < N_BLOCKS

    # Load weights once; |w| suffices since FP4 quant preserves sign:
    #   (w - w_q)^2 = (|w| - q_mag * s_d)^2.
    elem_offs = block_idx[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    elem_mask = block_mask[:, None]
    w_abs = tl.abs(tl.load(x_ptr + elem_offs, mask=elem_mask, other=0.0).to(tl.float32))

    global_amax = tl.load(global_amax_ptr).to(tl.float32)

    best_dss_loss = tl.full([BLOCKS_PER_PROGRAM], float("inf"), dtype=tl.float32)
    best_beta = tl.zeros([BLOCKS_PER_PROGRAM], dtype=tl.float32)
    best_coupled_loss = tl.full([BLOCKS_PER_PROGRAM], float("inf"), dtype=tl.float32)

    for k in tl.range(NUM_CANDIDATES):
        c = tl.load(candidates_ptr + k).to(tl.float32)
        s_d = c * global_amax / 6.0
        # global_amax == 0 implies w_abs == 0 (global_amax = max|w|), so every loss is 0.
        s_d_safe = tl.where(s_d == 0.0, 1.0, s_d)
        for j in tl.range(NUM_BETAS):
            beta = tl.load(betas_ptr + j).to(tl.float32)
            s_q = beta * s_d_safe
            q_mag = fp4_round_magnitude(w_abs / s_q)
            diff = w_abs - q_mag * s_d_safe
            loss = tl.sum(diff * diff, axis=1)  # [BLOCKS_PER_PROGRAM]

            is_better = loss < best_dss_loss
            best_dss_loss = tl.where(is_better, loss, best_dss_loss)
            best_beta = tl.where(is_better, beta, best_beta)

            # betas holds exactly one 1.0 entry -> this captures the coupled (s_q == s_d) slice.
            is_coupled = beta == 1.0
            best_coupled_loss = tl.where(
                is_coupled & (loss < best_coupled_loss), loss, best_coupled_loss
            )

    tl.store(coupled_loss_ptr + block_idx, best_coupled_loss, mask=block_mask)
    tl.store(dss_loss_ptr + block_idx, best_dss_loss, mask=block_mask)
    tl.store(best_beta_ptr + block_idx, best_beta, mask=block_mask)


def nvfp4_dss_diag_sweep(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    betas: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure DSS vs. coupled NVFP4 per-block reconstruction loss.

    Args:
        x: Weight tensor on CUDA. Total element count must be divisible by ``block_size``;
            layout is treated as a flat ``[N_BLOCKS, BLOCK_SIZE]`` (matching the calibrator).
        global_amax: Scalar FP32 global amax (``= reduce_amax(per_block_amax)``).
        betas: 1-D FP32 quant-scale multipliers ``s_q = beta * s_d``. Must contain exactly one
            entry equal to ``1.0`` (the coupled reference); its index drives ``coupled_loss``.
        block_size: NVFP4 block size (typically 16).

    Returns:
        ``(coupled_loss, dss_loss, best_beta)``, each shape ``[N_BLOCKS]`` fp32 on ``x``'s
        device. ``dss_loss <= coupled_loss`` elementwise by construction.
    """
    if not x.is_cuda:
        raise ValueError("nvfp4_dss_diag_sweep requires a CUDA tensor.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}.")
    if x.numel() % block_size != 0:
        raise ValueError(f"x.numel() ({x.numel()}) is not divisible by block_size ({block_size}).")

    betas = betas.detach().to(device=x.device, dtype=torch.float32).contiguous()
    if int((betas == 1.0).sum()) != 1:
        raise ValueError("betas must contain exactly one entry equal to 1.0 (coupled reference).")

    candidates = fp8_scale_candidates(x.device).to(dtype=torch.float32)

    n_blocks = x.numel() // block_size
    x_flat = x.contiguous().view(-1)
    global_amax_f32 = global_amax.detach().to(device=x.device, dtype=torch.float32).reshape(1)
    coupled_loss = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    dss_loss = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    best_beta = torch.empty(n_blocks, dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(n_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    with torch.cuda.device(x.device):
        _dss_diag_sweep_kernel[grid](
            x_flat,
            candidates,
            betas,
            global_amax_f32,
            coupled_loss,
            dss_loss,
            best_beta,
            n_blocks,
            BLOCK_SIZE=block_size,
            NUM_CANDIDATES=int(candidates.numel()),
            NUM_BETAS=int(betas.numel()),
        )
    return coupled_loss, dss_loss, best_beta
