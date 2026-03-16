# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pure-PyTorch reference implementation of NVFP4 "four over six" fake quantization.

Each 16-element block independently chooses between a 4-bit or 6-bit FP8 scale
encoding to minimize quantization error (MSE/MAE/abs_max), following the approach
described in arxiv:2512.02010.
"""

from __future__ import annotations

import torch

__all__ = ["nvfp4_4o6_fake_quant"]

_E2M1_MAX = 6
_E2M1_MAX_FOUR = 4
_E4M3_MAX_FOUROVERSIX = 256


def _fake_quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """E2M1 round-trip fake quantization (nearest rounding)."""
    step1 = torch.round(2 * x.abs()) / 2
    step2 = torch.round(x.abs())
    step3 = 2 * torch.round(x.abs() / 2)

    mask1 = x.abs() < 2
    mask2 = x.abs() < 4

    return x.sign() * (step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2))


def _quantize_to_nvfp4(
    x_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_expansion_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-block FP8 scales and return block-scaled activations.

    Args:
        x_blocks: Tensor of shape (num_blocks, block_size), float32.
        x_amax: Scalar float32 tensor — global amax of the full activation tensor.
        scale_expansion_factor: If provided, multiply computed scales by this factor
            (1.5 for the 4-bit path).

    Returns:
        (x_block_scaled, scales) where x_block_scaled has the same shape as x_blocks
        and scales is float8_e4m3fn of shape (num_blocks,).
    """
    if x_amax == 0:
        x_scales_hp = torch.zeros(
            x_blocks.shape[0],
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    else:
        encode_scale = (
            torch.tensor(
                _E2M1_MAX * _E4M3_MAX_FOUROVERSIX,
                dtype=x_amax.dtype,
                device=x_amax.device,
            )
            / x_amax
        )
        x_scales_hp = (
            x_blocks.abs().max(dim=-1).values
            / torch.tensor(_E2M1_MAX, dtype=x_amax.dtype, device=x_amax.device)
            * encode_scale
        )

    if scale_expansion_factor is not None:
        x_scales_hp = x_scales_hp * scale_expansion_factor

    x_scales = x_scales_hp.to(torch.float8_e4m3fn)

    decode_scale = (
        torch.tensor(
            _E2M1_MAX * _E4M3_MAX_FOUROVERSIX,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
        / x_amax
    )
    x_block_scaled = torch.where(
        x_scales.unsqueeze(1) != 0,
        x_blocks * (1 / (x_scales.to(x_amax.dtype).unsqueeze(1) / decode_scale)),
        torch.zeros_like(x_blocks),
    )

    return x_block_scaled, x_scales


def _select_fouroversix(
    x_blocks: torch.Tensor,
    x_scaled_6: torch.Tensor,
    scales_6: torch.Tensor,
    x_scaled_4: torch.Tensor,
    scales_4: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block adaptive 4-or-6 bit scale selection with fake dequantization.

    Args:
        x_blocks: Original blocks, shape (num_blocks, block_size), float32.
        x_scaled_6: Block-scaled activations from 6-bit path.
        scales_6: FP8 scales from 6-bit path, shape (num_blocks,).
        x_scaled_4: Block-scaled activations from 4-bit path.
        scales_4: FP8 scales from 4-bit path, shape (num_blocks,).
        x_amax: Scalar float32 global amax.
        scale_rule: Error metric — "mse", "mae", or "abs_max".

    Returns:
        (x_fake_quantized, scales) where x_fake_quantized has shape
        (num_blocks, block_size) and scales has shape (num_blocks, 1).
    """
    x_fq_6 = _fake_quantize_to_e2m1(x_scaled_6)
    x_fq_4 = _fake_quantize_to_e2m1(x_scaled_4)

    denom = torch.tensor(
        _E2M1_MAX * _E4M3_MAX_FOUROVERSIX,
        dtype=x_amax.dtype,
        device=x_amax.device,
    )
    x_deq_6 = x_fq_6.to(x_amax.dtype) * scales_6.unsqueeze(1).to(x_amax.dtype) * x_amax / denom
    x_deq_4 = x_fq_4.to(x_amax.dtype) * scales_4.unsqueeze(1).to(x_amax.dtype) * x_amax / denom

    if scale_rule == "abs_max":
        err_4 = (x_deq_4 - x_blocks).abs().max(dim=-1).values
        err_6 = (x_deq_6 - x_blocks).abs().max(dim=-1).values
    elif scale_rule == "mae":
        err_4 = (x_deq_4 - x_blocks).abs().sum(dim=-1)
        err_6 = (x_deq_6 - x_blocks).abs().sum(dim=-1)
    elif scale_rule == "mse":
        err_4 = ((x_deq_4 - x_blocks) ** 2).sum(dim=-1)
        err_6 = ((x_deq_6 - x_blocks) ** 2).sum(dim=-1)
    else:
        raise ValueError(
            f"Unknown scale_rule: {scale_rule!r}. Expected 'mse', 'mae', or 'abs_max'."
        )

    select_4 = (err_4 < err_6).unsqueeze(1)  # (num_blocks, 1)
    x_fake_quantized = torch.where(
        select_4,
        x_fq_4.reshape(x_blocks.shape[0], -1),
        x_fq_6.reshape(x_blocks.shape[0], -1),
    )
    scales = torch.where(
        select_4,
        scales_4.reshape(-1, 1).to(x_amax.dtype),
        scales_6.reshape(-1, 1).to(x_amax.dtype),
    )

    return x_fake_quantized, scales


def nvfp4_4o6_fake_quant(
    x: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: str = "mse",
    block_size: int = 16,
) -> torch.Tensor:
    """NVFP4 "four over six" fake quantization.

    Each block of ``block_size`` elements independently chooses between a 4-bit
    or 6-bit FP8 block scale to minimize the per-block quantization error, as
    described in arxiv:2512.02010.

    Args:
        x: Input activation tensor of any shape. The last dimension (flattened)
            must be divisible by ``block_size``.
        x_amax: Scalar float32 tensor — global absolute maximum of ``x``.
        scale_rule: Error metric used for scale selection. One of
            ``"mse"`` (default), ``"mae"``, or ``"abs_max"``.
        block_size: Number of elements per quantization block. Default: 16.

    Returns:
        Fake-quantized tensor with the same shape and dtype as ``x``.

    Raises:
        ValueError: If the total number of elements is not divisible by ``block_size``.
    """
    if x.numel() % block_size != 0:
        raise ValueError(
            f"Total number of elements ({x.numel()}) must be divisible by "
            f"block_size ({block_size})."
        )

    orig_shape = x.shape
    orig_dtype = x.dtype

    x_blocks = x.reshape(-1, block_size).float()
    x_amax = x_amax.float()

    x_scaled_6, scales_6 = _quantize_to_nvfp4(x_blocks, x_amax)
    x_scaled_4, scales_4 = _quantize_to_nvfp4(x_blocks, x_amax, scale_expansion_factor=1.5)

    x_fake_q, scales = _select_fouroversix(
        x_blocks, x_scaled_6, scales_6, x_scaled_4, scales_4, x_amax, scale_rule=scale_rule
    )

    # Dequantize: scales already float, shape (num_blocks, 1)
    denom = _E2M1_MAX * _E4M3_MAX_FOUROVERSIX
    x_out = x_fake_q * scales * x_amax / denom

    return x_out.reshape(orig_shape).to(orig_dtype)
