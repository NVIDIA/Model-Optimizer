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

"""Closed-form numerics for the bit-exact MXFP4 -> NVFP4 weight cast.

These helpers turn an MXFP4 source layer's E8M0 block scales into the per-tensor
``global_amax`` and per-NVFP4-block ``amax`` that pin NVFP4's two-level scale so
the cast reproduces the source MXFP4 weights bit-for-bit (see PR #1372 for the
derivation). They are pure tensor math with no model or checkpoint dependencies,
shared by the GPT-OSS (``examples/llm_ptq``) and DeepSeek-V4
(``examples/deepseek``) PTQ cast paths.
"""

import torch

__all__ = [
    "E2M1_MAX",
    "E4M3_KMAX",
    "E4M3_KMIN",
    "E4M3_MAX",
    "E8M0_BIAS",
    "mxfp4_to_nvfp4_global_amax",
    "mxfp4_to_nvfp4_per_block_amax",
]

E8M0_BIAS = 127  # E8M0 stores k_j as uint8 with bias 127
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_KMAX = 8
E4M3_KMIN = -9  # E4M3 represents 2^k exactly for k in [-9, 8]
# E2M1 magnitude grid indexed by the low 3 bits of an FP4 nibble.
_E2M1_MAGNITUDE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Cache of the E2M1 magnitude lookup table per (device, dtype) so we don't
# rebuild it for every layer in a batched cast.
_E2M1_MAG_CACHE: "dict[tuple, torch.Tensor]" = {}


def _e2m1_magnitude_table(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return ``_E2M1_MAGNITUDE`` as a tensor on the requested device, cached."""
    key = (device, dtype)
    cached = _E2M1_MAG_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(_E2M1_MAGNITUDE, dtype=dtype, device=device)
        _E2M1_MAG_CACHE[key] = cached
    return cached


def mxfp4_to_nvfp4_global_amax(e8m0_scales: torch.Tensor) -> tuple[float, dict]:
    """Closed-form per-tensor ``global_amax``: ``m = k_max - 8``, ``global_amax = 6 * 448 * 2^m``.

    Args:
        e8m0_scales: uint8 tensor of E8M0 scales for one MXFP4 source layer.

    Returns:
        global_amax: scalar (float) — pins NVFP4 scale_2 to 2^m.
        info: diagnostic dict with k_min, k_max, m, lossless-block stats.
    """
    # k_j = e8m0 - 127. MXFP4 quantize emits e8m0=0 (=> k=-127) for all-zero
    # blocks; treat those as "ignore me" when computing k_max.
    k = e8m0_scales.to(torch.int32) - E8M0_BIAS
    nonzero_mask = e8m0_scales > 0
    if nonzero_mask.any():
        k_nonzero = k[nonzero_mask]
        k_min = int(k_nonzero.min().item())
        k_max = int(k_nonzero.max().item())
    else:
        k_min = k_max = 0

    m = k_max - E4M3_KMAX
    global_amax = E2M1_MAX * E4M3_MAX * float(2.0**m)

    # A block is lossless under this cast iff k_max - k_j <= 17 (its k_j - m sits
    # in E4M3's [-9, 8] window). All-zero blocks are trivially lossless because
    # their reconstruction is 0 regardless of the snapped scale.
    n_total = e8m0_scales.numel()
    in_range = (k >= (k_max - 17)) | (~nonzero_mask)
    n_lossless = int(in_range.sum().item())
    pct_lossless = 100.0 * n_lossless / n_total if n_total else 100.0

    return global_amax, {
        "k_min": k_min,
        "k_max": k_max,
        "m": m,
        "n_total_blocks": n_total,
        "n_lossless_blocks": n_lossless,
        "pct_lossless": pct_lossless,
        "n_zero_blocks": int((~nonzero_mask).sum().item()),
    }


def mxfp4_to_nvfp4_per_block_amax(blocks: torch.Tensor, e8m0_scales: torch.Tensor) -> torch.Tensor:
    """Hybrid per-NVFP4-block amax for MXFP4 -> NVFP4 cast.

    Each MXFP4 block of 32 elements has one E8M0 exponent ``k_j``. Two cases
    based on whether ``k_j`` fits in NVFP4's E4M3 scale grid (with
    ``m = k_max - 8`` chosen by ``mxfp4_to_nvfp4_global_amax``):

    - **In-range** (``k_j - m`` in ``[-9, 8]``): ``6 * 2^k_j`` (closed-form
      ideal). The resulting per-block scale ``2^(k_j - m)`` is exactly
      representable in E4M3 — no rounding loss — and
      ``round_to_E2M1(value / 2^k_j)`` yields the original MXFP4 nibble
      verbatim. Bit-exact reconstruction.

    - **Out of range** (``|k_j - m| > 8/9``): ``max_nibble * 2^k_j``, i.e.
      ``max(|w_block|)`` where ``w`` is the MXFP4-dequantized block. This is
      the data-derived per-block amax. The per-block scale will still get
      clamped at the E4M3 boundary, but data-derived amax keeps the post-clamp
      scale closer to the block's actual magnitude than the closed-form ideal
      would, which reduces re-bucketing error for OOR blocks where
      ``max_nibble < 6``.

    Two NVFP4 blocks of 16 share each MXFP4 block's ``k_j``, so the result is
    expanded by ``repeat_interleave(2, dim=-1)``.

    Args:
        blocks: uint8 tensor of packed E2M1 nibbles, shape
            ``(..., num_mxfp4_blocks, 16)`` (16 bytes per 32-element MXFP4 block).
        e8m0_scales: uint8 tensor of E8M0 scales, shape
            ``(..., num_mxfp4_blocks)``.

    Returns:
        float32 tensor of shape ``(..., 2 * num_mxfp4_blocks)``.
    """
    if blocks.shape[-1] != 16 or blocks.shape[:-1] != e8m0_scales.shape:
        raise ValueError(
            f"shape mismatch: blocks {tuple(blocks.shape)} "
            "(expected (..., num_mxfp4_blocks, 16)) "
            f"vs scales {tuple(e8m0_scales.shape)}"
        )

    k = e8m0_scales.to(torch.int32) - E8M0_BIAS  # (..., num_mxfp4_blocks)
    pow2_k = torch.exp2(k.float())
    closed_form_ideal = E2M1_MAX * pow2_k  # (..., num_mxfp4_blocks)

    # ``m = k_max - 8`` over non-zero blocks. Compute via masked ``amax`` so
    # ``m`` stays a 0-d tensor and we avoid a GPU->CPU sync just to get a
    # Python int. All-zero scales fall through with the -E8M0_BIAS sentinel,
    # which leaves every block trivially in-range (closed_form_ideal == 0 there).
    nonzero = e8m0_scales > 0
    sentinel = torch.full_like(k, -E8M0_BIAS)
    k_max = torch.where(nonzero, k, sentinel).amax()
    delta = k - (k_max - E4M3_KMAX)
    in_range = (delta >= E4M3_KMIN) & (delta <= E4M3_KMAX)

    # Fast path: if every block fits E4M3's [-9, 8] window the per-block amax
    # is just the closed-form ideal, and we can skip the per-byte nibble scan
    # over the block tensor (which is 16x larger than the scales). For typical
    # MXFP4 checkpoints (e.g. gpt-oss-20b) this is the only path ever taken.
    if bool(in_range.all()):
        return closed_form_ideal.repeat_interleave(2, dim=-1)

    # OOR fallback: data-derived per-block amax = max(|w_block|) after MXFP4
    # dequant = ``max_nibble * 2^k_j``. The MXFP4 nibble is sign-magnitude with
    # sign in bit 3 and magnitude index in bits 0-2; we extract per-byte
    # magnitudes, take the byte-wise max, then reduce across the 16 bytes to
    # get the largest magnitude index in the 32-element block.
    low = blocks & 0x07
    high = (blocks >> 4) & 0x07
    max_idx = torch.maximum(low, high).amax(dim=-1).long()
    max_nibble = _e2m1_magnitude_table(blocks.device)[max_idx]
    data_derived = max_nibble * pow2_k

    per_block_amax_mxfp4 = torch.where(in_range, closed_form_ideal, data_derived)
    # Each MXFP4 block of 32 splits into two NVFP4 blocks of 16 sharing k_j.
    return per_block_amax_mxfp4.repeat_interleave(2, dim=-1)
