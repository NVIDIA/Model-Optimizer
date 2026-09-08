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

"""Headroom calibrator for the NVFP4 activation global scale."""

import warnings

import torch

from ..utils import reduce_block_amax
from .calibrator import _Calibrator

__all__ = ["NVFP4ActHeadroomCalibrator"]

# FP8-E4M3 normal dynamic range (max_normal / min_normal = 448 / 2**-6). Per-block scales
# outside this ratio cannot all be represented as normal FP8 values by one global scale.
_FP8_NORMAL_DYNAMIC_RANGE = 28672.0

# Per-block amaxes below ``upper / _ANCHOR_FLOOR_RATIO`` are ignored when locating the
# anchor, so a tail of near-zero blocks cannot drag the global scale down.
_ANCHOR_FLOOR_RATIO = 1e6


class NVFP4ActHeadroomCalibrator(_Calibrator):
    """Calibrates the NVFP4 activation global scale with headroom for unseen outliers.

    NVFP4 scales a tensor in two levels: an FP8-E4M3 scale per 16-element block, and one
    per-tensor global scale (``amax``). Plain max calibration sets the global scale from the
    largest block seen during calibration, which leaves no room above it: any activation
    larger than the calibration max saturates.

    This calibrator instead anchors the global scale to a low percentile of the per-block
    amax distribution::

        amax = max(rho * anchor, upper)

    where ``anchor`` is the per-block amax at ``anchor_percentile`` and ``upper`` is the
    per-block amax at ``upper_percentile``. Multiplying the low anchor by ``rho`` places the
    calibrated blocks in the lower part of the FP8 scale range and leaves the rest as upward
    headroom; ``upper`` is the top of the range the scale commits to representing.

    ``upper_percentile`` defaults to 99.99 rather than the literal maximum on purpose. A single
    freak block far above the rest would otherwise drag the global scale up so far that every
    other block's FP8 block scale falls below subnormal and flushes to zero -- losing the whole
    tensor to protect one value. The default clips those rare blocks instead. Set
    ``upper_percentile=100`` to use the literal observed max, which guarantees no calibration
    data is clipped at the cost of that exposure.

    Args:
        num_bits: quantizer ``num_bits`` (``(2, 1)`` for NVFP4); kept for interface parity.
        axis: unused (the global scale is per-tensor); kept for interface parity.
        unsigned: unused; kept for interface parity.
        block_size: NVFP4 block width along the last dim.
        anchor_percentile: percentile of the per-block amaxes used as the anchor; must be > 0.
        upper_percentile: percentile of the per-block amaxes used as the top of the calibrated
            range. ``100`` uses the literal observed max (no calibration data is clipped).
        rho: headroom factor applied to the anchor. Must be in ``(0, 28672)``.
        num_bins: number of log2 histogram bins.
        log2_min: low end of the log2 range covered by the histogram.
        log2_max: high end of the log2 range covered by the histogram.
    """

    def __init__(
        self,
        num_bits=(2, 1),
        axis=None,
        unsigned=False,
        *,
        block_size=16,
        anchor_percentile=1.0,
        upper_percentile=99.99,
        rho=16384.0,
        num_bins=512,
        log2_min=-40.0,
        log2_max=40.0,
    ):
        """Initialize."""
        super().__init__(num_bits, axis, unsigned)
        if not (0.0 < rho < _FP8_NORMAL_DYNAMIC_RANGE):
            raise ValueError(
                f"rho must be in (0, {_FP8_NORMAL_DYNAMIC_RANGE}); got {rho}. Larger rho gives "
                "more headroom above the calibrated range but less room below it."
            )
        if not (0.0 < anchor_percentile <= 100.0):
            raise ValueError(f"anchor_percentile must be in (0, 100]; got {anchor_percentile}.")
        self._block_size = block_size
        if not (0.0 < upper_percentile <= 100.0):
            raise ValueError(f"upper_percentile must be in (0, 100]; got {upper_percentile}.")
        self._anchor_percentile = float(anchor_percentile)
        self._upper_percentile = float(upper_percentile)
        self._rho = float(rho)
        self._num_bins = int(num_bins)
        self._log2_min = float(log2_min)
        self._log2_max = float(log2_max)
        self._hist = None  # int64 histogram over log2(per-block amax), created on first collect
        self._running_max = None  # literal per-tensor max, used as the fallback amax

    def _bin_index(self, log2_vals: torch.Tensor) -> torch.Tensor:
        frac = (log2_vals - self._log2_min) / (self._log2_max - self._log2_min)
        idx = (frac * self._num_bins).floor().long()
        return idx.clamp_(0, self._num_bins - 1)

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        """Accumulate the per-block amax histogram for one activation batch."""
        x = x.detach()
        # Zero-pad a ragged tail so activations whose last dim is not a multiple of the block
        # size are handled instead of tripping the divisibility assert in reduce_block_amax.
        # Padding with zeros cannot change any block's amax.
        remainder = x.shape[-1] % self._block_size
        if remainder:
            x = torch.nn.functional.pad(x, (0, self._block_size - remainder))
        # reduce_block_amax already takes abs().
        block_amax = reduce_block_amax(x, block_sizes={-1: self._block_size})
        block_amax = block_amax.flatten().float()

        assert not torch.any(torch.isnan(block_amax)), (
            f"detected nan values in amax. nan in original tensor: {torch.any(torch.isnan(x))}"
        )
        assert not torch.any(torch.isinf(block_amax)), (
            f"detected inf values in amax. inf in original tensor: {torch.any(torch.isinf(x))}"
        )

        cur_max = block_amax.max()
        self._running_max = (
            cur_max if self._running_max is None else torch.maximum(self._running_max, cur_max)
        )

        nonzero = block_amax[block_amax > 0]
        if nonzero.numel() == 0:
            return

        idx = self._bin_index(torch.log2(nonzero))
        counts = torch.bincount(idx, minlength=self._num_bins)
        if self._hist is None:
            self._hist = torch.zeros(self._num_bins, dtype=torch.long, device=x.device)
        self._hist += counts.to(self._hist.device)

    def _percentile(self, percentile: float, floor_value: float | None = None) -> float | None:
        """Value at ``percentile`` of the histogram, ignoring bins below ``floor_value``."""
        if self._hist is None:
            return None
        counts = self._hist.float()
        if floor_value is not None and floor_value > 0:
            floor_bin = int(self._bin_index(torch.log2(torch.tensor(floor_value))).item())
            counts[:floor_bin] = 0
        total = counts.sum()
        if total <= 0:
            return None
        target = percentile / 100.0 * total
        cdf = torch.cumsum(counts, dim=0)
        bin_idx = int(torch.searchsorted(cdf, target).clamp(0, self._num_bins - 1).item())
        # Reconstruct at the bin center. Which representative is used (center vs. a bin edge
        # vs. interpolating within the bin from the cdf) shifts every calibrated scale, so it
        # is kept fixed to keep scales comparable across runs and existing checkpoints.
        log2_val = self._log2_min + (bin_idx + 0.5) / self._num_bins * (
            self._log2_max - self._log2_min
        )
        return float(2.0**log2_val)

    @torch.no_grad()
    def compute_amax(self) -> torch.Tensor | None:
        """Return the calibrated NVFP4 activation global scale."""
        if self._hist is None or self._running_max is None:
            return None

        upper = (
            float(self._running_max)
            if self._upper_percentile >= 100.0
            else self._percentile(self._upper_percentile)
        )
        anchor = (
            self._percentile(
                self._anchor_percentile, floor_value=upper / _ANCHOR_FLOOR_RATIO if upper else None
            )
            if upper
            else None
        )
        if not upper or upper <= 0 or not anchor or anchor <= 0:
            # Percentiles collapsed despite a non-empty histogram: fall back to plain max.
            return self._running_max.clone()

        # ``upper`` is the top of the range the scale commits to representing; the anchor term
        # only ever raises the scale above it. Blocks above ``upper`` are clipped -- see the
        # class docstring for why that is preferred to chasing a lone outlier.
        headroom_amax = self._rho * anchor
        if headroom_amax <= upper:
            span = upper / anchor
            warnings.warn(
                f"[nvfp4_act_headroom] per-block amax range {span:.1f} leaves no headroom at "
                f"rho={self._rho:g}; the scale falls back to the top of the calibrated range. "
                + (
                    f"The range also exceeds the FP8 scale range "
                    f"({_FP8_NORMAL_DYNAMIC_RANGE:.0f}), so no permitted rho can clear it and no "
                    "single global scale holds both ends. Reduce the range with outlier "
                    "mitigation (SmoothQuant / per-channel / higher precision)."
                    if span > _FP8_NORMAL_DYNAMIC_RANGE
                    else "Lower anchor_percentile or raise rho for more headroom."
                ),
                stacklevel=2,
            )
            amax = upper
        else:
            amax = headroom_amax

        return torch.tensor(float(amax), dtype=torch.float32, device=self._running_max.device)

    def reset(self) -> None:
        """Reset the collected histogram."""
        self._hist = None
        self._running_max = None

    def __repr__(self):
        return (
            f"NVFP4ActHeadroomCalibrator(anchor_percentile={self._anchor_percentile}, "
            f"upper_percentile={self._upper_percentile}, rho={self._rho}, "
            f"block_size={self._block_size})"
        )
