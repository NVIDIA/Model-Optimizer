# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Calibrator for the NVFP4 activation global scale (``nvfp4_act_max``).

Calibrates the per-tensor NVFP4 global amax (``g_amax``) of an activation
quantizer from the distribution of its per-block (16-wide) amaxes, using the
``B_min``-anchored recipe documented in
``experimental/nvfp4_global_scale_study/``:

    g_amax = rho * B_min        (rho in (0, 28672), default 16384)

where ``B_min`` is a robust low percentile of the per-block amaxes. Unlike plain
``max`` calibration (which sets ``g_amax`` to the literal per-tensor max and so
sits on the saturation cliff), this spends the format's wide normal-FP8 window as
upward headroom so unseen activation outliers degrade gracefully rather than
clipping. See the study README §4 for the derivation.
"""

import warnings

import torch

from ..utils import reduce_block_amax
from .calibrator import _Calibrator

__all__ = ["NVFP4ActMaxCalibrator"]

# FP8-E4M3 normal dynamic range (max_normal / min_normal = 448 / 2**-6 = 7 * 2**12).
# The width of the well-conditioned NVFP4 global-scale window; see study README §2.
_FP8_NORMAL_DYNAMIC_RANGE = 28672.0


class NVFP4ActMaxCalibrator(_Calibrator):
    """Calibrates the NVFP4 activation global amax via the ``B_min``-anchored recipe.

    The calibrator accumulates a base-2 log-spaced histogram of the per-block
    amaxes seen during calibration (bounded memory), then at ``compute_amax``
    derives robust ``B_min`` / ``B_max`` percentiles and returns
    ``g_amax = clamp(rho * B_min, floor = margin * B_max)``.

    Args:
        num_bits: quantizer ``num_bits`` (``(2, 1)`` for NVFP4); kept for interface parity.
        axis: unused (the global amax is per-tensor); kept for interface parity.
        unsigned: unused; kept for interface parity.
        block_size: NVFP4 block width along the last dim (16).
        rho: window-split factor; ``g_amax = rho * B_min``. Must be in ``(0, 28672)``.
        b_min_percentile: low percentile (over represented blocks) used for ``B_min``.
        b_max_percentile: high percentile used for ``B_max`` (``100`` => literal max).
        margin: sanity-floor multiplier; ``g_amax >= margin * B_max``.
        num_bins: number of log2 histogram bins.
        log2_min / log2_max: log2 range covered by the histogram.
    """

    def __init__(
        self,
        num_bits=(2, 1),
        axis=None,
        unsigned=False,
        *,
        block_size=16,
        rho=16384.0,
        b_min_percentile=1.0,
        b_max_percentile=99.99,
        margin=1.0,
        num_bins=512,
        log2_min=-40.0,
        log2_max=40.0,
    ):
        """Initialize."""
        super().__init__(num_bits, axis, unsigned)
        if not (0.0 < rho < _FP8_NORMAL_DYNAMIC_RANGE):
            raise ValueError(
                f"rho must be in (0, {_FP8_NORMAL_DYNAMIC_RANGE}); got {rho}. Larger rho gives more "
                "upward (saturation) headroom but less downward (subnormal) cushion."
            )
        self._block_size = block_size
        self._rho = float(rho)
        self._b_min_percentile = float(b_min_percentile)
        self._b_max_percentile = float(b_max_percentile)
        self._margin = float(margin)
        self._num_bins = int(num_bins)
        self._log2_min = float(log2_min)
        self._log2_max = float(log2_max)
        self._hist = None  # int64 histogram over log2(per-block amax), lazily created on device
        self._running_max = (
            None  # literal per-tensor max (for b_max_percentile == 100 and the floor)
        )
        self._total = 0  # total per-block samples seen
        self._dead = 0  # blocks that were exactly zero
        self._stats = (
            None  # diagnostic stats recorded at compute_amax (see NVFP4_ACT_MAX_STATS_PATH)
        )

    def _bin_index(self, log2_vals: torch.Tensor) -> torch.Tensor:
        frac = (log2_vals - self._log2_min) / (self._log2_max - self._log2_min)
        idx = (frac * self._num_bins).floor().long()
        return idx.clamp_(0, self._num_bins - 1)

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        """Accumulate the per-block amax histogram for one activation batch."""
        # Per-block (16-wide) amax over the last dim; reduce_block_amax already takes abs().
        block_amax = reduce_block_amax(x.detach(), block_sizes={-1: self._block_size})
        block_amax = block_amax.flatten().float()

        cur_max = block_amax.max()
        self._running_max = (
            cur_max if self._running_max is None else torch.maximum(self._running_max, cur_max)
        )

        self._total += block_amax.numel()
        nonzero = block_amax[block_amax > 0]
        self._dead += block_amax.numel() - nonzero.numel()
        if nonzero.numel() == 0:
            return

        log2_vals = torch.log2(nonzero)
        idx = self._bin_index(log2_vals)
        counts = torch.bincount(idx, minlength=self._num_bins)
        if self._hist is None:
            self._hist = torch.zeros(self._num_bins, dtype=torch.long, device=x.device)
        self._hist += counts.to(self._hist.device)

    def _percentile(self, percentile: float, floor_value: float | None = None) -> float | None:
        """Return the value at ``percentile`` of the histogram, optionally ignoring bins below ``floor_value``."""
        if self._hist is None:
            return None
        counts = self._hist.float().clone()
        if floor_value is not None and floor_value > 0:
            floor_bin = int(self._bin_index(torch.log2(torch.tensor(floor_value))).item())
            counts[:floor_bin] = 0
        total = counts.sum()
        if total <= 0:
            return None
        target = percentile / 100.0 * total
        cdf = torch.cumsum(counts, dim=0)
        bin_idx = int(torch.searchsorted(cdf, target).clamp(0, self._num_bins - 1).item())
        # Bin center back to a value in original (linear) space.
        log2_val = self._log2_min + (bin_idx + 0.5) / self._num_bins * (
            self._log2_max - self._log2_min
        )
        return float(2.0**log2_val)

    @torch.no_grad()
    def compute_amax(self) -> torch.Tensor | None:
        """Return the calibrated NVFP4 activation global amax (``g_amax``).

        Also records a diagnostic ``self._stats`` dict (literal max, p1, p99.99, the values
        actually used, the chosen ``g_amax`` and which term set it) for offline analysis.
        """
        if self._hist is None or self._running_max is None:
            return None
        running_max = float(self._running_max)

        # Report percentiles, computed regardless of the formula knobs so the dump always
        # exposes the true distribution tails (p99.99 over all blocks; low percentiles over
        # represented blocks). p3/p5 let us study a higher B_min anchor for sparse activations.
        p99_99 = self._percentile(99.99)
        _lowfloor = p99_99 / 1e6 if p99_99 else None
        p1 = self._percentile(1.0, floor_value=_lowfloor)
        p3 = self._percentile(3.0, floor_value=_lowfloor)
        p5 = self._percentile(5.0, floor_value=_lowfloor)

        b_max = (
            running_max
            if self._b_max_percentile >= 100.0
            else self._percentile(self._b_max_percentile)
        )
        b_min = self._percentile(self._b_min_percentile, floor_value=b_max / 1e6) if b_max else None

        if b_max is None or b_max <= 0 or b_min is None or b_min <= 0:
            self._stats = {
                "n_total": int(self._total),
                "n_dead": int(self._dead),
                "literal_max": running_max,
                "p1": p1,
                "p3": p3,
                "p5": p5,
                "p99_99": p99_99,
                "g_amax": running_max,
                "term": "fallback_running_max",
            }
            return self._running_max.clone()

        if b_max / b_min > _FP8_NORMAL_DYNAMIC_RANGE:
            warnings.warn(
                f"[nvfp4_act_max] block-amax dynamic range B_max/B_min = {b_max / b_min:.1f} exceeds the "
                f"NVFP4 format window ({_FP8_NORMAL_DYNAMIC_RANGE:.0f}); no single global scale avoids both "
                "saturation and subnormal. Falling back to g_amax = B_max (no-saturation edge). Fix the "
                "range with outlier mitigation (SmoothQuant / per-channel / higher-precision), not g_amax."
            )
            g_amax = b_max
            term = "guardrail_b_max"
        else:
            # B_min-anchored: spend the format's window as upward headroom.
            cand_rho, cand_floor = self._rho * b_min, self._margin * b_max
            g_amax = max(cand_rho, cand_floor)
            term = "rho*B_min" if cand_rho >= cand_floor else "margin*B_max"

        self._stats = {
            "n_total": int(self._total),
            "n_dead": int(self._dead),
            "literal_max": running_max,
            "p1": p1,
            "p3": p3,
            "p5": p5,
            "p99_99": p99_99,
            "b_min_used": b_min,
            "b_max_used": b_max,
            "b_min_percentile": self._b_min_percentile,
            "b_max_percentile": self._b_max_percentile,
            "rho": self._rho,
            "g_amax": float(g_amax),
            "term": term,
            "tail_literal_over_p99_99": (running_max / p99_99) if p99_99 else None,
            "dyn_range_p99_99_over_p1": (p99_99 / p1) if (p99_99 and p1) else None,
        }
        return torch.tensor(float(g_amax), dtype=torch.float32, device=self._running_max.device)

    def reset(self) -> None:
        """Reset the collected histogram and statistics."""
        self._hist = None
        self._running_max = None
        self._total = 0
        self._dead = 0
        self._stats = None

    def __repr__(self):
        return (
            f"NVFP4ActMaxCalibrator(rho={self._rho}, b_min_percentile={self._b_min_percentile}, "
            f"b_max_percentile={self._b_max_percentile}, margin={self._margin}, "
            f"block_size={self._block_size})"
        )
