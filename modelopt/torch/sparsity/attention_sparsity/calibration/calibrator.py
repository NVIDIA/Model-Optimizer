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

"""Calibration framework for sparse attention methods."""

import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..stats_manager import SparseAttentionStatsManager
from ..utils import get_sparse_attention_modules


class DynamicThresholdCalibrator:
    """Dynamic threshold calibrator.

    The calibration fits the model::

        t = 1 - exp(-a * (S / (1 - S)) ^ b / L ^ c)

    to ``(t_j, S_ij, L_i)`` tuples collected from a forward pass. Taking logs
    yields a linear model in ``(log a, b, -c)``::

        log(-log(1-t_j)) = log(a) + b * logit(S_ij) - c * log(L_i)

    which is solved in closed form with ``np.linalg.lstsq``. At inference
    time, given target sparsity ``S``, the threshold is
    ``t = 1 - exp(-a * (S / (1-S))^b / L^c)``.

    Properties:
        - Bounded in ``(0, 1)`` by construction (no clamping required).
        - Correct asymptotes: ``t->0`` as ``S->0`` or ``L->inf``; ``t->1`` as
          ``S->1`` or ``L->0``.
    """

    def __init__(
        self,
        threshold_trials: list[float] | None = None,
    ):
        """Initialize dynamic threshold calibrator.

        Args:
            threshold_trials: List of thresholds to try during calibration.
                Should span a range that achieves sparsities from ~10% to ~95%.
        """
        # Default threshold trials if not provided
        self.threshold_trials = threshold_trials or [
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            2e-2,
            5e-2,
            1e-1,
            2e-1,
            3e-1,
            5e-1,
            7e-1,
            8e-1,
            9e-1,
            9.5e-1,
            9.9e-1,
        ]

    def calibrate(self, model: nn.Module, forward_loop: Callable, phase: str) -> dict[str, Any]:
        """Calibrate (a, b, c) for the dynamic threshold model.

        Algorithm: set thresholds = ``threshold_trials`` on all modules and
        run ONE forward pass. Each module returns a sparsity list (one entry
        per threshold) per sample. For each ``(t_j, L_i, S_ij)`` triple, form::

            y_ij = log(-log(1 - t_j))
            x_S  = logit(S_ij) = log(S_ij / (1 - S_ij))
            x_L  = log(L_i)

        The model ``log(-log(1-t)) = log(a) + b*logit(S) - c*log(L)`` is
        linear in ``(log a, b, -c)`` and solved with ``np.linalg.lstsq``.

        At inference time, given target sparsity ``S``, the threshold is
        ``t = 1 - exp(-a * (S / (1 - S))^b / L^c)``.

        Args:
            model: The model with sparse attention modules.
            forward_loop: Callable that takes model and forwards calibration data.
            phase: Phase to calibrate (``'prefill'`` or ``'decode'``).

        Returns:
            Dict with calibration results including ``a``, ``b``, ``c``,
            ``r_squared``, and ``num_data_points``.
        """
        # Extract attention modules
        attention_modules = get_sparse_attention_modules(model)

        if not attention_modules:
            raise ValueError("No sparse attention modules found for calibration")

        print(f"Starting dynamic threshold calibration ({phase} phase)")
        print(f"Threshold trials: {len(self.threshold_trials)}")

        # Stage 1: collect (t, L, S) triples in a single forward pass. All
        # threshold_trials are passed at once; each module returns a sparsity
        # list with one entry per threshold.
        print(f"\nStage 1: Collecting {phase} sparsity data for all thresholds in one pass...")

        all_data_points = []  # List of {"threshold", "length", "sparsity"}

        self._set_thresholds(attention_modules, self.threshold_trials)
        self._enable_calibration_mode(attention_modules)
        with torch.no_grad():
            forward_loop(model)
        per_sample_stats = self._extract_calibration_stats(attention_modules, phase=phase)
        self._disable_calibration_mode(attention_modules)

        for sample_stat in per_sample_stats:
            length = sample_stat["sample_length"]
            sparsity_list = sample_stat["sparsity"]
            for threshold, sparsity in zip(self.threshold_trials, sparsity_list):
                all_data_points.append(
                    {
                        "threshold": threshold,
                        "length": length,
                        "sparsity": sparsity,
                    }
                )

        if len(all_data_points) < 10:
            warnings.warn(
                f"Not enough data points for {phase} calibration. "
                f"Got {len(all_data_points)}, need at least 10."
            )
            return {}

        print(f"Collected {len(all_data_points)} individual (t, L, S) triples")

        # Stage 2: closed-form linear fit on the transformed coordinates.
        print("\nStage 2: Fitting threshold model (closed-form linear lstsq)...")

        thresholds_arr = np.array([pt["threshold"] for pt in all_data_points], dtype=np.float64)
        lengths_arr = np.array([pt["length"] for pt in all_data_points], dtype=np.float64)
        sparsities_arr = np.array([pt["sparsity"] for pt in all_data_points], dtype=np.float64)

        # Filter:
        #  - extreme sparsities (must be in (10%, 90%); extremes are unreliable)
        #  - t must be in (0, 1) for log(-log(1-t)) to be defined
        #  - L must be > 0
        valid_mask = (
            (sparsities_arr >= 0.10)
            & (sparsities_arr <= 0.90)
            & (thresholds_arr > 0)
            & (thresholds_arr < 1)
            & (lengths_arr > 0)
        )

        if int(valid_mask.sum()) < 3:
            warnings.warn(
                f"Not enough valid data points after filtering. Got {int(valid_mask.sum())}."
            )
            return {}

        # Uppercase L, S, X below match the regression notation used in
        # the model docstring above.
        t_obs = thresholds_arr[valid_mask]
        L = lengths_arr[valid_mask]  # noqa: N806
        S = sparsities_arr[valid_mask]  # noqa: N806

        y = np.log(-np.log(1.0 - t_obs))
        logit_S = np.log(S / (1.0 - S))  # noqa: N806
        log_L = np.log(L)  # noqa: N806

        # Design matrix X with columns [1, logit(S), -log(L)] so coefficients
        # are [log(a), b, c] in that order.
        X = np.column_stack([np.ones_like(y), logit_S, -log_L])  # noqa: N806

        try:
            coefs, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        except Exception as e:
            warnings.warn(f"Linear fit failed: {e}")
            return {}

        log_a, b, c = coefs.tolist()
        a = float(np.exp(log_a))

        pred = X @ coefs
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        min_observed_sparsity = float(S.min())
        max_observed_sparsity = float(S.max())

        print(f"\n{phase.capitalize()} Calibration Results:")
        print("  Model: t = 1 - exp(-a * (S/(1-S))^b / L^c)")
        print(f"  Fitted a:     {a:.6e}")
        print(f"  Fitted b:     {b:.4f}")
        print(f"  Fitted c:     {c:.4f}")
        print(f"  R-squared:    {r_squared:.6f}")
        print(
            f"  Observed sparsity range: [{min_observed_sparsity:.1%}, {max_observed_sparsity:.1%}]"
        )
        print(f"  Data points used: {int(valid_mask.sum())} / {len(all_data_points)}")

        # Show predicted threshold for a few (S, L) combinations.
        print("\nExample thresholds t(S, L) = 1 - exp(-a*(S/(1-S))^b / L^c):")
        print(f"  {'Target S':<12} {'L=4096':<14} {'L=16384':<14} {'L=65536':<14}")
        print(f"  {'-' * 12} {'-' * 14} {'-' * 14} {'-' * 14}")
        for target in [0.3, 0.5, 0.7, 0.9]:
            scale = a * (target / (1.0 - target)) ** b
            tvals = [1.0 - np.exp(-scale / (Lx**c)) for Lx in (4096, 16384, 65536)]
            note = ""
            if target < min_observed_sparsity or target > max_observed_sparsity:
                note = " (extrapolation)"
            print(f"  {target:<12.0%} {tvals[0]:<14.4e} {tvals[1]:<14.4e} {tvals[2]:<14.4e}{note}")

        # Per-threshold summary (handy for debugging).
        print("\nCalibration data summary (per threshold):")
        print(f"  {'Threshold':<12} {'Avg Sparsity':<14} {'Avg L':<12} {'Samples':<8}")
        print(f"  {'-' * 12} {'-' * 14} {'-' * 12} {'-' * 8}")
        by_threshold = defaultdict(list)
        for point in all_data_points:
            by_threshold[point["threshold"]].append(point)
        for threshold in sorted(by_threshold.keys()):
            points = by_threshold[threshold]
            avg_s = np.mean([p["sparsity"] for p in points])
            avg_l = np.mean([p["length"] for p in points])
            print(f"  {threshold:<12.4f} {avg_s:<14.2%} {avg_l:<12.1f} {len(points):<8}")

        return {
            "phase": phase,
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "r_squared": float(r_squared),
            "num_data_points": int(valid_mask.sum()),
            "total_samples": len(all_data_points),
            "calibration_type": "dynamic_threshold",
            "min_observed_sparsity": min_observed_sparsity,
            "max_observed_sparsity": max_observed_sparsity,
        }

    def _enable_calibration_mode(self, modules: list[nn.Module]):
        """Enable calibration mode on sparse attention modules."""
        for idx, module in enumerate(modules):
            # Create stats manager if needed
            if not module._stats_manager:
                module._stats_manager = SparseAttentionStatsManager(
                    module_name=f"sparse_attn_{idx}", enabled=True
                )
            else:
                # Re-enable if disabled
                module._stats_manager.enabled = True

            # Enable calibration mode with fresh stats
            module._stats_manager.set_calibration_mode(enabled=True, reset_history=True)
            module._sparse_method_instance.set_calibration_mode(True)

    def _disable_calibration_mode(self, modules: list[nn.Module]):
        """Disable calibration mode (but keep stats enabled if collect_stats=True)."""
        for module in modules:
            if module._stats_manager:
                module._stats_manager.set_calibration_mode(enabled=False)

            module._sparse_method_instance.set_calibration_mode(False)

    def _extract_calibration_stats(
        self, modules: list[nn.Module], phase: str | None = None
    ) -> list[dict]:
        """Extract per-sample calibration statistics from modules.

        Args:
            modules: List of attention modules
            phase: Optional phase to filter by ('prefill' or 'decode').
                   If None, returns all stats.

        Returns:
            List of per-sample statistics across all modules
        """
        # Collect from all stats managers
        all_per_sample_stats = []

        for module in modules:
            # Skip modules without stats manager
            if not hasattr(module, "_stats_manager") or module._stats_manager is None:
                continue

            manager_stats = module._stats_manager.get_calibration_stats(phase)
            if manager_stats:
                all_per_sample_stats.append(manager_stats)

        if not all_per_sample_stats:
            return []

        # Aggregate across modules by sample index
        num_samples = len(all_per_sample_stats[0])
        aggregated_stats = []

        for sample_idx in range(num_samples):
            sparsity_lists = []
            sample_length = 0

            for module_stats in all_per_sample_stats:
                if sample_idx < len(module_stats):
                    sample_stat = module_stats[sample_idx]
                    sparsity = sample_stat.get("sparsity", [])
                    sparsity_lists.append(sparsity if isinstance(sparsity, list) else [sparsity])
                    if not sample_length and "sample_length" in sample_stat:
                        sample_length = sample_stat["sample_length"]

            if not sparsity_lists:
                continue

            lengths = [len(s) for s in sparsity_lists]
            assert len(set(lengths)) == 1, (
                f"All modules must have the same number of thresholds, got {lengths}"
            )
            n = lengths[0]
            avg_sparsity = [float(np.mean([sl[i] for sl in sparsity_lists])) for i in range(n)]

            aggregated_stats.append(
                {
                    "sparsity": avg_sparsity,
                    "sample_length": sample_length,
                }
            )

        return aggregated_stats

    def _set_thresholds(self, modules: list[nn.Module], thresholds: list[float]):
        """Set thresholds list on sparse attention modules.

        Supports both flash_skip_softmax (sets ``thresholds`` attribute) and
        triton_skip_softmax (sets ``_threshold_trials`` attribute).
        """
        for module in modules:
            method = module._sparse_method_instance
            if hasattr(method, "_threshold_trials"):
                # triton_skip_softmax: calibration uses Triton calibration kernel
                method._threshold_trials = thresholds
            else:
                # flash_skip_softmax: calibration uses F.softmax patching
                method.thresholds = thresholds
