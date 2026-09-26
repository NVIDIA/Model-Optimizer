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
from scipy.optimize import curve_fit

from ..stats_manager import SparseAttentionStatsManager
from ..utils import get_sparse_attention_modules


class DynamicThresholdCalibrator:
    """Dynamic threshold calibrator using Exponential model.

    Calibration Algorithm:
        1. For each threshold λ_j in threshold_trials:
           - Run ALL samples through forward_loop
           - For each sample i with length L_i, collect sparsity S_ij
           - Compute scale_factor_ij = λ_j × L_i

        2. Fit Exponential model to ALL individual (sf_ij, S_ij) pairs:
           scale_factor = a * exp(b * sparsity)

        3. Return fitted a and b parameters

    At inference time (user specifies target_sparsity S*):
        scale_factor = a * exp(b * S*)
        threshold = scale_factor / seqlen

    Key insight: Using all individual data points (N_thresholds × N_samples)
    instead of per-threshold averages provides more accurate fitting without
    additional calibration time cost.
    """

    def __init__(
        self,
        threshold_trials: list[float] | None = None,
        fit_logspace: bool = False,
    ):
        """Initialize dynamic threshold calibrator.

        Args:
            threshold_trials: List of thresholds to try during calibration.
                Should span a range that achieves sparsities from ~10% to ~95%.
            fit_logspace: If True, fit the exponential model in log space
                (minimizes relative error). Recommended for diffusion models
                where scale_factors span many orders of magnitude.
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
        self.fit_logspace = fit_logspace

    def calibrate(self, model: nn.Module, forward_loop: Callable, phase: str) -> dict[str, Any]:
        """Calibrate a and b parameters for Exponential model.

        Algorithm:
            1. Set thresholds = threshold_trials on all modules, run ONE forward pass.
               Each module returns a sparsity list (one entry per threshold) per sample.
               Unpack to get (scale_factor_ij = λ_j × L_i, sparsity_ij) pairs.

            2. Fit Exponential model to ALL (sf_ij, S_ij) pairs:
               scale_factor = a * exp(b * sparsity)

            3. Return fitted a and b parameters

        At inference time (user specifies target_sparsity S*):
            scale_factor = a * exp(b * S*)
            threshold = scale_factor / seqlen

        Args:
            model: The model with sparse attention modules
            forward_loop: Callable that takes model and forwards calibration data
            phase: Phase to calibrate ('prefill' or 'decode')

        Returns:
            Dict with calibration results including a, b, r_squared, and num_data_points
        """
        # Extract attention modules
        attention_modules = get_sparse_attention_modules(model)

        if not attention_modules:
            raise ValueError("No sparse attention modules found for calibration")

        print(f"Starting Exponential model calibration ({phase} phase)")
        print(f"Threshold trials: {len(self.threshold_trials)}")

        # Stage 1: Collect ALL (scale_factor, sparsity) pairs in a single forward pass.
        # All threshold_trials are passed at once; each module returns a sparsity list
        # with one entry per threshold, eliminating the need for repeated forward passes.
        print(f"\nStage 1: Collecting {phase} sparsity data for all thresholds in one pass...")

        all_data_points = []  # List of {"threshold", "length", "scale_factor", "sparsity"}

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
                scale_factor = threshold * length
                all_data_points.append(
                    {
                        "threshold": threshold,
                        "length": length,
                        "scale_factor": scale_factor,
                        "sparsity": sparsity,
                    }
                )

        if len(all_data_points) < 10:
            warnings.warn(
                f"Not enough data points for {phase} calibration. "
                f"Got {len(all_data_points)}, need at least 10."
            )
            return {}

        print(f"Collected {len(all_data_points)} individual (scale_factor, sparsity) pairs")

        # Stage 2: Fit Exponential model: scale_factor = a * exp(b * sparsity)
        print("\nStage 2: Fitting Exponential model to all data points...")

        # Extract data for fitting
        scale_factors = np.array([pt["scale_factor"] for pt in all_data_points])
        sparsities = np.array([pt["sparsity"] for pt in all_data_points])

        # Filter out extreme sparsities (must be in (10%, 90%))
        # Extreme values are unreliable for fitting
        valid_mask = (sparsities >= 0.10) & (sparsities <= 0.90)
        if self.fit_logspace:
            valid_mask &= scale_factors > 0  # log requires positive values
        scale_factors = scale_factors[valid_mask]
        sparsities = sparsities[valid_mask]

        if len(scale_factors) < 3:
            warnings.warn(
                f"Not enough valid data points after filtering. Got {len(scale_factors)}."
            )
            return {}

        # Record observed sparsity range for feasibility checks at inference
        min_observed_sparsity = float(np.min(sparsities))
        max_observed_sparsity = float(np.max(sparsities))

        try:
            if self.fit_logspace:
                # Log-space fit: minimizes relative error. Recommended for
                # diffusion models where scale_factors span many orders of
                # magnitude (e.g. 0.06 to 57,000) — a linear-space fit would
                # be dominated by the largest values.
                log_scale_factors = np.log(scale_factors)

                def log_exponential(sparsity, log_a, b):
                    return log_a + b * sparsity

                popt, pcov = curve_fit(
                    log_exponential,
                    sparsities,
                    log_scale_factors,
                    p0=[0.0, 10.0],
                    maxfev=10000,
                )
                log_a, b = popt
                a = np.exp(log_a)

                # R-squared in log space (where the fit was performed)
                pred = log_exponential(sparsities, log_a, b)
                ss_res = np.sum((log_scale_factors - pred) ** 2)
                ss_tot = np.sum((log_scale_factors - np.mean(log_scale_factors)) ** 2)
            else:
                # Linear-space fit (default): minimizes absolute error.

                def exponential(sparsity, a, b):
                    return a * np.exp(b * sparsity)

                popt, pcov = curve_fit(
                    exponential,
                    sparsities,
                    scale_factors,
                    p0=[1.0, 5.0],
                    bounds=([0.0, 0.0], [np.inf, 20.0]),
                    maxfev=10000,
                )
                a, b = popt

                pred = exponential(sparsities, a, b)
                ss_res = np.sum((scale_factors - pred) ** 2)
                ss_tot = np.sum((scale_factors - np.mean(scale_factors)) ** 2)
        except Exception as e:
            warnings.warn(f"Curve fitting failed: {e}")
            return {}

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        fit_label = "log-space" if self.fit_logspace else "linear-space"
        print(f"\n{phase.capitalize()} Calibration Results (Exponential Model, {fit_label} fit):")
        print("  Model: scale_factor = a * exp(b * sparsity)")
        print(f"  Fitted a: {a:.6e}")
        print(f"  Fitted b: {b:.4f}")
        print(f"  R-squared: {r_squared:.6f}")
        print(
            f"  Observed sparsity range: [{min_observed_sparsity:.1%}, {max_observed_sparsity:.1%}]"
        )
        print(f"  Data points used: {int(np.sum(valid_mask))} / {len(all_data_points)}")

        # Show scale_factor for various target sparsities
        print("\nScale factors for different target sparsities:")
        print(f"  {'Target':<10} {'Scale Factor':<15} {'Note':<20}")
        print(f"  {'-' * 10} {'-' * 15} {'-' * 20}")
        for target in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            sf = a * np.exp(b * target)
            note = ""
            if target < min_observed_sparsity or target > max_observed_sparsity:
                note = "(extrapolation)"
            print(f"  {target:<10.0%} {sf:<15.4f} {note:<20}")

        # Print calibration data summary by threshold
        print("\nCalibration data summary (per threshold):")
        print(f"  {'Threshold':<12} {'Avg SF':<12} {'Avg Sparsity':<12} {'Samples':<8}")
        print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}")

        # Group by threshold for summary
        by_threshold = defaultdict(list)
        for point in all_data_points:
            by_threshold[point["threshold"]].append(point)

        for threshold in sorted(by_threshold.keys()):
            points = by_threshold[threshold]
            avg_sf = np.mean([p["scale_factor"] for p in points])
            avg_s = np.mean([p["sparsity"] for p in points])
            print(f"  {threshold:<12.4f} {avg_sf:<12.2f} {avg_s:<12.2%} {len(points):<8}")

        return {
            "phase": phase,
            "a": float(a),
            "b": float(b),
            "r_squared": float(r_squared),
            "num_data_points": int(np.sum(valid_mask)),
            "total_samples": len(all_data_points),
            "calibration_type": "exponential",
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
