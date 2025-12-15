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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..sparse_attention import SparseAttentionModule
from ..stats_manager import SparseAttentionStatsManager


class DynamicThresholdCalibrator:
    """Dynamic threshold calibrator using length-based linear relationship.

    Implements calibration algorithm:
    1. Find hyperparameter 'a' where threshold λ = a / context_length
    2. Use dataset with different lengths and test multiple thresholds
    3. For each sample, find optimal threshold closest to target sparsity
    4. Use linear regression to fit: threshold = a * (1/length)
    """

    @dataclass
    class SampleSparsity:
        """Sparsity results for a single calibration sample."""

        length: int
        threshold_sparsities: dict[float, float]

    def __init__(
        self,
        target_sparse_ratio: dict[str, float] | None = None,
        threshold_trials: list[float] | None = None,
    ):
        """Initialize dynamic threshold calibrator.

        Args:
            target_sparse_ratio: Target sparsity ratio dict with 'prefill' and 'decode' keys.
                Each value should be in range (0.0 to 1.0). Set to 0.0 to skip that phase.
            threshold_trials: List of thresholds to try during calibration
        """
        self.target_sparse_ratio = target_sparse_ratio
        self._target_sparse_ratio_dict = self.target_sparse_ratio

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
            5e-2,
            1e-1,
            5e-1,
        ]

        # Statistics tracking
        self.sparsity_results = []

    def calibrate(self, model: nn.Module, forward_loop: Callable, phase: str) -> dict[str, Any]:
        """Find optimal 'a' parameter for length-based threshold.

        Algorithm:
            1. Test all threshold trials by running forward_loop multiple times
            2. For each sample, find optimal threshold closest to target sparsity
            3. Use regression to find 'a' in: threshold = a / length

        Args:
            model: The model with sparse attention modules
            forward_loop: Callable that takes model and forwards calibration data
            phase: Phase to calibrate ('prefill' or 'decode')

        Returns:
            Dict with calibration results including scale_factor, or empty dict if failed
        """
        assert self._target_sparse_ratio_dict is not None, "target_sparse_ratio must be provided"
        target_sparsity = self._target_sparse_ratio_dict[phase]

        # Extract attention modules
        attention_modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]

        if not attention_modules:
            raise ValueError("No sparse attention modules found for calibration")

        print(f"Starting dynamic threshold calibration ({phase} phase)")
        print(f"Target sparsity: {target_sparsity}")
        print(f"Threshold trials: {len(self.threshold_trials)}")

        # Stage 1: Collect sparsity for all sample-threshold pairs
        print(f"\nStage 1: Collecting {phase} sparsity data...")

        # Run first threshold to discover samples and initialize results
        self._set_threshold(attention_modules, self.threshold_trials[0])
        self._enable_calibration_mode(attention_modules)
        with torch.no_grad():
            forward_loop(model)
        per_sample_stats = self._extract_calibration_stats(attention_modules, phase=phase)
        self._disable_calibration_mode(attention_modules)

        if not per_sample_stats:
            warnings.warn(f"No {phase} phase statistics collected. Check forward loop.")
            return {}

        # Initialize sparsity_results with sample info
        self.sparsity_results = [
            self.SampleSparsity(
                length=stat["sample_length"],
                threshold_sparsities={self.threshold_trials[0]: stat["sparsity"]},
            )
            for stat in per_sample_stats
        ]

        # Collect remaining thresholds
        for threshold in tqdm(self.threshold_trials[1:], desc=f"Testing thresholds ({phase})"):
            self._set_threshold(attention_modules, threshold)
            self._enable_calibration_mode(attention_modules)
            with torch.no_grad():
                forward_loop(model)
            per_sample_stats = self._extract_calibration_stats(attention_modules, phase=phase)
            self._disable_calibration_mode(attention_modules)

            for sample_idx, sample_stat in enumerate(per_sample_stats):
                if sample_idx < len(self.sparsity_results):
                    self.sparsity_results[sample_idx].threshold_sparsities[threshold] = sample_stat[
                        "sparsity"
                    ]

        if not self.sparsity_results:
            warnings.warn(f"No valid {phase} sparsity measurements collected during calibration")
            return {}

        print(f"Collected statistics for {len(self.sparsity_results)} samples")

        # Stage 2: Find optimal threshold for each sample and compute 'a'
        print(f"\nStage 2: Finding 'a' parameter for target sparsity {target_sparsity:.2f}")

        # Find optimal threshold for each sample
        optimal_pairs = []
        for sample_result in self.sparsity_results:
            # Find threshold closest to target sparsity
            best_threshold, achieved_sparsity = min(
                sample_result.threshold_sparsities.items(),
                key=lambda item: abs(item[1] - target_sparsity),
            )

            optimal_pairs.append(
                {
                    "length": sample_result.length,
                    "optimal_threshold": best_threshold,
                    "achieved_sparsity": achieved_sparsity,
                    "target_sparsity": target_sparsity,
                }
            )

        if not optimal_pairs:
            warnings.warn(
                f"No optimal threshold pairs found for {phase} target sparsity {target_sparsity}. "
                f"Collected {len(self.sparsity_results)} samples but none achieved target sparsity."
            )
            return {}

        # Linear regression: threshold = a * (1/length)
        lengths = np.array([p["length"] for p in optimal_pairs])
        thresholds = np.array([p["optimal_threshold"] for p in optimal_pairs])

        # X = 1/length, Y = threshold
        x = 1.0 / lengths
        y = thresholds

        # Least squares: scale_factor = sum(x*y) / sum(x^2)
        scale_factor = np.sum(x * y) / np.sum(x**2)

        # Calculate statistics
        scale_factors_per_sample = y * lengths
        scale_factor_std = np.std(scale_factors_per_sample)

        # Calculate R-squared for quality metric
        y_pred = scale_factor * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate average achieved sparsity
        avg_achieved_sparsity = np.mean([p["achieved_sparsity"] for p in optimal_pairs])

        print(f"\n{phase.capitalize()} Calibration Results:")
        print(f"  Threshold scale factor: {scale_factor:.6f} (std: {scale_factor_std:.6f})")
        print(f"  R-squared: {r_squared:.4f}")
        print(
            f"  Average achieved sparsity: {avg_achieved_sparsity:.2%} (target: {target_sparsity:.2%})"
        )
        print(f"\nExample thresholds with λ = {scale_factor:.6f} / length:")
        for length in [1024, 2048, 4096, 8192, 16384]:
            print(f"  Length {length:5d}: threshold = {scale_factor / length:.2e}")

        return {
            "phase": phase,
            "scale_factor": scale_factor,
            "scale_factor_std": scale_factor_std,
            "r_squared": r_squared,
            "num_samples": len(optimal_pairs),
            "target_sparsity": target_sparsity,
            "avg_achieved_sparsity": avg_achieved_sparsity,
            "optimal_pairs": optimal_pairs,
            "calibration_type": "length_based_dynamic",
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
            sparsities = []
            sample_length = 0

            for module_stats in all_per_sample_stats:
                if sample_idx < len(module_stats):
                    sample_stat = module_stats[sample_idx]
                    sparsities.append(sample_stat.get("sparsity", 0.0))
                    if not sample_length and "sample_length" in sample_stat:
                        sample_length = sample_stat["sample_length"]

            avg_sparsity = float(np.mean(sparsities)) if sparsities else 0.0

            aggregated_stats.append(
                {
                    "sparsity": avg_sparsity,
                    "sample_length": sample_length,
                }
            )

        return aggregated_stats

    def _set_threshold(self, modules: list[nn.Module], threshold: float):
        """Set threshold on sparse attention modules."""
        for module in modules:
            module._sparse_method_instance.threshold = threshold
