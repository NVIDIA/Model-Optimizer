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

"""Statistics manager for sparse attention modules."""


class SparseAttentionStatsManager:
    """Centralized statistics manager for sparse attention.

    This class is the single source of truth for all statistics collection
    in sparse attention modules. It handles both runtime aggregation and
    per-sample calibration statistics.

    Design principles:
    - Single responsibility: only stats management
    - No computation: receives pre-computed stats from methods
    - Optional: can be None if stats collection disabled
    - Zero overhead when disabled
    """

    def __init__(self, module_name: str, enabled: bool = True):
        """Initialize stats manager.

        Args:
            module_name: Name of the module this manager is attached to
            enabled: Whether stats collection is enabled
        """
        self.module_name = module_name
        self.enabled = enabled
        self.calibration_mode = False

        # Aggregated stats (running totals across all forward passes)
        self.aggregated_stats: dict = {
            "total_calls": 0,
            "total_blocks": 0,
            "sparse_blocks": 0,
            "phase_counts": {"prefill": 0, "decode": 0, "unknown": 0},
        }

        # Per-sample stats (only populated during calibration)
        self.per_sample_stats: list[dict] = []

    def collect(self, stats: dict):
        """Collect statistics from a single forward pass.

        Args:
            stats: Dictionary containing statistics from method computation.
                Expected keys: sparsity, phase, total_blocks, sparse_blocks,
                sample_length (optional)
        """
        if not self.enabled:
            return

        # Update aggregated stats
        self.aggregated_stats["total_calls"] += 1
        self.aggregated_stats["total_blocks"] += stats.get("total_blocks", 0)
        self.aggregated_stats["sparse_blocks"] += stats.get("sparse_blocks", 0)

        phase = stats.get("phase", "unknown")
        if phase in self.aggregated_stats["phase_counts"]:
            self.aggregated_stats["phase_counts"][phase] += 1

        # In calibration mode, store per-sample stats
        if self.calibration_mode:
            self.per_sample_stats.append(
                {
                    "module": self.module_name,
                    "sparsity": stats.get("sparsity", 0.0),
                    "sample_length": stats.get("sample_length", 0),
                    "phase": phase,
                }
            )

    def get_summary(self) -> dict:
        """Get aggregated statistics summary.

        Returns:
            Dictionary with module name, total calls, average sparsity,
            and phase distribution.
        """
        total_blocks = self.aggregated_stats["total_blocks"]
        if total_blocks > 0:
            avg_sparsity = self.aggregated_stats["sparse_blocks"] / total_blocks
        else:
            avg_sparsity = 0.0

        return {
            "module": self.module_name,
            "total_calls": self.aggregated_stats["total_calls"],
            "average_sparsity": avg_sparsity,
            "phase_distribution": self.aggregated_stats["phase_counts"].copy(),
        }

    def set_calibration_mode(self, enabled: bool, reset_history: bool = True):
        """Enable or disable calibration mode.

        In calibration mode, per-sample statistics are stored for detailed
        analysis. Otherwise, only aggregated stats are maintained.

        Args:
            enabled: Whether to enable calibration mode
            reset_history: Whether to clear per_sample_stats when enabling
        """
        self.calibration_mode = enabled
        if enabled and reset_history:
            self.per_sample_stats = []

    def reset(self):
        """Reset all statistics to initial state."""
        self.aggregated_stats = {
            "total_calls": 0,
            "total_blocks": 0,
            "sparse_blocks": 0,
            "phase_counts": {"prefill": 0, "decode": 0, "unknown": 0},
        }
        self.per_sample_stats = []

    def get_calibration_stats(self, phase: str | None = None) -> list[dict]:
        """Get per-sample calibration statistics, optionally filtered by phase.

        Args:
            phase: Optional phase to filter by ('prefill' or 'decode').
                   If None, returns all stats.

        Returns:
            List of per-sample statistics dictionaries.
            Empty list if not in calibration mode or no stats for that phase.
        """
        if phase is None:
            return self.per_sample_stats
        return [s for s in self.per_sample_stats if s.get("phase") == phase]
