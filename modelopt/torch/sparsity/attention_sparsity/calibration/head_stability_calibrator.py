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

"""Per-head stability calibrator for diffusion attention temporal caching.

Profiles per-head cosine similarity across consecutive denoising timesteps
to identify stable heads that can be cached and reused.

The calibrator sets ``HeadCacheState.calibrating = True`` on each module,
runs the forward loop, then reads ``similarity_scores`` that were recorded
by the kernel wrappers during the forward pass.  No monkey-patching needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from modelopt.torch.sparsity.attention_sparsity.methods.head_cache import HeadCacheState
from modelopt.torch.sparsity.attention_sparsity.utils import get_named_sparse_attention_modules


class HeadStabilityCalibrator:
    """Profiles per-head stability across denoising timesteps.

    Runs the pipeline once with calibration mode enabled on each
    ``HeadCacheState``, then reads the per-head cosine similarity scores
    recorded by the kernel wrappers.
    """

    def calibrate(
        self,
        model: torch.nn.Module,
        forward_loop: Callable,
    ) -> dict[str, dict[str, Any]]:
        """Run calibration to identify stable attention heads.

        Heads with mean cosine similarity >= ``similarity_threshold`` (from
        each module's config) are marked as stable and cached on alternating
        steps.  This gives each layer a different number of cached heads
        based on its actual stability.

        Args:
            model: Model with SparseAttentionModules using head_cache method.
            forward_loop: Callable that runs the full pipeline (e.g. one video
                generation).  Called as ``forward_loop(model)``.

        Returns:
            Dictionary mapping module names to per-head similarity scores.
        """
        # Find all head_cache modules and collect their states
        head_cache_modules: list[tuple[str, torch.nn.Module]] = []
        for name, module in get_named_sparse_attention_modules(model):
            if (
                hasattr(module, "_sparse_method_instance")
                and module._sparse_method_instance.name == "head_cache"
            ):
                head_cache_modules.append((name, module))

        if not head_cache_modules:
            print("No head_cache modules found for calibration.")
            return {}

        # Enable calibration mode on each module's HeadCacheState
        for _, module in head_cache_modules:
            if not hasattr(module, "_head_cache_state"):
                module._head_cache_state = HeadCacheState()
            state = module._head_cache_state
            state.reset_calibration()
            state.calibrating = True

        # Run forward loop — kernel wrappers will record similarity scores
        print(
            f"Running calibration forward loop for {len(head_cache_modules)} attention modules..."
        )
        with torch.no_grad():
            forward_loop(model)

        # Disable calibration and analyze results
        results = {}
        total_stable = 0
        total_heads = 0
        for name, module in head_cache_modules:
            state = module._head_cache_state
            state.calibrating = False

            # Compute mean similarity per head
            head_scores: dict[int, float] = {}
            for head_idx, sims in state.similarity_scores.items():
                head_scores[head_idx] = sum(sims) / len(sims) if sims else 0.0
            num_heads = len(head_scores)

            if num_heads == 0:
                print(f"  {name}: no self-attention calls recorded, skipping")
                state.reset_calibration()
                continue

            # Read similarity_threshold from the module's config
            threshold = getattr(module._sparse_method_instance, "similarity_threshold", 0.97)

            # Select heads above threshold as stable
            stable_heads = [h for h, sim in head_scores.items() if sim >= threshold]
            dynamic_heads = [h for h, sim in head_scores.items() if sim < threshold]

            # Keep at least 1 dynamic head if all are above threshold
            if len(dynamic_heads) == 0 and len(stable_heads) > 1:
                # Move the least stable head to dynamic
                sorted_by_sim = sorted(stable_heads, key=lambda h: head_scores[h])
                dynamic_heads = [sorted_by_sim[0]]
                stable_heads = sorted_by_sim[1:]

            # Set state on the module
            state.stable_heads = sorted(stable_heads)
            state.dynamic_heads = sorted(dynamic_heads)
            state.num_heads = num_heads
            state.reset()  # Clear cached output from calibration run

            total_stable += len(stable_heads)
            total_heads += num_heads

            results[name] = {
                "mean_similarity": [head_scores[h] for h in range(num_heads)],
                "stable_heads": stable_heads,
                "dynamic_heads": dynamic_heads,
            }

            # Print summary
            stable_sims = [head_scores[h] for h in stable_heads] if stable_heads else [0.0]
            dynamic_sims = [head_scores[h] for h in dynamic_heads] if dynamic_heads else [0.0]
            print(
                f"  {name}: {len(stable_heads)}/{num_heads} cached "
                f"(stable avg={sum(stable_sims) / len(stable_sims):.4f}, "
                f"dynamic avg={sum(dynamic_sims) / len(dynamic_sims):.4f})"
            )

            # Clean up calibration-only state
            state.reset_calibration()

        if total_heads > 0:
            print(
                f"\nTotal: {total_stable}/{total_heads} heads cached "
                f"({total_stable / total_heads:.0%}) with threshold={threshold}"
            )

        return results


def calibrate_head_cache(
    model: torch.nn.Module,
    forward_loop: Callable,
) -> dict[str, dict[str, Any]]:
    """Calibrate head caching by profiling per-head stability.

    Convenience function wrapping ``HeadStabilityCalibrator.calibrate``.

    Args:
        model: Model with ``head_cache`` sparse attention applied.
        forward_loop: Callable that runs one full pipeline generation.

    Returns:
        Per-module calibration results with similarity scores and head assignments.
    """
    calibrator = HeadStabilityCalibrator()
    return calibrator.calibrate(model, forward_loop)
