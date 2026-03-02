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

"""Per-head temporal caching method for diffusion attention.

Caches stable attention head outputs across denoising steps and reuses them
on alternating steps:
- Even steps: compute all heads, store output in cache
- Odd steps: recompute only dynamic heads, merge with cached stable heads

Stable heads are identified offline via cosine similarity calibration.
"""

import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from . import SparseAttentionMethod, register_sparse_method


@dataclass
class HeadCacheState:
    """Per-module state for head temporal caching.

    Attributes:
        stable_heads: Head indices to reuse from cache (set by calibrator).
        dynamic_heads: Head indices to recompute every step.
        cached_output: Cached output tensor ``[B, T, H, D]`` from previous compute step.
        step: Call counter. Even = compute all, odd = reuse stable.
        num_heads: Total number of attention heads.
        apply_sparse24: Whether to apply 2:4 sparsity on dynamic head subset.
        calibrating: When True, wrappers compute all heads and record per-head similarity.
        similarity_scores: Per-head cosine similarities across consecutive steps (filled during calibration).
        _prev_shape: Tracks output shape for auto-reset on new generation.
        _prev_calib_output: Previous output during calibration for similarity comparison.
    """

    stable_heads: list[int] = field(default_factory=list)
    dynamic_heads: list[int] = field(default_factory=list)
    cached_output: torch.Tensor | None = None
    step: int = 0
    num_heads: int = 0
    apply_sparse24: bool = False
    calibrating: bool = False
    similarity_scores: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    _prev_shape: tuple[int, ...] | None = None
    _prev_calib_output: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear cache and step counter (call before each new generation)."""
        self.cached_output = None
        self.step = 0
        self._prev_shape = None

    def reset_calibration(self) -> None:
        """Clear calibration state."""
        self.calibrating = False
        self.similarity_scores = defaultdict(list)
        self._prev_calib_output = None

    def record_output(self, output_4d: torch.Tensor) -> None:
        """Record per-head cosine similarity with previous output.

        Called by kernel wrappers during calibration. ``output_4d`` must be
        ``[B, T, H, D]``.
        """
        if self._prev_calib_output is not None and self._prev_calib_output.shape == output_4d.shape:
            n_heads = output_4d.shape[2]
            with torch.no_grad():
                for h in range(n_heads):
                    curr = output_4d[:, :, h, :].reshape(-1).float()
                    prev = self._prev_calib_output[:, :, h, :].reshape(-1).float()
                    sim = F.cosine_similarity(curr.unsqueeze(0), prev.unsqueeze(0)).item()
                    self.similarity_scores[h].append(sim)
        self._prev_calib_output = output_4d.detach().clone()


@register_sparse_method("head_cache")
class HeadCacheMethod(SparseAttentionMethod):
    """Per-head temporal caching for diffusion attention.

    Alternates between full computation (even steps) and partial reuse (odd steps).
    Stable heads are identified by offline calibration of cosine similarity
    across consecutive denoising timesteps.
    """

    def __init__(self, method_config: dict | None = None):
        """Initialize head cache method with configuration."""
        super().__init__()
        config = method_config or {}
        self.similarity_threshold: float = config.get("similarity_threshold", 0.97)
        self.apply_sparse24: bool = config.get("apply_sparse24", False)
        self.skip_diagonal_blocks: bool = config.get("skip_diagonal_blocks", True)
        self.backend: str = config.get("backend", "triton")

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Return a dummy all-True mask; head caching operates at output level."""
        mask = torch.ones_like(attention_scores, dtype=torch.bool)
        stats = {
            "sparsity": 0.0,  # actual ratio determined after calibration
            "phase": "head_cache",
            "type": "head_cache",
        }
        return mask, stats

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """No-op: head caching applies at the output level, not score level."""
        return attention_scores

    @contextlib.contextmanager
    def get_sparse_context(self, module: torch.nn.Module):
        """Set thread-local head cache context for kernel wrappers.

        Lazily creates ``module._head_cache_state`` on first call.
        """
        from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_head_cache_attention import (
            set_head_cache_context,
        )

        # Lazily create per-module state
        if not hasattr(module, "_head_cache_state"):
            module._head_cache_state = HeadCacheState(
                num_heads=0,
                apply_sparse24=self.apply_sparse24,
            )

        state = module._head_cache_state
        set_head_cache_context(True, state)
        try:
            yield
        finally:
            set_head_cache_context(False, None)

    def get_threshold_info(self) -> dict[str, Any]:
        """Return head cache configuration info."""
        return {
            "type": "head_cache",
            "value": self.similarity_threshold,
            "apply_sparse24": self.apply_sparse24,
        }

    @property
    def name(self) -> str:
        """Method identifier."""
        return "head_cache"
