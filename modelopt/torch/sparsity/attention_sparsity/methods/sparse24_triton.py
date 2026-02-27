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

"""2:4 structured sparse attention method for the Triton prefill kernel.

This method is used with backend="triton" and attn_implementation="modelopt_triton".
Sparsity is applied inside the Triton kernel during prefill; this class provides
the SparseAttentionMethod interface for config-driven setup and optional diagnostics.
"""

import contextlib
from typing import Any

import torch

from . import SparseAttentionMethod, register_sparse_method


def _sparse24_mask_along_last_dim(scores: torch.Tensor) -> torch.Tensor:
    """Compute 2:4 mask: for every 4 elements along the last dim, keep the 2 largest.

    Args:
        scores: Tensor of shape [..., N] with N divisible by 4.

    Returns:
        Boolean mask of same shape; True where the element is kept (top-2 of 4).
    """
    *prefix, n = scores.shape
    assert n % 4 == 0, "2:4 sparsity requires last dim divisible by 4"
    grouped = scores.reshape(*prefix, n // 4, 4)
    # topk(2) along dim=-1; indices [..., 0] and [..., 1] are the two largest
    _, top2_idx = torch.topk(grouped, k=2, dim=-1, largest=True, sorted=False)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, top2_idx, True)
    return mask.reshape(*prefix, n)


@register_sparse_method("sparse24_triton")
class Sparse24Triton(SparseAttentionMethod):
    """2:4 structured sparse attention for the Triton prefill kernel.

    When backend is "triton", sparsity is applied inside the kernel; this method
    provides the config interface and optional PyTorch-side diagnostics (e.g.
    calculate_sparsity for stats). No calibration; pattern is fixed (top-2 of every 4).
    """

    def __init__(self, method_config: dict | None = None):
        """Initialize 2:4 Triton sparse attention method.

        Args:
            method_config: Configuration dict. Uses skip_diagonal_blocks, is_causal;
                ignores threshold, br, bc (not used by 2:4).
        """
        super().__init__()
        config = method_config or {}
        self.skip_diagonal_blocks = config.get("skip_diagonal_blocks", True)
        self.is_causal = config.get("is_causal", True)
        self.backend = config.get("backend", "triton")

    def _infer_phase(self, attention_scores: torch.Tensor) -> str:
        """Infer phase from attention scores shape."""
        return "decode" if attention_scores.shape[2] == 1 else "prefill"

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Calculate 2:4 sparsity mask and statistics (PyTorch reference).

        Used for diagnostics when collect_stats is enabled. The actual sparsity
        during forward with backend="triton" is applied inside the Triton kernel.

        Args:
            attention_scores: [batch, heads, seq_q, seq_k]

        Returns:
            (sparse_mask, stats_dict)
        """
        assert attention_scores.dim() == 4, (
            f"Expected 4D attention scores, got shape {attention_scores.shape}"
        )
        batch, num_heads, seq_q, seq_k = attention_scores.shape
        phase = self._infer_phase(attention_scores)

        # Pad seq_k to multiple of 4 for 2:4 grouping
        pad = (4 - seq_k % 4) % 4
        if pad > 0:
            scores_padded = torch.nn.functional.pad(
                attention_scores, (0, pad), value=torch.finfo(attention_scores.dtype).min
            )
        else:
            scores_padded = attention_scores

        mask_padded = _sparse24_mask_along_last_dim(scores_padded)
        if pad > 0:
            sparse_mask = mask_padded[..., :seq_k].contiguous()
        else:
            sparse_mask = mask_padded

        # 2:4 keeps 2 of 4 -> 50% kept (0.5 sparsity ratio as "fraction sparse" = 0.5)
        sparsity = 0.5
        stats = {
            "sparsity": sparsity,
            "phase": phase,
            "total_blocks": (seq_k + pad) // 4 * seq_q * num_heads * batch,
            "sparse_blocks": int(0.5 * (seq_k + pad) // 4 * seq_q * num_heads * batch),
            "sample_length": seq_k,
        }
        return sparse_mask, stats

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply 2:4 sparsity mask to attention scores.

        Args:
            attention_scores: [batch, heads, seq_q, seq_k]
            sparse_mask: Optional pre-computed mask. If None, computes via calculate_sparsity.

        Returns:
            Masked scores (same shape); masked positions set to dtype min.
        """
        if sparse_mask is None:
            sparse_mask, _ = self.calculate_sparsity(attention_scores)
        mask_value = torch.finfo(attention_scores.dtype).min
        return attention_scores.masked_fill(~sparse_mask, mask_value)

    @contextlib.contextmanager
    def get_sparse_context(self, module: torch.nn.Module):
        """Set _apply_sparse24 and _skip_diagonal_blocks on module for the Triton kernel.

        Sets both module attributes (for the transformers ``ALL_ATTENTION_FUNCTIONS``
        path) and a thread-local (for the diffusers ``dispatch_attention_fn`` path).
        """
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            set_sparse24_context,
        )

        effective_skip = self.skip_diagonal_blocks and self.is_causal
        module._apply_sparse24 = True
        module._skip_diagonal_blocks = effective_skip
        set_sparse24_context(True, effective_skip)
        try:
            yield
        finally:
            module._apply_sparse24 = False
            set_sparse24_context(False, True)

    def get_threshold_info(self) -> dict[str, Any]:
        """Return fixed 2:4 pattern info."""
        return {"type": "fixed", "value": 0.5}

    @property
    def name(self) -> str:
        """Method identifier."""
        return "sparse24_triton"
