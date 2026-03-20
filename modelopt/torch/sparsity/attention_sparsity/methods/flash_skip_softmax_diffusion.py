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

"""Flash Attention-aware skip-softmax method for diffusion models.

Uses gap/log(seq_k) normalization and percentile-based calibration, designed
for diffusion models where the narrow sparsity range (0-29%) causes the
exponential calibration model to fail.

Key differences from flash_skip_softmax:
- Skip decision: gap >= threshold * log(seq_k) where gap = cummax - block_max
- Calibration: threshold = percentile(all_gaps / log(seq_k), (1 - target) * 100)
- No static thresholds — requires percentile calibration
"""

import math
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from . import SparseAttentionMethod, register_sparse_method

if TYPE_CHECKING:
    from ..sparse_attention import SparseAttentionModule


@register_sparse_method("flash_skip_softmax_diffusion")
class FlashSkipSoftmaxDiffusion(SparseAttentionMethod):
    """Flash Attention-aware skip-softmax for diffusion models.

    Uses gap/log(seq_k) normalization for sequence-length-invariant thresholds
    and percentile-based calibration for accurate sparsity targeting.
    """

    def __init__(self, method_config: dict | None = None):
        """Initialize diffusion skip-softmax method.

        Args:
            method_config: Configuration dict with br, bc, is_causal, backend.
                          Threshold comes from calibration, not config.
        """
        super().__init__()
        config = method_config or {}

        self.br = config.get("br", 128)
        self.bc = config.get("bc", 128)
        self.backend = config.get("backend", "pytorch")
        self.is_causal = config.get("is_causal", False)

        self._calibration_mode = False

    def set_calibration_mode(self, enabled: bool):
        """Set calibration mode."""
        self._calibration_mode = enabled

    def _reshape_to_blocks(
        self, tensor: torch.Tensor, br: int, bc: int
    ) -> tuple[torch.Tensor, ...]:
        """Reshape tensor into blocks for Flash Attention processing.

        Args:
            tensor: Input tensor of shape [batch, heads, seq_q, seq_k]
            br: Block row size
            bc: Block column size

        Returns:
            Tuple of (blocked_tensor, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k)
        """
        batch_size, num_heads, seq_q, seq_k = tensor.shape

        padded_seq_q = math.ceil(seq_q / br) * br
        padded_seq_k = math.ceil(seq_k / bc) * bc

        if padded_seq_q != seq_q or padded_seq_k != seq_k:
            pad_q = padded_seq_q - seq_q
            pad_k = padded_seq_k - seq_k
            pad_value = torch.finfo(tensor.dtype).min
            tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_q), value=pad_value)

        num_block_rows = padded_seq_q // br
        num_block_cols = padded_seq_k // bc

        blocked = tensor.view(batch_size, num_heads, num_block_rows, br, num_block_cols, bc)

        return blocked, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k

    def calc_correction_factor_and_p(
        self, attn_weights: torch.Tensor, phase: str
    ) -> tuple[torch.Tensor | None, dict]:
        """Calculate sparse mask using gap/log(seq_k) normalization.

        Algorithm:
        1. Reshape attention scores into blocks
        2. Compute block_max and cummax (left-to-right cumulative maximum)
        3. Compute gap = cummax - block_max (positive for non-peak blocks)
        4. Calibration: collect min-gap-per-block-column / log(seq_k) as numpy
        5. Inference: compare gap against threshold * log(seq_k)

        Args:
            attn_weights: Pre-softmax attention scores [batch, heads, seq_q, seq_k]
            phase: "prefill" (always for diffusion models)

        Returns:
            element_mask: Boolean mask or None (calibration mode)
            stats: Dict with sparsity stats and optionally normalized_gaps
        """
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape
        log_seq_k = math.log(seq_k)

        blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
            self._reshape_to_blocks(attn_weights, self.br, self.bc)
        )

        # block_max: per-row max over bc (same as Flash Attention / LLM version)
        # [batch, heads, block_rows, br, block_cols]
        block_max = blocked_attn.max(dim=-1)[0]
        del blocked_attn

        # cummax: per-row running maximum across tiles (left to right)
        block_max_cummax = block_max.cummax(dim=-1)[0]

        # gap = cummax - block_max (>= 0 for non-peak tiles)
        # [batch, heads, block_rows, br, block_cols]
        gap = block_max_cummax - block_max

        total_valid_blocks = batch_size * num_heads * num_block_rows * num_block_cols
        total_blocks = num_block_rows * num_block_cols

        if self._calibration_mode:
            # Block-level skip: a tile is skipped when ALL br rows have gap >= threshold,
            # i.e., when min(gap across rows) >= threshold.
            # Collect min-gap-per-block normalized by log(seq_k).
            # gap shape: [batch, heads, block_rows, br, block_cols]
            min_gap = gap.min(dim=-2)[0]  # [batch, heads, block_rows, block_cols]
            normalized_gap = min_gap / log_seq_k

            # Exclude padded tiles
            block_max_tile = block_max.min(dim=-2)[0]  # padded rows have dtype.min
            valid_mask = block_max_tile > torch.finfo(attn_weights.dtype).min
            valid_gaps = normalized_gap[valid_mask].detach().float().cpu().numpy()
            del gap, min_gap, normalized_gap, valid_mask, block_max, block_max_cummax
            del block_max_tile

            stats = {
                "sparsity": [0.0],
                "phase": phase,
                "total_blocks": total_blocks,
                "sparse_blocks": [0],
                "sample_length": seq_k,
                "normalized_gaps": valid_gaps,
            }
            return None, stats

        del block_max, block_max_cummax

        # Inference mode: apply threshold
        calibration_params = self.calibration_params
        if (
            calibration_params is not None
            and phase in calibration_params
            and "threshold" in calibration_params[phase]
        ):
            threshold = calibration_params[phase]["threshold"]
        else:
            # Not calibrated: no sparsity (keep everything)
            del gap
            stats = {
                "sparsity": [0.0],
                "phase": phase,
                "total_blocks": total_blocks,
                "sparse_blocks": [0],
                "sample_length": seq_k,
            }
            element_mask = torch.ones(
                batch_size,
                num_heads,
                seq_q,
                seq_k,
                dtype=torch.bool,
                device=attn_weights.device,
            )
            return element_mask, stats

        scaled_threshold = threshold * log_seq_k

        # Skip tile if ALL rows agree: gap >= threshold for every row in the block
        # Equivalently: keep tile if ANY row has gap < threshold
        # gap: [batch, heads, block_rows, br, block_cols]
        block_mask = (gap < scaled_threshold).any(dim=-2)  # [batch, heads, block_rows, block_cols]
        dense_blocks = block_mask.sum().item()
        del gap

        # Expand block mask to element level
        element_mask = (
            block_mask.unsqueeze(-2)
            .unsqueeze(-1)
            .expand(batch_size, num_heads, num_block_rows, self.br, num_block_cols, self.bc)
        )
        del block_mask
        element_mask = element_mask.reshape(batch_size, num_heads, padded_seq_q, padded_seq_k)
        element_mask = element_mask[:, :, :seq_q, :seq_k]

        sparsity = 1.0 - dense_blocks / total_valid_blocks

        stats = {
            "sparsity": [sparsity],
            "phase": phase,
            "total_blocks": total_blocks,
            "sparse_blocks": [int(sparsity * total_blocks)],
            "sample_length": seq_k,
        }

        return element_mask, stats

    @contextmanager
    def get_sparse_context(self, module: "SparseAttentionModule"):
        """Return context that activates skip-softmax sparse attention.

        For diffusers models this additionally:
        1. Sets the thread-local flag so the eager backend is activated.
        2. Switches the diffusers attention backend to ``modelopt_skip_softmax``.

        The ``F.softmax`` patch is always applied (works for both HF LLMs and
        diffusers models).
        """
        import torch.nn.functional as F_module

        from modelopt.torch.quantization.utils import replace_function

        original_softmax = F_module.softmax

        def sparse_softmax(input, dim=-1, *args, **kwargs):
            sparse_mask, stats = self.calculate_sparsity(input)
            module._last_stats = stats
            if not self._calibration_mode:
                input = self.apply_sparsity(input, sparse_mask)
            return original_softmax(input, dim, *args, **kwargs)

        with ExitStack() as stack:
            from ..kernels import set_skip_softmax_context

            set_skip_softmax_context(True)
            stack.callback(set_skip_softmax_context, False)

            try:
                from ..kernels.diffusers_eager_attention import get_skip_softmax_attention_backend

                stack.enter_context(get_skip_softmax_attention_backend())
            except (ImportError, RuntimeError):
                pass

            stack.enter_context(replace_function(torch.nn.functional, "softmax", sparse_softmax))
            yield

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict]:
        """Calculate sparsity mask and statistics.

        Args:
            attention_scores: Attention scores [batch, heads, seq_q, seq_k]

        Returns:
            Tuple of (sparse_mask, stats)
        """
        assert len(attention_scores.shape) == 4, (
            f"Expected 4D attention scores, got shape {attention_scores.shape}"
        )

        phase = "prefill"  # Diffusion models always use prefill

        sparse_mask, stats = self.calc_correction_factor_and_p(attention_scores, phase)
        self._last_stats = stats

        return sparse_mask, stats

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparsity mask to attention scores.

        Args:
            attention_scores: Attention scores [batch, heads, seq_q, seq_k]
            sparse_mask: Optional pre-computed boolean mask.

        Returns:
            Masked attention scores with sparse elements set to dtype minimum
        """
        if sparse_mask is None:
            sparse_mask, _ = self.calculate_sparsity(attention_scores)

        if sparse_mask is None:
            return attention_scores

        mask_value = torch.finfo(attention_scores.dtype).min
        return attention_scores.masked_fill(~sparse_mask, mask_value)

    def get_threshold_info(self) -> dict[str, Any]:
        """Get threshold information for display."""
        calibration_params = self.calibration_params
        target_sparse_ratio = self.target_sparse_ratio

        if calibration_params is not None and target_sparse_ratio is not None:
            example_lengths = [4608, 6144, 9216, 18432]
            phase_info = {}
            for phase, params in calibration_params.items():
                if "threshold" not in params:
                    continue
                threshold = params["threshold"]
                target_sparsity = target_sparse_ratio.get(phase, 0.2)
                phase_info[phase] = {
                    "threshold": threshold,
                    "target_sparsity": target_sparsity,
                    "example_scaled_thresholds": {
                        length: threshold * math.log(length) for length in example_lengths
                    },
                }
            return {
                "type": "dynamic_calibrated_percentile",
                "formula": "skip if gap >= threshold * log(seq_k)",
                "calibration_params": calibration_params,
                "target_sparse_ratio": target_sparse_ratio,
                "phases": phase_info,
            }
        else:
            return {"type": "none", "value": "requires calibration"}

    @property
    def name(self) -> str:
        """Method identifier."""
        return "flash_skip_softmax_diffusion"
