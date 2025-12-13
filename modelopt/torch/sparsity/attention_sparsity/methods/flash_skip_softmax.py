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

"""Flash Attention-aware softmax skip method for sparse attention.

This module implements block-wise sparsity that aligns with Flash Attention's
processing pattern for optimal performance.
"""

import math
from typing import Any

import numpy as np
import torch

from . import SparseAttentionMethod, register_sparse_method


@register_sparse_method("flash_skip_softmax")
class FlashSkipSoftmax(SparseAttentionMethod):
    """Flash Attention-aware softmax skip sparse attention method.

    Implements row-level block-wise sparsity aligned with Flash Attention's
    processing pattern for optimal performance and accuracy.
    """

    def __init__(self, method_config: dict | None = None):
        """Initialize Flash softmax skip method.

        Args:
            method_config: Configuration dict with threshold, br, bc, is_causal, etc.
                          All required fields should have defaults from SparseAttentionAttributeConfig.
        """
        config = method_config or {}

        # Extract configuration
        self.threshold_config = config["threshold"]
        self.br = config["br"]
        self.bc = config["bc"]
        self.backend = config["backend"]
        self.is_causal = config["is_causal"]

        # Optional parameters not in Pydantic config
        self.phase = config.get("phase", None)

        # Calibration mode: when True, prevent threshold updates to preserve calibrator's test threshold
        self._calibration_mode = False

        # Initialize threshold
        if isinstance(self.threshold_config, dict):
            self.threshold = self.threshold_config.get(
                "default", self.threshold_config.get("prefill", 1e-4)
            )
        else:
            self.threshold = self.threshold_config

    def set_calibration_mode(self, enabled: bool):
        """Set calibration mode to prevent _update_threshold from modifying the threshold."""
        self._calibration_mode = enabled

    def _update_threshold(self, phase: str):
        """Update threshold based on phase."""
        if isinstance(self.threshold_config, dict):
            self.threshold = self.threshold_config.get(
                phase, self.threshold_config.get("default", self.threshold)
            )

    def _infer_phase(self, attention_scores: torch.Tensor) -> str:
        """Infer phase from attention scores shape."""
        return "decode" if attention_scores.shape[2] == 1 else "prefill"

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

        # Calculate padding needed
        padded_seq_q = math.ceil(seq_q / br) * br
        padded_seq_k = math.ceil(seq_k / bc) * bc

        # Pad tensor if necessary
        if padded_seq_q != seq_q or padded_seq_k != seq_k:
            pad_q = padded_seq_q - seq_q
            pad_k = padded_seq_k - seq_k
            # Use dtype min instead of -inf for numerical stability
            pad_value = torch.finfo(tensor.dtype).min
            tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_q), value=pad_value)

        # Reshape to blocks
        num_block_rows = padded_seq_q // br
        num_block_cols = padded_seq_k // bc

        # Keep natural order for row-level processing: [batch, heads, block_rows, br, block_cols, bc]
        blocked = tensor.view(batch_size, num_heads, num_block_rows, br, num_block_cols, bc)

        return blocked, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k

    def calc_correction_factor_and_p(
        self, attn_weights: torch.Tensor, phase: str
    ) -> tuple[torch.Tensor, dict]:
        """Calculate sparse mask and statistics for Flash Attention.

        Implements block-wise sparsity compatible with Flash Attention's online softmax:
        1. Reshape attention scores into 128x128 blocks
        2. Track block-wise maximum values (simulating Flash Attention's row processing)
        3. Compute cumulative maximum across blocks (for online normalization)
        4. Apply threshold: mask blocks where p = score - cummax < log(threshold)
        5. Calculate correction factor and sparsity statistics

        Args:
            attn_weights: Pre-softmax attention scores [batch, heads, seq_q, seq_k]
            phase: "prefill" (seq_q > 1) or "decode" (seq_q = 1)

        Returns:
            element_mask: Boolean mask [batch, heads, seq_q, seq_k]
            stats: Dict with sparsity, correction_factor, total_blocks, etc.
        """
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape

        # Calculate threshold
        threshold_scale_factor = getattr(self, "threshold_scale_factor", None)
        if threshold_scale_factor:
            # Use calibrated dynamic threshold: λ = scale_factor / length
            log_threshold = np.log(threshold_scale_factor / seq_k)
        else:
            # Use static threshold from config
            log_threshold = np.log(self.threshold)

        if phase == "prefill":
            blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
                self._reshape_to_blocks(attn_weights, self.br, self.bc)
            )

            # Step 1: Compute maximum value in each block
            # For each 128x128 block, find max across the 128 columns
            # blocked_attn: [batch, heads, block_rows, br=128, block_cols, bc=128]
            # block_max: [batch, heads, block_rows, br=128, block_cols]
            block_max = blocked_attn.max(dim=-1)[0]

            # Step 2: Track cumulative maximum across blocks (left to right)
            # This simulates Flash Attention's online softmax normalization
            # block_max_cummax: [batch, heads, block_rows, br=128, block_cols]
            block_max_cummax = block_max.cummax(dim=-1)[0]

            # Step 3: Calculate correction factor (how often max changes)
            # Used by Flash Attention to adjust running sum when max increases
            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = float(torch.sum(block_max_larger) / torch.numel(block_max_larger))

            # Step 4: Normalize attention scores by cumulative max
            # p represents log-space difference: log(score) - log(cummax)
            p = blocked_attn - block_max_cummax[..., None]

            # Step 5: Apply threshold and create block-level mask
            # Keep blocks where at least one element exceeds log(threshold)
            p_larger_than_thresh = p > log_threshold
            # Reduce over bc (128 cols), then br (128 rows) to get block-level decision
            # Result: [batch, heads, block_rows, block_cols]
            block_mask = p_larger_than_thresh.any(dim=-1).any(dim=-2)

            # Step 6: Expand block mask back to element level
            # All 128x128 elements in a block share the same mask value
            # [batch, heads, block_rows, block_cols] -> [batch, heads, block_rows, br=128, block_cols, bc=128]
            element_mask = block_mask.unsqueeze(-2).unsqueeze(-1).expand_as(blocked_attn)

            # Step 7: Reshape to original attention shape and remove padding
            element_mask = element_mask.reshape(batch_size, num_heads, padded_seq_q, padded_seq_k)
            element_mask = element_mask[:, :, :seq_q, :seq_k]

            # Step 8: Calculate sparsity statistics
            # density = sum(mask) / numel(mask) * N / (N+1) for causal
            if self.is_causal:
                density = float(block_mask.sum() / block_mask.numel()) * (
                    num_block_rows / (num_block_rows + 1)
                )
            else:
                density = float(block_mask.sum() / block_mask.numel())
            sparsity = 1 - density
            total_blocks = num_block_rows * num_block_cols
        else:  # decode
            blocked_attn, _, num_block_cols, _, padded_seq_k = self._reshape_to_blocks(
                attn_weights, 1, self.bc
            )

            # Decode: Single query row attends to all past key blocks
            # blocked_attn: [batch, heads, 1, 1, num_block_cols, bc=128]

            # Step 1: Find maximum in each key block
            # block_max: [batch, heads, 1, 1, num_block_cols]
            block_max = blocked_attn.max(dim=-1)[0]

            # Step 2: Track cumulative maximum across key blocks (left to right)
            # Simulates Flash Attention's online softmax normalization
            block_max_cummax = block_max.cummax(dim=-1)[0]

            # Step 3: Calculate correction factor
            # Tracks how often the maximum increases (needed for Flash Attention rescaling)
            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = float(torch.sum(block_max_larger) / torch.numel(block_max_larger))

            # Step 4: Normalize scores by cumulative max
            # p = log(score) - log(cummax) in log-space
            p = blocked_attn - block_max_cummax[..., None]

            # Step 5: Apply threshold and create block mask
            # Keep blocks where at least one element exceeds threshold
            p_larger_than_thresh = p > log_threshold
            block_mask = p_larger_than_thresh.any(dim=-1, keepdim=False)

            # Step 6: Expand to element level and remove padding
            element_mask = block_mask[..., None].expand_as(blocked_attn)
            element_mask = element_mask.reshape(batch_size, num_heads, 1, padded_seq_k)
            element_mask = element_mask[:, :, :seq_q, :seq_k]

            # Step 7: Calculate sparsity statistics
            density = float(block_mask.sum() / block_mask.numel())
            sparsity = 1 - density
            total_blocks = num_block_cols

        # Create stats dictionary
        stats = {
            "correction_factor": correction_factor,
            "sparsity": sparsity,
            "phase": phase,
            "total_blocks": total_blocks,
            "sparse_blocks": int(sparsity * total_blocks),
            "sample_length": seq_k,
        }

        return element_mask, stats

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Calculate sparsity mask and statistics for Flash Attention.

        Args:
            attention_scores: Attention scores tensor with shape [batch, heads, seq_q, seq_k]

        Returns:
            Tuple of (sparse_mask, stats) where sparse_mask is boolean mask
        """
        # Attention scores are always 4D: [batch, heads, seq_q, seq_k]
        assert len(attention_scores.shape) == 4, (
            f"Expected 4D attention scores, got shape {attention_scores.shape}"
        )

        # Infer phase from tensor shape
        phase = self._infer_phase(attention_scores)

        # Update threshold for the detected phase (skip during calibration)
        if not self._calibration_mode:
            self._update_threshold(phase)

        # Calculate block-wise sparsity mask and stats
        sparse_mask, stats = self.calc_correction_factor_and_p(attention_scores, phase)

        # Store stats for module to collect
        self._last_stats = stats

        return sparse_mask, stats

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparsity mask to attention scores.

        Args:
            attention_scores: Attention scores tensor [batch, heads, seq_q, seq_k]
            sparse_mask: Optional pre-computed boolean mask. If None, calculates internally.

        Returns:
            Masked attention scores with sparse elements set to dtype minimum
        """
        if sparse_mask is None:
            sparse_mask, _ = self.calculate_sparsity(attention_scores)

        # Apply mask: set masked positions to minimum value (becomes 0 after softmax)
        mask_value = torch.finfo(attention_scores.dtype).min
        return attention_scores.masked_fill(~sparse_mask, mask_value)

    def get_threshold_info(self) -> dict[str, Any]:
        """Get threshold information for this method.

        Returns:
            Dictionary with threshold configuration and calibration info.
        """
        threshold_scale_factor = getattr(self, "threshold_scale_factor", None)

        if threshold_scale_factor is not None:
            # Calibrated dynamic threshold
            return {
                "type": "dynamic",
                "scale_factor": threshold_scale_factor,
                "formula": "λ / length",
                "example_lengths": {
                    1024: threshold_scale_factor / 1024,
                    2048: threshold_scale_factor / 2048,
                    4096: threshold_scale_factor / 4096,
                    8192: threshold_scale_factor / 8192,
                },
            }
        else:
            # Static threshold (single value or phase-specific dict)
            return {
                "type": "static",
                "value": self.threshold_config,
            }

    @property
    def name(self) -> str:
        """Method identifier."""
        return "flash_skip_softmax"
