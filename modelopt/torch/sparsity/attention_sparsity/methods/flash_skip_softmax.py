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
import torch.nn.functional as F

from modelopt.torch.quantization.utils import replace_function

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
            method_config: Configuration dict with thresholds, br, bc, is_causal, etc.
                          All required fields should have defaults from SparseAttentionAttributeConfig.
        """
        super().__init__()
        config = method_config or {}

        # Extract configuration
        self.thresholds_config = config["thresholds"]
        self.br = config["br"]
        self.bc = config["bc"]
        self.backend = config["backend"]
        self.is_causal = config["is_causal"]

        # Optional parameters not in Pydantic config
        self.phase = config.get("phase", None)

        # Initialize thresholds from dict config (prefill phase as default)
        self.thresholds = self.thresholds_config.get("prefill", [1e-3])

        # Calibration mode flag (prevents threshold updates during calibration)
        self._calibration_mode = False

    def set_calibration_mode(self, enabled: bool):
        """Set calibration mode to prevent _update_thresholds from modifying the thresholds."""
        self._calibration_mode = enabled

    def _update_thresholds(self, phase: str):
        """Update thresholds list based on phase."""
        self.thresholds = self.thresholds_config.get(phase, self.thresholds)

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

        # Check whether to use calibrated single-threshold path or multi-threshold list path
        calibration_params = self.calibration_params
        target_sparse_ratio = self.target_sparse_ratio
        use_calibration_params = (
            calibration_params is not None
            and phase in calibration_params
            and target_sparse_ratio is not None
        )

        if use_calibration_params:
            # Calibrated dynamic threshold: bypass thresholds list entirely
            assert calibration_params is not None and target_sparse_ratio is not None
            a = calibration_params[phase]["a"]
            b = calibration_params[phase]["b"]
            target_sparsity = target_sparse_ratio.get(phase, 0.5)
            scale_factor = a * np.exp(b * target_sparsity)
            log_thresholds = [np.log(scale_factor / seq_k)]
        else:
            log_thresholds = [np.log(t) for t in self.thresholds]

        if phase == "prefill":
            blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
                self._reshape_to_blocks(attn_weights, self.br, self.bc)
            )

            # Step 1: Compute maximum value in each block
            # blocked_attn: [batch, heads, block_rows, br, block_cols, bc]
            # block_max: [batch, heads, block_rows, br, block_cols]
            block_max = blocked_attn.max(dim=-1)[0]
            del blocked_attn  # free padded copy early; block_max holds what we need

            # Step 2: Track cumulative maximum across blocks (left to right)
            block_max_cummax = block_max.cummax(dim=-1)[0]

            # Step 3: Calculate correction factor
            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = (block_max_larger.sum() / block_max_larger.numel()).item()
            del block_max_larger

            # Pre-compute total_valid_blocks (same for all thresholds)
            if self.is_causal:
                num_causal_blocks = num_block_rows * (2 * num_block_cols - num_block_rows + 1) // 2
                total_valid_blocks = batch_size * num_heads * num_causal_blocks
                total_blocks = num_causal_blocks
            else:
                total_valid_blocks = batch_size * num_heads * num_block_rows * num_block_cols
                total_blocks = num_block_rows * num_block_cols

            # Step 4-5: Loop over thresholds, computing block mask and sparsity for each.
            # Only store block_mask for the first threshold (used for element_mask).
            # In calibration mode, skip element_mask entirely to save memory.
            # We compare block_max to block_max_cummax directly (avoids materializing the
            # full blocked_attn-sized intermediate tensor — saves ~1x attn weights per threshold).
            dense_blocks_list = []
            block_mask_0 = None
            block_diff = block_max - block_max_cummax
            for i, log_threshold in enumerate(log_thresholds):
                block_mask = (block_diff > log_threshold).any(dim=-2)

                dense_blocks_list.append(block_mask.sum().item())

                if i == 0 and not self._calibration_mode:
                    block_mask_0 = block_mask
                del block_mask

            del block_max, block_max_cummax

            # Step 6-7: Expand block_mask_0 to element level (skip in calibration mode)
            if not self._calibration_mode and block_mask_0 is not None:
                element_mask = (
                    block_mask_0.unsqueeze(-2)
                    .unsqueeze(-1)
                    .expand(batch_size, num_heads, num_block_rows, self.br, num_block_cols, self.bc)
                )
                del block_mask_0
                element_mask = element_mask.reshape(
                    batch_size, num_heads, padded_seq_q, padded_seq_k
                )
                element_mask = element_mask[:, :, :seq_q, :seq_k]
            else:
                element_mask = None

        else:  # decode
            blocked_attn, _, num_block_cols, _, padded_seq_k = self._reshape_to_blocks(
                attn_weights, 1, self.bc
            )

            # blocked_attn: [batch, heads, 1, 1, num_block_cols, bc]
            block_max = blocked_attn.max(dim=-1)[0]
            del blocked_attn  # free early; block_max holds what we need
            block_max_cummax = block_max.cummax(dim=-1)[0]

            block_max_larger = torch.ones_like(block_max)
            block_max_larger[..., 1:] = block_max[..., 1:] > block_max_cummax[..., :-1]
            correction_factor = (block_max_larger.sum() / block_max_larger.numel()).item()
            del block_max_larger

            total_valid_blocks = batch_size * num_heads * num_block_cols
            total_blocks = num_block_cols

            dense_blocks_list = []
            block_mask_0 = None
            for i, log_threshold in enumerate(log_thresholds):
                block_mask = block_max - block_max_cummax > log_threshold

                dense_blocks_list.append(block_mask.sum().item())

                if i == 0 and not self._calibration_mode:
                    block_mask_0 = block_mask
                del block_mask

            del block_max, block_max_cummax

            if not self._calibration_mode and block_mask_0 is not None:
                element_mask = block_mask_0[..., None].expand(
                    batch_size, num_heads, 1, 1, num_block_cols, self.bc
                )
                del block_mask_0
                element_mask = element_mask.reshape(batch_size, num_heads, 1, padded_seq_k)
                element_mask = element_mask[:, :, :seq_q, :seq_k]
            else:
                element_mask = None

        sparsity_list = [1.0 - d / total_valid_blocks for d in dense_blocks_list]

        sparsity_out = sparsity_list
        sparse_blocks_out = [int(s * total_blocks) for s in sparsity_list]

        stats = {
            "correction_factor": correction_factor,
            "sparsity": sparsity_out,
            "phase": phase,
            "total_blocks": total_blocks,
            "sparse_blocks": sparse_blocks_out,
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

        # Update thresholds for the detected phase (skip during calibration)
        if not self._calibration_mode:
            self._update_thresholds(phase)

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
        calibration_params = self.calibration_params
        target_sparse_ratio = self.target_sparse_ratio

        if calibration_params is not None and target_sparse_ratio is not None:
            # Per-phase calibrated dynamic threshold using Exponential model
            example_lengths = [1024, 4096, 16384, 65536, 131072]
            phase_info = {}
            for phase, params in calibration_params.items():
                a, b = params["a"], params["b"]
                target_sparsity = target_sparse_ratio.get(phase, 0.5)
                scale_factor = a * np.exp(b * target_sparsity)
                phase_info[phase] = {
                    "a": a,
                    "b": b,
                    "target_sparsity": target_sparsity,
                    "scale_factor": scale_factor,
                    "example_thresholds": {
                        length: scale_factor / length for length in example_lengths
                    },
                }
            return {
                "type": "dynamic_calibrated",
                "formula": "threshold = a * exp(b * target_sparsity) / seqlen",
                "calibration_params": calibration_params,
                "target_sparse_ratio": target_sparse_ratio,
                "phases": phase_info,
            }
        else:
            # Static thresholds (list per phase)
            return {
                "type": "static",
                "value": self.thresholds_config,
            }

    def get_sparse_context(self, module: torch.nn.Module):
        """Return a context manager that patches F.softmax with sparse masking."""
        original_softmax = F.softmax

        def sparse_softmax(input, dim=-1, *args, **kwargs):
            sparse_mask, stats = self.calculate_sparsity(input)
            module._last_stats = stats
            if not self._calibration_mode:
                input = self.apply_sparsity(input, sparse_mask)
            return original_softmax(input, dim, *args, **kwargs)

        return replace_function(torch.nn.functional, "softmax", sparse_softmax)

    @property
    def name(self) -> str:
        """Method identifier."""
        return "flash_skip_softmax"
