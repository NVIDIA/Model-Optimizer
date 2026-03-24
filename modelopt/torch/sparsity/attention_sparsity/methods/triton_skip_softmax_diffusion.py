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

"""Triton-based skip-softmax method for diffusion models.

Uses gap/log(seq_k) normalization for sequence-length-invariant thresholds
and percentile-based calibration.

Two modes controlled by ``_calibration_mode``:
- **Calibration**: eager attention with F.softmax patching to collect gap statistics.
- **Inference**: Triton FA kernel with fused tile skipping and gap/log(seq_k) normalization.
"""

import math
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

import torch

from . import SparseAttentionMethod, register_sparse_method

if TYPE_CHECKING:
    from ..sparse_attention import SparseAttentionModule


@register_sparse_method("triton_skip_softmax_diffusion")
class TritonSkipSoftmaxDiffusion(SparseAttentionMethod):
    """Triton-based skip-softmax for diffusion models.

    Uses gap/log(seq_k) normalization for sequence-length-invariant thresholds.
    During calibration, runs eager attention to collect gap statistics.
    During inference, runs Triton FA kernel with fused tile skipping.
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
        self.backend = config.get("backend", "triton")
        self.is_causal = config.get("is_causal", False)
        self.enable_v25 = config.get("enable_v25", False)

        # These are set by the Triton kernel integration and read by HF/diffusers backends
        self.skip_softmax_threshold: float | None = None
        self.skip_softmax_normalize_by_seqlen: bool = True

    def set_calibration_mode(self, enabled: bool):
        """Set calibration mode."""
        self._calibration_mode = enabled

    @property
    def _effective_threshold(self) -> float | None:
        """Get the effective threshold from calibration params."""
        if self.calibration_params is not None:
            prefill_params = self.calibration_params.get("prefill", {})
            if "threshold" in prefill_params:
                return prefill_params["threshold"]
        return self.skip_softmax_threshold

    # -----------------------------------------------------------------------
    # Calibration-mode helpers (eager attention path)
    # -----------------------------------------------------------------------

    def _reshape_to_blocks(
        self, tensor: torch.Tensor, br: int, bc: int
    ) -> tuple[torch.Tensor, ...]:
        """Reshape tensor into blocks for Flash Attention processing."""
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

    def _calc_gaps_and_mask(
        self, attn_weights: torch.Tensor, phase: str
    ) -> tuple[torch.Tensor | None, dict]:
        """Calculate sparse mask using gap/log(seq_k) normalization.

        During calibration: collects normalized gaps, returns no mask.
        During inference (eager fallback): applies threshold, returns element mask.
        """
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape
        log_seq_k = math.log(seq_k)

        blocked_attn, num_block_rows, num_block_cols, padded_seq_q, padded_seq_k = (
            self._reshape_to_blocks(attn_weights, self.br, self.bc)
        )

        # block_max: per-row max over bc
        block_max = blocked_attn.max(dim=-1)[0]
        del blocked_attn

        # cummax: per-row running maximum across tiles (left to right)
        block_max_cummax = block_max.cummax(dim=-1)[0]

        # gap = cummax - block_max (>= 0 for non-peak tiles)
        gap = block_max_cummax - block_max

        total_valid_blocks = batch_size * num_heads * num_block_rows * num_block_cols
        total_blocks = num_block_rows * num_block_cols

        if self._calibration_mode:
            # Collect per-row normalized gaps for percentile calibration.
            # gap shape: [batch, heads, block_rows, br, block_cols]
            # We collect ALL per-row gaps (not per-block min) because the
            # percentile on per-row gaps produces the correct threshold for
            # the block-level skip decision (skip block if ALL rows agree).
            normalized_gap = gap / log_seq_k

            # Exclude padded rows/tiles: padded positions have block_max = dtype.min
            valid_mask = block_max > torch.finfo(attn_weights.dtype).min
            valid_gaps = normalized_gap[valid_mask].detach().float().cpu().numpy()
            del gap, normalized_gap, valid_mask, block_max, block_max_cummax

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
        threshold = self._effective_threshold
        if threshold is None:
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

        # Keep tile if ANY row has gap < threshold
        block_mask = (gap < scaled_threshold).any(dim=-2)
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

    # -----------------------------------------------------------------------
    # Context manager: switches between calibration (eager) and inference (Triton)
    # -----------------------------------------------------------------------

    @contextmanager
    def get_sparse_context(self, module: "SparseAttentionModule"):
        """Return context that activates skip-softmax sparse attention.

        - Calibration mode: activates eager attention with F.softmax patching
          to collect gap statistics.
        - Inference mode: activates Triton FA kernel via the diffusers
          ``modelopt_triton`` backend with skip-softmax threshold.
        """
        if self._calibration_mode:
            yield from self._eager_calibration_context(module)
        else:
            yield from self._triton_inference_context(module)

    def _eager_calibration_context(self, module: "SparseAttentionModule"):
        """Context manager for eager calibration (F.softmax patching)."""
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

    def _triton_inference_context(self, module: "SparseAttentionModule"):
        """Context manager for Triton inference (fused kernel tile skipping)."""
        threshold = self._effective_threshold

        with ExitStack() as stack:
            if threshold is not None and threshold > 0.0:
                # Set skip-softmax config for the diffusers Triton backend
                try:
                    from ..kernels.diffusers_triton_attention import (
                        clear_triton_skip_softmax_config,
                        set_triton_skip_softmax_config,
                    )

                    set_triton_skip_softmax_config(
                        threshold=threshold,
                        normalize_by_seqlen=True,
                        enable_v25=self.enable_v25,
                    )
                    stack.callback(clear_triton_skip_softmax_config)
                except ImportError:
                    pass

                # Set skip-softmax config for the LTX-2 Triton backend
                try:
                    from ..kernels.ltx_triton_attention import (
                        clear_ltx_triton_context,
                        set_ltx_triton_context,
                    )

                    set_ltx_triton_context(
                        active=True,
                        threshold=threshold,
                        normalize_by_seqlen=True,
                        enable_v25=self.enable_v25,
                    )
                    stack.callback(clear_ltx_triton_context)
                except ImportError:
                    pass

                # Also set module flags for HF Triton backend (hf_triton_attention.py)
                module._apply_skip_softmax = True
                self.skip_softmax_threshold = threshold
                self.skip_softmax_normalize_by_seqlen = True
                stack.callback(setattr, module, "_apply_skip_softmax", False)

            # Activate the diffusers Triton backend
            try:
                from ..kernels.diffusers_triton_attention import get_triton_attention_backend

                stack.enter_context(get_triton_attention_backend())
            except (ImportError, RuntimeError):
                pass

            yield

    # -----------------------------------------------------------------------
    # SparseAttentionMethod interface
    # -----------------------------------------------------------------------

    def calculate_sparsity(
        self,
        attention_scores: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict]:
        """Calculate sparsity mask and statistics (eager path only)."""
        assert len(attention_scores.shape) == 4, (
            f"Expected 4D attention scores, got shape {attention_scores.shape}"
        )
        phase = "prefill"  # Diffusion models always use prefill
        sparse_mask, stats = self._calc_gaps_and_mask(attention_scores, phase)
        self._last_stats = stats
        return sparse_mask, stats

    def apply_sparsity(
        self,
        attention_scores: torch.Tensor,
        sparse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparsity mask to attention scores (eager path only)."""
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
            return {
                "type": "dynamic_calibrated_percentile",
                "formula": "skip if gap >= threshold * log(seq_k)",
                "calibration_params": calibration_params,
                "target_sparse_ratio": target_sparse_ratio,
            }
        return {"type": "none", "value": "requires calibration"}

    @property
    def name(self) -> str:
        """Method identifier."""
        return "triton_skip_softmax_diffusion"
