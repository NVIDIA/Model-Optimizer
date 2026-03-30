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
        self.enable_lite_attention = config.get("enable_lite_attention", False)
        self.lite_threshold = config.get("lite_threshold", -5.0)
        # LLM-style: normalize_by_seqlen=False uses log2(threshold) directly
        # Diffusion-style: normalize_by_seqlen=True uses -threshold * log2(seq_k)
        self.normalize_by_seqlen = config.get("normalize_by_seqlen", True)

        # These are set by the Triton kernel integration and read by HF/diffusers backends
        self.skip_softmax_threshold: float | None = None
        self.skip_softmax_normalize_by_seqlen: bool = self.normalize_by_seqlen

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
            # Collect per-tile min gaps for percentile calibration.
            # gap shape: [batch, heads, block_rows, br, block_cols]
            #
            # The kernel skips a tile only when ALL rows agree (min of row
            # gaps >= threshold). So the calibration must collect per-tile
            # min gaps — the minimum normalized gap across the br rows within
            # each tile — to match the kernel's skip decision granularity.
            normalized_gap = gap / log_seq_k if self.normalize_by_seqlen else gap

            # Exclude padded rows: set padded positions to +inf so they
            # don't affect the min (padded rows have block_max = dtype.min)
            valid_mask = block_max > torch.finfo(attn_weights.dtype).min
            normalized_gap[~valid_mask] = float("inf")

            # Per-tile min gap: min over br rows (dim=-2)
            # [batch, heads, block_rows, block_cols]
            tile_min_gap = normalized_gap.min(dim=-2)[0]

            # Exclude fully-padded tiles (all rows padded → min is still inf)
            tile_valid = tile_min_gap < float("inf")
            valid_gaps = tile_min_gap[tile_valid].detach().float().cpu().numpy()
            del gap, normalized_gap, valid_mask, block_max, block_max_cummax
            del tile_min_gap, tile_valid

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

    def _compute_tiled_gaps(self, query, key):
        """Compute per-tile min gaps from Q/K without materializing the full score matrix.

        Streams KV tiles left-to-right, maintaining a running per-row cummax.
        For each tile, computes the per-row gap and immediately reduces to
        min-over-rows — only 1 scalar per tile is kept.

        Memory: O(num_br × br) for running_max, not O(num_br × br × num_bc).

        Args:
            query: [B, seq_q, heads, head_dim] (diffusers convention)
            key: [B, seq_k, heads, head_dim]

        Returns:
            1D numpy array of per-tile min normalized gaps
        """
        import numpy as np

        B, seq_q, heads, head_dim = query.shape
        seq_k = key.shape[1]
        scale = 1.0 / math.sqrt(head_dim)
        log_seq_k = math.log(seq_k)
        br, bc = self.br, self.bc

        padded_q = math.ceil(seq_q / br) * br
        num_br = padded_q // br
        num_bc = math.ceil(seq_k / bc)

        all_tile_gaps = []

        for b in range(B):
            for h in range(heads):
                q_h = query[b, :, h, :]  # [seq_q, dim]
                k_h = key[b, :, h, :]    # [seq_k, dim]

                # Pad Q to multiple of br
                if padded_q > seq_q:
                    q_padded = torch.zeros(padded_q, head_dim, device=q_h.device, dtype=q_h.dtype)
                    q_padded[:seq_q] = q_h
                else:
                    q_padded = q_h
                q_blocks = q_padded.view(num_br, br, head_dim)  # [num_br, br, dim]

                # Mask for valid rows (padded Q rows should not affect min)
                valid_rows = torch.ones(num_br, br, device=q_h.device, dtype=torch.bool)
                if padded_q > seq_q:
                    last_valid = seq_q - (num_br - 1) * br
                    if last_valid < br:
                        valid_rows[num_br - 1, last_valid:] = False

                # Running per-row cummax: [num_br, br]
                running_max = torch.full(
                    (num_br, br), float("-inf"), device=q_h.device, dtype=q_h.dtype
                )

                # Stream KV tiles left to right
                for kbc in range(num_bc):
                    k_start = kbc * bc
                    k_end = min(k_start + bc, seq_k)
                    k_tile = k_h[k_start:k_end]  # [<=bc, dim]

                    # [num_br, br, <=bc] -> per-row max -> [num_br, br]
                    tile_row_max = torch.bmm(
                        q_blocks, k_tile.T.unsqueeze(0).expand(num_br, -1, -1)
                    ).mul_(scale).max(dim=-1)[0]

                    # gap = running_max - tile_row_max (before updating running_max)
                    # For the first tile, running_max=-inf so gap=-inf (will be
                    # replaced by 0 after max with tile_row_max)
                    gap = running_max - tile_row_max  # [num_br, br], >= 0 for non-peak
                    gap = gap.clamp(min=0.0)
                    if self.normalize_by_seqlen:
                        gap = gap / log_seq_k

                    # Invalidate padded rows
                    gap[~valid_rows] = float("inf")

                    # Per-tile min gap: min over br rows -> [num_br]
                    tile_min = gap.min(dim=1)[0]
                    # Exclude fully-invalid tiles
                    tile_valid_mask = tile_min < float("inf")
                    if tile_valid_mask.any():
                        all_tile_gaps.append(
                            tile_min[tile_valid_mask].detach().float().cpu().numpy()
                        )

                    # Update running cummax
                    running_max = torch.maximum(running_max, tile_row_max)

        if all_tile_gaps:
            return np.concatenate(all_tile_gaps)
        return np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Global calibration dispatch hook (shared across all modules)
    # ------------------------------------------------------------------
    _calib_dispatch_installed = False
    _calib_dispatch_lock = __import__("threading").Lock()
    _calib_active_modules: list = []  # list of (method, module) pairs
    _calib_call_idx = 0  # tracks which module the current dispatch call belongs to

    def _eager_calibration_context(self, module: "SparseAttentionModule"):
        """Context manager for calibration via dispatch_attention_fn hooking.

        Installs a single global hook on dispatch_attention_fn (shared across all
        modules). Each self-attention call increments a counter to map to the
        correct sparse attention module.
        """
        cls = type(self)

        with ExitStack() as stack:
            from ..kernels import set_skip_softmax_context

            set_skip_softmax_context(True)
            stack.callback(set_skip_softmax_context, False)

            # Register this module
            entry = (self, module)
            cls._calib_active_modules.append(entry)
            stack.callback(cls._calib_active_modules.remove, entry)

            # Install global dispatch hook once
            if not cls._calib_dispatch_installed:
                cls._install_calib_dispatch(stack)

            yield

    @classmethod
    def _install_calib_dispatch(cls, stack):
        """Install a single global dispatch_attention_fn hook for calibration."""
        from modelopt.torch.quantization.utils import replace_function

        try:
            import diffusers.models.attention_dispatch as _dispatch_mod
            from diffusers.models.attention_dispatch import (
                dispatch_attention_fn as _orig_dispatch,
            )
        except ImportError:
            return

        def _capturing_dispatch(query, key, value, *, backend=None, **kwargs):
            is_self_attn = query.shape[1] == key.shape[1] and query.shape[1] >= 1024

            if is_self_attn and cls._calib_active_modules:
                # Map call to module by round-robin index
                idx = cls._calib_call_idx % len(cls._calib_active_modules)
                cls._calib_call_idx += 1

                method_inst, module_inst = cls._calib_active_modules[idx]

                if method_inst._calibration_mode:
                    gaps = method_inst._compute_tiled_gaps(query, key)
                    if len(gaps) > 0:
                        stats = {
                            "sparsity": [0.0],
                            "phase": "prefill",
                            "total_blocks": 0,
                            "sparse_blocks": [0],
                            "sample_length": key.shape[1],
                            "normalized_gaps": gaps,
                        }
                        module_inst._last_stats = stats

            return _orig_dispatch(query, key, value, backend=backend, **kwargs)

        stack.enter_context(
            replace_function(_dispatch_mod, "dispatch_attention_fn", _capturing_dispatch)
        )

        # Also patch modules with direct import references
        import diffusers.models.transformers

        for attr in dir(diffusers.models.transformers):
            mod = getattr(diffusers.models.transformers, attr, None)
            if (
                mod is not None
                and hasattr(mod, "dispatch_attention_fn")
                and getattr(mod, "dispatch_attention_fn", None) is _orig_dispatch
            ):
                stack.enter_context(
                    replace_function(mod, "dispatch_attention_fn", _capturing_dispatch)
                )

        cls._calib_dispatch_installed = True
        stack.callback(setattr, cls, "_calib_dispatch_installed", False)
        stack.callback(setattr, cls, "_calib_call_idx", 0)

    def _triton_inference_context(self, module: "SparseAttentionModule"):
        """Context manager for Triton inference (fused kernel tile skipping)."""
        threshold = self._effective_threshold
        use_lite = self.enable_lite_attention

        with ExitStack() as stack:
            if use_lite or (threshold is not None and threshold > 0.0):
                # Set skip-softmax config for the diffusers Triton backend
                try:
                    from ..kernels.diffusers_triton_attention import (
                        clear_triton_skip_softmax_config,
                        set_triton_skip_softmax_config,
                    )

                    set_triton_skip_softmax_config(
                        threshold=threshold if not use_lite else None,
                        normalize_by_seqlen=self.normalize_by_seqlen,
                        enable_v25=self.enable_v25,
                    )
                    stack.callback(clear_triton_skip_softmax_config)
                except ImportError:
                    pass

                # Set config for the LTX-2 Triton backend
                try:
                    from ..kernels.ltx_triton_attention import (
                        clear_ltx_triton_context,
                        set_ltx_triton_context,
                    )

                    set_ltx_triton_context(
                        active=True,
                        threshold=threshold if not use_lite else None,
                        normalize_by_seqlen=self.normalize_by_seqlen,
                        enable_v25=self.enable_v25,
                        lite_threshold=self.lite_threshold if use_lite else None,
                    )
                    stack.callback(clear_ltx_triton_context)
                except ImportError:
                    pass

                if not use_lite:
                    # Also set module flags for HF Triton backend (hf_triton_attention.py)
                    module._apply_skip_softmax = True
                    self.skip_softmax_threshold = threshold
                    self.skip_softmax_normalize_by_seqlen = self.normalize_by_seqlen
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
