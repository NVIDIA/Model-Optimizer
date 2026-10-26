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

"""Skip-softmax method for attention via Triton kernel tile skipping.

Supports two modes:
- **Inference**: KV tiles with negligible scores are skipped in-kernel.
- **Calibration**: The Triton calibration kernel collects multi-threshold
  sparsity statistics without skipping any tiles.
"""

from contextlib import contextmanager

import torch

from .registry import SparseAttentionMethod, register_sparse_method


@register_sparse_method("triton_skip_softmax")
class TritonSkipSoftmaxMethod(SparseAttentionMethod):
    """Skip-softmax tile skipping via the Triton flash attention kernel.

    During prefill, KV tiles whose max attention score is far below the
    running softmax max are skipped entirely — no V load, no softmax
    update, no accumulation. This is a long-context optimization that
    benefits sequences with strong attention locality.

    Config params:
        skip_softmax_threshold: Tiles contributing less than this fraction
            are skipped. Typical values: 1e-3 to 1e-1. Set to 0 to disable.
    """

    def __init__(self, method_config=None):
        """Initialize with skip-softmax threshold from config."""
        super().__init__()
        method_config = method_config or {}
        self.skip_softmax_threshold = method_config.get("skip_softmax_threshold", 0.1)
        # Calibration state
        self._threshold_trials: list[float] | None = None
        # HF (modelopt_triton) backend calibration outputs, accumulated across
        # attention calls in one forward pass and read back in
        # ``_collect_calibration_stats``. The HF backend reads/writes these
        # directly on the method instance (no thread-local needed).
        self._hf_calibration_counters: torch.Tensor | None = None
        self._hf_calibration_seq_k: int | None = None
        self._hf_calibration_is_decode: bool = False
        # Runtime sparsity measurement
        self._measure_sparsity: bool = False
        self._sparsity_total: int = 0
        self._sparsity_skipped: int = 0

    @property
    def name(self) -> str:
        """Method name identifier."""
        return "triton_skip_softmax"

    def calculate_sparsity(self, attention_scores):
        """Return a no-op mask (skip decision is made inside the Triton kernel)."""
        mask = torch.ones_like(attention_scores, dtype=torch.bool)
        return mask, {}

    def apply_sparsity(self, attention_scores, sparse_mask=None):
        """Not supported — tile skipping is fused into the Triton kernel."""
        raise NotImplementedError(
            "triton_skip_softmax applies tile skipping inside the Triton kernel. "
            "Use backend='triton', not backend='pytorch'."
        )

    def get_sparse_context(self, module):
        """Return context manager that activates skip-softmax during forward.

        In calibration mode, configures the Triton backend to use the
        calibration kernel which collects multi-threshold sparsity stats.
        In inference mode, sets the skip threshold for tile skipping.
        """
        if self._calibration_mode and self._threshold_trials:
            return self._triton_calibration_context(module)
        return self._triton_inference_context(module)

    @contextmanager
    def _triton_inference_context(self, module):
        """Inference: activate skip-softmax with calibrated or fixed threshold."""
        module._apply_skip_softmax = True

        backend_kwargs: dict = {}
        if self._measure_sparsity:
            backend_kwargs["measure_sparsity"] = True

        # Priority: calibrated dynamic threshold > static threshold.
        scale_factor = self._get_scale_factor()
        if scale_factor is not None:
            self._set_triton_backends(scale_factor=scale_factor, **backend_kwargs)
        else:
            self._set_triton_backends(threshold=self.skip_softmax_threshold, **backend_kwargs)
        with self._get_diffusers_backend_context():
            try:
                yield
            finally:
                # Collect accumulated runtime sparsity counters before clearing
                if self._measure_sparsity:
                    self._collect_sparsity_counters()
                module._apply_skip_softmax = False
                self._clear_triton_backends()

    @contextmanager
    def _triton_calibration_context(self, module):
        """Calibration: collect multi-threshold sparsity stats via Triton kernel."""
        module._apply_skip_softmax = True
        # Reset the HF-backend calibration accumulators for this forward pass.
        # (The diffusers/LTX backends reset their own state in ``_set_triton_backends``.)
        self._hf_calibration_counters = None
        self._hf_calibration_seq_k = None
        self._hf_calibration_is_decode = False
        self._set_triton_backends(calibration_mode=True, threshold_trials=self._threshold_trials)
        with self._get_diffusers_backend_context():
            try:
                yield
                # After forward pass, extract counters and build stats
                self._collect_calibration_stats(module)
            finally:
                module._apply_skip_softmax = False
                self._clear_triton_backends()

    def _get_scale_factor(self, phase: str = "prefill") -> float | None:
        """Compute the scale_factor for ``phase`` from calibration params, or None.

        The scale_factor is sequence-length-independent. Callers divide by the
        actual ``seq_k`` at call time: ``threshold = scale_factor / seq_k``.
        """
        if self.calibration_params and self.target_sparse_ratio:
            import math
            import warnings

            params = self.calibration_params.get(phase, {})
            a = params.get("a", 0)
            b = params.get("b", 0)
            target = self.target_sparse_ratio.get(phase, 0.5)
            if a > 0 and b > 0:
                # Warn if target is outside the calibrated range
                min_s = params.get("min_observed_sparsity")
                max_s = params.get("max_observed_sparsity")
                if min_s is not None and target < min_s:
                    warnings.warn(
                        f"Target sparsity {target:.1%} is below the minimum observed "
                        f"during calibration ({min_s:.1%}). The model is extrapolating "
                        f"and runtime sparsity will likely be higher than the target.",
                        stacklevel=2,
                    )
                elif max_s is not None and target > max_s:
                    warnings.warn(
                        f"Target sparsity {target:.1%} is above the maximum observed "
                        f"during calibration ({max_s:.1%}). The model is extrapolating.",
                        stacklevel=2,
                    )
                return a * math.exp(b * target)
        return None

    def get_inference_threshold(self, seq_q: int, seq_k: int) -> float | None:
        """Return the skip threshold to apply for this call's phase.

        Picks the phase from the query length (``decode`` when ``seq_q == 1``,
        else ``prefill``) and returns the calibrated dynamic threshold
        ``scale_factor(phase) / seq_k`` when the phase is calibrated, otherwise
        the static ``skip_softmax_threshold`` (or ``None`` to disable). This is
        what the HF backend applies; it keeps prefill and decode on their own
        calibrated ``(a, b)`` instead of forcing decode onto prefill's.
        """
        phase = "decode" if seq_q <= 1 else "prefill"
        scale_factor = self._get_scale_factor(phase)
        if scale_factor is not None and seq_k > 0:
            return scale_factor / seq_k
        return self.skip_softmax_threshold or None

    @staticmethod
    @contextmanager
    def _get_diffusers_backend_context():
        """Activate the modelopt_triton diffusers backend if registered."""
        try:
            from modelopt.torch.kernels.sparsity.attention.diffusers_triton_attention import (
                get_triton_attention_backend,
            )

            with get_triton_attention_backend():
                yield
        except (ImportError, RuntimeError):
            yield

    def _set_triton_backends(self, **kwargs):
        """Set config on the diffusers and LTX Triton backends.

        The HF (modelopt_triton) backend reads its calibration config directly
        from this method instance during ``triton_attention_forward``, so it
        needs no separate configuration here.
        """
        try:
            from modelopt.torch.kernels.sparsity.attention.diffusers_triton_attention import (
                set_triton_skip_softmax_config,
            )

            set_triton_skip_softmax_config(**kwargs)
        except ImportError:
            pass
        try:
            from modelopt.torch.kernels.sparsity.attention.ltx_triton_attention import (
                set_ltx_triton_context,
            )

            set_ltx_triton_context(active=True, **kwargs)
        except ImportError:
            pass

    def _clear_triton_backends(self):
        """Clear config on the diffusers and LTX Triton backends."""
        try:
            from modelopt.torch.kernels.sparsity.attention.diffusers_triton_attention import (
                clear_triton_skip_softmax_config,
            )

            clear_triton_skip_softmax_config()
        except ImportError:
            pass
        try:
            from modelopt.torch.kernels.sparsity.attention.ltx_triton_attention import (
                clear_ltx_triton_context,
            )

            clear_ltx_triton_context()
        except ImportError:
            pass

    def _collect_calibration_stats(self, module):
        """Read Triton calibration counters and store as stats on the module."""
        counters = None
        seq_k = None
        # Diffusers/LTX (video) backends are prefill-only; only the HF backend
        # reports a phase, for decode-step calibration.
        phase = "prefill"

        try:
            from modelopt.torch.kernels.sparsity.attention.diffusers_triton_attention import (
                get_calibration_counters,
                get_calibration_seq_k,
            )

            counters = get_calibration_counters()
            seq_k = get_calibration_seq_k()
        except ImportError:
            pass

        if counters is None:
            try:
                from modelopt.torch.kernels.sparsity.attention.ltx_triton_attention import (
                    get_calibration_counters,
                    get_calibration_seq_k,
                )

                counters = get_calibration_counters()
                seq_k = get_calibration_seq_k()
            except ImportError:
                pass

        if counters is None:
            # HF (modelopt_triton) backend accumulates counters on this method
            # instance (``module._sparse_method_instance is self``).
            counters = self._hf_calibration_counters
            seq_k = self._hf_calibration_seq_k
            if counters is not None and self._hf_calibration_is_decode:
                phase = "decode"

        if counters is None or self._threshold_trials is None:
            return

        # counters: [num_thresholds, 2] — [:, 0]=total, [:, 1]=skipped
        total = counters[:, 0].float()
        skipped = counters[:, 1].float()
        sparsity_list = (skipped / total.clamp(min=1)).tolist()

        # Use actual KV sequence length from backend for the exponential model fit.
        # The calibrator uses: scale_factor = threshold * sample_length, so this
        # must be the real sequence length, not the total tile count.
        sample_length = seq_k if seq_k is not None else 0

        module._last_stats = {
            "sparsity": sparsity_list,
            "sample_length": sample_length,
            "phase": phase,
        }

    def get_threshold_info(self) -> dict:
        """Get threshold information for debugging/display."""
        scale_factor = self._get_scale_factor()
        if scale_factor is not None:
            return {
                "type": "dynamic_calibrated",
                "formula": "threshold = scale_factor / seq_k (computed at runtime)",
                "scale_factor": scale_factor,
                "calibration_params": self.calibration_params,
                "target_sparse_ratio": self.target_sparse_ratio,
            }
        return {
            "type": "static",
            "value": self.skip_softmax_threshold,
        }

    # ------------------------------------------------------------------
    # Runtime sparsity measurement
    # ------------------------------------------------------------------

    def enable_measure_sparsity(self, enabled: bool = True) -> None:
        """Enable or disable runtime sparsity measurement."""
        self._measure_sparsity = enabled

    def reset_sparsity_counters(self) -> None:
        """Reset accumulated sparsity counters to zero."""
        self._sparsity_total = 0
        self._sparsity_skipped = 0

    def get_sparsity_counters(self) -> tuple[int, int]:
        """Return accumulated ``(total_tiles, skipped_tiles)``."""
        return self._sparsity_total, self._sparsity_skipped

    def _collect_sparsity_counters(self) -> None:
        """Read runtime sparsity counters from the backend and accumulate."""
        try:
            from modelopt.torch.kernels.sparsity.attention.diffusers_triton_attention import (
                get_sparsity_counters,
            )

            total, skipped = get_sparsity_counters()
            self._sparsity_total += total
            self._sparsity_skipped += skipped
        except ImportError:
            pass
