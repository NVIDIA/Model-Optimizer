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

"""Triton flash attention wrapper for LTX-2 (ltx_core) skip-softmax sparse attention.

Two modes:
- **Inference**: ``attention()`` with skip-softmax tile skipping.
- **Calibration**: ``attention_calibrate()`` to collect multi-threshold stats.
"""

import math
import threading

import torch

# ``attention`` and ``attention_calibrate`` are resolved lazily inside the
# call-site functions below. Capturing them at module top-level would fetch
# ``None`` from the partially-loaded ``common.attention`` package during the
# sparsity↔common circular import chain.
from modelopt.torch.utils.logging import warn_rank_0

# Thread-local storage for skip-softmax configuration
_thread_local = threading.local()


def set_ltx_triton_context(
    active: bool,
    threshold: float | None = None,
    calibration_mode: bool = False,
    threshold_trials: list[float] | None = None,
    scale_factor: float | None = None,
    raw_threshold: float | None = None,
    **kwargs,
) -> None:
    """Set thread-local Triton config for LTX-2 attention."""
    _thread_local.active = active
    _thread_local.threshold = threshold
    _thread_local.calibration_mode = calibration_mode
    _thread_local.threshold_trials = threshold_trials
    _thread_local.scale_factor = scale_factor
    _thread_local.raw_threshold = raw_threshold
    if not calibration_mode:
        _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None


def clear_ltx_triton_context() -> None:
    """Clear thread-local Triton config."""
    _thread_local.active = False
    _thread_local.threshold = None
    _thread_local.calibration_mode = False
    _thread_local.threshold_trials = None
    _thread_local.scale_factor = None
    _thread_local.raw_threshold = None
    _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None


def _get_ltx_triton_context() -> tuple[bool, float | None, float | None]:
    """Return (active, threshold, scale_factor)."""
    return (
        getattr(_thread_local, "active", False),
        getattr(_thread_local, "threshold", None),
        getattr(_thread_local, "scale_factor", None),
    )


def get_calibration_counters() -> "torch.Tensor | None":
    """Return accumulated calibration counters ``[num_thresholds, 2]`` or None."""
    return getattr(_thread_local, "calibration_counters", None)


def get_calibration_seq_k() -> int | None:
    """Return KV sequence length observed during calibration, or None."""
    return getattr(_thread_local, "calibration_seq_k", None)


def _ltx_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    threshold: float | None = None,
) -> torch.Tensor:
    """Triton FA attention on LTX-2 layout ``[B, T, H*D]``."""
    b, seq_q, dim_total = q.shape
    dim_head = dim_total // heads
    seq_k = k.shape[1]
    device = q.device

    q_flat = q.view(b, seq_q, heads, dim_head).reshape(b * seq_q, heads, dim_head).contiguous()
    k_flat = k.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()
    v_flat = v.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()

    b_start_loc_q = torch.arange(b, device=device, dtype=torch.int32) * seq_q
    b_seq_len_q = torch.full((b,), seq_q, device=device, dtype=torch.int32)

    scale = 1.0 / math.sqrt(dim_head)

    kw: dict = {
        "b_start_loc": b_start_loc_q,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_q,
        "is_causal": False,
        "softmax_scale": scale,
    }

    if seq_q != seq_k:
        b_start_loc_k = torch.arange(b, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((b,), seq_k, device=device, dtype=torch.int32)
        kw["b_start_loc_k"] = b_start_loc_k
        kw["b_seq_len_k"] = b_seq_len_k
        kw["max_input_len_k"] = seq_k

    # --- Calibration mode ---
    calib_mode = getattr(_thread_local, "calibration_mode", False)
    if calib_mode:
        trials = getattr(_thread_local, "threshold_trials", None)
        from modelopt.torch.kernels.common.attention import attention_calibrate

        if trials and attention_calibrate is not None:
            o, counters = attention_calibrate(q_flat, k_flat, v_flat, **kw, threshold_trials=trials)

            prev = getattr(_thread_local, "calibration_counters", None)
            if prev is None:
                _thread_local.calibration_counters = counters
            else:
                _thread_local.calibration_counters = prev + counters

            # Store actual KV sequence length for calibration stats
            _thread_local.calibration_seq_k = seq_k

            return o.view(b, seq_q, heads * dim_head)

    # --- Inference mode: raw, dynamic, or static threshold ---
    raw_thresh = getattr(_thread_local, "raw_threshold", None)
    scale_factor = getattr(_thread_local, "scale_factor", None)
    if raw_thresh is not None:
        kw["skip_softmax_raw_threshold"] = raw_thresh
    elif scale_factor is not None and scale_factor > 0.0:
        kw["skip_softmax_threshold"] = scale_factor / seq_k
    elif threshold is not None and threshold > 0.0:
        kw["skip_softmax_threshold"] = threshold

    from modelopt.torch.kernels.common.attention import attention

    assert attention is not None, "Triton attention kernel not available (requires CUDA + triton)"
    o = attention(q_flat, k_flat, v_flat, **kw)
    return o.view(b, seq_q, heads * dim_head)


class _TritonLTXAttentionWrapper:
    """Wraps ltx_core attention_function for Triton dispatch."""

    def __init__(self, original_fn):
        self._original_fn = original_fn

    def __call__(self, q, k, v, heads, mask=None):
        active, threshold, _scale_factor = _get_ltx_triton_context()
        if active:
            return _ltx_triton_attention(q, k, v, heads, mask, threshold)
        return self._original_fn(q, k, v, heads, mask)


def register_ltx_triton_attention(model: torch.nn.Module) -> None:
    """Patch all ``ltx_core.Attention`` modules for Triton dispatch."""
    from ltx_core.model.transformer.attention import Attention

    for module in model.modules():
        if isinstance(module, Attention):
            warn_rank_0(
                "LTX-2 packages (ltx-core, ltx-pipelines, ltx-trainer) are provided by "
                "Lightricks and are NOT covered by the Apache 2.0 license governing NVIDIA "
                "Model Optimizer. You MUST comply with the LTX Community License Agreement "
                "when installing and using LTX-2 with NVIDIA Model Optimizer. Any derivative "
                "models or fine-tuned weights from LTX-2 (including quantized or distilled "
                "checkpoints) remain subject to the LTX Community License Agreement, not "
                "Apache 2.0. See: https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
                UserWarning,
                stacklevel=2,
            )
            fn = module.attention_function
            if not isinstance(fn, _TritonLTXAttentionWrapper):
                module.attention_function = _TritonLTXAttentionWrapper(fn)
