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

"""Triton flash attention backend for diffusers models.

Registers a ``modelopt_triton`` backend in diffusers' ``_AttentionBackendRegistry``
that converts the diffusers [B, S, H, D] layout to the Triton FA kernel's varlen
[total_tokens, H, D] format.

Two modes:
- **Inference**: Calls ``attention()`` with skip-softmax tile skipping.
- **Calibration**: Calls ``attention_calibrate()`` to collect multi-threshold
  sparsity statistics without skipping any tiles.
"""

import inspect
import math
import threading

import torch
from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    attention_backend,
)

from modelopt.torch.kernels import attention, attention_calibrate

_BACKEND_NAME = "modelopt_triton"
_BACKEND_REGISTERED = False

# Thread-local storage for per-forward skip-softmax configuration.
_thread_local = threading.local()


def set_triton_skip_softmax_config(
    threshold: float | None = None,
    calibration_mode: bool = False,
    threshold_trials: list[float] | None = None,
    scale_factor: float | None = None,
    raw_threshold: float | None = None,
    measure_sparsity: bool = False,
) -> None:
    """Set thread-local skip-softmax config for the next Triton attention call.

    Args:
        threshold: Skip-softmax threshold for inference mode (static).
        calibration_mode: If True, use the calibration kernel to collect
            multi-threshold sparsity stats instead of skipping tiles.
        threshold_trials: List of thresholds to measure sparsity for
            (only used when calibration_mode=True).
        scale_factor: Calibrated scale factor for dynamic threshold computation.
            When set, the actual threshold is computed as ``scale_factor / seq_k``
            at attention call time, adapting to the actual sequence length.
        raw_threshold: Raw ``skip_threshold_log2`` value passed directly to the
            kernel without conversion. Takes precedence over other thresholds.
        measure_sparsity: If True, count total and skipped tiles during
            inference via atomic counters in the forward kernel.
    """
    _thread_local.skip_threshold = threshold
    _thread_local.calibration_mode = calibration_mode
    _thread_local.threshold_trials = threshold_trials
    _thread_local.scale_factor = scale_factor
    _thread_local.raw_threshold = raw_threshold
    _thread_local.measure_sparsity = measure_sparsity
    # Accumulated counters across all attention calls in one forward pass
    _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None
    # Accumulated runtime sparsity counters (total_tiles, skipped_tiles)
    _thread_local.sparsity_total = 0
    _thread_local.sparsity_skipped = 0


def clear_triton_skip_softmax_config() -> None:
    """Clear thread-local skip-softmax config."""
    _thread_local.skip_threshold = None
    _thread_local.calibration_mode = False
    _thread_local.threshold_trials = None
    _thread_local.scale_factor = None
    _thread_local.raw_threshold = None
    _thread_local.measure_sparsity = False
    _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None
    _thread_local.sparsity_total = 0
    _thread_local.sparsity_skipped = 0


def get_calibration_counters() -> "torch.Tensor | None":
    """Return accumulated calibration counters ``[num_thresholds, 2]`` or None."""
    return getattr(_thread_local, "calibration_counters", None)


def get_calibration_seq_k() -> int | None:
    """Return KV sequence length observed during calibration, or None."""
    return getattr(_thread_local, "calibration_seq_k", None)


def get_sparsity_counters() -> tuple[int, int]:
    """Return accumulated runtime sparsity counters ``(total_tiles, skipped_tiles)``."""
    return (
        getattr(_thread_local, "sparsity_total", 0),
        getattr(_thread_local, "sparsity_skipped", 0),
    )


# ---------------------------------------------------------------------------
# Triton attention implementation for diffusers layout
# ---------------------------------------------------------------------------


def _diffusers_triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Compute attention via Triton FA kernel on diffusers layout ``[B, S, H, D]``."""
    batch, seq_q, num_heads_q, head_dim = query.shape
    seq_k = key.shape[1]
    device = query.device

    # Reshape from diffusers [B, S, H, D] -> flat [B*S, H, D]
    q = query.reshape(batch * seq_q, num_heads_q, head_dim).contiguous()
    k = key.reshape(batch * seq_k, key.shape[2], head_dim).contiguous()
    v = value.reshape(batch * seq_k, value.shape[2], head_dim).contiguous()

    # Build varlen metadata
    b_start_loc_q = torch.arange(batch, device=device, dtype=torch.int32) * seq_q
    b_seq_len_q = torch.full((batch,), seq_q, device=device, dtype=torch.int32)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    kw: dict = {
        "b_start_loc": b_start_loc_q,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_q,
        "is_causal": is_causal,
        "softmax_scale": scale,
    }

    if seq_q != seq_k:
        b_start_loc_k = torch.arange(batch, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((batch,), seq_k, device=device, dtype=torch.int32)
        kw["b_start_loc_k"] = b_start_loc_k
        kw["b_seq_len_k"] = b_seq_len_k
        kw["max_input_len_k"] = seq_k

    # --- Calibration mode: collect multi-threshold stats ---
    calib_mode = getattr(_thread_local, "calibration_mode", False)
    if calib_mode:
        trials = getattr(_thread_local, "threshold_trials", None)
        if trials and attention_calibrate is not None:
            o, counters = attention_calibrate(q, k, v, **kw, threshold_trials=trials)

            # Accumulate counters across all attention calls in this forward pass
            prev = getattr(_thread_local, "calibration_counters", None)
            if prev is None:
                _thread_local.calibration_counters = counters
            else:
                _thread_local.calibration_counters = prev + counters

            # Store actual KV sequence length for calibration stats
            _thread_local.calibration_seq_k = seq_k

            return o.view(batch, seq_q, num_heads_q, head_dim)

    # --- Inference mode: skip-softmax with raw, dynamic, or static threshold ---
    raw_thresh = getattr(_thread_local, "raw_threshold", None)
    if raw_thresh is not None:
        # Raw threshold: passed directly to kernel as skip_threshold_log2
        kw["skip_softmax_raw_threshold"] = raw_thresh
    else:
        scale_factor = getattr(_thread_local, "scale_factor", None)
        if scale_factor is not None and scale_factor > 0.0:
            # Dynamic threshold: adapt to actual sequence length
            kw["skip_softmax_threshold"] = scale_factor / seq_k
        else:
            threshold = getattr(_thread_local, "skip_threshold", None)
            if threshold is not None and threshold > 0.0:
                kw["skip_softmax_threshold"] = threshold

    assert attention is not None, "Triton attention kernel not available (requires CUDA + triton)"
    do_measure = getattr(_thread_local, "measure_sparsity", False)
    if do_measure:
        kw["measure_sparsity"] = True
    o = attention(q, k, v, **kw)

    # Accumulate runtime sparsity counters from the kernel output
    if do_measure and hasattr(o, "_sparsity_total"):
        prev_total = getattr(_thread_local, "sparsity_total", 0)
        prev_skipped = getattr(_thread_local, "sparsity_skipped", 0)
        _thread_local.sparsity_total = prev_total + o._sparsity_total
        _thread_local.sparsity_skipped = prev_skipped + o._sparsity_skipped

    return o.view(batch, seq_q, num_heads_q, head_dim)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_diffusers_triton_attention() -> None:
    """Register ``modelopt_triton`` backend in diffusers.

    Safe to call multiple times; registration happens only once.
    """
    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return

    new_member = str.__new__(AttentionBackendName, _BACKEND_NAME)
    new_member._name_ = "MODELOPT_TRITON"
    new_member._value_ = _BACKEND_NAME
    AttentionBackendName._member_map_["MODELOPT_TRITON"] = new_member
    AttentionBackendName._value2member_map_[_BACKEND_NAME] = new_member

    _AttentionBackendRegistry._backends[new_member] = _diffusers_triton_attention
    _AttentionBackendRegistry._constraints[new_member] = []
    _AttentionBackendRegistry._supported_arg_names[new_member] = set(
        inspect.signature(_diffusers_triton_attention).parameters.keys()
    )

    _BACKEND_REGISTERED = True


def get_triton_attention_backend():
    """Return a context manager that activates the modelopt_triton backend."""
    if not _BACKEND_REGISTERED:
        raise RuntimeError(
            "modelopt_triton backend not registered. "
            "Call register_diffusers_triton_attention() first."
        )
    return attention_backend(_BACKEND_NAME)
