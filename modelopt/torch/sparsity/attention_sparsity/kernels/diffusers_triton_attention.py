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
- **Inference**: Calls ``attention()`` with optional skip-softmax tile skipping,
  N:M sparse softmax, and/or NVFP4 P-matrix quantization.
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
    sparsity_n: int = 0,
    sparsity_m: int = 4,
    num_sink_tokens: int = 0,
    dense_window_size: int = 64,
) -> None:
    """Set thread-local sparse attention config for the next Triton attention call.

    This controls skip-softmax tile skipping and N:M sparsity. It does NOT touch
    ``quantize_p`` — that is managed independently by :func:`set_sage_attention_config`.

    Args:
        threshold: Skip-softmax threshold for inference mode.
        calibration_mode: If True, use the calibration kernel to collect
            multi-threshold sparsity stats instead of skipping tiles.
        threshold_trials: List of thresholds to measure sparsity for
            (only used when calibration_mode=True).
        sparsity_n: Keep top-N of every M attention scores (0 to disable N:M sparsity).
        sparsity_m: Group size for N:M sparsity (4 or 8).
        num_sink_tokens: KV positions before this index kept dense (attention sinks).
        dense_window_size: Tokens near the diagonal kept dense (absolute token count).
    """
    _thread_local.skip_threshold = threshold
    _thread_local.calibration_mode = calibration_mode
    _thread_local.threshold_trials = threshold_trials
    _thread_local.sparsity_n = sparsity_n
    _thread_local.sparsity_m = sparsity_m
    _thread_local.num_sink_tokens = num_sink_tokens
    _thread_local.dense_window_size = dense_window_size
    # Accumulated counters across all attention calls in one forward pass
    _thread_local.calibration_counters = None


def clear_triton_skip_softmax_config() -> None:
    """Clear thread-local sparse attention config.

    Only clears skip-softmax / N:M sparsity params. Does NOT reset ``quantize_p``
    so that :func:`set_sage_attention_config` remains active across attention layers.
    """
    _thread_local.skip_threshold = None
    _thread_local.calibration_mode = False
    _thread_local.threshold_trials = None
    _thread_local.sparsity_n = 0
    _thread_local.sparsity_m = 4
    _thread_local.num_sink_tokens = 0
    _thread_local.dense_window_size = 64
    _thread_local.calibration_counters = None


def set_sage_attention_config(
    quantize_p: bool = True,
    quantize_qkv: bool = False,
) -> None:
    """Set NVFP4 quantization flags for SageAttention.

    Manages ``quantize_p`` and ``quantize_qkv`` independently of sparse-attention
    params so either can be combined with skip-softmax / N:M sparsity without
    clobbering the other's state.

    Args:
        quantize_p: If True, quantize the post-softmax P tile to NVFP4 E2M1
            (per-tile max scaling, SageAttn v1/v2 style). Default True.
        quantize_qkv: If True, apply SageAttn-v3-style per-group microscaling NVFP4
            to Q, K, V (groups of 16 along head_dim) and finer per-group NVFP4 to P.
            Supersedes ``quantize_p`` when set. Default False.
    """
    _thread_local.quantize_p = quantize_p
    _thread_local.quantize_qkv = quantize_qkv


def clear_sage_attention_config() -> None:
    """Clear NVFP4 quantization flags."""
    _thread_local.quantize_p = False
    _thread_local.quantize_qkv = False


def get_calibration_counters() -> "torch.Tensor | None":
    """Return accumulated calibration counters ``[num_thresholds, 2]`` or None."""
    return getattr(_thread_local, "calibration_counters", None)


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

            return o.view(batch, seq_q, num_heads_q, head_dim)

    # --- Inference mode: optional skip-softmax, N:M sparsity, and/or NVFP4 quantization ---
    threshold = getattr(_thread_local, "skip_threshold", None)
    if threshold is not None and threshold > 0.0:
        kw["skip_softmax_threshold"] = threshold

    sparsity_n = getattr(_thread_local, "sparsity_n", 0)
    if sparsity_n > 0:
        kw["sparsity_n"] = sparsity_n
        kw["sparsity_m"] = getattr(_thread_local, "sparsity_m", 4)
        num_sink_tokens = getattr(_thread_local, "num_sink_tokens", 0)
        if num_sink_tokens > 0:
            kw["num_sink_tokens"] = num_sink_tokens
        dense_window_size = getattr(_thread_local, "dense_window_size", 64)
        if dense_window_size > 0:
            kw["dense_window_size"] = dense_window_size

    quantize_qkv = getattr(_thread_local, "quantize_qkv", False)
    if quantize_qkv:
        kw["quantize_qkv"] = True
    elif getattr(_thread_local, "quantize_p", False):
        kw["quantize_p"] = True

    assert attention is not None, "Triton attention kernel not available (requires CUDA + triton)"
    o = attention(q, k, v, **kw)
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
