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

"""HuggingFace attention backend using the Triton flash attention kernel.

Registers as attn_implementation="modelopt_triton" so HF models dispatch to the
Triton kernel natively. Handles format conversion between HF's [batch, heads, seq, dim]
and the kernel's flat packed [total_tokens, heads, dim] varlen format.
"""

from __future__ import annotations

import threading

import torch
import torch.nn as nn

from modelopt.torch.kernels.common.attention.triton_fa import attention

# ---------------------------------------------------------------------------
# Thread-local skip-softmax calibration config for the HF (modelopt_triton) backend
# ---------------------------------------------------------------------------
# Mirrors the diffusers/LTX backends: during calibration the Triton calibration
# kernel measures multi-threshold tile-skip statistics without skipping any tiles.
# Inference-time config (skip threshold / scale factor) is still read from the
# module/method attributes in ``triton_attention_forward`` — only calibration
# state lives here.
_thread_local = threading.local()


def set_hf_triton_skip_softmax_config(
    threshold: float | None = None,
    calibration_mode: bool = False,
    threshold_trials: list[float] | None = None,
    scale_factor: float | None = None,
    measure_sparsity: bool = False,
) -> None:
    """Set thread-local skip-softmax calibration config for the next forward.

    Accepts the same keyword arguments as the diffusers/LTX backends so the
    shared :class:`TritonSkipSoftmaxMethod` can configure all backends uniformly.
    Only the calibration fields are consumed by the HF backend; the inference
    fields (``threshold``/``scale_factor``/``measure_sparsity``) are accepted for
    signature compatibility but ignored here, since the HF inference path reads
    its threshold from the module/method attributes.

    Args:
        threshold: Ignored by the HF backend (inference threshold comes from the module).
        calibration_mode: If True, route prefill attention through the calibration kernel.
        threshold_trials: Thresholds to measure sparsity for (used when calibration_mode=True).
        scale_factor: Ignored by the HF backend.
        measure_sparsity: Ignored by the HF backend.
    """
    _thread_local.calibration_mode = calibration_mode
    _thread_local.threshold_trials = threshold_trials
    # Counters accumulated across all attention calls in one forward pass.
    _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None


def clear_hf_triton_skip_softmax_config() -> None:
    """Clear thread-local skip-softmax calibration config."""
    _thread_local.calibration_mode = False
    _thread_local.threshold_trials = None
    _thread_local.calibration_counters = None
    _thread_local.calibration_seq_k = None


def get_calibration_counters() -> torch.Tensor | None:
    """Return accumulated calibration counters ``[num_thresholds, 2]`` or None."""
    return getattr(_thread_local, "calibration_counters", None)


def get_calibration_seq_k() -> int | None:
    """Return KV sequence length observed during calibration, or None."""
    return getattr(_thread_local, "calibration_seq_k", None)


def _seq_lens_from_mask(
    attention_mask: torch.Tensor | None,
    fallback: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, bool]:
    """Derive per-sequence lengths from attention mask.

    Returns (b_seq_len, has_padding). If the mask is not a usable 2D format,
    returns (None, False).
    """
    if attention_mask is not None and attention_mask.dim() == 2:
        mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
        b_seq_len = mask.sum(dim=1).to(torch.int32).to(device)
        has_padding = bool((b_seq_len != fallback).any())
        return b_seq_len, has_padding
    return None, False


def triton_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention forward compatible with HF AttentionInterface.

    Converts HF tensors to varlen format, calls the Triton kernel, converts back.
    Handles both prefill (seq_len > 1) and decode (seq_len == 1).

    Args:
        module: The attention module (LlamaAttention etc.).
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_kv_heads, seq_k, head_dim].
        value: [batch, num_kv_heads, seq_k, head_dim].
        attention_mask: Optional; kernel handles causal masking internally.
            2D [batch, seq_len] masks are used to derive per-sequence lengths.
            Other formats (e.g. 4D causal masks) are ignored.
        scaling: Softmax scale (e.g. 1/sqrt(head_dim)).
        dropout: Ignored (kernel has no dropout); use 0 for eval.
        **kwargs: Reserved for future extensions.

    Returns:
        (attn_output, None) with attn_output [batch, seq_len, num_heads, head_dim].
    """
    batch, num_heads, seq_len, head_dim = query.shape
    seq_k = key.shape[2]
    num_kv_heads = key.shape[1]
    device = query.device
    is_decode = seq_len <= 1

    # Reshape from HF [batch, heads, seq, dim] -> flat [batch*seq, heads, dim]
    q = query.permute(0, 2, 1, 3).reshape(batch * seq_len, num_heads, head_dim).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim).contiguous()

    # Build varlen metadata
    b_seq_len_q, has_padding = _seq_lens_from_mask(attention_mask, seq_len, device)
    if b_seq_len_q is None:
        b_seq_len_q = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    kw = {
        "b_start_loc": torch.arange(batch, device=device, dtype=torch.int32) * seq_len,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_len,
        "is_causal": not is_decode,
        "softmax_scale": scaling,
    }
    # Decode: Q has 1 token, K/V have seq_k tokens (KV cache, no padding)
    if is_decode:
        kw["b_start_loc_k"] = torch.arange(batch, device=device, dtype=torch.int32) * seq_k
        kw["b_seq_len_k"] = torch.full((batch,), seq_k, device=device, dtype=torch.int32)
        kw["max_input_len_k"] = seq_k

    # --- Calibration mode: collect multi-threshold tile-skip stats (prefill only) ---
    # Run the calibration kernel, which computes full (non-skipped) attention while
    # counting, per candidate threshold, how many KV tiles would be skipped. ``kw`` at
    # this point holds only the base attention args that ``attention_calibrate`` accepts;
    # the sparse-attention kwargs below are intentionally not added in this branch.
    calib_mode = getattr(_thread_local, "calibration_mode", False)
    if calib_mode and not is_decode:
        trials = getattr(_thread_local, "threshold_trials", None)
        from modelopt.torch.kernels.common.attention import attention_calibrate

        if trials and attention_calibrate is not None:
            o, counters = attention_calibrate(q, k, v, **kw, threshold_trials=trials)

            # Accumulate counters across all attention calls in this forward pass.
            prev = getattr(_thread_local, "calibration_counters", None)
            _thread_local.calibration_counters = counters if prev is None else prev + counters
            _thread_local.calibration_seq_k = seq_k

            return (o.view(batch, seq_len, num_heads, head_dim), None)

    # Sparse attention params
    method = getattr(module, "_sparse_method_instance", None)

    # N:M sparse softmax: prefill only (no perf benefit for decode)
    if method is not None and not is_decode and getattr(module, "_apply_sparse_nm", False):
        kw["sparsity_n"] = method.sparsity_n
        kw["sparsity_m"] = method.sparsity_m
        kw["dense_sink_tokens"] = method.dense_sink_tokens
        kw["dense_recent_tokens"] = method.dense_recent_tokens

    # Skip-softmax: applies to both prefill and decode
    if method is not None and getattr(module, "_apply_skip_softmax", False):
        if method.skip_softmax_threshold:
            kw["skip_softmax_threshold"] = method.skip_softmax_threshold

    o = attention(q, k, v, **kw)

    attn_output = o.view(batch, seq_len, num_heads, head_dim)

    # Zero out padding positions (kernel produces NaN for all-padding rows due to 0/0).
    # Assumes right-padding (valid tokens at positions 0..n-1), which is the HF
    # convention during prefill. Left-padded inputs are not supported.
    if has_padding:
        pad_mask = torch.arange(seq_len, device=device)[None, :] >= b_seq_len_q[:, None]
        attn_output = attn_output.masked_fill(pad_mask[:, :, None, None], 0.0)

    return (attn_output, None)


def register_triton_attention() -> bool:
    """Register the Triton backend with HF AttentionInterface.

    Called by _set_attn_implementation() during sparsification. Must run before
    the model's first forward pass so HF dispatches to the Triton kernel.

    Returns:
        True if registration succeeded, False if transformers API not available.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except (ImportError, AttributeError):
        return False

    ALL_ATTENTION_FUNCTIONS.register("modelopt_triton", triton_attention_forward)
    return True


__all__ = [
    "clear_hf_triton_skip_softmax_config",
    "get_calibration_counters",
    "get_calibration_seq_k",
    "register_triton_attention",
    "set_hf_triton_skip_softmax_config",
    "triton_attention_forward",
]
