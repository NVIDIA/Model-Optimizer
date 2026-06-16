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

import torch
import torch.nn as nn

from modelopt.torch.kernels.common.attention.triton_fa import attention

# Skip-softmax calibration config and counters live on the module's
# ``_sparse_method_instance`` (HF passes the owning module to
# ``triton_attention_forward``), so no separate thread-local state is needed.


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


def _check_mask_supported(attention_mask: torch.Tensor | None, seq_q: int) -> None:
    """Reject attention masks this wrapper would silently misread.

    The wrapper only derives right-padded per-sequence lengths from 2D
    ``[batch, q_len]`` masks; anything else either loses padding info (4D
    masks) or corrupts the varlen metadata (FA2-style ``[batch, kv_len]``
    masks during cached decode).
    """

    def _unsupported(reason):
        return NotImplementedError(
            f"The ModelOpt Triton attention kernel does not support {reason}. "
            "Use unpadded (or uniform-length) right-padded inputs."
        )

    if attention_mask is None:
        return
    if attention_mask.dim() == 2:
        if attention_mask.shape[1] != seq_q:
            # FA2-style [batch, kv_len] mask during cached decode: the wrapper
            # would misread KV lengths as query lengths (out-of-bounds access).
            raise _unsupported("padded batches during cached decode")
        mask_bool = attention_mask.to(torch.bool)
        if not mask_bool[:, 0].all():
            raise _unsupported("left-padded inputs")
        # ``_seq_lens_from_mask`` derives lengths via ``sum(dim=1)``, which is only
        # correct when each row is a contiguous run of valid tokens followed by
        # padding. A hole (e.g. ``[1, 0, 1]``) would sum to the right count but
        # place the valid tokens at the wrong positions, so reject non-right-padded
        # masks (any valid token after a pad == row not monotonically non-increasing).
        if not (mask_bool[:, :-1].int() >= mask_bool[:, 1:].int()).all():
            raise _unsupported("non-contiguously padded inputs")
        return
    # 4D [batch, 1, q, kv] masks are ignored by the wrapper, which is safe only
    # when they encode pure causal structure (the kernel masks causally itself).
    # In a causal mask the newest query row sees every position; any masked
    # entry there means padding, windowing, or a non-causal/bias pattern.
    last_row = attention_mask[..., -1, :]
    hidden = ~last_row if attention_mask.dtype == torch.bool else last_row != 0
    if hidden.any():
        raise _unsupported("masks carrying padding or non-causal structure")


def validate_triton_attention_envelope(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> None:
    """Raise ``NotImplementedError`` for inputs outside this wrapper/kernel envelope.

    These limits do not come from the quantization or sparsity features layered
    on top — they document what the ``triton_fa`` kernel (causal or single-token
    decode only; no sliding window, attention sinks, logit softcapping, or
    dropout; head_dim >= 16) and this wrapper's varlen-metadata derivation
    (right-padded 2D masks only; no multi-token forwards over a longer KV cache)
    support. Callers that route arbitrary HF models onto the kernel dynamically
    (e.g. the quantization plugin's p_bmm_quantizer dispatch) should call this
    before dispatching, so unsupported models fail loudly instead of silently
    computing wrong attention. The sparse-attention path predates these checks
    and does not yet enforce them.
    """
    # Mistral-style models pass sliding_window as an interface kwarg instead of
    # setting it on the attention module, so check both.
    if getattr(module, "sliding_window", None) or kwargs.get("sliding_window"):
        raise NotImplementedError(
            "The ModelOpt Triton attention kernel does not support sliding-window attention layers."
        )
    # Semantic attention arguments the kernel does not implement: dropping them
    # would change the attention math.
    for name, reason in (("s_aux", "attention sinks"), ("softcap", "logit softcapping")):
        if kwargs.get(name) is not None:
            raise NotImplementedError(
                f"The ModelOpt Triton attention kernel does not support {reason} ('{name}')."
            )
    if kwargs.get("is_causal") is False or getattr(module, "is_causal", True) is False:
        raise NotImplementedError(
            "The ModelOpt Triton attention kernel does not support non-causal attention."
        )
    if kwargs.get("dropout"):
        raise NotImplementedError(
            "The ModelOpt Triton attention kernel does not support attention dropout; "
            "set attention_dropout=0 for training."
        )
    if query.shape[-1] < 16:
        raise NotImplementedError(
            f"The ModelOpt Triton attention kernel requires head_dim >= 16, got {query.shape[-1]}."
        )
    seq_q, seq_k = query.shape[2], key.shape[2]
    if seq_q > 1 and seq_k != seq_q:
        # The wrapper only passes K-side varlen metadata for single-token decode;
        # multi-token forwards over a longer KV cache would mis-index K/V.
        raise NotImplementedError(
            "The ModelOpt Triton attention kernel does not support multi-token "
            "forwards over a longer KV cache (chunked prefill or "
            "assisted/speculative decoding)."
        )
    _check_mask_supported(attention_mask, seq_q)


def triton_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    p_qdq: str | None = None,
    p_qdq_scale: float | None = None,
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
        p_qdq: Optional softmax fake quant-dequant mode ("fp8" or
            "nvfp4") forwarded to the kernel. Not passed by HF dispatch;
            used by direct callers such as the quantization plugin.
        p_qdq_scale: Optional per-tensor quantization scale for the
            softmax qdq; None uses the kernel default of 1.0 (an effective
            amax of 448 for FP8 / 6 * 448 for NVFP4).
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

    # Sparse-attention method instance. It carries the inference threshold and,
    # during calibration, both the calibration config and the accumulated
    # tile-skip counters. Available here because HF passes the owning module.
    method = getattr(module, "_sparse_method_instance", None)

    # Calibration mode: run the calibration kernel, which computes full attention
    # while counting, per candidate threshold, how many KV tiles would be skipped.
    # The sparse-attention kwargs below are intentionally not added in this branch.
    if method is not None and getattr(method, "_calibration_mode", False):
        trials = getattr(method, "_threshold_trials", None)
        # Deferred: the package __init__ imports this module, so importing
        # attention_calibrate at module top would be circular.
        from modelopt.torch.kernels.sparsity.attention.calibrate import attention_calibrate

        if trials and attention_calibrate is not None:
            o, counters = attention_calibrate(q, k, v, **kw, threshold_trials=trials)

            # Accumulate counters across all attention calls in this forward pass.
            # The method instance is per-module so the accumulator stays on one
            # device, but guard the add against a device mismatch just in case.
            prev = getattr(method, "_hf_calibration_counters", None)
            method._hf_calibration_counters = (
                counters if prev is None else prev + counters.to(prev.device)
            )
            method._hf_calibration_seq_k = seq_k
            method._hf_calibration_is_decode = is_decode

            return (o.view(batch, seq_len, num_heads, head_dim), None)

    # N:M sparse softmax: prefill only (no perf benefit for decode)
    if method is not None and not is_decode and getattr(module, "_apply_sparse_nm", False):
        kw["sparsity_n"] = method.sparsity_n
        kw["sparsity_m"] = method.sparsity_m
        kw["dense_sink_tokens"] = method.dense_sink_tokens
        kw["dense_recent_tokens"] = method.dense_recent_tokens

    # Skip-softmax: applies to both prefill and decode. Prefer the method's
    # per-phase calibrated dynamic threshold (scale_factor / seq_k); fall back
    # to the static threshold when uncalibrated.
    if method is not None and getattr(module, "_apply_skip_softmax", False):
        threshold = method.get_inference_threshold(seq_len, seq_k)
        if threshold:
            kw["skip_softmax_threshold"] = threshold

    if p_qdq is not None:
        kw["p_qdq"] = p_qdq
        if p_qdq_scale is not None:
            kw["p_qdq_scale"] = p_qdq_scale

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
    "register_triton_attention",
    "triton_attention_forward",
    "validate_triton_attention_envelope",
]
