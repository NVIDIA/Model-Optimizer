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

from modelopt.torch.kernels.triton_fa import attention


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

    # Sparse attention params
    method = getattr(module, "_sparse_method_instance", None)

    # N:M sparse softmax: prefill only (no perf benefit for decode)
    if method is not None and not is_decode and getattr(module, "_apply_sparse_nm", False):
        kw["sparsity_n"] = method.sparsity_n
        kw["sparsity_m"] = method.sparsity_m
        kw["num_sink_tokens"] = method.num_sink_tokens
        kw["dense_window_size"] = method.dense_window_size

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
    "register_triton_attention",
    "triton_attention_forward",
]
