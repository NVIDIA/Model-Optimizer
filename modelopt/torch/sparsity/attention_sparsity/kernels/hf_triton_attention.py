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

"""Hugging Face attention backend for the Triton unified attention kernel.

Registers the Triton kernel as attn_implementation="modelopt_triton" so HF models
use it natively without patching forward. Both prefill and decode use the unified
Triton kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modelopt.torch.sparsity.attention_sparsity.kernels.triton_unified_attention import (
    context_attention,
    context_attention_fwd,
)


def _attention_mask_supported_for_triton(attention_mask: torch.Tensor) -> bool:
    """Return True if mask shape is supported for packing (2D [batch, seq_len])."""
    return attention_mask.dim() == 2 and attention_mask.shape[0] > 0 and attention_mask.shape[1] > 0


def _packed_token_indices(
    seq_lens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vectorized (batch_idx, token_idx) for packing/unpacking variable-length sequences.

    Assumes valid tokens occupy positions ``0..seq_lens[b]-1`` in each batch
    element (right-padded layout). This matches the HF convention where padding
    tokens are appended after the valid content during prefill.

    Args:
        seq_lens: [batch] number of valid tokens per sequence.
        device: Target device.

    Returns:
        (batch_indices, token_indices) each of shape [total_valid_tokens].
    """
    total = int(seq_lens.sum().item())
    cumsum = torch.zeros(seq_lens.shape[0] + 1, device=device, dtype=torch.long)
    cumsum[1:] = torch.cumsum(seq_lens, dim=0)
    flat_idx = torch.arange(total, device=device, dtype=torch.long)
    batch_indices = torch.bucketize(flat_idx, cumsum[1:], right=True)
    token_indices = flat_idx - cumsum[batch_indices]
    return batch_indices, token_indices


def _derive_seq_lens_and_pack(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Derive b_seq_len and b_start_loc from 2D mask; pack q,k,v to contiguous [total, heads, dim].

    attention_mask: [batch, seq_len], 1 = valid, 0 = pad. Assumes valid tokens are
    at positions 0..n-1 (right-padded layout). The count of valid tokens per row
    determines the packing lengths.
    Returns: (q_packed, k_packed, v_packed, b_start_loc, b_seq_len, max_input_len).
    """
    batch = query.shape[0]
    device = query.device
    # Valid length per batch: number of ones (or non-zero) in the mask per row
    if attention_mask.dtype == torch.bool:
        seq_lens = attention_mask.sum(dim=1).long()
    else:
        seq_lens = (attention_mask != 0).sum(dim=1).long()
    seq_lens = seq_lens.to(device)
    b_start_loc = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(seq_lens, dim=0)
    b_start_loc = b_start_loc[:batch]
    b_seq_len = seq_lens.to(torch.int32)
    max_input_len = int(seq_lens.max().item())

    # Vectorized packing: query [batch, heads, seq, dim] -> [total, heads, dim]
    batch_indices, token_indices = _packed_token_indices(seq_lens, device)
    q_packed = query[batch_indices, :, token_indices, :].contiguous()
    k_packed = key[batch_indices, :, token_indices, :].contiguous()
    v_packed = value[batch_indices, :, token_indices, :].contiguous()
    return q_packed, k_packed, v_packed, b_start_loc, b_seq_len, max_input_len


def _unpack_attn_output(
    o_packed: torch.Tensor,
    batch: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    b_seq_len: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Scatter packed output [total_tokens, num_heads, head_dim] to [batch, seq_len, num_heads, head_dim]."""
    attn_output = torch.zeros(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    total = int(b_seq_len.sum().item())
    if total == 0:
        return attn_output
    batch_indices, token_indices = _packed_token_indices(b_seq_len.long(), device)
    attn_output[batch_indices, token_indices] = o_packed
    return attn_output


def _decode_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Decode attention via context_attention_fwd (one query token per sequence).

    Reshapes HF-format K/V from [batch, kv_heads, seq_k, dim] to flat packed
    [total_kv_tokens, kv_heads, dim] and calls context_attention_fwd with
    is_causal=False so the single query token attends to all K/V positions.

    Args:
        module: The attention module (unused; kept for API compatibility).
        query: [batch, num_heads, 1, head_dim].
        key: [batch, num_kv_heads, seq_k, head_dim].
        value: [batch, num_kv_heads, seq_k, head_dim].
        attention_mask: Optional 2D [batch, seq_k] mask; 1=valid, 0=pad.
        scaling: Softmax scale.

    Returns:
        attn_output: [batch, 1, num_heads, head_dim].
    """
    batch = query.shape[0]
    num_kv_heads = key.shape[1]
    seq_k = key.shape[2]
    head_dim = query.shape[3]
    device = query.device

    # Q: [batch, heads, 1, dim] -> [batch, heads, dim] (flat: 1 token per batch)
    q_flat = query.squeeze(2).contiguous()

    # K/V: [batch, kv_heads, seq_k, dim] -> [batch * seq_k, kv_heads, dim]
    k_flat = key.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim).contiguous()
    v_flat = value.permute(0, 2, 1, 3).reshape(batch * seq_k, num_kv_heads, head_dim).contiguous()

    # Q metadata: each batch element has 1 query token
    b_start_loc_q = torch.arange(batch, device=device, dtype=torch.int32)
    b_seq_len_q = torch.ones(batch, device=device, dtype=torch.int32)

    # K/V metadata: each batch element has seq_k tokens (or fewer if masked)
    if attention_mask is not None and _attention_mask_supported_for_triton(attention_mask):
        if attention_mask.dtype == torch.bool:
            b_seq_len_k = attention_mask.sum(dim=1).to(torch.int32).to(device)
        else:
            b_seq_len_k = (attention_mask != 0).sum(dim=1).to(torch.int32).to(device)
    else:
        b_seq_len_k = torch.full((batch,), seq_k, device=device, dtype=torch.int32)

    b_start_loc_k = torch.arange(batch, device=device, dtype=torch.int32) * seq_k

    o_flat = torch.empty_like(q_flat)
    context_attention_fwd(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        b_start_loc=b_start_loc_q,
        b_seq_len=b_seq_len_q,
        max_input_len=1,
        is_causal=False,
        softmax_scale=scaling,
        b_start_loc_k=b_start_loc_k,
        b_seq_len_k=b_seq_len_k,
        max_input_len_k=int(b_seq_len_k.max().item()),
    )

    # [batch, heads, dim] -> [batch, 1, heads, dim]
    return o_flat.unsqueeze(1)


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

    Uses the unified Triton kernel for both prefill (seq_len > 1) and decode
    (seq_len == 1). Same signature as eager_attention_forward.

    Args:
        module: The attention module (LlamaAttention etc.).
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_kv_heads, seq_k, head_dim].
        value: [batch, num_kv_heads, seq_k, head_dim].
        attention_mask: Optional; kernel handles causal internally.
            2D [batch, seq_k] masks are used to derive per-sequence lengths.
            Unsupported formats raise an error.
        scaling: Softmax scale (e.g. 1/sqrt(head_dim)).
        dropout: Ignored (kernel has no dropout); use 0 for eval.
        **kwargs: May contain apply_sparse24, skip_diagonal_blocks for 2:4 sparse attention.

    Returns:
        (attn_output, None) with attn_output [batch, seq_len, num_heads, head_dim].
    """
    batch, num_heads, seq_len, head_dim = query.shape
    seq_k = key.shape[2]
    is_cross_attention = seq_len != seq_k

    # Decode: one query token per sequence, full context in K/V
    if seq_len <= 1:
        attn_output = _decode_attention(module, query, key, value, attention_mask, scaling)
        return (attn_output, None)

    device = query.device
    num_kv_heads = key.shape[1]
    is_causal = not is_cross_attention
    apply_sparse24 = kwargs.get("apply_sparse24", getattr(module, "_apply_sparse24", False))
    skip_diagonal_blocks = kwargs.get(
        "skip_diagonal_blocks", getattr(module, "_skip_diagonal_blocks", True)
    )

    needs_grad = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )

    use_packed = attention_mask is not None and _attention_mask_supported_for_triton(attention_mask)
    if use_packed:
        q_packed, k_packed, v_packed, b_start_loc, b_seq_len, max_input_len = (
            _derive_seq_lens_and_pack(query, key, value, attention_mask)
        )
        fwd_kwargs = {
            "b_start_loc": b_start_loc,
            "b_seq_len": b_seq_len,
            "max_input_len": max_input_len,
            "is_causal": is_causal,
            "softmax_scale": scaling,
            "apply_sparse24": apply_sparse24,
            "skip_diagonal_blocks": skip_diagonal_blocks,
        }
        if needs_grad:
            o_packed = context_attention(q_packed, k_packed, v_packed, **fwd_kwargs)
        else:
            o_packed = torch.empty_like(q_packed)
            context_attention_fwd(q_packed, k_packed, v_packed, o_packed, **fwd_kwargs)
        attn_output = _unpack_attn_output(
            o_packed,
            batch,
            num_heads,
            head_dim,
            seq_len,
            b_seq_len,
            query.dtype,
            device,
        )
        return (attn_output, None)
    if attention_mask is not None:
        raise ValueError(
            f"Unsupported attention_mask format for modelopt_triton: "
            f"dim={attention_mask.dim()}, shape={attention_mask.shape}. "
            f"Only 2D [batch, seq_len] masks are supported."
        )

    q = query.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim).contiguous()
    b_start_loc_q = torch.arange(batch, device=device, dtype=torch.int32) * seq_len
    b_seq_len_q = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    if is_cross_attention:
        b_start_loc_k = torch.arange(batch, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((batch,), seq_k, device=device, dtype=torch.int32)
    else:
        b_start_loc_k = None
        b_seq_len_k = None

    fwd_kwargs = {
        "b_start_loc": b_start_loc_q,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_len,
        "is_causal": is_causal,
        "softmax_scale": scaling,
        "apply_sparse24": apply_sparse24,
        "skip_diagonal_blocks": skip_diagonal_blocks,
        "b_start_loc_k": b_start_loc_k,
        "b_seq_len_k": b_seq_len_k,
        "max_input_len_k": seq_k if is_cross_attention else None,
    }
    if needs_grad:
        o = context_attention(q, k, v, **fwd_kwargs)
    else:
        o = torch.empty_like(q)
        context_attention_fwd(q, k, v, o, **fwd_kwargs)
    attn_output = o.view(batch, seq_len, num_heads, head_dim)
    return (attn_output, None)


def register_triton_attention() -> bool:
    """Register the Triton backend with HF AttentionInterface.

    Call after importing this module so that attn_implementation="modelopt_triton"
    is available when loading models.

    Returns:
        True if registration succeeded.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("modelopt_triton", triton_attention_forward)
        return True
    except Exception:
        return False


def set_sparse24(
    model: nn.Module,
    apply_sparse24: bool = True,
    skip_diagonal_blocks: bool = True,
) -> None:
    """Set 2:4 sparse attention on all attention modules in the model.

    Prefer using ``mtsa.sparsify(model, SPARSE24_TRITON)`` from
    ``modelopt.torch.sparsity.attention_sparsity`` for config-driven setup,
    pattern-based layer selection, and consistency with other sparse methods.
    This helper remains for backward compatibility and one-off scripting.

    The Triton backend reads ``getattr(module, '_apply_sparse24', False)`` and
    ``getattr(module, '_skip_diagonal_blocks', True)`` when kwargs don't provide them.

    Limitations:
        - **Prefill-only sparsity:** 2:4 sparsity is applied during prefill only;
          decode uses the unified kernel without sparsity.
        - **Fixed 50% sparsity:** 2:4 keeps top 2 of every 4 attention scores;
          no threshold tuning or calibration.
        - **Mutually exclusive with flash_skip_softmax:** sparse24 requires
          ``attn_implementation="modelopt_triton"``; flash_skip_softmax requires
          ``attn_implementation="eager"``. They cannot be combined in one model.

    Args:
        model: Hugging Face model (e.g. LlamaForCausalLM).
        apply_sparse24: Whether to apply 2:4 sparsity to attention scores.
        skip_diagonal_blocks: If True, keep diagonal tiles dense (local attention).
    """
    for _, module in model.named_modules():
        # Match only actual attention modules (have o_proj + head_dim), not their children
        # like q_proj, k_proj, v_proj, rotary_emb, etc.
        if hasattr(module, "o_proj") and hasattr(module, "head_dim"):
            setattr(module, "_apply_sparse24", apply_sparse24)
            setattr(module, "_skip_diagonal_blocks", skip_diagonal_blocks)


__all__ = [
    "register_triton_attention",
    "set_sparse24",
    "triton_attention_forward",
]
