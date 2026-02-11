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

"""Hugging Face attention backend for the Triton prefill kernel.

Registers the Triton kernel as attn_implementation="modelopt_triton" so HF models
use it natively without patching forward. Decode (seq_len==1) falls back to eager.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modelopt.torch.sparsity.attention_sparsity.kernels.triton_prefill_attention import (
    context_attention_fwd,
)


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

    Uses context_attention_fwd for prefill (seq_len > 1); decode (seq_len == 1)
    falls back to eager. Same signature as eager_attention_forward.

    Args:
        module: The attention module (LlamaAttention etc.).
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_kv_heads, seq_k, head_dim].
        value: [batch, num_kv_heads, seq_k, head_dim].
        attention_mask: Optional; kernel handles causal internally.
        scaling: Softmax scale (e.g. 1/sqrt(head_dim)).
        dropout: Ignored (kernel has no dropout); use 0 for eval.
        **kwargs: May contain skip_threshold for skip-softmax.

    Returns:
        (attn_output, None) with attn_output [batch, num_heads, seq_len, head_dim].
    """
    batch, num_heads, seq_len, head_dim = query.shape
    if seq_len <= 1:
        from transformers.models.llama.modeling_llama import eager_attention_forward

        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )

    skip_threshold = kwargs.get("skip_threshold") or getattr(
        module, "_skip_threshold", None
    )

    q = query.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim).contiguous()
    num_kv_heads = key.shape[1]
    k = key.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim).contiguous()

    device = query.device
    b_start_loc = torch.arange(
        batch, device=device, dtype=torch.int32
    ) * seq_len
    b_seq_len = torch.full(
        (batch,), seq_len, device=device, dtype=torch.int32
    )

    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    o = torch.empty_like(q_float)

    context_attention_fwd(
        q_float,
        k_float,
        v_float,
        o,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=seq_len,
        is_causal=True,
        softmax_scale=scaling,
        skip_threshold=skip_threshold,
    )

    attn_output = o.to(query.dtype).view(batch, seq_len, num_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
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


def set_skip_threshold(model: nn.Module, threshold: float | None) -> None:
    """Set skip_threshold on all attention modules in the model.

    The Triton backend reads getattr(module, '_skip_threshold', None) when
    kwargs don't contain skip_threshold.

    Args:
        model: Hugging Face model (e.g. LlamaForCausalLM).
        threshold: Value in (0, 1) for skip-softmax; None to disable.
    """
    for name, module in model.named_modules():
        if "attention" in name.lower() and hasattr(module, "head_dim"):
            setattr(module, "_skip_threshold", threshold)


__all__ = [
    "register_triton_attention",
    "set_skip_threshold",
    "triton_attention_forward",
]
