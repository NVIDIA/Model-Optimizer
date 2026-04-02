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

"""HuggingFace attention backend using the VSA sparse attention kernel.

Registers as ``attn_implementation="modelopt_vsa"`` so HF models dispatch to the
VSA kernel natively.  The ``SparseAttentionModule`` wrapping each attention layer
exposes ``_sparse_method_instance`` (a :class:`VSA` object) which holds the tiling
metadata and calls the fastvideo Triton kernel.

HF provides Q, K, V in ``[batch, heads, seq_len, dim]`` — exactly the format VSA
expects — so no reshape is needed on the way in.  The output is transposed to HF's
expected ``[batch, seq_len, heads, dim]`` on the way out.
"""

import torch
import torch.nn as nn


def vsa_attention_forward(
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

    Retrieves the VSA method instance from the ``SparseAttentionModule`` that
    wraps *module* (via the DynamicModule class-swap) and delegates to
    :meth:`VSA.forward_attention`.

    Args:
        module: The attention module (e.g. ``LlamaAttention``).  After
            ``sparsify()`` its class has been swapped to include
            ``SparseAttentionModule``, so ``module._sparse_method_instance``
            is the :class:`VSA` instance.
        query: ``[batch, num_heads, seq_len, head_dim]``.
        key: ``[batch, num_kv_heads, seq_k, head_dim]``.
        value: ``[batch, num_kv_heads, seq_k, head_dim]``.
        attention_mask: Ignored by VSA (the kernel handles masking internally).
        scaling: Ignored (VSA uses its own scaling via the Triton kernel).
        dropout: Ignored.
        **kwargs: Reserved for future extensions.

    Returns:
        ``(attn_output, None)`` with ``attn_output`` in
        ``[batch, seq_len, num_heads, head_dim]`` (HF convention).
    """
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

    # When VSA is disabled, fall back to standard SDPA.
    is_enabled = isinstance(module, SparseAttentionModule) and module.is_enabled
    if not is_enabled:
        import torch.nn.functional as F

        # Expand KV heads for GQA (num_heads != num_kv_heads)
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_heads != num_kv_heads:
            reps = num_heads // num_kv_heads
            key = key.repeat_interleave(reps, dim=1)
            value = value.repeat_interleave(reps, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=scaling
        )
        return (attn_output.transpose(1, 2), None)

    vsa = module._sparse_method_instance

    output, stats = vsa.forward_attention(query, key, value)

    # Store stats for collection by SparseAttentionModule
    module._last_stats = stats

    # VSA returns [batch, heads, seq_len, dim] and HF expects [batch, seq_len, heads, dim]
    attn_output = output.transpose(1, 2)

    return (attn_output, None)


def register_vsa_attention() -> bool:
    """Register the VSA backend with HF ``ALL_ATTENTION_FUNCTIONS``.

    Called by ``_set_attn_implementation()`` during sparsification.  Must run
    before the model's first forward pass so HF dispatches to VSA.

    Returns:
        ``True`` if registration succeeded, ``False`` if the transformers API
        is not available.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except (ImportError, AttributeError):
        return False

    ALL_ATTENTION_FUNCTIONS.register("modelopt_vsa", vsa_attention_forward)
    return True


__all__ = [
    "register_vsa_attention",
    "vsa_attention_forward",
]
