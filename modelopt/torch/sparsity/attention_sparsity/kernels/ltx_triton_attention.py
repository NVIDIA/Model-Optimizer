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

"""LTX-2 attention backend for the Triton unified attention kernel with 2:4 sparsity.

Patches LTX-2 ``Attention`` modules so that ``self.attention_function`` routes to
the Triton sparse24 kernel when the thread-local context flags are set by
``SparseAttentionModule``.

The thread-local context (``apply_sparse24`` / ``skip_diagonal_blocks``) is shared
with the diffusers backend so the same ``Sparse24Triton.get_sparse_context`` context
manager drives both code paths.
"""

from __future__ import annotations

import torch

from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
    get_sparse24_context,
)
from modelopt.torch.sparsity.attention_sparsity.kernels.triton_unified_attention import (
    context_attention_fwd,
)

# ---------------------------------------------------------------------------
# Triton sparse24 attention adapted for LTX-2 tensor layout
# ---------------------------------------------------------------------------


def _ltx_sparse24_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    skip_diagonal_blocks: bool = True,
) -> torch.Tensor:
    """Run Triton sparse24 attention on LTX-2 layout tensors ``[B, T, H*D]``.

    Reshapes to the packed ``[total, H, D]`` format expected by
    ``context_attention_fwd``, runs the kernel with ``apply_sparse24=True``,
    and reshapes back.
    """
    batch, seq_len, hd = q.shape
    dim_head = hd // heads
    device = q.device

    # LTX-2 uses the same number of heads for Q, K, V (no GQA)
    q_packed = q.reshape(batch * seq_len, heads, dim_head).contiguous()
    k_packed = k.reshape(batch * seq_len, heads, dim_head).contiguous()
    v_packed = v.reshape(batch * seq_len, heads, dim_head).contiguous()
    o = torch.empty_like(q_packed)

    b_start_loc = torch.arange(batch, device=device, dtype=torch.int32) * seq_len
    b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    context_attention_fwd(
        q_packed,
        k_packed,
        v_packed,
        o,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=None,
        apply_sparse24=True,
        skip_diagonal_blocks=skip_diagonal_blocks,
    )

    return o.view(batch, seq_len, heads * dim_head)


# ---------------------------------------------------------------------------
# Wrapper that intercepts LTX-2 attention_function calls
# ---------------------------------------------------------------------------


class _Sparse24LTXAttentionWrapper:
    """Wraps an LTX-2 ``AttentionCallable``; routes to Triton when sparse context is active.

    When the thread-local ``apply_sparse24`` flag is set (by
    ``Sparse24Triton.get_sparse_context``) and the query/key sequence lengths
    match (self-attention), the call is forwarded to the Triton sparse24 kernel.
    Otherwise the original attention function is called unchanged.
    """

    def __init__(self, original_fn):
        self.original_fn = original_fn

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        apply_sparse24, skip_diagonal_blocks = get_sparse24_context()

        seq_q = q.shape[1]
        seq_k = k.shape[1]
        can_use_triton = apply_sparse24 and seq_q == seq_k

        if can_use_triton:
            return _ltx_sparse24_attention(q, k, v, heads, skip_diagonal_blocks)

        return self.original_fn(q, k, v, heads, mask)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_ltx_triton_attention(model: torch.nn.Module) -> bool:
    """Patch LTX-2 ``Attention`` modules to route to the sparse24 Triton kernel.

    Iterates over all modules in *model* and, for each ``ltx_core`` ``Attention``
    instance, replaces ``attention_function`` with a wrapper that checks the
    thread-local sparse24 context.

    Args:
        model: The model (or sub-model) whose attention modules should be patched.

    Returns:
        True if at least one module was patched.
    """
    try:
        from ltx_core.model.transformer.attention import Attention as LTXAttention
    except ImportError:
        return False

    patched = 0
    for module in model.modules():
        if isinstance(module, LTXAttention):
            # Don't double-wrap
            if not isinstance(module.attention_function, _Sparse24LTXAttentionWrapper):
                module.attention_function = _Sparse24LTXAttentionWrapper(module.attention_function)
                patched += 1

    return patched > 0


__all__ = [
    "register_ltx_triton_attention",
]
