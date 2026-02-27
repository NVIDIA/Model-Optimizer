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

"""Diffusers attention backend for the Triton unified attention kernel with 2:4 sparsity.

Registers into diffusers' ``_AttentionBackendRegistry`` so that diffusers models
(Wan, LTX, Flux, etc.) can use the Triton sparse24 kernel via the standard
``dispatch_attention_fn`` path.

A thread-local context carries ``apply_sparse24`` / ``skip_diagonal_blocks`` flags
from the ``SparseAttentionModule`` wrapper down into the dispatch function, avoiding
any per-model processor changes.
"""

from __future__ import annotations

import inspect
import threading

import torch

from modelopt.torch.sparsity.attention_sparsity.kernels.triton_unified_attention import (
    context_attention_fwd,
)

# ---------------------------------------------------------------------------
# Thread-local sparse24 context
# ---------------------------------------------------------------------------

_sparse24_tls = threading.local()


def set_sparse24_context(apply_sparse24: bool, skip_diagonal_blocks: bool = True) -> None:
    """Set the thread-local sparse24 flags (called by Sparse24Triton.get_sparse_context)."""
    _sparse24_tls.apply_sparse24 = apply_sparse24
    _sparse24_tls.skip_diagonal_blocks = skip_diagonal_blocks


def get_sparse24_context() -> tuple[bool, bool]:
    """Read the thread-local sparse24 flags."""
    return (
        getattr(_sparse24_tls, "apply_sparse24", False),
        getattr(_sparse24_tls, "skip_diagonal_blocks", True),
    )


# ---------------------------------------------------------------------------
# Triton sparse24 attention adapted for diffusers tensor layout
# ---------------------------------------------------------------------------


def _diffusers_sparse24_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    skip_diagonal_blocks: bool = True,
) -> torch.Tensor:
    """Run Triton sparse24 attention on diffusers-layout tensors ``[B, S, H, D]``.

    Reshapes to the packed ``[total, H, D]`` format expected by
    ``context_attention_fwd``, runs the kernel with ``apply_sparse24=True``,
    and reshapes back.
    """
    batch, seq_len, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    device = query.device

    q = query.reshape(batch * seq_len, num_heads, head_dim).contiguous()
    k = key.reshape(batch * seq_len, num_kv_heads, head_dim).contiguous()
    v = value.reshape(batch * seq_len, num_kv_heads, head_dim).contiguous()
    o = torch.empty_like(q)

    b_start_loc = torch.arange(batch, device=device, dtype=torch.int32) * seq_len
    b_seq_len = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=seq_len,
        is_causal=is_causal,
        softmax_scale=scale,
        apply_sparse24=True,
        skip_diagonal_blocks=skip_diagonal_blocks,
    )

    return o.view(batch, seq_len, num_heads, head_dim)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_BACKEND_KEY = "modelopt_sparse24"


def register_diffusers_triton_attention() -> bool:
    """Register the sparse24 Triton kernel into diffusers' attention backend registry.

    When the backend is active, ``dispatch_attention_fn`` checks a thread-local
    flag.  If ``apply_sparse24`` is set (by ``SparseAttentionModule``), the call
    is routed to the Triton kernel.  Otherwise it falls through to whatever
    backend was active before registration.

    Returns:
        True if registration succeeded.
    """
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    except ImportError:
        return False

    if _BACKEND_KEY in _AttentionBackendRegistry._backends:
        return True

    original_backend_name = _AttentionBackendRegistry._active_backend
    original_fn = _AttentionBackendRegistry._backends.get(original_backend_name)
    original_arg_names = _AttentionBackendRegistry._supported_arg_names.get(
        original_backend_name, set()
    )

    def modelopt_sparse24_backend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        apply_sparse24, skip_diagonal_blocks = get_sparse24_context()

        seq_q = query.shape[1]
        seq_k = key.shape[1]
        can_use_triton = apply_sparse24 and seq_q == seq_k

        if can_use_triton:
            return _diffusers_sparse24_attention(
                query,
                key,
                value,
                is_causal=is_causal,
                scale=scale,
                skip_diagonal_blocks=skip_diagonal_blocks,
            )

        all_kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,
            **kwargs,
        }
        filtered = {k: v for k, v in all_kwargs.items() if k in original_arg_names}
        return original_fn(**filtered)

    _AttentionBackendRegistry._backends[_BACKEND_KEY] = modelopt_sparse24_backend
    _AttentionBackendRegistry._supported_arg_names[_BACKEND_KEY] = set(
        inspect.signature(modelopt_sparse24_backend).parameters.keys()
    )
    _AttentionBackendRegistry._constraints[_BACKEND_KEY] = []
    _AttentionBackendRegistry._active_backend = _BACKEND_KEY
    return True


__all__ = [
    "get_sparse24_context",
    "register_diffusers_triton_attention",
    "set_sparse24_context",
]
