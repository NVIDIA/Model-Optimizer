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

"""Eager attention backend for diffusers skip-softmax sparse attention.

Registers a ``modelopt_skip_softmax`` backend in diffusers'
``_AttentionBackendRegistry`` that computes attention eagerly with an explicit
``F.softmax`` call.  This allows the existing softmax-patching mechanism in
``SparseAttentionModule`` to intercept and apply block-wise sparsity.

Used during **calibration only** — inference uses the Triton FA kernel.
"""

import inspect
import math

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    attention_backend,
)

_BACKEND_NAME = "modelopt_skip_softmax"
_BACKEND_REGISTERED = False


# ---------------------------------------------------------------------------
# Eager attention implementation
# ---------------------------------------------------------------------------


def _diffusers_eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Compute attention eagerly on diffusers layout ``[B, S, H, D]``.

    The explicit ``F.softmax`` call is what the skip-softmax patch intercepts.
    """
    # Diffusers convention: [B, S, H, D] → transpose to [B, H, S, D]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Handle GQA: repeat K/V heads to match Q heads
    if enable_gqa and query.shape[1] != key.shape[1]:
        num_heads_q = query.shape[1]
        num_heads_kv = key.shape[1]
        n_rep = num_heads_q // num_heads_kv
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # Q @ K^T * scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Apply causal mask if needed
    if is_causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.full((seq_q, seq_k), float("-inf"), device=scores.device, dtype=scores.dtype),
            diagonal=seq_k - seq_q + 1,
        )
        scores = scores + causal_mask

    # F.softmax — this is where the skip-softmax patch intercepts
    scores = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        scores = F.dropout(scores, p=dropout_p, training=True)

    # scores @ V
    out = torch.matmul(scores, value)

    # Transpose back: [B, H, S, D] → [B, S, H, D]
    out = out.transpose(1, 2)
    return out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_diffusers_eager_attention() -> None:
    """Register ``modelopt_skip_softmax`` backend in diffusers.

    Safe to call multiple times; registration happens only once.
    """
    global _BACKEND_REGISTERED
    if _BACKEND_REGISTERED:
        return

    # Extend the AttentionBackendName enum with our custom value
    new_member = str.__new__(AttentionBackendName, _BACKEND_NAME)
    new_member._name_ = "MODELOPT_SKIP_SOFTMAX"
    new_member._value_ = _BACKEND_NAME
    AttentionBackendName._member_map_["MODELOPT_SKIP_SOFTMAX"] = new_member
    AttentionBackendName._value2member_map_[_BACKEND_NAME] = new_member

    # Register the backend function
    _AttentionBackendRegistry._backends[new_member] = _diffusers_eager_attention
    _AttentionBackendRegistry._constraints[new_member] = []
    _AttentionBackendRegistry._supported_arg_names[new_member] = set(
        inspect.signature(_diffusers_eager_attention).parameters.keys()
    )

    _BACKEND_REGISTERED = True


def get_skip_softmax_attention_backend():
    """Return a context manager that activates the modelopt_skip_softmax backend.

    Raises RuntimeError if the backend has not been registered yet.
    """
    if not _BACKEND_REGISTERED:
        raise RuntimeError(
            "modelopt_skip_softmax backend not registered. "
            "Call register_diffusers_eager_attention() first."
        )
    return attention_backend(_BACKEND_NAME)
