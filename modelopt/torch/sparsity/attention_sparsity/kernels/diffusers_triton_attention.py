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
[total_tokens, H, D] format. Supports skip-softmax tile skipping for sparse attention.

Used during **inference** -- calibration uses the eager backend instead.
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

from modelopt.torch.kernels.triton_fa import attention

_BACKEND_NAME = "modelopt_triton"
_BACKEND_REGISTERED = False

# Thread-local storage for per-forward skip-softmax configuration.
# The method's get_sparse_context() sets these before each forward pass.
_thread_local = threading.local()


def set_triton_skip_softmax_config(threshold: float | None = None) -> None:
    """Set thread-local skip-softmax config for the next Triton attention call."""
    _thread_local.skip_threshold = threshold


def clear_triton_skip_softmax_config() -> None:
    """Clear thread-local skip-softmax config."""
    _thread_local.skip_threshold = None


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
    """Compute attention via Triton FA kernel on diffusers layout ``[B, S, H, D]``.

    Converts to the kernel's varlen format, calls the Triton FA kernel, and
    converts back.
    """
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

    # If Q and KV have different sequence lengths, pass separate KV metadata
    if seq_q != seq_k:
        b_start_loc_k = torch.arange(batch, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((batch,), seq_k, device=device, dtype=torch.int32)
        kw["b_start_loc_k"] = b_start_loc_k
        kw["b_seq_len_k"] = b_seq_len_k
        kw["max_input_len_k"] = seq_k

    # Read skip-softmax config from thread-local storage
    threshold = getattr(_thread_local, "skip_threshold", None)
    if threshold is not None and threshold > 0.0:
        kw["skip_softmax_threshold"] = threshold

    o = attention(q, k, v, **kw)

    # Reshape back: [B*S, H, D] -> [B, S, H, D]
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

    # Extend the AttentionBackendName enum with our custom value
    new_member = str.__new__(AttentionBackendName, _BACKEND_NAME)
    new_member._name_ = "MODELOPT_TRITON"
    new_member._value_ = _BACKEND_NAME
    AttentionBackendName._member_map_["MODELOPT_TRITON"] = new_member
    AttentionBackendName._value2member_map_[_BACKEND_NAME] = new_member

    # Register the backend function
    _AttentionBackendRegistry._backends[new_member] = _diffusers_triton_attention
    _AttentionBackendRegistry._constraints[new_member] = []
    _AttentionBackendRegistry._supported_arg_names[new_member] = set(
        inspect.signature(_diffusers_triton_attention).parameters.keys()
    )

    _BACKEND_REGISTERED = True


def get_triton_attention_backend():
    """Return a context manager that activates the modelopt_triton backend.

    Raises RuntimeError if the backend has not been registered yet.
    """
    if not _BACKEND_REGISTERED:
        raise RuntimeError(
            "modelopt_triton backend not registered. "
            "Call register_diffusers_triton_attention() first."
        )
    return attention_backend(_BACKEND_NAME)
