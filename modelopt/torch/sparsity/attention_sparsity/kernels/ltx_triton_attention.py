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

"""Triton flash attention wrapper for LTX-2 (ltx_core) skip-softmax sparse attention.

Patches ``Attention`` modules from ``ltx_core`` so that when the Triton
skip-softmax flag is active, attention is computed via the Triton FA kernel
with fused tile skipping.

Used during **inference** -- calibration uses the eager wrapper instead.
"""

import math
import threading

import torch
from ltx_core.model.transformer.attention import Attention

from modelopt.torch.kernels import attention

# Thread-local storage for skip-softmax configuration
_thread_local = threading.local()


def set_ltx_triton_context(
    active: bool,
    threshold: float | None = None,
) -> None:
    """Set thread-local Triton config for LTX-2 attention."""
    _thread_local.active = active
    _thread_local.threshold = threshold


def clear_ltx_triton_context() -> None:
    """Clear thread-local Triton config."""
    _thread_local.active = False
    _thread_local.threshold = None


def _get_ltx_triton_context() -> tuple[bool, float | None]:
    """Return (active, threshold)."""
    return (
        getattr(_thread_local, "active", False),
        getattr(_thread_local, "threshold", None),
    )


def _ltx_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    threshold: float | None = None,
) -> torch.Tensor:
    """Triton FA attention on LTX-2 layout ``[B, T, H*D]``.

    Converts from LTX-2's fused-head layout to the Triton kernel's varlen
    format, calls the kernel with skip-softmax, and converts back.
    """
    b, seq_q, dim_total = q.shape
    dim_head = dim_total // heads
    seq_k = k.shape[1]
    device = q.device

    # LTX-2 layout: [B, T, H*D] -> reshape to [B, T, H, D] -> flat [B*T, H, D]
    q_flat = q.view(b, seq_q, heads, dim_head).reshape(b * seq_q, heads, dim_head).contiguous()
    k_flat = k.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()
    v_flat = v.view(b, seq_k, heads, dim_head).reshape(b * seq_k, heads, dim_head).contiguous()

    # Build varlen metadata
    b_start_loc_q = torch.arange(b, device=device, dtype=torch.int32) * seq_q
    b_seq_len_q = torch.full((b,), seq_q, device=device, dtype=torch.int32)

    scale = 1.0 / math.sqrt(dim_head)

    kw: dict = {
        "b_start_loc": b_start_loc_q,
        "b_seq_len": b_seq_len_q,
        "max_input_len": seq_q,
        "is_causal": False,  # Diffusion uses bidirectional attention
        "softmax_scale": scale,
    }

    # Handle different Q/KV sequence lengths
    if seq_q != seq_k:
        b_start_loc_k = torch.arange(b, device=device, dtype=torch.int32) * seq_k
        b_seq_len_k = torch.full((b,), seq_k, device=device, dtype=torch.int32)
        kw["b_start_loc_k"] = b_start_loc_k
        kw["b_seq_len_k"] = b_seq_len_k
        kw["max_input_len_k"] = seq_k

    # Skip-softmax threshold
    if threshold is not None and threshold > 0.0:
        kw["skip_softmax_threshold"] = threshold

    assert attention is not None, "Triton attention kernel not available (requires CUDA + triton)"
    o = attention(q_flat, k_flat, v_flat, **kw)

    # Reshape back: [B*T, H, D] -> [B, T, H*D]
    return o.view(b, seq_q, heads * dim_head)


class _TritonLTXAttentionWrapper:
    """Wraps an ``attention_function`` callable from ltx_core.

    When the thread-local Triton skip-softmax flag is active, routes to the
    Triton FA kernel.  Otherwise calls the original function.
    """

    def __init__(self, original_fn):
        self._original_fn = original_fn

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active, threshold = _get_ltx_triton_context()
        if active:
            return _ltx_triton_attention(q, k, v, heads, mask, threshold)
        return self._original_fn(q, k, v, heads, mask)


def register_ltx_triton_attention(model: torch.nn.Module) -> None:
    """Walk *model* and patch all ``ltx_core.Attention`` modules for Triton dispatch.

    Safe to call multiple times -- already-wrapped modules are skipped.
    """
    for module in model.modules():
        if isinstance(module, Attention):
            fn = module.attention_function
            if not isinstance(fn, _TritonLTXAttentionWrapper):
                module.attention_function = _TritonLTXAttentionWrapper(fn)
