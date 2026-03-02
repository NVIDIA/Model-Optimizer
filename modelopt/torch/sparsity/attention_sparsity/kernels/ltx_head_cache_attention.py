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

"""LTX-2 attention backend for per-head temporal caching.

Patches LTX-2 ``Attention`` modules so that ``self.attention_function`` routes
to the head-cache wrapper when the thread-local context is set by
``HeadCacheMethod.get_sparse_context``.

Thread-local context is shared with the diffusers backend so the same
``HeadCacheMethod.get_sparse_context`` context manager drives both code paths.
"""

from __future__ import annotations

import threading

import torch

# ---------------------------------------------------------------------------
# Thread-local head cache context (shared with diffusers backend)
# ---------------------------------------------------------------------------

_head_cache_tls = threading.local()


def set_head_cache_context(enabled: bool, cache_state) -> None:
    """Set thread-local head cache state (called by HeadCacheMethod.get_sparse_context)."""
    _head_cache_tls.enabled = enabled
    _head_cache_tls.cache_state = cache_state


def get_head_cache_context():
    """Read thread-local head cache state.

    Returns:
        Tuple of (enabled: bool, cache_state: HeadCacheState | None).
    """
    return (
        getattr(_head_cache_tls, "enabled", False),
        getattr(_head_cache_tls, "cache_state", None),
    )


# ---------------------------------------------------------------------------
# Core head-cache attention logic for LTX-2 layout [B, T, H*D]
# ---------------------------------------------------------------------------


def _head_cache_ltx_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    cache_state,
    original_fn,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute head-cache attention for LTX-2 layout tensors ``[B, T, H*D]``.

    Even steps: compute all heads with original_fn, cache output.
    Odd steps: compute only dynamic heads, merge with cached stable heads.
    """
    B, T, HD = q.shape
    D = HD // heads
    output_shape = (B, T, heads, D)

    # Auto-reset if output shape changed (new generation or different batch)
    if cache_state._prev_shape is not None and cache_state._prev_shape != output_shape:
        cache_state.reset()
    cache_state._prev_shape = output_shape

    # Calibration mode: compute all heads, record per-head similarity, return
    if cache_state.calibrating:
        output = original_fn(q, k, v, heads, mask)
        cache_state.record_output(output.view(B, T, heads, D))
        return output

    # If not calibrated (no stable heads selected), fall back to full compute
    if not cache_state.stable_heads:
        output = original_fn(q, k, v, heads, mask)
        # Still cache for potential future use and increment step
        cache_state.cached_output = output.view(B, T, heads, D).detach().clone()
        cache_state.step += 1
        return output

    is_compute_step = cache_state.step % 2 == 0

    if is_compute_step:
        # Even step: compute ALL heads, store cache
        output = original_fn(q, k, v, heads, mask)
        cache_state.cached_output = output.view(B, T, heads, D).detach().clone()
        cache_state.step += 1
        return output

    # Odd step: compute only dynamic heads, merge with cache
    stable_heads = cache_state.stable_heads
    dynamic_heads = cache_state.dynamic_heads
    num_dynamic = len(dynamic_heads)

    # Reshape to per-head view: [B, T, H, D]
    q_4d = q.view(B, T, heads, D)
    k_4d = k.view(B, T, heads, D)
    v_4d = v.view(B, T, heads, D)

    # Gather dynamic head slices
    dyn_idx = torch.tensor(dynamic_heads, device=q.device)
    q_dyn = q_4d[:, :, dyn_idx, :].reshape(B, T, num_dynamic * D)
    k_dyn = k_4d[:, :, dyn_idx, :].reshape(B, T, num_dynamic * D)
    v_dyn = v_4d[:, :, dyn_idx, :].reshape(B, T, num_dynamic * D)

    # Optionally apply 2:4 sparsity on dynamic heads
    if cache_state.apply_sparse24:
        try:
            from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention import (
                _ltx_sparse24_attention,
            )

            dyn_output = _ltx_sparse24_attention(q_dyn, k_dyn, v_dyn, num_dynamic)
        except ImportError:
            dyn_output = original_fn(q_dyn, k_dyn, v_dyn, num_dynamic, mask)
    else:
        dyn_output = original_fn(q_dyn, k_dyn, v_dyn, num_dynamic, mask)

    dyn_output_4d = dyn_output.view(B, T, num_dynamic, D)

    # Build full output: merge cached stable + computed dynamic
    output_4d = torch.empty(B, T, heads, D, device=q.device, dtype=q.dtype)

    # Copy cached stable heads
    stable_idx = torch.tensor(stable_heads, device=q.device)
    output_4d[:, :, stable_idx, :] = cache_state.cached_output[:, :, stable_idx, :]

    # Place computed dynamic heads
    output_4d[:, :, dyn_idx, :] = dyn_output_4d

    # Update cache with merged output
    cache_state.cached_output = output_4d.detach().clone()
    cache_state.step += 1

    return output_4d.reshape(B, T, HD)


# ---------------------------------------------------------------------------
# Wrapper that intercepts LTX-2 attention_function calls
# ---------------------------------------------------------------------------


class _HeadCacheLTXAttentionWrapper:
    """Wraps an LTX-2 ``AttentionCallable``; routes to head cache when context is active.

    When the thread-local head cache is enabled (by ``HeadCacheMethod.get_sparse_context``)
    and query/key sequence lengths match (self-attention), the call is forwarded to the
    head-cache logic. Otherwise the original attention function is called unchanged.
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
        enabled, cache_state = get_head_cache_context()

        seq_q = q.shape[1]
        seq_k = k.shape[1]
        can_use_cache = enabled and cache_state is not None and seq_q == seq_k

        if can_use_cache:
            return _head_cache_ltx_attention(q, k, v, heads, cache_state, self.original_fn, mask)

        return self.original_fn(q, k, v, heads, mask)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_ltx_head_cache_attention(model: torch.nn.Module) -> bool:
    """Patch LTX-2 ``Attention`` modules to route to the head-cache wrapper.

    Iterates over all modules in *model* and, for each ``ltx_core`` ``Attention``
    instance, replaces ``attention_function`` with a wrapper that checks the
    thread-local head cache context.

    If the attention_function is already a sparse24 wrapper, wraps its
    ``original_fn`` to avoid double-wrapping conflicts.

    Args:
        model: The model whose attention modules should be patched.

    Returns:
        True if at least one module was patched.
    """
    try:
        from ltx_core.model.transformer.attention import Attention as LTXAttention
    except ImportError:
        return False

    from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_triton_attention import (
        _Sparse24LTXAttentionWrapper,
    )

    patched = 0
    for module in model.modules():
        if isinstance(module, LTXAttention):
            fn = module.attention_function
            # Don't double-wrap with head cache
            if isinstance(fn, _HeadCacheLTXAttentionWrapper):
                continue
            # If already wrapped by sparse24, wrap the inner original_fn
            if isinstance(fn, _Sparse24LTXAttentionWrapper):
                fn.original_fn = _HeadCacheLTXAttentionWrapper(fn.original_fn)
            else:
                module.attention_function = _HeadCacheLTXAttentionWrapper(fn)
            patched += 1

    return patched > 0


__all__ = [
    "get_head_cache_context",
    "register_ltx_head_cache_attention",
    "set_head_cache_context",
]
