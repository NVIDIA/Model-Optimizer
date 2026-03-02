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

"""Diffusers attention backend for per-head temporal caching.

Registers into diffusers' ``_AttentionBackendRegistry`` so that diffusers models
can use head caching via the standard ``dispatch_attention_fn`` path.

The thread-local context is shared with the LTX-2 backend
(``kernels/ltx_head_cache_attention.py``).
"""

from __future__ import annotations

import inspect

import torch

from modelopt.torch.sparsity.attention_sparsity.kernels.ltx_head_cache_attention import (
    get_head_cache_context,
)

# ---------------------------------------------------------------------------
# Core head-cache attention for diffusers layout [B, S, H, D]
# ---------------------------------------------------------------------------


def _diffusers_head_cache_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache_state,
    original_fn,
    original_arg_names: set[str],
    is_causal: bool = False,
    scale: float | None = None,
    **extra_kwargs,
) -> torch.Tensor:
    """Execute head-cache attention for diffusers layout tensors ``[B, S, H, D]``.

    Even steps: compute all heads with original_fn, cache output.
    Odd steps: compute only dynamic heads, merge with cached stable heads.
    """
    B, S, H, D = query.shape
    output_shape = (B, S, H, D)

    # Auto-reset if output shape changed (new generation or different batch)
    if cache_state._prev_shape is not None and cache_state._prev_shape != output_shape:
        cache_state.reset()
    cache_state._prev_shape = output_shape

    def _call_original(q, k, v, **kw):
        all_kwargs = {"query": q, "key": k, "value": v, "is_causal": is_causal, "scale": scale}
        all_kwargs.update(kw)
        all_kwargs.update(extra_kwargs)
        filtered = {k_: v_ for k_, v_ in all_kwargs.items() if k_ in original_arg_names}
        return original_fn(**filtered)

    # Calibration mode: compute all heads, record per-head similarity, return
    if cache_state.calibrating:
        output = _call_original(query, key, value)
        cache_state.record_output(output)  # already [B, S, H, D]
        return output

    # If not calibrated, fall back to full compute
    if not cache_state.stable_heads:
        output = _call_original(query, key, value)
        cache_state.cached_output = output.detach().clone()
        cache_state.step += 1
        return output

    is_compute_step = cache_state.step % 2 == 0

    if is_compute_step:
        # Even step: compute ALL heads, store cache
        output = _call_original(query, key, value)
        cache_state.cached_output = output.detach().clone()
        cache_state.step += 1
        return output

    # Odd step: compute only dynamic heads, merge with cache
    stable_heads = cache_state.stable_heads
    dynamic_heads = cache_state.dynamic_heads

    # Gather dynamic head slices along dim=2 (head dimension)
    dyn_idx = torch.tensor(dynamic_heads, device=query.device)
    q_dyn = query[:, :, dyn_idx, :]
    k_dyn = key[:, :, dyn_idx, :]
    v_dyn = value[:, :, dyn_idx, :]

    # Optionally apply 2:4 sparsity on dynamic heads
    if cache_state.apply_sparse24:
        try:
            from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
                _diffusers_sparse24_attention,
            )

            dyn_output = _diffusers_sparse24_attention(
                q_dyn,
                k_dyn,
                v_dyn,
                is_causal=is_causal,
                scale=scale,
            )
        except ImportError:
            dyn_output = _call_original(q_dyn, k_dyn, v_dyn)
    else:
        dyn_output = _call_original(q_dyn, k_dyn, v_dyn)

    # Build full output: merge cached stable + computed dynamic
    output = torch.empty(B, S, H, D, device=query.device, dtype=query.dtype)

    stable_idx = torch.tensor(stable_heads, device=query.device)
    output[:, :, stable_idx, :] = cache_state.cached_output[:, :, stable_idx, :]
    output[:, :, dyn_idx, :] = dyn_output

    # Update cache with merged output
    cache_state.cached_output = output.detach().clone()
    cache_state.step += 1

    return output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_BACKEND_KEY = "modelopt_head_cache"


def register_diffusers_head_cache_attention() -> bool:
    """Register the head-cache backend into diffusers' attention backend registry.

    When the backend is active, ``dispatch_attention_fn`` checks the thread-local
    head cache flag. If enabled (by ``HeadCacheMethod.get_sparse_context``) and
    self-attention (seq_q == seq_k), routes to head-cache logic. Otherwise falls
    through to the previous active backend.

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

    def modelopt_head_cache_backend(
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
        enabled, cache_state = get_head_cache_context()

        seq_q = query.shape[1]
        seq_k = key.shape[1]
        can_use_cache = enabled and cache_state is not None and seq_q == seq_k

        if can_use_cache:
            return _diffusers_head_cache_attention(
                query,
                key,
                value,
                cache_state=cache_state,
                original_fn=original_fn,
                original_arg_names=original_arg_names,
                is_causal=is_causal,
                scale=scale,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=enable_gqa,
                **kwargs,
            )

        # Fallback to original backend
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

    _AttentionBackendRegistry._backends[_BACKEND_KEY] = modelopt_head_cache_backend
    _AttentionBackendRegistry._supported_arg_names[_BACKEND_KEY] = set(
        inspect.signature(modelopt_head_cache_backend).parameters.keys()
    )
    _AttentionBackendRegistry._constraints[_BACKEND_KEY] = []
    _AttentionBackendRegistry._active_backend = _BACKEND_KEY
    return True


__all__ = [
    "register_diffusers_head_cache_attention",
]
