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

"""ModelOpt sparse attention plugin for diffusers WAN models.

Registers ``WanAttention`` with the ``SparseAttentionRegistry`` and provides
``ModelOptWanAttnProcessor``, a drop-in replacement for ``WanAttnProcessor`` that
calls the ModelOpt Triton flash-attention kernel (``modelopt.torch.kernels.triton_fa``)
with optional N:M sparsity or skip-softmax acceleration.

Integration:
- No diffusers-side changes required.
- Call ``mtsa.sparsify(model, config)`` with ``backend="diffusers_triton"`` in the
  layer config to activate this path.
- ``register_wan_sparse_attention`` is appended to ``CUSTOM_MODEL_PLUGINS`` so it is
  called automatically during ``mtsa.sparsify()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..sparse_attention import SparseAttentionModule, SparseAttentionRegistry
from . import CUSTOM_MODEL_PLUGINS


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings (matches WanAttnProcessor._apply_rotary_emb logic)."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class ModelOptWanAttnProcessor:
    """WAN attention processor backed by the ModelOpt Triton kernel.

    Replaces ``WanAttnProcessor`` on each ``WanAttention`` module that is wrapped
    by a ``WanSparseAttentionModule``.  The Triton kernel is called directly with
    flat-packed (varlen) tensors; no HuggingFace attention-dispatch machinery is
    used.

    Args:
        sparse_kw: Extra keyword arguments forwarded to ``triton_fa.attention()``.
            Typical entries: ``sparsity_n``, ``sparsity_m``, ``num_sink_tokens``,
            ``dense_window_size``, ``skip_softmax_threshold``.
    """

    _attention_backend = None  # kept for diffusers compatibility checks

    def __init__(self, sparse_kw: dict | None = None) -> None:
        """Initialize with optional sparse kwargs for the Triton kernel."""
        self.sparse_kw: dict = sparse_kw or {}
        # Controlled by WanSparseAttentionModule.forward() on every call.
        self._enabled: bool = True

    # ------------------------------------------------------------------
    # Triton helper
    # ------------------------------------------------------------------

    def _triton_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run triton_fa.attention on BSND-layout tensors.

        Args:
            query: ``[batch, seq_q, num_heads, head_dim]``
            key:   ``[batch, seq_kv, num_kv_heads, head_dim]``
            value: ``[batch, seq_kv, num_kv_heads, head_dim]``

        Returns:
            Output tensor of shape ``[batch, seq_q, num_heads, head_dim]``.
        """
        from modelopt.torch.kernels.triton_fa import attention as triton_attention

        batch, seq_q, num_heads, head_dim = query.shape
        seq_kv = key.shape[1]
        num_kv_heads = key.shape[2]

        q = query.reshape(batch * seq_q, num_heads, head_dim).contiguous()
        k = key.reshape(batch * seq_kv, num_kv_heads, head_dim).contiguous()
        v = value.reshape(batch * seq_kv, num_kv_heads, head_dim).contiguous()

        b_start_loc = torch.arange(batch, device=q.device, dtype=torch.int32) * seq_q
        b_seq_len = torch.full((batch,), seq_q, device=q.device, dtype=torch.int32)

        if seq_kv != seq_q:
            b_start_loc_k = torch.arange(batch, device=q.device, dtype=torch.int32) * seq_kv
            b_seq_len_k = torch.full((batch,), seq_kv, device=q.device, dtype=torch.int32)
            max_input_len_k: int | None = seq_kv
        else:
            b_start_loc_k = None
            b_seq_len_k = None
            max_input_len_k = None

        out = triton_attention(
            q,
            k,
            v,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=seq_q,
            is_causal=False,
            b_start_loc_k=b_start_loc_k,
            b_seq_len_k=b_seq_len_k,
            max_input_len_k=max_input_len_k,
            **self.sparse_kw,
        )
        return out.view(batch, seq_q, num_heads, head_dim)

    # ------------------------------------------------------------------
    # Processor entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        attn: WanAttention,  # type: ignore[name-defined]  # noqa: F821
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass — replaces WanAttnProcessor.__call__."""
        if not self._enabled:
            # Fallback: use standard PyTorch SDPA
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            return self._wan_forward_sdpa(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                dispatch_fn=dispatch_attention_fn,
            )

        return self._wan_forward_triton(
            attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_qkv(
        attn: WanAttention,  # type: ignore[name-defined]  # noqa: F821
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, norm, unflatten, and apply rotary embeddings."""
        enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(enc).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(enc)
            value = attn.to_v(enc)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        return query, key, value

    def _wan_forward_triton(
        self,
        attn: WanAttention,  # type: ignore[name-defined]  # noqa: F821
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Triton-backed WAN attention (self-attention and I2V cross-attention)."""
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the text-encoder context length (WAN hardcoded constant)
            assert encoder_hidden_states is not None
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self._prepare_qkv(
            attn, hidden_states, encoder_hidden_states, rotary_emb
        )

        # I2V: image cross-attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            if attn.fused_projections:
                key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
            else:
                key_img = attn.add_k_proj(encoder_hidden_states_img)
                value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self._triton_attention(query, key_img, value_img)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        # Main attention (self or text cross-attention)
        hidden_states = self._triton_attention(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _wan_forward_sdpa(
        self,
        attn: WanAttention,  # type: ignore[name-defined]  # noqa: F821
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        dispatch_fn,
    ) -> torch.Tensor:
        """Fallback path using diffusers dispatch_attention_fn (SDPA)."""
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            assert encoder_hidden_states is not None
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self._prepare_qkv(
            attn, hidden_states, encoder_hidden_states, rotary_emb
        )

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            if attn.fused_projections:
                key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
            else:
                key_img = attn.add_k_proj(encoder_hidden_states_img)
                value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = dispatch_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanSparseAttentionModule(SparseAttentionModule):
    """SparseAttentionModule specialisation for diffusers WanAttention.

    Installs a ``ModelOptWanAttnProcessor`` on the wrapped ``WanAttention`` so
    that the ModelOpt Triton kernel is used for attention computation.  The
    processor is re-installed whenever the sparse config changes.
    """

    def _setup(self) -> None:
        super()._setup()
        self._update_processor()

    def set_from_attribute_config(self, attribute_cfg=None) -> None:
        """Set config and refresh the Triton processor with updated sparse kwargs."""
        super().set_from_attribute_config(attribute_cfg)
        self._update_processor()

    def _build_sparse_kw(self) -> dict:
        """Extract triton_fa kwargs from the current method config."""
        cfg = getattr(self, "_method_config", {})
        method = getattr(self, "_method", "")
        kw: dict = {}
        if method == "triton_sparse_softmax":
            kw["sparsity_n"] = cfg.get("sparsity_n", 2)
            kw["sparsity_m"] = cfg.get("sparsity_m", 4)
            kw["num_sink_tokens"] = cfg.get("num_sink_tokens", 0)
            kw["dense_window_size"] = cfg.get("dense_window_size", 64)
        elif method == "triton_skip_softmax":
            kw["skip_softmax_threshold"] = cfg.get("skip_softmax_threshold", 0.1)
        if cfg.get("quantize_p", False):
            kw["quantize_p"] = True
        return kw

    def _update_processor(self) -> None:
        """Install or refresh the ModelOptWanAttnProcessor."""
        proc = ModelOptWanAttnProcessor(sparse_kw=self._build_sparse_kw())
        # set_processor is a method on WanAttention; DynamicModule delegates via __getattr__
        self.set_processor(proc)

    def forward(self, *args, **kwargs):
        """Forward that calls ModelOptWanAttnProcessor directly.

        WanAttention is a new-style diffusers AttentionModuleMixin that uses
        dispatch_attention_fn internally — it does NOT call self.processor().
        So we intercept here and call the processor directly, bypassing the
        original WanAttention.forward().

        The processor's __call__ handles both paths:
          - enabled=True  → _wan_forward_triton (ModelOpt Triton kernel)
          - enabled=False → _wan_forward_sdpa   (dispatch_attention_fn fallback)
        """
        proc = getattr(self, "processor", None)
        if isinstance(proc, ModelOptWanAttnProcessor):
            proc._enabled = self.is_enabled
            return proc(self, *args, **kwargs)
        # Fallback: original WanAttention.forward() via DynamicModule
        return super(SparseAttentionModule, self).forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Plugin registration callback
# ---------------------------------------------------------------------------


def register_wan_sparse_attention(model: nn.Module) -> bool:
    """Register WanAttention modules for ModelOpt sparse attention.

    Called automatically by ``register_custom_model_plugins_on_the_fly`` during
    ``mtsa.sparsify()``.

    Args:
        model: The diffusers model being sparsified.

    Returns:
        True if at least one WanAttention type was registered.
    """
    try:
        from diffusers.models.transformers.transformer_wan import WanAttention
    except ImportError:
        return False

    if not any(isinstance(m, WanAttention) for m in model.modules()):
        return False

    # Register once — idempotent if already registered
    if WanAttention not in SparseAttentionRegistry:
        SparseAttentionRegistry.register({WanAttention: "WanAttention"})(WanSparseAttentionModule)

    return True


CUSTOM_MODEL_PLUGINS.append(register_wan_sparse_attention)

__all__ = [
    "ModelOptWanAttnProcessor",
    "WanSparseAttentionModule",
    "register_wan_sparse_attention",
]
