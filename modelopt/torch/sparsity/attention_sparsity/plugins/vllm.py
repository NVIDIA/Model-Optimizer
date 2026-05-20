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

"""ModelOpt sparse attention backend for vLLM.

Registers a custom vLLM attention backend that uses the ModelOpt Triton kernel
with paged KV cache support. Integration approach:

- No module replacement — the Attention module stays intact with all its state
- Only ``impl`` is swapped from FlashAttentionImpl to ModelOptSparseAttentionImpl
- KV cache update is handled by vLLM (inherited ``do_kv_cache_update``)
- Only ``forward()`` is overridden to call our Triton kernel for both prefill and decode

For MLA (Multi-Latent Attention) models like DeepSeek, a different strategy is used:
the MLA impl's prefill methods are monkey-patched to call our Triton kernel in
contiguous (non-paged) mode, since MLA decompresses KV latents before attention.
"""

import importlib.util
import types

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from modelopt.torch.kernels.common.attention.triton_fa import attention as triton_attention
from modelopt.torch.kernels.common.attention.triton_fa import attention_with_lse

_HAS_MLA = (
    importlib.util.find_spec("vllm.model_executor.layers.attention.mla_attention") is not None
)

logger = init_logger(__name__)

_LOGGED_CONFIGS: set[tuple] = set()


class ModelOptSparseAttentionImpl(FlashAttentionImpl):
    """Attention implementation that uses the ModelOpt Triton kernel.

    Inherits from FlashAttentionImpl to reuse:
    - __init__ (all configuration)
    - do_kv_cache_update (KV cache writing)
    Only overrides forward() to replace the attention computation.
    """

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with ModelOpt Triton sparse attention kernel."""
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        is_prefill = attn_metadata.max_query_len > 1

        # Unpack paged KV cache: [2, num_blocks, page_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(0)
        page_size = key_cache.shape[1]

        # Per-layer sparse kwargs (set by _replace_attention_impl in the worker)
        sparse_kw = getattr(self, "sparse_kw", {})

        # Log once per unique sparse_kw config so each variant is reported
        # the first time it actually executes (prefill or decode).
        kw_key = tuple(sorted(sparse_kw.items()))
        if kw_key not in _LOGGED_CONFIGS:
            variants = []
            if "sparsity_n" in sparse_kw:
                variants.append(
                    f"{sparse_kw['sparsity_n']}:{sparse_kw['sparsity_m']} N:M"
                )
            if "skip_softmax_threshold" in sparse_kw:
                variants.append(
                    f"skip_softmax(threshold={sparse_kw['skip_softmax_threshold']})"
                )
            variant_str = " + ".join(variants) if variants else "dense"
            msg = (
                f"[ModelOpt] sparse Triton kernel active: kernel={variant_str} "
                f"phase={'prefill' if is_prefill else 'decode'} "
                f"layer={layer.__class__.__name__} "
                f"sparse_kw={sparse_kw}"
            )
            logger.warning(msg)
            print(msg, flush=True)
            _LOGGED_CONFIGS.add(kw_key)

        # Prepare metadata for our kernel
        q = query[:num_actual_tokens].contiguous()
        cu_seqlens_q = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        batch = seq_lens.shape[0]

        b_start_loc = cu_seqlens_q[:batch]
        b_seq_len = cu_seqlens_q[1 : batch + 1] - cu_seqlens_q[:batch]

        # Dummy K/V for paged mode: not used by the kernel (KV are read from
        # k_cache/v_cache via block_table), but shape[1] must be num_kv_heads
        # so the kernel computes the correct GQA ratio (num_q_heads // num_kv_heads).
        k_dummy = torch.empty(0, self.num_kv_heads, self.head_size, device=q.device, dtype=q.dtype)

        # Call ModelOpt Triton kernel with paged KV.
        # b_seq_len is the query length (e.g., 6 for prefill, 1 for decode).
        # b_seq_len_k is the total KV length including cache (e.g., 6 for first
        # prefill, 7/8/... for subsequent decode steps).
        triton_out = triton_attention(
            q,
            k=k_dummy,
            v=k_dummy,
            # Query metadata
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=attn_metadata.max_query_len,
            is_causal=is_prefill,  # causal for prefill, non-causal for decode
            softmax_scale=self.scale,
            # KV metadata
            b_start_loc_k=None,  # paged mode: KV offsets not needed
            b_seq_len_k=seq_lens,  # total KV length per sequence
            max_input_len_k=attn_metadata.max_seq_len,
            # Paged KV cache
            k_cache=key_cache,  # [num_blocks, page_size, num_kv_heads, head_dim]
            v_cache=value_cache,  # [num_blocks, page_size, num_kv_heads, head_dim]
            block_table=attn_metadata.block_table,  # [batch, max_blocks]
            page_size=page_size,  # tokens per page in the KV cache
            **sparse_kw,
        )

        output[:num_actual_tokens] = triton_out
        return output


class ModelOptSparseAttentionBackend(FlashAttentionBackend):
    """Attention backend that uses ModelOpt's sparse Triton kernel.

    Inherits everything from FlashAttentionBackend except get_impl_cls and get_name.
    """

    @staticmethod
    def get_name() -> str:
        """Return backend name."""
        return "MODELOPT_SPARSE"

    @staticmethod
    def get_impl_cls() -> type:
        """Return the attention implementation class."""
        return ModelOptSparseAttentionImpl


# ---------------------------------------------------------------------------
# MLA (Multi-Latent Attention) sparse prefill support
# ---------------------------------------------------------------------------
# MLA models (DeepSeek) decompress KV latents to full Q, K, V tensors before
# calling attention in the prefill path. We replace the prefill methods on the
# MLA impl to use our Triton kernel in contiguous mode. V is zero-padded to
# match Q/K head_dim; the caller (_forward_prefill) slices the output back to
# V's head_dim when _pad_v=True.
#
# Decode is unchanged — it uses specialized MLA-aware backends (FlashInfer MLA,
# FlashMLA, TRT-LLM) that operate on compressed latents.
# ---------------------------------------------------------------------------


def _modelopt_mla_run_prefill_new_tokens(self, prefill, q, k, v, return_softmax_lse):
    """ModelOpt sparse attention for MLA new tokens (causal).

    Replaces ``MLACommonImpl._run_prefill_new_tokens`` when sparse attention
    is enabled. Pads V to Q's head_dim, calls the ModelOpt Triton kernel in
    contiguous mode, and returns LSE in ``[num_heads, total_tokens]`` format.
    """
    padded_v = F.pad(v, [0, q.shape[-1] - v.shape[-1]]) if v.shape[-1] < q.shape[-1] else v

    cu = prefill.query_start_loc
    batch = cu.shape[0] - 1
    b_start_loc = cu[:batch]
    b_seq_len = cu[1 : batch + 1] - cu[:batch]

    o, lse = attention_with_lse(
        q,
        k,
        padded_v,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=prefill.max_query_len,
        is_causal=True,
        softmax_scale=self.scale,
        **self._modelopt_sparse_kw,
    )

    if return_softmax_lse:
        return o, lse.transpose(0, 1).contiguous()
    return o


def _modelopt_mla_run_prefill_context_chunk(self, prefill, chunk_idx, q, k, v):
    """ModelOpt sparse attention for MLA context chunks (non-causal).

    Replaces ``MLACommonImpl._run_prefill_context_chunk``.  Always returns
    ``(output, lse)`` since chunked context needs LSE for merging.
    """
    padded_v = F.pad(v, [0, q.shape[-1] - v.shape[-1]]) if v.shape[-1] < q.shape[-1] else v

    cu_q = prefill.query_start_loc
    cu_k = prefill.chunked_context.cu_seq_lens[chunk_idx]
    batch = cu_q.shape[0] - 1

    sparse_kw = dict(self._modelopt_sparse_kw)
    sparse_kw["dense_window_size"] = 0  # no dense window for non-causal context

    o, lse = attention_with_lse(
        q,
        k,
        padded_v,
        b_start_loc=cu_q[:batch],
        b_seq_len=cu_q[1 : batch + 1] - cu_q[:batch],
        max_input_len=prefill.max_query_len,
        is_causal=False,
        softmax_scale=self.scale,
        b_start_loc_k=cu_k[:batch],
        b_seq_len_k=cu_k[1 : batch + 1] - cu_k[:batch],
        max_input_len_k=prefill.chunked_context.max_seq_lens[chunk_idx],
        **sparse_kw,
    )

    return o, lse.transpose(0, 1).contiguous()


def patch_mla_impl_for_sparse(impl, sparse_kw: dict) -> None:
    """Monkey-patch an MLACommonImpl to use ModelOpt sparse prefill.

    Sets ``_pad_v=True`` so that ``_forward_prefill`` slices the output back
    to ``v_head_dim`` after attention. Replaces the prefill method pointers
    with our Triton-kernel-based implementations.

    Args:
        impl: An ``MLACommonImpl`` instance (or subclass like ``FlashInferMLAImpl``).
        sparse_kw: Sparse attention config dict with keys like ``sparsity_n``,
            ``sparsity_m``, ``num_sink_tokens``, ``dense_window_size``.
    """
    impl._modelopt_sparse_kw = sparse_kw
    impl._pad_v = True
    impl._run_prefill_new_tokens = types.MethodType(_modelopt_mla_run_prefill_new_tokens, impl)
    impl._run_prefill_context_chunk = types.MethodType(
        _modelopt_mla_run_prefill_context_chunk, impl
    )
