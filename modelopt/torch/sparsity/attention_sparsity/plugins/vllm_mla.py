# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ModelOpt MLA attention adapter for vLLM's TRITON_MLA backend.

Same integration philosophy as the regular-attention adapter in
``plugins/vllm.py``: the ``MLAAttention`` module stays intact (its module-level
``kv_c``/``k_pe`` quantizers provide the write-once latent-cache QDQ before the
native cache write), and only ``layer.impl`` is reclassed. vLLM keeps owning
projections, RoPE, cache writes, metadata, chunked-context gathering, state
merging, and the V up-projection:

- ``forward_mha`` (all prefill) temporarily swaps the per-layer ModelOpt
  prefill backend into the prefill metadata and delegates to the inherited
  implementation, so the ``kv_b_proj`` projection and chunked-context plumbing
  are reused rather than forked. The backend routes the two attention calls
  (causal new tokens; non-causal context chunks with LSE) to
  :func:`mla_prefill_attention` with fused Q/K/P/V QDQ and optional 2:4
  score sparsity (prefill only).
- ``forward_mqa`` (decode) fake-quantizes the absorbed query (FP32 QDQ
  carrier) and calls :func:`mla_attention_decode`, which fuses the P QDQ.
  BMM1-K and BMM2-V both consume the write-once quantized latent cache as-is
  (single stored representation; no on-read re-quantization).
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

from modelopt.torch.kernels.quantization.attention.mla import (
    mla_attention_decode,
    mla_prefill_attention,
)

from . import vllm as attention_plugin

__all__ = [
    "ModelOptMLAImpl",
    "clone_mla_impl",
    "mla_quant_kw_from_layer",
    "select_mla_impl_cls",
]

# Fixed split count: decode P QDQ numerics follow the split-local schedule, so
# a fixed value keeps results reproducible across batch shapes and devices.
_MLA_DECODE_NUM_KV_SPLITS = 32


@dataclass(frozen=True, slots=True)
class _MLAQuantKw:
    """Kernel quantization kwargs resolved once per layer at install time."""

    prefill: dict[str, Any]
    decode: dict[str, Any]

    @property
    def any_active(self) -> bool:
        return (
            any(self.prefill[f"{op}_quant"] is not None for op in ("q", "k", "p", "v"))
            or self.decode["p_qdq"] is not None
        )


def mla_quant_kw_from_layer(layer, *, query_in_kernel: bool) -> _MLAQuantKw:
    """Resolve the MLA kernels' quantization kwargs from the layer's quantizers.

    ``kv_c``/``k_pe`` quantizers are module-level (write-once latent-cache QDQ)
    and therefore never appear here. With ``query_in_kernel`` False (FP8 Q),
    the module-level quantizer already QDQ'd the 192-d query, so neither
    kernel applies a Q transform.
    """
    q_qdq, q_amax = attention_plugin._bmm_qdq_from_layer(layer, "q_bmm_quantizer", None)
    k_qdq, k_amax = attention_plugin._bmm_qdq_from_layer(layer, "k_mha_bmm_quantizer", None)
    v_qdq, v_amax = attention_plugin._bmm_qdq_from_layer(layer, "v_mha_bmm_quantizer", None)
    p_qdq, p_amax = attention_plugin._p_qdq_from_layer(layer)
    return _MLAQuantKw(
        prefill={
            "q_quant": q_qdq if query_in_kernel else None,
            "q_amax": q_amax,
            "k_quant": k_qdq,
            "k_amax": k_amax,
            "p_quant": p_qdq,
            "p_amax": p_amax,
            "v_quant": v_qdq,
            "v_amax": v_amax,
        },
        decode={"p_qdq": p_qdq, "p_qdq_amax": p_amax},
    )


class _ModelOptMLAPrefillBackend(MLAPrefillBackend):
    """Per-layer prefill backend carrying this layer's quant/sparse kwargs.

    Swapped into ``prefill_metadata.prefill_backend`` for the duration of one
    ``forward_mha`` call (the builder-stamped backend is shared across layers,
    so per-layer state cannot live there permanently).
    """

    def __init__(self, base_backend: MLAPrefillBackend, quant_kw: dict, sparse_kw: dict):
        super().__init__(
            num_heads=base_backend.num_heads,
            scale=base_backend.scale,
            kv_lora_rank=base_backend.kv_lora_rank,
            qk_nope_head_dim=base_backend.qk_nope_head_dim,
            qk_rope_head_dim=base_backend.qk_rope_head_dim,
            v_head_dim=base_backend.v_head_dim,
            vllm_config=base_backend.vllm_config,
        )
        self._quant_kw = dict(quant_kw)
        self._sparse_kw = dict(sparse_kw)

    @staticmethod
    def get_name() -> str:
        return "MODELOPT_MLA"

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output_scale is not None:
            raise NotImplementedError(
                "ModelOpt MLA attention does not support fused FP8 output quantization"
            )
        pm = self._prefill_metadata
        result = mla_prefill_attention(
            q,
            k,
            v,
            cu_seqlens_q=pm.query_start_loc,
            cu_seqlens_k=pm.query_start_loc,
            max_seqlen_q=pm.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_lse=return_softmax_lse,
            **self._quant_kw,
            **self._sparse_kw,
        )
        if out is not None and not return_softmax_lse:
            out.copy_(result)
            return out
        return result

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pm = self._prefill_metadata
        assert pm.chunked_context is not None
        # No sparse kwargs here: the 2:4 dense-window semantics are
        # suffix-relative, so cached-context chunks run dense.
        return mla_prefill_attention(
            q,
            k,
            v,
            cu_seqlens_q=pm.query_start_loc,
            cu_seqlens_k=pm.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=pm.max_query_len,
            softmax_scale=self.scale,
            causal=False,
            return_lse=True,
            **self._quant_kw,
        )


class ModelOptMLAImpl(TritonMLAImpl):
    """TRITON_MLA impl adapter routing prefill and decode to ModelOpt kernels.

    Instances are created by :func:`clone_mla_impl` (state-preserving reclass);
    ``quant_kw`` (an :class:`_MLAQuantKw`) and ``sparse_kw`` are stashed by the
    installer before the impl is published on the layer.
    """

    quant_kw: _MLAQuantKw
    sparse_kw: dict[str, Any]

    def do_kv_cache_update(self, kv_c_normed, k_pe, *args, **kwargs):
        """Write-once latent QDQ, applied here (cache write) not in the module.

        vLLM calls this before ``forward_impl`` and passes ``forward_impl`` the
        same latent tensors. Quantizing here — **out of place**, so the caller's
        tensors stay bf16 — writes a quantized cache (read by decode) while the
        prefill ``kv_b_proj`` projection still consumes bf16 latent. That makes
        the prefill K/V operands single-quant (quantized once in the prefill
        kernel) instead of double-quant. Quantizer refs are stashed by the
        installer; absent/disabled quantizers pass through unchanged.

        Note: cached-context prefill chunks gather from this quantized cache and
        re-quantize the projected operands, so they remain double-quant — that
        is inherent to reading a stored quantized latent and is not addressed
        here.
        """
        kv_q = getattr(self, "_kv_c_quantizer", None)
        kpe_q = getattr(self, "_k_pe_quantizer", None)
        if kv_q is not None and getattr(kv_q, "is_enabled", False):
            kv_c_normed = kv_q(kv_c_normed)
        if kpe_q is not None and getattr(kpe_q, "is_enabled", False):
            k_pe = kpe_q(k_pe)
        return super().do_kv_cache_update(kv_c_normed, k_pe, *args, **kwargs)

    def _get_prefill_backend(self, prefill_metadata) -> _ModelOptMLAPrefillBackend:
        backend = self.__dict__.get("_modelopt_prefill_backend")
        if backend is None:
            backend = _ModelOptMLAPrefillBackend(
                prefill_metadata.prefill_backend,
                quant_kw=self.quant_kw.prefill,
                sparse_kw=self.sparse_kw,
            )
            self._modelopt_prefill_backend = backend
        return backend

    def forward_mha(
        self,
        q,
        kv_c_normed,
        k_pe,
        kv_c_and_k_pe_cache,
        attn_metadata,
        k_scale,
        output,
        output_scale=None,
    ) -> None:
        """Run MLA prefill with the ModelOpt prefill backend swapped in."""
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        if not self.quant_kw.any_active and not self.sparse_kw:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        backend = self._get_prefill_backend(prefill_metadata)
        backend.prepare_metadata(prefill_metadata)
        saved = prefill_metadata.prefill_backend
        prefill_metadata.prefill_backend = backend
        try:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )
        finally:
            prefill_metadata.prefill_backend = saved

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        """Run absorbed-MLA decode through the ModelOpt split-K kernel."""
        query_in_kernel = getattr(layer, "_query_quant_in_kernel", False)
        if self.quant_kw.decode["p_qdq"] is None and not query_in_kernel:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if query_in_kernel:
            # FP32 QDQ carrier on the absorbed query: the hardware-faithful
            # emulation of the decode BMM1 A-operand.
            q = layer.q_bmm_quantizer(q.float())
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        return mla_attention_decode(
            q,
            kv_c_and_k_pe_cache,
            decode_meta.block_table,
            decode_meta.seq_lens,
            softmax_scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            num_kv_splits=_MLA_DECODE_NUM_KV_SPLITS,
            out_dtype=kv_c_and_k_pe_cache.dtype,
            return_lse=True,
            **self.quant_kw.decode,
        )


def select_mla_impl_cls(impl) -> type | None:
    """Return the ModelOpt MLA adapter class matching a native implementation."""
    if isinstance(impl, ModelOptMLAImpl):
        return type(impl)
    if isinstance(impl, TritonMLAImpl):
        return ModelOptMLAImpl
    return None


def clone_mla_impl(old_impl) -> ModelOptMLAImpl:
    """Create the MLA adapter while preserving vLLM's initialized impl state."""
    new_cls = select_mla_impl_cls(old_impl)
    if new_cls is None:
        raise TypeError(
            f"MLA backend {type(old_impl).__name__} is not supported; launch vLLM with "
            "--attention-backend TRITON_MLA"
        )
    new_impl = object.__new__(new_cls)
    new_impl.__dict__.update(vars(old_impl))
    # A re-install over an existing adapter must not inherit the cached prefill
    # backend, which froze the previous install's quant/sparse kwargs.
    new_impl.__dict__.pop("_modelopt_prefill_backend", None)
    return new_impl
