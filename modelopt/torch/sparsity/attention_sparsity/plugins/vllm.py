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
- ``forward()`` calls ModelOpt Triton only when a validated sparse path is active

Vllm-free config helpers (``match_sparse_config`` / ``load_from_checkpoint_metadata``)
live in ``plugins/sparse_attn_config.py`` and are unit-testable without vLLM.
"""

import math
import warnings

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from modelopt.torch.kernels.common.attention.decode_attention import (
    _ONWRITE_BLOCK_N,
    attention_decode,
    fake_quant_v_onwrite,
)
from modelopt.torch.kernels.common.attention.triton_fa import attention as triton_attention


def _target_sparse_ratio_for_phase(target_sparse_ratio, phase: str) -> float:
    """Return target sparsity for a phase, defaulting old checkpoint metadata."""
    if isinstance(target_sparse_ratio, (float, int)):
        return float(target_sparse_ratio)
    if isinstance(target_sparse_ratio, dict):
        return float(target_sparse_ratio.get(phase, 0.5))
    return 0.5


def _resolve_skip_softmax_calibration(
    sparse_kw: dict,
    *,
    is_prefill: bool,
    max_seq_len: int,
) -> None:
    """Convert exported calibration params into the scalar threshold kernel API."""
    threshold_scale_factor = sparse_kw.pop("threshold_scale_factor", None)
    sparse_target_ratio = sparse_kw.pop("target_sparse_ratio", None)
    if threshold_scale_factor is None:
        return

    phase = "prefill" if is_prefill else "decode"
    params = threshold_scale_factor.get(phase) if isinstance(threshold_scale_factor, dict) else None
    if not isinstance(params, dict):
        return

    try:
        a = float(params["a"])
        b = float(params["b"])
        seq_len = int(max_seq_len)
    except (KeyError, TypeError, ValueError):
        return
    if a <= 0.0 or seq_len <= 0:
        return

    target = _target_sparse_ratio_for_phase(sparse_target_ratio, phase)
    scale_factor = a * math.exp(b * target)
    # The current Triton kernel accepts one scalar threshold per launch. Use
    # the max KV length in the scheduled batch; shorter sequences are denser.
    threshold = scale_factor / seq_len
    if threshold >= 1.0:
        warnings.warn(
            "Disabling calibrated skip-softmax for this vLLM launch because "
            f"the derived threshold is outside the valid lambda range: "
            f"phase={phase}, seq_len={seq_len}, scale_factor={scale_factor:.6g}, "
            f"target_sparse_ratio={target:.6g}, threshold={threshold:.6g}.",
            stacklevel=2,
        )
        return
    sparse_kw["skip_softmax_threshold"] = threshold


def _build_sparse_kw(layer_cfg: dict) -> dict:
    """Convert one checkpoint layer config into kernel kwargs."""
    sparse_kw = {}
    sparsity_n = layer_cfg.get("sparsity_n", 0)
    if sparsity_n > 0:
        sparse_kw["sparsity_n"] = sparsity_n
        sparse_kw["sparsity_m"] = layer_cfg.get("sparsity_m", 4)
        sparse_kw["dense_sink_tokens"] = layer_cfg.get("dense_sink_tokens", 0)
        sparse_kw["dense_recent_tokens"] = layer_cfg.get("dense_recent_tokens", 64)

    threshold = layer_cfg.get("skip_softmax_threshold")
    if threshold is not None:
        sparse_kw["skip_softmax_threshold"] = threshold
    threshold_scale_factor = layer_cfg.get("threshold_scale_factor")
    if threshold_scale_factor is not None:
        sparse_kw["threshold_scale_factor"] = threshold_scale_factor
        sparse_kw["target_sparse_ratio"] = layer_cfg.get("target_sparse_ratio")

    return sparse_kw


def _bmm_qdq_from_layer(layer, attr: str, default_amax: float | None):
    """Map a layer's ``attr`` BMM2 quantizer to ``(qdq mode, amax)`` for the kernels.

    Mirrors ``_QuantAttention._p_qdq_mode``: per-tensor FP8 -> ``"fp8"``; dynamic NVFP4
    at block size 16 -> ``"nvfp4"``; otherwise (missing/disabled quantizer or an
    unsupported format) -> ``(None, default_amax)``. A calibrated per-tensor ``_amax``,
    if present, is forwarded so the kernel uses it instead of ``default_amax``.
    """
    q = getattr(layer, attr, None)
    if q is None or not getattr(q, "is_enabled", False):
        return None, default_amax
    if q.is_fp8:
        mode = "fp8"
    elif q.is_nvfp4_dynamic and (q.block_sizes or {}).get(-1, None) == 16:
        mode = "nvfp4"
    else:
        return None, default_amax
    amax = getattr(q, "_amax", None)
    if amax is not None:
        if amax.numel() != 1:
            raise NotImplementedError(
                f"{attr} via the sparse Triton kernel supports only a per-tensor "
                f"(scalar) amax, got shape {tuple(amax.shape)}."
            )
        return mode, float(amax)
    return mode, default_amax


def _p_qdq_from_layer(layer) -> tuple[str | None, float]:
    """P (softmax) BMM2 operand from ``p_bmm_quantizer``; default amax 1.0 (P in [0, 1])."""
    return _bmm_qdq_from_layer(layer, "p_bmm_quantizer", 1.0)


def _v_qdq_from_layer(layer) -> tuple[str | None, float | None]:
    """V (value) BMM2 operand from ``v_bmm_quantizer``.

    Default amax ``None`` -> the constant 1.0 global scale (V's dynamic per-16
    block amax carries the range).
    """
    return _bmm_qdq_from_layer(layer, "v_bmm_quantizer", None)


def _bake_v_onwrite(value_cache, block_table, seq_lens, query_lens, page_size, v_qdq_amax, decode):
    """Quantize-on-write the newly-complete BLOCK_N V tiles so decode reads them as-is.

    Bakes the tiles that completed since the previous step (``[prev, seq)`` clipped to whole
    ``_ONWRITE_BLOCK_N`` tiles) into the paged cache, NVFP4, with the same scale the kernel applies
    on read — turning decode V quant from ``O(S²)`` on-read into ``O(S)`` quantize-once. NVFP4-only;
    the caller skips it for FP8 V.
    """
    bn = _ONWRITE_BLOCK_N
    prev = (seq_lens - query_lens).to(torch.int32)
    seqk = seq_lens.to(torch.int32)
    v_lo = (prev // bn) * bn
    v_hi = (seqk // bn) * bn
    v_scale = 1.0 if v_qdq_amax is None else v_qdq_amax / (6.0 * 448.0)
    fake_quant_v_onwrite(
        value_cache,
        block_table,
        v_lo,
        v_hi,
        page_size=page_size,
        v_qdq_scale=v_scale,
        decode=decode,
    )


class ModelOptSparseAttentionImpl(FlashAttentionImpl):
    """Attention implementation that uses the ModelOpt Triton kernel.

    Inherits from FlashAttentionImpl to reuse:
    - __init__ (all configuration)
    - do_kv_cache_update (KV cache writing)
    Only overrides forward() to replace sparse prefill attention computation.
    """

    def _forward_vllm_flash_attn(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Delegate a launch back to vLLM's native FlashAttention impl."""
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

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

        if getattr(attn_metadata, "use_cascade", False):
            # vLLM cascade metadata splits the request into shared-prefix and
            # suffix pieces. The ModelOpt paged kernel consumes plain per-request
            # KV lengths, so delegate cascade launches back to vLLM's impl.
            return self._forward_vllm_flash_attn(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        cu_seqlens_q = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        batch = seq_lens.shape[0]
        b_start_loc = cu_seqlens_q[:batch]
        b_seq_len = cu_seqlens_q[1 : batch + 1] - cu_seqlens_q[:batch]

        # Standard decode schedules one query token per request. Chunked
        # prefill and mixed prefill/decode launches use the prefill path.
        is_decode_only = attn_metadata.max_query_len <= 1
        is_causal = getattr(attn_metadata, "causal", not is_decode_only)

        # Unpack paged KV cache: [2, num_blocks, page_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(0)
        page_size = key_cache.shape[1]

        # BMM2-operand quant from the layer's p/v_bmm_quantizer (config-driven, exported);
        # None when no/disabled/unsupported quantizer. P or V quant alone (no sparsity)
        # still engages the kernel below. K/Q quant is applied upstream by the
        # _QuantVLLMAttention pre-step (head_dim axis); only the keys-axis BMM2 operands
        # (P, V) are quantized in-kernel here.
        p_qdq, p_qdq_amax = _p_qdq_from_layer(layer)
        v_qdq, v_qdq_amax = _v_qdq_from_layer(layer)

        # Per-layer sparse kwargs (set by _replace_attention_impl in the worker)
        sparse_kw = dict(getattr(self, "sparse_kw", {}))
        _resolve_skip_softmax_calibration(
            sparse_kw,
            is_prefill=not is_decode_only,
            max_seq_len=attn_metadata.max_seq_len,
        )
        if is_decode_only:
            # N:M sparse softmax is prefill-only.
            for name in ("sparsity_n", "sparsity_m", "dense_sink_tokens", "dense_recent_tokens"):
                sparse_kw.pop(name, None)
            threshold = sparse_kw.get("skip_softmax_threshold")
            if threshold is None and p_qdq is None and v_qdq is None:
                # No decode sparsity and no BMM quant for this launch; use vLLM FlashAttention.
                return self._forward_vllm_flash_attn(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            # NVFP4 V: bake the newly-complete tiles on write so the decode kernel reads them
            # as-is (O(S)) instead of re-fake-quantizing the whole cache every step (O(S²)).
            # FP8 V stays on-read. Bake BEFORE the attention so it reads the baked cache.
            v_cache_quantized = v_qdq == "nvfp4"
            if v_cache_quantized:
                _bake_v_onwrite(
                    value_cache,
                    attn_metadata.block_table,
                    seq_lens,
                    b_seq_len,
                    page_size,
                    v_qdq_amax,
                    decode=True,
                )
            # Decode runs on the dedicated paged decode kernel (one query vector per
            # request, split-K): skip-softmax and/or P quant, not the prefill kernel.
            return self._forward_sparse_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                page_size=page_size,
                seq_lens=seq_lens,
                block_table=attn_metadata.block_table,
                num_actual_tokens=num_actual_tokens,
                skip_softmax_threshold=threshold,
                p_qdq=p_qdq,
                p_qdq_amax=p_qdq_amax,
                v_qdq=v_qdq,
                v_qdq_amax=v_qdq_amax,
                v_cache_quantized=v_cache_quantized,
                output=output,
            )
        if not sparse_kw and p_qdq is None and v_qdq is None:
            # No sparse feature and no BMM2 quant active (e.g. dynamic calibration
            # disabled the threshold); avoid swapping in the ModelOpt kernel.
            return self._forward_vllm_flash_attn(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # Prepare metadata for our kernel
        q = query[:num_actual_tokens].contiguous()
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
            is_causal=is_causal,
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
            p_qdq=p_qdq,
            p_qdq_amax=p_qdq_amax,
            v_qdq=v_qdq,
            v_qdq_amax=v_qdq_amax,
            **sparse_kw,
        )

        output[:num_actual_tokens] = triton_out
        # NVFP4 V: prefill attention read raw V on-read above (O(S), single pass); now bake the
        # prompt's complete tiles on write so the following decode steps inherit a pre-quantized
        # cache and stay O(S). (Chunked prefill re-reads earlier baked tiles on-read with a small
        # idempotency drift; a prefill-side V_CACHE_QUANTIZED gate would remove it — follow-up.)
        if v_qdq == "nvfp4":
            _bake_v_onwrite(
                value_cache,
                attn_metadata.block_table,
                seq_lens,
                b_seq_len,
                page_size,
                v_qdq_amax,
                decode=False,
            )
        return output

    def _forward_sparse_decode(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_size: int,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_actual_tokens: int,
        skip_softmax_threshold: float | None = None,
        p_qdq: str | None = None,
        p_qdq_amax: float = 1.0,
        v_qdq: str | None = None,
        v_qdq_amax: float | None = None,
        v_cache_quantized: bool = False,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode via the dedicated paged decode kernel (skip-softmax and/or P/V quant).

        Standard decode schedules exactly one query token per request, so the
        ``num_actual_tokens`` query rows are the per-request decode queries. The
        decode kernel computes one query vector per ``(request, head)`` over the
        paged cache (split-K over the KV sequence) and applies the same prefix-max
        skip criterion as the prefill kernel, so realized decode sparsity matches
        the calibrated ``(a, b)``. The prefill kernel would tile this single query
        token into ``BLOCK_M`` rows, wasting ~127/128 of the work.
        """
        q = query[:num_actual_tokens].contiguous()  # [batch, num_q_heads, head_dim]
        decode_out = attention_decode(
            q,
            key_cache,
            value_cache,
            block_table[:num_actual_tokens],
            seq_lens[:num_actual_tokens],
            softmax_scale=self.scale,
            skip_softmax_threshold=skip_softmax_threshold,
            page_size=page_size,
            p_qdq=p_qdq,
            p_qdq_amax=p_qdq_amax,
            v_qdq=v_qdq,
            v_qdq_amax=v_qdq_amax,
            v_cache_quantized=v_cache_quantized,
        )
        output[:num_actual_tokens] = decode_out
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


def _clone_sparse_impl(old_impl):
    """Create a sparse impl while preserving vLLM's initialized runtime state."""
    if getattr(old_impl, "sinks", None) is not None:
        # vLLM passes sinks to FlashAttention as s_aux; our Triton path does not support sinks yet.
        raise NotImplementedError(
            "ModelOptSparseAttentionImpl does not support vLLM FlashAttention sinks yet."
        )

    try:
        old_state = vars(old_impl)
    except TypeError as err:
        raise TypeError(
            "Cannot clone vLLM attention impl state: old impl does not expose __dict__."
        ) from err

    new_impl = ModelOptSparseAttentionImpl.__new__(ModelOptSparseAttentionImpl)
    new_impl.__dict__.update(old_state)
    return new_impl
