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

Installs backend-matched vLLM attention implementations that use the ModelOpt
Triton kernel with paged KV cache support. Integration approach:

- No module replacement — the Attention module stays intact with all its state
- Only ``impl`` is swapped to the matching FlashAttention or FlashInfer adapter
- KV cache update follows the selected backend's native version-specific contract
- ``forward()`` calls ModelOpt Triton only when a validated transform is active

Vllm-free config helpers (``match_sparse_config`` / ``load_from_checkpoint_metadata``)
live in ``plugins/sparse_attn_config.py`` and are unit-testable without vLLM.
"""

import functools
import inspect
import math
import warnings
from dataclasses import dataclass

import torch
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

from modelopt.torch.kernels.common.attention.decode_attention import (
    attention_decode as triton_decode_attention,
)
from modelopt.torch.kernels.common.attention.triton_fa import attention as triton_attention
from modelopt.torch.kernels.quantization.attention.v_qdq import fake_quant_v_onwrite


def _target_sparse_ratio_for_phase(target_sparse_ratio, phase: str) -> float:
    """Return target sparsity for a phase, defaulting old checkpoint metadata."""
    if isinstance(target_sparse_ratio, float | int):
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
    """Map an enabled BMM2 quantizer to the kernel's NVFP4 mode and scalar amax."""
    quantizer = getattr(layer, attr, None)
    if quantizer is None or not getattr(quantizer, "is_enabled", False):
        return None, default_amax
    if (
        not getattr(quantizer, "is_nvfp4_dynamic", False)
        or (quantizer.block_sizes or {}).get(-1) != 16
    ):
        raise NotImplementedError(
            f"{attr} is enabled with an unsupported format; only dynamic block-16 NVFP4 is supported"
        )
    amax = getattr(quantizer, "_amax", None)
    if amax is None:
        return "nvfp4", default_amax
    if getattr(amax, "numel", lambda: 1)() != 1:
        raise NotImplementedError(f"{attr} requires a scalar amax, got shape {tuple(amax.shape)}")
    return "nvfp4", float(amax)


def _p_qdq_from_layer(layer) -> tuple[str | None, float]:
    return _bmm_qdq_from_layer(layer, "p_bmm_quantizer", 1.0)


def _v_qdq_from_layer(layer) -> tuple[str | None, float | None]:
    return _bmm_qdq_from_layer(layer, "v_bmm_quantizer", None)


def _quant_kw_from_impl(impl, layer):
    """Resolve the compact P/V QDQ contract once for one attention launch."""
    quant_kw = getattr(impl, "quant_kw", None)
    if quant_kw is None:
        p_qdq, p_qdq_amax = _p_qdq_from_layer(layer)
        v_qdq, v_qdq_amax = _v_qdq_from_layer(layer)
    else:
        p_qdq, p_qdq_amax = quant_kw["p_qdq"], quant_kw["p_qdq_amax"]
        v_qdq, v_qdq_amax = quant_kw["v_qdq"], quant_kw["v_qdq_amax"]
    return p_qdq, p_qdq_amax, v_qdq, v_qdq_amax


def _any_quant_active(layer, p_qdq, v_qdq) -> bool:
    """Return whether native fallback would omit any Q/K/P/V transform."""
    k_quantizer = getattr(layer, "k_bmm_quantizer", None)
    return bool(
        p_qdq
        or v_qdq
        or getattr(layer, "_query_quant_in_kernel", False)
        or getattr(k_quantizer, "is_enabled", False)
    )


def _should_run_modelopt_kernel(sparse_kw, quant_active: bool, force_kernel: bool) -> bool:
    """Return whether a launch has effective work for the ModelOpt kernel."""
    return bool(sparse_kw or quant_active or force_kernel)


def _mixed_batch_needs_dense_base(quant_active: bool, force_kernel: bool) -> bool:
    """Return whether sparse-only mixed phases need one native output seed."""
    return not quant_active and not force_kernel


@dataclass(frozen=True, slots=True)
class _ResolvedForward:
    p_qdq: str | None
    p_qdq_amax: float
    v_qdq: str | None
    v_qdq_amax: float | None
    quant_active: bool
    force_kernel: bool


def _resolve_forward(
    impl,
    layer,
    attn_metadata,
    output_scale,
    output_block_scale,
    *,
    require_flashinfer_metadata: bool = False,
) -> _ResolvedForward | None:
    """Resolve shared transform state or request the backend's native path."""
    p_qdq, p_qdq_amax, v_qdq, v_qdq_amax = _quant_kw_from_impl(impl, layer)
    quant_active = _any_quant_active(layer, p_qdq, v_qdq)
    force_kernel = getattr(layer, "_modelopt_force_kernel", False)
    transform_active = _should_run_modelopt_kernel(
        getattr(impl, "sparse_kw", None), quant_active, force_kernel
    )

    if getattr(attn_metadata, "use_cascade", False):
        if transform_active:
            raise NotImplementedError(
                "vLLM cascade attention is incompatible with an active ModelOpt attention transform"
            )
        return None

    if require_flashinfer_metadata:
        missing = [name for name in _FLASHINFER_METADATA_FIELDS if not hasattr(attn_metadata, name)]
        if missing:
            if transform_active:
                raise NotImplementedError(
                    "FlashInfer metadata is missing the ModelOpt attention transform "
                    f"fields: {', '.join(missing)}"
                )
            return None

    if transform_active and (output_scale is not None or output_block_scale is not None):
        raise NotImplementedError("Fused attention output quantization is unsupported")

    return _ResolvedForward(
        p_qdq=p_qdq,
        p_qdq_amax=p_qdq_amax,
        v_qdq=v_qdq,
        v_qdq_amax=v_qdq_amax,
        quant_active=quant_active,
        force_kernel=force_kernel,
    )


# Resolution guards raw configured transforms; dispatch rechecks effective
# sparse work after calibration and decode-only pruning.
def _forward_modelopt(
    impl,
    *,
    layer,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    is_causal: bool,
    output: torch.Tensor,
    p_qdq: str | None,
    p_qdq_amax: float,
    v_qdq: str | None,
    v_qdq_amax: float | None,
    quant_active: bool,
    force_kernel: bool,
    dense_fallback,
    prepare_modelopt=None,
) -> torch.Tensor:
    """Run the compact ModelOpt path over a backend-normalized paged cache."""
    batch = seq_lens.shape[0]
    b_start_loc = cu_seqlens_q[:batch]
    b_seq_len = cu_seqlens_q[1 : batch + 1] - cu_seqlens_q[:batch]
    is_decode_only = max_query_len <= 1
    page_size = key_cache.shape[1]

    sparse_kw = dict(getattr(impl, "sparse_kw", {}))
    _resolve_skip_softmax_calibration(
        sparse_kw,
        is_prefill=not is_decode_only,
        max_seq_len=max_seq_len,
    )
    if is_decode_only:
        # N:M sparse softmax is prefill-only.
        for name in ("sparsity_n", "sparsity_m", "dense_sink_tokens", "dense_recent_tokens"):
            sparse_kw.pop(name, None)
    if not _should_run_modelopt_kernel(
        sparse_kw,
        quant_active,
        force_kernel,
    ):
        # Dynamic calibration can disable sparse work for a launch. Preserve the
        # backend's native dense path when no ModelOpt transform remains active.
        # ``_modelopt_force_kernel`` (isolation knob) keeps the bf16 kernel path
        # even with all quant disabled, so the on/off PPL diff holds the kernel
        # constant.
        return dense_fallback()
    if prepare_modelopt is not None:
        prepare_modelopt()

    v_cache_quantized = v_qdq == "nvfp4"
    if v_cache_quantized:
        v_qdq_scale = 1.0 if v_qdq_amax is None else v_qdq_amax / (6.0 * 448.0)
        if not (math.isfinite(v_qdq_scale) and v_qdq_scale > 0):
            raise ValueError(f"v_bmm_quantizer amax must be finite and positive, got {v_qdq_amax}")
        prev = seq_lens - b_seq_len
        fake_quant_v_onwrite(
            value_cache,
            block_table,
            (prev // 16) * 16,
            (seq_lens // 16) * 16,
            max_new_tokens=max_query_len,
            page_size=page_size,
            v_qdq_scale=v_qdq_scale,
            decode=is_decode_only,
        )

    q = query[:num_actual_tokens].contiguous()
    if getattr(layer, "_query_quant_in_kernel", False):
        valid_q = torch.arange(q.shape[0], device=q.device) < cu_seqlens_q[-1]
        q = q.masked_fill(~valid_q[:, None, None], 0)
        q = layer.q_bmm_quantizer(q.float())
    use_split_k_decode = (
        is_decode_only
        and "skip_softmax_threshold" not in sparse_kw
        and (p_qdq == "nvfp4" or v_qdq == "nvfp4")
    )
    if use_split_k_decode:
        triton_out = triton_decode_attention(
            q[:batch],
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=impl.scale,
            page_size=page_size,
            p_qdq=p_qdq,
            p_qdq_amax=p_qdq_amax,
            v_qdq=v_qdq,
            v_qdq_amax=v_qdq_amax,
            v_cache_quantized=v_cache_quantized,
        )
        output[:batch] = triton_out
        return output

    # Paged mode reads K/V through the cache. The dummy shape provides the GQA ratio.
    k_dummy = torch.empty(0, impl.num_kv_heads, impl.head_size, device=q.device, dtype=q.dtype)
    triton_out = triton_attention(
        q,
        k=k_dummy,
        v=k_dummy,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=max_query_len,
        is_causal=is_causal,
        softmax_scale=impl.scale,
        b_start_loc_k=None,
        b_seq_len_k=seq_lens,
        max_input_len_k=max_seq_len,
        k_cache=key_cache,
        v_cache=value_cache,
        block_table=block_table,
        page_size=page_size,
        p_qdq=p_qdq,
        p_qdq_amax=p_qdq_amax,
        v_qdq=v_qdq,
        v_qdq_amax=v_qdq_amax,
        v_cache_quantized=v_cache_quantized,
        **sparse_kw,
    )
    output[:num_actual_tokens] = triton_out
    return output


class ModelOptSparseAttentionImpl(FlashAttentionImpl):
    """FlashAttention adapter for the compact ModelOpt Triton path."""

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
            return output.fill_(0)

        def native_forward():
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

        resolved = _resolve_forward(
            self,
            layer,
            attn_metadata,
            output_scale,
            output_block_scale,
        )
        if resolved is None:
            return native_forward()

        key_cache, value_cache = kv_cache.unbind(0)
        is_decode_only = attn_metadata.max_query_len <= 1
        return _forward_modelopt(
            self,
            layer=layer,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            cu_seqlens_q=attn_metadata.query_start_loc,
            num_actual_tokens=attn_metadata.num_actual_tokens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=attn_metadata.max_seq_len,
            is_causal=getattr(attn_metadata, "causal", not is_decode_only),
            output=output,
            p_qdq=resolved.p_qdq,
            p_qdq_amax=resolved.p_qdq_amax,
            v_qdq=resolved.v_qdq,
            v_qdq_amax=resolved.v_qdq_amax,
            quant_active=resolved.quant_active,
            force_kernel=resolved.force_kernel,
            dense_fallback=native_forward,
        )


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


_FLASHINFER_PATCHED = False
_FLASHINFER_IMPL_CLS: type | None = None
_FLASHINFER_METADATA_FIELDS = {
    "_modelopt_block_table": "block_table_tensor",
    "_modelopt_seq_lens": "seq_lens",
    "_modelopt_query_start_loc": "query_start_loc",
    "_modelopt_num_actual_tokens": "num_actual_tokens",
    "_modelopt_max_query_len": "max_query_len",
    "_modelopt_max_seq_len": "max_seq_len",
    "_modelopt_causal": "causal",
}


def _reset_flashinfer_state_for_tests() -> None:
    """Clear lazy state without unwrapping the process-wide builder patch."""
    global _FLASHINFER_PATCHED, _FLASHINFER_IMPL_CLS
    _FLASHINFER_PATCHED = False
    _FLASHINFER_IMPL_CLS = None


def patch_flashinfer_metadata_builder() -> bool:
    """Attach the common paged metadata needed by the ModelOpt kernels."""
    global _FLASHINFER_PATCHED
    if _FLASHINFER_PATCHED:
        return True
    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    except ImportError:
        return False

    orig_build = FlashInferMetadataBuilder.build
    if getattr(orig_build, "_modelopt_sparse_metadata_patch", False):
        _FLASHINFER_PATCHED = True
        return True
    # vLLM compatibility contract: build has a named ``common_attn_metadata``
    # argument and returns a mutable metadata object that accepts attached fields.
    build_sig = inspect.signature(orig_build)

    @functools.wraps(orig_build)
    def build(*args, **kwargs):
        metadata = orig_build(*args, **kwargs)
        common = build_sig.bind(*args, **kwargs).arguments["common_attn_metadata"]
        for target, source in _FLASHINFER_METADATA_FIELDS.items():
            setattr(metadata, target, getattr(common, source))
        return metadata

    setattr(build, "_modelopt_sparse_metadata_patch", True)
    FlashInferMetadataBuilder.build = build
    _FLASHINFER_PATCHED = True
    return True


def _flashinfer_cache_write(layer, key, value, kv_cache, attn_metadata, impl) -> None:
    """Issue FlashInfer's native paged K/V cache write."""
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        kv_cache[:, 0],
        kv_cache[:, 1],
        attn_metadata.slot_mapping,
        impl.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )


def _maybe_update_flashinfer_cache(layer, key, value, kv_cache, attn_metadata, impl) -> None:
    """Write K/V when the selected vLLM release performs updates in forward."""
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    if not getattr(FlashInferBackend, "forward_includes_kv_cache_update", True):
        return
    if getattr(impl, "kv_sharing_target_layer_name", None) is not None:
        return
    _flashinfer_cache_write(layer, key, value, kv_cache, attn_metadata, impl)


def _flashinfer_forward(
    impl,
    native_forward,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=None,
    output_scale=None,
    output_block_scale=None,
):
    """Run the FlashInfer adapter with module-scope, directly testable logic."""
    assert output is not None, "Output tensor must be provided."
    if attn_metadata is None:
        return output.fill_(0)

    dense_output = None
    cache_prepared = False

    def dense_fallback():
        nonlocal cache_prepared, dense_output
        if dense_output is None:
            dense_output = native_forward(
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
            cache_prepared = True
        return dense_output

    def prepare_modelopt():
        nonlocal cache_prepared
        if not cache_prepared:
            _maybe_update_flashinfer_cache(layer, key, value, kv_cache, attn_metadata, impl)
            cache_prepared = True

    resolved = _resolve_forward(
        impl,
        layer,
        attn_metadata,
        output_scale,
        output_block_scale,
        require_flashinfer_metadata=True,
    )
    if resolved is None:
        return dense_fallback()

    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "FlashInfer KV cache must have logical shape [blocks, 2, page, heads, dim]"
        )

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    max_query_len = attn_metadata._modelopt_max_query_len
    is_decode_only = max_query_len <= 1
    common_kw = {
        "layer": layer,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "max_seq_len": attn_metadata._modelopt_max_seq_len,
        "is_causal": getattr(attn_metadata, "_modelopt_causal", not is_decode_only),
        "p_qdq": resolved.p_qdq,
        "p_qdq_amax": resolved.p_qdq_amax,
        "v_qdq": resolved.v_qdq,
        "v_qdq_amax": resolved.v_qdq_amax,
        "quant_active": resolved.quant_active,
        "force_kernel": resolved.force_kernel,
        "dense_fallback": dense_fallback,
        "prepare_modelopt": prepare_modelopt,
    }
    num_decodes = getattr(attn_metadata, "num_decodes", 0)
    num_prefills = getattr(attn_metadata, "num_prefills", 0)
    if not (num_decodes and num_prefills):
        return _forward_modelopt(
            impl,
            query=query,
            block_table=attn_metadata._modelopt_block_table,
            seq_lens=attn_metadata._modelopt_seq_lens,
            cu_seqlens_q=attn_metadata._modelopt_query_start_loc,
            num_actual_tokens=attn_metadata._modelopt_num_actual_tokens,
            max_query_len=max_query_len,
            output=output,
            **common_kw,
        )

    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefill_tokens = attn_metadata.num_prefill_tokens
    if num_decode_tokens % num_decodes:
        raise NotImplementedError("Non-uniform mixed FlashInfer decode is unsupported")
    if num_decode_tokens + num_prefill_tokens != attn_metadata._modelopt_num_actual_tokens:
        raise ValueError("FlashInfer mixed-batch token counts do not match common metadata")

    # Sparse-only launches may have an inactive phase (for example N:M
    # is prefill-only). Compute the native result once, then overwrite
    # each active phase with its ModelOpt result.
    if _mixed_batch_needs_dense_base(
        resolved.quant_active,
        resolved.force_kernel,
    ):
        dense_fallback()

    block_table = attn_metadata._modelopt_block_table
    seq_lens = attn_metadata._modelopt_seq_lens
    cu_seqlens_q = attn_metadata._modelopt_query_start_loc
    _forward_modelopt(
        impl,
        query=query[:num_decode_tokens],
        block_table=block_table[:num_decodes],
        seq_lens=seq_lens[:num_decodes],
        cu_seqlens_q=cu_seqlens_q[: num_decodes + 1],
        num_actual_tokens=num_decode_tokens,
        max_query_len=num_decode_tokens // num_decodes,
        output=output[:num_decode_tokens],
        **common_kw,
    )
    prefill_start = num_decode_tokens
    prefill_cu_seqlens_q = cu_seqlens_q[num_decodes:] - cu_seqlens_q[num_decodes]
    _forward_modelopt(
        impl,
        query=query[prefill_start : prefill_start + num_prefill_tokens],
        block_table=block_table[num_decodes : num_decodes + num_prefills],
        seq_lens=seq_lens[num_decodes : num_decodes + num_prefills],
        cu_seqlens_q=prefill_cu_seqlens_q,
        num_actual_tokens=num_prefill_tokens,
        max_query_len=max_query_len,
        output=output[prefill_start : prefill_start + num_prefill_tokens],
        **common_kw,
    )
    return output


def get_flashinfer_sparse_impl_cls() -> type:
    """Return the lazy FlashInfer adapter without requiring it for FA users."""
    global _FLASHINFER_IMPL_CLS
    if _FLASHINFER_IMPL_CLS is not None:
        return _FLASHINFER_IMPL_CLS

    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    class ModelOptSparseFlashInferImpl(FlashInferImpl):
        """FlashInfer adapter for the compact ModelOpt Triton path."""

        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=None,
            output_scale=None,
            output_block_scale=None,
        ):
            return _flashinfer_forward(
                self,
                super().forward,
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

    _FLASHINFER_IMPL_CLS = ModelOptSparseFlashInferImpl
    return _FLASHINFER_IMPL_CLS


def select_sparse_impl_cls(impl) -> type | None:
    """Return the ModelOpt adapter matching a native vLLM implementation."""
    if isinstance(impl, ModelOptSparseAttentionImpl):
        return None
    if _FLASHINFER_IMPL_CLS is not None and isinstance(impl, _FLASHINFER_IMPL_CLS):
        return None
    if isinstance(impl, FlashAttentionImpl):
        return ModelOptSparseAttentionImpl
    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    except ImportError:
        return None
    if isinstance(impl, FlashInferImpl) and patch_flashinfer_metadata_builder():
        return get_flashinfer_sparse_impl_cls()
    return None


def _clone_sparse_impl(old_impl, new_cls=None):
    """Create a sparse impl while preserving vLLM's initialized runtime state."""
    if new_cls is None:
        new_cls = select_sparse_impl_cls(old_impl)
    if new_cls is None:
        raise TypeError(f"Unsupported vLLM attention implementation: {type(old_impl).__name__}")
    if getattr(old_impl, "sinks", None) is not None:
        raise NotImplementedError(f"{new_cls.__name__} does not support attention sinks yet.")

    try:
        old_state = vars(old_impl)
    except TypeError as err:
        raise TypeError(
            "Cannot clone vLLM attention impl state: old impl does not expose __dict__."
        ) from err

    new_impl = object.__new__(new_cls)
    new_impl.__dict__.update(old_state)
    return new_impl
