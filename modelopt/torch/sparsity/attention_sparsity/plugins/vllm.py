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

from modelopt.torch.kernels.common.attention.triton_fa import attention as triton_attention
from modelopt.torch.kernels.sparsity.attention.calibrate import attention_calibrate


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


class _SparseCalibrationMixin:
    """Backend-agnostic skip-softmax calibration shared by the sparse impls.

    A backend-specific impl extracts the dense paged metadata (per-request query
    offsets/lengths, KV lengths, block table) and the K/V caches from its own
    attention-metadata and cache layout, then calls :meth:`_forward_calibrate`.
    The per-request measurement, dense-output write, and stats recording are
    identical across backends (FlashAttention, FlashInfer, ...), so only the
    extraction differs. ``iter_sparse_impls`` recognizes any impl that mixes
    this in.

    Calibration state (``_calibrate``, ``_calib_threshold_trials``,
    ``_calib_records``) is attached by :func:`enable_calibration`.
    """

    # Provided at runtime by the vLLM AttentionImpl base class.
    scale: float
    num_kv_heads: int
    head_size: int
    # Attached by enable_calibration().
    _calib_threshold_trials: list[float] | None
    _calib_records: list[dict]

    def _forward_calibrate(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_size: int,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_actual_tokens: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Measure per-request tile-skip stats via the paged Triton calibration kernel.

        Each scheduled request is calibrated independently (batch=1) so its KV
        length is the per-sample length the exponential fit needs, and so the
        kernel keeps the uniform-length contract it was validated against. The
        kernel computes full attention, so ``output`` is written densely and the
        forward pass is numerically unchanged.

        Phase and causality are decided per request: ``q_len == 1`` is a decode
        step (full-cache, non-causal); ``q_len > 1`` is (chunked) prefill (causal
        — the kernel offsets the query into the KV span). A mixed prefill/decode
        batch therefore contributes correctly to both phase fits.
        """
        trials = self._calib_threshold_trials
        batch = seq_lens.shape[0]

        q = query[:num_actual_tokens].contiguous()
        # Dummy K/V: in paged mode KV is read from the cache via block_table.
        # Only shape[1] (num_kv_heads) is consulted, to compute the GQA ratio.
        k_dummy = torch.empty(0, self.num_kv_heads, self.head_size, device=q.device, dtype=q.dtype)

        for i in range(batch):
            q_len = int(b_seq_len[i].item())
            if q_len <= 0:
                continue
            q_start = int(b_start_loc[i].item())
            seq_k = int(seq_lens[i].item())
            phase = "decode" if q_len <= 1 else "prefill"

            oi, counters = attention_calibrate(
                q[q_start : q_start + q_len],
                k_dummy,
                k_dummy,
                b_start_loc=torch.zeros(1, device=q.device, dtype=torch.int32),
                b_seq_len=b_seq_len[i : i + 1].to(torch.int32),
                max_input_len=q_len,
                is_causal=q_len > 1,
                softmax_scale=self.scale,
                b_seq_len_k=seq_lens[i : i + 1].to(torch.int32),
                max_input_len_k=seq_k,
                threshold_trials=trials,
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=block_table[i : i + 1],
                page_size=page_size,
            )
            output[q_start : q_start + q_len] = oi

            total = counters[:, 0].float()
            skipped = counters[:, 1].float()
            sparsity = (skipped / total.clamp(min=1)).tolist()
            self._calib_records.append(
                {"phase": phase, "sample_length": seq_k, "sparsity": sparsity}
            )

        return output


class ModelOptSparseAttentionImpl(_SparseCalibrationMixin, FlashAttentionImpl):
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

        # Calibration mode: measure multi-threshold tile-skip statistics with the
        # Triton calibration kernel (full attention + counting) instead of running
        # the sparse inference kernel. Output stays dense so generation proceeds
        # normally and decode-step calibration sees a correct cache.
        if getattr(self, "_calibrate", False) and getattr(self, "_calib_threshold_trials", None):
            return self._forward_calibrate(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                page_size=page_size,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                seq_lens=seq_lens,
                block_table=attn_metadata.block_table,
                num_actual_tokens=num_actual_tokens,
                output=output,
            )

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
            if set(sparse_kw) <= {"skip_softmax_threshold"}:
                # The current ModelOpt paged kernel is only validated for
                # sparse prefill in vLLM. Decode-only skip-softmax would route
                # through the dense Triton path for every non-skipped tile, so
                # keep decode on vLLM FlashAttention until that path is covered.
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
        if not sparse_kw:
            # Dynamic calibration can disable sparse work for a launch, e.g.
            # short-prefill thresholds outside the valid lambda range. Avoid
            # swapping in the ModelOpt dense kernel when no sparse feature is active.
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


def _clone_sparse_impl(old_impl, new_cls: type = ModelOptSparseAttentionImpl):
    """Re-class a vLLM attention impl into ``new_cls``, preserving its state.

    The new impl shares the backend impl's initialized runtime state (config,
    scales, kv-cache dtype) so ``do_kv_cache_update`` and the dense-fallback
    ``super().forward()`` keep working. ``new_cls`` selects the backend-specific
    sparse impl (FlashAttention vs FlashInfer).
    """
    if getattr(old_impl, "sinks", None) is not None:
        # vLLM passes sinks to FlashAttention as s_aux; our Triton path does not support sinks yet.
        raise NotImplementedError(
            f"{new_cls.__name__} does not support vLLM FlashAttention sinks yet."
        )

    try:
        old_state = vars(old_impl)
    except TypeError as err:
        raise TypeError(
            "Cannot clone vLLM attention impl state: old impl does not expose __dict__."
        ) from err

    new_impl = object.__new__(new_cls)
    new_impl.__dict__.update(old_state)
    return new_impl


# ---------------------------------------------------------------------------
# FlashInfer backend support
# ---------------------------------------------------------------------------
# FlashInfer's per-step metadata only retains planned wrappers, not the dense
# block_table / seq_lens / query_start_loc the calibration kernel needs. Those
# live on the CommonAttentionMetadata the builder consumes, so we stash them onto
# the produced FlashInferMetadata (``_modelopt_*``) and read them back in
# forward. The KV cache is ``[num_blocks, 2, page_size, num_kv_heads, head_dim]``
# (``[:, 0]`` = K, ``[:, 1]`` = V); strides are passed through, so this is correct
# for both NHD and HND physical layouts.

_FLASHINFER_PATCHED = False
_FLASHINFER_IMPL_CLS: type | None = None


def patch_flashinfer_metadata_builder() -> bool:
    """Stash the dense common metadata onto ``FlashInferMetadata`` at build time.

    Idempotent. Returns ``True`` if the FlashInfer builder is now patched,
    ``False`` if the FlashInfer backend is unavailable.
    """
    global _FLASHINFER_PATCHED
    if _FLASHINFER_PATCHED:
        return True
    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    except ImportError:
        return False

    orig_build = FlashInferMetadataBuilder.build

    def build(self, common_attn_metadata, *args, **kwargs):
        metadata = orig_build(self, common_attn_metadata, *args, **kwargs)
        metadata._modelopt_block_table = common_attn_metadata.block_table_tensor
        metadata._modelopt_seq_lens = common_attn_metadata.seq_lens
        metadata._modelopt_query_start_loc = common_attn_metadata.query_start_loc
        metadata._modelopt_num_actual_tokens = common_attn_metadata.num_actual_tokens
        return metadata

    FlashInferMetadataBuilder.build = build
    _FLASHINFER_PATCHED = True
    return True


def get_flashinfer_sparse_impl_cls() -> type:
    """Build (once) and return ``ModelOptSparseFlashInferImpl``.

    Defined lazily so importing this module does not require the FlashInfer
    backend (and its ``flashinfer`` dependency) to be installed.
    """
    global _FLASHINFER_IMPL_CLS
    if _FLASHINFER_IMPL_CLS is not None:
        return _FLASHINFER_IMPL_CLS

    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    class ModelOptSparseFlashInferImpl(_SparseCalibrationMixin, FlashInferImpl):
        """FlashInfer attention impl with ModelOpt skip-softmax calibration.

        Outside calibration it is a plain ``FlashInferImpl`` (delegates to
        ``super().forward``). In calibration mode it measures multi-threshold
        tile-skip stats over the paged cache with the Triton calibration kernel,
        using the dense metadata stashed by ``patch_flashinfer_metadata_builder``.
        """

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
            """Calibrate when enabled, otherwise run native FlashInfer."""
            calibrating = getattr(self, "_calibrate", False) and getattr(
                self, "_calib_threshold_trials", None
            )
            # Fall back to native FlashInfer for: inference, profiling
            # (attn_metadata is None), cascade, or an unpatched builder.
            if (
                not calibrating
                or attn_metadata is None
                or getattr(attn_metadata, "use_cascade", False)
                or not hasattr(attn_metadata, "_modelopt_block_table")
            ):
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

            assert output is not None, "Output tensor must be provided."
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            seq_lens = attn_metadata._modelopt_seq_lens
            cu_seqlens_q = attn_metadata._modelopt_query_start_loc
            batch = seq_lens.shape[0]
            return self._forward_calibrate(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                page_size=key_cache.shape[1],
                b_start_loc=cu_seqlens_q[:batch],
                b_seq_len=cu_seqlens_q[1 : batch + 1] - cu_seqlens_q[:batch],
                seq_lens=seq_lens,
                block_table=attn_metadata._modelopt_block_table,
                num_actual_tokens=attn_metadata._modelopt_num_actual_tokens,
                output=output,
            )

    _FLASHINFER_IMPL_CLS = ModelOptSparseFlashInferImpl
    return _FLASHINFER_IMPL_CLS


# ---------------------------------------------------------------------------
# Calibration driver helpers
# ---------------------------------------------------------------------------
# These run skip-softmax calibration *through* the vLLM integration: the model
# is loaded under vLLM with ModelOptSparseAttentionImpl on each attention layer
# (see examples/vllm_serve/sparse_attn_worker.py), calibration mode is turned on,
# a few prompts are generated, and the collected per-threshold tile-skip counts
# are fit to the same exponential model (a, b) the HF path produces — so the
# result drops straight into the existing export/inference path.


def iter_sparse_impls(model):
    """Yield every ModelOpt sparse attention impl reachable from a vLLM model.

    Walks ``model.named_modules()`` and returns the swapped ``impl`` of each
    attention layer (any backend — FlashAttention, FlashInfer — that mixes in
    ``_SparseCalibrationMixin``). Used by the calibration driver to toggle
    calibration mode and harvest stats without knowing vLLM's module layout.
    """
    for _, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if isinstance(impl, _SparseCalibrationMixin):
            yield impl


def enable_calibration(impls, threshold_trials: list[float]) -> None:
    """Put a set of sparse impls into calibration mode and clear prior records."""
    if not threshold_trials:
        raise ValueError("threshold_trials must be a non-empty list for calibration.")
    for impl in impls:
        impl._calibrate = True
        impl._calib_threshold_trials = list(threshold_trials)
        impl._calib_records = []


def disable_calibration(impls) -> None:
    """Turn off calibration mode (collected records are left intact)."""
    for impl in impls:
        impl._calibrate = False


def collect_calibration_stats(impls) -> dict[str, list[dict]]:
    """Group every impl's per-request records by phase.

    Returns a dict ``{"prefill": [...], "decode": [...]}`` where each entry is a
    ``{"sample_length", "sparsity"}`` record ready for
    :meth:`DynamicThresholdCalibrator.calibrate_from_stats`.
    """
    per_phase: dict[str, list[dict]] = {"prefill": [], "decode": []}
    for impl in impls:
        for record in getattr(impl, "_calib_records", []):
            per_phase.setdefault(record["phase"], []).append(record)
    return per_phase


def fit_calibration(
    impls,
    threshold_trials: list[float],
    *,
    fit_logspace: bool = False,
) -> dict[str, dict[str, float]]:
    """Fit the exponential skip-softmax model from collected vLLM stats.

    Reuses :class:`DynamicThresholdCalibrator` so the vLLM-calibrated ``(a, b)``
    are identical in form to the HF path and export unchanged via
    ``threshold_scale_factor``.

    Returns:
        ``{phase: {"a", "b", "min_observed_sparsity", "max_observed_sparsity"}}``
        for each phase that produced a valid fit.
    """
    from ..calibration.calibrator import DynamicThresholdCalibrator

    per_phase = collect_calibration_stats(impls)
    calibration_params: dict[str, dict[str, float]] = {}
    for phase, stats in per_phase.items():
        if not stats:
            continue
        calibrator = DynamicThresholdCalibrator(
            threshold_trials=list(threshold_trials), fit_logspace=fit_logspace
        )
        result = calibrator.calibrate_from_stats(stats, phase=phase)
        if "a" in result and "b" in result:
            params = {"a": result["a"], "b": result["b"]}
            for key in ("min_observed_sparsity", "max_observed_sparsity"):
                if key in result:
                    params[key] = result[key]
            calibration_params[phase] = params
    return calibration_params
