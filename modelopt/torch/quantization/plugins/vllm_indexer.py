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

"""Fake quantization of the sparse-attention indexer query, K cache and scorer in vLLM.

Covers the CSA indexer of DeepSeek-V4 and the k-pool indexer of GLM-5.3-Flash.

``indexer_k_quantizer`` fake-quantizes the key the serving kernel quantizes into the indexer K
cache, and ``indexer_q_quantizer`` the query it quantizes for scoring. vLLM only materializes both
inside fused kernels that already produced FP8, so the FP8 tensors (the cache entries written in the
current step, the query) are dequantized, fake-quantized and quantized again with the kernels' scale
rule. The names avoid the ``*[kv]_bmm_quantizer`` globs, so the KV-cache presets leave them
disabled.

GLM-5.3-Flash's DeepGEMM scorer computes ``score[q, k] = k_scale[k] * sum_h W[q, h] *
ReLU(q_h . k)``. ``indexer_scorer_kwargs_quantizer`` passes keyword arguments to that kernel, per
layer, e.g. the numerics options of a DeepGEMM build that declares them in
``MQA_LOGITS_NUMERICS``. :func:`register_indexer_scorer_kwargs_quantizer` adds quantizers that turn
their own configs into such arguments. They start disabled.
"""

import functools
import importlib
import inspect
import weakref
from collections.abc import Callable
from types import ModuleType

import torch
from vllm.forward_context import get_forward_context

from ..config import QuantizerAttributeConfig
from ..nn import (
    QuantModule,
    QuantModuleRegistry,
    TensorQuantizer,
    is_registered_quant_backend,
    register_quant_backend,
)
from .vllm import create_parallel_state
from .vllm_layer_scope import register_layer_owner, wrap_layer_callee, wrap_layer_op

__all__ = ["register_indexer_scorer_kwargs_quantizer"]

# NotImplementedError: some vLLM model packages reject unsupported platforms at import
# (e.g. glm5next on XPU).
try:
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer as VllmDeepseekV4Indexer
except (ImportError, NotImplementedError):
    VllmDeepseekV4Indexer = None

# Older vLLM releases have no DeepGEMM wrappers; GLM-5.3-Flash's indexer needs them.
try:
    from vllm.utils import deep_gemm as vllm_deep_gemm
except ImportError:
    vllm_deep_gemm = None


def _import_glm5next_indexer() -> tuple[type | None, ModuleType | None, ModuleType | None]:
    """Return GLM-5.3-Flash's ``Indexer``, the module of its indexer op and the op's ``kpool_ops``."""
    for indexer_path, op_path in (
        # vLLM main: the op module is platform-dispatched, follow it to the bound variant.
        ("vllm.models.glm5next.common.attention", "vllm.models.glm5next.sparse_indexer"),
        # vLLM 0.28
        (
            "vllm.models.glm5next.nvidia.attention",
            "vllm.model_executor.layers.sparse_attn_indexer_kpool",
        ),
    ):
        try:
            indexer_cls = importlib.import_module(indexer_path).Indexer
            op_module = importlib.import_module(op_path)
            op_module = importlib.import_module(op_module.SparseAttnIndexerKpool.__module__)
            kpool_ops = op_module.kpool_ops
        except (ImportError, AttributeError, NotImplementedError):
            continue
        return indexer_cls, op_module, kpool_ops
    return None, None, None


VllmGlm5NextIndexer, _glm5next_op_module, _glm5next_kpool_ops = _import_glm5next_indexer()

_INDEXER_FP8_MAX = 448.0
_INDEXER_K_SCALE_BYTES = 4


def _indexer_ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """Power-of-two FP8 scale used by vLLM's indexer q and K kernels (``scale_fmt="ue8m0"``)."""
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / _INDEXER_FP8_MAX)))


def _fake_quantize_fp8_rows(
    x: torch.Tensor, quantizer: TensorQuantizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize the dequantized rows ``x`` and quantize them back to E4M3 like vLLM's kernels.

    Returns the E4M3 rows and their power-of-two fp32 scales (shape ``x.shape[:-1]``).
    """
    x = quantizer(x)
    scale = _indexer_ue8m0_scale(x.abs().amax(dim=-1))
    values = (x / scale[..., None]).clamp(-_INDEXER_FP8_MAX, _INDEXER_FP8_MAX)
    return values.to(torch.float8_e4m3fn), scale


def _requantize_fp8_indexer_k_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: torch.Tensor,
    quantizer: TensorQuantizer,
) -> None:
    """Fake-quantize the FP8 indexer K cache entries at ``slot_mapping`` in place.

    ``kv_cache`` is the ``[num_blocks, block_size, head_dim + 4]`` uint8 cache: per block,
    ``block_size`` E4M3 rows then ``block_size`` fp32 power-of-two scales. Entries are dequantized
    (exactly) to fp32, quantized and re-stored with the kernels' scale rule, so the QDQ input
    carries the FP8 rounding of the fused write. Shapes stay static for CUDA graphs: rows with
    ``valid == False`` are redirected to slot 0 (vLLM's null block) and write back the bytes they
    read.
    """
    num_blocks, block_size, row_bytes = kv_cache.shape
    head_dim = row_bytes - _INDEXER_K_SCALE_BYTES
    device = kv_cache.device
    flat = kv_cache.view(num_blocks, block_size * row_bytes)

    slots = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping)).to(torch.int64)
    block = (slots // block_size)[:, None]
    pos = slots % block_size
    value_idx = pos[:, None] * head_dim + torch.arange(head_dim, device=device)
    scale_idx = (
        block_size * head_dim
        + pos[:, None] * _INDEXER_K_SCALE_BYTES
        + torch.arange(_INDEXER_K_SCALE_BYTES, device=device)
    )

    old_values = flat[block, value_idx]
    old_scales = flat[block, scale_idx]
    scale = old_scales.contiguous().view(torch.float32).squeeze(-1)
    k = old_values.view(torch.float8_e4m3fn).to(torch.float32) * scale[:, None]

    # Zero the redirected rows so they cannot influence calibration or dynamic scales.
    k = torch.where(valid[:, None], k, torch.zeros_like(k))
    new_values, new_scale = _fake_quantize_fp8_rows(k, quantizer)
    new_values = new_values.view(torch.uint8)
    new_scales = new_scale.contiguous().view(torch.uint8).view(-1, _INDEXER_K_SCALE_BYTES)

    keep = valid[:, None]
    flat[block, value_idx] = torch.where(keep, new_values, old_values)
    flat[block, scale_idx] = torch.where(keep, new_scales, old_scales)


def _native_positional_index(module: torch.nn.Module, method_name: str, name: str) -> int:
    """Index of ``name`` among the positional arguments of the framework's own ``method_name``.

    Looks past the ModelOpt mixins in the converted class's MRO so their ``*args, **kwargs``
    overrides do not hide the real signature.
    """
    for cls in type(module).__mro__:
        if method_name in cls.__dict__ and not issubclass(cls, QuantModule):
            return list(inspect.signature(cls.__dict__[method_name]).parameters).index(name) - 1
    raise AttributeError(f"{type(module).__name__} has no native {method_name}")


def _get_arg(args: tuple, kwargs: dict, name: str, pos: int):
    return kwargs[name] if name in kwargs else args[pos]


# The built-in quantizer whose ``backend_extra_args`` are the scorer kernel's keyword arguments.
_SCORER_KWARGS_QUANTIZER = "indexer_scorer_kwargs_quantizer"
_SCORER_KWARGS_BACKEND = "scorer_kwargs"
# Scorer keyword arguments of stock DeepGEMM; a build declares the others it accepts, with their
# values, in ``deep_gemm.MQA_LOGITS_NUMERICS``.
_STOCK_SCORER_KWARGS = ("logits_dtype",)


def _passed_scorer_kwargs(quantizer: TensorQuantizer) -> dict:
    """``indexer_scorer_kwargs_quantizer``: its ``backend_extra_args``, passed on as they are."""
    if quantizer.backend != _SCORER_KWARGS_BACKEND:
        raise ValueError(
            f"{_SCORER_KWARGS_QUANTIZER} needs backend '{_SCORER_KWARGS_BACKEND}', "
            f"got {quantizer.backend!r}."
        )
    return dict(quantizer.backend_extra_args or {})


# The quantizers whose configs become keyword arguments of the GLM-5.3-Flash indexer's DeepGEMM
# scorer kernel, with the functions that turn an enabled one into those arguments.
_SCORER_KWARGS_QUANTIZERS: dict[str, Callable[[TensorQuantizer], dict]] = {
    _SCORER_KWARGS_QUANTIZER: _passed_scorer_kwargs
}


def register_indexer_scorer_kwargs_quantizer(
    name: str, scorer_kwargs: Callable[[TensorQuantizer], dict]
) -> None:
    """Add the quantizer ``name`` to the vLLM indexers that are converted from now on.

    It starts disabled. Enabled, ``scorer_kwargs(quantizer)`` turns its config into keyword
    arguments of the GLM-5.3-Flash indexer's DeepGEMM scorer kernel, which every scorer call of
    that layer then gets while the quantizer quantizes; it raises for a config it cannot express.
    Other indexers reject the quantizer. The built-in ``indexer_scorer_kwargs_quantizer`` passes its
    ``backend_extra_args`` (``backend: scorer_kwargs``) on as they are.
    """
    _SCORER_KWARGS_QUANTIZERS[name] = scorer_kwargs


def _scorer_kwargs_backend(inputs: torch.Tensor, quantizer: TensorQuantizer) -> torch.Tensor:
    raise RuntimeError(
        f"backend '{_SCORER_KWARGS_BACKEND}' passes keyword arguments to the DeepGEMM indexer "
        "scorer kernel and never runs on a tensor."
    )


if not is_registered_quant_backend(_SCORER_KWARGS_BACKEND):
    register_quant_backend(_SCORER_KWARGS_BACKEND, _scorer_kwargs_backend)


def _scorer_logits_dtype(value) -> torch.dtype:
    """The scorer's ``logits_dtype``, also from its name in a config (``bfloat16``)."""
    dtype = getattr(torch, value, None) if isinstance(value, str) else value
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"The scorer's logits_dtype is float32 or bfloat16, got {value!r}.")
    return dtype


def _widen_logits(logits: torch.Tensor) -> torch.Tensor:
    """FP32 copy of ``logits`` with the same strides, which vLLM's top-k kernels read explicitly."""
    widened = torch.empty_strided(
        logits.shape, logits.stride(), dtype=torch.float32, device=logits.device
    )
    return widened.copy_(logits)


class _QuantVLLMIndexerBase(QuantModule):
    """Owner of the indexer quantizers for one vLLM indexer layout."""

    def _setup(self):
        self.indexer_q_quantizer = TensorQuantizer()
        self.indexer_k_quantizer = TensorQuantizer()
        for name in _SCORER_KWARGS_QUANTIZERS:  # opt-in, unlike the q and K quantizers
            setattr(self, name, TensorQuantizer(QuantizerAttributeConfig(enable=False)))
        self.parallel_state = create_parallel_state()

    def forward(self, *args, **kwargs):
        # The registry matches subclasses that share ``forward``; overriding it keeps the converted
        # class from matching again. vLLM's MLA wrapper holds the attention's indexer too
        # (``MLAModules(indexer=...)``), so ``replace_quant_module`` visits the indexer twice and
        # would otherwise try to convert it a second time (inconsistent MRO).
        return super().forward(*args, **kwargs)

    def modelopt_post_restore(self, prefix: str = ""):
        """Also reject restored scorer quantizers that the layout cannot apply."""
        super().modelopt_post_restore(prefix)
        self._validate_scorer_quantizers()

    def _enabled_scorer_quantizers(self) -> list[str]:
        # A quantizer registered after the conversion is missing.
        return [
            name
            for name in _SCORER_KWARGS_QUANTIZERS
            if hasattr(self, name) and getattr(self, name).is_enabled
        ]

    def _validate_scorer_quantizers(self) -> None:
        """Reject enabled scorer quantizers; the layouts that implement them override this."""
        if enabled := self._enabled_scorer_quantizers():
            raise NotImplementedError(
                f"{', '.join(enabled)}: the indexer scorer quantizers are only implemented for the "
                "GLM-5.3-Flash k-pool indexer."
            )


class _QuantVLLMDeepseekV4Indexer(_QuantVLLMIndexerBase):
    """DeepSeek-V4 indexer: fused kernels FP8-quantize the query and cache the compressed key.

    After the forward, the cache entries the compressor wrote (one per ``compress_ratio`` tokens)
    are re-quantized through ``indexer_k_quantizer`` and the returned query through
    ``indexer_q_quantizer``. vLLM uses both without the Hadamard rotation that the DeepSeek
    reference applies.
    """

    def _setup(self):
        super()._setup()
        self._positions_pos = _native_positional_index(self, "forward", "positions")
        self._indexer_weights_pos = _native_positional_index(self, "forward", "indexer_weights")

    def forward(self, *args, **kwargs):
        self._validate_scorer_quantizers()
        q, q_scale, weights = super().forward(*args, **kwargs)
        if self.indexer_k_quantizer.is_enabled:
            self._check_fp8_indexer()
            self._requantize_written_keys(_get_arg(args, kwargs, "positions", self._positions_pos))
        if self.indexer_q_quantizer.is_enabled and q is not None:  # None: short-context shortcut
            self._check_fp8_indexer()
            indexer_weights = _get_arg(args, kwargs, "indexer_weights", self._indexer_weights_pos)
            q, weights = self._requantize_query(q, weights, indexer_weights)
        return q, q_scale, weights

    def _check_fp8_indexer(self) -> None:
        if getattr(self, "use_fp4_kv", False):
            raise NotImplementedError(
                "The indexer quantizers re-quantize the FP8 indexer query and cache; serve with "
                "the default indexer_kv_dtype instead of 'mxfp4'."
            )

    def _requantize_query(
        self, q: torch.Tensor, weights: torch.Tensor, indexer_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-quantize the FP8 query; the kernel folds its power-of-two scale into ``weights``.

        The kernel returns ``indexer_weights * q_scale * softmax_scale * n_head**-0.5`` as
        ``weights`` and no ``q_scale``: divide the other factors out, round to the power of two and
        swap in the new scale.
        """
        base = indexer_weights.float() * self.softmax_scale * self.n_head**-0.5
        # A head with zero weight does not score; zero its row instead of recovering its scale.
        live = (weights != 0) & (base != 0)
        ratio = torch.where(live, weights / torch.where(live, base, 1.0), 1.0)
        old_scale = torch.exp2(torch.round(torch.log2(ratio)))
        x = torch.where(live[..., None], q.float() * old_scale[..., None], 0.0)
        q, new_scale = _fake_quantize_fp8_rows(x, self.indexer_q_quantizer)
        return q, weights * (new_scale / old_scale)

    def _requantize_written_keys(self, positions: torch.Tensor) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):  # profiling run: nothing was written
            return
        # The compress kernel skips tokens without a compressor-state slot or an indexer-cache
        # slot, and only the last token of each compression group produces an entry.
        state_slots = attn_metadata[self.compressor.state_cache.prefix].slot_mapping
        num_tokens = state_slots.shape[0]
        k_slots = attn_metadata[self.k_cache.prefix].slot_mapping[:num_tokens]
        positions = positions[:num_tokens]
        valid = (k_slots >= 0) & (state_slots >= 0) & ((positions + 1) % self.compress_ratio == 0)
        _requantize_fp8_indexer_k_cache(
            self.k_cache.kv_cache, k_slots, valid, self.indexer_k_quantizer
        )


# GLM-5.3-Flash indexers converted in this process, for the check that the K cache is written inside
# their indexer op.
_glm5next_indexers: weakref.WeakSet = weakref.WeakSet()


def _kpool_prefill_written(arguments: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Slots written by ``kpool_compress_and_write_cache``: ``loc`` where valid and unmasked."""
    loc = arguments["loc"]
    valid = loc >= 0
    if arguments.get("write_mask") is not None:
        valid = valid & arguments["write_mask"]
    if not arguments.get("write_cache", True):
        valid = torch.zeros_like(valid)
    return loc, valid


def _kpool_decode_written(arguments: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Slots written by the decode update: pool-completing tokens with valid slot and position."""
    slots = arguments["slot_mapping"].reshape(-1)
    positions = arguments["positions"].reshape(-1)
    pool_size = arguments["pool_size"]
    valid = (slots >= 0) & (positions >= 0) & (positions % pool_size == pool_size - 1)
    return slots, valid


def _requantize_written_pools(written_slots: Callable) -> Callable:
    """A K-cache writer call of an indexer: the kernel, then a re-quantization of what it wrote."""

    def call(indexer, original: Callable, bound: inspect.BoundArguments):
        kv_cache = bound.arguments["kv_cache"]
        if kv_cache.data_ptr() != indexer.k_cache.kv_cache.data_ptr():
            raise RuntimeError(
                f"The indexer op of {indexer.k_cache.prefix} writes the K cache of another layer."
            )
        out = original(*bound.args, **bound.kwargs)
        slots, valid = written_slots(bound.arguments)
        _requantize_fp8_indexer_k_cache(kv_cache, slots, valid, indexer.indexer_k_quantizer)
        return out

    return call


def _reject_k_cache_write_outside_op() -> None:
    """Reject a K-cache write outside the indexer op, which alone names the layer of the cache."""
    if any(indexer.indexer_k_quantizer.is_enabled for indexer in _glm5next_indexers):
        raise RuntimeError(
            "vLLM wrote the GLM-5.3-Flash indexer K cache outside the indexer op, so "
            "indexer_k_quantizer cannot tell whose cache it is."
        )


# GLM-5.3-Flash's K-cache writers, with the slots that a call writes.
_KPOOL_CACHE_WRITERS = {
    "kpool_compress_and_write_cache": _kpool_prefill_written,
    "kpool_decode_update_and_maybe_write_cache_batched": _kpool_decode_written,
}
# GLM-5.3-Flash's indexer op, which writes the K cache and scores and selects the keys.
_GLM5NEXT_INDEXER_OP = "sparse_attn_indexer_kpool"


def _wrap_deep_gemm_scorer(deep_gemm_utils: ModuleType, paged: bool) -> None:
    """Wrap vLLM's DeepGEMM scorer in ``deep_gemm_utils`` to apply the scorer quantizers.

    ``paged``: the decode scorer, else the prefill one. The indexer op imports the scorer from the
    module on every call.
    """
    wrap_layer_callee(
        deep_gemm_utils,
        "fp8_fp4_paged_mqa_logits" if paged else "fp8_fp4_mqa_logits",
        lambda indexer, original, bound: indexer._score(deep_gemm_utils, original, bound, paged),
        applies=lambda indexer: bool(indexer._enabled_scorer_quantizers()),
    )


def _scorer_deep_gemm(deep_gemm_utils: ModuleType, scorer_kwargs: dict) -> ModuleType:
    """The DeepGEMM module vLLM loaded, checked to accept the scorer keyword arguments here.

    Stock DeepGEMM takes ``logits_dtype``; a build declares the others it accepts, with their
    values, in ``MQA_LOGITS_NUMERICS``. The scorer only runs for long sequences, so this fails
    early instead.
    """
    if hasattr(deep_gemm_utils, "_lazy_init"):  # like vLLM's wrappers: JIT cache directory, PDL
        deep_gemm_utils._lazy_init()
    deep_gemm = deep_gemm_utils._import_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError("The indexer scorer quantizers need DeepGEMM, which vLLM did not find.")
    declared = getattr(deep_gemm, "MQA_LOGITS_NUMERICS", None) or {}
    undeclared = {
        key: value
        for key, value in scorer_kwargs.items()
        if key not in _STOCK_SCORER_KWARGS and value not in declared.get(key, ())
    }
    if undeclared:
        raise RuntimeError(
            f"deep_gemm {getattr(deep_gemm, '__version__', '')} from "
            f"{getattr(deep_gemm, '__file__', '?')} does not declare the indexer scorer keyword "
            f"arguments {undeclared} in MQA_LOGITS_NUMERICS. Serve with a DeepGEMM build that "
            "takes them."
        )
    return deep_gemm


def _call_deep_gemm_scorer(
    deep_gemm_utils: ModuleType,
    arguments: dict,
    weights: torch.Tensor,
    scorer_kwargs: dict,
    paged: bool,
) -> torch.Tensor:
    """Call DeepGEMM's scorer with ``scorer_kwargs``, which vLLM's wrappers do not pass on."""
    deep_gemm = _scorer_deep_gemm(deep_gemm_utils, scorer_kwargs)
    kwargs = {"clean_logits": arguments["clean_logits"], **scorer_kwargs}
    if not paged:
        return deep_gemm.fp8_fp4_mqa_logits(
            arguments["q"],
            arguments["kv"],
            weights,
            arguments["cu_seqlen_ks"],
            arguments["cu_seqlen_ke"],
            **kwargs,
        )
    # Like vLLM's wrapper: DeepGEMM needs a unit last stride, and .contiguous() keeps a size-1
    # dim's stride.
    block_tables = arguments["block_tables"]
    if block_tables.dim() >= 2 and block_tables.stride(-1) != 1:
        block_tables = block_tables.clone(memory_format=torch.contiguous_format)
    if arguments.get("indices") is not None:
        kwargs["indices"] = arguments["indices"]
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        arguments["q"],
        arguments["kv_cache"],
        weights,
        arguments["context_lens"],
        block_tables,
        arguments["schedule_metadata"],
        arguments["max_model_len"],
        **kwargs,
    )


# GLM-5.3-Flash's fused Hadamard + FP8 quantization of the indexer query.
_GLM5NEXT_QUERY_KERNEL = "fwht128_quant_fp8"


class _QuantVLLMGlm5NextIndexer(_QuantVLLMIndexerBase):
    """GLM-5.3-Flash indexer: kernels Hadamard-rotate and FP8-quantize the query and pooled keys.

    The query kernel is swapped for a re-quantizing wrapper while ``forward`` runs; CUDA graph
    capture records the wrapper. The indexer op writes the K cache and scores the keys of the layer
    that it names, so the op and the kernel entry points it calls are wrapped process-wide: the op
    makes its layer's indexer the layer owner (see :mod:`.vllm_layer_scope`), the K-cache writers
    re-quantize the pools they wrote, on prefill and on decode pool completion, and vLLM's DeepGEMM
    scorer entry points apply the scorer quantizers.
    """

    kpool_ops: ModuleType | None = _glm5next_kpool_ops
    # The module whose ``sparse_attn_indexer_kpool`` the indexer op calls.
    op_module: ModuleType | None = _glm5next_op_module
    # The op imports its DeepGEMM scorer kernels from this module.
    deep_gemm_utils: ModuleType | None = vllm_deep_gemm
    # The native forward looks up its query kernel in this module.
    indexer_module: ModuleType | None = (
        inspect.getmodule(VllmGlm5NextIndexer) if VllmGlm5NextIndexer is not None else None
    )

    def _setup(self):
        super()._setup()
        # Imported together with the registered indexer class.
        assert self.kpool_ops is not None and self.op_module is not None
        assert self.deep_gemm_utils is not None
        _glm5next_indexers.add(self)
        # The op names the layer by its K cache, which vLLM registers under that name.
        register_layer_owner(self.k_cache, self)
        # Every layer calls these; only the first call wraps.
        wrap_layer_op(
            self.op_module,
            _GLM5NEXT_INDEXER_OP,
            "k_cache_prefix",
            decorator=getattr(self.op_module, "eager_break_during_capture", None),
        )
        for name, written_slots in _KPOOL_CACHE_WRITERS.items():
            wrap_layer_callee(
                self.kpool_ops,
                name,
                _requantize_written_pools(written_slots),
                applies=lambda indexer: indexer.indexer_k_quantizer.is_enabled,
                outside=_reject_k_cache_write_outside_op,
            )
        # Prefill scores keys gathered from the K cache into one buffer, per query range.
        _wrap_deep_gemm_scorer(self.deep_gemm_utils, paged=False)
        # Decode scores keys read straight from the paged K cache.
        _wrap_deep_gemm_scorer(self.deep_gemm_utils, paged=True)

    def forward(self, *args, **kwargs):
        # The scorer only runs for sequences longer than index_topk, which calibration and warmup
        # may never reach: fail here rather than on the first long request.
        self._validate_scorer_quantizers()
        if not self.indexer_q_quantizer.is_enabled:
            return super().forward(*args, **kwargs)
        module = self.indexer_module
        quant_fn = getattr(module, _GLM5NEXT_QUERY_KERNEL, None)
        if quant_fn is None:
            raise NotImplementedError(
                "indexer_q_quantizer: this vLLM version's GLM-5.3-Flash indexer does not quantize "
                f"the query with {_GLM5NEXT_QUERY_KERNEL}."
            )
        # Swap the module global for this call only; vLLM runs one forward at a time.
        setattr(module, _GLM5NEXT_QUERY_KERNEL, functools.partial(self._quantize_query, quant_fn))
        try:
            return super().forward(*args, **kwargs)
        finally:
            setattr(module, _GLM5NEXT_QUERY_KERNEL, quant_fn)

    def _quantize_query(
        self, quant_fn: Callable, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_fp8, q_scale = quant_fn(q)  # rotated query [rows, 128] and its [rows, 1] scales
        q_fp8, q_scale = _fake_quantize_fp8_rows(q_fp8.float() * q_scale, self.indexer_q_quantizer)
        return q_fp8, q_scale[:, None]

    def _validate_scorer_quantizers(self) -> None:
        """Reject what the scorer cannot apply, also while calibrating.

        That includes a DeepGEMM that does not take the enabled quantizers' keyword arguments.
        """
        if scorer_kwargs := self._scorer_kwargs(quantizing=False):
            assert self.deep_gemm_utils is not None  # checked in _setup
            _scorer_deep_gemm(self.deep_gemm_utils, scorer_kwargs)

    def _scorer_kwargs(self, quantizing: bool = True) -> dict:
        """The scorer keyword arguments of the enabled scorer-kwargs quantizers.

        ``quantizing``: only of those that quantize now; calibration runs the stock kernel.
        """
        kwargs: dict = {}
        for name in self._enabled_scorer_quantizers():
            quantizer = getattr(self, name)
            if not isinstance(quantizer, TensorQuantizer):
                raise ValueError(f"{name} takes a single format, not a list of formats.")
            if quantizing and not quantizer._if_quant:
                continue
            for key, value in _SCORER_KWARGS_QUANTIZERS[name](quantizer).items():
                if key in kwargs and kwargs[key] != value:
                    raise ValueError(
                        f"{name} sets the scorer's {key} to {value!r}, another quantizer to "
                        f"{kwargs[key]!r}."
                    )
                kwargs[key] = value
        if "logits_dtype" in kwargs:
            kwargs["logits_dtype"] = _scorer_logits_dtype(kwargs["logits_dtype"])
        return kwargs

    def _score(
        self,
        deep_gemm_utils: ModuleType,
        original: Callable,
        bound: inspect.BoundArguments,
        paged: bool,
    ) -> torch.Tensor:
        """Run the call ``bound`` of vLLM's scorer ``original`` with the scorer quantizers.

        ``weights`` is the effective W: the query scale and the model's normalization are folded in.
        """
        arguments = bound.arguments
        scorer_kwargs = self._scorer_kwargs()
        weights = arguments["weights"]
        if scorer_kwargs.get("logits_dtype") == torch.bfloat16:  # DeepGEMM's BF16 scorer
            weights = weights.to(torch.bfloat16)
        if scorer_kwargs:
            logits = _call_deep_gemm_scorer(
                deep_gemm_utils, arguments, weights, scorer_kwargs, paged
            )
        else:
            logits = original(*bound.args, **bound.kwargs)
        if logits.dtype != torch.float32:
            logits = _widen_logits(logits)
        # vLLM's prefill top-k returns out-of-range indices for rows with NaN scores, which the
        # sparse attention then reads out of bounds; it handles -inf and +inf. An overflowing FP16
        # accumulation or rounding turns scores into NaN (e.g. +inf - inf in the head sum).
        return logits.nan_to_num_(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))


if VllmDeepseekV4Indexer is not None:
    QuantModuleRegistry.register({VllmDeepseekV4Indexer: "vllm_DeepseekV4Indexer"})(
        _QuantVLLMDeepseekV4Indexer
    )

if VllmGlm5NextIndexer is not None:
    QuantModuleRegistry.register({VllmGlm5NextIndexer: "vllm_Glm5NextIndexer"})(
        _QuantVLLMGlm5NextIndexer
    )
