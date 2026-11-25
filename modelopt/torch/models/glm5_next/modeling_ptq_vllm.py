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

"""GLM-5.3-Flash PTQ modeling for vLLM (HF model type ``glm5_next``)."""

import functools
import importlib
import inspect
import weakref
from collections.abc import Callable
from types import ModuleType

import torch

from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins import vllm_indexer
from modelopt.torch.quantization.plugins.vllm_indexer import (
    _STOCK_SCORER_KWARGS,
    _fake_quantize_fp8_rows,
    _QuantVLLMIndexerBase,
    _requantize_fp8_indexer_k_cache,
    _scorer_logits_dtype,
    _widen_logits,
)
from modelopt.torch.quantization.plugins.vllm_layer_scope import (
    current_layer,
    current_layer_owner,
    patch_function,
    register_layer_owner,
    wrap_layer_op,
)

__all__: list[str] = []

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


# Once per function: the patched kernels run on every step.
_signature = functools.cache(inspect.signature)


def _bind(fn: Callable, args: tuple, kwargs: dict) -> inspect.BoundArguments:
    """The call ``fn(*args, **kwargs)`` with its arguments by name, defaults included."""
    bound = _signature(fn).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _write_k_cache(written_slots: Callable, original: Callable, *args, **kwargs):
    """A call of a K-cache writer; ``written_slots`` gives the slots that the call writes.

    In the indexer op of a layer with ``indexer_k_quantizer`` on: the kernel, then a re-quantization
    of the pools it wrote.
    """
    if current_layer() is None:
        # Only the indexer op names the layer of the cache.
        if any(indexer.indexer_k_quantizer.is_enabled for indexer in _glm5next_indexers):
            raise RuntimeError(
                "vLLM wrote the GLM-5.3-Flash indexer K cache outside the indexer op, so "
                "indexer_k_quantizer cannot tell whose cache it is."
            )
        return original(*args, **kwargs)
    indexer = current_layer_owner()
    if indexer is None or not indexer.indexer_k_quantizer.is_enabled:
        return original(*args, **kwargs)
    bound = _bind(original, args, kwargs)
    kv_cache = bound.arguments["kv_cache"]
    if kv_cache.data_ptr() != indexer.k_cache.kv_cache.data_ptr():
        raise RuntimeError(
            f"The indexer op of {indexer.k_cache.prefix} writes the K cache of another layer."
        )
    out = original(*args, **kwargs)
    slots, valid = written_slots(bound.arguments)
    _requantize_fp8_indexer_k_cache(kv_cache, slots, valid, indexer.indexer_k_quantizer)
    return out


def _run_scorer(paged: bool, original: Callable, *args, **kwargs) -> torch.Tensor:
    """A call of vLLM's DeepGEMM scorer: with the scorer quantizers of the layer whose op runs.

    ``paged``: the decode scorer, else the prefill one.
    """
    indexer = current_layer_owner()
    if indexer is None or not indexer._enabled_scorer_quantizers():
        return original(*args, **kwargs)
    return indexer._score(original, _bind(original, args, kwargs), paged)


# GLM-5.3-Flash's K-cache writers, patched to re-quantize the slots that a call writes.
_KPOOL_CACHE_WRITERS = {
    "kpool_compress_and_write_cache": functools.partial(_write_k_cache, _kpool_prefill_written),
    "kpool_decode_update_and_maybe_write_cache_batched": functools.partial(
        _write_k_cache, _kpool_decode_written
    ),
}
# vLLM's DeepGEMM scorers, patched to apply the scorer quantizers. The indexer op imports them from
# vLLM's module on every call. Prefill scores keys gathered from the K cache into one buffer, per
# query range; decode scores keys read straight from the paged K cache.
_DEEP_GEMM_SCORERS = {
    "fp8_fp4_mqa_logits": functools.partial(_run_scorer, False),
    "fp8_fp4_paged_mqa_logits": functools.partial(_run_scorer, True),
}
# GLM-5.3-Flash's indexer op, which writes the K cache and scores and selects the keys.
_GLM5NEXT_INDEXER_OP = "sparse_attn_indexer_kpool"


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
    makes its layer the current layer, whose indexer is the layer owner (see
    :mod:`.vllm_layer_scope`), the K-cache writers re-quantize the pools they wrote, on prefill and
    on decode pool completion, and vLLM's DeepGEMM scorer entry points apply the scorer quantizers.
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
        # The op names the layer by its K cache.
        register_layer_owner(self.k_cache.prefix, self)
        # Every layer calls these; only the first call wraps.
        wrap_layer_op(
            self.op_module,
            _GLM5NEXT_INDEXER_OP,
            "k_cache_prefix",
            decorator=getattr(self.op_module, "eager_break_during_capture", None),
        )
        for name, wrapper in _KPOOL_CACHE_WRITERS.items():
            patch_function(self.kpool_ops, name, wrapper)
        for name, wrapper in _DEEP_GEMM_SCORERS.items():
            patch_function(self.deep_gemm_utils, name, wrapper)

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
            # Looked up on the plugin at call time, like the base class does, so both read the same
            # registry even if it is rebound.
            for key, value in vllm_indexer._SCORER_KWARGS_QUANTIZERS[name](quantizer).items():
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
        self, original: Callable, bound: inspect.BoundArguments, paged: bool
    ) -> torch.Tensor:
        """Run the call ``bound`` of vLLM's scorer ``original`` with the scorer quantizers.

        ``weights`` is the effective W: the query scale and the model's normalization are folded in.
        """
        assert self.deep_gemm_utils is not None  # checked in _setup
        arguments = bound.arguments
        scorer_kwargs = self._scorer_kwargs()
        weights = arguments["weights"]
        if scorer_kwargs.get("logits_dtype") == torch.bfloat16:  # DeepGEMM's BF16 scorer
            weights = weights.to(torch.bfloat16)
        if scorer_kwargs:
            logits = _call_deep_gemm_scorer(
                self.deep_gemm_utils, arguments, weights, scorer_kwargs, paged
            )
        else:
            logits = original(*bound.args, **bound.kwargs)
        if logits.dtype != torch.float32:
            logits = _widen_logits(logits)
        # vLLM's prefill top-k returns out-of-range indices for rows with NaN scores, which the
        # sparse attention then reads out of bounds; it handles -inf and +inf. An overflowing FP16
        # accumulation or rounding turns scores into NaN (e.g. +inf - inf in the head sum).
        return logits.nan_to_num_(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))


if VllmGlm5NextIndexer is not None:
    QuantModuleRegistry.register({VllmGlm5NextIndexer: "vllm_Glm5NextIndexer"})(
        _QuantVLLMGlm5NextIndexer
    )
