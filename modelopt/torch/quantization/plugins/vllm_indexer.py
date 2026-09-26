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

"""Fake quantization of the sparse-attention indexer K cache in vLLM.

Covers the DSA indexers of DeepSeek-V3.2 and GLM-5.x, the CSA indexer of DeepSeek-V4 and the
k-pool indexer of GLM-5.3-Flash.

``indexer_k_quantizer`` fake-quantizes the tensor the serving kernel quantizes into the indexer K
cache. Where vLLM only materializes that key inside a fused kernel that already wrote FP8 to the
cache, the entries written in the current step are read back, fake-quantized and re-stored. The
name avoids the ``*[kv]_bmm_quantizer`` globs, so the KV-cache presets leave it disabled.
"""

import functools
import importlib
import inspect
import weakref
from collections.abc import Callable
from types import ModuleType

import torch
from vllm.forward_context import get_forward_context

from ..nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from .vllm import create_parallel_state

__all__ = []

try:
    # The release that moved the K quantize+insert behind ``Indexer.indexer_op``.
    importlib.import_module("vllm.model_executor.layers.sparse_attn_indexer")
    from vllm.model_executor.models.deepseek_v2 import Indexer as VllmDeepseekV2Indexer
except ImportError:
    VllmDeepseekV2Indexer = None

# NotImplementedError: these packages reject unsupported platforms (XPU) in their __init__.
try:
    from vllm.models.deepseek_v32.attention import DeepseekV32Attention as VllmDeepseekV32Attention
    from vllm.models.deepseek_v32.attention import DeepseekV32Indexer as VllmDeepseekV32Indexer
except (ImportError, NotImplementedError):
    VllmDeepseekV32Attention = VllmDeepseekV32Indexer = None

try:
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer as VllmDeepseekV4Indexer
except (ImportError, NotImplementedError):
    VllmDeepseekV4Indexer = None


def _import_glm5next_indexer() -> tuple[type | None, ModuleType | None]:
    """Return GLM-5.3-Flash's ``Indexer`` and the module its indexer op binds as ``kpool_ops``."""
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
        return indexer_cls, kpool_ops
    return None, None


VllmGlm5NextIndexer, _glm5next_kpool_ops = _import_glm5next_indexer()

_INDEXER_K_FP8_MAX = 448.0
_INDEXER_K_SCALE_BYTES = 4


def _indexer_k_ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """Power-of-two FP8 scale used by vLLM's indexer K cache kernels (``scale_fmt="ue8m0"``)."""
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / _INDEXER_K_FP8_MAX)))


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
    k = quantizer(k)
    new_scale = _indexer_k_ue8m0_scale(k.abs().amax(dim=-1))
    new_values = (
        (k / new_scale[:, None])
        .clamp(-_INDEXER_K_FP8_MAX, _INDEXER_K_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
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


def _set_arg(args: tuple, kwargs: dict, name: str, pos: int, value) -> tuple[tuple, dict]:
    if name in kwargs:
        return args, {**kwargs, name: value}
    return (*args[:pos], value, *args[pos + 1 :]), kwargs


class _QuantVLLMIndexerBase(QuantModule):
    """Owner of ``indexer_k_quantizer`` for one vLLM indexer layout."""

    def _setup(self):
        self.indexer_k_quantizer = TensorQuantizer()
        self.parallel_state = create_parallel_state()

    def forward(self, *args, **kwargs):
        # The registry matches subclasses that share ``forward``; overriding it keeps the converted
        # class from matching again. vLLM's MLA wrapper holds the attention's indexer too
        # (``MLAModules(indexer=...)``), so ``replace_quant_module`` visits the indexer twice and
        # would otherwise try to convert it a second time (inconsistent MRO).
        return super().forward(*args, **kwargs)


class _QuantVLLMIndexerOpIndexer(_QuantVLLMIndexerBase):
    """DeepSeek-V3.2 / GLM-5 ``Indexer`` whose forward hands the bf16 key to ``indexer_op``.

    ``indexer_op`` (``SparseAttnIndexer(hidden_states, q_quant, k, weights)``) quantizes ``k`` to
    FP8 and inserts it into the indexer K cache, so the key is fake-quantized right before that
    call.
    """

    def _setup(self):
        super()._setup()
        self.indexer_op.register_forward_pre_hook(self._quantize_k, with_kwargs=True)

    def _quantize_k(self, module, args, kwargs):
        k = _get_arg(args, kwargs, "k", 2)
        if k is None:
            return None
        return _set_arg(args, kwargs, "k", 2, self.indexer_k_quantizer(k))


class _QuantVLLMDeepseekV32Attention(QuantModule):
    """Applies ``indexer.indexer_k_quantizer`` to the fused DeepSeek-V3.2 / GLM-5 attention.

    ``forward`` writes the FP8 key into the indexer cache inside ``fused_norm_rope``, so the entries
    written for this step are re-quantized before ``_sparse_indexer_and_attn`` scores them. Under
    prefill context parallelism (vLLM >= 0.28) the bf16 key is handed to that method instead and
    fake-quantized directly. The quantizer lives on ``self.indexer`` (``_QuantVLLMIndexerBase``).
    """

    def _setup(self):
        self.parallel_state = create_parallel_state()
        try:
            self._index_k_pos = _native_positional_index(
                self, "_sparse_indexer_and_attn", "index_k"
            )
        except ValueError:  # vLLM 0.27 opt-in layout: the cache is always written by the kernel
            self._index_k_pos = None

    def forward(self, *args, **kwargs):
        # Keeps the converted class from matching the registry again, see _QuantVLLMIndexerBase.
        return super().forward(*args, **kwargs)

    def _indexer_k_quantizer(self) -> TensorQuantizer | None:
        if self.indexer is None or self.skip_topk:
            return None
        quantizer = getattr(self.indexer, "indexer_k_quantizer", None)
        return quantizer if quantizer is not None and quantizer.is_enabled else None

    def _sparse_indexer_and_attn(self, *args, **kwargs):
        quantizer = self._indexer_k_quantizer()
        if quantizer is not None:
            index_k = None
            if self._index_k_pos is not None:
                index_k = _get_arg(args, kwargs, "index_k", self._index_k_pos)
            if index_k is not None:
                # Prefill context parallelism: ``fused_norm_rope`` leaves the indexer cache alone
                # and hands over the normed, RoPE'd bf16 key, which ``sparse_attn_indexer`` then
                # quantizes to FP8 and inserts. Fake-quantize it before that insert.
                args, kwargs = _set_arg(
                    args, kwargs, "index_k", self._index_k_pos, quantizer(index_k)
                )
            else:
                # Otherwise (and always on the vLLM 0.27 layout) ``fused_norm_rope`` has already
                # written this step's keys to the FP8 cache. Re-quantize only those rows, found
                # through this step's slot mapping; rows of earlier steps were done when written.
                forward_context = get_forward_context()
                # The indexer cache has its own slot mapping; the kernel writes with that one.
                slot_mapping = forward_context.slot_mapping.get(self.indexer.k_cache.prefix)
                # Profiling runs (no attn_metadata) write nothing.
                if forward_context.attn_metadata is not None and slot_mapping is not None:
                    _requantize_fp8_indexer_k_cache(
                        self.indexer.k_cache.kv_cache, slot_mapping, slot_mapping >= 0, quantizer
                    )
        return super()._sparse_indexer_and_attn(*args, **kwargs)


class _QuantVLLMDeepseekV4Indexer(_QuantVLLMIndexerBase):
    """DeepSeek-V4 indexer: the compressor kernel FP8-quantizes and caches the compressed key.

    After the compressor ran, the entries it wrote (one per ``compress_ratio`` tokens) are
    re-quantized through ``indexer_k_quantizer``. vLLM caches the key without the Hadamard rotation
    that the DeepSeek reference applies.
    """

    def _setup(self):
        super()._setup()
        self._positions_pos = _native_positional_index(self, "forward", "positions")

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if self.indexer_k_quantizer.is_enabled:
            self._requantize_written_keys(_get_arg(args, kwargs, "positions", self._positions_pos))
        return out

    def _requantize_written_keys(self, positions: torch.Tensor) -> None:
        if getattr(self, "use_fp4_kv", False):
            raise NotImplementedError(
                "indexer_k_quantizer re-quantizes the FP8 indexer cache; serve with the default "
                "indexer_kv_dtype instead of 'mxfp4'."
            )
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


# GLM-5.3-Flash indexers converted in this process, looked up by the cache tensor a kernel writes.
_glm5next_indexers: weakref.WeakSet = weakref.WeakSet()


def _glm5next_quantizer_for(kv_cache: torch.Tensor) -> TensorQuantizer | None:
    """The enabled ``indexer_k_quantizer`` of the indexer that owns ``kv_cache``, if any."""
    for indexer in _glm5next_indexers:
        quantizer = indexer.indexer_k_quantizer
        if quantizer.is_enabled and indexer.k_cache.kv_cache.data_ptr() == kv_cache.data_ptr():
            return quantizer
    return None


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


def _wrap_kpool_cache_writer(kpool_ops: ModuleType, name: str, written_slots: Callable) -> None:
    """Wrap ``kpool_ops.<name>`` so the pools it wrote are re-quantized after every call.

    The wrap is permanent: the indexer op is a breakable-cudagraph eager break that vLLM replays
    without re-entering ``Indexer.forward``, so a per-forward patch would miss every replayed step.
    """
    original = getattr(kpool_ops, name)
    if getattr(original, "_modelopt_indexer_k_wrapped", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        kv_cache = bound.arguments["kv_cache"]
        quantizer = _glm5next_quantizer_for(kv_cache)
        if quantizer is not None:
            slots, valid = written_slots(bound.arguments)
            _requantize_fp8_indexer_k_cache(kv_cache, slots, valid, quantizer)
        return out

    wrapper._modelopt_indexer_k_wrapped = True  # type: ignore[attr-defined]
    setattr(kpool_ops, name, wrapper)


def _install_kpool_cache_hooks(kpool_ops: ModuleType) -> None:
    _wrap_kpool_cache_writer(kpool_ops, "kpool_compress_and_write_cache", _kpool_prefill_written)
    _wrap_kpool_cache_writer(
        kpool_ops, "kpool_decode_update_and_maybe_write_cache_batched", _kpool_decode_written
    )


class _QuantVLLMGlm5NextIndexer(_QuantVLLMIndexerBase):
    """GLM-5.3-Flash indexer: kpool kernels Hadamard-rotate, FP8-quantize and cache the pooled keys.

    The kernel entry points that write the indexer K cache are wrapped process-wide and re-quantize
    the pools they wrote, on prefill and on decode pool completion.
    """

    kpool_ops: ModuleType | None = _glm5next_kpool_ops

    def _setup(self):
        super()._setup()
        assert self.kpool_ops is not None  # imported together with the registered indexer class
        _glm5next_indexers.add(self)
        _install_kpool_cache_hooks(self.kpool_ops)


if VllmDeepseekV2Indexer is not None:
    QuantModuleRegistry.register({VllmDeepseekV2Indexer: "vllm_DeepseekV2Indexer"})(
        _QuantVLLMIndexerOpIndexer
    )

# ``vllm.models.deepseek_v32`` (default for DeepSeek-V3.2 / GLM-5 on vLLM >= 0.28, opt-in before)
# never calls its indexer's forward: the attention writes the cache in ``fused_norm_rope``.
if VllmDeepseekV32Attention is not None and hasattr(
    VllmDeepseekV32Attention, "_sparse_indexer_and_attn"
):
    QuantModuleRegistry.register({VllmDeepseekV32Indexer: "vllm_DeepseekV32Indexer"})(
        _QuantVLLMIndexerBase
    )
    QuantModuleRegistry.register({VllmDeepseekV32Attention: "vllm_DeepseekV32Attention"})(
        _QuantVLLMDeepseekV32Attention
    )

if VllmDeepseekV4Indexer is not None:
    QuantModuleRegistry.register({VllmDeepseekV4Indexer: "vllm_DeepseekV4Indexer"})(
        _QuantVLLMDeepseekV4Indexer
    )

if VllmGlm5NextIndexer is not None:
    QuantModuleRegistry.register({VllmGlm5NextIndexer: "vllm_Glm5NextIndexer"})(
        _QuantVLLMGlm5NextIndexer
    )
