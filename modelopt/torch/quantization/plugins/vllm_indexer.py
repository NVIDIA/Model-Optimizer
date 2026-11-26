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

Covers the CSA indexer of DeepSeek-V4 and the k-pool indexer of GLM-5.3-Flash, whose layouts live
with their models (``modelopt/torch/models/<model_type>/modeling_ptq_vllm.py``).

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

import inspect
from collections.abc import Callable

import torch

from ..config import QuantizerAttributeConfig
from ..nn import QuantModule, TensorQuantizer, is_registered_quant_backend, register_quant_backend
from .vllm import create_parallel_state

__all__ = ["register_indexer_scorer_kwargs_quantizer"]

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
    ``backend_extra_args`` (``backend: scorer_kwargs``) on as they are. Like for the built-in one,
    ``mtq.quantize`` raises when a config enables the quantizer but no indexer has it.
    """
    from ..model_quant import _PLUGIN_QUANTIZERS

    _SCORER_KWARGS_QUANTIZERS[name] = scorer_kwargs
    _PLUGIN_QUANTIZERS[name] = _PLUGIN_QUANTIZERS[_SCORER_KWARGS_QUANTIZER]


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
