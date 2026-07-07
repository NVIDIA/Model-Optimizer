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

"""Custom vLLM workers for checkpoint-driven sparse and fixed-NVFP4 attention."""

import importlib
import os
from collections import Counter
from functools import cache
from types import SimpleNamespace
from typing import NamedTuple

try:
    _has_legacy_attention_layer = importlib.util.find_spec("vllm.attention.layer") is not None
except (ModuleNotFoundError, ValueError):
    _has_legacy_attention_layer = False

if _has_legacy_attention_layer:
    from vllm.attention.layer import Attention as VLLMAttention
else:
    from vllm.model_executor.layers.attention import Attention as VLLMAttention

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _build_sparse_kw,
    _clone_sparse_impl,
    _p_qdq_from_layer,
    _v_qdq_from_layer,
    select_sparse_impl_cls,
)

__all__ = ["SparseAttnWorker", "QuantSparseAttnWorker"]  # noqa: RUF022

_NVFP4_CFG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}
_BMM_CFG = [
    {"quantizer_name": "*_bmm_quantizer", "enable": False},
    *(
        {"quantizer_name": f"*{name}_bmm_quantizer", "cfg": _NVFP4_CFG, "enable": True}
        for name in ("q", "k", "p", "v")
    ),
]


class _AttentionPlan(NamedTuple):
    module: object
    new_impl: object
    sparse_kw: dict
    device: object | None
    dtype: object | None


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _sparse_kwargs(name: str, sparse_cfg: dict | None) -> dict:
    if sparse_cfg is None:
        return {}
    layer_cfg = match_sparse_config(name, sparse_cfg)
    if layer_cfg is None or not layer_cfg.get("enable", True):
        return {}
    return _build_sparse_kw(layer_cfg)


@cache
def _load_quant_api(vllm_version: str):
    # Keep sparse-only module loading independent of quant-specific vLLM APIs.
    import torch
    from packaging import version

    if version.parse(vllm_version) < version.parse("0.14.0"):
        raise RuntimeError("The compact NVFP4 attention worker requires vLLM >= 0.14.0")

    from vllm.config import compilation
    from vllm.v1.attention import backend

    from modelopt.torch.quantization import conversion
    from modelopt.torch.quantization import nn as quant_nn
    from modelopt.torch.quantization.plugins import vllm as quant_plugin

    return SimpleNamespace(
        torch=torch,
        compilation=compilation,
        backend=backend,
        conversion=conversion,
        nn=quant_nn,
        plugin=quant_plugin,
    )


def _quant_api():
    import vllm

    return _load_quant_api(vllm.__version__)


def _cudagraph_mode(worker, api):
    config = getattr(worker.model_runner, "vllm_config", None)
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    return mode if mode is not None else api.compilation.CUDAGraphMode.NONE


def _global_errors(worker, api) -> list[str]:
    config = worker.model_runner.vllm_config
    parallel = config.parallel_config
    cache_config, model_config = config.cache_config, config.model_config
    errors = []
    if getattr(parallel, "decode_context_parallel_size", 1) != 1:
        errors.append("decode_context_parallel_size must be 1")
    if getattr(parallel, "enable_dbo", False) or getattr(parallel, "use_ubatching", False):
        errors.append("DBO/ubatching is unsupported")
    if getattr(cache_config, "enable_prefix_caching", False):
        errors.append("prefix caching is unsupported")
    if getattr(config, "kv_transfer_config", None) is not None:
        errors.append("KV transfer is unsupported")
    if getattr(config, "speculative_config", None) is not None:
        errors.append("speculative decoding is unsupported")
    if _cudagraph_mode(worker, api).mixed_mode() == api.compilation.CUDAGraphMode.FULL:
        errors.append("FULL mixed-batch cudagraph mode is unsupported")
    if getattr(model_config, "dtype", None) not in (api.torch.float16, api.torch.bfloat16):
        errors.append("resolved model/KV-cache dtype must be fp16 or bf16")
    cache_dtype = getattr(cache_config, "cache_dtype", "auto")
    if str(cache_dtype) not in {"auto", "bfloat16", "float16", "torch.bfloat16", "torch.float16"}:
        errors.append(f"resolved KV-cache dtype {cache_dtype!r} must be fp16 or bf16")
    return errors


def _quant_layer_errors(module, api) -> list[str]:
    impl = getattr(module, "impl", None)
    errors = []
    if type(module) is not api.plugin.vllm_attention.Attention:
        errors.append(f"layout {type(module).__name__} is not regular decoder self-attention")
    if getattr(module, "attn_type", None) != api.backend.AttentionType.DECODER:
        errors.append("attn_type must be DECODER")
    head_size = getattr(module, "head_size", None)
    if not isinstance(head_size, int) or head_size % 16:
        errors.append(f"head_size={head_size!r} must be a multiple of 16")
    head_size_v = getattr(module, "head_size_v", head_size)
    if head_size_v != head_size:
        errors.append(f"head_size_v={head_size_v!r} must equal head_size={head_size!r}")
    if getattr(module, "sliding_window", None) is not None:
        errors.append("sliding_window is unsupported")
    if getattr(module, "kv_sharing_target_layer_name", None) is not None:
        errors.append("cross-layer KV sharing is unsupported")
    if str(getattr(module, "kv_cache_dtype", "")).startswith("fp8"):
        errors.append("FP8 KV cache is unsupported")
    if getattr(impl, "alibi_slopes", None) is not None:
        errors.append("ALiBi is unsupported")
    if getattr(impl, "logits_soft_cap", None):
        errors.append("logits soft cap is unsupported")
    if getattr(impl, "sinks", None) is not None or getattr(impl, "has_sinks", False):
        errors.append("attention sinks are unsupported")
    return errors


def _sparse_graph_error(sparse_kw: dict, mode, api) -> str | None:
    """Reject decode calibration whose live length would be frozen by a full graph."""
    params = sparse_kw.get("threshold_scale_factor")
    if (
        mode.decode_mode() == api.compilation.CUDAGraphMode.FULL
        and isinstance(params, dict)
        and isinstance(params.get("decode"), dict)
    ):
        return "calibrated decode skip-softmax requires a non-FULL CUDA graph mode"
    return None


def _select_new_impl(module):
    """Clone the module's attention impl into its sparse-capable subclass; return (impl, error)."""
    try:
        cls = select_sparse_impl_cls(module.impl)
    except (NotImplementedError, TypeError) as err:
        return None, str(err)
    if cls is None:
        return None, (
            f"backend {type(module.impl).__name__} is not supported; "
            "expected FlashAttentionImpl or FlashInferImpl"
        )
    return _clone_sparse_impl(module.impl, cls), None


def _raise_unsupported(errors: list[str], policy: str) -> None:
    if errors:
        raise NotImplementedError(
            f"Unsupported ModelOpt {policy} plan:\n  - " + "\n  - ".join(errors)
        )


def _sparse_plans(worker):
    """Plans for checkpoint-driven sparse attention; skips layers without a sparse config."""
    model = _unwrapped_model(worker)
    detected = load_from_checkpoint_metadata(
        getattr(worker.model_runner.model_config, "hf_config", None)
    )
    if detected is None:
        print(
            "[ModelOpt] No sparse_attention_config found in the checkpoint; "
            "skipping sparse attention. Run examples/llm_sparsity/attention_sparsity/"
            "hf_sa.py to calibrate and export a checkpoint with the config embedded."
        )
        return None
    sparse_cfg, sparse_algo = detected
    print(f"[ModelOpt] Sparse attention config: algo -> {sparse_algo}")
    plans, errors = [], []
    for name, module in model.named_modules():
        if not isinstance(module, VLLMAttention):
            continue
        sparse_kw = _sparse_kwargs(name, sparse_cfg)
        if not sparse_kw:
            continue
        new_impl, error = _select_new_impl(module)
        if error:
            errors.append(f"{name or '<root>'}: {error}")
        else:
            plans.append(_AttentionPlan(module, new_impl, sparse_kw, None, None))
    _raise_unsupported(errors, "sparse attention")
    return tuple(plans)


def _quant_plans(worker):
    """Plans for fixed-NVFP4 attention on every decoder self-attention layer (+ optional sparsity)."""
    api = _quant_api()
    model = _unwrapped_model(worker)
    model_config = worker.model_runner.model_config
    detected = load_from_checkpoint_metadata(getattr(model_config, "hf_config", None))
    sparse_cfg = detected[0] if detected is not None else None
    errors = _global_errors(worker, api)
    mode = _cudagraph_mode(worker, api)
    plans, attention_count = [], 0
    for name, module in model.named_modules():
        if not isinstance(module, api.plugin._ATTENTION_TYPES):
            continue
        attention_count += 1
        reasons = _quant_layer_errors(module, api)
        # Prefer the model compute dtype (fp16/bf16); _get_device_dtype's buffer scan
        # can otherwise report fp32 from the attention module's scale buffers.
        device, dtype = api.plugin._get_device_dtype(module)
        if getattr(model_config, "dtype", None) in (api.torch.float16, api.torch.bfloat16):
            dtype = model_config.dtype
        if device is None or dtype is None:
            reasons.append("device/dtype could not be resolved")
        elif dtype not in (api.torch.float16, api.torch.bfloat16):
            reasons.append(f"resolved dtype {dtype} must be fp16 or bf16")
        sparse_kw = _sparse_kwargs(name, sparse_cfg)
        if graph_error := _sparse_graph_error(sparse_kw, mode, api):
            reasons.append(graph_error)
        new_impl, error = _select_new_impl(module)
        if error:
            reasons.append(error)
        if reasons:
            errors.extend(f"{name or '<root>'}: {reason}" for reason in reasons)
        else:
            plans.append(_AttentionPlan(module, new_impl, sparse_kw, device, dtype))
    if attention_count == 0:
        errors.append("no regular attention layers were found")
    _raise_unsupported(errors, "attention")
    return tuple(plans)


def _install_sparse_plans(plans) -> None:
    for plan in plans:
        plan.new_impl.sparse_kw = plan.sparse_kw
        plan.module.impl = plan.new_impl
    installed = dict(Counter(type(plan.new_impl).__name__ for plan in plans))
    print(
        f"[ModelOpt] Sparse attention: replaced impl on {len(plans)} attention layers: {installed}"
    )


def _install_quant_plans(worker, plans) -> None:
    api = _quant_api()
    quant_off = os.environ.get("MODELOPT_ATTN_QUANT_OFF") == "1"
    for plan in plans:
        module = plan.module
        module.device, module.dtype = plan.device, plan.dtype
        api.nn.QuantModuleRegistry.convert(module)
        module.p_bmm_quantizer = api.nn.TensorQuantizer()
        api.conversion.set_quantizer_by_cfg(module, _BMM_CFG)
        if quant_off:
            # Isolation knob: keep the ModelOpt kernel fixed while disabling all
            # Q/K/P/V NVFP4 transforms, so quant-on minus quant-off isolates the
            # fakequant loss without triggering the native dense fallback.
            for name in ("q", "k", "p", "v"):
                quantizer = getattr(module, f"{name}_bmm_quantizer", None)
                if quantizer is not None:
                    quantizer.disable()
        api.plugin._set_vllm_attention_kv_default_amax(module, plan.device)
        plan.new_impl.sparse_kw = plan.sparse_kw
        p_qdq, p_qdq_amax = _p_qdq_from_layer(module)
        v_qdq, v_qdq_amax = _v_qdq_from_layer(module)
        plan.new_impl.quant_kw = {
            "p_qdq": p_qdq,
            "p_qdq_amax": p_qdq_amax,
            "v_qdq": v_qdq,
            "v_qdq_amax": v_qdq_amax,
        }
        module.impl = plan.new_impl
        module._query_quant_in_kernel = not quant_off
        module._value_quant_in_kernel = not quant_off
        if quant_off:
            module._modelopt_force_kernel = True
    worker.model_runner.cascade_attn_enabled = False
    installed = dict(Counter(type(plan.new_impl).__name__ for plan in plans))
    print(f"[ModelOpt] Installed NVFP4 quant+sparse attention on {len(plans)} layers: {installed}")


def _install_attention(worker, *, quantize: bool) -> None:
    if quantize:
        _install_quant_plans(worker, _quant_plans(worker))
    else:
        plans = _sparse_plans(worker)
        if plans is not None:
            _install_sparse_plans(plans)


class _ModelOptAttentionWorker(BaseWorker):
    quantize_attention = False

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        _install_attention(self, quantize=self.quantize_attention)


class SparseAttnWorker(_ModelOptAttentionWorker):
    """Install checkpoint-driven ModelOpt sparse attention after model load."""


class QuantSparseAttnWorker(_ModelOptAttentionWorker):
    """Install fixed NVFP4 attention plus optional checkpoint sparsity."""

    quantize_attention = True

    def determine_available_memory(self) -> int:
        api = _quant_api()
        with api.torch.inference_mode(), api.plugin.disable_compilation(_unwrapped_model(self)):
            return BaseWorker.determine_available_memory(self)
