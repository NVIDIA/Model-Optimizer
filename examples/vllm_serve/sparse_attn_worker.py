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
    name: str
    module: object
    new_impl: object
    sparse_kw: dict
    device: object | None
    dtype: object | None


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


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


def _global_errors(worker, api=None) -> list[str]:
    api = api or _quant_api()
    config = worker.model_runner.vllm_config
    parallel, cache, model_config = config.parallel_config, config.cache_config, config.model_config
    errors = []
    if getattr(parallel, "decode_context_parallel_size", 1) != 1:
        errors.append("decode_context_parallel_size must be 1")
    if getattr(parallel, "enable_dbo", False) or getattr(parallel, "use_ubatching", False):
        errors.append("DBO/ubatching is unsupported")
    if getattr(cache, "enable_prefix_caching", False):
        errors.append("prefix caching is unsupported")
    if getattr(config, "kv_transfer_config", None) is not None:
        errors.append("KV transfer is unsupported")
    if getattr(config, "speculative_config", None) is not None:
        errors.append("speculative decoding is unsupported")
    if _cudagraph_mode(worker, api).mixed_mode() == api.compilation.CUDAGraphMode.FULL:
        errors.append("FULL mixed-batch cudagraph mode is unsupported")
    if getattr(model_config, "dtype", None) not in (api.torch.float16, api.torch.bfloat16):
        errors.append("resolved model/KV-cache dtype must be fp16 or bf16")
    cache_dtype = getattr(cache, "cache_dtype", "auto")
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


def _validated_device_dtype(module, model_config, api):
    device, dtype = api.plugin._get_device_dtype(module)
    model_dtype = getattr(model_config, "dtype", None)
    if model_dtype in (api.torch.float16, api.torch.bfloat16):
        dtype = model_dtype
    if device is None or dtype is None:
        return device, dtype, "device/dtype could not be resolved"
    if dtype not in (api.torch.float16, api.torch.bfloat16):
        return device, dtype, f"resolved dtype {dtype} must be fp16 or bf16"
    return device, dtype, None


def _sparse_graph_error(sparse_kw: dict, mode, api=None) -> str | None:
    """Reject decode calibration whose live length would be frozen by a full graph."""
    api = api or _quant_api()
    params = sparse_kw.get("threshold_scale_factor")
    if (
        mode.decode_mode() == api.compilation.CUDAGraphMode.FULL
        and isinstance(params, dict)
        and isinstance(params.get("decode"), dict)
    ):
        return "calibrated decode skip-softmax requires a non-FULL CUDA graph mode"
    return None


def _validated_attention_plans(worker, *, quantize: bool):
    """Validate and clone every selected attention adapter before mutating layers."""
    api = _quant_api() if quantize else None
    model = _unwrapped_model(worker)
    model_config = worker.model_runner.model_config
    detected = load_from_checkpoint_metadata(getattr(model_config, "hf_config", None))
    if not quantize and detected is None:
        print(
            "[ModelOpt] No sparse_attention_config found in the checkpoint; "
            "skipping sparse attention. Run examples/llm_sparsity/"
            "attention_sparsity/hf_sa.py to calibrate and export a checkpoint "
            "with the config embedded."
        )
        return None

    sparse_cfg = detected[0] if detected is not None else None
    if quantize:
        assert api is not None
        errors = _global_errors(worker, api)
        mode = _cudagraph_mode(worker, api)
        attention_types = api.plugin._ATTENTION_TYPES
    else:
        assert detected is not None
        print(f"[ModelOpt] Sparse attention config: algo -> {detected[1]}")
        errors, mode, attention_types = [], None, (VLLMAttention,)
    plans = []
    attention_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, attention_types):
            continue
        if quantize:
            assert api is not None
            attention_count += 1
            reasons = _quant_layer_errors(module, api)
            device, dtype, dtype_error = _validated_device_dtype(module, model_config, api)
            if dtype_error:
                reasons.append(dtype_error)
            layer_cfg = match_sparse_config(name, sparse_cfg) if sparse_cfg is not None else None
            sparse_kw = (
                _build_sparse_kw(layer_cfg)
                if layer_cfg is not None and layer_cfg.get("enable", True)
                else {}
            )
            if graph_error := _sparse_graph_error(sparse_kw, mode, api):
                reasons.append(graph_error)
        else:
            assert sparse_cfg is not None
            layer_cfg = match_sparse_config(name, sparse_cfg)
            if layer_cfg is None or not layer_cfg.get("enable", True):
                continue
            sparse_kw = _build_sparse_kw(layer_cfg)
            if not sparse_kw:
                continue
            reasons, device, dtype = [], None, None

        new_impl = None
        try:
            new_impl_cls = select_sparse_impl_cls(module.impl)
            if new_impl_cls is None:
                backend = type(module.impl).__name__
                message = (
                    f"backend {backend} is not supported; expected FlashAttentionImpl or FlashInferImpl"
                    if quantize
                    else f"unsupported backend {backend}"
                )
                reasons.append(message)
            else:
                new_impl = _clone_sparse_impl(module.impl, new_impl_cls)
        except (NotImplementedError, TypeError) as err:
            reasons.append(str(err))
        layer_name = name or "<root>"
        if reasons:
            errors.extend(f"{layer_name}: {reason}" for reason in reasons)
        else:
            plans.append(_AttentionPlan(name, module, new_impl, sparse_kw, device, dtype))

    if quantize and attention_count == 0:
        errors.append("no regular attention layers were found")
    if errors:
        policy = "attention" if quantize else "sparse attention"
        raise NotImplementedError(
            f"Unsupported ModelOpt {policy} plan:\n  - " + "\n  - ".join(errors)
        )
    return tuple(plans)


def _install_sparse_plans(plans) -> None:
    installed = {}
    for plan in plans:
        plan.new_impl.sparse_kw = plan.sparse_kw
        plan.module.impl = plan.new_impl
        impl_name = type(plan.new_impl).__name__
        installed[impl_name] = installed.get(impl_name, 0) + 1
    print(
        f"[ModelOpt] Sparse attention: replaced impl on {len(plans)} attention layers: {installed}"
    )


def _install_quant_plans(worker, plans) -> None:
    api = _quant_api()
    quant_off = os.environ.get("MODELOPT_ATTN_QUANT_OFF") == "1"
    installed = {}
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
        impl_name = type(plan.new_impl).__name__
        installed[impl_name] = installed.get(impl_name, 0) + 1
    worker.model_runner.cascade_attn_enabled = False
    print(f"[ModelOpt] Installed NVFP4 quant+sparse attention on {len(plans)} layers: {installed}")


def _install_attention(worker, *, quantize: bool) -> None:
    plans = _validated_attention_plans(worker, quantize=quantize)
    if plans is None:
        return
    if quantize:
        _install_quant_plans(worker, plans)
    else:
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
