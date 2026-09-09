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

"""Install ModelOpt attention transforms into a loaded vLLM model."""

import importlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import vllm
from packaging import version
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from . import vllm as attention_plugin
from .sparse_attn_config import load_from_checkpoint_metadata, match_sparse_config

__all__ = [
    "VllmAttentionInstallReport",
    "install_vllm_nvfp4_attention",
    "install_vllm_sparse_attention_from_checkpoint",
]


def _import_attention_type() -> type:
    """Import the concrete vLLM Attention type across supported releases."""
    for module_name in (
        "vllm.attention.layer",
        "vllm.model_executor.layers.attention",
        "vllm.attention",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "Attention"):
            return module.Attention
    raise ImportError("No supported vLLM Attention module was found")


_VLLM_ATTENTION = _import_attention_type()


@dataclass(frozen=True, slots=True)
class VllmAttentionInstallReport:
    """Summary of attention modules changed by a vLLM runtime installation."""

    installed_layers: tuple[str, ...] = ()
    quantized_layers: tuple[str, ...] = ()
    sparse_layers: tuple[str, ...] = ()
    backend_counts: Mapping[str, int] = field(default_factory=dict)
    sparse_algorithm: str | None = None
    cascade_disabled: bool = False

    @property
    def installed_count(self) -> int:
        """Number of attention implementations installed."""
        return len(self.installed_layers)

    @property
    def transforms_active(self) -> bool:
        """Whether the installation activated any ModelOpt attention transform."""
        return bool(self.installed_layers)


@dataclass(frozen=True, slots=True)
class _AttentionPlan:
    name: str
    module: torch.nn.Module
    new_impl: Any
    sparse_kw: dict[str, Any]
    device: object | None
    dtype: torch.dtype | None
    requires_flashinfer_patch: bool


@dataclass(frozen=True, slots=True)
class _InstallPlan:
    model_runner: Any
    layers: tuple[_AttentionPlan, ...]
    quantize: bool
    sparse_algorithm: str | None
    q_format: str = "nvfp4"
    k_format: str = "nvfp4"
    p_format: str = "nvfp4"
    v_format: str = "nvfp4"


def _unwrapped_model(model_runner):
    model = model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _model_config(model_runner):
    model_config = getattr(model_runner, "model_config", None)
    if model_config is not None:
        return model_config
    return getattr(getattr(model_runner, "vllm_config", None), "model_config", None)


def _resolve_sparse_config(model_runner, sparse_cfg) -> tuple[dict | None, str | None]:
    if sparse_cfg is None:
        return None, None
    if isinstance(sparse_cfg, str):
        if sparse_cfg != "checkpoint":
            raise TypeError("sparse_cfg must be 'checkpoint', a mapping, or None")
        detected = load_from_checkpoint_metadata(
            getattr(_model_config(model_runner), "hf_config", None)
        )
        if detected is None:
            return None, None
        return detected
    if not isinstance(sparse_cfg, Mapping):
        raise TypeError("sparse_cfg must be 'checkpoint', a mapping, or None")
    return dict(sparse_cfg), "EXPLICIT"


def _sparse_kwargs(name: str, sparse_cfg: dict | None) -> dict[str, Any]:
    if sparse_cfg is None:
        return {}
    layer_cfg = match_sparse_config(name, sparse_cfg)
    if layer_cfg is None or not layer_cfg.get("enable", True):
        return {}
    return attention_plugin._build_sparse_kw(layer_cfg)


def _require_supported_vllm() -> None:
    # The adapters rely on vLLM writing the current K/V before impl.forward;
    # that external cache-update contract starts in vLLM 0.15.
    if version.parse(vllm.__version__) < version.parse("0.15.0"):
        raise RuntimeError("ModelOpt vLLM attention transforms require vLLM >= 0.15.0")


def _cudagraph_mode(model_runner):
    # Imported lazily so missing quant-only APIs do not affect a sparse no-op.
    from vllm.config.compilation import CUDAGraphMode

    config = getattr(model_runner, "vllm_config", None)
    compilation = getattr(config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    return mode if mode is not None else CUDAGraphMode.NONE


def _global_errors(model_runner) -> list[str]:
    config = getattr(model_runner, "vllm_config", None)
    if config is None:
        return ["model_runner.vllm_config is required"]

    from vllm.config.compilation import CUDAGraphMode

    parallel = getattr(config, "parallel_config", None)
    cache_config = getattr(config, "cache_config", None)
    model_config = _model_config(model_runner)
    if parallel is None or cache_config is None or model_config is None:
        return ["vLLM parallel, cache, and model configs are required"]

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
    if _cudagraph_mode(model_runner).mixed_mode() == CUDAGraphMode.FULL:
        errors.append("FULL mixed-batch cudagraph mode is unsupported")
    if getattr(model_config, "dtype", None) not in (torch.float16, torch.bfloat16):
        errors.append("resolved model/KV-cache dtype must be fp16 or bf16")
    cache_dtype = getattr(cache_config, "cache_dtype", "auto")
    if str(cache_dtype) not in {
        "auto",
        "bfloat16",
        "float16",
        "torch.bfloat16",
        "torch.float16",
    }:
        errors.append(f"resolved KV-cache dtype {cache_dtype!r} must be fp16 or bf16")
    return errors


def _layer_errors(module) -> list[str]:
    impl = getattr(module, "impl", None)
    errors = []
    original_layout = next(
        (
            cls
            for cls in type(module).__mro__
            if cls is _VLLM_ATTENTION
            or (cls.__module__.startswith("vllm.") and issubclass(cls, _VLLM_ATTENTION))
        ),
        None,
    )
    if original_layout is not _VLLM_ATTENTION:
        errors.append(f"layout {type(module).__name__} is not regular decoder self-attention")
    attn_type = getattr(module, "attn_type", None)
    if getattr(attn_type, "value", attn_type) != "decoder":
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


def _device_capability_error(device) -> str | None:
    if device is None:
        return None
    device = torch.device(device)
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        return (
            "NVFP4 attention requires CUDA compute capability >= 8.9 (Ada/Hopper+); "
            f"got sm_{major}{minor}"
        )
    return None


def _sparse_graph_error(sparse_kw: dict[str, Any], mode) -> str | None:
    from vllm.config.compilation import CUDAGraphMode

    params = sparse_kw.get("threshold_scale_factor")
    if (
        mode.decode_mode() == CUDAGraphMode.FULL
        and isinstance(params, dict)
        and isinstance(params.get("decode"), dict)
    ):
        return "calibrated decode skip-softmax requires a non-FULL CUDA graph mode"
    return None


def _is_modelopt_flashinfer_impl(impl) -> bool:
    flashinfer_cls = attention_plugin._FLASHINFER_IMPL_CLS
    return flashinfer_cls is not None and isinstance(impl, flashinfer_cls)


def _select_new_impl(module) -> tuple[object | None, bool, str | None]:
    old_impl = module.impl
    try:
        if isinstance(old_impl, attention_plugin.ModelOptSparseAttentionImpl):
            new_cls = type(old_impl)
            requires_flashinfer_patch = False
        elif _is_modelopt_flashinfer_impl(old_impl):
            new_cls = type(old_impl)
            requires_flashinfer_patch = True
        elif isinstance(old_impl, FlashAttentionImpl):
            new_cls = attention_plugin.ModelOptSparseAttentionImpl
            requires_flashinfer_patch = False
        else:
            try:
                flashinfer_module = importlib.import_module("vllm.v1.attention.backends.flashinfer")
            except ImportError:
                flashinfer_impl_cls = ()
            else:
                flashinfer_impl_cls = flashinfer_module.FlashInferImpl
            if not isinstance(old_impl, flashinfer_impl_cls):
                return (
                    None,
                    False,
                    (
                        f"backend {type(old_impl).__name__} is not supported; "
                        "expected FlashAttentionImpl or FlashInferImpl"
                    ),
                )
            new_cls = attention_plugin.get_flashinfer_sparse_impl_cls()
            requires_flashinfer_patch = True
        return (
            attention_plugin._clone_sparse_impl(old_impl, new_cls),
            requires_flashinfer_patch,
            None,
        )
    except (NotImplementedError, TypeError) as err:
        return None, False, str(err)


def _load_quant_plugin():
    # Quantization is optional for the sparse-only public entry point.
    from modelopt.torch.quantization.plugins import vllm as quant_plugin

    return quant_plugin


def _raise_unsupported(errors: list[str], policy: str) -> None:
    if errors:
        raise NotImplementedError(
            f"Unsupported ModelOpt {policy} plan:\n  - " + "\n  - ".join(errors)
        )


def _plan_vllm_attention(
    model_runner,
    *,
    quantize: bool,
    sparse_cfg,
    q_format: str = "nvfp4",
    k_format: str = "nvfp4",
    p_format: str = "nvfp4",
    v_format: str = "nvfp4",
) -> _InstallPlan:
    model = _unwrapped_model(model_runner)
    resolved_sparse_cfg, sparse_algorithm = _resolve_sparse_config(model_runner, sparse_cfg)
    candidates = []
    attention_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, _VLLM_ATTENTION):
            continue
        attention_count += 1
        sparse_kw = _sparse_kwargs(name, resolved_sparse_cfg)
        if quantize or sparse_kw:
            candidates.append((name, module, sparse_kw))

    if not candidates and not quantize:
        return _InstallPlan(
            model_runner, (), False, sparse_algorithm, q_format, k_format, p_format, v_format
        )

    _require_supported_vllm()
    errors = _global_errors(model_runner) if quantize else []
    mode = _cudagraph_mode(model_runner) if quantize else None
    quant_plugin: Any = _load_quant_plugin() if quantize else None
    plans = []
    for name, module, sparse_kw in candidates:
        reasons = _layer_errors(module)
        device = dtype = None
        if quantize:
            device, dtype = quant_plugin._get_device_dtype(module)
            model_dtype = getattr(_model_config(model_runner), "dtype", None)
            if model_dtype in (torch.float16, torch.bfloat16):
                dtype = model_dtype
            if device is None or dtype is None:
                reasons.append("device/dtype could not be resolved")
            elif dtype not in (torch.float16, torch.bfloat16):
                reasons.append(f"resolved dtype {dtype} must be fp16 or bf16")
            if capability_error := _device_capability_error(device):
                reasons.append(capability_error)
        if quantize:
            if graph_error := _sparse_graph_error(sparse_kw, mode):
                reasons.append(graph_error)
        new_impl, requires_flashinfer_patch, backend_error = _select_new_impl(module)
        if backend_error:
            reasons.append(backend_error)
        if reasons:
            errors.extend(f"{name or '<root>'}: {reason}" for reason in reasons)
            continue
        plans.append(
            _AttentionPlan(
                name,
                module,
                new_impl,
                sparse_kw,
                device,
                dtype,
                requires_flashinfer_patch,
            )
        )
    if quantize and attention_count == 0:
        errors.append("no regular attention layers were found")
    _raise_unsupported(errors, "NVFP4 attention" if quantize else "sparse attention")
    return _InstallPlan(
        model_runner,
        tuple(plans),
        quantize,
        sparse_algorithm,
        q_format,
        k_format,
        p_format,
        v_format,
    )


def _build_report(plan: _InstallPlan) -> VllmAttentionInstallReport:
    names = tuple(layer.name for layer in plan.layers)
    sparse_names = tuple(layer.name for layer in plan.layers if layer.sparse_kw)
    return VllmAttentionInstallReport(
        installed_layers=names,
        quantized_layers=names if plan.quantize else (),
        sparse_layers=sparse_names,
        backend_counts=dict(Counter(type(layer.new_impl).__name__ for layer in plan.layers)),
        sparse_algorithm=plan.sparse_algorithm,
        cascade_disabled=bool(plan.layers),
    )


def _apply_vllm_attention_plans(plan: _InstallPlan) -> VllmAttentionInstallReport:
    if not plan.layers:
        return _build_report(plan)

    if any(layer.requires_flashinfer_patch for layer in plan.layers):
        if not attention_plugin.patch_flashinfer_metadata_builder():
            raise RuntimeError("Unable to prepare FlashInfer metadata for ModelOpt attention")

    quant_plugin: Any = _load_quant_plugin() if plan.quantize else None
    # Compatibility validation is all-layers-first. From here, any exception is
    # fatal; publish each layer immediately after configuration so earlier layers
    # are never left configured on their native implementation.
    plan.model_runner.cascade_attn_enabled = False
    for layer in plan.layers:
        layer.new_impl.sparse_kw = layer.sparse_kw
        if plan.quantize:
            # Pass cfg only for non-default formats: keeps the default call
            # signature stable for callers/fakes that predate the cfg parameter.
            _cfg_kwargs = (
                {
                    "cfg": quant_plugin.build_vllm_attention_quant_cfg(
                        q_format=plan.q_format,
                        k_format=plan.k_format,
                        p_format=plan.p_format,
                        v_format=plan.v_format,
                    )
                }
                if (plan.q_format, plan.k_format, plan.p_format, plan.v_format)
                != ("nvfp4", "nvfp4", "nvfp4", "nvfp4")
                else {}
            )
            converted = quant_plugin.configure_vllm_nvfp4_attention_quantizers(
                layer.module,
                device=layer.device,
                dtype=layer.dtype,
                **_cfg_kwargs,
            )
            if converted is not None and converted is not layer.module:
                raise RuntimeError("vLLM attention quantization must convert modules in place")
            p_qdq, p_qdq_amax = attention_plugin._p_qdq_from_layer(layer.module)
            v_qdq, v_qdq_amax = attention_plugin._v_qdq_from_layer(layer.module)
            if plan.v_format == "fp8":
                # Per-tensor FP8 V is quantized module-level BEFORE the cache
                # write (each token is self-contained; no block geometry), so
                # the kernel sees pre-QDQ'd V and needs no V transform at all.
                v_qdq, v_qdq_amax = None, None
            layer.new_impl.quant_kw = {
                "p_qdq": p_qdq,
                "p_qdq_amax": p_qdq_amax,
                "v_qdq": v_qdq,
                "v_qdq_amax": v_qdq_amax,
            }

        missing = object()
        old_query_flag = getattr(layer.module, "_query_quant_in_kernel", missing)
        old_value_flag = getattr(layer.module, "_value_quant_in_kernel", missing)
        if plan.quantize:
            # fp8 Q is module-level (bf16 losslessly carries E4M3 QDQ values);
            # the kernel then runs a plain bf16 BMM1 with no Q transform.
            layer.module._query_quant_in_kernel = plan.q_format != "fp8"
            layer.module._value_quant_in_kernel = plan.v_format != "fp8"
        try:
            # Publish the adapter last so a native impl never runs with in-kernel
            # quantization flags that only the ModelOpt adapter understands.
            layer.module.impl = layer.new_impl
        except Exception:
            if plan.quantize:
                for name, value in (
                    ("_query_quant_in_kernel", old_query_flag),
                    ("_value_quant_in_kernel", old_value_flag),
                ):
                    if value is missing:
                        delattr(layer.module, name)
                    else:
                        setattr(layer.module, name, value)
            raise
    return _build_report(plan)


def install_vllm_sparse_attention_from_checkpoint(
    model_runner,
) -> VllmAttentionInstallReport:
    """Install checkpoint-configured sparse attention into a loaded vLLM model.

    Missing or inactive ``sparse_attention_config`` metadata is a no-op. All
    known compatibility errors are validated before any attention module is
    changed.
    """
    return _apply_vllm_attention_plans(
        _plan_vllm_attention(model_runner, quantize=False, sparse_cfg="checkpoint")
    )


def install_vllm_nvfp4_attention(
    model_runner,
    *,
    sparse_cfg="checkpoint",
    q_format: str = "nvfp4",
    k_format: str = "nvfp4",
    p_format: str = "nvfp4",
    v_format: str = "nvfp4",
) -> VllmAttentionInstallReport:
    """Install fixed NVFP4 attention with optional checkpoint sparsity.

    Args:
        model_runner: A loaded vLLM model runner.
        sparse_cfg: ``"checkpoint"`` to consume optional exported metadata, a
            resolved sparse config dict, or ``None`` for NVFP4-only attention.
    """
    for name, fmt in (
        ("q_format", q_format),
        ("k_format", k_format),
        ("p_format", p_format),
        ("v_format", v_format),
    ):
        if fmt not in ("nvfp4", "fp8"):
            raise ValueError(f"{name} must be 'nvfp4' or 'fp8', got {fmt!r}")
    return _apply_vllm_attention_plans(
        _plan_vllm_attention(
            model_runner,
            quantize=True,
            sparse_cfg=sparse_cfg,
            q_format=q_format,
            k_format=k_format,
            p_format=p_format,
            v_format=v_format,
        )
    )
