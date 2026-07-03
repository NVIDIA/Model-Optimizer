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

"""Fixed NVFP4 Q/K/P/V worker for ModelOpt sparse attention on vLLM."""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.vllm import (
    _ATTENTION_TYPES,
    _get_device_dtype,
    _set_vllm_attention_kv_default_amax,
    disable_compilation,
    vllm_attention,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.sparse_attn_config import (
    load_from_checkpoint_metadata,
    match_sparse_config,
)
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    _build_sparse_kw,
    _clone_sparse_impl,
    _p_qdq_from_layer,
    _v_qdq_from_layer,
)

__all__ = ["QuantSparseAttnWorker"]

VLLMAttention = vllm_attention.Attention
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


def _unwrapped_model(worker):
    model = worker.model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _global_errors(worker) -> list[str]:
    config = worker.model_runner.vllm_config
    parallel = config.parallel_config
    cache = config.cache_config
    model_config = config.model_config
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
    if config.compilation_config.cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL:
        errors.append("FULL mixed-batch cudagraph mode is unsupported")
    if getattr(model_config, "dtype", None) not in (torch.float16, torch.bfloat16):
        errors.append("resolved model/KV-cache dtype must be fp16 or bf16")
    cache_dtype = getattr(cache, "cache_dtype", "auto")
    if str(cache_dtype) not in {"auto", "bfloat16", "float16", "torch.bfloat16", "torch.float16"}:
        errors.append(f"resolved KV-cache dtype {cache_dtype!r} must be fp16 or bf16")
    return errors


def _layer_errors(module) -> list[str]:
    impl = getattr(module, "impl", None)
    errors = []
    if type(module) is not VLLMAttention:
        errors.append(f"layout {type(module).__name__} is not regular decoder self-attention")
    if getattr(module, "attn_type", None) != AttentionType.DECODER:
        errors.append("attn_type must be DECODER")
    if not isinstance(impl, FlashAttentionImpl):
        errors.append(f"backend {type(impl).__name__} is not FlashAttentionImpl")
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


def _validated_attention_plans(worker):
    """Validate every attention layout without mutation, then return regular-layer tuples."""
    model = _unwrapped_model(worker)
    detected = load_from_checkpoint_metadata(worker.model_runner.model_config.hf_config)
    sparse_cfg = detected[0] if detected is not None else None
    errors = _global_errors(worker)
    plans = []
    attention_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, _ATTENTION_TYPES):
            continue
        attention_count += 1
        reasons = _layer_errors(module)
        device, dtype = _get_device_dtype(module)
        if device is None or dtype is None:
            reasons.append("device/dtype could not be resolved")
        elif dtype not in (torch.float16, torch.bfloat16):
            reasons.append(f"resolved dtype {dtype} must be fp16 or bf16")
        if reasons:
            errors.extend(f"{name or '<root>'}: {reason}" for reason in reasons)
            continue
        layer_cfg = match_sparse_config(name, sparse_cfg) if sparse_cfg is not None else None
        sparse_kw = (
            _build_sparse_kw(layer_cfg)
            if layer_cfg is not None and layer_cfg.get("enable", True)
            else {}
        )
        plans.append((name, module, sparse_kw, device, dtype))
    if attention_count == 0:
        errors.append("no regular attention layers were found")
    if errors:
        raise NotImplementedError(
            "Unsupported ModelOpt attention plan:\n  - " + "\n  - ".join(errors)
        )
    return plans


def _install_quant_sparse_attn(worker) -> None:
    plans = _validated_attention_plans(worker)
    for _name, module, sparse_kw, device, dtype in plans:
        module.device, module.dtype = device, dtype
        QuantModuleRegistry.convert(module)
        module.p_bmm_quantizer = TensorQuantizer()
        set_quantizer_by_cfg(module, _BMM_CFG)
        _set_vllm_attention_kv_default_amax(module, device)
        new_impl = _clone_sparse_impl(module.impl)
        new_impl.sparse_kw = sparse_kw
        p_qdq, p_qdq_amax = _p_qdq_from_layer(module)
        v_qdq, v_qdq_amax = _v_qdq_from_layer(module)
        new_impl.quant_kw = {
            "p_qdq": p_qdq,
            "p_qdq_amax": p_qdq_amax,
            "v_qdq": v_qdq,
            "v_qdq_amax": v_qdq_amax,
        }
        module.impl = new_impl
        module._value_quant_in_kernel = True
    worker.model_runner.cascade_attn_enabled = False
    print(f"[ModelOpt] Installed NVFP4 quant+sparse attention on {len(plans)} layers")


class QuantSparseAttnWorker(BaseWorker):
    """Install the fixed attention-only recipe before vLLM warmup and graph capture."""

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        with disable_compilation(_unwrapped_model(self)):
            return BaseWorker.determine_available_memory(self)

    def compile_or_warm_up_model(self) -> float:
        _install_quant_sparse_attn(self)
        return BaseWorker.compile_or_warm_up_model(self)
