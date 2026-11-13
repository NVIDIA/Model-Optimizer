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

"""Kimi-K3 PTQ modeling for vLLM (HF model type ``kimi_k3``)."""

import sys

import torch
import vllm.model_executor.layers.linear as vllm_linear

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.custom import CUSTOM_MODEL_PLUGINS
from modelopt.torch.quantization.plugins.vllm import _VLLMParallelLinear
from modelopt.torch.quantization.plugins.vllm_mega_moe import _QuantVLLMMegaMoEExperts

__all__: list[str] = []

# NotImplementedError: some vLLM model packages reject unsupported platforms at import.
try:
    from vllm.models.kimi_k3.nvidia.model import KimiK3MegaMoEExperts
except (ImportError, NotImplementedError):
    KimiK3MegaMoEExperts = None


def _kimi_mlp_bf16_linear_supports_fake_quant(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.dtype == torch.bfloat16
        and weight.layout == torch.strided
        and not weight.is_quantized
    )


def _kimi_down_proj_has_enabled_quantizers(down_proj: torch.nn.Module) -> bool:
    return any(
        (quantizer := getattr(down_proj, name, None)) is not None and quantizer.is_enabled
        for name in _VLLMParallelLinear._QUANTIZER_NAMES
    )


class _KimiGemmRsFakeQuantGuard:
    """Keep vLLM 0.28 GEMM-RS when the down projection is not fake-quantized."""

    def __init__(self, down_proj: vllm_linear.RowParallelLinear) -> None:
        self.down_proj = down_proj

    def __bool__(self) -> bool:
        return not _kimi_down_proj_has_enabled_quantizers(self.down_proj)


class _KimiGemmRsArFakeQuantGuard:
    """Keep Kimi's fused collective GEMM for projections without enabled fake quantizers."""

    def __init__(self, gemm_rs_ar, down_proj: _VLLMParallelLinear) -> None:
        self.gemm_rs_ar = gemm_rs_ar
        self.down_proj = down_proj

    def should_run(self, x: torch.Tensor) -> bool:
        return not _kimi_down_proj_has_enabled_quantizers(
            self.down_proj
        ) and self.gemm_rs_ar.should_run(x)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.gemm_rs_ar(x, weight)


def _prepare_kimi_mlp_fake_quant(module: torch.nn.Module, kimi_method_cls: type | None) -> None:
    """Mark supported Kimi MLP linears and bypass fused GEMM when quantized."""
    for linear in (module.gate_up_proj, module.down_proj):
        if (
            kimi_method_cls is not None
            and type(getattr(linear, "quant_method", None)) is kimi_method_cls
            and _kimi_mlp_bf16_linear_supports_fake_quant(linear)
        ):
            linear._fake_quant_supported_by_model = True

    down_proj = module.down_proj
    if not isinstance(down_proj, (vllm_linear.RowParallelLinear, _VLLMParallelLinear)):
        return
    # vLLM 0.28 checks this flag before calling down_proj.forward.
    if getattr(module, "run_gemm_rs", False) is True:
        module.run_gemm_rs = _KimiGemmRsFakeQuantGuard(down_proj)
    gemm_rs_ar = getattr(module, "gemm_rs_ar", None)
    if gemm_rs_ar is not None and not isinstance(gemm_rs_ar, _KimiGemmRsArFakeQuantGuard):
        # QuantModule conversion updates down_proj in place, preserving this reference.
        module.gemm_rs_ar = _KimiGemmRsArFakeQuantGuard(gemm_rs_ar, down_proj)


def prepare_kimi_mlps_on_the_fly(model: torch.nn.Module) -> None:
    """Prepare loaded Kimi MLP modules before conversion."""
    # Inspect loaded Kimi modules so other models never import K3 during conversion.
    kimi_module = sys.modules.get("vllm.models.kimi_k3.nvidia.model")
    kimi_mlp_cls = getattr(kimi_module, "KimiMLP", None)
    if kimi_mlp_cls is None:
        return
    low_latency_module = sys.modules.get("vllm.models.kimi_k3.nvidia.low_latency_gemm")
    kimi_method_cls = getattr(low_latency_module, "KimiK3LowLatencyLinearMethod", None)
    for m in model.modules():
        if isinstance(m, kimi_mlp_cls):
            _prepare_kimi_mlp_fake_quant(m, kimi_method_cls)


CUSTOM_MODEL_PLUGINS.add(prepare_kimi_mlps_on_the_fly)

if KimiK3MegaMoEExperts is not None:
    # Overrides forward, so DeepSeek-V4's registration does not cover it.
    QuantModuleRegistry.register({KimiK3MegaMoEExperts: "vllm_KimiK3MegaMoEExperts"})(
        _QuantVLLMMegaMoEExperts
    )
