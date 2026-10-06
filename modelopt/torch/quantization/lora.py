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

"""Mergeable low-rank adaptation of fake-quantized linear weights."""

import fnmatch
import math

import torch
from torch import nn

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.opt.mode import ModeDescriptor

from .config import QuantizeConfig
from .conversion import restore_quantizer_state, update_quantize_metadata
from .mode import QuantizeModeRegistry
from .nn import QuantModule, QuantModuleRegistry
from .qtensor import QTensorWrapper

__all__ = ["QuantLoRAConfig", "enable_quant_lora", "merge_quant_lora"]


class QuantLoRAConfig(ModeloptBaseConfig):
    """Configuration for adapters applied to combined weights before fake quantization."""

    rank: int = ModeloptField(8, gt=0)
    alpha: float = ModeloptField(16.0, gt=0, allow_inf_nan=False)
    target_modules: list[str] = ModeloptField(["*"], min_length=1)


QuantLoRARegistry = _DMRegistryCls("quant_lora")


@QuantLoRARegistry.register({QuantModuleRegistry[nn.Linear]: "nn.Linear"})
class _QuantLoRALinear(DynamicModule):
    def _setup(self, config: QuantLoRAConfig):
        """Initialize factors with a zero effective update."""
        weight = self._parameters["weight"]
        self.lora_A = nn.Parameter(weight.new_empty(config.rank, weight.shape[1]))
        self.lora_B = nn.Parameter(weight.new_zeros(weight.shape[0], config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scale = config.alpha / config.rank

    @staticmethod
    def _effective_weight(module, weight):
        """Combine the frozen weight and trainable low-rank update."""
        return weight + (module.lora_scale * module.lora_B @ module.lora_A).to(weight.dtype)

    def forward(self, *args, **kwargs):
        """Run the quantized base layer on the combined weight."""
        # Keep the combined weight outside quantizers and TE custom autograd functions.
        weight = self._parameters["weight"]
        self._parameters["weight"] = self._effective_weight(self, weight)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._parameters["weight"] = weight

    @torch.no_grad()
    def merge_lora(self):
        """Fold the update into the base weight and remove the factors."""
        weight = self._parameters["weight"]
        weight.copy_(self._effective_weight(self, weight))
        self.export()
        del self.lora_A, self.lora_B, self.lora_scale


def enable_quant_lora(model: nn.Module, config: dict | QuantLoRAConfig | None = None) -> nn.Module:
    """Freeze a fake-quantized backbone and add zero-initialized, mergeable adapters.

    The forward quantizes ``W + (alpha / rank) * B @ A``. Save and restore using
    ModelOpt checkpoint APIs to reconstruct adapters before loading tensor/optimizer state.
    """
    if any(isinstance(module, _QuantLoRALinear) for module in model.modules()):
        raise ValueError("Quantization-aware LoRA is already enabled.")
    return apply_mode(model, [("quant_lora", config or {})], registry=QuantizeModeRegistry)


def merge_quant_lora(model: nn.Module) -> nn.Module:
    """Merge adapters into floating-point weights, retaining quantizers for deployment export."""
    if not any(isinstance(module, _QuantLoRALinear) for module in model.modules()):
        return model
    return apply_mode(model, "quant_lora_export", registry=QuantizeModeRegistry)


def _convert_quant_lora(model, config):
    """Validate all targets before freezing the backbone and adding factors."""
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantModule) or not hasattr(module, "weight_quantizer"):
            continue
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in config.target_modules):
            continue
        weight = module._parameters.get("weight")
        if isinstance(weight, QTensorWrapper):
            raise ValueError(
                "Quantization-aware LoRA requires uncompressed floating-point weights."
            )
        if weight is None or weight.ndim != 2 or type(module) not in QuantLoRARegistry:
            raise ValueError(
                f"Quantization-aware LoRA does not support layer {name}: {type(module)}"
            )
        targets.append(module)
    if not targets:
        raise ValueError("No supported fake-quantized linear layers match target_modules.")
    model.requires_grad_(False)
    for module in targets:
        QuantLoRARegistry[type(module)].convert(module, config=config)
    metadata = {}
    _update_quant_lora(model, config, metadata)
    return model, metadata


def _restore_quant_lora(model, config, metadata):
    """Restore quantizers and recreate factors before loading checkpoint tensors."""
    restore_quantizer_state(model, QuantizeConfig(), metadata)
    return _convert_quant_lora(model, config)[0]


def _update_quant_lora(model, config, metadata):
    """Record quantizer metadata for adapter and merged checkpoints."""
    update_quantize_metadata(model, QuantizeConfig(), metadata)


def _merge_quant_lora(model, config):
    """Merge every adapter and record the remaining quantizers."""
    for module in list(model.modules()):
        if isinstance(module, _QuantLoRALinear):
            module.merge_lora()
    metadata = {}
    _update_quant_lora(model, config, metadata)
    return model, metadata


def _restore_merged_quant_lora(model, config, metadata):
    """Remove reconstructed factors before restoring the merged checkpoint state."""
    model = _merge_quant_lora(model, config)[0]
    return restore_quantizer_state(model, QuantizeConfig(), metadata)


@QuantizeModeRegistry.register_mode
class QuantLoRAModeDescriptor(ModeDescriptor):
    """Record adapters after quantization and reconstruct them before checkpoint loading."""

    name = "quant_lora"
    config_class = QuantLoRAConfig
    export_mode = "quant_lora_export"
    next_prohibited_modes = {"quant_lora", "quantize", "auto_quantize", "real_quantize"}
    convert = staticmethod(_convert_quant_lora)
    restore = staticmethod(_restore_quant_lora)
    update_for_save = staticmethod(_update_quant_lora)
    update_for_new_mode = staticmethod(_update_quant_lora)


@QuantizeModeRegistry.register_mode
class QuantLoRAExportModeDescriptor(ModeDescriptor):
    """Record merged adapters so exported checkpoints restore without adapter parameters."""

    name = "quant_lora_export"
    config_class = ModeloptBaseConfig
    is_export_mode = True
    convert = staticmethod(_merge_quant_lora)
    restore = staticmethod(_restore_merged_quant_lora)
    update_for_save = staticmethod(_update_quant_lora)
    update_for_new_mode = staticmethod(_update_quant_lora)
