# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face PEFT ownership for the trainable SVDQuant residual."""

import re
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import Linear as LoraLinear

from ..nn import SVDQuantLinear

__all__ = []

_SVDQUANT_ADAPTER_NAME = "modelopt_svdquant"


class _SVDQuantPeftLinear(LoraLinear):
    """PEFT layer implementing ``Q(W_residual)x + B(Ax)`` for SVDQuant."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the AWQ pre-scale exactly once to both SVDQuant branches."""
        base_layer = self.get_base_layer()
        scaled_x = base_layer._apply_pre_quant_scale(x)
        with base_layer.input_quantizer.disable_pre_quant_scale():
            return super().forward(scaled_x, *args, **kwargs)


def _svdquant_peft_config(target_names: list[str], rank: int) -> LoraConfig:
    target_regex = "^(?:" + "|".join(re.escape(name) for name in target_names) + ")$"
    config = LoraConfig(
        task_type=None,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=target_regex,
        bias="none",
        lora_bias=False,
        modules_to_save=None,
        fan_in_fan_out=False,
        use_rslora=False,
        use_dora=False,
        init_lora_weights=False,
        inference_mode=False,
    )
    config._register_custom_module({SVDQuantLinear: _SVDQuantPeftLinear})
    return config


def _delete_quantizer_svdquant_factors(weight_quantizer: nn.Module) -> None:
    """Remove calibration factors stored as buffers or plain quantizer attributes."""
    for public_name in ("svdquant_lora_a", "svdquant_lora_b"):
        for storage_name in (f"_{public_name}", public_name):
            if (
                storage_name in weight_quantizer.__dict__
                or storage_name in weight_quantizer._buffers
            ):
                delattr(weight_quantizer, storage_name)
                break


def _inject_svdquant_peft(
    model: nn.Module,
    target_names: list[str],
    rank: int,
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> dict[str, Any]:
    target_names = sorted(target_names)
    base_trainability = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    config = _svdquant_peft_config(target_names, rank)
    inject_adapter_in_model(
        config,
        model,
        adapter_name=_SVDQUANT_ADAPTER_NAME,
        low_cpu_mem_usage=False,
    )
    for parameter, requires_grad in base_trainability:
        parameter.requires_grad_(requires_grad)

    for name in target_names:
        module = model.get_submodule(name)
        lora_a = module.lora_A[_SVDQUANT_ADAPTER_NAME].weight
        lora_b = module.lora_B[_SVDQUANT_ADAPTER_NAME].weight
        if factors is not None:
            source_a, source_b = factors[name]
            with torch.no_grad():
                lora_a.copy_(source_a.to(device=lora_a.device, dtype=lora_a.dtype))
                lora_b.copy_(source_b.to(device=lora_b.device, dtype=lora_b.dtype))
            _delete_quantizer_svdquant_factors(module.get_base_layer().weight_quantizer)
        lora_a.requires_grad_(True)
        lora_b.requires_grad_(True)

    return {
        "rank": rank,
        "target_modules": target_names,
    }


def _externalize_svdquant_lora(model: nn.Module, rank: int) -> dict[str, Any] | None:
    """Move calibrated SVDQuant factors from weight quantizers into HF PEFT."""
    factors = {}
    for name, module in model.named_modules():
        if not isinstance(module, SVDQuantLinear):
            continue
        lora_a = getattr(module.weight_quantizer, "svdquant_lora_a", None)
        lora_b = getattr(module.weight_quantizer, "svdquant_lora_b", None)
        if lora_a is not None and lora_b is not None:
            factors[name] = (lora_a.detach(), lora_b.detach())
    if not factors:
        return None
    return _inject_svdquant_peft(model, list(factors), rank, factors)


def _restore_svdquant_peft(model: nn.Module, metadata: dict[str, Any]) -> None:
    """Rebuild the PEFT topology before the complete model state is loaded."""
    _inject_svdquant_peft(
        model,
        metadata["target_modules"],
        metadata["rank"],
        factors=None,
    )
