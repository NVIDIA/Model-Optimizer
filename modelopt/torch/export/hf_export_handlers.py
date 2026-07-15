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

"""Built-in module handlers for unified Hugging Face export."""

import collections.abc
import warnings

import torch.nn as nn

from modelopt.torch.quantization.utils import fsdp2_aware_weight_update

from .layer_utils import get_expert_linear_names, is_quantlinear, set_expert_quantizer_amax
from .model_config import QUANTIZATION_NONE
from .moe_utils import _export_fused_experts
from .quant_utils import get_quantization_format
from .registry import ExportContext, ExportModuleRegistry, PrepareMoEInputsRegistry

__all__: list[str] = []


def _has_fused_experts_quantizers(module: nn.Module) -> bool:
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    return hasattr(module, f"{first_proj_attr}_weight_quantizers")


def _export_weight(
    module: nn.Module,
    ctx: ExportContext,
    weight_name: str = "weight",
) -> None:
    # Imported lazily to avoid a cycle: unified_export_hf imports this module to
    # install the built-in handlers while retaining this legacy helper's import path.
    from .unified_export_hf import _export_quantized_weight

    _export_quantized_weight(module, ctx.dtype, weight_name, _tied_cache=ctx.tied_cache)


# Preparation handlers are registered in the same precedence as the legacy MoE prepass.


# Keyed on the mixin class name too: the generated class is normally named
# "QuantDbrxExperts", but _DMRegistryCls falls back to a module-prefixed name on
# collision, while "_QuantDbrxExperts" remains in the generated class's MRO.
@PrepareMoEInputsRegistry.register("QuantDbrxExperts", "_QuantDbrxExperts")
def _prepare_dbrx_experts(name: str, moe_module: nn.Module, ctx: ExportContext) -> None:
    """Fill missing input amax values for DBRX per-expert ModuleLists."""
    experts_mlp = moe_module.experts.mlp
    for linear_name in get_expert_linear_names(moe_module):
        if hasattr(experts_mlp, linear_name):
            linear_modulelist = getattr(experts_mlp, linear_name)
            if hasattr(linear_modulelist, "__iter__"):
                set_expert_quantizer_amax(
                    modules=list(linear_modulelist),
                    quantizer_attrs=["input_quantizer"],
                )


@PrepareMoEInputsRegistry.register(predicate=_has_fused_experts_quantizers)
def _prepare_fused_experts(name: str, moe_module: nn.Module, ctx: ExportContext) -> None:
    """Mark fused experts handled; their missing amax fallback occurs during export."""


@PrepareMoEInputsRegistry.register("Llama4TextExperts", "GptOssExperts")
def _prepare_bmm_experts(name: str, moe_module: nn.Module, ctx: ExportContext) -> None:
    """Fill missing input amax values for fused BMM-style experts."""
    # Both use gate_up_proj and down_proj with singular input quantizers
    # (gate_up_proj_input_quantizer/down_proj_input_quantizer); the weight-side
    # amax fallback and weight export happen in _export_bmm_experts.
    for linear_name in ["gate_up_proj", "down_proj"]:
        if hasattr(moe_module.experts, linear_name):
            linear_module = getattr(moe_module.experts, linear_name)
            if hasattr(linear_module, "input_quantizer"):
                set_expert_quantizer_amax(
                    modules=[linear_module],
                    quantizer_attrs=["input_quantizer"],
                )


@PrepareMoEInputsRegistry.register(
    predicate=lambda module: isinstance(module, collections.abc.Iterable)
)
def _prepare_iterable_experts(name: str, moe_module: nn.Module, ctx: ExportContext) -> None:
    """Fill missing input amax values for iterable per-expert submodules."""
    expert_linear_names = get_expert_linear_names(moe_module)
    linear_name = None
    try:
        for linear_name in expert_linear_names:
            set_expert_quantizer_amax(
                modules=[getattr(expert, linear_name) for expert in moe_module.experts],
                quantizer_attrs=["input_quantizer"],
            )
    except AttributeError as e:
        expert_types = [type(expert).__name__ for expert in moe_module.experts]
        raise AttributeError(
            f"Failed to access attribute '{linear_name}' on experts. "
            f"MoE module type: {type(moe_module).__name__}, "
            f"Expert types: {expert_types}, "
            f"Expected linear names: {expert_linear_names}. "
            f"This suggests the get_expert_linear_names function may need "
            f"to be updated for this model architecture. "
            f"Original error: {e}"
        ) from e


# Export handlers are registered in the same precedence as the legacy model walk.


@ExportModuleRegistry.register(
    "QuantMoELinear", predicate=lambda module: hasattr(module, "experts")
)
def _export_moe_linear(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Fill missing input amax before child expert QuantLinears are exported."""
    set_expert_quantizer_amax(list(module.experts), quantizer_attrs="input_quantizer")


@ExportModuleRegistry.register(predicate=_has_fused_experts_quantizers)
def _export_fused_experts_module(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Split and quantize a fused-experts module with plural weight quantizers."""
    with fsdp2_aware_weight_update(ctx.model, module, reshard=False):
        _export_fused_experts(
            module,
            ctx.dtype,
            _moe_tied_cache=ctx.moe_tied_cache,
            _tied_cache=ctx.tied_cache,
        )


@ExportModuleRegistry.register(predicate=is_quantlinear)
def _export_quant_linear(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export a standard quantized linear layer."""
    if get_quantization_format(module) == QUANTIZATION_NONE:
        return
    try:
        with fsdp2_aware_weight_update(ctx.model, module, reshard=False):
            _export_weight(module, ctx)
    except AssertionError as e:
        raise AssertionError(
            f"Failed to export module '{name}' (type={type(module).__name__}): {e}"
        ) from e


@ExportModuleRegistry.register(
    nn.Embedding, predicate=lambda module: hasattr(module, "weight_quantizer")
)
def _export_quant_embedding(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export a quantized embedding table unless its weight is tied."""
    if get_quantization_format(module) == QUANTIZATION_NONE:
        return
    # Packing replaces .weight, which would sever any Python-level weight tie and
    # leave the other module pointing at a stale float Parameter.
    tied_to = [
        other_name
        for other_name, other_module in ctx.model.named_modules()
        if other_module is not module and getattr(other_module, "weight", None) is module.weight
    ]
    if tied_to:
        warnings.warn(
            f"Skipping quantized weight packing for embedding '{name}': its "
            f"weight Parameter is shared with {tied_to} (weight tying). Packing "
            "would break the tie and produce stale weights in the tied module(s). "
            "The embedding will be exported as its fake-quantized float weight."
        )
        return
    try:
        with fsdp2_aware_weight_update(ctx.model, module, reshard=False):
            _export_weight(module, ctx)
    except AssertionError as e:
        raise AssertionError(
            f"Failed to export embedding '{name}' (type={type(module).__name__}): {e}"
        ) from e


@ExportModuleRegistry.register("Llama4TextExperts", "GptOssExperts")
def _export_bmm_experts(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export fused BMM-style expert weights and quantization metadata."""
    if get_quantization_format(module) == QUANTIZATION_NONE:
        return
    # TODO: consolidate uncalibrated experts handling logic
    set_expert_quantizer_amax(
        modules=module,
        quantizer_attrs=["gate_up_proj_weight_quantizer", "down_proj_weight_quantizer"],
    )
    set_expert_quantizer_amax(
        modules=module,
        quantizer_attrs=["gate_up_proj_input_quantizer", "down_proj_input_quantizer"],
    )
    with fsdp2_aware_weight_update(ctx.model, module, reshard=False):
        for weight_name in ["gate_up_proj", "down_proj"]:
            _export_weight(module, ctx, weight_name)
