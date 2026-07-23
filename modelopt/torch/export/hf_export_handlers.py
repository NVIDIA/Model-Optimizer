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

import torch
import torch.nn as nn

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, quantizer_attr_names

from .layer_utils import get_expert_linear_names, is_quantlinear, set_expert_quantizer_amax
from .model_config import QUANTIZATION_NONE, QUANTIZATION_NVFP4
from .moe_utils import _export_fused_experts
from .quant_utils import (
    get_activation_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    to_quantized_weight,
)
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


def _validate_dynamic_nvfp4_quantizer(
    quantizer: TensorQuantizer | None,
    role: str,
) -> TensorQuantizer:
    """Validate the exact quantizer contract used by Conv3d implicit GEMM."""
    if not isinstance(quantizer, TensorQuantizer):
        raise NotImplementedError(
            f"Conv3d {role} quantizer must be a TensorQuantizer, got "
            f"{type(quantizer).__name__ if quantizer is not None else 'None'}."
        )
    if not quantizer.is_enabled or not quantizer._if_quant:
        raise NotImplementedError(f"Conv3d {role} quantizer must be enabled for quantization.")
    if (
        not quantizer.is_nvfp4_dynamic
        or quantizer.block_sizes.get(-1) != 16
        or quantizer.axis is not None
    ):
        raise NotImplementedError(
            f"Conv3d {role} quantizer must use per-tensor dynamic block-16 NVFP4."
        )
    return quantizer


def _export_quantized_conv3d_weight(
    sub_module: nn.Module,
    dtype: torch.dtype,
    weight_name: str = "weight",
) -> None:
    """Export a supported Conv3d as logical flattened-K dynamic NVFP4 tensors.

    The only supported contract is an ordinary, ungrouped Conv3d with full
    NVFP4 weight/input quantization and a dynamic weight quantizer. Other
    formats fail closed because their live fake-quant path does not share this
    canonical flattened reduction axis.
    """
    if not isinstance(sub_module, nn.Conv3d):
        raise TypeError(f"Expected nn.Conv3d, got {type(sub_module).__name__}.")
    if sub_module.groups != 1:
        raise NotImplementedError(
            f"Grouped Conv3d export is not supported; got groups={sub_module.groups}."
        )

    quantizer_attrs = quantizer_attr_names(weight_name)
    weight: nn.Parameter = getattr(sub_module, weight_name)
    if weight.ndim != 5:
        raise ValueError(f"Conv3d weight must be rank 5, got shape {tuple(weight.shape)}.")
    if not weight.is_floating_point():
        raise ValueError(f"Conv3d weight must be floating point, got {weight.dtype}.")

    weight_quantizer = _validate_dynamic_nvfp4_quantizer(
        getattr(sub_module, quantizer_attrs.weight_quantizer, None), "weight"
    )
    input_quantizer = _validate_dynamic_nvfp4_quantizer(
        getattr(sub_module, quantizer_attrs.input_quantizer, None), "input"
    )
    output_quantizer = getattr(sub_module, quantizer_attrs.output_quantizer, None)
    if output_quantizer is not None and getattr(output_quantizer, "is_enabled", False):
        raise NotImplementedError("Conv3d output quantization is not supported for export.")
    quantization_format = get_quantization_format(sub_module)
    if quantization_format != QUANTIZATION_NVFP4:
        raise NotImplementedError(
            "Conv3d export supports only full dynamic NVFP4; "
            f"got quantization format {quantization_format!r}."
        )

    block_size = get_weight_block_size(sub_module, weight_name)
    weight_flat = weight.reshape(weight.shape[0], -1)
    k_flat = weight_flat.shape[-1]
    k_padded = ((k_flat + block_size - 1) // block_size) * block_size
    if k_padded != k_flat:
        weight_flat = torch.nn.functional.pad(weight_flat, (0, k_padded - k_flat))

    weight_scale_2 = None
    if weight_quantizer.amax is not None:
        if weight_quantizer.amax.numel() != 1 or not torch.all(
            torch.isfinite(weight_quantizer.amax) & (weight_quantizer.amax > 0)
        ):
            raise ValueError(
                "Conv3d weight quantizer amax must be a finite positive scalar when calibrated."
            )
        weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)
    if weight_scale_2 is None:
        weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2(weight_flat)
    weight_scale_2 = weight_scale_2.to(device=weight.device, dtype=torch.float32).squeeze()
    if not torch.all(torch.isfinite(weight_scale_2)) or not torch.all(weight_scale_2 > 0):
        raise ValueError(
            f"Conv3d weight_scale_2 must be finite and positive, got {weight_scale_2}."
        )
    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight_flat,
        block_size=block_size,
        weights_scaling_factor_2=weight_scale_2,
    )[0]
    quantized_weight = to_quantized_weight(
        weight_flat.to(dtype),
        weight_scale,
        QUANTIZATION_NVFP4,
        weight_scale_2,
        block_size,
    )

    if input_quantizer.amax is None:
        raise ValueError("Conv3d input quantizer must be calibrated before export.")
    if input_quantizer.amax.numel() != 1 or not torch.all(
        torch.isfinite(input_quantizer.amax) & (input_quantizer.amax > 0)
    ):
        raise ValueError("Conv3d input quantizer amax must be a finite positive scalar.")
    input_scale = get_activation_scaling_factor(
        sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
    )

    # Commit only after every validation and tensor conversion succeeds.
    setattr(sub_module, weight_name, nn.Parameter(quantized_weight, requires_grad=False))
    sub_module.register_buffer(quantizer_attrs.weight_scale, weight_scale)
    sub_module.register_buffer(quantizer_attrs.weight_scale_2, weight_scale_2)
    sub_module.register_buffer(quantizer_attrs.input_scale, input_scale.squeeze())

    torch.cuda.empty_cache()


@ExportModuleRegistry.register(
    "WanCausalConv3d", predicate=lambda module: hasattr(module, "weight_quantizer")
)
def _export_quant_conv3d(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export a supported quantized Wan Conv3d through its MRO registry match."""
    quantizers = [
        getattr(module, attr, None)
        for attr in ("weight_quantizer", "input_quantizer", "output_quantizer")
    ]
    if not any(getattr(quantizer, "is_enabled", False) for quantizer in quantizers):
        return

    try:
        with fsdp2_aware_weight_update(ctx.model, module, reshard=False):
            _export_quantized_conv3d_weight(module, ctx.dtype)
    except (AssertionError, NotImplementedError, TypeError, ValueError) as e:
        message = f"Failed to export Conv3d '{name}' (type={type(module).__name__}): {e}"
        raise type(e)(message) from e


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
