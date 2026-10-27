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
import contextlib
import warnings
from dataclasses import dataclass

import torch.nn as nn

from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, module_name_maps
from modelopt.torch.quantization.utils.core_utils import _get_enclosing_fsdp_module
from modelopt.torch.utils.distributed import distribution_plan

from .layer_utils import get_expert_linear_names, is_quantlinear, set_expert_quantizer_amax
from .moe_utils import _export_fused_experts, _export_fused_experts_keep_fused
from .quant_format import QUANTIZATION_NONE
from .quant_utils import get_quantization_format
from .registry import ExportContext, ExportModuleRegistry, PrepareMoEInputsRegistry

__all__: list[str] = []


def _has_fused_experts_quantizers(module: nn.Module) -> bool:
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    return hasattr(module, f"{first_proj_attr}_weight_quantizers")


@dataclass
class _HeldWindow:
    """An FSDP2 unit's weight-update window, and the modules packed in it so far."""

    unit: nn.Module
    modules: list[nn.Module]
    stack: contextlib.ExitStack


def _hold_fsdp2_weight_window(ctx: ExportContext, module: nn.Module) -> None:
    """Hold open the ``fsdp2_aware_weight_update`` window of the FSDP2 unit owning ``module``'s weights.

    The window unshards the whole unit and reshards it when it closes, so the export packs every
    module of a unit in one window: it stays open from one handler to the next until the export
    reaches another unit, and :func:`close_fsdp2_weight_window` closes the last. Each module joins
    the list the window was opened with, whose weights it re-registers with FSDP2 when it closes.

    The window opens on the innermost unit enclosing the module, but only when the root it is handed
    is wrapped itself: on an unwrapped root it does nothing. Expert parallelism FSDP2-shards only the
    dense layers and leaves the root unwrapped, where the weight would stay a sharded DTensor, which
    packing mishandles (``DTensor << int`` is a silent no-op in torch 2.11, so NVFP4 packed unrelated
    codes next to correct scales). So there that unit is handed in as the root, found by walking up
    the module's name.
    """
    from torch.distributed.fsdp import FSDPModule

    if ctx.module_names is None:
        ctx.module_names = module_name_maps(ctx.model)
    unit = _get_enclosing_fsdp_module(module, ctx.model, ctx.module_names) or ctx.model
    held = ctx.fsdp2_window
    if held is not None and held.unit is unit:
        held.modules.append(module)
        return
    close_fsdp2_weight_window(ctx)
    if isinstance(ctx.model, FSDPModule):
        root, names = ctx.model, ctx.module_names
    else:
        # Names relative to the unit, which the window resolves the modules' parameters against.
        root, names = unit, module_name_maps(unit)
    held = _HeldWindow(unit, [module], contextlib.ExitStack())
    held.stack.enter_context(fsdp2_aware_weight_update(root, held.modules, names=names))
    ctx.fsdp2_window = held


def close_fsdp2_weight_window(ctx: ExportContext) -> None:
    """Close the weight-update window the handlers hold open, if any, resharding its unit."""
    held, ctx.fsdp2_window = ctx.fsdp2_window, None
    if held is not None:
        held.stack.close()


def _export_weight(
    module: nn.Module,
    ctx: ExportContext,
    weight_name: str = "weight",
) -> None:
    # Imported lazily to avoid a cycle: unified_export_hf imports this module to
    # install the built-in handlers while retaining this legacy helper's import path.
    from .unified_export_hf import _export_quantized_weight

    # The no-gather writer packs while the model is still sharded, so unlike the
    # collect_export_tensors path there is no open weight-access window here and the weight is a
    # DTensor. FP8 tolerates that; NVFP4 does not -- get_weights_scaling_factor does an
    # index_put_, which DTensor cannot dispatch. Pack in the unit's window for that path only, which
    # reshards it once the export moves on: a layer left unsharded holds all of its weights on every
    # rank, and FSDP2 later puts its own shards back over anything packed in it outside a window.
    if ctx.keep_fused_experts and distribution_plan(ctx.model).is_fsdp_dense:
        _hold_fsdp2_weight_window(ctx, module)
    _export_quantized_weight(module, ctx.dtype, weight_name)


# Preparation handlers are registered in the same precedence as the legacy MoE prepass.


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
    expert_linear_names = get_expert_linear_names(moe_module, ctx.model_type)
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


def _is_quant_moe_linear(module: nn.Module) -> bool:
    """Whether ``module`` is an expert-indexed ``MoELinear`` expanded by ``_QuantMoELinear``.

    Matched by wrapper type first, not only by the dynamically generated class name
    (``Quant`` + the model's own class name): the wrapper is registered structurally, so a
    compatible remote-code class under any other name — or a second one, whose generated
    name gets uniquified — would bypass this handler and export without the input-amax
    fallback. The name check is kept as a fallback so stand-in modules match too.

    The wrapper lives in the Step PTQ modeling module, imported lazily like other export
    modules do to avoid an import cycle with ``modelopt.torch.quantization``.
    """
    if not hasattr(module, "experts"):
        return False
    try:
        from modelopt.torch.models.step3p5.modeling_ptq import _QuantMoELinear

        if isinstance(module, _QuantMoELinear):
            return True
    except ImportError:
        pass
    return any(cls.__name__ == "QuantMoELinear" for cls in type(module).__mro__)


@ExportModuleRegistry.register(predicate=_is_quant_moe_linear)
def _export_moe_linear(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Fill missing input amax before child expert QuantLinears are exported."""
    set_expert_quantizer_amax(list(module.experts), quantizer_attrs="input_quantizer")


@ExportModuleRegistry.register(predicate=_has_fused_experts_quantizers)
def _export_fused_experts_module(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Split and quantize a fused-experts module with plural weight quantizers.

    Under FSDP2 each rank holds only some experts, so it packs just those (kept fused) and the
    split into per-expert keys is deferred until the gather brings all experts together.

    Tied experts are packed independently and their duplicate keys are dropped by name
    in postprocess_state_dict; no per-module dedup cache is used.

    Distributed no-gather export (``ctx.keep_fused_experts``) folds the quantizers into the fused
    weight IN PLACE, quantizing only this rank's local experts and keeping the result ``Shard(0)`` --
    the per-expert split happens later on the sharded weight (``split_fused_experts_state_dict``).
    Otherwise (single process, or gather export) it splits into full per-expert submodules here.
    """
    attr = name.rpartition(".")[2]
    if ctx.keep_fused_experts and attr != "experts":
        # The split into per-expert keys finds a fused weight by its ``experts.<proj>`` key, as the
        # writer does, so one named otherwise would be written fused and fail only at load.
        raise NotImplementedError(
            f"{name}: the keep-fused export splits fused experts into per-expert keys by their "
            f"`experts` attribute name, and this module is named {attr!r}."
        )
    if ctx.keep_fused_experts and distribution_plan(ctx.model).is_fsdp_dense:
        # FSDP2 owns the fused weights here: fold them inside a weight-update window, which hands
        # FSDP2 the packed weights. Replaced behind its back, they would be overwritten with its
        # original shards the next time the layer reshards, and exported unquantized.
        _hold_fsdp2_weight_window(ctx, module)
    if ctx.keep_fused_experts:
        # No-gather distributed export: fold each rank's LOCAL experts and leave the weight
        # fused + sharded; the per-expert split happens after consolidation.
        _export_fused_experts_keep_fused(module, ctx.dtype)
    else:
        _export_fused_experts(module, ctx.dtype)


@ExportModuleRegistry.register(predicate=is_quantlinear)
def _export_quant_linear(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export a standard quantized linear layer.

    The caller has already made the weight readable, so this packs it the same way for every
    parallelism setup.
    """
    if get_quantization_format(module) == QUANTIZATION_NONE:
        return
    try:
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
        _export_weight(module, ctx)
    except AssertionError as e:
        raise AssertionError(
            f"Failed to export embedding '{name}' (type={type(module).__name__}): {e}"
        ) from e


@ExportModuleRegistry.register("Llama4TextExperts", "GptOssExperts")
def _export_bmm_experts(name: str, module: nn.Module, ctx: ExportContext) -> None:
    """Export fused BMM-style expert weights (Llama4 / GPT-OSS).

    Its weight quantizer has one amax covering all experts, so under FSDP2 each rank can pack
    just the experts it owns and produce identical bytes -- no gather or unshard needed.
    """
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
    for weight_name in ["gate_up_proj", "down_proj"]:
        _export_weight(module, ctx, weight_name)
