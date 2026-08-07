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

"""Model-level preparation shared by every unified HF exporter.

Everything here runs on the whole model before any weight is packed: dtype resolution,
MoE input-quantizer preparation, resmoothing and shared-input fusion, quant-config
adjustments, and the transformers patches needed while writing artifacts.

It imports only leaf helpers (layer_utils, model_config, model_utils, quant_utils,
registry, diffusers_utils) and never an exporter, which is what lets all three exporters
import it without a cycle.
"""

import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.quantization import set_quantizer_by_cfg_context
from modelopt.torch.quantization.nn import SequentialQuantizer
from modelopt.torch.quantization.utils import fsdp2_aware_weight_update
from modelopt.torch.utils.dataset_utils import _disable_use_cache

from .diffusers_utils import get_qkv_group_key, is_qkv_projection
from .layer_utils import (
    get_experts_list,
    is_layernorm,
    is_moe,
    is_quantlinear,
    sync_moe_gate_up_amax,
)
from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4_SVDQUANT,
)
from .model_utils import get_language_model_from_vl, is_multimodal_model
from .quant_utils import (
    fuse_prequant_layernorm,
    fuse_prequant_to_linear,
    get_quantization_format,
    preprocess_linear_fusion,
)
from .registry import ExportContext, PrepareMoEInputsRegistry


def _is_enabled_quantizer(quantizer):
    if hasattr(quantizer, "is_enabled") and quantizer.is_enabled:
        return True

    if isinstance(quantizer, SequentialQuantizer):
        return any(q.is_enabled for q in quantizer)

    return False


def collect_shared_input_modules(
    model: nn.Module,
    dummy_forward_fn: Callable[[], None],
    collect_layernorms: bool = False,
) -> tuple[dict, dict | None]:
    """Collect modules that share the same input using forward hooks.

    This is a common helper for both LLM and diffusion model fusion.

    Args:
        model: The model to analyze.
        dummy_forward_fn: A callable that runs a dummy forward pass on the model.
            Should be a function that takes no arguments.
        collect_layernorms: If True, also collect layernorm output mappings (for AWQ).

    Returns:
        A tuple of (input_to_linear, output_to_layernorm).
        input_to_linear: Dict mapping input tensor to list of modules sharing that input.
        output_to_layernorm: Dict mapping layernorm output to the layernorm module (or None).
    """
    input_to_linear: dict = defaultdict(list)
    output_to_layernorm: dict | None = defaultdict(lambda: None) if collect_layernorms else None

    def _input_hook(module, input, output):
        """Update dictionary with list of all modules that share the same input."""
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            # TODO: Handle DBRX MoE case
            input_to_linear[input[0]].append(module)

    def _output_hook(module, input, output):
        """Update dictionary with mapping of layernorms and their outputs."""
        if output_to_layernorm is not None and isinstance(output, torch.Tensor):
            output_to_layernorm[output] = module

    handles = []

    # Register hooks on all quantized linear modules (and optionally layernorms)
    for name, module in model.named_modules():
        if collect_layernorms and is_layernorm(module):
            module.name = name
            handle = module.register_forward_hook(_output_hook)
            handles.append(handle)
        elif is_quantlinear(module) and (
            _is_enabled_quantizer(module.input_quantizer)
            or _is_enabled_quantizer(module.weight_quantizer)
        ):
            module.name = name
            handle = module.register_forward_hook(_input_hook)
            handles.append(handle)

    if not handles:
        return input_to_linear, output_to_layernorm

    # Run dummy forward pass to collect modules sharing same input.
    # `_disable_use_cache` keeps the probe forward working on configs that don't
    # set `use_cache` (e.g., stepfun-ai/Step-3.5-Flash's Step3p5Config).
    try:
        with (
            torch.no_grad(),
            set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]),
            _disable_use_cache(model),
        ):
            dummy_forward_fn()
    finally:
        # Always remove hooks
        for handle in handles:
            handle.remove()

    return input_to_linear, output_to_layernorm


def _fuse_shared_input_modules(
    model: nn.Module,
    input_to_linear: dict,
    output_to_layernorm: dict | None = None,
    qkv_only: bool = False,
    fuse_layernorms: bool = False,
    quantization_format: str | None = None,
) -> dict[str, list[str]]:
    """Fuse modules that share the same input.

    This is a common helper for both LLM and diffusion model fusion.

    Args:
        model: The model being processed (for FSDP-aware updates).
        input_to_linear: Dict mapping input tensor to list of modules sharing that input.
        output_to_layernorm: Dict mapping layernorm output to the layernorm module (optional).
        qkv_only: If True, only fuse QKV projection layers (for diffusion models).
        fuse_layernorms: If True, also fuse layernorms with pre_quant_scale (for AWQ).
        quantization_format: The quantization format of the model.

    Returns:
        Dict mapping first module name to list of all fused module names.
    """
    fused_linears = {}
    fused_count = 0

    for tensor, modules in input_to_linear.items():
        # Get quantization format for this group of modules
        # (must be re-evaluated per group as different modules may have different formats)
        group_quant_format = get_quantization_format(modules[0]) if modules else quantization_format

        if len(modules) > 1 and group_quant_format not in [
            QUANTIZATION_FP8,
            QUANTIZATION_NONE,
            QUANTIZATION_FP8_PB_REAL,
        ]:
            if qkv_only:
                # Filter to only include QKV projection layers (diffusion models)
                qkv_modules = [m for m in modules if is_qkv_projection(getattr(m, "name", ""))]

                if len(qkv_modules) > 1:
                    # Group QKV modules by their parent attention block
                    qkv_groups: dict[str, list[nn.Module]] = defaultdict(list)
                    for m in qkv_modules:
                        group_key = get_qkv_group_key(getattr(m, "name", ""))
                        qkv_groups[group_key].append(m)

                    # Fuse each group separately
                    for group_key, group_modules in qkv_groups.items():
                        if len(group_modules) >= 2:
                            preprocess_linear_fusion(group_modules, resmooth_only=False)
                            fused_count += 1
                            module_names = [getattr(m, "name", "unknown") for m in group_modules]
                            print(f"  Fused QKV group: {module_names}")
            else:
                # Fuse all modules that have the same input (LLM models)
                with fsdp2_aware_weight_update(model, modules):
                    preprocess_linear_fusion(modules)
                fused_linears[modules[0].name] = [module.name for module in modules]
                fused_count += 1

            # Fuse layernorms (for AWQ)
            if (
                fuse_layernorms
                and output_to_layernorm is not None
                and group_quant_format is not None
                and group_quant_format != QUANTIZATION_NONE
                and "awq" in group_quant_format
                and tensor in output_to_layernorm
            ):
                with fsdp2_aware_weight_update(model, output_to_layernorm[tensor]):
                    fuse_prequant_layernorm(output_to_layernorm[tensor], modules)

    if qkv_only:
        if fused_count > 0:
            print(f"Fused {fused_count} QKV group(s) for unified amax values.")
        else:
            print("No QKV groups found to fuse.")

    return fused_linears


def requantize_resmooth_fused_llm_layers(model: torch.nn.Module):
    """Group modules that take the same input and register shared parameters in module."""
    # TODO: Handle DBRX MoE
    quantization_format = get_quantization_format(model)
    model_type = type(model).__name__.lower()
    module_names = set()

    # NVFP4 SVDQuant does not need pre-quant scale fusion (either into previous linear or layernorm) because
    # 1) its kernel handles pre-quant scale.
    # 2) fusing into previous linear will need to change the lora_up in up_proj which may cause issue in
    #    the later gate up fusion.
    # Fuse pre_quant_scale to the linear weights if possible
    if quantization_format is not None and "nvfp4_awq" in quantization_format.lower():
        fuse_prequant_to_linear(model)

    # Pre-process MoE experts
    for name, module in model.named_modules():
        module_names.add(name)

        # For MoE models update pre_quant_scale to average pre_quant_scale amongst experts
        if is_moe(module) and (
            quantization_format is not QUANTIZATION_NONE
            and ("awq" in quantization_format or quantization_format == QUANTIZATION_NVFP4_SVDQUANT)
        ):
            # update_experts_avg_prequant_scale(module)
            grouped_experts = get_experts_list(module, model_type)
            for modules in grouped_experts:
                with fsdp2_aware_weight_update(model, modules):
                    preprocess_linear_fusion(modules, resmooth_only=True)

    # Define the dummy forward function for LLM
    def llm_dummy_forward():
        fake_input = torch.ones([1, 2], dtype=torch.long).to(model.device)
        decoder_fake_input = fake_input

        # Check if this is a VL model that needs special input handling
        is_vl_model = is_multimodal_model(model)

        if model_type.startswith("whisper"):
            # For Whisper models, we need to pass a fake input with the specific sequence length
            from transformers import AutoFeatureExtractor

            feature_extractor = AutoFeatureExtractor.from_pretrained(model.name_or_path)
            fake_input = torch.ones(
                [1, model.config.num_mel_bins, feature_extractor.nb_max_frames], dtype=model.dtype
            ).to(model.device)

        if is_vl_model and "nemotron" in model_type:
            # For Nemotron VL models, run optimization on just the language model/decoder.
            # This avoids needing pixel_values for the vision encoder.
            language_model_lineage = get_language_model_from_vl(model)

            if language_model_lineage is not None:
                language_model = language_model_lineage[-1]
                print(
                    f"Running optimization on language model with fake_input shape: {fake_input.shape}"
                )
                # Pass use_cache=False to avoid KV cache issues in encoder-decoder models
                language_model(fake_input, use_cache=False)
            else:
                raise ValueError(
                    f"Cannot extract language_model from Nemotron VL model (type: {model_type}). "
                    "This is required for requantization/resmoothing optimization. "
                    "Please ensure the model architecture is supported or file an issue."
                )
        elif getattr(model.config, "is_encoder_decoder", False):
            # For other encoder-decoder models (non-VL), pass both encoder and decoder input ids
            model(fake_input, decoder_input_ids=decoder_fake_input)
        elif hasattr(model, "get_dummy_inputs"):
            # For speculative decoding models (EAGLE, etc.), use model-provided dummy inputs
            model(**model.get_dummy_inputs())
        else:
            model(fake_input)

    input_to_linear, output_to_layernorm = collect_shared_input_modules(
        model, llm_dummy_forward, collect_layernorms=True
    )

    fused_linears = _fuse_shared_input_modules(
        model,
        input_to_linear,
        output_to_layernorm,
        qkv_only=False,
        fuse_layernorms=True,
        quantization_format=quantization_format,
    )

    # The dummy forward may not be able to activate all the experts.
    # Process experts by naming rules like experts.0, experts.1, etc.
    for name, modules_fused in fused_linears.items():
        if re.search(r"experts?\.\d+", name):
            expert_id = 0
            while True:
                new_expert_name = re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", name, count=1)
                if new_expert_name in fused_linears:
                    expert_id += 1
                    continue
                if new_expert_name not in module_names:
                    break

                new_expert_modules = []
                for name_fused in modules_fused:
                    new_expert_name = re.sub(r"(experts?\.)\d+", rf"\g<1>{expert_id}", name_fused)
                    assert new_expert_name in module_names
                    new_expert_modules.append(model.get_submodule(new_expert_name))

                with fsdp2_aware_weight_update(model, new_expert_modules):
                    preprocess_linear_fusion(new_expert_modules)

                expert_id += 1


def _resolve_export_dtype(model: nn.Module, dtype: torch.dtype | None) -> torch.dtype:
    """Return the export dtype, defaulting to the model's own and warning on a mismatch."""
    if dtype is None:
        return model.config.torch_dtype
    if dtype != model.config.torch_dtype:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )
    return dtype


def _prepare_moe_inputs(model: nn.Module, dtype: torch.dtype, is_modelopt_qlora: bool) -> None:
    """Handle input quantizers of experts that are not calibrated.

    Each MoE block is dispatched by its experts container to the matching preparation
    handler.
    """
    prepare_ctx = ExportContext(model=model, dtype=dtype, is_modelopt_qlora=is_modelopt_qlora)
    for name, sub_module in model.named_modules():
        if is_moe(sub_module) and hasattr(sub_module, "experts"):
            handler = PrepareMoEInputsRegistry.match(sub_module.experts)
            if handler is None:
                # Unsupported MoE model structure
                raise NotImplementedError(
                    f"MoE model with experts type '{type(sub_module.experts).__name__}' is not supported in export."
                    f"Please file an issue or add support for this model architecture."
                )
            handler(name, sub_module, prepare_ctx)


def _add_mtp_exclusions(model: nn.Module, quant_config: dict) -> None:
    """Add MTP layer prefixes to exclude_modules if they were excluded from quantization.

    This ensures they appear in ``quantization_config["ignore"]`` in ``config.json``.
    """
    mtp_layer_prefixes = getattr(model, "_mtp_layer_prefixes", None)
    if mtp_layer_prefixes:
        exclude_modules = quant_config["quantization"].setdefault("exclude_modules", [])
        for prefix in mtp_layer_prefixes:
            # Add wildcard pattern to exclude all submodules under this MTP layer
            pattern = f"{prefix}*"
            if pattern not in exclude_modules:
                exclude_modules.append(pattern)
                print(f"Adding MTP layer to quantization_config ignore: {pattern}")


def _warn_on_unsynced_moe_gate_up(model: nn.Module) -> None:
    """Safety net for gate/up weight quantizer amaxes that resmoothing did not reach.

    ``requantize_resmooth_fused_llm_layers`` can miss experts that the dummy forward
    never activated, or that use non-standard expert naming.
    """
    synced = sync_moe_gate_up_amax(model)
    if synced:
        warnings.warn(
            f"Found {synced} MoE expert gate/up projection pair(s) with mismatched "
            f"weight_scale_2 after requantize_resmooth_fused_llm_layers. "
            f"This typically means the dummy forward did not activate these experts. "
            f"Taking element-wise max of amaxes for serving-engine fusion."
        )


# TODO: Remove this workaround once HuggingFace fixes revert_weight_conversion to handle
# scalar (0-d) tensors. transformers' Chunk.convert() calls torch.chunk() on quantization
# scale buffers that are 0-d scalars, raising RuntimeError ("chunk expects at least a
# 1-dimensional tensor"). Confirmed in transformers 5.12.0.
# See: transformers/core_model_loading.py, Chunk.convert()
def _revert_weight_conversion_noop(model: Any, state_dict: dict) -> dict:
    """No-op replacement for transformers' revert_weight_conversion."""
    return state_dict


def _try_patch_module(mod_path: str) -> tuple[Any, Any] | None:
    """Try to patch revert_weight_conversion in a single module."""
    import importlib

    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, "revert_weight_conversion"):
            original = getattr(mod, "revert_weight_conversion")
            setattr(mod, "revert_weight_conversion", _revert_weight_conversion_noop)
            return (mod, original)
    except (ImportError, AttributeError):
        pass
    return None


def _patch_revert_weight_conversion() -> list[tuple[Any, Any]]:
    """Patch revert_weight_conversion in transformers to avoid RuntimeError on scalar tensors."""
    patches: list[tuple[Any, Any]] = []
    for mod_path in [
        "transformers.core_model_loading",
        "transformers.modeling_utils",
    ]:
        result = _try_patch_module(mod_path)
        if result is not None:
            patches.append(result)
    return patches


def _unpatch_revert_weight_conversion(patches: list[tuple[Any, Any]]) -> None:
    """Restore the original revert_weight_conversion functions."""
    for mod, original in patches:
        mod.revert_weight_conversion = original


def _sanitize_generation_config_for_save(model: torch.nn.Module) -> None:
    """Force ``do_sample=True`` when generation_config has ``top_k``/``top_p`` set.

    Newer transformers reject ``do_sample=False`` mixed with sampling attrs in
    ``save_pretrained``'s strict validate.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    if getattr(gc, "top_k", None) is not None or getattr(gc, "top_p", None) is not None:
        gc.do_sample = True
