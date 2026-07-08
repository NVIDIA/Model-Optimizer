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

"""Code that export quantized Hugging Face models for deployment."""

import collections.abc
import json
import re
import tempfile
import warnings
from builtins import ValueError
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .diffusers_utils import build_layerwise_quant_metadata, pad_nvfp4_weights, swizzle_nvfp4_scales

try:
    import diffusers

    from .diffusers_utils import (
        generate_diffusion_dummy_forward_fn,
        get_diffusion_components,
        get_diffusion_model_type,
        get_qkv_group_key,
        hide_quantizers_from_state_dict,
        infer_dtype_from_model,
        is_diffusers_object,
        is_qkv_projection,
        merge_diffusion_checkpoint,
    )

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

from torch.distributed.fsdp import FSDPModule

from modelopt.torch.quantization import set_quantizer_by_cfg_context
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.qtensor import MXFP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.utils import fsdp2_aware_weight_update, quantizer_attr_names
from modelopt.torch.utils.dataset_utils import _disable_use_cache

try:
    from modelopt.torch.sparsity.attention_sparsity.conversion import export_sparse_attention_config
except ImportError:
    export_sparse_attention_config = None

from .convert_hf_config import convert_hf_quant_config_format
from .layer_utils import (
    get_expert_linear_names,
    get_experts_list,
    is_layernorm,
    is_moe,
    is_quantlinear,
    set_expert_quantizer_amax,
    sync_moe_gate_up_amax,
)
from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PC_PT,
    QUANTIZATION_MXFP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_NVFP4_SVDQUANT,
    QUANTIZATION_W4A8_AWQ,
    QUANTIZATION_W4A8_NVFP4_FP8,
    QUANTIZATION_W4A16_NVFP4,
)
from .model_utils import _reorder_canonical_first, get_language_model_from_vl, is_multimodal_model
from .moe_utils import _export_fused_experts
from .plugins import SpeculativeDecodingExporter, has_spec_opt, sanitize_hf_config_for_deployment
from .quant_aware_conversion import (
    build_reverse_name_mapper,
    revert_quant_config_names,
    revert_weight_conversion_quant_aware,
)
from .quant_utils import (
    fuse_prequant_layernorm,
    fuse_prequant_to_linear,
    get_activation_scaling_factor,
    get_quant_config,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    has_quantized_modules,
    maybe_transpose_expert_weight_dimensions,
    postprocess_state_dict,
    preprocess_linear_fusion,
    sync_tied_input_amax,
    to_quantized_weight,
)

__all__ = ["export_hf_checkpoint", "export_speculative_decoding"]


def _is_enabled_quantizer(quantizer):
    if hasattr(quantizer, "is_enabled") and quantizer.is_enabled:
        return True

    if isinstance(quantizer, SequentialQuantizer):
        return any(q.is_enabled for q in quantizer)

    return False


def _save_component_state_dict_safetensors(
    component: nn.Module,
    component_export_dir: Path,
) -> None:
    """Save component state dict as a plain safetensors file.

    Args:
        component: The nn.Module to save.
        component_export_dir: Directory to save model.safetensors and config.json.
    """
    cpu_state_dict = {k: v.detach().contiguous().cpu() for k, v in component.state_dict().items()}
    metadata = {
        "_export_format": "safetensors_state_dict",
        "_class_name": type(component).__name__,
    }

    save_file(
        cpu_state_dict,
        str(component_export_dir / "model.safetensors"),
        metadata=metadata,
    )

    with open(component_export_dir / "config.json", "w") as f:
        json.dump(metadata, f, indent=4)


def _postprocess_safetensors(
    export_dir: Path,
    pipe: Any | None = None,
    hf_quant_config: dict | None = None,
    **kwargs,
) -> None:
    """Post-process saved safetensors files for deployment compatibility.

    Loads each ``.safetensors`` file in *export_dir* and applies all requested
    transformations in order, then re-saves in-place with updated metadata:

    1. **Merge** with base checkpoint — combines quantized transformer weights with
       non-transformer components (VAE, vocoder, text encoders) from a base
       ``.safetensors`` file to produce a single-file checkpoint (e.g., for ComfyUI).
    2. **Pad** NVFP4 weight/scale tensors — ensures dimensions are multiples of 16
       for hardware alignment requirements.
    3. **Swizzle** NVFP4 block scales — rearranges from flat layout to cuBLAS 2-D
       block-scaling-factors tiled layout for optimized inference.
    4. **Inject metadata** — embeds ``quantization_config`` and per-layer
       ``_quantization_metadata`` so inference runtimes can detect and handle
       quantized layers.

    All of these target single-file deployment runtimes (e.g. ComfyUI) and are
    opt-in; ModelOpt itself reads the quant config from ``config.json`` on reload. If
    the caller passes none of ``merged_base_safetensor_path``, ``padding_strategy``,
    ``enable_swizzle_layout``, or ``enable_layerwise_quant_metadata``, this function
    does nothing and leaves the standard exported checkpoint untouched.

    Args:
        export_dir: Directory containing the saved ``.safetensors`` file(s).
        pipe: The diffusion pipeline / model.  Used to infer the model type
            (via :func:`get_diffusion_model_type`) when
            ``merged_base_safetensor_path`` is set.
        hf_quant_config: Quantization config dict to embed in metadata.
        **kwargs: Runtime-specific keyword arguments:
            merged_base_safetensor_path (str, optional): When provided, merges
                the exported transformer weights with non-transformer components
                (VAE, vocoder, text encoders, etc.) from this base safetensors
                file to produce a single-file checkpoint compatible with ComfyUI.
                Value should be the path to a full base model ``.safetensors``
                file (e.g. ``"path/to/ltx-2-19b-dev.safetensors"``).
            enable_layerwise_quant_metadata (bool, optional): When True, embeds
                ``quantization_config`` and per-layer ``_quantization_metadata`` in the
                safetensors header so single-file runtimes (e.g., ComfyUI) can identify
                which layers are quantized and in what format. Defaults to False (no
                header metadata; this alone leaves the export untouched).
            enable_swizzle_layout (bool, optional): When True, rearranges NVFP4
                block scales from ModelOpt's flat layout to cuBLAS 2-D tiled
                layout. Required for runtimes that consume cuBLAS block-scaled
                GEMM (e.g., comfy_kitchen). Defaults to False.
            padding_strategy (str | None, optional): Padding strategy for NVFP4
                weight and scale tensors. ``"row"`` pads rows to multiples of
                16 (columns assumed already aligned). ``"row_col"`` pads both
                dimensions. ``None`` (default) disables padding. Independent of
                ``enable_swizzle_layout``.

    """
    merged_base_safetensor_path: str | None = kwargs.get("merged_base_safetensor_path")
    enable_layerwise_quant_metadata: bool = kwargs.get("enable_layerwise_quant_metadata", False)
    enable_swizzle_layout: bool = kwargs.get("enable_swizzle_layout", False)
    padding_strategy: str | None = kwargs.get("padding_strategy")

    # This post-processing only produces single-file deployment checkpoints (e.g.
    # ComfyUI): merging with a base checkpoint, NVFP4 padding/swizzling, and embedding
    # quant metadata in the safetensors header. None of it is read back by ModelOpt
    # (the diffusers reload uses ``config.json``), so if the user has not opted into any
    # of these options there is nothing to do — leave the exported checkpoint untouched.
    if not (
        merged_base_safetensor_path is not None
        or padding_strategy is not None
        or enable_swizzle_layout
        or enable_layerwise_quant_metadata
    ):
        return

    safetensor_files = sorted(export_dir.glob("*.safetensors"))
    if not safetensor_files:
        return

    if list(export_dir.glob("*.safetensors.index.json")) and (
        merged_base_safetensor_path is not None or enable_layerwise_quant_metadata
    ):
        raise NotImplementedError(
            "Post-processing sharded safetensors is not supported. "
            "Export with a larger max_shard_size or disable merge/metadata options."
        )

    model_type: str | None = None
    if merged_base_safetensor_path is not None:
        if pipe is None:
            raise ValueError("`pipe` must be provided when `merged_base_safetensor_path` is set.")
        model_type = get_diffusion_model_type(pipe)

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt") as f:
            metadata = dict(f.metadata() or {})
            sd = {k: f.get_tensor(k).clone() for k in f.keys()}  # noqa: SIM118

        if merged_base_safetensor_path is not None and model_type is not None:
            sd, base_metadata = merge_diffusion_checkpoint(
                sd, merged_base_safetensor_path, model_type, hf_quant_config=None
            )
            base_metadata.update(metadata)
            metadata = base_metadata

        if padding_strategy is not None:
            sd = pad_nvfp4_weights(sd, padding_strategy)

        if enable_swizzle_layout:
            sd = swizzle_nvfp4_scales(sd)

        if hf_quant_config is not None:
            metadata["quantization_config"] = json.dumps(hf_quant_config)
            if enable_layerwise_quant_metadata:
                metadata["_quantization_metadata"] = build_layerwise_quant_metadata(
                    sd, hf_quant_config
                )

        save_file(sd, str(sf_path), metadata=metadata)


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


def _export_quantized_weight(
    sub_module: nn.Module,
    dtype: torch.dtype,
    weight_name: str = "weight",
    _tied_cache: dict[int, nn.Module] | None = None,
):
    """For the given weight attr of the sub_module, export the quantization info of it.

    The export includes converting weight tensor to correct quantized values and quantized dtype,
    and registering scaling factors.

    Tied-weight dedup is opt-in via ``_tied_cache``: the setattr below replaces
    ``.weight`` with a fresh ``nn.Parameter`` wrapping packed bytes, breaking
    any HF-level tie. When the caller passes a ``_tied_cache`` dict (keyed by
    the pre-pack ``weight.data_ptr()``), the alias step at the end re-points
    ``weight`` / ``weight_scale`` / ``weight_scale_2`` at a previously-processed
    module sharing the same source memory so the downstream data_ptr dedup can
    collapse them. The cache is owned by the caller (typically
    ``_export_transformers_checkpoint``) and scoped to one export invocation;
    when ``_tied_cache`` is ``None`` (the default) the alias step is skipped
    entirely. Uses memory identity only — no ``_tied_weights_keys`` lookup,
    no-op for non-tied modules.
    """
    quantization_format = get_quantization_format(sub_module)
    if quantization_format == QUANTIZATION_NONE:
        return

    block_size = get_weight_block_size(sub_module, weight_name)
    quantizer_attrs = quantizer_attr_names(weight_name)
    weight: nn.Parameter = getattr(sub_module, weight_name)

    # Capture source identity BEFORE any tensor-creating operation below.
    # For HF-tied weights this matches across all modules sharing the
    # underlying Parameter; the cache lookup at the end of this function
    # uses it to detect ties whose Python identity is about to be broken
    # by the setattr on `weight_name` further down.
    _tied_source_data_ptr = weight.data_ptr()
    weight_quantizer: TensorQuantizer | SequentialQuantizer = getattr(
        sub_module, quantizer_attrs.weight_quantizer
    )
    input_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.input_quantizer, None
    )
    output_quantizer: TensorQuantizer | SequentialQuantizer | None = getattr(
        sub_module, quantizer_attrs.output_quantizer, None
    )

    if quantization_format == QUANTIZATION_FP8:
        # Convert amax to float32
        weight_quantizer._amax = weight_quantizer._amax.to(torch.float32)

        if weight_quantizer._amax.dim() == 1:
            # Per-tensor amax
            weight_scaling_factor = torch.tensor(
                weight_quantizer.amax.item() / weight_quantizer.maxbound
            )
        else:
            # Per-channel amax
            weight_scaling_factor = torch.tensor(weight_quantizer.amax / weight_quantizer.maxbound)

        sub_module.register_buffer(
            quantizer_attrs.weight_scale,
            weight_scaling_factor,
        )

        if hasattr(input_quantizer, "_amax"):
            assert input_quantizer is not None
            input_quantizer._amax = input_quantizer._amax.to(torch.float32)

            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

        if hasattr(output_quantizer, "_amax"):
            assert output_quantizer is not None
            output_quantizer._amax = output_quantizer._amax.to(torch.float32)
    else:
        # Register weight_scale and input_scale
        if quantization_format == QUANTIZATION_FP8_PB_REAL:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale,
                weight_quantizer._scale.to(torch.float32),
            )
            del weight_quantizer._scale
        elif quantization_format == QUANTIZATION_MXFP8:
            # MXFP8 uses dynamic block quantization with E8M0 scales (uint8)
            weight = getattr(sub_module, weight_name)
            e8m0_scale = MXFP8QTensor.get_weights_scaling_factor_from_quantizer(
                weight, weight_quantizer
            )
            sub_module.register_buffer(quantizer_attrs.weight_scale, e8m0_scale)
            if hasattr(weight_quantizer, "_scale") and weight_quantizer._scale is not None:
                del weight_quantizer._scale
        else:
            sub_module.register_buffer(
                quantizer_attrs.weight_scale, get_weight_scaling_factor(sub_module, weight_name)
            )

        if (
            input_quantizer is not None
            and "disabled" not in repr(input_quantizer)
            and input_quantizer.amax is not None
        ):
            sub_module.register_buffer(
                quantizer_attrs.input_scale,
                get_activation_scaling_factor(
                    sub_module, input_quantizer_name=quantizer_attrs.input_quantizer
                ).squeeze(),
            )

    if quantization_format in [
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_NVFP4,
        QUANTIZATION_W4A16_NVFP4,
        QUANTIZATION_W4A8_AWQ,
        QUANTIZATION_W4A8_NVFP4_FP8,
    ]:
        # Register weight_scale_2
        sub_module.register_buffer(
            quantizer_attrs.weight_scale_2,
            get_weight_scaling_factor_2(sub_module, weight_name).squeeze(),
        )

    weight_scale: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale, None)
    weight_scale_2: torch.Tensor | None = getattr(sub_module, quantizer_attrs.weight_scale_2, None)

    # Transpose weight for bmm-style expert quantization (llama4, gpt-oss)
    # Check if this is a BMM-style expert weight that needs transposition
    is_bmm_expert_weight = weight.dim() == 3 and any(
        expert_type in type(sub_module).__name__
        for expert_type in ["Llama4TextExperts", "GptOssExperts"]
    )
    # NVFP4StaticQuantizer + BMM-style experts: route through the static-aware
    # ``_from_quantizer`` helper so the pinned per-block ``_amax`` (e.g. set by
    # the MXFP4->NVFP4 cast to ``6 * 2^k_j``) is used to derive the FP8
    # per-block scale. The plain ``get_weights_scaling_factor`` would ignore
    # ``_amax`` and recompute per-block max from the BF16 weight, which
    # rebuckets nibbles and loses bit-exactness when ``max_nibble < 6``.

    if quantization_format in [
        QUANTIZATION_NVFP4,
        QUANTIZATION_NVFP4_AWQ,
        QUANTIZATION_NVFP4_SVDQUANT,
        QUANTIZATION_W4A16_NVFP4,
    ]:
        # Transpose weight from (num_experts, input_dim, output_dim) to (num_experts, output_dim, input_dim)
        # for NVFP4 quantization functions that expect input_dim as the last dimension for block quantization
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )

        if NVFP4QTensor._is_static_quantizer(weight_quantizer):
            weight_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                weight_quantizer,
                weight,
                weight_scale_2,
            )[0]
        else:
            weight_scale = NVFP4QTensor.get_weights_scaling_factor(
                weight,
                block_size=block_size,
                weights_scaling_factor_2=weight_scale_2,
            )[0]

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        quantized_weight, weight_scale = maybe_transpose_expert_weight_dimensions(
            quantized_weight, weight_scale, is_bmm_expert_weight=is_bmm_expert_weight
        )
    elif quantization_format == QUANTIZATION_FP8_PC_PT and is_bmm_expert_weight:
        # For FP8_PC_PT with BMM-style experts, transpose only the weight (not weight_scale)
        weight, _ = maybe_transpose_expert_weight_dimensions(
            weight, is_bmm_expert_weight=is_bmm_expert_weight
        )

        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

        # Transpose back to original BMM format
        quantized_weight, _ = maybe_transpose_expert_weight_dimensions(
            quantized_weight, is_bmm_expert_weight=is_bmm_expert_weight
        )
    else:
        quantized_weight = to_quantized_weight(
            weight.to(dtype),
            weight_scale,
            quantization_format,
            weight_scale_2,
            block_size,
        )

    setattr(sub_module, weight_name, nn.Parameter(quantized_weight, requires_grad=False))

    # Register the corrected weight_scale as a buffer
    if weight_scale is not None:
        sub_module.register_buffer(quantizer_attrs.weight_scale, weight_scale)

    # Tied-weight dedup: if a previously-processed module shared the same
    # source weight memory, alias the packed weight + scale buffers so the
    # downstream data_ptr dedup in postprocess_state_dict can collapse them.
    # input_scale is safe to alias because sync_tied_input_amax (earlier in
    # this export) already max-merged the per-side amaxes. Gated on the
    # caller-owned _tied_cache so the dedup state is scoped to one export.
    if _tied_cache is not None:
        _prior = _tied_cache.get(_tied_source_data_ptr)
        if _prior is not None and _prior is not sub_module:
            if hasattr(_prior, weight_name):
                setattr(sub_module, weight_name, getattr(_prior, weight_name))
            for _attr in (
                quantizer_attrs.weight_scale,
                quantizer_attrs.weight_scale_2,
                quantizer_attrs.input_scale,
            ):
                if not hasattr(_prior, _attr):
                    continue
                if _attr in sub_module._buffers:
                    del sub_module._buffers[_attr]
                elif hasattr(sub_module, _attr):
                    delattr(sub_module, _attr)
                sub_module.register_buffer(_attr, getattr(_prior, _attr))
        else:
            _tied_cache[_tied_source_data_ptr] = sub_module

    torch.cuda.empty_cache()


def _process_quantized_modules(
    model: nn.Module,
    dtype: torch.dtype,
    is_modelopt_qlora: bool = False,
) -> None:
    """Process all quantized modules in model, export weights in-place.

    This function iterates through all modules in the model and exports quantized weights
    for modules that have quantization enabled. It handles both standard linear layers
    and specialized expert modules (Llama4TextExperts, GptOssExperts).

    Args:
        model: The model containing quantized modules.
        dtype: The data type for weight conversion.
        is_modelopt_qlora: Whether the model is a modelopt-trained QLoRA model.
            If True, modules with base_layer attribute are skipped.
    """
    # Per-call tied-weight dedup caches. Created fresh on every invocation
    # so cache state is scoped to one export and cannot leak into a later
    # call (a process-global cache would carry stale entries whose data_ptr
    # keys can be recycled by PyTorch's allocator across exports — silent
    # false-positive aliasing). int keys hold dense Linear / per-expert
    # wrapper dedup; tuple keys hold MoE fused-experts module dedup.
    _tied_cache: dict[int, nn.Module] = {}
    _moe_tied_cache: dict[tuple[int, int], nn.Module] = {}
    fsdp_module_to_reshard = None

    for name, sub_module in model.named_modules():
        # Optimization to perform resharding only once per decoder layer to avoid extra communication overhead
        if isinstance(sub_module, FSDPModule):
            # Every time we encounter a new FSDPModule, the previous decoder layer is fully processed.
            # We need to reshard the previous FSDPModule to prevent potential OOM.
            # This hack reduces the number of unshard reshard operations, to avoid unnecessary communication.
            if fsdp_module_to_reshard is not None:
                fsdp_module_to_reshard.reshard()

            fsdp_module_to_reshard = sub_module

        # We skip QuantLoraLinear module for modelopt QLoRA
        if is_modelopt_qlora and (hasattr(sub_module, "base_layer")):
            continue

        # Step-3.5 QuantMoELinear reconstructs packed MoE tensors from child
        # expert QuantLinears after export. Fill missing input amax here, before
        # named_modules() reaches those children, so every expert emits input_scale.
        if type(sub_module).__name__ == "QuantMoELinear" and hasattr(sub_module, "experts"):
            set_expert_quantizer_amax(list(sub_module.experts), quantizer_attrs="input_quantizer")
            continue

        # Preprocessing: restore unpacked weight so the export path can read
        # the live quantizer state. Falls through to the export branches below.
        if hasattr(sub_module, "weight_packed") or (
            "QuantFP8Linear" in type(sub_module).__name__ and sub_module.weight.element_size() <= 1
        ):
            sub_module.unpack_weight()

        first_proj_attr = getattr(sub_module, "_first_proj_attr", "gate_up_proj")
        if hasattr(sub_module, f"{first_proj_attr}_weight_quantizers"):
            # _QuantFusedExperts uses plural `<first_proj>_weight_quantizers`
            # (ModuleList), which get_quantization_format's singular-weight_quantizer
            # check misses. Handle it explicitly before the format gate so fused-experts
            # get split + quantized.
            with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                _export_fused_experts(
                    sub_module,
                    dtype,
                    _moe_tied_cache=_moe_tied_cache,
                    _tied_cache=_tied_cache,
                )
        elif get_quantization_format(sub_module) != QUANTIZATION_NONE:
            # Skip QuantMoELinear - it's handled separately in _reconstruct_fused_moe_linear
            if type(sub_module).__name__ == "QuantMoELinear":
                continue
            if is_quantlinear(sub_module):
                try:
                    with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                        _export_quantized_weight(sub_module, dtype, _tied_cache=_tied_cache)
                except AssertionError as e:
                    raise AssertionError(
                        f"Failed to export module '{name}' (type={type(sub_module).__name__}): {e}"
                    ) from e
            elif isinstance(sub_module, nn.Embedding) and hasattr(sub_module, "weight_quantizer"):
                # Quantized nn.Embedding: pack the embedding table the same way as Linear
                # weights so downstream loaders see the NVFP4/FP8/INT-packed bytes + scales.
                # Skip packing when the embedding's weight is tied to another module
                # (e.g. tied_word_embeddings → lm_head): _export_quantized_weight reassigns
                # the .weight attribute to a new uint8 Parameter, which severs the Python-
                # level tie and leaves the other module pointing at a stale float Parameter.
                tied_to = [
                    other_name
                    for other_name, other_module in model.named_modules()
                    if other_module is not sub_module
                    and getattr(other_module, "weight", None) is sub_module.weight
                ]
                if tied_to:
                    warnings.warn(
                        f"Skipping quantized weight packing for embedding '{name}': its "
                        f"weight Parameter is shared with {tied_to} (weight tying). Packing "
                        "would break the tie and produce stale weights in the tied module(s). "
                        "The embedding will be exported as its fake-quantized float weight."
                    )
                else:
                    try:
                        with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                            _export_quantized_weight(sub_module, dtype, _tied_cache=_tied_cache)
                    except AssertionError as e:
                        raise AssertionError(
                            f"Failed to export embedding '{name}' (type={type(sub_module).__name__}): {e}"
                        ) from e
            elif (
                "Llama4TextExperts" in type(sub_module).__name__
                or "GptOssExperts" in type(sub_module).__name__
            ):
                # TODO: consolidate uncalibrated experts handling logic
                # Handle weight quantizers amax values using smart fallback logic
                set_expert_quantizer_amax(
                    modules=sub_module,
                    quantizer_attrs=["gate_up_proj_weight_quantizer", "down_proj_weight_quantizer"],
                )
                # Handle input quantizers amax values using smart fallback logic
                set_expert_quantizer_amax(
                    modules=sub_module,
                    quantizer_attrs=["gate_up_proj_input_quantizer", "down_proj_input_quantizer"],
                )
                # Export the quantized weights
                with fsdp2_aware_weight_update(model, sub_module, reshard=False):
                    for weight_name in ["gate_up_proj", "down_proj"]:
                        _export_quantized_weight(
                            sub_module, dtype, weight_name, _tied_cache=_tied_cache
                        )


def _export_transformers_checkpoint(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    is_modelopt_qlora: bool = False,
    **kwargs,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Args:
        model: the full torch model to export. The actual quantized model may be a submodule.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        accelerator: the accelerator instance in case of distributed export setup.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
    """
    if dtype is None:
        dtype = model.config.torch_dtype
    elif dtype != model.config.torch_dtype:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )

    accelerator = kwargs.get("accelerator")

    # Handle input quantizers of experts that are not calibrated
    for _, sub_module in model.named_modules():
        if is_moe(sub_module) and hasattr(sub_module, "experts"):
            expert_linear_names = get_expert_linear_names(sub_module)
            first_proj_attr = getattr(sub_module.experts, "_first_proj_attr", "gate_up_proj")
            has_fused_experts_quantizers = hasattr(
                sub_module.experts, f"{first_proj_attr}_weight_quantizers"
            )
            for linear_name in expert_linear_names:
                # Handle DBRX experts specifically
                if "QuantDbrxExperts" in type(sub_module.experts).__name__:
                    # For DBRX, experts are in sub_module.experts.mlp and linear layers are ModuleLists
                    experts_mlp = sub_module.experts.mlp
                    if hasattr(experts_mlp, linear_name):
                        linear_modulelist = getattr(experts_mlp, linear_name)
                        if hasattr(linear_modulelist, "__iter__"):
                            set_expert_quantizer_amax(
                                modules=list(linear_modulelist),
                                quantizer_attrs=["input_quantizer"],
                            )
                elif has_fused_experts_quantizers:
                    # _QuantFusedExperts: amax fallback is handled in _export_fused_experts
                    break
                elif (
                    "QuantGptOssExperts" in type(sub_module.experts).__name__
                    or "QuantLlama4TextExperts" in type(sub_module.experts).__name__
                ):
                    # Handle GPT-OSS / Llama4 fused experts specifically.
                    # Both use gate_up_proj and down_proj with singular input quantizers
                    # (gate_up_proj_input_quantizer/down_proj_input_quantizer); the actual
                    # amax fallback and weight export is performed in _process_quantized_modules.
                    gpt_oss_linear_names = ["gate_up_proj", "down_proj"]
                    for linear_name in gpt_oss_linear_names:
                        if hasattr(sub_module.experts, linear_name):
                            linear_module = getattr(sub_module.experts, linear_name)
                            if hasattr(linear_module, "input_quantizer"):
                                set_expert_quantizer_amax(
                                    modules=[linear_module],
                                    quantizer_attrs=["input_quantizer"],
                                )
                elif isinstance(sub_module.experts, collections.abc.Iterable):
                    # For other MoE models (like Mixtral) with iterable experts
                    try:
                        set_expert_quantizer_amax(
                            modules=[getattr(expert, linear_name) for expert in sub_module.experts],
                            quantizer_attrs=["input_quantizer"],
                        )
                    except AttributeError as e:
                        # Provide more helpful debugging information
                        expert_types = [type(expert).__name__ for expert in sub_module.experts]
                        raise AttributeError(
                            f"Failed to access attribute '{linear_name}' on experts. "
                            f"MoE module type: {type(sub_module).__name__}, "
                            f"Expert types: {expert_types}, "
                            f"Expected linear names: {expert_linear_names}. "
                            f"This suggests the get_expert_linear_names function may need "
                            f"to be updated for this model architecture. "
                            f"Original error: {e}"
                        ) from e
                else:
                    # Unsupported MoE model structure
                    raise NotImplementedError(
                        f"MoE model with experts type '{type(sub_module.experts).__name__}' is not supported in export."
                        f"Please file an issue or add support for this model architecture."
                    )

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    # Remove all hooks from the model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except ImportError:
        warnings.warn("accelerate is not installed, hooks will not be removed")

    quant_config = get_quant_config(model, is_modelopt_qlora=is_modelopt_qlora)

    # Add MTP layer prefixes to exclude_modules if they were excluded from quantization
    # This ensures they appear in quantization_config["ignore"] in config.json
    mtp_layer_prefixes = getattr(model, "_mtp_layer_prefixes", None)
    if mtp_layer_prefixes:
        exclude_modules = quant_config["quantization"].setdefault("exclude_modules", [])
        for prefix in mtp_layer_prefixes:
            # Add wildcard pattern to exclude all submodules under this MTP layer
            pattern = f"{prefix}*"
            if pattern not in exclude_modules:
                exclude_modules.append(pattern)
                print(f"Adding MTP layer to quantization_config ignore: {pattern}")

    # Safety net: sync any gate/up weight quantizer amaxes that
    # requantize_resmooth_fused_llm_layers did not reach (e.g. experts not
    # activated during the dummy forward, or non-standard expert naming).
    synced = sync_moe_gate_up_amax(model)
    if synced:
        warnings.warn(
            f"Found {synced} MoE expert gate/up projection pair(s) with mismatched "
            f"weight_scale_2 after requantize_resmooth_fused_llm_layers. "
            f"This typically means the dummy forward did not activate these experts. "
            f"Taking element-wise max of amaxes for serving-engine fusion."
        )

    # Merge per-side input_quantizer amaxes BEFORE _process_quantized_modules,
    # so the merged value flows into input_scale derivation downstream.
    synced_input = sync_tied_input_amax(model)
    if synced_input:
        print(
            f"sync_tied_input_amax: max-merged input_quantizer amaxes across "
            f"{synced_input} tied module group(s)"
        )

    # Process all quantized modules and export weights
    _process_quantized_modules(model, dtype, is_modelopt_qlora)

    # Reconstruct fused MoELinear: per-expert _QuantLinear weights → original 3D format
    from modelopt.torch.quantization.plugins.huggingface import _reconstruct_fused_moe_linear

    _reconstruct_fused_moe_linear(model)

    if accelerator is not None:
        # Gather state_dict from all ranks
        quantized_state_dict = accelerator.get_state_dict(model)
    else:
        quantized_state_dict = model.state_dict()

    # We define kv cache scale as amax / 448 for both FP8 and NVFP4 KV cache quantization.
    kv_cache_max_bound = 448
    kv_cache_format = quant_config["quantization"]["kv_cache_quant_algo"]

    # Reorder so canonical-side tied keys (per HF's _tied_weights_keys)
    # iterate first into postprocess_state_dict's first-wins data_ptr dedup.
    # Self-gated to DiffusionGemma inside _reorder_canonical_first; no-op
    # for every other model.
    quantized_state_dict = _reorder_canonical_first(quantized_state_dict, model)

    quantized_state_dict = postprocess_state_dict(
        quantized_state_dict, kv_cache_max_bound, kv_cache_format, is_modelopt_qlora
    )

    return quantized_state_dict, quant_config


def _fuse_qkv_linears_diffusion(
    model: nn.Module,
    dummy_forward_fn: Callable[[], None] | None = None,
    strict: bool = False,
) -> None:
    """Fuse QKV linear layers that share the same input for diffusion models.

    This function uses forward hooks to dynamically identify linear modules that
    share the same input tensor (e.g., q_proj, k_proj, v_proj in attention).
    For these modules, it unifies their input and weight amax values.

    Note: This is a simplified version for diffusion models that:
    - Handles QKV fusion (shared input detection)
    - Filters to only fuse actual QKV projection layers (not AdaLN, FFN, etc.)
    - Skips pre_quant_scale *fusion* (the export path promotes pre_quant_scale to
      module-level keys separately; see _promote_quantizer_tensors_to_module)
    - Skips FFN fusion with layernorm (TODO for future)

    Args:
        model: The diffusion model component (e.g., transformer, unet).
        dummy_forward_fn: Optional callable to run a dummy forward pass. Use this
            for diffusion-like models whose forward signature is not compatible
            with `generate_diffusion_dummy_inputs`.
    """
    quantization_format = get_quantization_format(model)

    if quantization_format == QUANTIZATION_NONE:
        return

    if dummy_forward_fn is None:
        dummy_forward_fn = generate_diffusion_dummy_forward_fn(model)

    # Collect modules sharing the same input
    try:
        input_to_linear, _ = collect_shared_input_modules(
            model, dummy_forward_fn, collect_layernorms=False
        )
    except Exception as e:
        if strict:
            raise RuntimeError(
                f"QKV fusion dummy forward failed for {type(model).__name__}; a working "
                f"dummy forward is required to export this model correctly. Original error: {e}"
            ) from e
        print(f"Warning: Failed to run dummy forward for QKV fusion: {e}")
        print("Skipping QKV fusion. Quantization may still work but amax values won't be unified.")
        return

    if not input_to_linear:
        print("No quantized linear modules found for QKV fusion.")
        return

    # Fuse the collected modules (QKV only for diffusion)
    _fuse_shared_input_modules(
        model,
        input_to_linear,
        output_to_layernorm=None,
        qkv_only=True,
        fuse_layernorms=False,
        quantization_format=quantization_format,
    )


def _detect_svdquant_rank(component: nn.Module) -> int | None:
    """Return the single SVDQuant low-rank dimension shared by the SVDQuant linears.

    ``svdquant_lora_a`` has shape ``(rank, in_features)``, so its first dimension is
    the low-rank size. A single global ``lora_rank`` is written to the checkpoint
    config, so all SVDQuant linears are expected to share one rank; an inconsistency
    is raised rather than silently recording one module's rank for all. Returns
    ``None`` when no SVDQuant LoRA factors are present.
    """
    ranks: set[int] = set()
    for _, sub_module in component.named_modules():
        weight_quantizer = getattr(sub_module, "weight_quantizer", None)
        lora_a = getattr(weight_quantizer, "svdquant_lora_a", None)
        if lora_a is not None:
            ranks.add(int(lora_a.shape[0]))
    if not ranks:
        return None
    if len(ranks) > 1:
        raise ValueError(f"Inconsistent SVDQuant ranks across modules: {sorted(ranks)}")
    return next(iter(ranks))


def _promote_quantizer_tensors_to_module(component: nn.Module) -> None:
    """Promote quantizer-owned export tensors onto their parent linear module.

    The diffusers export path saves via ``save_pretrained`` inside
    :func:`hide_quantizers_from_state_dict` (which deletes the ``weight_quantizer``
    / ``input_quantizer`` submodules) and -- unlike the transformers path -- does
    NOT run :func:`postprocess_state_dict`. Without this step the AWQ smoothing
    scale and the SVDQuant low-rank factors would be dropped from the exported
    checkpoint. We register them as module buffers under clean, AWQ-aligned keys
    so they are embedded in the component's main safetensors:

    - ``input_quantizer._pre_quant_scale`` -> ``<module>.pre_quant_scale``
      (the same key the transformers/AWQ path produces via postprocess_state_dict)
    - ``weight_quantizer.svdquant_lora_a`` -> ``<module>.svdquant_lora_a``
    - ``weight_quantizer.svdquant_lora_b`` -> ``<module>.svdquant_lora_b``

    This runs after :func:`_process_quantized_modules` (which leaves these
    quantizer buffers in place) and before ``save_pretrained``.
    """
    for _, sub_module in component.named_modules():
        if not is_quantlinear(sub_module):
            continue

        # register_buffer overwrites an existing buffer of the same name, so a
        # repeated export refreshes (rather than keeps stale) promoted tensors.
        input_quantizer = getattr(sub_module, "input_quantizer", None)
        pre_quant_scale = getattr(input_quantizer, "_pre_quant_scale", None)
        if pre_quant_scale is not None:
            sub_module.register_buffer("pre_quant_scale", pre_quant_scale.detach().clone())

        weight_quantizer = getattr(sub_module, "weight_quantizer", None)
        lora_a = getattr(weight_quantizer, "svdquant_lora_a", None)
        lora_b = getattr(weight_quantizer, "svdquant_lora_b", None)
        if lora_a is not None and lora_b is not None:
            sub_module.register_buffer("svdquant_lora_a", lora_a.detach().clone())
            sub_module.register_buffer("svdquant_lora_b", lora_b.detach().clone())


def _remove_promoted_quantizer_tensors(component: nn.Module) -> None:
    """Undo :func:`_promote_quantizer_tensors_to_module`.

    Removes the temporary module-level export buffers (``svdquant_lora_a/b`` and
    ``pre_quant_scale``) so the live module is unchanged after export, keeping
    repeated export / post-export module reuse correct. The quantizer-owned tensors
    (``weight_quantizer.svdquant_lora_a/b``, ``input_quantizer._pre_quant_scale``)
    are left untouched.
    """
    for _, sub_module in component.named_modules():
        for buffer_name in ("svdquant_lora_a", "svdquant_lora_b", "pre_quant_scale"):
            if buffer_name in getattr(sub_module, "_buffers", {}):
                del sub_module._buffers[buffer_name]


def _export_diffusers_checkpoint(
    pipe: Any,
    dtype: torch.dtype | None,
    export_dir: Path,
    components: list[str] | None,
    max_shard_size: int | str = "10GB",
    **kwargs,
) -> None:
    """Internal: Export diffusion(-like) model/pipeline checkpoint.

    This function handles the export of:
    - diffusers models: DiffusionPipeline and individual ModelMixin components.
    - LTX-2 pipelines (duck-typed): exports stage-1 transformer only.

    Args:
        pipe: The model or pipeline to export.
        dtype: The data type for weight conversion. If None, will be inferred from model.
        export_dir: The directory to save the exported checkpoint.
        components: Optional list of component names to export. Only used for pipelines.
            If None, all components are exported.
        max_shard_size: Maximum size of each shard file. If the model exceeds this size,
            it will be sharded into multiple files and a .safetensors.index.json will be
            created. Use smaller values like "5GB" or "2GB" to force sharding.
        **kwargs: Runtime-specific post-processing options forwarded to
            :func:`_postprocess_safetensors`. See its docstring for details.
    """
    export_dir = Path(export_dir)

    # Get all pipeline components (nn.Module, tokenizers, schedulers, etc.)
    all_components = get_diffusion_components(pipe, components)

    if not all_components:
        warnings.warn("No exportable components found in the model.")
        return

    # Separate nn.Module components for quantization-aware export
    module_components = {
        name: comp for name, comp in all_components.items() if isinstance(comp, nn.Module)
    }

    # Best-effort diffusers pipeline check (kept for folder layout + model_index.json behavior)
    is_diffusers_pipe = False
    if HAS_DIFFUSERS:
        try:
            from diffusers import DiffusionPipeline as _DiffusionPipeline

            is_diffusers_pipe = isinstance(pipe, _DiffusionPipeline)
        except Exception:
            is_diffusers_pipe = False

    # Export each nn.Module component with quantization handling
    for component_name, component in module_components.items():
        is_quantized = has_quantized_modules(component)
        status = "quantized" if is_quantized else "non-quantized"
        print(f"Exporting component: {component_name} ({status})")

        # Determine component export directory
        # For pipelines, each component goes in a subfolder
        if is_diffusers_pipe:
            component_export_dir = export_dir / component_name
        else:
            component_export_dir = export_dir

        component_export_dir.mkdir(parents=True, exist_ok=True)

        # Infer dtype if not provided
        component_dtype = dtype if dtype is not None else infer_dtype_from_model(component)

        if is_quantized:
            # Fuse QKV linears that share the same input (unify amax values)
            # This is similar to requantize_resmooth_fused_llm_layers but simplified for diffusion
            # TODO: Add FFN fusion for AWQ-style quantization (pre_quant_scale is
            # promoted to module keys at export by _promote_quantizer_tensors_to_module below)
            print(f"  Running QKV fusion for {component_name}...")
            # Qwen-Image's packed-latent forward signature is non-standard; if the
            # dummy forward fails for it, fail loudly rather than silently skipping
            # fusion (which would export un-unified amax values).
            is_qwen_component = "qwen" in type(component).__name__.lower()
            _fuse_qkv_linears_diffusion(component, strict=is_qwen_component)

            # Process quantized modules (convert weights, register scales)
            _process_quantized_modules(component, component_dtype, is_modelopt_qlora=False)

            # Promote quantizer-owned tensors (AWQ pre_quant_scale and SVDQuant
            # LoRA factors) onto the module so they survive
            # hide_quantizers_from_state_dict and are embedded in the component's
            # main safetensors under clean, AWQ-aligned keys.
            _promote_quantizer_tensors_to_module(component)

            # Build the quantization config + save inside try/finally so the temporary
            # promoted buffers are always removed, even if save / post-process / config
            # update raises (keeps the live module reusable for a repeated export).
            try:
                quant_config = get_quant_config(component, is_modelopt_qlora=False)
                if quant_config:
                    quantization_details = quant_config.get("quantization", {})
                    # Record the SVDQuant low-rank size so consumers know the LoRA shape.
                    if quantization_details.get("quant_algo") == "NVFP4_SVD":
                        svdquant_rank = _detect_svdquant_rank(component)
                        if svdquant_rank is not None:
                            quantization_details["lora_rank"] = svdquant_rank
                hf_quant_config = (
                    convert_hf_quant_config_format(quant_config) if quant_config else None
                )

                # Save the component
                # - diffusers ModelMixin.save_pretrained does NOT accept state_dict parameter
                # - for non-diffusers modules (e.g., LTX-2 transformer), fall back to torch.save
                if hasattr(component, "save_pretrained"):
                    with hide_quantizers_from_state_dict(component):
                        component.save_pretrained(
                            component_export_dir, max_shard_size=max_shard_size
                        )
                else:
                    with hide_quantizers_from_state_dict(component):
                        _save_component_state_dict_safetensors(component, component_export_dir)

                # Post-process — merge, metadata, padding, swizzle
                _postprocess_safetensors(
                    component_export_dir,
                    pipe,
                    hf_quant_config=hf_quant_config,
                    **kwargs,
                )

                # Update config.json with quantization info
                if hf_quant_config is not None:
                    config_path = component_export_dir / "config.json"
                    if config_path.exists():
                        with open(config_path) as file:
                            config_data = json.load(file)
                        config_data["quantization_config"] = hf_quant_config
                        with open(config_path, "w") as file:
                            json.dump(config_data, file, indent=4)
            finally:
                # Drop the temporary promoted export buffers so the live module is
                # unchanged after export (supports repeated export / module reuse).
                _remove_promoted_quantizer_tensors(component)
        # Non-quantized component: just save as-is
        elif hasattr(component, "save_pretrained"):
            component.save_pretrained(component_export_dir, max_shard_size=max_shard_size)
        else:
            _save_component_state_dict_safetensors(component, component_export_dir)

        # Update config.json with sparse attention info (both quantized and non-quantized)
        if export_sparse_attention_config is not None:
            sparse_attn_config = export_sparse_attention_config(component)
            if sparse_attn_config is not None:
                config_path = component_export_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as file:
                        config_data = json.load(file)
                    config_data["sparse_attention_config"] = sparse_attn_config
                    with open(config_path, "w") as file:
                        json.dump(config_data, file, indent=4)
                    print(f"  Added sparse_attention_config to {config_path.name}")

        print(f"  Saved to: {component_export_dir}")

    # Export non-nn.Module components (tokenizers, schedulers, feature extractors, etc.)
    if is_diffusers_pipe:
        for component_name, component in all_components.items():
            # Skip nn.Module components (already handled above)
            if isinstance(component, nn.Module):
                continue

            component_export_dir = export_dir / component_name
            component_export_dir.mkdir(parents=True, exist_ok=True)

            print(f"Exporting component: {component_name} ({type(component).__name__})")

            # Handle different component types
            if hasattr(component, "save_pretrained"):
                # Tokenizers, feature extractors, image processors
                component.save_pretrained(component_export_dir)
            elif hasattr(component, "save_config"):
                # Schedulers
                component.save_config(component_export_dir)
            else:
                warnings.warn(
                    f"Component '{component_name}' of type {type(component).__name__} "
                    "does not have save_pretrained or save_config method. Skipping."
                )
                continue

            print(f"  Saved to: {component_export_dir}")

    # For pipelines, also save model_index.json
    if is_diffusers_pipe:
        model_index_path = export_dir / "model_index.json"
        is_partial_export = components is not None

        # For full export, preserve original model_index.json when possible.
        # For partial export, skip this to avoid listing non-exported components.
        if not is_partial_export:
            source_path = getattr(pipe, "name_or_path", None) or getattr(
                getattr(pipe, "config", None), "_name_or_path", None
            )
            if source_path:
                candidate_model_index = Path(source_path) / "model_index.json"
                if candidate_model_index.exists():
                    with open(candidate_model_index) as file:
                        model_index = json.load(file)
                    with open(model_index_path, "w") as file:
                        json.dump(model_index, file, indent=4)

        # Full-export fallback to Diffusers-native config serialization.
        # Partial export skips this for the same reason as above.
        if not is_partial_export and not model_index_path.exists() and hasattr(pipe, "save_config"):
            pipe.save_config(export_dir)

        # Last resort: synthesize a minimal model_index.json from exported components.
        if not model_index_path.exists() and hasattr(pipe, "config") and pipe.config is not None:
            model_index = {
                "_class_name": type(pipe).__name__,
                "_diffusers_version": diffusers.__version__,
            }
            for name, comp in all_components.items():
                module = type(comp).__module__
                library = module.split(".")[0]
                model_index[name] = [library, type(comp).__name__]

            with open(model_index_path, "w") as file:
                json.dump(model_index, file, indent=4)

    print(f"Export complete. Saved to: {export_dir}")


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


def export_speculative_decoding(
    model: torch.nn.Module,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
) -> None:
    """Export speculative decoding HuggingFace model checkpoint."""
    assert has_spec_opt(model), "Model is not optimized for speculative decoding."

    exporter: SpeculativeDecodingExporter = model.get_exporter()
    exporter.export(export_dir, dtype)


def export_hf_checkpoint(
    model: Any,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    max_shard_size: int | str = "10GB",
    **kwargs,
):
    """Export quantized HuggingFace model checkpoint (transformers or diffusers).

    This function automatically detects whether the model is from transformers
    or diffusers and applies the appropriate export logic.

    Args:
        model: The full torch model to export. The actual quantized model may be a submodule.
            Supports both transformers models (e.g., LlamaForCausalLM) and diffusers
            models/pipelines (e.g., StableDiffusionPipeline, UNet2DConditionModel).
        dtype: The weights data type to export the unquantized layers or the default
            model data type if None.
        export_dir: The target export path.
        save_modelopt_state: Whether to save the modelopt state_dict.
        components: Only used for diffusers pipelines. Optional list of component names
            to export. If None, all quantized components are exported.
        extra_state_dict: Extra state dictionary to add to the exported model.
        max_shard_size: Maximum size of each safetensors shard file. Defaults to "10GB".
        **kwargs: Runtime-specific post-processing options forwarded to
            :func:`_postprocess_safetensors` for diffusion model exports.
            See its docstring for supported keys.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    is_diffusers_obj = False
    if HAS_DIFFUSERS:
        is_diffusers_obj = is_diffusers_object(model)
    if is_diffusers_obj:
        _export_diffusers_checkpoint(
            model,
            dtype,
            export_dir,
            components,
            max_shard_size,
            **kwargs,
        )
        return

    try:
        post_state_dict, hf_quant_config = _export_transformers_checkpoint(model, dtype, **kwargs)

        # Remove hf_quantizer from model so post_state_dict can be exported.
        if getattr(model, "hf_quantizer", None) is not None:
            model.hf_quantizer = None

        export_state_dict = {**post_state_dict, **(extra_state_dict or {})}

        # transformers may have applied a load-time conversion_mapping (fused gate_up_proj,
        # renamed MoE leaves, reordered model/language_model prefix), so the in-memory names
        # differ from the original hub checkpoint. Reverse it quantization-aware so exported
        # tensor names stay aligned with the hub checkpoint (the unified-checkpoint contract).
        # transformers' own revert_weight_conversion errors on 0-d scalar scale tensors, so we
        # do it here. The same rename is applied to the quant-config module references
        # (exclude_modules / quantized_layers keys) so a deployment loader matches them against
        # the reverted hub-named modules (otherwise an excluded BF16 layer is loaded as quantized
        # and fails). Best-effort and atomic: any failure (an op we cannot reverse yet,
        # transformers API drift, unexpected shapes) falls back to the in-memory names for BOTH
        # weights and config so they stay mutually consistent.
        try:
            name_mapper = build_reverse_name_mapper(model)
            export_state_dict = revert_weight_conversion_quant_aware(model, export_state_dict)
            if name_mapper is not None and hf_quant_config:
                revert_quant_config_names(hf_quant_config.get("quantization", {}), name_mapper)
        except Exception as exc:
            warnings.warn(
                f"Quant-aware reverse weight conversion skipped ({exc}); exported tensor "
                "names may not match the original HF hub checkpoint."
            )

        # Only treat the export as quantized when at least one quant_algo field is set.
        # get_quant_config always returns a dict (even for sparsity-only or unmodified models),
        # so emitting hf_quant_config.json unconditionally produces a file with
        # "quant_algo": null that downstream loaders (e.g. TensorRT-LLM) reject as a
        # malformed pre-quantized checkpoint.
        quantization_details = (hf_quant_config or {}).get("quantization", {})
        is_quantized_export = (
            quantization_details.get("quant_algo") is not None
            or quantization_details.get("kv_cache_quant_algo") is not None
        )

        if is_quantized_export:
            # Save hf_quant_config.json for backward compatibility
            with open(f"{export_dir}/hf_quant_config.json", "w") as file:
                json.dump(hf_quant_config, file, indent=4)

            hf_quant_config = convert_hf_quant_config_format(hf_quant_config)
        else:
            hf_quant_config = None

        # Keep transformers' own revert_weight_conversion disabled (the quant-aware reverse
        # above replaces it): it can't handle quantized state dicts (RuntimeError on 0-d scalar
        # scale tensors). Patch both the source and importing module since modeling_utils does
        # `from core_model_loading import revert_weight_conversion`.
        _patches = _patch_revert_weight_conversion()

        _sanitize_generation_config_for_save(model)

        try:
            model.save_pretrained(
                export_dir,
                state_dict=export_state_dict,
                save_modelopt_state=save_modelopt_state,
                max_shard_size=max_shard_size,
            )
        finally:
            _unpatch_revert_weight_conversion(_patches)

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        sanitize_hf_config_for_deployment(config_data, model)

        if hf_quant_config is not None:
            config_data["quantization_config"] = hf_quant_config

        # Add sparse attention config if available
        if export_sparse_attention_config is not None:
            sparse_attn_config = export_sparse_attention_config(model)
            if sparse_attn_config is not None:
                config_data["sparse_attention_config"] = sparse_attn_config

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)

    except Exception as e:
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            " can be saved with torch.save for further inspection."
        )
        raise e
