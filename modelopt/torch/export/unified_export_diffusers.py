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

"""Unified HF checkpoint export for diffusers models.

Split out of :mod:`unified_export_hf`, which it shares almost nothing with: the
transformers and diffusers paths meet only at the dispatch in ``export_hf_checkpoint``.
What they do share -- model preparation and per-module weight packing -- lives in
:mod:`hf_export_prep` and :mod:`hf_weight_export`.
"""

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .convert_hf_config import convert_hf_quant_config_format
from .diffusers_utils import build_layerwise_quant_metadata, pad_nvfp4_weights, swizzle_nvfp4_scales
from .hf_export_prep import _fuse_shared_input_modules, collect_shared_input_modules
from .hf_weight_export import _process_quantized_modules
from .layer_utils import is_quantlinear
from .model_config import QUANTIZATION_NONE
from .quant_utils import get_quant_config, get_quantization_format, has_quantized_modules

try:
    import diffusers

    from .diffusers_utils import (
        generate_diffusion_dummy_forward_fn,
        get_diffusion_components,
        get_diffusion_model_type,
        hide_quantizers_from_state_dict,
        infer_dtype_from_model,
        merge_diffusion_checkpoint,
    )

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    from modelopt.torch.sparsity.attention_sparsity.conversion import export_sparse_attention_config
except ImportError:
    export_sparse_attention_config = None


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
