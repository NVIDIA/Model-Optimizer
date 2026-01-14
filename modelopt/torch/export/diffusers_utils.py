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

import warnings
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline

from .layer_utils import is_quantlinear


def generate_diffusion_dummy_inputs(
    model: nn.Module, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor] | None:
    """Generate dummy inputs for diffusion model forward pass.

    Different diffusion models have very different input formats:
    - DiTTransformer2DModel: 4D hidden_states + class_labels
    - FluxTransformer2DModel: 3D hidden_states + encoder_hidden_states + img_ids + txt_ids + pooled_projections
    - SD3Transformer2DModel: 4D hidden_states + encoder_hidden_states + pooled_projections
    - UNet2DConditionModel: 4D sample + timestep + encoder_hidden_states

    Args:
        model: The diffusion model component.
        device: Device to create tensors on.
        dtype: Data type for tensors.

    Returns:
        Dictionary of dummy inputs, or None if model type is not supported.
    """
    model_class_name = type(model).__name__
    batch_size = 1

    # Try to import specific model classes for isinstance checks
    try:
        from diffusers.models.transformers import FluxTransformer2DModel

        is_flux = isinstance(model, FluxTransformer2DModel)
    except ImportError:
        is_flux = "flux" in model_class_name.lower()

    try:
        from diffusers.models.transformers import SD3Transformer2DModel

        is_sd3 = isinstance(model, SD3Transformer2DModel)
    except ImportError:
        is_sd3 = "sd3" in model_class_name.lower()

    try:
        from diffusers.models.transformers import DiTTransformer2DModel

        is_dit = isinstance(model, DiTTransformer2DModel)
    except ImportError:
        is_dit = model_class_name == "DiTTransformer2DModel"

    try:
        from diffusers.models.unets import UNet2DConditionModel

        is_unet = isinstance(model, UNet2DConditionModel)
    except ImportError:
        is_unet = "unet" in model_class_name.lower()

    cfg = getattr(model, "config", None)

    if is_flux:
        # FluxTransformer2DModel: 3D hidden_states (batch, seq_len, in_channels)
        # Requires: hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids
        in_channels = getattr(cfg, "in_channels", 64)
        joint_attention_dim = getattr(cfg, "joint_attention_dim", 4096)
        pooled_projection_dim = getattr(cfg, "pooled_projection_dim", 768)
        guidance_embeds = getattr(cfg, "guidance_embeds", False)

        # Use small dimensions for dummy forward
        img_seq_len = 16  # 4x4 latent grid
        text_seq_len = 8

        dummy_inputs = {
            "hidden_states": torch.randn(
                batch_size, img_seq_len, in_channels, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, joint_attention_dim, device=device, dtype=dtype
            ),
            "pooled_projections": torch.randn(
                batch_size, pooled_projection_dim, device=device, dtype=dtype
            ),
            "timestep": torch.tensor([0.5], device=device, dtype=dtype).expand(batch_size),
            "img_ids": torch.zeros(img_seq_len, 3, device=device, dtype=torch.float32),
            "txt_ids": torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32),
            "return_dict": False,
        }
        if guidance_embeds:
            dummy_inputs["guidance"] = torch.tensor([3.5], device=device, dtype=torch.float32)
        return dummy_inputs

    elif is_sd3:
        # SD3Transformer2DModel: 4D hidden_states (batch, channels, height, width)
        # Requires: hidden_states, encoder_hidden_states, pooled_projections, timestep
        in_channels = getattr(cfg, "in_channels", 16)
        sample_size = getattr(cfg, "sample_size", 128)
        joint_attention_dim = getattr(cfg, "joint_attention_dim", 4096)
        pooled_projection_dim = getattr(cfg, "pooled_projection_dim", 2048)

        # Use smaller sample size for speed
        test_size = min(sample_size, 32)
        text_seq_len = 8

        return {
            "hidden_states": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, joint_attention_dim, device=device, dtype=dtype
            ),
            "pooled_projections": torch.randn(
                batch_size, pooled_projection_dim, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), device=device),
            "return_dict": False,
        }

    elif is_dit:
        # DiTTransformer2DModel: 4D hidden_states (batch, in_channels, height, width)
        # Requires: hidden_states, timestep, class_labels
        in_channels = getattr(cfg, "in_channels", 4)
        sample_size = getattr(cfg, "sample_size", 32)
        num_embeds_ada_norm = getattr(cfg, "num_embeds_ada_norm", 1000)

        # Use smaller sample size for speed
        test_size = min(sample_size, 16)

        return {
            "hidden_states": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, num_embeds_ada_norm, (batch_size,), device=device),
            "class_labels": torch.randint(0, num_embeds_ada_norm, (batch_size,), device=device),
            "return_dict": False,
        }

    elif is_unet:
        # UNet2DConditionModel: 4D sample (batch, in_channels, height, width)
        # Requires: sample, timestep, encoder_hidden_states
        in_channels = getattr(cfg, "in_channels", 4)
        sample_size = getattr(cfg, "sample_size", 64)
        cross_attention_dim = getattr(cfg, "cross_attention_dim", 768)

        # Use smaller sample size for speed
        test_size = min(sample_size, 32)
        text_seq_len = 8

        dummy_inputs = {
            "sample": torch.randn(
                batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), device=device),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, cross_attention_dim, device=device, dtype=dtype
            ),
            "return_dict": False,
        }

        # Handle SDXL additional conditioning
        if getattr(cfg, "addition_embed_type", None) == "text_time":
            # SDXL requires text_embeds and time_ids
            add_embed_dim = getattr(cfg, "projection_class_embeddings_input_dim", 2816)
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": torch.randn(
                    batch_size, add_embed_dim - 6 * 256, device=device, dtype=dtype
                ),
                "time_ids": torch.randn(batch_size, 6, device=device, dtype=dtype),
            }
        return dummy_inputs

    # Try generic transformer handling for other model types
    # Check if model has common transformer attributes
    elif cfg is not None:
        # Many transformers use 4D hidden_states with in_channels and sample_size
        if hasattr(cfg, "in_channels") and hasattr(cfg, "sample_size"):
            in_channels = cfg.in_channels
            sample_size = cfg.sample_size
            test_size = min(sample_size, 32)

            dummy_inputs = {
                "hidden_states": torch.randn(
                    batch_size, in_channels, test_size, test_size, device=device, dtype=dtype
                ),
                "timestep": torch.randint(0, 1000, (batch_size,), device=device),
                "return_dict": False,
            }

            # Add encoder_hidden_states if model has cross attention
            if hasattr(cfg, "joint_attention_dim"):
                text_seq_len = 8
                dummy_inputs["encoder_hidden_states"] = torch.randn(
                    batch_size, text_seq_len, cfg.joint_attention_dim, device=device, dtype=dtype
                )
                if hasattr(cfg, "pooled_projection_dim"):
                    dummy_inputs["pooled_projections"] = torch.randn(
                        batch_size, cfg.pooled_projection_dim, device=device, dtype=dtype
                    )
            elif hasattr(cfg, "cross_attention_dim"):
                text_seq_len = 8
                dummy_inputs["encoder_hidden_states"] = torch.randn(
                    batch_size, text_seq_len, cfg.cross_attention_dim, device=device, dtype=dtype
                )

            return dummy_inputs

    return None


def is_qkv_projection(module_name: str) -> bool:
    """Check if a module name corresponds to a QKV projection layer.

    In diffusers, QKV projections typically have names like:
    - to_q, to_k, to_v (most common in diffusers attention)
    - q_proj, k_proj, v_proj
    - query, key, value
    - add_q_proj, add_k_proj, add_v_proj (for additional attention in some models)

    We exclude:
    - norm*.linear (AdaLayerNorm modulation layers)
    - proj_out, proj_mlp (output projections)
    - ff.*, mlp.* (feed-forward layers)
    - to_out (output projection)

    Args:
        module_name: The full module name path.

    Returns:
        True if this is a QKV projection layer.
    """
    # Get the last component of the module name
    name_parts = module_name.split(".")
    last_part = name_parts[-1] if name_parts else ""
    second_last = name_parts[-2] if len(name_parts) >= 2 else ""

    # QKV projection patterns (positive matches)
    qkv_patterns = [
        "to_q",
        "to_k",
        "to_v",
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "value",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_added_q",
        "to_added_k",
        "to_added_v",
    ]

    # Check if the last part matches any QKV pattern
    if last_part in qkv_patterns:
        return True

    # Also check second-to-last for cases like "attn.to_q.weight"
    return second_last in qkv_patterns


def get_qkv_group_key(module_name: str) -> str:
    """Extract the parent attention block path and QKV type for grouping.

    QKV projections should only be fused within the same attention block AND
    for the same type of attention (main vs added/cross).

    Examples:
        - 'transformer_blocks.0.attn.to_q' -> 'transformer_blocks.0.attn.main'
        - 'transformer_blocks.0.attn.to_k' -> 'transformer_blocks.0.attn.main'
        - 'transformer_blocks.5.attn.add_q_proj' -> 'transformer_blocks.5.attn.add'
        - 'transformer_blocks.5.attn.add_k_proj' -> 'transformer_blocks.5.attn.add'

    Args:
        module_name: The full module name path.

    Returns:
        A string key representing the attention block and QKV type for grouping.
    """
    name_parts = module_name.split(".")
    last_part = name_parts[-1] if name_parts else ""

    # Determine if this is "main" QKV or "added" QKV (for cross-attention in some models)
    added_patterns = [
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_added_q",
        "to_added_k",
        "to_added_v",
    ]
    qkv_type = "add" if last_part in added_patterns else "main"

    # Find the parent attention block by removing the QKV projection name
    # e.g., 'transformer_blocks.0.attn.to_q' -> 'transformer_blocks.0.attn'
    parent_parts = name_parts[:-1]
    parent_path = ".".join(parent_parts) if parent_parts else ""

    return f"{parent_path}.{qkv_type}"


def get_diffusers_components(
    model: DiffusionPipeline,
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Get all exportable components from a diffusers pipeline.

    This function extracts all components from a DiffusionPipeline including
    nn.Module models, tokenizers, schedulers, feature extractors, etc.

    Args:
        model: The diffusers pipeline.
        components: Optional list of component names to filter. If None, all
            components are returned.

    Returns:
        Dictionary mapping component names to their instances (can be nn.Module,
        tokenizers, schedulers, etc.).
    """
    if isinstance(model, DiffusionPipeline):
        # Get all components from the pipeline
        all_components = {name: comp for name, comp in model.components.items() if comp is not None}

        # If specific components requested, filter to only those
        if components is not None:
            filtered = {name: comp for name, comp in all_components.items() if name in components}
            # Warn about requested components that don't exist
            missing = set(components) - set(filtered.keys())
            if missing:
                warnings.warn(f"Requested components not found in pipeline: {missing}")
            return filtered

        return all_components
    else:
        raise TypeError(f"Expected DiffusionPipeline for now, got {type(model).__name__}")


@contextmanager
def hide_quantizers_from_state_dict(model: nn.Module):
    """Context manager that temporarily removes quantizer modules from the model.

    This allows save_pretrained to save the model without quantizer buffers like _amax.
    The quantizers are restored after exiting the context.

    Args:
        model: The model with quantizers to temporarily hide.

    Yields:
        None - the model can be saved within the context.
    """
    # Store references to quantizers that we'll temporarily remove
    quantizer_backup: dict[str, dict[str, nn.Module]] = {}

    for name, module in model.named_modules():
        if is_quantlinear(module):
            backup = {}
            for attr in ["weight_quantizer", "input_quantizer", "output_quantizer"]:
                if hasattr(module, attr):
                    backup[attr] = getattr(module, attr)
                    delattr(module, attr)
            if backup:
                quantizer_backup[name] = backup

    try:
        yield
    finally:
        # Restore quantizers
        for name, backup in quantizer_backup.items():
            module = model.get_submodule(name)
            for attr, quantizer in backup.items():
                setattr(module, attr, quantizer)


def infer_dtype_from_model(model: nn.Module) -> torch.dtype:
    """Infer the dtype from a model's parameters.

    Args:
        model: The model to infer dtype from.

    Returns:
        The dtype of the model's parameters, defaulting to float16 if no parameters found.
    """
    for param in model.parameters():
        return param.dtype
    return torch.float16
