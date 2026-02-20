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
from collections.abc import Callable
from contextlib import contextmanager
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn

from .layer_utils import is_quantlinear

DiffusionPipeline: type[Any] | None
ModelMixin: type[Any] | None
try:  # diffusers is optional for LTX-2 export paths
    from diffusers import DiffusionPipeline as _DiffusionPipeline
    from diffusers import ModelMixin as _ModelMixin

    DiffusionPipeline = _DiffusionPipeline
    ModelMixin = _ModelMixin
    _HAS_DIFFUSERS = True
except Exception:  # pragma: no cover
    DiffusionPipeline = None
    ModelMixin = None
    _HAS_DIFFUSERS = False

TI2VidTwoStagesPipeline: type[Any] | None
try:  # optional for LTX-2 export paths
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline as _TI2VidTwoStagesPipeline

    TI2VidTwoStagesPipeline = _TI2VidTwoStagesPipeline
except Exception:  # pragma: no cover
    TI2VidTwoStagesPipeline = None


def is_diffusers_object(model: Any) -> bool:
    """Return True if model is a diffusers pipeline/component or LTX-2 pipeline."""
    if not _HAS_DIFFUSERS:
        return False

    diffusers_types: tuple[type, ...] = ()
    if DiffusionPipeline is not None:
        diffusers_types = (*diffusers_types, DiffusionPipeline)
    if ModelMixin is not None:
        diffusers_types = (*diffusers_types, ModelMixin)
    if TI2VidTwoStagesPipeline is not None:
        diffusers_types = (*diffusers_types, TI2VidTwoStagesPipeline)

    if not diffusers_types:
        return False

    return isinstance(model, diffusers_types)


def generate_diffusion_dummy_inputs(
    model: nn.Module, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor] | None:
    """Generate dummy inputs for diffusion model forward pass.

    Different diffusion models have very different input formats:
    - DiTTransformer2DModel: 4D hidden_states + class_labels
    - FluxTransformer2DModel: 3D hidden_states + encoder_hidden_states + img_ids + txt_ids + pooled_projections
    - SD3Transformer2DModel: 4D hidden_states + encoder_hidden_states + pooled_projections
    - UNet2DConditionModel: 4D sample + timestep + encoder_hidden_states
    - WanTransformer3DModel: 5D hidden_states + encoder_hidden_states + timestep

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
    def _is_model_type(module_path: str, class_name: str, fallback: bool) -> bool:
        try:
            module = import_module(module_path)
            return isinstance(model, getattr(module, class_name))
        except (ImportError, AttributeError):
            return fallback

    is_flux = _is_model_type(
        "diffusers.models.transformers",
        "FluxTransformer2DModel",
        "flux" in model_class_name.lower(),
    )
    is_sd3 = _is_model_type(
        "diffusers.models.transformers",
        "SD3Transformer2DModel",
        "sd3" in model_class_name.lower(),
    )
    is_dit = _is_model_type(
        "diffusers.models.transformers",
        "DiTTransformer2DModel",
        model_class_name == "DiTTransformer2DModel",
    )
    is_wan = _is_model_type(
        "diffusers.models.transformers",
        "WanTransformer3DModel",
        "wan" in model_class_name.lower(),
    )
    is_unet = _is_model_type(
        "diffusers.models.unets",
        "UNet2DConditionModel",
        "unet" in model_class_name.lower(),
    )

    cfg = getattr(model, "config", None)

    def _flux_inputs() -> dict[str, torch.Tensor]:
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

    def _sd3_inputs() -> dict[str, torch.Tensor]:
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

    def _dit_inputs() -> dict[str, torch.Tensor]:
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

    def _unet_inputs() -> dict[str, torch.Tensor]:
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

    def _wan_inputs() -> dict[str, torch.Tensor]:
        # WanTransformer3DModel: 5D hidden_states (batch, channels, frames, height, width)
        # Requires: hidden_states, encoder_hidden_states, timestep
        in_channels = getattr(cfg, "in_channels", 16)
        text_dim = getattr(cfg, "text_dim", 4096)
        max_seq_len = getattr(cfg, "rope_max_seq_len", 512)

        patch_dtype = getattr(getattr(model, "patch_embedding", None), "weight", None)
        patch_dtype = patch_dtype.dtype if patch_dtype is not None else dtype
        text_embedder = getattr(getattr(model, "condition_embedder", None), "text_embedder", None)
        text_dtype = (
            text_embedder.linear_1.weight.dtype
            if text_embedder is not None and hasattr(text_embedder, "linear_1")
            else dtype
        )

        # Wan expects num_frames = 4 * n + 1; keep n small for dummy forward
        num_frames = 5
        text_seq_len = min(max_seq_len, 512)

        # Keep spatial dims small and divisible by patch size (default 2x2)
        height = 8
        width = 8

        return {
            "hidden_states": torch.randn(
                batch_size, in_channels, num_frames, height, width, device=device, dtype=patch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, text_seq_len, text_dim, device=device, dtype=text_dtype
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), device=device),
            "return_dict": False,
        }

    def _generic_transformer_inputs() -> dict[str, torch.Tensor] | None:
        # Try generic transformer handling for other model types
        # Check if model has common transformer attributes
        if cfg is None:
            return None
        if not (hasattr(cfg, "in_channels") and hasattr(cfg, "sample_size")):
            return None

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

    model_input_builders = [
        ("flux", is_flux, _flux_inputs),
        ("sd3", is_sd3, _sd3_inputs),
        ("dit", is_dit, _dit_inputs),
        ("wan", is_wan, _wan_inputs),
        ("unet", is_unet, _unet_inputs),
    ]

    for _, matches, build_inputs in model_input_builders:
        if matches:
            return build_inputs()

    generic_inputs = _generic_transformer_inputs()
    if generic_inputs is not None:
        return generic_inputs

    return None


def generate_diffusion_dummy_forward_fn(model: nn.Module) -> Callable[[], None]:
    """Create a dummy forward function for diffusion(-like) models.

    - For diffusers components, this uses `generate_diffusion_dummy_inputs()` and calls `model(**kwargs)`.
    - For LTX-2 stage-1 transformer (X0Model), the forward signature is
      `model(video: Modality|None, audio: Modality|None, perturbations: BatchedPerturbationConfig)`,
      so we build tiny `ltx_core` dataclasses and call the model directly.
    """
    # Duck-typed LTX-2 stage-1 transformer wrapper
    velocity_model = getattr(model, "velocity_model", None)
    if velocity_model is not None:

        def _ltx2_dummy_forward() -> None:
            try:
                from ltx_core.guidance.perturbations import BatchedPerturbationConfig
                from ltx_core.model.transformer.modality import Modality
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "LTX-2 export requires `ltx_core` to be installed (Modality, BatchedPerturbationConfig)."
                ) from e

            # Small shapes for speed/memory
            batch_size = 1
            v_seq_len = 8
            a_seq_len = 8
            ctx_len = 4

            device = next(model.parameters()).device
            default_dtype = next(model.parameters()).dtype

            def _param_dtype(module: Any, fallback: torch.dtype) -> torch.dtype:
                w = getattr(getattr(module, "weight", None), "dtype", None)
                return w if isinstance(w, torch.dtype) else fallback

            def _positions(bounds_dims: int, seq_len: int) -> torch.Tensor:
                # [B, dims, seq_len, 2] bounds (start/end)
                pos = torch.zeros(
                    (batch_size, bounds_dims, seq_len, 2), device=device, dtype=torch.float32
                )
                pos[..., 1] = 1.0
                return pos

            has_video = hasattr(velocity_model, "patchify_proj") and hasattr(
                velocity_model, "caption_projection"
            )
            has_audio = hasattr(velocity_model, "audio_patchify_proj") and hasattr(
                velocity_model, "audio_caption_projection"
            )
            if not has_video and not has_audio:
                raise ValueError(
                    "Unsupported LTX-2 velocity model: missing both video and audio preprocessors."
                )

            video = None
            if has_video:
                v_in = int(velocity_model.patchify_proj.in_features)
                v_caption_in = int(velocity_model.caption_projection.linear_1.in_features)
                v_latent_dtype = _param_dtype(velocity_model.patchify_proj, default_dtype)
                v_ctx_dtype = _param_dtype(
                    velocity_model.caption_projection.linear_1, default_dtype
                )
                video = Modality(
                    enabled=True,
                    latent=torch.randn(
                        batch_size, v_seq_len, v_in, device=device, dtype=v_latent_dtype
                    ),
                    # LTX `X0Model` uses `timesteps` as the sigma tensor in `to_denoised(sample, velocity, sigma)`.
                    # It must be broadcastable to `[B, T, D]`, so we use `[B, T, 1]`.
                    timesteps=torch.full(
                        (batch_size, v_seq_len, 1), 0.5, device=device, dtype=torch.float32
                    ),
                    positions=_positions(bounds_dims=3, seq_len=v_seq_len),
                    context=torch.randn(
                        batch_size, ctx_len, v_caption_in, device=device, dtype=v_ctx_dtype
                    ),
                    context_mask=None,
                )

            audio = None
            if has_audio:
                a_in = int(velocity_model.audio_patchify_proj.in_features)
                a_caption_in = int(velocity_model.audio_caption_projection.linear_1.in_features)
                a_latent_dtype = _param_dtype(velocity_model.audio_patchify_proj, default_dtype)
                a_ctx_dtype = _param_dtype(
                    velocity_model.audio_caption_projection.linear_1, default_dtype
                )
                audio = Modality(
                    enabled=True,
                    latent=torch.randn(
                        batch_size, a_seq_len, a_in, device=device, dtype=a_latent_dtype
                    ),
                    timesteps=torch.full(
                        (batch_size, a_seq_len, 1), 0.5, device=device, dtype=torch.float32
                    ),
                    positions=_positions(bounds_dims=1, seq_len=a_seq_len),
                    context=torch.randn(
                        batch_size, ctx_len, a_caption_in, device=device, dtype=a_ctx_dtype
                    ),
                    context_mask=None,
                )

            perturbations = BatchedPerturbationConfig.empty(batch_size)
            model(video, audio, perturbations)

        return _ltx2_dummy_forward

    # Default: diffusers-style `model(**kwargs)`
    def _diffusers_dummy_forward() -> None:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dummy_inputs = generate_diffusion_dummy_inputs(model, device, dtype)
        if dummy_inputs is None:
            raise ValueError(
                f"Unknown model type '{type(model).__name__}', cannot generate dummy inputs."
            )
        model(**dummy_inputs)

    return _diffusers_dummy_forward


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

    # Check last or second-to-last for cases like "attn.to_q.weight"
    return last_part in qkv_patterns or second_last in qkv_patterns


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


def get_diffusion_components(
    model: Any,
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Get all exportable components from a diffusion(-like) pipeline.

    Supports:
    - diffusers `DiffusionPipeline`: returns `pipeline.components`
    - diffusers component `nn.Module` (e.g., UNet / transformer)
    - LTX-2 pipeline (duck-typed): returns stage-1 transformer only as `stage_1_transformer`

    Args:
        model: The pipeline or component.
        components: Optional list of component names to filter. If None, all
            components are returned.

    Returns:
        Dictionary mapping component names to their instances (can be nn.Module,
        tokenizers, schedulers, etc.).
    """
    # LTX-2 pipeline: duck-typed stage-1 transformer export
    stage_1 = getattr(model, "stage_1_model_ledger", None)
    transformer_fn = getattr(stage_1, "transformer", None)
    if stage_1 is not None and callable(transformer_fn):
        all_components: dict[str, Any] = {"stage_1_transformer": stage_1.transformer()}
        if components is not None:
            filtered = {name: comp for name, comp in all_components.items() if name in components}
            missing = set(components) - set(filtered.keys())
            if missing:
                warnings.warn(f"Requested components not found in pipeline: {missing}")
            return filtered
        return all_components

    # diffusers pipeline
    if _HAS_DIFFUSERS and DiffusionPipeline is not None and isinstance(model, DiffusionPipeline):
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

    if isinstance(model, nn.Module):
        # Single component model (e.g., UNet2DConditionModel, DiTTransformer2DModel, FluxTransformer2DModel)
        component_name = type(model).__name__
        all_components = {component_name: model}

        if components is not None:
            filtered = {name: comp for name, comp in all_components.items() if name in components}
            missing = set(components) - set(filtered.keys())
            if missing:
                warnings.warn(f"Requested components not found in pipeline: {missing}")
            return filtered

        return all_components

    raise TypeError(f"Expected DiffusionPipeline or nn.Module, got {type(model).__name__}")


# Backward-compatible alias
get_diffusers_components = get_diffusion_components


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
