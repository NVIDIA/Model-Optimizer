# Adapted from https://github.com/huggingface/diffusers/blob/275869dcae4ebcfee6a80253fdabc56033335020/src/diffusers/models/transformers/transformer_qwenimage.py

# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""Qwen-Image adapter and explicit output-projection conversion for PDD.

The masked joint-attention execution follows NVIDIA FastGen merge request 210.
"""

from __future__ import annotations

import math
import types
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..config import PDDConfig
from ..methods.pdd import PDDLayerSpec, PDDOutputProjection
from .qwen_image import build_img_shapes, pack_latents, unpack_latents

__all__ = [
    "QWEN_IMAGE_PDD_EXECUTION",
    "QWEN_IMAGE_PDD_LAYER_SPEC",
    "QwenImagePDDAdapter",
    "convert_qwen_image_to_pdd",
    "enable_qwen_image_pdd_forward",
    "freeze_qwen_image_pdd_unused_parameters",
    "require_qwen_image_pdd_forward",
    "restore_qwen_image_pdd_projection",
]

QWEN_IMAGE_PDD_EXECUTION = "qwen_image_pdd_masked_joint_attention_v1"

QWEN_IMAGE_PDD_LAYER_SPEC = PDDLayerSpec(
    projection_path="transformer.proj_out",
    head_layout="channel_major",
)

_CONTROLLED_MODEL_KWARGS = {
    "additional_t_cond",
    "attention_kwargs",
    "controlnet_block_samples",
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "guidance",
    "hidden_states",
    "img_shapes",
    "max_txt_seq_len",
    "return_dict",
    "timestep",
    "txt_seq_lens",
    "_pdd_fusion",
}

_QWEN_IMAGE_PDD_EXECUTION_ATTRIBUTE = "_modelopt_qwen_image_pdd_execution"


def _config_guidance_embeds(transformer: nn.Module) -> bool:
    config = getattr(transformer, "config", None)
    if isinstance(config, Mapping):
        return bool(config.get("guidance_embeds", False))
    return bool(getattr(config, "guidance_embeds", False))


def _config_value(transformer: nn.Module, name: str, default: Any = None) -> Any:
    config = getattr(transformer, "config", None)
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _qwen_image_pdd_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_hidden_states_mask: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
    img_shapes: list[Any] | None = None,
    txt_seq_lens: list[int] | None = None,
    guidance: torch.Tensor | None = None,
    max_txt_seq_len: int | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples: Any = None,
    additional_t_cond: torch.Tensor | None = None,
    return_dict: bool = True,
    _pdd_fusion: tuple[int, int, torch.Tensor] | None = None,
) -> Any:
    """Run Qwen-Image with the masked joint-attention contract required by PDD."""
    if not isinstance(timestep, torch.Tensor) or timestep.dtype != torch.float32:
        raise TypeError("Qwen PDD timestep must remain FP32 at transformer entry.")
    if not isinstance(encoder_hidden_states, torch.Tensor) or not isinstance(
        encoder_hidden_states_mask, torch.Tensor
    ):
        raise TypeError("Qwen PDD requires text embeddings and their attention mask.")
    if attention_kwargs:
        raise ValueError("Qwen PDD does not support nonempty attention_kwargs.")
    if guidance is not None:
        raise ValueError("Qwen PDD does not support transformer guidance embeddings.")
    if controlnet_block_samples is not None:
        raise ValueError("Qwen PDD does not support ControlNet block samples.")
    if additional_t_cond is not None:
        raise ValueError("Qwen PDD does not support additional timestep conditioning.")
    del txt_seq_lens
    max_txt_seq_len = encoder_hidden_states.shape[1]
    encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
    temb = self.time_text_embed(timestep, hidden_states)
    image_rotary_emb = self.pos_embed(
        img_shapes,
        max_txt_seq_len=max_txt_seq_len,
        device=hidden_states.device,
    )
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    if _pdd_fusion is None:
        output = self.proj_out(hidden_states)
    else:
        if not isinstance(self.proj_out, PDDOutputProjection):
            raise TypeError("Qwen PDD fusion requires a PDDOutputProjection.")
        output = self.proj_out(hidden_states, fusion=_pdd_fusion)
    if not return_dict:
        return (output,)

    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    return Transformer2DModelOutput(sample=output)


def _is_qwen_image_pdd_forward(model: nn.Module) -> bool:
    forward = model.__dict__.get("forward")
    return (
        isinstance(forward, types.MethodType)
        and forward.__func__ is _qwen_image_pdd_forward
        and forward.__self__ is model
        and getattr(model, _QWEN_IMAGE_PDD_EXECUTION_ATTRIBUTE, None) == QWEN_IMAGE_PDD_EXECUTION
    )


def require_qwen_image_pdd_forward(model: nn.Module) -> str:
    """Require and return the semantic label for the bound Qwen PDD forward."""
    if not isinstance(model, nn.Module) or not _is_qwen_image_pdd_forward(model):
        raise RuntimeError("Qwen-Image PDD requires its masked joint-attention forward.")
    return QWEN_IMAGE_PDD_EXECUTION


def freeze_qwen_image_pdd_unused_parameters(transformer: nn.Module) -> tuple[str, ...]:
    """Freeze final-block text outputs that the Qwen PDD forward does not consume."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise RuntimeError("Qwen PDD requires a nonempty transformer_blocks ModuleList.")

    final_block = blocks[-1]
    local_names = (
        "attn.to_add_out.weight",
        "attn.to_add_out.bias",
        "txt_mlp.net.0.proj.weight",
        "txt_mlp.net.0.proj.bias",
        "txt_mlp.net.2.weight",
        "txt_mlp.net.2.bias",
    )
    frozen_names = []
    for local_name in local_names:
        try:
            parameter = final_block.get_parameter(local_name)
        except AttributeError as error:
            raise RuntimeError(
                f"Qwen PDD final block is missing required parameter {local_name!r}."
            ) from error
        parameter.requires_grad_(False)
        frozen_names.append(f"transformer_blocks.{len(blocks) - 1}.{local_name}")
    return tuple(frozen_names)


def enable_qwen_image_pdd_forward(transformer: nn.Module) -> nn.Module:
    """Bind the masked joint-attention PDD forward to a loaded Qwen transformer."""
    if not isinstance(transformer, nn.Module):
        raise TypeError(f"transformer must be nn.Module, got {type(transformer).__name__}.")
    if _is_qwen_image_pdd_forward(transformer):
        return transformer

    # Diffusers is an optional dependency used only by the Qwen example.
    from diffusers import QwenImageTransformer2DModel

    if not isinstance(transformer, QwenImageTransformer2DModel):
        raise TypeError(
            "Qwen PDD forward binding requires QwenImageTransformer2DModel, "
            f"got {type(transformer).__name__}."
        )
    existing_forward = transformer.__dict__.get("forward")
    if existing_forward is not None:
        raise RuntimeError("Qwen root already has a different instance-level forward override.")
    if _config_guidance_embeds(transformer):
        raise ValueError("Qwen PDD does not support transformer guidance embeddings.")
    if getattr(transformer, "peft_config", None):
        raise ValueError("Qwen PDD does not support active PEFT adapters.")
    if any(getattr(module, "fused_projections", False) for module in transformer.modules()):
        raise ValueError("Qwen PDD does not support fused QKV projections.")
    for name in ("zero_cond_t", "use_additional_t_cond", "use_layer3d_rope"):
        if bool(_config_value(transformer, name, False)):
            raise ValueError(f"Qwen PDD requires {name}=False.")

    transformer.forward = types.MethodType(_qwen_image_pdd_forward, transformer)
    setattr(transformer, _QWEN_IMAGE_PDD_EXECUTION_ATTRIBUTE, QWEN_IMAGE_PDD_EXECUTION)
    require_qwen_image_pdd_forward(transformer)
    return transformer


def _validate_qwen_pdd_config(config: PDDConfig) -> None:
    if not isinstance(config, PDDConfig):
        raise TypeError(f"config must be PDDConfig, got {type(config).__name__}.")


def _output_projection(transformer: nn.Module) -> nn.Linear:
    projection = getattr(transformer, "proj_out", None)
    if not isinstance(projection, nn.Linear):
        raise TypeError(
            "Qwen-Image transformer must register an nn.Linear at 'proj_out', got "
            f"{type(projection).__name__}."
        )
    return projection


def _require_pdd_projection(
    model: nn.Module,
    grid_size: int,
    base_out_features: int,
    *,
    fused: bool = False,
) -> nn.Linear:
    projection = _output_projection(model)
    if projection.out_features != grid_size * base_out_features:
        raise ValueError(
            f"Qwen PDD proj_out has {projection.out_features} outputs; expected "
            f"{grid_size * base_out_features} ({grid_size} heads x {base_out_features})."
        )
    if isinstance(projection, PDDOutputProjection):
        if projection.grid_size != grid_size or projection.layer_spec != QWEN_IMAGE_PDD_LAYER_SPEC:
            raise ValueError("Qwen PDD projection metadata does not match its configuration.")
    elif fused:
        raise TypeError("Qwen fused PDD inference requires a PDDOutputProjection.")
    return projection


def convert_qwen_image_to_pdd(
    transformer: nn.Module,
    config: PDDConfig,
) -> PDDOutputProjection:
    """Replace a loaded, unwrapped Qwen transformer's ``proj_out`` for PDD.

    The full metadata path remains ``transformer.proj_out`` even though this
    helper receives the active transformer component directly. Callers must run
    conversion before device/distributed wrappers and optimizer construction.
    """
    _validate_qwen_pdd_config(config)
    if not isinstance(transformer, nn.Module):
        raise TypeError(f"transformer must be nn.Module, got {type(transformer).__name__}.")
    if _config_guidance_embeds(transformer):
        raise ValueError("Qwen-Image PDD does not support transformer guidance embeddings.")
    current = _output_projection(transformer)

    projection = PDDOutputProjection.from_linear(
        current,
        config.grid_size,
        QWEN_IMAGE_PDD_LAYER_SPEC,
    )
    if projection is not current:
        transformer.proj_out = projection
    return projection


def restore_qwen_image_pdd_projection(
    transformer: nn.Module,
    config: PDDConfig,
) -> PDDOutputProjection:
    """Restore PDD fusion behavior on a serialized, already-widened Qwen projection."""
    _validate_qwen_pdd_config(config)
    if not isinstance(transformer, nn.Module):
        raise TypeError(f"transformer must be nn.Module, got {type(transformer).__name__}.")
    if _config_guidance_embeds(transformer):
        raise ValueError("Qwen-Image PDD does not support transformer guidance embeddings.")
    current = _output_projection(transformer)
    if isinstance(current, PDDOutputProjection):
        return PDDOutputProjection.from_linear(
            current,
            config.grid_size,
            QWEN_IMAGE_PDD_LAYER_SPEC,
        )

    base_out_features = _config_value(transformer, "in_channels")
    if type(base_out_features) is not int or base_out_features <= 0:
        raise ValueError("Qwen-Image transformer config must define positive in_channels.")
    expected_out_features = config.grid_size * base_out_features
    if current.out_features != expected_out_features:
        raise ValueError(
            "Serialized Qwen PDD proj_out has the wrong width: expected "
            f"{expected_out_features}, got {current.out_features}."
        )

    projection = PDDOutputProjection(
        current.in_features,
        base_out_features,
        config.grid_size,
        QWEN_IMAGE_PDD_LAYER_SPEC,
        bias=current.bias is not None,
        device="meta",
        dtype=current.weight.dtype,
    )
    projection.weight = current.weight
    projection.bias = current.bias
    projection.train(current.training)
    transformer.proj_out = projection
    return projection


class QwenImagePDDAdapter:
    """Adapt raw Qwen packed-token calls to the framework-neutral PDD protocol."""

    def __init__(
        self,
        config: PDDConfig,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """Validate the fixed Qwen continuous-time and packed-CFG contract."""
        _validate_qwen_pdd_config(config)
        if config.guidance_scale is not None and not math.isfinite(config.guidance_scale):
            raise ValueError("guidance_scale must be finite when Qwen teacher CFG is enabled.")
        if compute_dtype is not None and (
            not isinstance(compute_dtype, torch.dtype) or not compute_dtype.is_floating_point
        ):
            raise TypeError("compute_dtype must be a real floating-point torch dtype or None.")

        self.config = config
        self.guidance_scale = (
            None if config.guidance_scale is None else float(config.guidance_scale)
        )
        self.compute_dtype = compute_dtype

    @staticmethod
    def _parse_condition(
        condition: Any,
        *,
        state: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(condition, tuple) or len(condition) != 2:
            raise TypeError(f"{name} must be a tuple of (encoder_hidden_states, attention_mask).")
        encoder_hidden_states, attention_mask = condition
        if not isinstance(encoder_hidden_states, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError(f"{name} entries must be tensors.")
        if encoder_hidden_states.ndim != 3 or attention_mask.ndim != 2:
            raise ValueError(
                f"{name} requires batched embeddings and a 2D mask, got "
                f"{tuple(encoder_hidden_states.shape)} and {tuple(attention_mask.shape)}."
            )
        if not encoder_hidden_states.dtype.is_floating_point:
            raise TypeError(f"{name} embeddings must use a real floating-point dtype.")
        if attention_mask.dtype.is_floating_point or attention_mask.dtype.is_complex:
            raise TypeError(f"{name} mask must use an integer or bool dtype.")
        if attention_mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError(f"{name} mask shape must match the embedding batch and sequence axes.")
        if encoder_hidden_states.shape[0] != state.shape[0]:
            raise ValueError(f"{name} batch size must match state batch size {state.shape[0]}.")
        if encoder_hidden_states.device != state.device or attention_mask.device != state.device:
            raise ValueError(f"{name} tensors must be on {state.device}.")
        return encoder_hidden_states, attention_mask

    def _model_dtype(self, model: nn.Module, fallback: torch.dtype) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        for parameter in model.parameters():
            if parameter.dtype.is_floating_point:
                return parameter.dtype
        return fallback

    @staticmethod
    def _extract_packed_output(output: Any) -> torch.Tensor:
        if (
            not isinstance(output, tuple)
            or len(output) != 1
            or not isinstance(output[0], torch.Tensor)
        ):
            raise TypeError("Qwen-Image PDD requires a one-tensor tuple from return_dict=False.")
        packed = output[0]
        if packed.ndim != 3:
            raise ValueError(
                f"Qwen-Image model output must be packed [B, P, F], got {tuple(packed.shape)}."
            )
        return packed

    def _prepare_model_call(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        model_kwargs: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.dtype, list[list[tuple[int, int, int]]]]:
        require_qwen_image_pdd_forward(model)
        if _config_guidance_embeds(model):
            raise ValueError("Qwen-Image PDD does not support transformer guidance embeddings.")
        conflicts = sorted(_CONTROLLED_MODEL_KWARGS.intersection(model_kwargs))
        if conflicts:
            raise ValueError(f"Qwen-Image PDD model_kwargs contains controlled keys: {conflicts}.")
        model_dtype = self._model_dtype(model, state.dtype)
        if model_dtype != torch.bfloat16:
            raise TypeError("Qwen PDD execution requires BF16 compute.")
        if time.dtype != torch.float32:
            raise TypeError("Qwen PDD execution requires FP32 time.")
        packed_state = pack_latents(state).to(model_dtype)
        if time.shape != (packed_state.shape[0],) or time.device != packed_state.device:
            raise ValueError("Qwen PDD time must have shape [batch] on the state device.")
        return (
            packed_state,
            model_dtype,
            build_img_shapes(state.shape[0], state.shape[2], state.shape[3]),
        )

    def _call_packed(
        self,
        model: nn.Module,
        packed_state: torch.Tensor,
        time: torch.Tensor,
        model_kwargs: Mapping[str, Any],
        *,
        prepared_condition: tuple[torch.Tensor, torch.Tensor],
        model_dtype: torch.dtype,
        img_shapes: list[list[tuple[int, int, int]]],
        fusion: tuple[int, int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states, attention_mask = prepared_condition

        encoder_hidden_states = encoder_hidden_states.to(model_dtype)
        call_kwargs = dict(model_kwargs)
        if fusion is not None:
            call_kwargs["_pdd_fusion"] = fusion
        output = model(
            hidden_states=packed_state,
            timestep=time,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=attention_mask,
            img_shapes=img_shapes,
            return_dict=False,
            **call_kwargs,
        )
        return self._extract_packed_output(output)

    @staticmethod
    def _expected_packed_shape(state: torch.Tensor, *, output_features: int) -> torch.Size:
        return torch.Size(
            (
                state.shape[0],
                (state.shape[2] // 2) * (state.shape[3] // 2),
                output_features,
            )
        )

    def _unpack_all_heads(self, packed: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = state.shape
        base_packed_features = channels * 4
        expected = self._expected_packed_shape(
            state,
            output_features=self.config.grid_size * base_packed_features,
        )
        if packed.shape != expected:
            raise ValueError(
                f"unfused Qwen PDD output must have shape {tuple(expected)}, "
                f"got {tuple(packed.shape)}."
            )
        by_head = packed.reshape(
            batch_size,
            packed.shape[1],
            self.config.grid_size,
            base_packed_features,
        ).permute(0, 2, 1, 3)
        flat = by_head.reshape(
            batch_size * self.config.grid_size,
            packed.shape[1],
            base_packed_features,
        )
        unpacked = unpack_latents(flat, height, width)
        return unpacked.reshape(batch_size, self.config.grid_size, channels, height, width)

    def _unpack_single(self, packed: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        expected = self._expected_packed_shape(state, output_features=state.shape[1] * 4)
        if packed.shape != expected:
            raise ValueError(
                f"fused/base Qwen output must have shape {tuple(expected)}, "
                f"got {tuple(packed.shape)}."
            )
        return unpack_latents(packed, state.shape[2], state.shape[3])

    def student_all_heads(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return unpacked PDD interval velocities from one Qwen call."""
        _require_pdd_projection(
            model,
            self.config.grid_size,
            base_out_features=state.shape[1] * 4,
        )
        packed_state, model_dtype, img_shapes = self._prepare_model_call(
            model, state, time, model_kwargs
        )
        prepared_condition = self._parse_condition(condition, state=state, name="condition")
        packed = self._call_packed(
            model,
            packed_state,
            time,
            model_kwargs,
            prepared_condition=prepared_condition,
            model_dtype=model_dtype,
            img_shapes=img_shapes,
        )
        return self._unpack_all_heads(packed, state)

    def student_fused_block(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        start: int,
        end: int,
        grid: torch.Tensor,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Run one conditional Qwen call with its final projection fused for a block."""
        projection = _output_projection(model)
        base_out_features = (
            projection.base_out_features
            if isinstance(projection, PDDOutputProjection)
            else state.shape[1] * 4
        )
        _require_pdd_projection(
            model,
            self.config.grid_size,
            base_out_features,
            fused=True,
        )
        packed_state, model_dtype, img_shapes = self._prepare_model_call(
            model, state, time, model_kwargs
        )
        prepared_condition = self._parse_condition(condition, state=state, name="condition")
        packed = self._call_packed(
            model,
            packed_state,
            time,
            model_kwargs,
            prepared_condition=prepared_condition,
            model_dtype=model_dtype,
            img_shapes=img_shapes,
            fusion=(start, end, grid),
        )
        return self._unpack_single(packed, state)

    @torch.no_grad()
    def teacher_velocity(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        negative_condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return conditional or fixed two-pass packed-CFG Qwen teacher velocity."""
        packed_state, model_dtype, img_shapes = self._prepare_model_call(
            model, state, time, model_kwargs
        )
        prepared_condition = self._parse_condition(condition, state=state, name="condition")
        guidance_scale = self.guidance_scale
        prepared_negative_condition = prepared_condition
        if guidance_scale is not None:
            prepared_negative_condition = self._parse_condition(
                negative_condition,
                state=state,
                name="negative_condition",
            )

        conditional = self._call_packed(
            model,
            packed_state,
            time,
            model_kwargs,
            prepared_condition=prepared_condition,
            model_dtype=model_dtype,
            img_shapes=img_shapes,
        )
        if guidance_scale is None:
            return self._unpack_single(conditional, state)

        unconditional = self._call_packed(
            model,
            packed_state,
            time,
            model_kwargs,
            prepared_condition=prepared_negative_condition,
            model_dtype=model_dtype,
            img_shapes=img_shapes,
        )
        expected = self._expected_packed_shape(state, output_features=state.shape[1] * 4)
        if conditional.shape != expected or unconditional.shape != expected:
            raise ValueError(
                f"Qwen teacher outputs must both have shape {tuple(expected)}, got "
                f"{tuple(conditional.shape)} and {tuple(unconditional.shape)}."
            )

        # Qwen-Image applies its native CFG rescale independently to every
        # packed image token before unpacking the velocity.
        guided_low_precision = conditional + (float(guidance_scale) - 1.0) * (
            conditional - unconditional
        )
        conditional_fp32 = conditional.to(torch.float32)
        guided_fp32 = guided_low_precision.to(torch.float32)
        conditional_norm = torch.linalg.vector_norm(
            conditional_fp32,
            dim=-1,
            keepdim=True,
        )
        guided_norm = torch.linalg.vector_norm(
            guided_fp32,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-5)
        guided = (guided_fp32 * (conditional_norm / guided_norm)).to(conditional.dtype)
        return self._unpack_single(guided, state)
