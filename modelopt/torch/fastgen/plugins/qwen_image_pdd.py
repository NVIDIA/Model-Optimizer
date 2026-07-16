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

"""Qwen-Image adapter and explicit output-projection conversion for PDD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..config import PDDConfig
from ..methods.pdd import PDDLayerSpec, PDDOutputProjection
from .qwen_image import build_img_shapes, pack_latents, unpack_latents

__all__ = [
    "QWEN_IMAGE_PDD_LAYER_SPEC",
    "QwenImagePDDAdapter",
    "convert_qwen_image_to_pdd",
]

QWEN_IMAGE_PDD_LAYER_SPEC = PDDLayerSpec(
    projection_path="transformer.proj_out",
    head_layout="channel_major",
)

_CONTROLLED_MODEL_KWARGS = {
    "encoder_hidden_states",
    "encoder_hidden_states_mask",
    "guidance",
    "hidden_states",
    "img_shapes",
    "return_dict",
    "timestep",
    "txt_seq_lens",
}


def _config_guidance_embeds(transformer: nn.Module) -> bool:
    config = getattr(transformer, "config", None)
    if isinstance(config, Mapping):
        return bool(config.get("guidance_embeds", False))
    return bool(getattr(config, "guidance_embeds", False))


def _validate_qwen_pdd_config(config: PDDConfig) -> None:
    if not isinstance(config, PDDConfig):
        raise TypeError(f"config must be PDDConfig, got {type(config).__name__}.")
    if config.num_train_timesteps is not None:
        raise ValueError(
            "Qwen-Image PDD requires num_train_timesteps=None because the adapter "
            "forwards normalized continuous grid time."
        )


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
    try:
        current = transformer.get_submodule("proj_out")
    except AttributeError as error:
        raise ValueError(
            "Qwen-Image transformer must register an nn.Linear at 'proj_out'."
        ) from error
    if not isinstance(current, nn.Linear):
        raise TypeError(f"Qwen-Image proj_out must be nn.Linear, got {type(current).__name__}.")

    projection = PDDOutputProjection.from_linear(
        current,
        config.grid_size,
        QWEN_IMAGE_PDD_LAYER_SPEC,
    )
    if projection is not current:
        transformer.proj_out = projection
        if transformer.get_submodule("proj_out") is not projection:
            raise RuntimeError("Qwen-Image proj_out replacement did not remain registered.")
    return projection


class QwenImagePDDAdapter:
    """Adapt raw Qwen packed-token calls to the framework-neutral PDD protocol."""

    def __init__(
        self,
        config: PDDConfig,
        *,
        guidance_rescale: float = 1.0,
        guidance_eps: float = 1e-5,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """Validate the fixed Qwen continuous-time and packed-CFG contract."""
        _validate_qwen_pdd_config(config)
        if isinstance(guidance_rescale, bool) or not isinstance(guidance_rescale, int | float):
            raise TypeError("guidance_rescale must be a real number.")
        if not math.isfinite(guidance_rescale) or not 0.0 <= guidance_rescale <= 1.0:
            raise ValueError("guidance_rescale must be finite and in [0, 1].")
        if isinstance(guidance_eps, bool) or not isinstance(guidance_eps, int | float):
            raise TypeError("guidance_eps must be a real number.")
        if not math.isfinite(guidance_eps) or guidance_eps <= 0.0:
            raise ValueError("guidance_eps must be finite and > 0.")
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
        self.guidance_rescale = float(guidance_rescale)
        self.guidance_eps = float(guidance_eps)
        self.compute_dtype = compute_dtype

    @staticmethod
    def _validate_state_and_time(state: torch.Tensor, time: torch.Tensor) -> None:
        if state.ndim != 4:
            raise ValueError(
                f"Qwen-Image PDD state must have shape [B, C, H, W], got {tuple(state.shape)}."
            )
        if state.shape[2] % 2 or state.shape[3] % 2:
            raise ValueError("Qwen-Image PDD requires even latent height and width.")
        if time.shape != (state.shape[0],):
            raise ValueError(
                f"Qwen-Image PDD time must have shape ({state.shape[0]},), got {tuple(time.shape)}."
            )
        if time.device != state.device:
            raise ValueError(f"time must be on {state.device}, got {time.device}.")
        if not time.dtype.is_floating_point:
            raise TypeError(f"time must use a real floating-point dtype, got {time.dtype}.")

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
        if encoder_hidden_states.ndim < 2 or attention_mask.ndim != 2:
            raise ValueError(
                f"{name} requires batched embeddings and a 2D mask, got "
                f"{tuple(encoder_hidden_states.shape)} and {tuple(attention_mask.shape)}."
            )
        if not encoder_hidden_states.dtype.is_floating_point:
            raise TypeError(f"{name} embeddings must use a real floating-point dtype.")
        if (
            attention_mask.dtype.is_floating_point
            or attention_mask.dtype.is_complex
            or attention_mask.shape[1] != encoder_hidden_states.shape[1]
        ):
            raise ValueError(
                f"{name} mask must be an integer/bool tensor matching the embedding "
                "sequence length."
            )
        batch_size = state.shape[0]
        if encoder_hidden_states.shape[0] != batch_size or attention_mask.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match state batch size {batch_size}.")
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
        if isinstance(output, tuple):
            if not output:
                raise TypeError("Qwen-Image model returned an empty tuple.")
            packed = output[0]
        elif isinstance(output, torch.Tensor):
            packed = output
        elif hasattr(output, "sample"):
            packed = output.sample
        else:
            raise TypeError(
                "Qwen-Image PDD could not extract a tensor from model output of type "
                f"{type(output).__name__}."
            )
        if not isinstance(packed, torch.Tensor):
            raise TypeError("Qwen-Image model output payload must be a tensor.")
        if packed.ndim != 3:
            raise ValueError(
                f"Qwen-Image model output must be packed [B, P, F], got {tuple(packed.shape)}."
            )
        return packed

    def _prepare_call(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: Any,
        model_kwargs: Mapping[str, Any],
        *,
        condition_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_state_and_time(state, time)
        if _config_guidance_embeds(model):
            raise ValueError("Qwen-Image PDD does not support transformer guidance embeddings.")
        encoder_hidden_states, attention_mask = self._parse_condition(
            condition,
            state=state,
            name=condition_name,
        )
        conflicts = sorted(_CONTROLLED_MODEL_KWARGS.intersection(model_kwargs))
        if conflicts:
            raise ValueError(f"Qwen-Image PDD model_kwargs contains controlled keys: {conflicts}.")
        return encoder_hidden_states, attention_mask

    def _call_packed(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: Any,
        model_kwargs: Mapping[str, Any],
        *,
        condition_name: str,
    ) -> torch.Tensor:
        encoder_hidden_states, attention_mask = self._prepare_call(
            model,
            state,
            time,
            condition,
            model_kwargs,
            condition_name=condition_name,
        )

        batch_size, _, height, width = state.shape
        packed_state = pack_latents(state).to(self._model_dtype(model, state.dtype))
        output = model(
            hidden_states=packed_state,
            timestep=time,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=attention_mask,
            img_shapes=build_img_shapes(batch_size, height, width),
            guidance=None,
            return_dict=False,
            **model_kwargs,
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

    @staticmethod
    def _projection(model: nn.Module, grid_size: int) -> PDDOutputProjection:
        try:
            projection = model.get_submodule("proj_out")
        except AttributeError as error:
            raise ValueError(
                "Qwen student must register a PDD projection at 'proj_out'."
            ) from error
        if not isinstance(projection, PDDOutputProjection):
            raise TypeError(
                "Qwen student proj_out must be converted to PDDOutputProjection before use."
            )
        if projection.grid_size != grid_size:
            raise ValueError(
                f"Qwen PDD projection grid_size={projection.grid_size} does not match "
                f"config grid_size={grid_size}."
            )
        if projection.layer_spec != QWEN_IMAGE_PDD_LAYER_SPEC:
            raise ValueError("Qwen PDD projection carries an incompatible layer specification.")
        return projection

    def student_all_heads(
        self,
        model: nn.Module,
        state: torch.Tensor,
        time: torch.Tensor,
        *,
        condition: Any = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Return unpacked canonical interval velocities from one Qwen call."""
        self._projection(model, self.config.grid_size)
        packed = self._call_packed(
            model,
            state,
            time,
            condition,
            model_kwargs,
            condition_name="condition",
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
        projection = self._projection(model, self.config.grid_size)
        with projection.fuse_block(start, end, grid):
            packed = self._call_packed(
                model,
                state,
                time,
                condition,
                model_kwargs,
                condition_name="condition",
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
        guidance_scale = self.guidance_scale
        if guidance_scale is not None and negative_condition is None:
            raise ValueError("negative_condition is required when Qwen teacher CFG is enabled.")
        if guidance_scale is not None:
            # Validate both collective-participating calls before either model
            # call so malformed rank-local conditioning cannot split call counts.
            self._prepare_call(
                model,
                state,
                time,
                condition,
                model_kwargs,
                condition_name="condition",
            )
            self._prepare_call(
                model,
                state,
                time,
                negative_condition,
                model_kwargs,
                condition_name="negative_condition",
            )

        conditional = self._call_packed(
            model,
            state,
            time,
            condition,
            model_kwargs,
            condition_name="condition",
        )
        if guidance_scale is None:
            return self._unpack_single(conditional, state)

        unconditional = self._call_packed(
            model,
            state,
            time,
            negative_condition,
            model_kwargs,
            condition_name="negative_condition",
        )
        expected = self._expected_packed_shape(state, output_features=state.shape[1] * 4)
        if conditional.shape != expected or unconditional.shape != expected:
            raise ValueError(
                f"Qwen teacher outputs must both have shape {tuple(expected)}, got "
                f"{tuple(conditional.shape)} and {tuple(unconditional.shape)}."
            )

        # FastGen applies CFG in the model-output dtype, including its BF16
        # rounding, before cfg_rescale promotes the result for norm math.
        guided_model_dtype = conditional + (float(guidance_scale) - 1.0) * (
            conditional - unconditional
        )
        conditional_fp32 = conditional.to(torch.float32)
        guided = guided_model_dtype.to(torch.float32)
        # MR210 applies CFG after unpacking Qwen output to NCHW and leaves
        # ``rescale_dims`` unset, so the norm spans every non-batch element.
        # Reducing packed [P, F] here is algebraically identical because
        # pack/unpack only reshapes and permutes those elements.
        norm_dims = tuple(range(1, conditional_fp32.ndim))
        conditional_norm = torch.linalg.vector_norm(
            conditional_fp32,
            dim=norm_dims,
            keepdim=True,
        )
        guided_norm = torch.linalg.vector_norm(guided, dim=norm_dims, keepdim=True)
        factor = self.guidance_rescale * conditional_norm / guided_norm.clamp_min(
            self.guidance_eps
        ) + (1.0 - self.guidance_rescale)
        # FastGen's cfg_rescale returns to the conditional model-output dtype
        # before PDD promotes the teacher target for FP32 loss math.
        return self._unpack_single((guided * factor).to(conditional.dtype), state)
