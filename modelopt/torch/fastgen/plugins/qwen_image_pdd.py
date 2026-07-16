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

import hashlib
import inspect
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..config import PDDConfig
from ..methods.pdd import PDDLayerSpec, PDDOutputProjection
from .qwen_image import build_img_shapes, pack_latents, unpack_latents

__all__ = [
    "QWEN_IMAGE_PDD_FORWARD_SUBSTRATE",
    "QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID",
    "QWEN_IMAGE_PDD_LAYER_SPEC",
    "QwenImagePDDAdapter",
    "adopt_qwen_image_mr210_forward",
    "convert_qwen_image_to_pdd",
    "require_qwen_image_pdd_forward_substrate",
]

QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID = (
    "pdd_qwen_mr210_c8100b1347b278511336dccfc074a461457216ec_"
    "qwen_33706683487ba16d133b99b73be27b21164c53335441d77b1dcabbfca970f70e"
)
QWEN_IMAGE_PDD_FORWARD_SUBSTRATE = {
    "id": QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID,
    "fastgen_commit": "c8100b1347b278511336dccfc074a461457216ec",
    "fastgen_qwen_source_sha256": (
        "33706683487ba16d133b99b73be27b21164c53335441d77b1dcabbfca970f70e"
    ),
    "diffusers_version": "0.38.0",
    "diffusers_qwen_source_sha256": (
        "34c864b0b066a4a9eb84e40e1bb77b7df303c165e7910600b402a0f5f8d8f94e"
    ),
    "diffusers_embeddings_source_sha256": (
        "d7a90ef799569e3f0fab41cadde1ecba023abd053af956c022bbfc097662a302"
    ),
}

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
}

_QWEN_IMAGE_ROOT_CHILDREN = (
    "pos_embed",
    "time_text_embed",
    "txt_norm",
    "img_in",
    "txt_in",
    "transformer_blocks",
    "norm_out",
    "proj_out",
)
_ADOPTED_QWEN_TYPES: dict[type[nn.Module], type[nn.Module]] = {}


def require_qwen_image_pdd_forward_substrate(value: Any) -> dict[str, str]:
    """Return the authenticated MR210 Qwen substrate or reject it exactly."""
    if not isinstance(value, Mapping) or dict(value) != QWEN_IMAGE_PDD_FORWARD_SUBSTRATE:
        raise ValueError(
            "Qwen-Image PDD requires the authenticated MR210 forward substrate "
            f"{QWEN_IMAGE_PDD_FORWARD_SUBSTRATE_ID!r}."
        )
    return dict(QWEN_IMAGE_PDD_FORWARD_SUBSTRATE)


def _sha256_source(owner: type[Any]) -> str:
    source = inspect.getsourcefile(owner)
    if source is None:
        raise RuntimeError(f"cannot locate source for {owner.__module__}.{owner.__qualname__}.")
    with open(source, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_qwen_source_identity(transformer: nn.Module) -> None:
    transformer_type = type(transformer)
    if (
        transformer_type.__module__ != "diffusers.models.transformers.transformer_qwenimage"
        or transformer_type.__name__ != "QwenImageTransformer2DModel"
    ):
        raise TypeError("MR210 adoption requires the pinned Diffusers QwenImageTransformer2DModel.")
    if (
        _sha256_source(transformer_type)
        != QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_qwen_source_sha256"]
    ):
        raise RuntimeError("Diffusers Qwen transformer source does not match the MR210 substrate.")
    embedding_types = (
        type(transformer.time_text_embed.time_proj),
        type(transformer.time_text_embed.timestep_embedder),
    )
    if any(
        _sha256_source(embedding_type)
        != QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_embeddings_source_sha256"]
        for embedding_type in embedding_types
    ):
        raise RuntimeError(
            "Diffusers timestep embedding source does not match the MR210 substrate."
        )
    try:
        import diffusers  # Optional dependency required only by the Qwen adoption path.
    except ImportError as error:  # pragma: no cover - the transformer itself requires Diffusers
        raise RuntimeError("Diffusers is required for Qwen-Image PDD adoption.") from error
    if diffusers.__version__ != QWEN_IMAGE_PDD_FORWARD_SUBSTRATE["diffusers_version"]:
        raise RuntimeError(
            "Diffusers version does not match the authenticated Qwen MR210 substrate."
        )


def _config_value(transformer: nn.Module, name: str, default: Any = None) -> Any:
    config = getattr(transformer, "config", None)
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _require_binary_prefix_mask(
    encoder_hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(encoder_hidden_states.shape[:2]):
        raise ValueError("Qwen MR210 mask must match the text batch and sequence dimensions.")
    if mask.dtype.is_floating_point or mask.dtype.is_complex:
        raise TypeError("Qwen MR210 mask must use an integer or boolean dtype.")
    if mask.device != encoder_hidden_states.device:
        raise ValueError("Qwen MR210 mask and text embeddings must share a device.")
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("Qwen MR210 masks must have nonempty batch and sequence dimensions.")
    mask_int = mask.to(torch.int64)
    mask_bool = mask.bool()
    binary = torch.all((mask_int == 0) | (mask_int == 1))
    prefix = torch.all(mask_int[:, 1:] <= mask_int[:, :-1])
    lengths = mask_int.sum(dim=1)
    valid_lengths = torch.all(lengths > 0) & (lengths.max() == mask.shape[1])
    zero_padding = torch.all(encoder_hidden_states[~mask_bool] == 0)
    if not bool((binary & prefix & valid_lengths & zero_padding).item()):
        raise ValueError(
            "Qwen MR210 requires nonempty binary prefix masks, a longest unpadded row, "
            "and zero padding."
        )


class _QwenImageMR210ForwardMixin:
    """Execute FastGen MR210's Qwen forward without altering Diffusers classes.

    Source contract: ``fastgen/networks/QwenImage/network.py`` at
    ``c8100b1347b278511336dccfc074a461457216ec``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        img_shapes: list[Any] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: Any = None,
        additional_t_cond: Any = None,
        return_dict: bool = True,
        *,
        max_txt_seq_len: int | None = None,
    ) -> Any:
        """Run the source-locked MR210 regular-output path."""
        if hidden_states.ndim != 3 or hidden_states.dtype != torch.bfloat16:
            raise TypeError("Qwen MR210 hidden_states must be packed BF16 [B, P, C].")
        if (
            not isinstance(encoder_hidden_states, torch.Tensor)
            or encoder_hidden_states.ndim != 3
            or encoder_hidden_states.dtype != torch.bfloat16
        ):
            raise TypeError("Qwen MR210 encoder_hidden_states must be BF16 [B, S, D].")
        if not isinstance(encoder_hidden_states_mask, torch.Tensor):
            raise TypeError("Qwen MR210 requires encoder_hidden_states_mask.")
        if not isinstance(timestep, torch.Tensor) or timestep.dtype != torch.float32:
            raise TypeError("Qwen MR210 timestep must remain FP32 at transformer entry.")
        if timestep.shape != (hidden_states.shape[0],):
            raise ValueError("Qwen MR210 timestep must contain one value per batch item.")
        if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
            raise ValueError("Qwen MR210 image and text batch sizes must match.")
        if img_shapes is None or len(img_shapes) != hidden_states.shape[0]:
            raise ValueError("Qwen MR210 img_shapes must contain one entry per batch item.")
        if txt_seq_lens is not None:
            raise ValueError("Qwen MR210 does not support txt_seq_lens.")
        if guidance is not None:
            raise ValueError("Qwen MR210 does not support transformer guidance embeddings.")
        if attention_kwargs:
            raise ValueError("Qwen MR210 does not support nonempty attention_kwargs.")
        if controlnet_block_samples is not None:
            raise ValueError("Qwen MR210 does not support ControlNet residuals.")
        if additional_t_cond is not None:
            raise ValueError("Qwen MR210 does not support additional time conditioning.")
        if type(return_dict) is not bool:
            raise TypeError("return_dict must be bool.")
        _require_binary_prefix_mask(encoder_hidden_states, encoder_hidden_states_mask)
        sequence_length = encoder_hidden_states.shape[1]
        if max_txt_seq_len is not None and max_txt_seq_len != sequence_length:
            raise ValueError("Qwen MR210 max_txt_seq_len must equal the padded text length.")

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        if timestep.dtype != torch.float32:
            raise RuntimeError("Qwen MR210 timestep was rounded before time_text_embed.")
        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(
            img_shapes,
            max_txt_seq_len=sequence_length,
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
                    joint_attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if not return_dict:
            return (output,)
        try:
            from diffusers.models.modeling_outputs import (  # Optional Qwen runtime dependency.
                Transformer2DModelOutput,
            )
        except ImportError as error:  # pragma: no cover - adoption already requires Diffusers
            raise RuntimeError("Diffusers output types are unavailable.") from error
        return Transformer2DModelOutput(sample=output)


def _adopted_qwen_type(base: type[nn.Module]) -> type[nn.Module]:
    adopted = _ADOPTED_QWEN_TYPES.get(base)
    if adopted is None:
        adopted = type(
            f"ModelOptMR210{base.__name__}",
            (_QwenImageMR210ForwardMixin, base),
            {"__module__": __name__},
        )
        _ADOPTED_QWEN_TYPES[base] = adopted
    return adopted


def adopt_qwen_image_mr210_forward(transformer: nn.Module) -> nn.Module:
    """Adopt loaded Qwen children into a Diffusers-compatible MR210 forward root."""
    if isinstance(transformer, _QwenImageMR210ForwardMixin):
        return transformer
    if not isinstance(transformer, nn.Module):
        raise TypeError(f"transformer must be nn.Module, got {type(transformer).__name__}.")
    _require_qwen_source_identity(transformer)
    if tuple(transformer._modules) != _QWEN_IMAGE_ROOT_CHILDREN:
        raise RuntimeError("Qwen root child layout does not match the authenticated substrate.")
    if transformer._parameters or transformer._buffers:
        raise RuntimeError("Qwen root unexpectedly registers direct parameters or buffers.")
    if _config_guidance_embeds(transformer):
        raise ValueError("Qwen MR210 does not support transformer guidance embeddings.")
    if getattr(transformer, "peft_config", None):
        raise ValueError("Qwen MR210 does not support active PEFT adapters.")
    if any(getattr(module, "fused_projections", False) for module in transformer.modules()):
        raise ValueError("Qwen MR210 does not support fused QKV projections.")
    for name in ("zero_cond_t", "use_additional_t_cond", "use_layer3d_rope"):
        if bool(_config_value(transformer, name, False)):
            raise ValueError(f"Qwen MR210 requires {name}=False.")
    hook_names = (
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
    )
    if any(getattr(transformer, name, None) for name in hook_names):
        raise RuntimeError("Qwen root hooks must be empty before MR210 adoption.")

    adopted_type = _adopted_qwen_type(type(transformer))
    adopted = adopted_type.__new__(adopted_type)
    nn.Module.__init__(adopted)
    adopted._internal_dict = transformer._internal_dict
    for name in ("out_channels", "inner_dim", "gradient_checkpointing", "zero_cond_t"):
        setattr(adopted, name, getattr(transformer, name))
    if hasattr(transformer, "_gradient_checkpointing_func"):
        adopted._gradient_checkpointing_func = transformer._gradient_checkpointing_func
    for name, child in transformer._modules.items():
        adopted.add_module(name, child)
    adopted.train(transformer.training)
    if tuple(adopted.state_dict()) != tuple(transformer.state_dict()):
        raise RuntimeError("Qwen state keys changed during MR210 adoption.")
    if any(
        adopted_parameter is not source_parameter
        for adopted_parameter, source_parameter in zip(
            adopted.parameters(), transformer.parameters(), strict=True
        )
    ):
        raise RuntimeError("Qwen parameter identity changed during MR210 adoption.")
    return adopted


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
        if encoder_hidden_states.ndim != 3 or attention_mask.ndim != 2:
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
        _require_binary_prefix_mask(encoder_hidden_states, attention_mask)
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
        model_dtype = self._model_dtype(model, state.dtype)
        if model_dtype != torch.bfloat16:
            raise TypeError("authenticated Qwen MR210 execution requires BF16 compute.")
        if time.dtype != torch.float32:
            raise TypeError("authenticated Qwen MR210 execution requires FP32 time.")
        packed_state = pack_latents(state).to(model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(model_dtype)
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
