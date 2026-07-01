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

"""Qwen-Image-Edit plumbing for DMD2.

``QwenImageEditPlusPipeline`` conditions the transformer in two complementary ways:

* the Qwen2.5-VL prompt embedding contains the edit instruction and visual context; and
* one or more clean VAE reference-image latents are packed and appended after the noisy
  target-image tokens.

Only the target image is diffused. DMD2 therefore keeps its external latent contract as
``[B, C, H, W]`` and forwards reference latents through the model kwargs under
``conditioning_latents``. This plugin packs ``[target, reference_1, ...]`` for every model
call, constructs the matching ``img_shapes``, and discards the reference-token suffix from
the model prediction before returning to the shared DMD math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .qwen_image import (
    QwenImageDMDPipeline,
    pack_latents,
    unpack_latents,
    update_feature_capture_shape,
)
from .qwen_image import attach_feature_capture as _attach_qwen_feature_capture
from .qwen_image import remove_feature_capture as _remove_qwen_feature_capture

if TYPE_CHECKING:
    from ..config import DMDConfig

__all__ = [
    "QwenImageEditDMDPipeline",
]


class QwenImageEditDMDPipeline(QwenImageDMDPipeline):
    """DMD2 pipeline for Qwen-Image-Edit's target-plus-reference token layout.

    ``conditioning_latents`` must be supplied to each ``compute_*_loss`` call as a
    non-empty list or tuple. Each entry is one clean reference image with shape
    ``[B, C, H_ref, W_ref]``. Reference images may have different spatial shapes from
    each other and from the target, but their batch/channel/device/dtype must match the
    target latent. The reference order must match the order used when constructing the
    multimodal prompt embedding.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        config: DMDConfig,
        *,
        discriminator: nn.Module | None = None,
        guidance: float | None = None,
    ) -> None:
        """Initialize the shared Qwen pipeline and retain its timestep/guidance checks."""
        super().__init__(
            student=student,
            teacher=teacher,
            fake_score=fake_score,
            config=config,
            discriminator=discriminator,
            guidance=guidance,
        )

    @staticmethod
    def _validate_conditioning_latents(
        conditioning_latents: Any,
        target: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Validate and normalize the reference-latent sequence for one model call."""
        if not isinstance(conditioning_latents, (list, tuple)) or not conditioning_latents:
            raise ValueError(
                "QwenImageEditDMDPipeline requires non-empty `conditioning_latents` as a "
                "list or tuple of [B, C, H, W] tensors."
            )

        b, c, _h, _w = target.shape
        normalized: list[torch.Tensor] = []
        for index, latent in enumerate(conditioning_latents):
            if not torch.is_tensor(latent):
                raise TypeError(
                    f"conditioning_latents[{index}] must be a Tensor, got {type(latent).__name__}."
                )
            if latent.ndim != 4:
                raise ValueError(
                    f"conditioning_latents[{index}] must have shape [B, C, H, W], got "
                    f"{latent.ndim}D tensor with shape {tuple(latent.shape)}."
                )
            if latent.shape[0] != b or latent.shape[1] != c:
                raise ValueError(
                    f"conditioning_latents[{index}] batch/channels {tuple(latent.shape[:2])} "
                    f"must match target {(b, c)}."
                )
            if latent.shape[2] % 2 or latent.shape[3] % 2:
                raise ValueError(
                    f"conditioning_latents[{index}] requires even spatial dims for Qwen "
                    f"packing, got H={latent.shape[2]}, W={latent.shape[3]}."
                )
            if latent.device != target.device:
                raise ValueError(
                    f"conditioning_latents[{index}] is on {latent.device}, but target is on "
                    f"{target.device}."
                )
            if latent.dtype != target.dtype:
                raise ValueError(
                    f"conditioning_latents[{index}] has dtype {latent.dtype}, but target has "
                    f"dtype {target.dtype}."
                )
            normalized.append(latent)
        return normalized

    def _call_model(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Pack target + references, call the transformer, and return target prediction only."""
        if hidden_states.ndim != 4:
            raise ValueError(
                "QwenImageEditDMDPipeline._call_model expects 4D hidden_states "
                f"[B, C, H, W] (got {hidden_states.ndim}D)."
            )
        b, _c, h, w = hidden_states.shape

        call_kwargs: dict[str, Any] = dict(model_kwargs)
        conditioning_latents = self._validate_conditioning_latents(
            call_kwargs.pop("conditioning_latents", None), hidden_states
        )

        target_packed = pack_latents(hidden_states)
        update_feature_capture_shape(model, h, w)
        conditioning_packed = [pack_latents(latent) for latent in conditioning_latents]
        packed = torch.cat([target_packed, *conditioning_packed], dim=1)
        target_num_patches = target_packed.shape[1]

        per_sample_shapes = [(1, h // 2, w // 2)] + [
            (1, latent.shape[2] // 2, latent.shape[3] // 2) for latent in conditioning_latents
        ]
        img_shapes = [list(per_sample_shapes) for _ in range(b)]

        # These values are owned by this wrapper. Drop caller copies so duplicate kwargs
        # cannot leak through to the diffusers transformer.
        call_kwargs.pop("hidden_states", None)
        encoder_hidden_states_mask = call_kwargs.pop("encoder_hidden_states_mask", None)
        call_kwargs.pop("img_shapes", None)
        call_kwargs.pop("guidance", None)
        call_kwargs.pop("return_dict", None)
        # Stable Diffusers derives text lengths from encoder_hidden_states_mask.
        call_kwargs.pop("txt_seq_lens", None)

        guidance = None
        if self._guidance_value is not None:
            guidance = torch.full(
                (b,),
                float(self._guidance_value),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        out = model(
            hidden_states=packed,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=img_shapes,
            guidance=guidance,
            return_dict=False,
            **call_kwargs,
        )

        if isinstance(out, tuple):
            raw_packed = out[0]
        elif isinstance(out, torch.Tensor):
            raw_packed = out
        elif hasattr(out, "sample"):
            raw_packed = out.sample
        else:
            raise TypeError(
                "QwenImageEditDMDPipeline._call_model could not extract a tensor from "
                f"output of type {type(out).__name__!r}."
            )

        if raw_packed.ndim != 3:
            raise ValueError(
                "QwenImageEditDMDPipeline expected packed model output [B, tokens, C*4], "
                f"got shape {tuple(raw_packed.shape)}."
            )
        if raw_packed.shape[0] != b:
            raise ValueError(
                f"Packed model output batch {raw_packed.shape[0]} does not match target batch {b}."
            )
        if raw_packed.shape[1] < target_num_patches:
            raise ValueError(
                f"Packed model output has {raw_packed.shape[1]} tokens but the target prefix "
                f"requires {target_num_patches}."
            )

        # QwenImageEditPlusPipeline treats only the leading target tokens as the denoising
        # prediction. The appended reference-token outputs are conditioning-only.
        target_prediction = raw_packed[:, :target_num_patches]
        return unpack_latents(target_prediction, h, w)


def attach_feature_capture(
    teacher: nn.Module,
    feature_indices: list[int],
    h_lat: int,
    w_lat: int,
    *,
    blocks_attr: str = "transformer_blocks",
) -> None:
    """Capture only the target-token prefix from Qwen-Image-Edit teacher blocks."""
    _attach_qwen_feature_capture(
        teacher,
        feature_indices,
        h_lat,
        w_lat,
        blocks_attr=blocks_attr,
        target_prefix_only=True,
    )


def remove_feature_capture(teacher: nn.Module) -> None:
    """Remove feature hooks installed through :func:`attach_feature_capture`."""
    _remove_qwen_feature_capture(teacher)
