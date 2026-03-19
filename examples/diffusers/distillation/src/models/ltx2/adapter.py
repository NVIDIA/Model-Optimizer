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

"""LTX-2 training forward adapter.

Handles patchification, Modality construction, position generation,
first-frame conditioning, and loss computation for the LTX-2 model.

The LTX-2 transformer takes Modality objects:
    model(video=Modality, audio=Modality|None, perturbations=None)
    -> (video_pred: Tensor, audio_pred: Tensor|None)

Predictions and targets are in patchified space [B, seq_len, C].
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import StrategyOutputs

_LTX2_LATENT_CHANNELS = 128


class LTX2TrainingForwardAdapter:
    """Flow matching training forward adapter for LTX-2.

    Stateful: holds the patchifier and scale factor references needed for
    position generation. These are lightweight ltx-core objects.

    Text embeddings in the batch are expected to be post-connector
    (pre-applied during preprocessing or cached by the pipeline).

    Args:
        fps: Frames per second used to scale temporal position coordinates.
        first_frame_conditioning_p: Probability of conditioning on the first
            frame during training. 0.0 disables first-frame conditioning.
    """

    MOCK_LATENT_SHAPE: tuple[int, ...] = (128, 4, 32, 32)
    MOCK_TEXT_EMBED_DIM: int = 3840

    def __init__(self, fps: float = 25.0, first_frame_conditioning_p: float = 0.0) -> None:
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.types import SpatioTemporalScaleFactors

        self._patchifier = VideoLatentPatchifier(patch_size=1)
        self._scale_factors = SpatioTemporalScaleFactors.default()
        self._fps = fps
        self._first_frame_p = first_frame_conditioning_p

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        noise: Tensor,
        timesteps: Tensor,
    ) -> StrategyOutputs:
        latents = batch["latents"]  # [B, C, F, H, W]
        text_embeds = batch["text_embeds"]  # [B, L, D]  (post-connector)
        text_mask = batch.get("text_mask")  # [B, L] or None

        b, _c, n_f, n_h, n_w = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Flow matching: noisy = (1 - sigma) * clean + sigma * noise
        sigmas = timesteps.view(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        targets = noise - latents

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        noisy_patchified = self._patchifier.patchify(noisy_latents)
        targets_patchified = self._patchifier.patchify(targets)
        clean_patchified = self._patchifier.patchify(latents)
        seq_len = noisy_patchified.shape[1]

        # First-frame conditioning mask [B, seq_len]
        conditioning_mask = self._create_conditioning_mask(b, seq_len, n_h, n_w, device)

        # Replace conditioning tokens with clean latents
        if conditioning_mask.any():
            cond_expanded = conditioning_mask.unsqueeze(-1)
            noisy_patchified = torch.where(cond_expanded, clean_patchified, noisy_patchified)

        # Per-token timesteps: 0 for conditioning tokens, sigma for target tokens
        per_token_ts = timesteps.view(-1, 1).expand(-1, seq_len).clone()
        if conditioning_mask.any():
            per_token_ts[conditioning_mask] = 0.0

        # Position embeddings [B, 3, seq_len, 2]
        positions = self._get_positions(n_f, n_h, n_w, b, device, dtype)

        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            latent=noisy_patchified,
            timesteps=per_token_ts,
            positions=positions,
            context=text_embeds,
            context_mask=text_mask,
        )

        # Loss mask: True where we compute loss (non-conditioning tokens)
        loss_mask = ~conditioning_mask

        forward_kwargs = {
            "video": video_modality,
            "audio": None,
            "perturbations": None,
        }

        return StrategyOutputs(
            noisy_input=noisy_patchified,
            targets=targets_patchified,
            loss_mask=loss_mask,
            forward_kwargs=forward_kwargs,
        )

    def forward_model(self, model: nn.Module, inputs: StrategyOutputs) -> Tensor:
        video_pred, _audio_pred = model(**inputs.forward_kwargs)
        return video_pred  # [B, seq_len, C]

    def compute_task_loss(self, pred: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
        loss = (pred - targets).pow(2)
        if loss_mask is not None and loss_mask.numel() > 0:
            mask = loss_mask.unsqueeze(-1).float()
            loss = loss.mul(mask).div(mask.mean().clamp(min=1e-8))
        return loss.mean()

    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        from ltx_core.components.patchifiers import get_pixel_coords
        from ltx_core.types import VideoLatentShape

        latent_shape = VideoLatentShape(
            batch=batch_size,
            channels=_LTX2_LATENT_CHANNELS,
            frames=num_frames,
            height=height,
            width=width,
        )
        latent_coords = self._patchifier.get_patch_grid_bounds(
            output_shape=latent_shape, device=device
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self._scale_factors,
            causal_fix=True,
        ).to(dtype)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / self._fps
        return pixel_coords

    def get_output_transforms(self, model: nn.Module) -> dict[str, Callable]:
        # BasicAVTransformerBlock returns (TransformerArgs | None, TransformerArgs | None).
        # Extract the video hidden state from the first element.
        transforms: dict[str, Callable] = {}
        if hasattr(model, "transformer_blocks"):
            for i in range(len(model.transformer_blocks)):
                path = f"transformer_blocks.{i}"
                transforms[path] = lambda out, _i=i: out[0].x
        return transforms

    def _create_conditioning_mask(
        self,
        batch_size: int,
        seq_len: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        """Boolean mask where True = first-frame conditioning token."""
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if self._first_frame_p > 0 and random.random() < self._first_frame_p:
            first_frame_tokens = height * width
            if first_frame_tokens < seq_len:
                mask[:, :first_frame_tokens] = True
        return mask


def create_ltx2_adapter(**kwargs) -> LTX2TrainingForwardAdapter:
    """Factory for LTX2TrainingForwardAdapter.

    Accepts fps and first_frame_conditioning_p as keyword args.
    """
    return LTX2TrainingForwardAdapter(
        fps=kwargs.get("fps", 25.0),
        first_frame_conditioning_p=kwargs.get("first_frame_conditioning_p", 0.0),
    )
