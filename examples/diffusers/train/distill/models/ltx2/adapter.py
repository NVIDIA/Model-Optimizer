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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ...interfaces import BackboneInputs

if TYPE_CHECKING:
    from collections.abc import Callable

_LTX2_LATENT_CHANNELS = 128


class LTX2TrainingForwardAdapter:
    """Flow matching training forward adapter for LTX-2.

    Stateful: holds the patchifier, scale factors, and text embedding
    connectors needed for training.

    Text embeddings in the batch are raw Gemma features (pre-connector).
    The connector is run here in prepare_inputs to produce video_embeds
    and audio_embeds. The transformer's internal caption_projection (MLP)
    is then applied during the model forward pass.

    Args:
        fps: Frames per second used to scale temporal position coordinates.
        first_frame_conditioning_p: Probability of conditioning on the first
            frame during training. 0.0 disables first-frame conditioning.
    """

    MOCK_LATENT_SHAPE: tuple[int, ...] = (128, 4, 32, 32)
    MOCK_TEXT_EMBED_DIM: int = 3840
    # Audio latent shape: [C=8, T=25, F=16] (1 second of audio at ~25 frames/sec)
    MOCK_AUDIO_LATENT_SHAPE: tuple[int, ...] = (8, 25, 16)

    def __init__(self, fps: float = 25.0, first_frame_conditioning_p: float = 0.0) -> None:
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.types import SpatioTemporalScaleFactors

        self.patch_size = 1
        self._patchifier = VideoLatentPatchifier(patch_size=self.patch_size)
        self._audio_patchifier = AudioPatchifier(patch_size=self.patch_size)
        self._scale_factors = SpatioTemporalScaleFactors.default()
        self._fps = fps
        self._first_frame_p = first_frame_conditioning_p

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        noise: Tensor,
        timesteps: Tensor,
        pipeline=None,
    ) -> BackboneInputs:
        latents = batch["latents"]  # [B, C, F, H, W]
        text_embeds = batch["text_embeds"]  # [B, L, D] raw text features
        text_mask = batch.get("text_mask")  # [B, L] attention mask

        b, _c, n_f, n_h, n_w = latents.shape
        device = latents.device
        dtype = latents.dtype

        # Process raw text embeddings through pipeline (runs connectors)
        assert pipeline is not None, (
            "LTX-2 adapter requires a pipeline for text embedding connectors."
        )
        processed = pipeline.process_text_embeddings(text_embeds, text_mask)
        video_embeds = processed.video_context
        audio_embeds = processed.audio_context

        # --- Video ---
        sigmas = timesteps.view(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        targets = noise - latents

        noisy_patchified = self._patchifier.patchify(noisy_latents)
        targets_patchified = self._patchifier.patchify(targets)
        clean_patchified = self._patchifier.patchify(latents)
        seq_len = noisy_patchified.shape[1]

        conditioning_mask = self._create_conditioning_mask(b, seq_len, n_h, n_w, device)
        if conditioning_mask.any():
            cond_expanded = conditioning_mask.unsqueeze(-1)
            noisy_patchified = torch.where(cond_expanded, clean_patchified, noisy_patchified)

        per_token_ts = timesteps.view(-1, 1).expand(-1, seq_len).clone()
        if conditioning_mask.any():
            per_token_ts[conditioning_mask] = 0.0

        positions = self._get_positions(n_f, n_h, n_w, b, device, dtype)

        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            latent=noisy_patchified,
            timesteps=per_token_ts,
            positions=positions,
            context=video_embeds,
            context_mask=None,
        )

        video_loss_mask = ~conditioning_mask

        # --- Audio (optional) ---
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if "audio_latents" in batch:
            audio_latents = batch["audio_latents"]  # [B, C=8, T, F=16]

            # Patchify: [B, C, T, F] -> [B, T, C*F]
            audio_patchified = self._audio_patchifier.patchify(audio_latents)
            audio_seq_len = audio_patchified.shape[1]

            audio_noise = torch.randn_like(audio_patchified)
            sigmas_audio = timesteps.view(-1, 1, 1)  # Same sigma as video
            noisy_audio = (1.0 - sigmas_audio) * audio_patchified + sigmas_audio * audio_noise
            audio_targets = audio_noise - audio_patchified

            audio_timesteps = timesteps.view(-1, 1).expand(-1, audio_seq_len)
            audio_positions = self._get_audio_positions(audio_seq_len, b, device, dtype)

            audio_modality = Modality(
                enabled=True,
                latent=noisy_audio,
                timesteps=audio_timesteps,
                positions=audio_positions,
                context=audio_embeds,
                context_mask=None,
            )

            audio_loss_mask = torch.ones(b, audio_seq_len, dtype=torch.bool, device=device)

        forward_kwargs = {
            "video": video_modality,
            "audio": audio_modality,
            "perturbations": None,
        }

        return BackboneInputs(
            noisy_input=noisy_patchified,
            targets=targets_patchified,
            loss_mask=video_loss_mask,
            forward_kwargs=forward_kwargs,
            audio_targets=audio_targets,
            audio_loss_mask=audio_loss_mask,
        )

    def forward_model(
        self, model: nn.Module, inputs: BackboneInputs
    ) -> tuple[Tensor, Tensor | None]:
        video_pred, audio_pred = model(**inputs.forward_kwargs)
        return video_pred, audio_pred

    def _masked_mse(self, pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
        """MSE loss with optional boolean mask."""
        loss = (pred - target).pow(2)
        if mask is not None and mask.numel() > 0:
            m = mask.unsqueeze(-1).float()
            loss = loss.mul(m).div(m.mean().clamp(min=1e-8))
        return loss.mean()

    def compute_task_loss(
        self, model_output: tuple[Tensor, Tensor | None], inputs: BackboneInputs
    ) -> Tensor:
        video_pred, audio_pred = model_output
        loss = self._masked_mse(video_pred, inputs.targets, inputs.loss_mask)
        if audio_pred is not None and inputs.audio_targets is not None:
            loss = loss + (audio_pred - inputs.audio_targets).pow(2).mean()
        return loss

    def compute_distillation_loss(
        self,
        student_output: tuple[Tensor, Tensor | None],
        teacher_output: tuple[Tensor, Tensor | None],
        inputs: BackboneInputs,
    ) -> Tensor:
        s_video, s_audio = student_output
        t_video, t_audio = teacher_output
        loss = self._masked_mse(s_video, t_video, inputs.loss_mask)
        if s_audio is not None and t_audio is not None:
            loss = loss + (s_audio - t_audio).pow(2).mean()
        return loss

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

    def _get_audio_positions(
        self,
        num_time_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate audio position embeddings [B, 1, T, 2].

        Audio positions are temporal-only (no spatial H×W), containing
        [start_sec, end_sec] for each audio latent frame.
        """
        from ltx_core.types import AudioLatentShape

        audio_shape = AudioLatentShape(
            batch=batch_size,
            channels=8,
            frames=num_time_steps,
            mel_bins=16,
        )
        latent_coords = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_shape, device=device
        )
        return latent_coords.to(dtype)

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
