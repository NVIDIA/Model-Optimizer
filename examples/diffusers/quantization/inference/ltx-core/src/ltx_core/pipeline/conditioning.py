# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Self

import torch
from torch._prims_common import DeviceLikeType

from ltx_core.pipeline.components.patchifiers import (
    AudioLatentShape,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)


@dataclass
class ConditioningResult:
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor


class ConditioningMethod(Enum):
    """
    Method to condition the model.
    """

    REPLACE = "replace"
    """
    Replace the latent video frame with the conditioning.
    """
    APPEND = "append"
    """
    Append the new conditioning to the video latent, with positional embedding pointing to the conditioned frame.
    """


class ConditioningBuilder(Protocol):
    """
    Builds conditioning for the model.
    """

    def build(self, device: DeviceLikeType, dtype: torch.dtype) -> ConditioningResult: ...

    def unbuild(self, patchified_latent: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ConditioningItem:
    latent: torch.Tensor
    strength: float
    frame_idx: int
    frame_count: int
    method: ConditioningMethod


class VideoConditioningBuilder(ConditioningBuilder):
    """
    Builds conditioning for the video model.
    """

    def __init__(
        self,
        patchifier: VideoLatentPatchifier,
        batch: int,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        in_channels: int = 128,
        scale_factors: tuple[int, int, int] = (8, 32, 32),
        causal_fix: bool = True,
    ):
        self._patchifier = patchifier
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._fps = fps
        self._in_channels = in_channels
        self._batch = batch
        self._conditioning_items: list[ConditioningItem] = []

        if causal_fix and num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 1 for causal fix, got {num_frames}")

        if not causal_fix and num_frames % 8 != 0:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 0 for non-causal fix, got {num_frames}")

        self._scale_factors = scale_factors
        self._causal_fix = causal_fix

    @staticmethod
    def from_builder(builder: type[Self]) -> Self:
        builder_copy = copy.copy(builder)
        builder_copy._conditioning_items = copy.copy(builder._conditioning_items)
        return builder_copy

    def with_single_frame(
        self, image_latent: torch.Tensor, strength: float, frame_idx: int, method: ConditioningMethod
    ) -> Self:
        builder_copy = VideoConditioningBuilder.from_builder(self)
        builder_copy._conditioning_items.append(
            ConditioningItem(latent=image_latent, strength=strength, frame_idx=frame_idx, frame_count=1, method=method)
        )
        return builder_copy

    def build(self, device: DeviceLikeType, dtype: torch.dtype, generator: torch.Generator) -> ConditioningResult:
        latent = torch.randn(
            self._batch,
            self._in_channels,
            self._num_frames // self._scale_factors[0] + self._causal_fix,
            self._height // self._scale_factors[1],
            self._width // self._scale_factors[2],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        denoise_mask = torch.ones(latent.shape[0], 1, *latent.shape[2:], device=device, dtype=dtype)

        patchified_latent = self._patchifier.patchify(latent)
        patchified_denoise_mask = self._patchifier.patchify(denoise_mask)
        latent_coords = self._patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                height=self._height // self._scale_factors[1],
                width=self._width // self._scale_factors[2],
                frames=self._num_frames // self._scale_factors[0] + self._causal_fix,
                batch=self._batch,
                channels=self._in_channels,
            ),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self._scale_factors,
            causal_fix=self._causal_fix,
        ).to(dtype)

        self.apply_conditioning_(patchified_latent, patchified_denoise_mask)

        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / self._fps

        return ConditioningResult(
            latent=patchified_latent,
            denoise_mask=patchified_denoise_mask.squeeze(-1),
            positions=pixel_coords,
        )

    def apply_conditioning_(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> None:
        for item in self._conditioning_items:
            if item.method == ConditioningMethod.APPEND:
                raise NotImplementedError("APPEND method is not implemented yet.")
            if item.frame_idx != 0:
                raise NotImplementedError("Only first frame can be conditioned for now.")
            patchified_item = self._patchifier.patchify(item.latent)
            start_token = 0
            end_token = patchified_item.shape[1]

            latent[:, start_token:end_token].lerp_(patchified_item, item.strength)
            denoise_mask[:, start_token:end_token].fill_(1.0 - item.strength)

    def unbuild(self, patchified_latent: torch.Tensor) -> torch.Tensor:
        return self._patchifier.unpatchify(
            patchified_latent,
            output_shape=VideoLatentShape(
                height=self._height // self._scale_factors[1],
                width=self._width // self._scale_factors[2],
                frames=self._num_frames // self._scale_factors[0] + self._causal_fix,
                batch=self._batch,
                channels=self._in_channels,
            ),
        )


class AudioConditioningBuilder(ConditioningBuilder):
    """
    Builds conditioning for the audio model.
    """

    def __init__(self, patchifier: AudioPatchifier, batch: int, duration: float, channels: int = 8, mel_bins: int = 16):
        self._patchifier = patchifier
        self._duration = duration
        self._in_channels = channels
        self._mel_bins = mel_bins
        self._batch = batch
        self._latents_per_second = (
            float(self._patchifier.sample_rate)
            / float(self._patchifier.hop_length)
            / float(self._patchifier.audio_latent_downsample_factor)
        )

    def build(self, device: DeviceLikeType, dtype: torch.dtype, generator: torch.Generator) -> ConditioningResult:
        latent_length = int(self._duration * self._latents_per_second)
        latent = torch.randn(
            self._batch,
            self._in_channels,
            latent_length,
            self._mel_bins,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        patchified_latent = self._patchifier.patchify(latent)
        latent_coords = self._patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                channels=self._in_channels,
                frames=latent_length,
                mel_bins=self._mel_bins,
                batch=self._batch,
            ),
            device=device,
        )

        return ConditioningResult(
            latent=patchified_latent,
            denoise_mask=torch.ones(patchified_latent.shape[:2], device=device, dtype=dtype),
            positions=latent_coords,
        )

    def apply_conditioning_(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> None:
        pass

    def unbuild(self, patchified_latent: torch.Tensor) -> torch.Tensor:
        return self._patchifier.unpatchify(
            patchified_latent,
            output_shape=AudioLatentShape(
                batch=self._batch,
                channels=self._in_channels,
                mel_bins=self._mel_bins,
                frames=int(self._duration * self._latents_per_second),
            ),
        )
