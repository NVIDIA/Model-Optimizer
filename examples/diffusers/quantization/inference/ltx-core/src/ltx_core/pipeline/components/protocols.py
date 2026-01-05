# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Amit Pintz.

from collections import namedtuple
from typing import Protocol, Tuple

import torch

VideoLatentShape = namedtuple(
    "VideoLatentShape",
    [
        "batch",
        "channels",
        "frames",
        "height",
        "width",
    ],
)

AudioLatentShape = namedtuple(
    "AudioLatentShape",
    [
        "batch",
        "channels",
        "frames",
        "mel_bins",
    ],
)


class Patchifier(Protocol):
    """
    Protocol for patchifiers that convert latent tensors into patches and assemble them back.
    """

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        ...
        """
        Convert latent tensors into flattened patch tokens.

        Args:
            latents: Latent tensor to patchify.

        Returns:
            Flattened patch tokens tensor.
        """

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape | VideoLatentShape,
    ) -> torch.Tensor:
        """
        Converts latent tensors between spatio-temporal formats and flattened sequence representations.

        Args:
            latents: Patch tokens that must be rearranged back into the latent grid constructed by `patchify`.
            output_shape: Shape of the output tensor. Note that output_shape is either AudioLatentShape or
            VideoLatentShape.

        Returns:
            Dense latent tensor restored from the flattened representation.
        """

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        ...
        """
        Returns the patch size as a tuple of (temporal, height, width) dimensions
        """

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...
        """
        Compute metadata describing where each latent patch resides within the
        grid specified by `output_shape`.

        Args:
            output_shape: Target grid layout for the patches.
            device: Target device for the returned tensor.

        Returns:
            Tensor containing patch coordinate metadata such as spatial or temporal intervals.
        """


class SchedulerProtocol(Protocol):
    """
    Protocol for schedulers that provide a sigmas schedule tensor for a
    given number of steps.
    """

    def execute(self, steps: int, **kwargs) -> torch.Tensor: ...


class GuiderProtocol(Protocol):
    """
    Protocol for guiders that provide a delta tensor for given conditions.
    """

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor, *extra_conds: torch.Tensor) -> torch.Tensor: ...


class DiffusionStepProtocol(Protocol):
    """
    Protocol for diffusion steps that provide a next sample tensor for a given current sample tensor,
    velocity tensor, and sigmas tensor.
    """

    def step(
        self, sample: torch.Tensor, velocity: torch.Tensor, sigmas: torch.Tensor, step_index: int
    ) -> torch.Tensor: ...
