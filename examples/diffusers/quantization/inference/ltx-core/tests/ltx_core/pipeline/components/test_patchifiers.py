# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Ivan Zorin

import torch

from ltx_core.pipeline.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.pipeline.components.protocols import AudioLatentShape, VideoLatentShape


def test_video_latent_patchifier() -> None:
    # Setup
    batch_size = 2
    channels = 128
    frames = 8
    height = 32
    width = 32
    patch_size = 4

    # Create patchifier
    patchifier = VideoLatentPatchifier(patch_size=patch_size)
    assert patchifier is not None
    assert patchifier.patch_size == (1, patch_size, patch_size)

    output_shape = VideoLatentShape(
        batch=batch_size,
        channels=channels,
        frames=frames,
        height=height,
        width=width,
    )
    # Create random latents
    latents = torch.randn(batch_size, channels, frames, height, width)

    # Test patchify
    patches = patchifier.patchify(latents)
    coords = patchifier.get_patch_grid_bounds(
        output_shape=output_shape,
        device=latents.device,
    )

    expected_num_patches = frames * (height // patch_size) * (width // patch_size)
    expected_features = channels * patch_size * patch_size

    assert patches.shape == (batch_size, expected_num_patches, expected_features)
    assert coords.shape == (batch_size, 3, expected_num_patches, 2)

    # Test unpatchify
    reconstructed_latents = patchifier.unpatchify(patches, output_shape)
    reconstructed_coords = patchifier.get_patch_grid_bounds(
        output_shape=output_shape,
        device=reconstructed_latents.device,
    )

    # Verify roundtrip
    assert torch.allclose(latents, reconstructed_latents, atol=1e-6), (
        f"Unpatchified latents do not match original latents: {latents.shape} != {reconstructed_latents.shape}"
    )
    assert torch.allclose(coords, reconstructed_coords, atol=1e-6), (
        f"Coordinates of unpatchified latents do not match: {coords} != {reconstructed_coords}"
    )


def test_audio_patchifier() -> None:
    batch_size = 2
    channels = 4
    frames = 12
    freq_bins = 16
    patchifier = AudioPatchifier(patch_size=16)

    latents = torch.randn(batch_size, channels, frames, freq_bins)
    patches = patchifier.patchify(latents)

    expected_features = channels * freq_bins
    assert patches.shape == (batch_size, frames, expected_features)

    output_shape = AudioLatentShape(
        batch=batch_size,
        channels=channels,
        frames=frames,
        mel_bins=freq_bins,
    )
    coords = patchifier.get_patch_grid_bounds(
        output_shape=output_shape,
        device=latents.device,
    )
    assert coords.shape == (batch_size, 1, frames, 2)

    reconstructed_latents = patchifier.unpatchify(patches, output_shape)
    assert torch.allclose(latents, reconstructed_latents, atol=1e-6)

    reconstructed_coords = patchifier.get_patch_grid_bounds(
        output_shape=output_shape,
        device=reconstructed_latents.device,
    )
    assert torch.allclose(coords, reconstructed_coords, atol=1e-6)
