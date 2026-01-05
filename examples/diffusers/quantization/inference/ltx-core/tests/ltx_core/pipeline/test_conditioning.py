# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import torch

from ltx_core.pipeline.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.pipeline.conditioning import AudioConditioningBuilder, ConditioningMethod, VideoConditioningBuilder


def test_video_conditioning_builder() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    builder = VideoConditioningBuilder(patchifier=patchifier, batch=2, width=128, height=128, num_frames=17, fps=25)
    conditioning = builder.build(device="cpu", dtype=torch.float32, generator=torch.Generator(device="cpu"))
    assert conditioning.latent.shape == (2, 48, 128)
    assert conditioning.denoise_mask.shape == (2, 48)
    assert conditioning.positions.shape == (2, 3, 48, 2)

    image_latent = torch.ones(2, 128, 1, 4, 4)
    strength = 0.4
    builder = builder.with_single_frame(
        image_latent=image_latent, strength=strength, frame_idx=0, method=ConditioningMethod.REPLACE
    )
    conditioning = builder.build(device="cpu", dtype=torch.float32, generator=torch.Generator(device="cpu"))
    assert conditioning.latent.shape == (2, 48, 128)
    assert conditioning.denoise_mask.shape == (2, 48)
    assert conditioning.positions.shape == (2, 3, 48, 2)

    latent_height, latent_width = 4, 4

    cond_denoise_mask_view = conditioning.denoise_mask[:, : latent_height * latent_width]
    uncond_denoise_mask_view = conditioning.denoise_mask[:, latent_height * latent_width :]
    assert torch.allclose(cond_denoise_mask_view, torch.full_like(cond_denoise_mask_view, 1.0 - strength))
    assert torch.allclose(uncond_denoise_mask_view, torch.ones_like(uncond_denoise_mask_view))


def test_video_conditioning_builder_causal_fix_false() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    builder = VideoConditioningBuilder(
        patchifier=patchifier, batch=2, width=128, height=128, num_frames=16, fps=25, causal_fix=False
    )
    conditioning = builder.build(device="cpu", dtype=torch.float32, generator=torch.Generator(device="cpu"))
    assert conditioning.latent.shape == (2, 32, 128)
    assert conditioning.denoise_mask.shape == (2, 32)
    assert conditioning.positions.shape == (2, 3, 32, 2)


def test_audio_conditioning_builder() -> None:
    patchifier = AudioPatchifier(patch_size=1)
    builder = AudioConditioningBuilder(patchifier=patchifier, batch=2, duration=10.0)
    conditioning = builder.build(device="cpu", dtype=torch.float32, generator=torch.Generator(device="cpu"))
    assert conditioning.latent.shape == (2, 250, 128)
    assert conditioning.denoise_mask.shape == (2, 250)
    assert conditioning.positions.shape == (2, 1, 250, 2)


if __name__ == "__main__":
    test_video_conditioning_builder()
    test_video_conditioning_builder_causal_fix_false()
    test_audio_conditioning_builder()
