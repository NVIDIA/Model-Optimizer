# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Ivan Zorin

from typing import Callable

import numpy as np
import pytest
import torch
from skimage import data, io
from skimage.transform import resize
from tests.ltx_core.utils import resolve_model_path

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.video_vae.model_configurator import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VAEDecoderConfigurator,
    VAEEncoderConfigurator,
)

SAVE_IMAGES = False


VAE_CONFIG = {
    "vae": {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "encoder_spatial_padding_mode": "zeros",
        "decoder_spatial_padding_mode": "reflect",
        "encoder_blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
        ],
        "decoder_blocks": [
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 5, "inject_noise": False}],
        ],
        "scaling_factor": 1.0,
        "norm_layer": "pixel_norm",
        "patch_size": 4,
        "latent_log_var": "uniform",
        "use_quant_conv": False,
        "causal_decoder": False,
        "timestep_conditioning": True,
        "normalize_latent_channels": False,
    }
}


def test_encoder_instantiation() -> None:
    vae_encoder = VAEEncoderConfigurator.from_config(VAE_CONFIG)
    assert vae_encoder is not None


def test_decoder_instantiation() -> None:
    vae_decoder = VAEDecoderConfigurator.from_config(VAE_CONFIG)
    assert vae_decoder is not None


def test_encoder_loading() -> None:
    model_path = resolve_model_path()

    vae_encoder = Builder(
        model_path=model_path,
        model_class_configurator=VAEEncoderConfigurator,
        model_sd_key_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build()
    assert vae_encoder is not None


def test_decoder_loading() -> None:
    model_path = resolve_model_path()

    vae_decoder = Builder(
        model_path=model_path,
        model_class_configurator=VAEDecoderConfigurator,
        model_sd_key_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build()
    assert vae_decoder is not None


@pytest.mark.parametrize(
    ("image_name", "image_func"),
    [
        ("astronaut", data.astronaut),
        ("chelsea", data.chelsea),
        ("coffee", data.coffee),
    ],
)
def test_encode_decode_cycle(image_name: str, image_func: Callable) -> None:
    # Load weights from $MODEL_PATH or fall back to the default checkpoint
    model_path = resolve_model_path()

    dtype = torch.bfloat16
    device = torch.device("cuda")

    vae_decoder = Builder(
        model_path=model_path,
        model_class_configurator=VAEDecoderConfigurator,
        model_sd_key_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build()

    vae_encoder = Builder(
        model_path=model_path,
        model_class_configurator=VAEEncoderConfigurator,
        model_sd_key_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build()

    vae_encoder.to(dtype=dtype, device=device)
    vae_decoder.to(dtype=dtype, device=device)

    # Prepare Image
    image = image_func()

    # Resize if needed to match target shape
    target_shape = (512, 512)
    if image.shape[:2] != target_shape:
        # resize returns float 0-1
        image = resize(image, target_shape, anti_aliasing=True)
        # Convert to 0-255 uint8 range to match original pipeline assumption
        image = (image * 255).astype(np.uint8)

    # Normalize to [-1, 1]
    image = np.array(image).astype(np.float32) / 127.5 - 1.0

    # Convert to tensor (B, C, F, H, W)
    # Replicate the image 33 times to create a video
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    sample_video = image_tensor.repeat(1, 1, 33, 1, 1).to(device=device, dtype=dtype)

    # Run VAE
    with torch.autocast(device_type="cuda", dtype=dtype):
        encoded = vae_encoder(sample_video)
        assert not torch.isnan(encoded).any(), f"Encoded tensor contains NaNs for {image_name}"
        decoded = vae_decoder(encoded)
        assert not torch.isnan(decoded).any(), f"Decoded tensor contains NaNs for {image_name}"

    # Verify reconstruction shape
    assert decoded.shape == sample_video.shape, f"Shape mismatch for {image_name}"

    # Verify reconstruction error
    diff = (sample_video - decoded).float()
    mse = diff.pow(2).mean().item()

    # Assert MSE is reasonable
    # MSE threshold < 0.05 is conservative but safe for diverse images.
    assert mse < 0.02, f"MSE too high for {image_name}: {mse:.4f}"

    if SAVE_IMAGES:
        img_out = decoded[0, :, 0].detach().float().cpu().numpy()
        img_out = (img_out + 1.0) * 127.5
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)
        img_out = np.transpose(img_out, (1, 2, 0))  # (H, W, C)
        io.imsave(f"test_output_{image_name}.png", img_out)

    # Cleanup
    del encoded, decoded, sample_video, image_tensor, diff
    torch.cuda.empty_cache()
