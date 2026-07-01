# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Qwen-Image-Edit preprocessing support.

Qwen-Image-Edit-2511 conditions the denoiser through two independent paths:

* every reference image is encoded by the Qwen2.5-VL prompt encoder together with the edit
  instruction; and
* every reference image is encoded by the Qwen-Image VAE and appended to the noisy target
  tokens by the denoiser adapter.

This processor intentionally keeps those two representations separate in the cache.  Target
latents use the same sampled-VAE convention as :class:`QwenImageProcessor`, while reference
latents use the deterministic posterior mode used by ``QwenImageEditPlusPipeline`` at inference.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch

from .qwen_image import QwenImageProcessor
from .registry import ProcessorRegistry

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_CONDITION_IMAGE_AREA = 384 * 384
_VAE_IMAGE_AREA = 1024 * 1024


def _dimensions_for_area(image: Image.Image, area: int) -> tuple[int, int]:
    """Return ``(width, height)`` preserving aspect ratio, rounded to multiples of 32."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    ratio = width / height
    resized_width = max(32, round(math.sqrt(area * ratio) / 32) * 32)
    resized_height = max(32, round(math.sqrt(area / ratio) / 32) * 32)
    return resized_width, resized_height


def _posterior_mode(encoder_output: Any) -> torch.Tensor:
    """Extract the deterministic VAE posterior mode across diffusers return variants."""

    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access VAE latents from encoder output")


@ProcessorRegistry.register("qwen_image_edit")
class QwenImageEditProcessor(QwenImageProcessor):
    """Precompute the full Qwen-Image-Edit-2511 conditioning contract."""

    @property
    def model_type(self) -> str:
        return "qwen_image_edit"

    @property
    def default_model_name(self) -> str:
        return "Qwen/Qwen-Image-Edit-2511"

    def load_models(self, model_name: str, device: str) -> dict[str, Any]:
        """Load only the VAE and multimodal prompt encoder needed for caching."""

        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as exc:  # pragma: no cover - depends on the runtime environment
            raise ImportError(
                "Qwen-Image-Edit-2511 preprocessing requires a diffusers release that provides "
                "QwenImageEditPlusPipeline (introduced in diffusers 0.36.0). Full 2511 "
                "denoising also requires `zero_cond_t` support (stable diffusers>=0.37)."
            ) from exc

        logger.info("[Qwen-Image-Edit] Loading preprocessing models from %s", model_name)
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )
        pipeline.vae.to(device=device, dtype=torch.bfloat16).eval()
        pipeline.text_encoder.to(device).eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"vae": pipeline.vae, "pipeline": pipeline}

    def _vision_images(self, images: list[Image.Image], pipeline: Any) -> list[Image.Image]:
        """Resize references exactly as the Edit Plus prompt-encoding path does."""

        prepared = []
        for image in images:
            width, height = _dimensions_for_area(image, _CONDITION_IMAGE_AREA)
            prepared.append(pipeline.image_processor.resize(image, height, width))
        return prepared

    def encode_conditioning_images(
        self,
        images: list[Image.Image],
        models: dict[str, Any],
        device: str,
        *,
        max_pixels: int = _VAE_IMAGE_AREA,
    ) -> list[torch.Tensor]:
        """Encode one or more references with deterministic VAE posterior modes.

        Each returned tensor is ``[C, H/8, W/8]``.  References are kept as a list because the
        Edit Plus model permits different aspect ratios for different references.
        """

        if not images:
            raise ValueError("Qwen-Image-Edit requires at least one conditioning image")
        vae = models["vae"]
        pipeline = models["pipeline"]
        latents = []
        for image in images:
            width, height = _dimensions_for_area(image, max_pixels)
            image_tensor = pipeline.image_processor.preprocess(image, height, width).unsqueeze(2)
            image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                latent = _posterior_mode(vae.encode(image_tensor))

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
            latents.append(latent.detach().cpu().to(torch.float16).squeeze(2).squeeze(0))
        return latents

    def encode_multimodal_text(
        self,
        prompt: str,
        images: list[Image.Image],
        models: dict[str, Any],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Encode an instruction and its reference images with Qwen2.5-VL."""

        if not images:
            raise ValueError("Qwen-Image-Edit prompt encoding requires at least one image")
        pipeline = models["pipeline"]
        vision_images = self._vision_images(images, pipeline)
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=prompt,
                image=vision_images,
                device=device,
            )
        result = {"prompt_embeds": prompt_embeds.detach().cpu().to(torch.bfloat16)}
        if prompt_embeds_mask is not None:
            result["prompt_embeds_mask"] = prompt_embeds_mask.detach().cpu().to(torch.long)
        return result

    def encode_edit_prompts(
        self,
        prompt: str,
        negative_prompt: str,
        images: list[Image.Image],
        models: dict[str, Any],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Encode positive and per-sample negative multimodal conditioning."""

        positive = self.encode_multimodal_text(prompt, images, models, device)
        negative = self.encode_multimodal_text(negative_prompt, images, models, device)
        result = dict(positive)
        result["negative_prompt_embeds"] = negative["prompt_embeds"]
        if "prompt_embeds_mask" in negative:
            result["negative_prompt_embeds_mask"] = negative["prompt_embeds_mask"]
        return result

    def encode_text(
        self,
        prompt: str,
        models: dict[str, Any],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Reject text-only use, which would silently omit image tokens from the cache."""

        raise ValueError(
            "QwenImageEditProcessor.encode_text cannot encode a text-only prompt. Use "
            "encode_multimodal_text/encode_edit_prompts with the conditioning images."
        )

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: dict[str, Any],
        device: str,
    ) -> bool:
        """Decode a target latent with the VAE's actual dtype and validate finiteness."""

        try:
            vae = models["vae"]
            vae_dtype = next(vae.parameters()).dtype
            value = latent.unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_dtype)
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(device=device, dtype=vae_dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(device=device, dtype=vae_dtype)
            )
            with torch.no_grad():
                decoded = vae.decode(value * latents_std + latents_mean).sample[:, :, 0]
            return (
                decoded.ndim == 4 and decoded.shape[1] == 3 and torch.isfinite(decoded).all().item()
            )
        except Exception as exc:
            logger.warning("[Qwen-Image-Edit] Latent verification failed: %s", exc)
            return False

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct an image-edit cache record consumed by ``ImageToImageDataset``."""

        conditioning_latents = metadata.get("conditioning_latents")
        if not isinstance(conditioning_latents, list) or not conditioning_latents:
            raise ValueError("metadata['conditioning_latents'] must be a non-empty tensor list")
        required_text = ("prompt_embeds", "negative_prompt_embeds")
        missing = [key for key in required_text if key not in text_encodings]
        if missing:
            raise KeyError(f"Missing edit text encodings: {missing}")

        cache = {
            "latent": latent,
            "conditioning_latents": conditioning_latents,
            "prompt_embeds": text_encodings["prompt_embeds"],
            "negative_prompt_embeds": text_encodings["negative_prompt_embeds"],
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "negative_prompt": metadata["negative_prompt"],
            "image_path": metadata["image_path"],
            "conditioning_image_paths": metadata["conditioning_image_paths"],
            "conditioning_resolutions": metadata["conditioning_resolutions"],
            "target_latent_shape": tuple(latent.shape),
            "conditioning_latent_shapes": [tuple(value.shape) for value in conditioning_latents],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            "sample_id": metadata["sample_id"],
            "model_type": self.model_type,
        }
        for key in ("prompt_embeds_mask", "negative_prompt_embeds_mask"):
            if key in text_encodings:
                cache[key] = text_encodings[key]
        if metadata.get("source_metadata") is not None:
            cache["source_metadata"] = metadata["source_metadata"]
        return cache


__all__ = ["QwenImageEditProcessor"]
