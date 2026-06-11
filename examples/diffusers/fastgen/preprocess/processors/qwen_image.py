# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/processors/qwen_image.py
# Vendored into the fastgen example so preprocessing runs against stock nemo_automodel
# without the un-packaged AutoModel ``tools/`` tree. Self-contained (relative ``.base``/``.registry`` imports). Original license below.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""
Qwen-Image model processor for preprocessing.

Handles Qwen/Qwen-Image T2I models with:
- VAE for image encoding
- Qwen2 text encoder for text conditioning
"""

import logging
from typing import Any, Dict

import torch
from torch import autocast

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("qwen_image")
class QwenImageProcessor(BaseModelProcessor):
    """
    Processor for Qwen-Image T2I models.

    Qwen-Image uses a VAE for image encoding and a Qwen2 text encoder
    for text conditioning.
    """

    @property
    def model_type(self) -> str:
        return "qwen_image"

    @property
    def default_model_name(self) -> str:
        return "Qwen/Qwen-Image"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load Qwen-Image models.

        Args:
            model_name: HuggingFace model path (e.g., 'Qwen/Qwen-Image')
            device: Device to load models on

        Returns:
            Dict containing:
                - vae: AutoencoderKL
                - tokenizer: Qwen2 tokenizer
                - text_encoder: Qwen2 text encoder
        """
        from diffusers import QwenImagePipeline

        logger.info("[Qwen-Image] Loading models from %s...", model_name)

        # Load pipeline without transformer (not needed for preprocessing)
        pipeline = QwenImagePipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )

        models = {}

        logger.info("  Configuring VAE...")
        models["vae"] = pipeline.vae.to(device=device, dtype=torch.bfloat16)
        models["vae"].eval()

        logger.info("  Configuring Qwen2 text encoder...")
        pipeline.text_encoder.to(device)
        pipeline.text_encoder.eval()

        # Keep pipeline for encode_prompt — it owns the tokenizer, text_encoder,
        # chat template, and system-token dropping logic.
        models["pipeline"] = pipeline

        torch.cuda.empty_cache()

        logger.info("[Qwen-Image] Models loaded successfully!")
        return models

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode image to latent space using VAE.

        Args:
            image_tensor: Image tensor (1, 3, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            Latent tensor (C, H//8, W//8), FP16
        """
        vae = models["vae"]
        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

        # Qwen-Image VAE expects 5D input (B, C, T, H, W) — add frame dim for single image
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(2)

        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()

        # Normalize using per-channel latents_mean / latents_std
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latent = (latent - latents_mean) / latents_std

        # Remove frame dim if added, then batch dim → (C, H, W)
        return latent.detach().cpu().to(torch.float16).squeeze(2).squeeze(0)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using the QwenImagePipeline's encode_prompt.

        Delegates to the diffusers pipeline which applies the correct chat
        template, tokenization, system-token dropping, and attention masking.

        Args:
            prompt: Text prompt
            models: Dict containing 'pipeline' (QwenImagePipeline)
            device: Device to use

        Returns:
            Dict containing:
                - prompt_embeds: Qwen2 hidden states [1, seq_len, hidden_dim]
                - prompt_embeds_mask: Attention mask [1, seq_len]
        """
        pipeline = models["pipeline"]

        with torch.no_grad():
            prompt_embeds, _ = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
            )

        return {
            "prompt_embeds": prompt_embeds.detach().cpu().to(torch.bfloat16),
        }

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify latent can be decoded back to reasonable image.

        Args:
            latent: Encoded latent (C, H, W)
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            True if verification passes
        """
        try:
            vae = models["vae"]
            device_type = "cuda" if "cuda" in device else "cpu"

            # (C, H, W) → (B, C, T, H, W) for Qwen-Image VAE
            latent = latent.unsqueeze(0).unsqueeze(2).to(device).float()

            with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float32):
                # Denormalize: reverse (latent - mean) / std
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, latent.dtype)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, latent.dtype)
                latent = latent * latents_std + latents_mean
                decoded = vae.decode(latent).sample

            # decoded is 5D (B, C, T, H, W) — take first frame
            decoded = decoded[:, :, 0]
            _, c, h, w = decoded.shape
            if c != 3:
                return False

            if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                return False

            return True

        except Exception as e:
            logger.warning("[Qwen-Image] Verification failed: %s", e)
            return False

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for Qwen-Image.

        Args:
            latent: Encoded latent
            text_encodings: Dict from encode_text()
            metadata: Additional metadata

        Returns:
            Dict to save with torch.save()
        """
        return {
            # Image latent
            "latent": latent,
            # Text embeddings
            "prompt_embeds": text_encodings["prompt_embeds"],
            # Metadata
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "image_path": metadata["image_path"],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            # Model info
            "model_type": self.model_type,
        }
