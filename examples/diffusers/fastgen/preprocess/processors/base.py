# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/processors/base.py
# Vendored into the fastgen example so preprocessing runs against stock nemo_automodel
# without the un-packaged AutoModel ``tools/`` tree. Self-contained (no internal import changes). Original license below.
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

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from PIL import Image


class BaseModelProcessor(ABC):
    """
    Abstract base class for model-specific preprocessing logic.

    Each model architecture (FLUX, SDXL, SD1.5, SD3, etc.) should have its own
    processor implementation that handles:
    - Model loading (VAE, text encoders)
    - Image encoding to latent space
    - Text encoding to embeddings
    - Verification of encoded latents
    - Cache data structure formatting
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return the model type identifier.

        Returns:
            str: Model type (e.g., 'flux', 'sdxl', 'sd15', 'sd3')
        """
        pass

    @property
    def default_model_name(self) -> str:
        """
        Return the default HuggingFace model path for this processor.

        Returns:
            str: Default model name/path
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not specify a default model name")

    @abstractmethod
    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load all required models for this architecture.

        Args:
            model_name: HuggingFace model name/path
            device: Device to load models on (e.g., 'cuda', 'cuda:0', 'cpu')

        Returns:
            Dict containing all loaded models and tokenizers
        """
        pass

    @abstractmethod
    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode image tensor to latent space.

        Args:
            image_tensor: Image tensor of shape (1, C, H, W), normalized to [-1, 1]
            models: Dict of loaded models from load_models()
            device: Device to use for encoding

        Returns:
            Latent tensor (typically shape (C, H//8, W//8) for most VAEs)
        """
        pass

    @abstractmethod
    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            models: Dict of loaded models from load_models()
            device: Device to use for encoding

        Returns:
            Dict containing all text embeddings (keys vary by model type)
        """
        pass

    @abstractmethod
    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify that a latent can be decoded back to a reasonable image.

        Args:
            latent: Encoded latent tensor
            models: Dict of loaded models from load_models()
            device: Device to use for verification

        Returns:
            True if verification passes, False otherwise
        """
        pass

    @abstractmethod
    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct the cache dictionary to save.

        Args:
            latent: Encoded latent tensor
            text_encodings: Dict of text embeddings from encode_text()
            metadata: Dict containing:
                - original_resolution: Tuple[int, int]
                - bucket_resolution: Tuple[int, int]
                - crop_offset: Tuple[int, int]
                - prompt: str
                - image_path: str
                - bucket_id: str
                - tier: str
                - aspect_ratio: float

        Returns:
            Dict to be saved with torch.save()
        """
        pass

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to normalized tensor.

        Default implementation handles standard preprocessing.
        Override if model requires different preprocessing.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor of shape (1, 3, H, W), normalized to [-1, 1]
        """
        import numpy as np

        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)

        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor

    def get_vae_scaling_factor(self, models: Dict[str, Any]) -> float:
        """
        Get the VAE scaling factor for this model.

        Args:
            models: Dict of loaded models

        Returns:
            Scaling factor (typically from vae.config.scaling_factor)
        """
        if "vae" in models and hasattr(models["vae"], "config"):
            return models["vae"].config.scaling_factor
        return 0.18215  # Default for most models
