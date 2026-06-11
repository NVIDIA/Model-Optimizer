# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/nemo_automodel/components/datasets/diffusion/text_to_image_dataset.py
# Local modifications for the self-contained fastgen example: the relative import of the
# unpatched ``BaseMultiresolutionDataset`` is rewritten to its absolute ``nemo_automodel``
# package path. Includes the DMD2 ``prompt_embeds_mask`` emission. Original license below.
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

from pathlib import Path
from typing import Dict

import torch

from nemo_automodel.components.datasets.diffusion.base_dataset import BaseMultiresolutionDataset


class TextToImageDataset(BaseMultiresolutionDataset):
    """Text-to-Image dataset with hierarchical bucket organization."""

    def __init__(
        self,
        cache_dir: str,
        train_text_encoder: bool = False,
    ):
        """
        Args:
            cache_dir: Directory containing preprocessed cache
            train_text_encoder: If True, returns tokens instead of embeddings
        """
        self.train_text_encoder = train_text_encoder
        super().__init__(cache_dir, quantization=64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        item = self.metadata[idx]
        cache_file = Path(item["cache_file"]).resolve()
        cache_dir = Path(self.cache_dir).resolve()

        try:
            cache_file.relative_to(cache_dir)
        except ValueError as e:
            raise ValueError(f"Cache file {cache_file} is outside cache directory {cache_dir}") from e

        # Load cached data
        data = torch.load(cache_file, map_location="cpu", weights_only=True)

        # Prepare output - support both bucket_resolution and crop_resolution keys
        resolution_key = "bucket_resolution" if "bucket_resolution" in item else "crop_resolution"
        output = {
            "latent": data["latent"],
            "crop_resolution": torch.tensor(item[resolution_key]),
            "original_resolution": torch.tensor(item["original_resolution"]),
            "crop_offset": torch.tensor(data["crop_offset"]),
            "prompt": data["prompt"],
            "image_path": data["image_path"],
            "bucket_id": item["bucket_id"],
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }

        if self.train_text_encoder:
            output["clip_tokens"] = data["clip_tokens"].squeeze(0)
            output["t5_tokens"] = data["t5_tokens"].squeeze(0)
        else:
            # Model-agnostic: include whichever text embedding keys the cache provides
            if "clip_hidden" in data:
                output["clip_hidden"] = data["clip_hidden"].squeeze(0)
            if "pooled_prompt_embeds" in data:
                output["pooled_prompt_embeds"] = data["pooled_prompt_embeds"].squeeze(0)
            if "prompt_embeds" in data:
                output["prompt_embeds"] = data["prompt_embeds"].squeeze(0)
                if "prompt_embeds_mask" in data:
                    output["prompt_embeds_mask"] = data["prompt_embeds_mask"].squeeze(0)
                elif "text_mask" in data:
                    output["prompt_embeds_mask"] = data["text_mask"].squeeze(0)
                else:
                    output["prompt_embeds_mask"] = torch.ones(
                        output["prompt_embeds"].shape[0],
                        dtype=torch.long,
                    )

        return output
