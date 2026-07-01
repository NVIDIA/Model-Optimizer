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

"""Cached image-to-image dataset for Qwen-Image-Edit DMD2 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from nemo_automodel.components.datasets.diffusion.base_dataset import BaseMultiresolutionDataset


def _remove_cache_batch_dim(
    tensor: torch.Tensor,
    name: str,
    unbatched_ndim: int,
) -> torch.Tensor:
    """Remove the singleton encoder batch dimension while rejecting malformed caches."""

    if not torch.is_tensor(tensor):
        raise TypeError(f"Cached {name!r} must be a tensor, got {type(tensor).__name__}")
    if tensor.ndim == unbatched_ndim + 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != unbatched_ndim:
        raise ValueError(
            f"Cached {name!r} must be {unbatched_ndim}D after removing an optional "
            f"singleton batch dimension, got shape {tuple(tensor.shape)}"
        )
    return tensor


class ImageToImageDataset(BaseMultiresolutionDataset):
    """Read target/reference latents and per-sample multimodal prompt embeddings."""

    def __init__(self, cache_dir: str, train_text_encoder: bool = False):
        if train_text_encoder:
            raise NotImplementedError(
                "Qwen-Image-Edit requires cached multimodal embeddings; on-the-fly text encoder "
                "training is not supported by ImageToImageDataset."
            )
        self.train_text_encoder = False
        super().__init__(cache_dir, quantization=64)

    def _validated_cache_file(self, item: dict[str, Any]) -> Path:
        cache_file = Path(item["cache_file"]).resolve()
        cache_dir = Path(self.cache_dir).resolve()
        try:
            cache_file.relative_to(cache_dir)
        except ValueError as exc:
            raise ValueError(
                f"Cache file {cache_file} is outside cache directory {cache_dir}"
            ) from exc
        return cache_file

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.metadata[idx]
        data = torch.load(
            self._validated_cache_file(item),
            map_location="cpu",
            weights_only=True,
        )
        target_latent = data.get("latent")
        if not torch.is_tensor(target_latent) or target_latent.ndim != 3:
            raise ValueError(f"Cache item {idx} target latent must have shape [C,H,W]")

        conditioning_latents = data.get("conditioning_latents")
        if not isinstance(conditioning_latents, list) or not conditioning_latents:
            raise ValueError(
                f"Cache item {idx} must contain a non-empty `conditioning_latents` list"
            )
        if not all(torch.is_tensor(latent) and latent.ndim == 3 for latent in conditioning_latents):
            raise ValueError(f"Cache item {idx} conditioning latents must all have shape [C,H,W]")

        resolution_key = "bucket_resolution" if "bucket_resolution" in item else "crop_resolution"
        prompt_embeds = _remove_cache_batch_dim(
            data["prompt_embeds"], "prompt_embeds", unbatched_ndim=2
        )
        negative_prompt_embeds = _remove_cache_batch_dim(
            data["negative_prompt_embeds"], "negative_prompt_embeds", unbatched_ndim=2
        )
        prompt_mask = data.get("prompt_embeds_mask")
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_embeds.shape[0], dtype=torch.long)
        else:
            prompt_mask = _remove_cache_batch_dim(
                prompt_mask, "prompt_embeds_mask", unbatched_ndim=1
            ).long()
        negative_mask = data.get("negative_prompt_embeds_mask")
        if negative_mask is None:
            negative_mask = torch.ones(negative_prompt_embeds.shape[0], dtype=torch.long)
        else:
            negative_mask = _remove_cache_batch_dim(
                negative_mask,
                "negative_prompt_embeds_mask",
                unbatched_ndim=1,
            ).long()

        output = {
            "latent": target_latent,
            "conditioning_latents": conditioning_latents,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_mask,
            "crop_resolution": torch.tensor(item[resolution_key]),
            "original_resolution": torch.tensor(item["original_resolution"]),
            "crop_offset": torch.tensor(data["crop_offset"]),
            "prompt": data["prompt"],
            "negative_prompt": data.get("negative_prompt", " "),
            "image_path": data["image_path"],
            "conditioning_image_paths": data["conditioning_image_paths"],
            "conditioning_resolutions": data.get(
                "conditioning_resolutions",
                [None] * len(conditioning_latents),
            ),
            "target_latent_shape": data.get("target_latent_shape", tuple(target_latent.shape)),
            "conditioning_latent_shapes": data.get(
                "conditioning_latent_shapes",
                [tuple(value.shape) for value in conditioning_latents],
            ),
            "sample_id": data.get("sample_id", str(idx)),
            "bucket_id": item["bucket_id"],
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }
        if "source_metadata" in data:
            output["source_metadata"] = data["source_metadata"]
        return output


__all__ = ["ImageToImageDataset"]
