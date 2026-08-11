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

import json
from pathlib import Path

import torch
from nemo_automodel.components.datasets.diffusion.base_dataset import BaseMultiresolutionDataset

from .paths import resolve_cache_root, resolve_under_root
from .splits import make_train_validation_indices

__all__ = ["TextToImageDataset"]


class TextToImageDataset(BaseMultiresolutionDataset):
    """Text-to-Image dataset with hierarchical bucket organization."""

    def __init__(
        self,
        cache_dir: str | Path,
        train_text_encoder: bool = False,
        prompt_only: bool = False,
        split: str | None = None,
        validation_count: int | None = None,
        split_seed: int = 2026,
    ):
        """
        Args:
            cache_dir: Directory containing preprocessed cache
            train_text_encoder: If True, returns tokens instead of embeddings
            prompt_only: Omit cached image latents from returned samples.
            split: Optional deterministic ``"train"`` or ``"validation"`` selection.
            validation_count: Number of validation samples when ``split`` is set.
            split_seed: Local seed used to construct deterministic split membership.
        """
        if split not in (None, "train", "validation"):
            raise ValueError("split must be null, 'train', or 'validation'")
        if split is not None and validation_count is None:
            raise ValueError("validation_count is required when split is set")
        self.train_text_encoder = train_text_encoder
        self.prompt_only = prompt_only
        self.cache_root = resolve_cache_root(cache_dir)
        self._split = split
        self._validation_count = validation_count
        self._split_seed = split_seed
        self._resolved_cache_files: dict[int, Path] = {}
        super().__init__(str(self.cache_root), quantization=64)

    def _load_metadata(self) -> list[dict]:
        """Load contained metadata and preserve original expansion ordinals as sample IDs."""
        metadata_file = resolve_under_root(self.cache_root, "metadata.json", "metadata index")
        index_bytes = metadata_file.read_bytes()
        index = json.loads(index_bytes)
        if not isinstance(index, dict) or not isinstance(index.get("shards"), list):
            raise ValueError(
                f"Invalid metadata format in {metadata_file}. Expected dict with 'shards' list."
            )

        complete_metadata: list[dict] = []
        for shard_index, shard_name in enumerate(index["shards"]):
            if not isinstance(shard_name, str) or not shard_name:
                raise TypeError(f"metadata shard {shard_index} must be a nonempty string")
            shard_path = resolve_under_root(
                self.cache_root, shard_name, f"metadata shard {shard_index}"
            )
            shard_bytes = shard_path.read_bytes()
            shard = json.loads(shard_bytes)
            if not isinstance(shard, list):
                raise ValueError(f"metadata shard {shard_path} must contain a list")
            for shard_item_index, item in enumerate(shard):
                if not isinstance(item, dict):
                    raise TypeError(
                        f"metadata shard {shard_path} item {shard_item_index} must be a dict"
                    )
                cache_file = item.get("cache_file")
                if not isinstance(cache_file, str) or not cache_file:
                    raise TypeError(
                        f"metadata shard {shard_path} item {shard_item_index} has invalid cache_file"
                    )
                complete_metadata.append(dict(item))

        if not complete_metadata:
            raise ValueError(f"No samples found in {metadata_file}")
        self.total_num_samples = len(complete_metadata)
        if self._split is None:
            self.sample_ids = list(range(self.total_num_samples))
        else:
            if self._validation_count is None:
                raise RuntimeError("validation_count was not resolved for the requested split")
            train, validation = make_train_validation_indices(
                self.total_num_samples,
                self._validation_count,
                self._split_seed,
            )
            self.sample_ids = train if self._split == "train" else validation
        return [complete_metadata[index] for index in self.sample_ids]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single sample."""
        item = self.metadata[idx]
        sample_id = self.sample_ids[idx]
        cache_file = self._resolved_cache_files.get(idx)
        if cache_file is None:
            # Resolve lazily so every rank checks only the payloads it actually reads instead of
            # issuing a full-cache metadata-stat storm at construction time.
            cache_file = resolve_under_root(
                self.cache_root, item["cache_file"], f"sample cache file {sample_id}"
            )
            self._resolved_cache_files[idx] = cache_file

        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        # Prepare output - support both bucket_resolution and crop_resolution keys
        resolution_key = "bucket_resolution" if "bucket_resolution" in item else "crop_resolution"
        output = {
            "crop_resolution": torch.tensor(item[resolution_key]),
            "original_resolution": torch.tensor(item["original_resolution"]),
            "crop_offset": torch.tensor(data["crop_offset"]),
            "prompt": data["prompt"],
            "image_path": data["image_path"],
            "bucket_id": item["bucket_id"],
            "aspect_ratio": item.get("aspect_ratio", 1.0),
        }
        if not self.prompt_only:
            output["latent"] = data["latent"]
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
