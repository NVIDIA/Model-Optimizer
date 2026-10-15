# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset and dataloader utilities for precomputed latents + text embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)


class LatentDataset(Dataset):
    """Loads precomputed latents and text embeddings from a directory.

    Expected directory layout:
        data_root/
            sample_000000.safetensors   # contains: latents, text_embeds, text_mask
            sample_000001.safetensors
            ...
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.files = sorted(self.data_root.glob("*.safetensors"))
        if not self.files:
            raise FileNotFoundError(f"No .safetensors files found in {self.data_root}")
        logger.info(f"LatentDataset: {len(self.files)} samples from {self.data_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return load_file(str(self.files[idx]))


class MockDataset(Dataset):
    """Generates random tensors for pipeline testing without real data.

    Args:
        num_samples: Number of mock samples.
        latent_shape: (C, F, H, W) shape of video latents.
        text_embed_dim: Dimension of text embeddings.
        text_seq_len: Sequence length for text embeddings.
        audio_latent_shape: (C, T, F) shape of audio latents, or None to skip audio.
    """

    def __init__(
        self,
        num_samples: int = 100,
        latent_shape: tuple[int, ...] = (48, 4, 32, 32),
        text_embed_dim: int = 4096,
        text_seq_len: int = 512,
        audio_latent_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_samples = num_samples
        self.latent_shape = latent_shape
        self.text_embed_dim = text_embed_dim
        self.text_seq_len = text_seq_len
        self.audio_latent_shape = audio_latent_shape
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "latents": torch.randn(self.latent_shape, dtype=self.dtype),
            "text_embeds": torch.randn(self.text_seq_len, self.text_embed_dim, dtype=self.dtype),
            "text_mask": torch.ones(self.text_seq_len, dtype=torch.int64),
        }
        if self.audio_latent_shape is not None:
            sample["audio_latents"] = torch.randn(self.audio_latent_shape, dtype=self.dtype)
        return sample


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    shuffle: bool = True,
    distributed: bool = False,
) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
