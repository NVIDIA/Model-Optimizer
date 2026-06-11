# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/nemo_automodel/components/datasets/diffusion/mock_dataloader.py
# Provides MockTextToImageDataset + build_mock_t2i_dataloader for the smoke config.
# Self-contained on torch (no internal imports rewritten). Original license below.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Mock dataloader for automodel WAN training tests.

This module provides a mock dataset and dataloader that generates random
tensors with the correct shapes for WAN 2.1 training, allowing functional
tests to run without requiring real data.
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class MockWanDataset(Dataset):
    """Mock dataset that generates random data matching WAN 2.1 expected format.

    Args:
        length: Number of samples in the dataset.
        num_channels: Number of latent channels (default: 16 for WAN).
        num_frame_latents: Number of temporal latent frames.
        spatial_h: Height of spatial latents.
        spatial_w: Width of spatial latents.
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings (default: 4096 for UMT5).
        device: Device to place tensors on.
    """

    def __init__(
        self,
        length: int = 1024,
        num_channels: int = 16,
        num_frame_latents: int = 16,
        spatial_h: int = 30,
        spatial_w: int = 52,
        text_seq_len: int = 77,
        text_embed_dim: int = 4096,
        device: str = "cpu",
    ) -> None:
        self.length = max(int(length), 1)
        self.num_channels = num_channels
        self.num_frame_latents = num_frame_latents
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.text_seq_len = text_seq_len
        self.text_embed_dim = text_embed_dim
        self.device = device

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a mock sample with random data.

        Returns:
            Dict containing:
                - text_embeddings: [1, text_seq_len, text_embed_dim]
                - video_latents: [1, num_channels, num_frame_latents, spatial_h, spatial_w]
                - metadata: empty dict
                - file_info: mock file info
        """
        # Generate random video latents: (1, C, T, H, W)
        video_latents = torch.randn(
            1,
            self.num_channels,
            self.num_frame_latents,
            self.spatial_h,
            self.spatial_w,
            dtype=torch.float32,
            device=self.device,
        )

        # Generate random text embeddings: (1, seq_len, embed_dim)
        text_embeddings = torch.randn(
            1,
            self.text_seq_len,
            self.text_embed_dim,
            dtype=torch.float32,
            device=self.device,
        )

        return {
            "text_embeddings": text_embeddings,
            "video_latents": video_latents,
            "metadata": {},
            "file_info": {
                "meta_filename": f"mock_sample_{idx}.meta",
                "original_filename": f"mock_video_{idx}.mp4",
                "original_video_path": f"/mock/path/video_{idx}.mp4",
                "deterministic_latents": True,
                "memory_optimization": False,
                "num_frames": self.num_frame_latents * 4,  # Approximate original frames
            },
        }


def mock_collate_fn(batch):
    """Collate function for mock dataset, matching the real collate_fn behavior."""
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)
    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)

    return {
        "text_embeddings": text_embeddings,
        "video_latents": video_latents,
        "metadata": [item["metadata"] for item in batch],
        "file_info": [item["file_info"] for item in batch],
    }


def build_mock_dataloader(
    *,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
    length: int = 1024,
    num_channels: int = 16,
    num_frame_latents: int = 16,
    spatial_h: int = 30,
    spatial_w: int = 52,
    text_seq_len: int = 77,
    text_embed_dim: int = 4096,
    shuffle: bool = True,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a mock dataloader for WAN training tests.

    This function follows the same interface as build_dataloader but generates
    random data instead of loading from .meta files.

    Args:
        dp_rank: Data parallel rank.
        dp_world_size: Data parallel world size.
        batch_size: Batch size per GPU.
        num_workers: Number of dataloader workers.
        device: Device to place tensors on.
        length: Number of samples in mock dataset.
        num_channels: Number of latent channels (default: 16).
        num_frame_latents: Number of temporal latent frames.
        spatial_h: Height of spatial latents.
        spatial_w: Width of spatial latents.
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings.
        shuffle: Whether to shuffle data.

    Returns:
        Tuple of (DataLoader, DistributedSampler or None).
    """
    dataset = MockWanDataset(
        length=length,
        num_channels=num_channels,
        num_frame_latents=num_frame_latents,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        text_seq_len=text_seq_len,
        text_embed_dim=text_embed_dim,
        device=device,
    )

    sampler = None
    if dp_world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=shuffle,
            drop_last=False,
        )

    use_pin_memory = device == "cpu"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=mock_collate_fn,
        pin_memory=use_pin_memory,
    )

    return dataloader, sampler


class MockTextToImageDataset(Dataset):
    """Mock dataset producing 4D ``image_latents`` for text-to-image training tests.

    Mirrors :class:`MockWanDataset` but emits the keys expected by
    ``collate_fn_text_to_image`` and the Qwen-Image / Flux flow-matching adapters
    (``image_latents`` instead of ``video_latents``).

    Args:
        length: Number of samples in the dataset.
        num_channels: Number of latent channels (default: 16 for the Qwen-Image VAE).
        spatial_h: Height of spatial latents. Must be even for 2x2 patch packing.
        spatial_w: Width of spatial latents. Must be even for 2x2 patch packing.
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings (default: 3584 for Qwen2.5-VL).
        device: Device to place tensors on.
    """

    def __init__(
        self,
        length: int = 1024,
        num_channels: int = 16,
        spatial_h: int = 32,
        spatial_w: int = 32,
        text_seq_len: int = 512,
        text_embed_dim: int = 3584,
        device: str = "cpu",
    ) -> None:
        if spatial_h % 2 or spatial_w % 2:
            raise ValueError(
                f"MockTextToImageDataset requires even spatial dims for 2x2 patch packing, "
                f"got spatial_h={spatial_h}, spatial_w={spatial_w}."
            )
        self.length = max(int(length), 1)
        self.num_channels = num_channels
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.text_seq_len = text_seq_len
        self.text_embed_dim = text_embed_dim
        self.device = device

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a mock sample with random data.

        Returns:
            Dict containing:
                - text_embeddings: [1, text_seq_len, text_embed_dim]
                - negative_text_embeddings: [1, text_seq_len, text_embed_dim]
                  (random; consumed only when downstream CFG is enabled)
                - image_latents: [1, num_channels, spatial_h, spatial_w]
                - metadata: empty dict
                - file_info: mock file info
        """
        image_latents = torch.randn(
            1,
            self.num_channels,
            self.spatial_h,
            self.spatial_w,
            dtype=torch.float32,
            device=self.device,
        )
        text_embeddings = torch.randn(
            1,
            self.text_seq_len,
            self.text_embed_dim,
            dtype=torch.float32,
            device=self.device,
        )
        text_embeddings_mask = torch.ones(
            1,
            self.text_seq_len,
            dtype=torch.long,
            device=self.device,
        )
        # Distinct random tensor so the CFG branch is exercised. Real-data
        # training uses a precomputed empty-prompt embedding instead.
        negative_text_embeddings = torch.randn(
            1,
            self.text_seq_len,
            self.text_embed_dim,
            dtype=torch.float32,
            device=self.device,
        )
        negative_text_embeddings_mask = torch.ones_like(text_embeddings_mask)
        return {
            "text_embeddings": text_embeddings,
            "text_embeddings_mask": text_embeddings_mask,
            "negative_text_embeddings": negative_text_embeddings,
            "negative_text_embeddings_mask": negative_text_embeddings_mask,
            "image_latents": image_latents,
            "metadata": {},
            "file_info": {
                "meta_filename": f"mock_sample_{idx}.meta",
                "original_filename": f"mock_image_{idx}.png",
                "original_image_path": f"/mock/path/image_{idx}.png",
                "deterministic_latents": True,
                "memory_optimization": False,
            },
        }


def mock_t2i_collate_fn(batch):
    """Collate function for the mock T2I dataset (4D ``image_latents``)."""
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)
    image_latents = torch.cat([item["image_latents"] for item in batch], dim=0)

    out = {
        "text_embeddings": text_embeddings,
        "image_latents": image_latents,
        "data_type": "image",
        "metadata": [item["metadata"] for item in batch],
        "file_info": [item["file_info"] for item in batch],
    }
    # Optional negative embeddings — present for CFG-aware DMD2 training; old
    # consumers that ignore the key see no change.
    if "negative_text_embeddings" in batch[0]:
        out["negative_text_embeddings"] = torch.cat(
            [item["negative_text_embeddings"] for item in batch], dim=0
        )
    if "text_embeddings_mask" in batch[0]:
        out["text_embeddings_mask"] = torch.cat(
            [item["text_embeddings_mask"] for item in batch], dim=0
        )
    if "negative_text_embeddings_mask" in batch[0]:
        out["negative_text_embeddings_mask"] = torch.cat(
            [item["negative_text_embeddings_mask"] for item in batch], dim=0
        )
    return out


def build_mock_t2i_dataloader(
    *,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "cpu",
    length: int = 1024,
    num_channels: int = 16,
    spatial_h: int = 32,
    spatial_w: int = 32,
    text_seq_len: int = 512,
    text_embed_dim: int = 3584,
    shuffle: bool = True,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a mock dataloader for text-to-image training tests.

    Sibling of :func:`build_mock_dataloader` for diffusion examples that operate on 4D
    image latents (Qwen-Image, Flux, SD3). Generates random tensors with the correct
    shape — no real data required.

    Args:
        dp_rank: Data parallel rank.
        dp_world_size: Data parallel world size.
        batch_size: Batch size per GPU.
        num_workers: Number of dataloader workers.
        device: Device to place tensors on.
        length: Number of samples in mock dataset.
        num_channels: Number of latent channels (default: 16 for the Qwen-Image VAE).
        spatial_h: Height of spatial latents (must be even).
        spatial_w: Width of spatial latents (must be even).
        text_seq_len: Length of text sequence.
        text_embed_dim: Dimension of text embeddings (default: 3584 for Qwen2.5-VL).
        shuffle: Whether to shuffle data.

    Returns:
        Tuple of (DataLoader, DistributedSampler or None).
    """
    dataset = MockTextToImageDataset(
        length=length,
        num_channels=num_channels,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        text_seq_len=text_seq_len,
        text_embed_dim=text_embed_dim,
        device=device,
    )

    sampler = None
    if dp_world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=shuffle,
            drop_last=False,
        )

    use_pin_memory = device == "cpu"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=mock_t2i_collate_fn,
        pin_memory=use_pin_memory,
    )

    return dataloader, sampler
