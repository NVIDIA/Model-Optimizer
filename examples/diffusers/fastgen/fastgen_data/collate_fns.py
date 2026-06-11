# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/nemo_automodel/components/datasets/diffusion/collate_fns.py
# Local modifications for the self-contained fastgen example: relative imports of the
# unpatched upstream helpers (``sampler``, ``text_to_video_dataset``) are rewritten to
# absolute ``nemo_automodel`` package paths so this file works against stock upstream;
# ``TextToImageDataset`` is imported from the vendored sibling. Includes the DMD2
# negative-prompt-embedding + ``prompt_embeds_mask`` additions. Original license below.
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
Collate functions and dataloader builders for multiresolution diffusion training.

Supports both image and video pipelines via the FlowMatchingPipeline
expected batch format.
"""

import functools
import logging
from collections.abc import Callable

import torch
from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
from nemo_automodel.components.datasets.diffusion.text_to_video_dataset import (
    TextToVideoDataset,
    collate_optional_video_fields,
)
from torchdata.stateful_dataloader import StatefulDataLoader

from .text_to_image_dataset import TextToImageDataset

logger = logging.getLogger(__name__)


def collate_fn_production(batch: list[dict]) -> dict:
    """Production collate function with verification."""
    # Verify all samples have same resolution
    resolutions = [tuple(item["crop_resolution"].tolist()) for item in batch]
    assert len(set(resolutions)) == 1, f"Mixed resolutions in batch: {set(resolutions)}"

    # Stack tensors
    latents = torch.stack([item["latent"] for item in batch])
    crop_resolutions = torch.stack([item["crop_resolution"] for item in batch])
    original_resolutions = torch.stack([item["original_resolution"] for item in batch])
    crop_offsets = torch.stack([item["crop_offset"] for item in batch])

    # Collect metadata
    prompts = [item["prompt"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    bucket_ids = [item["bucket_id"] for item in batch]
    aspect_ratios = [item["aspect_ratio"] for item in batch]

    output = {
        "latent": latents,
        "crop_resolution": crop_resolutions,
        "original_resolution": original_resolutions,
        "crop_offset": crop_offsets,
        "prompt": prompts,
        "image_path": image_paths,
        "bucket_id": bucket_ids,
        "aspect_ratio": aspect_ratios,
    }

    # Handle text encodings — model-agnostic: stack whichever keys are present
    for key in (
        "clip_hidden",
        "pooled_prompt_embeds",
        "prompt_embeds",
        "prompt_embeds_mask",
        "clip_tokens",
        "t5_tokens",
    ):
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])

    return output


def collate_fn_text_to_image(
    batch: list[dict],
    negative_text_embeddings: torch.Tensor | None = None,
    negative_text_embeddings_mask: torch.Tensor | None = None,
) -> dict:
    """
    Text-to-image collate function that transforms multiresolution batch output
    to match FlowMatchingPipeline expected format.

    Args:
        batch: List of samples from TextToImageDataset.
        negative_text_embeddings: Optional static negative-prompt embedding of
            shape ``[seq, dim]``. When provided, it is broadcast across the
            batch dimension and attached to the output as
            ``negative_text_embeddings`` (shape ``[B, seq, dim]``). Consumed by
            DMD2 CFG; ignored when ``guidance_scale`` is null.

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    # First, use the production collate to stack tensors
    production_batch = collate_fn_production(batch)

    # Keep latent as 4D [B, C, H, W] for image (not video)
    latent = production_batch["latent"]

    # Use "image_latents" key for 4D tensors
    image_batch = {
        "image_latents": latent,
        "data_type": "image",
        "metadata": {
            "prompts": production_batch.get("prompt", []),
            "image_paths": production_batch.get("image_path", []),
            "bucket_ids": production_batch.get("bucket_id", []),
            "aspect_ratios": production_batch.get("aspect_ratio", []),
            "crop_resolution": production_batch.get("crop_resolution"),
            "original_resolution": production_batch.get("original_resolution"),
            "crop_offset": production_batch.get("crop_offset"),
        },
    }

    # Handle text embeddings (pre-encoded vs tokenized)
    if "prompt_embeds" in production_batch:
        # Pre-encoded text embeddings
        image_batch["text_embeddings"] = production_batch["prompt_embeds"]
        # Include optional model-specific fields if present
        if "pooled_prompt_embeds" in production_batch:
            image_batch["pooled_prompt_embeds"] = production_batch["pooled_prompt_embeds"]
        if "clip_hidden" in production_batch:
            image_batch["clip_hidden"] = production_batch["clip_hidden"]
        if "prompt_embeds_mask" in production_batch:
            image_batch["text_embeddings_mask"] = production_batch["prompt_embeds_mask"]
    else:
        # Tokenized - need to encode during training (not supported yet)
        image_batch["t5_tokens"] = production_batch["t5_tokens"]
        image_batch["clip_tokens"] = production_batch["clip_tokens"]
        raise NotImplementedError(
            "On-the-fly text encoding not yet supported. Please use pre-encoded text embeddings in your dataset."
        )

    if negative_text_embeddings is not None:
        # Broadcast the static [seq, dim] embedding to [B, seq, dim].
        batch_size = latent.shape[0]
        neg = negative_text_embeddings
        if neg.dim() == 2:
            neg = neg.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif neg.dim() == 3 and neg.shape[0] != batch_size:
            neg = neg.expand(batch_size, -1, -1).contiguous()
        image_batch["negative_text_embeddings"] = neg
        if negative_text_embeddings_mask is not None:
            neg_mask = negative_text_embeddings_mask
            if neg_mask.dim() == 1:
                neg_mask = neg_mask.unsqueeze(0).expand(batch_size, -1).contiguous()
            elif neg_mask.dim() == 2 and neg_mask.shape[0] != batch_size:
                neg_mask = neg_mask.expand(batch_size, -1).contiguous()
            image_batch["negative_text_embeddings_mask"] = neg_mask

    return image_batch


def _build_multiresolution_dataloader_core(
    *,
    dataset,
    collate_fn: Callable,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    base_resolution: tuple[int, int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> tuple[StatefulDataLoader, SequentialBucketSampler]:
    """Internal helper: create sampler + DataLoader from dataset and collate fn."""
    sampler = SequentialBucketSampler(
        dataset,
        base_batch_size=batch_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle_buckets=shuffle,
        shuffle_within_bucket=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_replicas=dp_world_size,
        rank=dp_rank,
    )

    dataloader = StatefulDataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return dataloader, sampler


def build_text_to_image_multiresolution_dataloader(
    *,
    # TextToImageDataset parameters
    cache_dir: str,
    train_text_encoder: bool = False,
    # Dataloader parameters
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: tuple[int, int] = (256, 256),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    negative_prompt_embedding_path: str | None = None,
) -> tuple[StatefulDataLoader, SequentialBucketSampler]:
    """
    Build a text-to-image multiresolution dataloader for TrainDiffusionRecipe.

    This wraps the existing TextToImageDataset and SequentialBucketSampler
    with a text-to-image collate function.

    Args:
        cache_dir: Directory containing preprocessed cache (metadata.json, shards, and resolution subdirs)
        train_text_encoder: If True, returns tokens instead of embeddings
        batch_size: Batch size per GPU
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        base_resolution: Base resolution for dynamic batch sizing
        drop_last: Drop incomplete batches
        shuffle: Shuffle data
        dynamic_batch_size: Scale batch size by resolution
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler)
    """
    logger.info("Building text-to-image multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  train_text_encoder: {train_text_encoder}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    dataset = TextToImageDataset(
        cache_dir=cache_dir,
        train_text_encoder=train_text_encoder,
    )

    # Optional negative-prompt embedding for DMD2 CFG. Loaded once and bound
    # into the collate via ``functools.partial``; broadcast to every batch.
    collate_fn: Callable = collate_fn_text_to_image
    if negative_prompt_embedding_path is not None:
        payload = torch.load(negative_prompt_embedding_path, map_location="cpu", weights_only=False)
        neg_embed = payload["embed"] if isinstance(payload, dict) else payload
        if not torch.is_tensor(neg_embed):
            raise TypeError(
                f"negative_prompt_embedding_path={negative_prompt_embedding_path!r} payload "
                f"must contain a tensor (or a dict with 'embed' key); got {type(neg_embed).__name__}."
            )
        neg_mask = None
        if isinstance(payload, dict):
            neg_mask = payload.get("mask")
            if neg_mask is None:
                neg_mask = payload.get("prompt_embeds_mask")
            if neg_mask is None:
                neg_mask = payload.get("text_mask")
        if neg_mask is not None and not torch.is_tensor(neg_mask):
            raise TypeError(
                f"negative_prompt_embedding_path={negative_prompt_embedding_path!r} mask "
                f"must be a tensor when present; got {type(neg_mask).__name__}."
            )
        if neg_mask is None:
            neg_mask = torch.ones(neg_embed.shape[:-1], dtype=torch.long)
        logger.info(
            "  Loaded negative_prompt_embedding from %s | shape=%s dtype=%s mask_shape=%s",
            negative_prompt_embedding_path,
            tuple(neg_embed.shape),
            neg_embed.dtype,
            tuple(neg_mask.shape),
        )
        collate_fn = functools.partial(
            collate_fn_text_to_image,
            negative_text_embeddings=neg_embed,
            negative_text_embeddings_mask=neg_mask,
        )

    dataloader, sampler = _build_multiresolution_dataloader_core(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler


def collate_fn_video(batch: list[dict], model_type: str = "wan") -> dict:
    """
    Video-compatible collate function for multiresolution video training.

    Concatenates video_latents (5D) and text_embeddings (3D) along the batch dim,
    matching the format expected by FlowMatchingPipeline with SimpleAdapter.

    Args:
        batch: List of samples from TextToVideoDataset
        model_type: Model type for model-specific field handling

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    # Verify all samples have the same bucket resolution
    resolutions = [tuple(item["bucket_resolution"].tolist()) for item in batch]
    assert len(set(resolutions)) == 1, f"Mixed resolutions in batch: {set(resolutions)}"

    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)

    result = {
        "video_latents": video_latents,
        "text_embeddings": text_embeddings,
        "data_type": "video",
    }

    # Collate model-specific optional fields
    collate_optional_video_fields(batch, result)

    return result


def build_video_multiresolution_dataloader(
    *,
    cache_dir: str,
    model_type: str = "wan",
    device: str = "cpu",
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: tuple[int, int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> tuple[StatefulDataLoader, SequentialBucketSampler]:
    """
    Build a multiresolution video dataloader for TrainDiffusionRecipe.

    Uses TextToVideoDataset with SequentialBucketSampler for bucket-based
    multiresolution video training (e.g. Wan, Hunyuan).

    Args:
        cache_dir: Directory containing preprocessed cache (metadata.json + shards + WxH/*.meta)
        model_type: Model type ("wan", "hunyuan", etc.)
        device: Device to load tensors to
        batch_size: Batch size per GPU
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        base_resolution: Base resolution for dynamic batch sizing
        drop_last: Drop incomplete batches
        shuffle: Shuffle data
        dynamic_batch_size: Scale batch size by resolution
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler)
    """
    logger.info("Building video multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    dataset = TextToVideoDataset(
        cache_dir=cache_dir,
        model_type=model_type,
        device=device,
    )

    collate = functools.partial(collate_fn_video, model_type=model_type)

    dataloader, sampler = _build_multiresolution_dataloader_core(
        dataset=dataset,
        collate_fn=collate,
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler
