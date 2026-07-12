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

"""DMD2 text-to-image collate + dataloader builder for the fastgen example.

Self-contained on **stock** ``nemo_automodel`` (no AutoModel patch required):

* :func:`collate_fn_text_to_image` builds the DMD2 batch directly from the vendored
  :class:`TextToImageDataset` per-item output (``image_latents`` / ``text_embeddings`` /
  ``text_embeddings_mask`` + an optional broadcast ``negative_text_embeddings`` for CFG). It
  deliberately does **not** call the stock ``collate_fn_production``: released
  ``nemo_automodel`` (0.4.0) unconditionally stacks model-specific token keys
  (``clip_tokens`` / ``t5_tokens``) that the Qwen-Image cache does not produce, which would
  raise ``KeyError``. The vendored dataset and this collate are a matched pair, so coupling
  them directly keeps the example self-contained on stock 0.4.0.
* :func:`build_text_to_image_multiresolution_dataloader` builds the vendored dataset + the
  stock bucket sampler (:class:`SequentialBucketSampler`) + a ``StatefulDataLoader``,
  optionally binding a static negative-prompt embedding into the collate via
  ``functools.partial``.
"""

import functools
import logging

import torch
from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from .text_to_image_dataset import TextToImageDataset

logger = logging.getLogger(__name__)


def collate_fn_text_to_image(
    batch: list[dict],
    negative_text_embeddings: torch.Tensor | None = None,
    negative_text_embeddings_mask: torch.Tensor | None = None,
) -> dict:
    """Build the DMD2 text-to-image batch (latents + text embeddings/mask + CFG negatives).

    Args:
        batch: Samples from :class:`TextToImageDataset` (pre-encoded ``prompt_embeds`` path).
        negative_text_embeddings: Optional static negative-prompt embedding of shape
            ``[seq, dim]``. When provided it is broadcast across the batch and attached as
            ``negative_text_embeddings`` (shape ``[B, seq, dim]``); consumed by DMD2 CFG and
            ignored when ``guidance_scale`` is null.
        negative_text_embeddings_mask: Optional mask for the negative embedding.

    Returns:
        Dict with ``image_latents`` / ``text_embeddings`` / ``text_embeddings_mask`` (and, when
        provided, the broadcast ``negative_text_embeddings`` / ``negative_text_embeddings_mask``).
    """
    if "prompt_embeds" not in batch[0]:
        raise NotImplementedError(
            "On-the-fly text encoding is not supported; preprocess to pre-encoded `prompt_embeds`."
        )

    # Bucket sampling yields one resolution per batch.
    resolutions = {tuple(item["crop_resolution"].tolist()) for item in batch}
    assert len(resolutions) == 1, f"Mixed resolutions in batch: {resolutions}"

    # Stack only the keys the DMD2 pipeline consumes, straight from the vendored dataset's
    # per-item output. We do NOT call the stock ``collate_fn_production`` (see module docstring):
    # released nemo_automodel 0.4.0 unconditionally stacks ``clip_tokens`` / ``t5_tokens``, which
    # the Qwen-Image cache omits.
    image_batch = {
        "image_latents": torch.stack([item["latent"] for item in batch]),
        "data_type": "image",
        "text_embeddings": torch.stack([item["prompt_embeds"] for item in batch]),
        "metadata": {
            "prompts": [item["prompt"] for item in batch],
            "image_paths": [item["image_path"] for item in batch],
            "bucket_ids": [item["bucket_id"] for item in batch],
            "aspect_ratios": [item["aspect_ratio"] for item in batch],
            "crop_resolution": torch.stack([item["crop_resolution"] for item in batch]),
            "original_resolution": torch.stack([item["original_resolution"] for item in batch]),
            "crop_offset": torch.stack([item["crop_offset"] for item in batch]),
        },
    }

    # Optional model-specific embedding fields, when a dataset provides them.
    for key in ("pooled_prompt_embeds", "clip_hidden"):
        if key in batch[0]:
            image_batch[key] = torch.stack([item[key] for item in batch])

    # DMD2 text mask: the stock production collate does not stack ``prompt_embeds_mask``.
    if "prompt_embeds_mask" in batch[0]:
        image_batch["text_embeddings_mask"] = torch.stack(
            [item["prompt_embeds_mask"] for item in batch]
        )

    if negative_text_embeddings is not None:
        # Broadcast the static [seq, dim] embedding to [B, seq, dim].
        batch_size = image_batch["image_latents"].shape[0]
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


def _load_negative_prompt_embedding(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(embed, mask)`` from a negative-prompt-embedding file.

    Accepts a dict with an ``embed`` tensor (and an optional ``mask`` /
    ``prompt_embeds_mask`` / ``text_mask``) or a bare embedding tensor; a missing mask
    defaults to all-ones.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    neg_embed = payload["embed"] if isinstance(payload, dict) else payload
    if not torch.is_tensor(neg_embed):
        raise TypeError(
            f"negative_prompt_embedding_path={path!r} payload must contain a tensor "
            f"(or a dict with 'embed' key); got {type(neg_embed).__name__}."
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
            f"negative_prompt_embedding_path={path!r} mask must be a tensor when present; "
            f"got {type(neg_mask).__name__}."
        )
    if neg_mask is None:
        neg_mask = torch.ones(neg_embed.shape[:-1], dtype=torch.long)
    return neg_embed, neg_mask


def build_text_to_image_multiresolution_dataloader(
    *,
    cache_dir: str,
    train_text_encoder: bool = False,
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
    """Build the DMD2 text-to-image multiresolution dataloader for ``TrainDiffusionRecipe``.

    Args:
        cache_dir: Directory with the preprocessed cache (metadata.json, shards, resolution
            subdirs).
        train_text_encoder: If True, the dataset returns tokens instead of embeddings.
        batch_size: Batch size per GPU.
        dp_rank: Data-parallel rank.
        dp_world_size: Data-parallel world size.
        base_resolution: Base resolution for dynamic batch sizing.
        drop_last: Drop incomplete batches.
        shuffle: Shuffle buckets and samples within a bucket.
        dynamic_batch_size: Scale batch size by resolution.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        prefetch_factor: Prefetch batches per worker.
        negative_prompt_embedding_path: Optional ``.pt`` with a static negative-prompt
            embedding, bound into the collate and broadcast to every batch (DMD2 CFG).

    Returns:
        ``(StatefulDataLoader, SequentialBucketSampler)``.
    """
    dataset = TextToImageDataset(cache_dir=cache_dir, train_text_encoder=train_text_encoder)

    # Optional negative-prompt embedding for DMD2 CFG: load once, bind into the collate.
    collate_fn = collate_fn_text_to_image
    if negative_prompt_embedding_path is not None:
        neg_embed, neg_mask = _load_negative_prompt_embedding(negative_prompt_embedding_path)
        logger.info(
            "Loaded negative_prompt_embedding from %s | shape=%s dtype=%s mask_shape=%s",
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

    logger.info(
        "text-to-image dataloader | cache_dir=%s size=%d batches/epoch=%d batch_size=%d dp=%d/%d",
        cache_dir,
        len(dataset),
        len(sampler),
        batch_size,
        dp_rank,
        dp_world_size,
    )
    return dataloader, sampler
