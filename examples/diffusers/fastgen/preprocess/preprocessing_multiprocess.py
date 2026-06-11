# Vendored from NVIDIA-NeMo/Automodel @ e42584e3 (Apache-2.0):
#   https://github.com/NVIDIA-NeMo/Automodel/blob/e42584e303397e9bd34643407b8a57d7def88ce9/tools/diffusion/preprocessing_multiprocess.py
# Vendored into the fastgen example so preprocessing runs against stock nemo_automodel
# without the un-packaged AutoModel ``tools/`` tree. The processors import is rewritten to the
# vendored ``.processors`` package; ``MultiTierBucketCalculator`` is imported from stock upstream.
# Original license below.
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
Unified preprocessing tool for images and videos.

Supports:
- Images: FLUX (and other image models)
- Videos: Wan2.1, HunyuanVideo-1.5

Usage:
    # Image preprocessing
    python examples/diffusers/fastgen/preprocess_qwen_image.py image \\
        --image_dir /path/to/images \\
        --output_dir /path/to/cache \\
        --processor flux

    # Video preprocessing
    python examples/diffusers/fastgen/preprocess_qwen_image.py video \\
        --video_dir /path/to/videos \\
        --output_dir /path/to/cache \\
        --processor wan \\
        --resolution_preset 512p

    # List available processors
    python examples/diffusers/fastgen/preprocess_qwen_image.py --list_processors
"""

import argparse
import hashlib
import json
import logging
import os
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from nemo_automodel.components.datasets.diffusion.multi_tier_bucketing import (
    MultiTierBucketCalculator,
)
from PIL import Image
from tqdm import tqdm

from .processors import BaseModelProcessor, ProcessorRegistry, get_caption_loader

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# =============================================================================
# Global worker state (initialized once per process)
# =============================================================================
_worker_models: dict[str, Any] | None = None
_worker_processor: BaseModelProcessor | None = None
_worker_calculator: MultiTierBucketCalculator | None = None
_worker_device: str | None = None
_worker_config: dict[str, Any] | None = None


# =============================================================================
# Common Utility Functions
# =============================================================================


def _get_media_files(media_dir: Path, extensions: set) -> list[Path]:
    """Recursively get all media files with given extensions using os.walk()."""
    media_files = []
    for root, dirs, files in os.walk(media_dir):
        root_path = Path(root)
        for file in files:
            if "." in file:
                ext = file.lower().rsplit(".", 1)[-1]
                if ext in extensions:
                    media_files.append(root_path / file)
    return sorted(media_files)


def _save_metadata_shards(
    all_metadata: list[dict],
    output_dir: Path,
    processor_name: str,
    model_name: str,
    model_type: str,
    shard_size: int,
    extra_fields: dict[str, Any],
    shard_rank: int = 0,
    shard_world: int = 1,
) -> None:
    """Save metadata in shards and write config file.

    When shard_world > 1, the index file and shard filenames are namespaced with
    the rank so that multiple jobs sharing an output directory don't overwrite
    each other. Merge the per-rank index files afterwards with a separate
    script to produce a single unified metadata.json.
    """
    sharded = shard_world > 1
    shard_prefix = f"r{shard_rank:02d}_" if sharded else ""
    index_filename = f"metadata_r{shard_rank:02d}.json" if sharded else "metadata.json"

    shard_files = []
    for chunk_start in range(0, len(all_metadata), shard_size):
        chunk_data = all_metadata[chunk_start : chunk_start + shard_size]
        chunk_idx = chunk_start // shard_size
        shard_file = output_dir / f"metadata_shard_{shard_prefix}s{chunk_idx:04d}.json"
        with open(shard_file, "w") as f:
            json.dump(chunk_data, f, indent=2)
        shard_files.append(shard_file.name)

    metadata = {
        "processor": processor_name,
        "model_name": model_name,
        "model_type": model_type,
        "total_items": len(all_metadata),
        "num_shards": len(shard_files),
        "shard_size": shard_size,
        "shards": shard_files,
        **extra_fields,
    }
    if sharded:
        metadata["shard_rank"] = shard_rank
        metadata["shard_world"] = shard_world

    with open(output_dir / index_filename, "w") as f:
        json.dump(metadata, f, indent=2)


def _print_bucket_distribution(all_metadata: list[dict]) -> None:
    """Print bucket resolution distribution."""
    bucket_counts: dict[str, int] = {}
    for item in all_metadata:
        res = f"{item['bucket_resolution'][0]}x{item['bucket_resolution'][1]}"
        bucket_counts[res] = bucket_counts.get(res, 0) + 1

    logger.info("Bucket distribution:")
    for res in sorted(bucket_counts.keys()):
        logger.info("  %s: %d", res, bucket_counts[res])


# =============================================================================
# Image Preprocessing Functions
# =============================================================================


def _init_worker(processor_name: str, model_name: str, gpu_id: int, max_pixels: int):
    """Initialize worker process with models on assigned GPU."""
    global _worker_models, _worker_processor, _worker_calculator, _worker_device

    # Set CUDA_VISIBLE_DEVICES to isolate this GPU for the worker process.
    # After this, the selected GPU becomes cuda:0 (not cuda:{gpu_id}).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_device = "cuda:0"

    _worker_processor = ProcessorRegistry.get(processor_name)
    _worker_models = _worker_processor.load_models(model_name, _worker_device)
    _worker_calculator = MultiTierBucketCalculator(quantization=64, max_pixels=max_pixels)

    logger.info("Worker initialized on GPU %d", gpu_id)


def _load_all_captions(
    image_files: list[Path],
    caption_field: str = "internvl",
    caption_format: str = "jsonl",
    verbose: bool = True,
) -> dict[str, str]:
    """Pre-load all captions from caption files. Returns filename->caption dict.

    Args:
        image_files: List of image paths to look up captions for.
        caption_field: Field name inside each caption record (e.g. "internvl", "caption").
        caption_format: One of "sidecar", "meta_json", "jsonl". Selects which CaptionLoader to use.
        verbose: If True, log progress and statistics.
    """
    loader = get_caption_loader(caption_format)
    captions, stats = loader.load_captions_with_stats(image_files, caption_field, verbose=verbose)

    if verbose:
        logger.info(
            "Loaded %d captions from %d caption files (format=%s)",
            stats.loaded_count,
            stats.files_parsed,
            caption_format,
        )
        if stats.files_missing > 0:
            logger.info(
                "  %d caption files not found (will use filename fallback)", stats.files_missing
            )
        if stats.captions_missing > 0:
            logger.info("  %d images will use filename as caption", stats.captions_missing)

    return captions


def _process_image(args: tuple) -> dict | None:
    """Process a single image using pre-initialized worker state."""
    image_path, output_dir, verify, caption = args

    try:
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        bucket = _worker_calculator.get_bucket_for_image(orig_width, orig_height)
        target_width, target_height = bucket["resolution"]

        resized_image, crop_offset = _worker_calculator.resize_and_crop(
            image, target_width, target_height, crop_mode="center"
        )

        image_tensor = _worker_processor.preprocess_image(resized_image)
        latent = _worker_processor.encode_image(image_tensor, _worker_models, _worker_device)

        if verify and not _worker_processor.verify_latent(latent, _worker_models, _worker_device):
            logger.warning("Verification failed: %s", image_path)
            return None

        # Use pre-loaded caption with fallback to filename
        if not caption:
            caption = Path(image_path).stem.replace("_", " ")

        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Save cache file
        resolution = f"{target_width}x{target_height}"
        cache_subdir = Path(output_dir) / resolution
        cache_subdir.mkdir(parents=True, exist_ok=True)

        cache_hash = hashlib.md5(f"{Path(image_path).absolute()}_{resolution}".encode()).hexdigest()
        cache_file = cache_subdir / f"{cache_hash}.pt"

        metadata = {
            "original_resolution": (orig_width, orig_height),
            "bucket_resolution": (target_width, target_height),
            "crop_offset": crop_offset,
            "prompt": caption,
            "image_path": str(Path(image_path).absolute()),
            "bucket_id": bucket["id"],
            "aspect_ratio": bucket["aspect_ratio"],
        }

        cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)
        torch.save(cache_data, cache_file)

        return {
            "cache_file": str(cache_file),
            "image_path": str(Path(image_path).absolute()),
            "bucket_resolution": [target_width, target_height],
            "original_resolution": [orig_width, orig_height],
            "prompt": caption,
            "bucket_id": bucket["id"],
            "aspect_ratio": bucket["aspect_ratio"],
            "pixels": target_width * target_height,
            "model_type": _worker_processor.model_type,
        }

    except Exception as e:
        logger.error("Error processing %s: %s", image_path, e)
        logger.debug(traceback.format_exc())
        return None


def _process_shard_on_gpu(
    gpu_id: int,
    image_files: list[Path],
    output_dir: str,
    processor_name: str,
    model_name: str,
    verify: bool,
    caption_cache: dict[str, str],
    max_pixels: int,
) -> list[dict]:
    """Process a shard of images on a specific GPU."""
    _init_worker(processor_name, model_name, gpu_id, max_pixels)

    results = []
    for image_path in tqdm(image_files, desc=f"GPU {gpu_id}", position=gpu_id):
        # Get caption from cache (or None if not found)
        caption = caption_cache.get(image_path.name)
        result = _process_image((str(image_path), output_dir, verify, caption))
        if result:
            results.append(result)

    return results


def preprocess_dataset(
    image_dir: str,
    output_dir: str,
    processor_name: str,
    model_name: str | None = None,
    shard_size: int = 10000,
    verify: bool = False,
    caption_field: str = "internvl",
    caption_format: str = "jsonl",
    max_images: int | None = None,
    max_pixels: int = 256 * 256,
    shard_idx: int = 0,
    shard_count: int = 1,
):
    """
    Preprocess image dataset with one process per GPU.

    Args:
        image_dir: Directory containing images
        output_dir: Output directory for cache
        processor_name: Name of processor to use (e.g., 'flux', 'sdxl')
        model_name: HuggingFace model name (uses processor default if None)
        shard_size: Number of images per metadata shard
        verify: Whether to verify latents can be decoded
        caption_field: Field name inside each caption record (e.g. 'internvl', 'caption')
        caption_format: One of 'sidecar', 'meta_json', 'jsonl' (selects the CaptionLoader)
        max_images: Maximum number of images to process
        max_pixels: Maximum pixels per image
        shard_idx: Rank of this job within a multi-job sweep (0-indexed). Each rank
            processes image_files[shard_idx::shard_count].
        shard_count: Total number of jobs in the sweep. Default 1 (single-job mode).
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get processor and resolve model name
    processor = ProcessorRegistry.get(processor_name)
    if model_name is None:
        model_name = processor.default_model_name

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    logger.info("Processor: %s (%s)", processor_name, processor.model_type)
    logger.info("Model: %s", model_name)
    logger.info("GPUs: %d", num_gpus)
    logger.info("Max pixels: %d", max_pixels)

    # Get all image files
    logger.info("Scanning for images...")
    image_files = _get_media_files(image_dir, IMAGE_EXTENSIONS)

    if max_images is not None:
        image_files = image_files[:max_images]

    if shard_count > 1:
        image_files = image_files[shard_idx::shard_count]
        logger.info("Shard %d/%d: %d images on this rank", shard_idx, shard_count, len(image_files))

    logger.info("Processing %d images", len(image_files))

    if not image_files:
        return

    caption_cache = _load_all_captions(
        image_files, caption_field, caption_format=caption_format, verbose=True
    )

    # Split images across GPUs
    chunks = [image_files[i::num_gpus] for i in range(num_gpus)]

    # Process with one worker per GPU
    all_metadata = []

    with Pool(processes=num_gpus) as pool:
        args = [
            (
                gpu_id,
                chunks[gpu_id],
                str(output_dir),
                processor_name,
                model_name,
                verify,
                caption_cache,
                max_pixels,
            )
            for gpu_id in range(num_gpus)
        ]

        results = pool.starmap(_process_shard_on_gpu, args)

        for gpu_results in results:
            all_metadata.extend(gpu_results)

    # Save metadata
    _save_metadata_shards(
        all_metadata,
        output_dir,
        processor_name,
        model_name,
        processor.model_type,
        shard_size,
        {
            "caption_field": caption_field,
            "caption_format": caption_format,
            "max_pixels": max_pixels,
        },
        shard_rank=shard_idx,
        shard_world=shard_count,
    )

    # Print summary
    logger.info("=" * 50)
    logger.info("COMPLETE: %d/%d images", len(all_metadata), len(image_files))
    logger.info("Output: %s", output_dir)
    _print_bucket_distribution(all_metadata)


# =============================================================================
# Video Preprocessing Functions
# =============================================================================


def _init_video_worker(
    processor_name: str,
    model_name: str,
    gpu_id: int,
    max_pixels: int,
    video_config: dict[str, Any],
):
    """Initialize video worker process with models on assigned GPU."""
    global _worker_models, _worker_processor, _worker_calculator, _worker_device, _worker_config

    # Set CUDA_VISIBLE_DEVICES to isolate this GPU for the worker process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_device = "cuda:0"
    _worker_config = video_config

    _worker_processor = ProcessorRegistry.get(processor_name)
    _worker_models = _worker_processor.load_models(model_name, _worker_device)

    # Create bucket calculator with processor's quantization (8 for video, 64 for image)
    quantization = getattr(_worker_processor, "quantization", 8)
    _worker_calculator = MultiTierBucketCalculator(quantization=quantization, max_pixels=max_pixels)

    logger.info("Video worker initialized on GPU %d (quantization=%d)", gpu_id, quantization)


def _get_video_dimensions(video_path: str) -> tuple[int, int, int]:
    """Get video dimensions and frame count using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return width, height, frame_count


def _extract_evenly_spaced_frames(
    video_path: str,
    num_frames: int,
    target_size: tuple[int, int],
    resize_mode: str = "bilinear",
    center_crop: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract evenly-spaced frames. Returns (frames, source_indices)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate evenly-spaced frame indices
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()

    target_height, target_width = target_size

    # Map resize modes to OpenCV interpolation
    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interp_map.get(resize_mode, cv2.INTER_LINEAR)

    frames = []
    actual_indices = []

    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and optionally center crop
        if center_crop:
            # Calculate scale to cover target area
            scale = max(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

            # Center crop
            start_x = (new_width - target_width) // 2
            start_y = (new_height - target_height) // 2
            frame = frame[start_y : start_y + target_height, start_x : start_x + target_width]
        else:
            # Direct resize (may change aspect ratio)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

        frames.append(frame)
        actual_indices.append(target_idx)

    cap.release()
    return frames, actual_indices


def _frame_to_video_tensor(frame: np.ndarray, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Convert frame (H,W,C) to video tensor (1,C,1,H,W) normalized to [-1,1]."""
    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(frame).float().permute(2, 0, 1)

    # Normalize to [-1, 1]
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5

    # Add batch and temporal dimensions: (C, H, W) -> (1, C, 1, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(2)

    return tensor.to(dtype)


# =============================================================================
# Video Processing Helper Functions
# =============================================================================


def _resolve_video_resolution(
    orig_width: int,
    orig_height: int,
    config: dict[str, Any],
) -> tuple[int, int, str | None, float]:
    """Resolve target resolution. Returns (width, height, bucket_id, aspect_ratio)."""
    target_height = config.get("target_height")
    target_width = config.get("target_width")

    if target_height is not None and target_width is not None:
        # Explicit size: no bucketing
        return target_width, target_height, None, target_width / target_height
    else:
        # Use bucket calculator to find best resolution
        bucket = _worker_calculator.get_bucket_for_image(orig_width, orig_height)
        return (
            bucket["resolution"][0],
            bucket["resolution"][1],
            bucket["id"],
            bucket["aspect_ratio"],
        )


def _save_cache_file(
    cache_data: dict[str, Any],
    output_dir: str,
    resolution: str,
    cache_hash: str,
    output_format: str,
) -> Path:
    """Save cache data to file. Returns path to saved file."""
    cache_subdir = Path(output_dir) / resolution
    cache_subdir.mkdir(parents=True, exist_ok=True)

    if output_format == "meta":
        cache_file = cache_subdir / f"{cache_hash}.meta"
        torch.save(cache_data, cache_file)
    else:  # pt format
        cache_file = cache_subdir / f"{cache_hash}.pt"
        torch.save(cache_data, cache_file)

    return cache_file


def _build_result_dict(
    cache_file: Path,
    video_path: str,
    target_width: int,
    target_height: int,
    orig_width: int,
    orig_height: int,
    caption: str,
    bucket_id: str | None,
    aspect_ratio: float,
    num_frames: int = 1,
    frame_index: int | None = None,
    total_frames_extracted: int | None = None,
    source_frame_index: int | None = None,
) -> dict[str, Any]:
    """Build a result dictionary for a processed video/frame."""
    result = {
        "cache_file": str(cache_file),
        "video_path": str(Path(video_path).absolute()),
        "bucket_resolution": [target_width, target_height],
        "original_resolution": [orig_width, orig_height],
        "num_frames": num_frames,
        "prompt": caption,
        "bucket_id": bucket_id,
        "aspect_ratio": aspect_ratio,
        "pixels": target_width * target_height,
        "model_type": _worker_processor.model_type,
    }

    # Add frame-specific fields if provided
    if frame_index is not None:
        result["frame_index"] = frame_index
    if total_frames_extracted is not None:
        result["total_frames_extracted"] = total_frames_extracted
    if source_frame_index is not None:
        result["source_frame_index"] = source_frame_index

    return result


def _process_video_frames_mode(args: tuple) -> list[dict]:
    """Process video in frames mode - each frame becomes a separate sample."""
    video_path, output_dir, caption, config = args

    try:
        # Get video dimensions
        orig_width, orig_height, total_frames = _get_video_dimensions(video_path)

        # Resolve target resolution (handles bucketing vs explicit size)
        target_width, target_height, bucket_id, aspect_ratio = _resolve_video_resolution(
            orig_width, orig_height, config
        )

        # Extract evenly-spaced frames
        num_frames = config.get("num_frames", 10)
        frames, source_frame_indices = _extract_evenly_spaced_frames(
            video_path,
            num_frames=num_frames,
            target_size=(target_height, target_width),
            resize_mode=config.get("resize_mode", "bilinear"),
            center_crop=config.get("center_crop", True),
        )

        if not frames:
            logger.warning("No frames extracted from %s", video_path)
            return []

        total_frames_extracted = len(frames)

        # Use caption with fallback to filename
        if not caption:
            caption = Path(video_path).stem.replace("_", " ")

        # Encode text ONCE (reuse for all frames)
        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Process each frame individually
        results = []
        deterministic = config.get("deterministic", True)
        output_format = config.get("output_format", "meta")
        resolution = f"{target_width}x{target_height}"

        for frame_idx, (frame, source_idx) in enumerate(zip(frames, source_frame_indices)):
            # Convert single frame to 1-frame video tensor
            video_tensor = _frame_to_video_tensor(frame)

            # Encode with VAE
            latent = _worker_processor.encode_video(
                video_tensor,
                _worker_models,
                _worker_device,
                deterministic=deterministic,
            )

            # Prepare metadata for this frame
            # Note: first_frame and image_embeds are omitted in frames mode
            # (frames mode is intended for t2v training, not i2v conditioning)
            metadata = {
                "original_resolution": (orig_width, orig_height),
                "bucket_resolution": (target_width, target_height),
                "bucket_id": bucket_id,
                "aspect_ratio": aspect_ratio,
                "num_frames": 1,  # Always 1 for frame mode
                "total_original_frames": total_frames,
                "prompt": caption,
                "video_path": str(Path(video_path).absolute()),
                "deterministic": deterministic,
                "mode": "frames",
                "frame_index": frame_idx + 1,  # 1-based index
                "total_frames_extracted": total_frames_extracted,
                "source_frame_index": source_idx,  # 0-based index in source video
            }

            # Get cache data from processor
            cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)

            # Include frame index in hash to ensure unique filenames
            cache_hash = hashlib.md5(
                f"{Path(video_path).absolute()}_{resolution}_frame{frame_idx}".encode()
            ).hexdigest()

            # Save cache file using helper
            cache_file = _save_cache_file(
                cache_data, output_dir, resolution, cache_hash, output_format
            )

            # Build result dict using helper
            results.append(
                _build_result_dict(
                    cache_file=cache_file,
                    video_path=video_path,
                    target_width=target_width,
                    target_height=target_height,
                    orig_width=orig_width,
                    orig_height=orig_height,
                    caption=caption,
                    bucket_id=bucket_id,
                    aspect_ratio=aspect_ratio,
                    num_frames=1,
                    frame_index=frame_idx + 1,
                    total_frames_extracted=total_frames_extracted,
                    source_frame_index=source_idx,
                )
            )

        return results

    except Exception as e:
        logger.error("Error processing %s in frames mode: %s", video_path, e)
        logger.debug(traceback.format_exc())
        return []


def _process_video_video_mode(args: tuple) -> dict | None:
    """Process video in video mode - multi-frame encoding as single sample."""
    video_path, output_dir, caption, config = args

    try:
        # Get video dimensions
        orig_width, orig_height, total_frames = _get_video_dimensions(video_path)

        # Resolve target resolution (handles bucketing vs explicit size)
        target_width, target_height, bucket_id, aspect_ratio = _resolve_video_resolution(
            orig_width, orig_height, config
        )

        # Load video with target resolution
        num_frames = config.get("num_frames")
        target_frames = config.get("target_frames")

        video_tensor, first_frame = _worker_processor.load_video(
            video_path,
            target_size=(target_height, target_width),
            num_frames=target_frames or num_frames,
            resize_mode=config.get("resize_mode", "bilinear"),
            center_crop=config.get("center_crop", True),
        )

        actual_frames = video_tensor.shape[2]  # (1, C, T, H, W)

        # Use caption with fallback to filename
        if not caption:
            caption = Path(video_path).stem.replace("_", " ")

        # Encode video
        deterministic = config.get("deterministic", True)
        latent = _worker_processor.encode_video(
            video_tensor,
            _worker_models,
            _worker_device,
            deterministic=deterministic,
        )

        # Encode text
        text_encodings = _worker_processor.encode_text(caption, _worker_models, _worker_device)

        # Encode first frame for i2v (if processor supports it)
        image_embeds = None
        if hasattr(_worker_processor, "encode_first_frame"):
            image_embeds = _worker_processor.encode_first_frame(
                first_frame, _worker_models, _worker_device
            )

        # Prepare metadata
        metadata = {
            "original_resolution": (orig_width, orig_height),
            "bucket_resolution": (target_width, target_height),
            "bucket_id": bucket_id,
            "aspect_ratio": aspect_ratio,
            "num_frames": actual_frames,
            "total_original_frames": total_frames,
            "prompt": caption,
            "video_path": str(Path(video_path).absolute()),
            "first_frame": first_frame,
            "image_embeds": image_embeds,
            "deterministic": deterministic,
            "mode": config.get("mode", "video"),
        }

        # Get cache data from processor
        cache_data = _worker_processor.get_cache_data(latent, text_encodings, metadata)

        # Save cache file using helper
        output_format = config.get("output_format", "meta")
        resolution = f"{target_width}x{target_height}"
        cache_hash = hashlib.md5(
            f"{Path(video_path).absolute()}_{resolution}_{actual_frames}".encode()
        ).hexdigest()
        cache_file = _save_cache_file(cache_data, output_dir, resolution, cache_hash, output_format)

        # Build result dict using helper
        return _build_result_dict(
            cache_file=cache_file,
            video_path=video_path,
            target_width=target_width,
            target_height=target_height,
            orig_width=orig_width,
            orig_height=orig_height,
            caption=caption,
            bucket_id=bucket_id,
            aspect_ratio=aspect_ratio,
            num_frames=actual_frames,
        )

    except Exception as e:
        logger.error("Error processing %s: %s", video_path, e)
        logger.debug(traceback.format_exc())
        return None


def _process_video(args: tuple) -> list[dict]:
    """Process a single video. Dispatches to frames or video mode based on config."""
    video_path, output_dir, caption, config = args
    mode = config.get("mode", "video")

    if mode == "frames":
        return _process_video_frames_mode(args)
    else:
        # Wrap single result in a list for consistent return type
        result = _process_video_video_mode(args)
        return [result] if result is not None else []


def _process_video_shard_on_gpu(
    gpu_id: int,
    video_files: list[Path],
    output_dir: str,
    processor_name: str,
    model_name: str,
    caption_cache: dict[str, str],
    max_pixels: int,
    video_config: dict[str, Any],
) -> list[dict]:
    """Process a shard of videos on a specific GPU."""
    _init_video_worker(processor_name, model_name, gpu_id, max_pixels, video_config)

    results = []
    for video_path in tqdm(video_files, desc=f"GPU {gpu_id}", position=gpu_id):
        caption = caption_cache.get(video_path.name)
        # _process_video now always returns List[Dict] for consistent handling
        results.extend(_process_video((str(video_path), output_dir, caption, video_config)))

    return results


def preprocess_video_dataset(
    video_dir: str,
    output_dir: str,
    processor_name: str,
    model_name: str | None = None,
    mode: str = "video",
    num_frames: int = 10,
    target_frames: int | None = None,
    resolution_preset: str | None = None,
    max_pixels: int | None = None,
    target_height: int | None = None,
    target_width: int | None = None,
    resize_mode: str = "bilinear",
    center_crop: bool = True,
    deterministic: bool = True,
    output_format: str = "meta",
    caption_format: str = "sidecar",
    caption_field: str = "caption",
    shard_size: int = 10000,
    max_videos: int | None = None,
):
    """
    Preprocess video dataset with one process per GPU.

    Args:
        video_dir: Directory containing videos
        output_dir: Output directory for cache
        processor_name: Name of processor ('wan', 'hunyuan')
        model_name: HuggingFace model name (uses processor default if None)
        mode: Processing mode ('video' or 'frames')
        num_frames: Number of frames for 'frames' mode
        target_frames: Target frame count (for HunyuanVideo 4n+1)
        resolution_preset: Resolution preset ('256p', '512p', '768p', '1024p', '1536p')
        max_pixels: Custom pixel budget (mutually exclusive with resolution_preset)
        target_height: Explicit target height (disables bucketing)
        target_width: Explicit target width (disables bucketing)
        resize_mode: Interpolation mode for resizing
        center_crop: Whether to center crop
        deterministic: Use deterministic latent encoding
        output_format: Output format ('meta' or 'pt')
        caption_format: Caption format ('sidecar', 'meta_json', 'jsonl')
        caption_field: Field name for captions
        shard_size: Number of videos per metadata shard
        max_videos: Maximum number of videos to process
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get processor and resolve model name
    processor = ProcessorRegistry.get(processor_name)
    if model_name is None:
        model_name = processor.default_model_name

    # Determine max_pixels
    if resolution_preset:
        if resolution_preset not in MultiTierBucketCalculator.RESOLUTION_PRESETS:
            raise ValueError(
                f"Unknown preset '{resolution_preset}'. "
                f"Available: {list(MultiTierBucketCalculator.RESOLUTION_PRESETS.keys())}"
            )
        max_pixels = MultiTierBucketCalculator.RESOLUTION_PRESETS[resolution_preset]
    elif max_pixels is None and target_height is None:
        # Default to 512p for videos
        max_pixels = 512 * 512

    # If explicit size given, disable bucketing
    use_bucketing = target_height is None or target_width is None
    if not use_bucketing and max_pixels is None:
        max_pixels = target_height * target_width  # Use explicit size as pixel budget

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    logger.info("Processor: %s (%s)", processor_name, processor.model_type)
    logger.info("Model: %s", model_name)
    logger.info("GPUs: %d", num_gpus)
    logger.info("Mode: %s", mode)
    if use_bucketing:
        logger.info("Max pixels: %d (bucketing enabled)", max_pixels)
        logger.info("Quantization: %d", getattr(processor, "quantization", 8))
    else:
        logger.info("Target size: %dx%d (bucketing disabled)", target_width, target_height)

    if hasattr(processor, "frame_constraint") and processor.frame_constraint:
        logger.info("Frame constraint: %s", processor.frame_constraint)

    # Get all video files
    logger.info("Scanning for videos...")
    video_files = _get_media_files(video_dir, VIDEO_EXTENSIONS)

    if max_videos is not None:
        video_files = video_files[:max_videos]

    logger.info("Found %d videos", len(video_files))

    if not video_files:
        return

    # Load captions using appropriate loader
    logger.info("Loading captions (format: %s, field: %s)...", caption_format, caption_field)
    caption_loader = get_caption_loader(caption_format)
    caption_cache = caption_loader.load_captions(video_files, caption_field)
    logger.info("  Loaded %d captions", len(caption_cache))

    # Video config for workers
    video_config = {
        "mode": mode,
        "num_frames": num_frames,
        "target_frames": target_frames,
        "target_height": target_height if not use_bucketing else None,
        "target_width": target_width if not use_bucketing else None,
        "resize_mode": resize_mode,
        "center_crop": center_crop,
        "deterministic": deterministic,
        "output_format": output_format,
    }

    # Split videos across GPUs
    chunks = [video_files[i::num_gpus] for i in range(num_gpus)]

    # Process with one worker per GPU
    all_metadata = []

    with Pool(processes=num_gpus) as pool:
        args = [
            (
                gpu_id,
                chunks[gpu_id],
                str(output_dir),
                processor_name,
                model_name,
                caption_cache,
                max_pixels,
                video_config,
            )
            for gpu_id in range(num_gpus)
        ]

        results = pool.starmap(_process_video_shard_on_gpu, args)

        for gpu_results in results:
            all_metadata.extend(gpu_results)

    # Save metadata
    _save_metadata_shards(
        all_metadata,
        output_dir,
        processor_name,
        model_name,
        processor.model_type,
        shard_size,
        {
            "caption_format": caption_format,
            "caption_field": caption_field,
            "max_pixels": max_pixels,
            "mode": mode,
            "target_frames": target_frames,
        },
    )

    # Print summary
    logger.info("=" * 50)
    logger.info("COMPLETE: %d/%d videos", len(all_metadata), len(video_files))
    logger.info("Output: %s", output_dir)
    _print_bucket_distribution(all_metadata)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Run image or video preprocessing from the command line."""

    parser = argparse.ArgumentParser(
        description="Unified preprocessing tool for images and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image preprocessing with FLUX
  python examples/diffusers/fastgen/preprocess_qwen_image.py image \\
      --image_dir /data/images --output_dir /cache --processor flux

  # Video preprocessing with Wan2.1
  python examples/diffusers/fastgen/preprocess_qwen_image.py video \\
      --video_dir /data/videos --output_dir /cache --processor wan \\
      --resolution_preset 512p --caption_format sidecar

  # Video preprocessing with HunyuanVideo
  python examples/diffusers/fastgen/preprocess_qwen_image.py video \\
      --video_dir /data/videos --output_dir /cache --processor hunyuan \\
      --target_frames 121 --caption_format meta_json
        """,
    )

    parser.add_argument(
        "--list_processors", action="store_true", help="List available processors and exit"
    )

    subparsers = parser.add_subparsers(dest="command", help="Preprocessing type")

    # ===================
    # Image subcommand
    # ===================
    image_parser = subparsers.add_parser("image", help="Preprocess images")
    image_parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    image_parser.add_argument(
        "--output_dir", type=str, required=True, help="Output cache directory"
    )
    image_parser.add_argument(
        "--processor", type=str, default="qwen_image", help="Processor name (default: qwen_image)"
    )
    image_parser.add_argument(
        "--model_name", type=str, default=None, help="Model name (uses processor default)"
    )
    image_parser.add_argument("--shard_size", type=int, default=10000, help="Metadata shard size")
    image_parser.add_argument("--verify", action="store_true", help="Verify latents can be decoded")
    image_parser.add_argument(
        "--caption_field",
        type=str,
        default="internvl",
        help="Field name inside each caption record (e.g. 'internvl', 'caption')",
    )
    image_parser.add_argument(
        "--caption_format",
        type=str,
        default="jsonl",
        choices=["sidecar", "meta_json", "jsonl"],
        help="Caption file format (default: jsonl for backward compat)",
    )
    image_parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    image_parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Rank within a multi-job sweep (0-indexed). Default 0.",
    )
    image_parser.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="Total jobs in the sweep. When >1, this rank processes image_files[shard_idx::shard_count].",
    )

    # Resolution options (mutually exclusive)
    image_res_group = image_parser.add_mutually_exclusive_group()
    image_res_group.add_argument(
        "--resolution_preset",
        type=str,
        choices=["256p", "512p", "768p", "1024p", "1536p"],
        help="Resolution preset for bucketing",
    )
    image_res_group.add_argument("--max_pixels", type=int, help="Custom max pixel budget")

    # ===================
    # Video subcommand
    # ===================
    video_parser = subparsers.add_parser("video", help="Preprocess videos")
    video_parser.add_argument("--video_dir", type=str, required=True, help="Input video directory")
    video_parser.add_argument(
        "--output_dir", type=str, required=True, help="Output cache directory"
    )
    video_parser.add_argument(
        "--processor",
        type=str,
        required=True,
        choices=["wan", "wan2.1", "hunyuan", "hunyuanvideo", "hunyuanvideo-1.5"],
    )
    video_parser.add_argument(
        "--model_name", type=str, default=None, help="Model name (uses processor default)"
    )
    video_parser.add_argument(
        "--mode", type=str, default="video", choices=["video", "frames"], help="Processing mode"
    )
    video_parser.add_argument(
        "--num_frames", type=int, default=10, help="Frames to extract in 'frames' mode"
    )
    video_parser.add_argument(
        "--target_frames",
        type=int,
        default=None,
        help="Target frame count (e.g., 121 for HunyuanVideo)",
    )

    # Resolution options
    video_res_group = video_parser.add_mutually_exclusive_group()
    video_res_group.add_argument(
        "--resolution_preset",
        type=str,
        choices=["256p", "512p", "768p", "1024p", "1536p"],
        help="Resolution preset (videos bucketed by aspect ratio)",
    )
    video_res_group.add_argument("--max_pixels", type=int, help="Custom pixel budget for bucketing")

    # Explicit size options (disables bucketing)
    video_parser.add_argument(
        "--height", type=int, default=None, help="Explicit height (disables bucketing)"
    )
    video_parser.add_argument(
        "--width", type=int, default=None, help="Explicit width (disables bucketing)"
    )

    video_parser.add_argument(
        "--resize_mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode",
    )
    video_parser.add_argument(
        "--center_crop", action="store_true", default=True, help="Center crop (default: True)"
    )
    video_parser.add_argument(
        "--no_center_crop", dest="center_crop", action="store_false", help="Disable center crop"
    )
    video_parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic encoding (default: True)",
    )
    video_parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic (sampled) encoding",
    )
    video_parser.add_argument(
        "--caption_format",
        type=str,
        default="sidecar",
        choices=["sidecar", "meta_json", "jsonl"],
        help="Caption format",
    )
    video_parser.add_argument(
        "--caption_field", type=str, default="caption", help="Caption field name"
    )
    video_parser.add_argument(
        "--output_format",
        type=str,
        default="meta",
        choices=["meta", "pt"],
        help="Output file format",
    )
    video_parser.add_argument("--shard_size", type=int, default=10000, help="Metadata shard size")
    video_parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")

    args = parser.parse_args()

    # Handle --list_processors
    if args.list_processors:
        logger.info("Available processors:")
        for name in ProcessorRegistry.list_available():
            proc = ProcessorRegistry.get(name)
            quantization = getattr(proc, "quantization", 64)
            logger.info("  %s:", name)
            logger.info("    type: %s", proc.model_type)
            logger.info("    media: image")
            logger.info("    quantization: %d", quantization)
        return

    # Handle subcommands
    if args.command == "image":
        if args.resolution_preset:
            max_pixels = MultiTierBucketCalculator.RESOLUTION_PRESETS[args.resolution_preset]
        elif args.max_pixels:
            max_pixels = args.max_pixels
        else:
            max_pixels = 256 * 256

        preprocess_dataset(
            args.image_dir,
            args.output_dir,
            args.processor,
            args.model_name,
            args.shard_size,
            args.verify,
            args.caption_field,
            args.caption_format,
            args.max_images,
            max_pixels,
            args.shard_idx,
            args.shard_count,
        )

    elif args.command == "video":
        # Validate explicit size args
        if (args.height is None) != (args.width is None):
            parser.error("Both --height and --width must be specified together")

        preprocess_video_dataset(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            processor_name=args.processor,
            model_name=args.model_name,
            mode=args.mode,
            num_frames=args.num_frames,
            target_frames=args.target_frames,
            resolution_preset=args.resolution_preset,
            max_pixels=args.max_pixels,
            target_height=args.height,
            target_width=args.width,
            resize_mode=args.resize_mode,
            center_crop=args.center_crop,
            deterministic=args.deterministic,
            output_format=args.output_format,
            caption_format=args.caption_format,
            caption_field=args.caption_field,
            shard_size=args.shard_size,
            max_videos=args.max_videos,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
