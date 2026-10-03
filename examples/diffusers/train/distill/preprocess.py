#!/usr/bin/env python3
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

"""Preprocess raw videos + captions into precomputed latents + text embeddings.

Two-phase pipeline (only one heavy model on GPU at a time):
1. Load text encoder -> encode all captions -> save -> unload text encoder
2. Load VAE encoder -> encode all videos -> save -> unload VAE

Usage:
    python -m distill.preprocess \
        --model_name wan \
        --model_path /path/to/Wan2.2/checkpoint \
        --dataset /path/to/dataset.json \
        --output_dir /path/to/precomputed \
        --video_column video \
        --caption_column caption
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from .interfaces import free_gpu_memory
from .models import get_model_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_dataset_file(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    elif p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    elif p.suffix == ".csv":
        import csv

        with open(p) as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}. Use .json, .jsonl, or .csv")


def load_video(path: str) -> torch.Tensor:
    """Load a video file and return as [C, F, H, W] float tensor in [0, 1]."""
    import torchvision.io as tvio

    frames, _, _info = tvio.read_video(path, output_format="TCHW")
    # frames: [F, C, H, W] uint8 -> [C, F, H, W] float [0, 1]
    video = frames.permute(1, 0, 2, 3).float() / 255.0
    return video


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for distillation training")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model backend name (e.g. 'wan')"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        help="Model variant (e.g. 'ti2v-5B', 't2v-A14B' for Wan)",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to text encoder model (e.g. Gemma dir for LTX-2)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset file (.json/.jsonl/.csv)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for precomputed data"
    )
    parser.add_argument(
        "--video_column", type=str, default="video", help="Column name for video paths"
    )
    parser.add_argument(
        "--caption_column", type=str, default="caption", help="Column name for captions"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load dataset metadata
    samples = load_dataset_file(args.dataset)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get model backend (we only need the inference pipeline)
    _, _, pipeline_cls = get_model_backend(args.model_name, variant=args.model_variant)
    if pipeline_cls is None:
        raise ValueError(f"Model '{args.model_name}' has no inference pipeline for preprocessing")
    pipeline = pipeline_cls()

    # Build a lightweight model config for the pipeline
    from .config import ModelConfig

    model_config = ModelConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        dtype=args.dtype,
    )

    # --- Phase 1: Encode text ---
    logger.info("Phase 1: Encoding captions with text encoder ...")
    pipeline.load_components(model_config, "cuda", dtype)

    captions = [s[args.caption_column] for s in samples]
    # Each entry: (text_emb, text_mask) -- embeddings + attention mask.
    # Backend keys vary (Wan: "context", LTX-2: "prompt_embeds" + "prompt_attention_mask").
    # We normalize to "text_embeds" + "text_mask" in the saved file.
    text_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(0, len(captions), args.batch_size):
        batch_captions = captions[i : i + args.batch_size]
        cached = pipeline.encode_prompts(batch_captions, "", "cuda")
        for emb in cached:
            pos = emb.positive
            if "prompt_embeds" in pos:
                # LTX-2: raw Gemma features + real attention mask
                text_data.append((pos["prompt_embeds"].cpu(), pos["prompt_attention_mask"].cpu()))
            else:
                # Wan / other: first key is the embedding, mask is all-ones
                emb_key = next(iter(pos))
                emb_tensor = pos[emb_key].cpu()
                mask = torch.ones(emb_tensor.shape[0], dtype=torch.int64)
                text_data.append((emb_tensor, mask))
        if (i + args.batch_size) % 100 == 0:
            logger.info(
                f"  Text encoding: {min(i + args.batch_size, len(captions))}/{len(captions)}"
            )

    # Unload text encoder to free memory for VAE
    pipeline.unload_text_encoder()
    logger.info(f"Phase 1 complete: {len(text_data)} captions encoded")

    # --- Phase 2: Encode videos ---
    logger.info("Phase 2: Encoding videos with VAE ...")
    video_paths = [s[args.video_column] for s in samples]

    for i, (video_path, (text_emb, text_mask)) in enumerate(zip(video_paths, text_data)):
        output_path = out_dir / f"sample_{i:06d}.safetensors"
        if output_path.exists():
            continue

        video = load_video(video_path)
        latents = pipeline.encode_videos([video], "cuda")
        latent = latents[0]

        save_file(
            {
                "latents": latent,
                "text_embeds": text_emb,
                "text_mask": text_mask,
            },
            str(output_path),
        )

        if (i + 1) % 100 == 0:
            logger.info(f"  Video encoding: {i + 1}/{len(video_paths)}")

    pipeline.offload_to_cpu()
    free_gpu_memory()
    logger.info(f"Phase 2 complete: {len(video_paths)} videos encoded")
    logger.info(f"Precomputed data saved to {out_dir}")


if __name__ == "__main__":
    main()
