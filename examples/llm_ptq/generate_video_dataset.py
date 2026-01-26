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

"""Script to pre-generate processed video dataset for Qwen3-Omni quantization."""

import argparse
import os

import torch
from transformers import AutoProcessor

from modelopt.torch.utils.video_dataset_utils import (
    Qwen3OmniVideoProcessor,
    get_video_dataset_dataloader,
)


def main():
    parser = argparse.ArgumentParser(description="Generate processed video dataset cache")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help="Model name or path for loading the processor",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="finevideo",
        help="Name of the video dataset to process",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset cache",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for processing",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio extraction from videos",
    )
    args = parser.parse_args()

    use_audio = not args.no_audio

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading processor from {args.model_name}...")
    hf_processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Creating Qwen3OmniVideoProcessor (use_audio={use_audio}, dtype={args.dtype})...")
    processor = Qwen3OmniVideoProcessor(
        tokenizer=hf_processor,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        use_audio_in_video=use_audio,
    )

    print(f"Processing {args.num_samples} samples from {args.dataset_name}...")
    print(f"Cache directory: {args.cache_dir}")

    # This will process and save to cache
    _ = get_video_dataset_dataloader(
        dataset_name=args.dataset_name,
        processor=processor,
        batch_size=1,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
    )

    # Cleanup temp files
    processor.cleanup()

    cache_path = os.path.join(args.cache_dir, f"{args.dataset_name}_n{args.num_samples}_processed")
    print(f"\nDone! Processed dataset saved to: {cache_path}")


if __name__ == "__main__":
    main()
