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

"""Create a tiny synthetic dataset for testing the preprocess pipeline.

Generates a few short random-noise MP4 videos + a metadata JSON file.

Usage:
    python tests/create_test_data.py --output_dir /tmp/test_videos --num_samples 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def create_test_videos(
    output_dir: str,
    num_samples: int = 4,
    num_frames: int = 17,
    height: int = 128,
    width: int = 128,
    fps: float = 24.0,
) -> None:
    import torchvision.io as tvio

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_dir = out / "videos"
    video_dir.mkdir(exist_ok=True)

    samples = []
    captions = [
        "A cat sitting on a windowsill watching birds outside",
        "Ocean waves crashing on a sandy beach at sunset",
        "A person walking through a snowy forest trail",
        "Colorful fireworks lighting up the night sky",
        "A dog running through a green meadow",
        "Rain falling on a quiet city street at night",
        "A butterfly landing on a bright red flower",
        "Steam rising from a cup of hot coffee",
    ]

    for i in range(num_samples):
        # Random noise video: [F, H, W, C] uint8
        frames = torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)
        video_path = video_dir / f"video_{i:04d}.mp4"
        tvio.write_video(str(video_path), frames, fps=fps)

        samples.append(
            {
                "video": str(video_path),
                "caption": captions[i % len(captions)],
            }
        )

    metadata_path = out / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Created {num_samples} test videos in {video_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/tmp/test_preprocess_data")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    create_test_videos(args.output_dir, args.num_samples, args.num_frames, args.height, args.width)
