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

"""Utility to dump imagenet data for calibration."""

import argparse
import os
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image


def main():
    """Prepares calibration data from ImageNet dataset and saves input dictionary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=500,
        help="Number[1-100000] of images to use in calibration.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Whether to save the image tensor data in FP16 format."
    )
    parser.add_argument(
        "--output_path", type=str, default="calib.npy", help="Path to output npy file."
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="zh-plus/tiny-imagenet",
        help="HF dataset card or local ImageNet root dir (expects train/<synset>/*.JPEG).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "timm model name (e.g. tf_efficientnet_b0). When set, derives input_size, mean, "
            "and std from the model's default data config. Overrides --input_size."
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Spatial resolution for calibration images. Ignored when --model_name is set.",
    )

    args = parser.parse_args()

    if args.model_name is not None:
        import timm  # optional dependency: only required when --model_name is set

        data_config = timm.data.resolve_model_data_config(
            timm.create_model(args.model_name, pretrained=False)
        )
        input_size = data_config["input_size"][1]  # (C, H, W) -> H
        mean = list(data_config["mean"])
        std = list(data_config["std"])
    else:
        input_size = args.input_size
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transforms = T.Compose(
        [
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    if os.path.isdir(args.imagenet_path):
        all_images = sorted(Path(args.imagenet_path, "train").rglob("*.JPEG"))
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(all_images), size=args.calibration_data_size, replace=False)
        images = [Image.open(all_images[i]).convert("RGB") for i in chosen]
    else:
        dataset = load_dataset(args.imagenet_path)
        images = dataset["train"][0 : args.calibration_data_size]["image"]

    calib_tensor = np.stack([transforms(image).numpy() for image in images], axis=0)
    if args.fp16:
        calib_tensor = calib_tensor.astype(np.float16)
    np.save(args.output_path, calib_tensor)


if __name__ == "__main__":
    main()
