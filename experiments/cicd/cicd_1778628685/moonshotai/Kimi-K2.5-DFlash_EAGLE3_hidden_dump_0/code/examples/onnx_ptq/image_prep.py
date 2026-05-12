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

import numpy as np
import timm
from datasets import load_dataset


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

    args = parser.parse_args()
    dataset = load_dataset("zh-plus/tiny-imagenet")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = dataset["train"][0 : args.calibration_data_size]["image"]

    calib_tensor = [transforms(image) for image in images]

    calib_tensor = np.stack(calib_tensor, axis=0)
    if args.fp16:
        calib_tensor = calib_tensor.astype(np.float16)
    np.save(args.output_path, calib_tensor)


if __name__ == "__main__":
    main()
