# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import sys
from pathlib import Path

from evaluate import get_image_input
from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.onnx_ptq.quantization_utils import NpzCalibrationWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FAR3D encoder calibration batches")
    parser.add_argument("config", help="Path to the FAR3D configuration file")
    parser.add_argument("encoder_onnx")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--sample-skip-interval", type=int, default=20)
    return parser.parse_args()


def build_validation_loader(config_path, num_samples, sample_skip_interval):
    cfg = Config.fromfile(config_path)
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        for dataset_cfg in cfg.data.test:
            dataset_cfg.test_mode = True
        samples_per_gpu = max(
            dataset_cfg.pop("samples_per_gpu", 1) for dataset_cfg in cfg.data.test
        )
        if samples_per_gpu > 1:
            for dataset_cfg in cfg.data.test:
                dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    sample_indices = range(
        sample_skip_interval - 1,
        min(len(dataset), num_samples * sample_skip_interval),
        sample_skip_interval,
    )
    return build_dataloader(
        Subset(dataset, sample_indices),
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )


def main():
    args = parse_args()
    if args.num_samples < 1 or args.sample_skip_interval < 1:
        raise ValueError("Sample count and skip interval must be positive")

    writer = NpzCalibrationWriter(args.output_dir, args.encoder_onnx)
    loader = build_validation_loader(args.config, args.num_samples, args.sample_skip_interval)
    for data in loader:
        writer.write({"img": get_image_input(data)})

    if writer.count != args.num_samples:
        raise RuntimeError(f"Prepared {writer.count} batches; expected {args.num_samples}")
    print(f"Saved {writer.count} calibration batches to {args.output_dir}")


if __name__ == "__main__":
    main()
