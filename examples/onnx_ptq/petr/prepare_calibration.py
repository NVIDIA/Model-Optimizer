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

import torch
from evaluate import build_runtime, get_backbone_inputs, get_head_inputs
from mmcv import DictAction
from mmdet3d.datasets import build_dataloader
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.onnx_ptq.quantization_utils import NpzCalibrationWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PETR calibration batches")
    parser.add_argument("version", choices=("v1", "v2"))
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("backbone_onnx")
    parser.add_argument("head_onnx")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--sample-skip-interval", type=int, default=10)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples < 1 or args.sample_skip_interval < 1:
        raise ValueError("Sample count and skip interval must be positive")

    cfg, dataset, _, model = build_runtime(args.config, args.checkpoint, args.cfg_options)
    stop = min(len(dataset), args.num_samples * args.sample_skip_interval)
    subset = Subset(dataset, range(args.sample_skip_interval - 1, stop, args.sample_skip_interval))
    loader = build_dataloader(
        subset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    backbone_writer = NpzCalibrationWriter(args.output_dir / "backbone", args.backbone_onnx)
    head_writer = NpzCalibrationWriter(args.output_dir / "head", args.head_onnx)
    with torch.no_grad():
        for data in loader:
            images = data["img"][0].data[0].cuda()
            img_metas = data["img_metas"][0].data[0]
            backbone_writer.write(get_backbone_inputs(args.version, model, images, img_metas))
            if head_writer.count == 0:
                features = model.extract_img_feat(images.clone(), img_metas)
                head_writer.write(get_head_inputs(args.version, model, features, img_metas))

    if backbone_writer.count != args.num_samples or head_writer.count != 1:
        raise RuntimeError(
            f"Prepared {backbone_writer.count} backbone and {head_writer.count} head batches; "
            f"expected {args.num_samples} and 1"
        )
    print(
        f"Saved {backbone_writer.count} backbone and one head calibration batch "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
