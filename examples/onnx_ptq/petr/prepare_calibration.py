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
from pathlib import Path

import numpy as np
import onnx
import torch
from evaluate import PETRPipeline, build_runtime
from mmdet3d.datasets import build_dataloader
from torch.utils.data import Subset


class CalibrationWriter:
    def __init__(self, output_dir, onnx_path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.glob("*.npz")):
            raise FileExistsError(f"{self.output_dir} already contains calibration batches")
        graph = onnx.load(onnx_path, load_external_data=False).graph
        self.dtypes = {
            value.name: onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            for value in graph.input
        }
        self.saved = 0

    def __call__(self, values):
        missing = self.dtypes.keys() - values.keys()
        unexpected = values.keys() - self.dtypes.keys()
        if missing or unexpected:
            raise ValueError(
                f"Calibration input mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        batch = {
            name: values[name].detach().cpu().numpy().astype(dtype, copy=False)
            for name, dtype in self.dtypes.items()
        }
        np.savez(self.output_dir / f"batch_{self.saved:04d}.npz", **batch)
        self.saved += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PETR ONNX calibration batches")
    parser.add_argument("version", choices=("v1", "v2"))
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("backbone_onnx")
    parser.add_argument("head_onnx")
    parser.add_argument("backbone_engine")
    parser.add_argument("head_engine")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--sample-skip-interval", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples < 1 or args.sample_skip_interval < 1:
        raise ValueError("Sample count and skip interval must be positive")
    cfg, dataset, _, model = build_runtime(args.config, args.checkpoint)
    stop = min(len(dataset), args.num_samples * args.sample_skip_interval)
    subset = Subset(dataset, range(args.sample_skip_interval - 1, stop, args.sample_skip_interval))
    loader = build_dataloader(
        subset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    backbone_writer = CalibrationWriter(args.output_dir / "backbone", args.backbone_onnx)
    head_writer = CalibrationWriter(args.output_dir / "head", args.head_onnx)
    pipeline = PETRPipeline(
        args.version,
        model,
        args.backbone_engine,
        args.head_engine,
        backbone_input_callback=backbone_writer,
        head_input_callback=head_writer,
    )
    stream = torch.cuda.Stream()
    for data in loader:
        pipeline(stream, data)
    if backbone_writer.saved != args.num_samples or head_writer.saved != args.num_samples:
        raise RuntimeError(
            f"Prepared {backbone_writer.saved} backbone and {head_writer.saved} head batches; "
            f"expected {args.num_samples}"
        )
    print(f"Saved {args.num_samples} calibration batches to {args.output_dir}")


if __name__ == "__main__":
    main()
