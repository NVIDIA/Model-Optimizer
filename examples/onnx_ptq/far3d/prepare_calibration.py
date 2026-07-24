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
import torch
from evaluate import Far3DPipeline
from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from torch.utils.data import Subset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FAR3D calibration batches")
    parser.add_argument("config", help="Path to the FAR3D configuration file")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--encoder-engine")
    parser.add_argument("--decoder-engine")
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
    dataset = Subset(dataset, sample_indices)
    return build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )


class DecoderCalibrationWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.saved = 0

    def __call__(self, inputs):
        batch = {name: value.detach().cpu().numpy() for name, value in inputs.items()}
        np.savez(self.output_dir / f"batch_{self.saved:04d}.npz", **batch)
        self.saved += 1


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if args.sample_skip_interval < 1:
        raise ValueError("--sample-skip-interval must be positive")
    if bool(args.encoder_engine) != bool(args.decoder_engine):
        raise ValueError("--encoder-engine and --decoder-engine must be specified together")

    encoder_dir = args.output_dir / "encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    if any(encoder_dir.glob("*.npy")):
        raise FileExistsError(
            f"{encoder_dir} already contains calibration batches; use an empty directory"
        )

    decoder_writer = pipeline = None
    if args.encoder_engine:
        decoder_dir = args.output_dir / "decoder"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        if any(decoder_dir.glob("*.npz")):
            raise FileExistsError(
                f"{decoder_dir} already contains calibration batches; use an empty directory"
            )
        decoder_writer = DecoderCalibrationWriter(decoder_dir)
        pipeline = Far3DPipeline(
            args.encoder_engine,
            args.decoder_engine,
            decoder_input_callback=decoder_writer,
        )
        stream = torch.cuda.Stream()

    saved = 0
    data_loader = build_validation_loader(args.config, args.num_samples, args.sample_skip_interval)
    for data in data_loader:
        images = data["img"][0].data[0].cpu().permute(0, 1, 3, 4, 2).numpy()
        np.save(encoder_dir / f"batch_{saved:04d}.npy", images)
        if pipeline:
            pipeline(stream, data)
        saved += 1
        if saved == args.num_samples:
            break

    if saved < args.num_samples:
        raise RuntimeError(
            f"Only prepared {saved} of {args.num_samples} requested calibration batches"
        )
    if decoder_writer and decoder_writer.saved != saved:
        raise RuntimeError(f"Prepared {saved} encoder and {decoder_writer.saved} decoder batches")
    print(f"Saved {saved} encoder calibration batches to {encoder_dir}")
    if decoder_writer:
        print(
            f"Saved {decoder_writer.saved} decoder calibration batches to {decoder_writer.output_dir}"
        )


if __name__ == "__main__":
    main()
