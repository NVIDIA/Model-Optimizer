# Adapted from https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq/tools/calib_data_prep.py.
#
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
import ctypes
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.onnx_ptq.quantization_utils import NpzCalibrationWriter


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Prepare BEVFormer calibration batches")
    parser.add_argument("config", help="Path to the BEVFormer TensorRT configuration")
    parser.add_argument("--onnx", required=True, type=Path, help="Original ONNX model")
    parser.add_argument("--engine", required=True, type=Path, help="FP16 TensorRT engine")
    parser.add_argument("--trt-plugin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(arguments)


def build_inputs(data, prev_bev, previous_frame):
    image = data["img"][0].data[0].numpy().astype(np.float32, copy=False)
    metadata = data["img_metas"][0].data[0][0]
    can_bus = metadata["can_bus"]
    same_scene = metadata["scene_token"] == previous_frame["scene_token"]
    position, angle = can_bus[:3].copy(), can_bus[-1].copy()
    if same_scene:
        can_bus[:3] -= previous_frame["position"]
        can_bus[-1] -= previous_frame["angle"]
    else:
        can_bus[:3] = 0
        can_bus[-1] = 0
        prev_bev = np.zeros_like(prev_bev)

    previous_frame.update(scene_token=metadata["scene_token"], position=position, angle=angle)
    return {
        "image": image,
        "prev_bev": prev_bev,
        "use_prev_bev": np.array([same_scene], dtype=np.float32),
        "can_bus": can_bus.astype(np.float32),
        "lidar2img": np.stack(metadata["lidar2img"])[None].astype(np.float32),
    }


def run_feedback(runner, stream, inputs, output_name) -> np.ndarray:
    with torch.cuda.stream(stream), torch.no_grad():
        device_inputs = {
            name: torch.from_numpy(value).cuda(non_blocking=True) for name, value in inputs.items()
        }
        outputs = runner(stream, **device_inputs)
    stream.synchronize()
    return outputs[output_name].detach().cpu().numpy()


def prepare_batches(loader, runner, onnx_path, prev_bev_shape, output_dir, num_samples, stream):
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} must be empty")
        output_dir.rmdir()

    with TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as staging_dir:
        writer = NpzCalibrationWriter(staging_dir, onnx_path)
        prev_bev = np.zeros(prev_bev_shape, dtype=np.float32)
        previous_frame = {"scene_token": None, "position": 0, "angle": 0}
        output_name = runner.resolve_name("bev_embed")
        for data in loader:
            inputs = build_inputs(data, prev_bev, previous_frame)
            prev_bev = run_feedback(runner, stream, inputs, output_name)
            writer.write(inputs)
            if writer.count == num_samples:
                break

        if writer.count != num_samples:
            raise RuntimeError(f"Prepared {writer.count} of {num_samples} requested samples")
        Path(staging_dir).replace(output_dir)
    return writer.count


def main(arguments=None):
    args = parse_args(arguments)
    if args.num_samples < 1 or args.workers < 0:
        raise ValueError("Sample count must be positive and workers must be non-negative")
    for path in (args.onnx, args.engine, args.trt_plugin):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep optional BEVFormer dependencies out of CPU-only module imports.
    from mmcv import Config
    from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset

    config = Config.fromfile(args.config)
    loader = build_dataloader(
        build_dataset(cfg=config.data.quant),
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        shuffle=False,
        dist=False,
    )

    _plugin = ctypes.CDLL(str(args.trt_plugin), mode=ctypes.RTLD_GLOBAL)
    from examples.onnx_ptq.trt_runner import TensorRTRunner

    runner = TensorRTRunner(args.engine)
    saved = prepare_batches(
        loader,
        runner,
        args.onnx,
        (config.bev_h_ * config.bev_w_, 1, config._dim_),
        args.output_dir,
        args.num_samples,
        torch.cuda.Stream(),
    )
    print(f"Saved {saved} calibration batches to {args.output_dir}")


if __name__ == "__main__":
    main()
