# Adapted from https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq/tools/calib_data_prep.py.
#
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
import copy
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BEVFormer calibration batches")
    parser.add_argument("config", help="Path to the BEVFormer TensorRT configuration")
    parser.add_argument("--onnx", required=True, type=Path, help="Post-processed ONNX model")
    parser.add_argument("--trt-plugin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def create_session(onnx_path, trt_plugin):
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime TensorRTExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.log_severity_level = 1
    providers = [
        (
            "TensorrtExecutionProvider",
            {"device_id": 0, "trt_extra_plugin_lib_paths": str(trt_plugin)},
        ),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)


def build_inputs(data, prev_bev, previous_frame):
    image = data["img"][0].data[0].numpy().astype(np.float32, copy=False)
    metadata = data["img_metas"][0].data[0][0]
    use_prev_bev = np.array(
        [metadata["scene_token"] == previous_frame["scene_token"]], dtype=np.float32
    )
    previous_frame["scene_token"] = metadata["scene_token"]
    position = copy.deepcopy(metadata["can_bus"][:3])
    angle = copy.deepcopy(metadata["can_bus"][-1])
    if use_prev_bev[0]:
        metadata["can_bus"][:3] -= previous_frame["position"]
        metadata["can_bus"][-1] -= previous_frame["angle"]
    else:
        metadata["can_bus"][:3] = 0
        metadata["can_bus"][-1] = 0
        prev_bev = np.zeros_like(prev_bev)

    inputs = {
        "image": image,
        "prev_bev": prev_bev,
        "use_prev_bev": use_prev_bev,
        "can_bus": metadata["can_bus"].astype(np.float32),
        "lidar2img": np.expand_dims(np.stack(metadata["lidar2img"], axis=0), axis=0).astype(
            np.float32
        ),
    }
    previous_frame["position"] = position
    previous_frame["angle"] = angle
    return inputs


def prepare_batches(loader, session, output_names, prev_bev_shape, output_dir, num_samples):
    prev_bev = np.zeros(prev_bev_shape, dtype=np.float32)
    previous_frame = {"scene_token": None, "position": 0, "angle": 0}
    saved = 0
    with tempfile.TemporaryDirectory(dir=output_dir) as staging_dir:
        staging_dir = Path(staging_dir)
        for data in loader:
            inputs = build_inputs(data, prev_bev, previous_frame)
            outputs = dict(zip(output_names, session.run(output_names, inputs), strict=True))
            np.savez(staging_dir / f"batch_{saved:04d}.npz", **inputs)
            prev_bev = outputs["bev_embed"]
            saved += 1
            if saved == num_samples:
                break

        if saved != num_samples:
            raise RuntimeError(f"Prepared {saved} of {num_samples} requested samples")
        for batch_path in sorted(staging_dir.glob("*.npz")):
            batch_path.replace(output_dir / batch_path.name)
    return saved


def main():
    # Keep optional BEVFormer dependencies out of CPU-only unit test imports.
    from mmcv import Config
    from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset

    args = parse_args()
    if args.num_samples < 1 or args.workers < 0:
        raise ValueError("Sample count must be positive and workers must be non-negative")
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not args.trt_plugin.is_file():
        raise FileNotFoundError(args.trt_plugin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"{args.output_dir} must be empty")

    config = Config.fromfile(args.config)
    dataset = build_dataset(cfg=config.data.quant)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        shuffle=False,
        dist=False,
    )
    session = create_session(args.onnx, args.trt_plugin)
    input_names = {value.name for value in session.get_inputs()}
    if input_names != set(config.input_shapes):
        raise ValueError(
            f"Configuration inputs {sorted(config.input_shapes)} do not match ONNX inputs "
            f"{sorted(input_names)}"
        )
    output_names = [value.name for value in session.get_outputs()]
    if "bev_embed" not in output_names:
        raise ValueError("ONNX model has no bev_embed output")

    saved = prepare_batches(
        loader,
        session,
        output_names,
        (config.bev_h_ * config.bev_w_, 1, config._dim_),
        args.output_dir,
        args.num_samples,
    )
    print(f"Saved {saved} calibration batches to {args.output_dir}")


if __name__ == "__main__":
    main()
