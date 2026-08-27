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

from modelopt.onnx.quantization import quantize
from modelopt.onnx.quantization.calib_utils import NpzCalibrationReader


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Quantize the BEVFormer ONNX model")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--trt-plugins", required=True, nargs="+", type=Path)
    parser.add_argument("--quantization-mode", choices=("int8", "fp8"), default="int8")
    parser.add_argument("--calibration-method", choices=("entropy", "max"))
    parser.add_argument("--max-batches", type=int, default=600)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    for plugin in args.trt_plugins:
        if not plugin.is_file():
            raise FileNotFoundError(plugin)
    if args.output is None:
        args.output = args.onnx.with_name(f"{args.onnx.stem}.{args.quantization_mode}.onnx")

    calibration_method = args.calibration_method or (
        "entropy" if args.quantization_mode == "int8" else "max"
    )
    quantize(
        onnx_path=str(args.onnx),
        quantize_mode=args.quantization_mode,
        calibration_data_reader=NpzCalibrationReader(
            args.calibration_dir, args.onnx, max_batches=args.max_batches
        ),
        calibration_method=calibration_method,
        calibration_eps=["trt", "cuda:0", "cpu"],
        op_types_to_exclude=["MatMul"],
        disable_mha_qdq=True,
        trt_plugins=[str(plugin) for plugin in args.trt_plugins],
        high_precision_dtype="fp16",
        output_path=str(args.output),
    )


if __name__ == "__main__":
    main()
