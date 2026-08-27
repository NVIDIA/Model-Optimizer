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

from modelopt.onnx.quantization import quantize
from modelopt.onnx.quantization.calib_utils import NpzCalibrationReader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.onnx_ptq.quantization_utils import find_vovnet_nodes_to_exclude


def default_output(onnx_path, precision):
    path = Path(onnx_path)
    return str(path.with_name(f"{path.stem}.{precision}{path.suffix}"))


def quantize_model(
    onnx_path,
    calibration_dir,
    precision,
    output_path,
    nodes_to_exclude=(),
    max_calibration_batches=512,
):
    quantize(
        onnx_path=onnx_path,
        quantize_mode=precision,
        calibration_data_reader=NpzCalibrationReader(
            calibration_dir, onnx_path, max_batches=max_calibration_batches
        ),
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        nodes_to_exclude=list(nodes_to_exclude),
        high_precision_dtype="fp16",
        output_path=output_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize PETR ONNX backbone and head models")
    parser.add_argument("--backbone-onnx", required=True)
    parser.add_argument("--head-onnx", required=True)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--precision", choices=("int8", "fp8"), default="int8")
    parser.add_argument(
        "--max-calibration-batches",
        type=int,
        default=512,
        help="Maximum number of calibration batches to load",
    )
    parser.add_argument("--quantize-head", action="store_true")
    parser.add_argument("--backbone-output")
    parser.add_argument("--head-output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_calibration_batches < 1:
        raise ValueError("--max-calibration-batches must be positive")
    backbone_output = args.backbone_output or default_output(args.backbone_onnx, args.precision)
    excluded = find_vovnet_nodes_to_exclude(args.backbone_onnx)
    print(f"Excluding {len(excluded)} accuracy-sensitive backbone nodes")
    quantize_model(
        args.backbone_onnx,
        args.calibration_dir / "backbone",
        args.precision,
        backbone_output,
        nodes_to_exclude=excluded,
        max_calibration_batches=args.max_calibration_batches,
    )
    if args.quantize_head:
        head_output = args.head_output or default_output(args.head_onnx, args.precision)
        quantize_model(
            args.head_onnx,
            args.calibration_dir / "head",
            args.precision,
            head_output,
            max_calibration_batches=args.max_calibration_batches,
        )
    else:
        print("Keeping the head in FP16; use --quantize-head to quantize it")


if __name__ == "__main__":
    main()
