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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.onnx_ptq.quantization_utils import (
    NpzCalibrationReader,
    find_vovnet_nodes_to_exclude,
    temporary_onnx_copy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a VoVNet ONNX image encoder")
    parser.add_argument("onnx_path")
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("--precision", choices=("int8", "fp8"), default="int8")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path)
    output_path = (
        Path(args.output)
        if args.output
        else onnx_path.with_name(f"{onnx_path.stem}.{args.precision}{onnx_path.suffix}")
    )
    if output_path.resolve() == onnx_path.resolve():
        raise ValueError("Output path must differ from the source ONNX path")
    excluded_nodes = find_vovnet_nodes_to_exclude(onnx_path)
    print(f"Excluding {len(excluded_nodes)} accuracy-sensitive VoVNet nodes")
    with temporary_onnx_copy(onnx_path) as temporary_onnx:
        quantize(
            onnx_path=str(temporary_onnx),
            quantize_mode=args.precision,
            calibration_data_reader=NpzCalibrationReader(args.calibration_dir),
            calibration_method="max",
            calibration_eps=["cuda:0", "cpu"],
            nodes_to_exclude=excluded_nodes,
            high_precision_dtype="fp16",
            output_path=str(output_path),
        )


if __name__ == "__main__":
    main()
