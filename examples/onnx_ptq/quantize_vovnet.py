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
import shutil
import sys
import tempfile
from pathlib import Path

from modelopt.onnx.quantization import quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.onnx_ptq.quantization_utils import NpzCalibrationReader, find_vovnet_nodes_to_exclude


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
    output_path = args.output or onnx_path.with_name(
        f"{onnx_path.stem}.{args.precision}{onnx_path.suffix}"
    )
    excluded_nodes = find_vovnet_nodes_to_exclude(onnx_path)
    print(f"Excluding {len(excluded_nodes)} accuracy-sensitive VoVNet nodes")
    # Shape inference updates its input in place; a sibling copy preserves external-data paths.
    temporary_file = tempfile.NamedTemporaryFile(
        dir=onnx_path.parent,
        prefix=f".{onnx_path.stem}.",
        suffix=onnx_path.suffix,
        delete=False,
    )
    temporary_onnx = Path(temporary_file.name)
    temporary_file.close()
    try:
        shutil.copyfile(onnx_path, temporary_onnx)
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
    finally:
        temporary_onnx.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
