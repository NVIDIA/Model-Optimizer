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

import numpy as np

from modelopt.onnx.quantization import quantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantization_utils import (
    FileCalibrationReader,
    NpzCalibrationReader,
    find_vovnet_nodes_to_exclude,
)


class EncoderCalibrationReader(FileCalibrationReader):
    def __init__(self, calibration_dir, max_batches=512):
        super().__init__(calibration_dir, "*.npy", max_batches)

    def load(self, batch_path):
        return {"img": np.load(batch_path, allow_pickle=False)}


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the FAR3D ONNX models")
    parser.add_argument("--encoder-onnx", required=True, help="Path to far3d.encoder.onnx")
    parser.add_argument("--decoder-onnx", required=True, help="Path to far3d.decoder.onnx")
    parser.add_argument(
        "--calibration-dir", required=True, help="Directory created by prepare_calibration.py"
    )
    parser.add_argument("--quantization-mode", choices=("int8", "fp8"), default="int8")
    parser.add_argument(
        "--max-calibration-batches",
        type=int,
        default=512,
        help="Maximum number of calibration batches to load",
    )
    parser.add_argument("--encoder-output")
    parser.add_argument("--decoder-output")
    parser.add_argument(
        "--fp16-decoder",
        action="store_true",
        help="Skip decoder quantization and use the original mixed-precision decoder",
    )
    return parser.parse_args()


def quantize_encoder(args):
    encoder_dir = Path(args.calibration_dir)
    if (encoder_dir / "encoder").is_dir():
        encoder_dir /= "encoder"
    excluded_nodes = find_vovnet_nodes_to_exclude(args.encoder_onnx)
    print(f"Excluding {len(excluded_nodes)} accuracy-sensitive nodes from quantization")
    quantize(
        onnx_path=args.encoder_onnx,
        quantize_mode=args.quantization_mode,
        calibration_data_reader=EncoderCalibrationReader(
            encoder_dir, max_batches=args.max_calibration_batches
        ),
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        nodes_to_exclude=excluded_nodes,
        high_precision_dtype="fp16",
        output_path=args.encoder_output,
    )


def quantize_decoder(args):
    decoder_dir = Path(args.calibration_dir) / "decoder"
    quantize(
        onnx_path=args.decoder_onnx,
        quantize_mode=args.quantization_mode,
        calibration_data_reader=NpzCalibrationReader(
            decoder_dir, args.decoder_onnx, max_batches=args.max_calibration_batches
        ),
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        high_precision_dtype="fp16" if args.quantization_mode == "fp8" else "fp32",
        output_path=args.decoder_output,
    )


def main():
    args = parse_args()
    if args.max_calibration_batches < 1:
        raise ValueError("--max-calibration-batches must be positive")
    if args.encoder_output is None:
        args.encoder_output = f"far3d.encoder.{args.quantization_mode}.onnx"
    if args.decoder_output is None:
        args.decoder_output = f"far3d.decoder.{args.quantization_mode}.onnx"
    quantize_encoder(args)
    if args.fp16_decoder:
        print("Skipping decoder quantization; use the original mixed-precision decoder ONNX")
    else:
        quantize_decoder(args)


if __name__ == "__main__":
    main()
