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
import re
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization import quantize
from modelopt.onnx.utils import topologically_sort_graph_nodes


class FileCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, pattern):
        self.batch_paths = sorted(Path(calibration_dir).glob(pattern))
        if not self.batch_paths:
            raise ValueError(f"No {pattern} calibration batches found in {calibration_dir}")
        self.rewind()

    def get_next(self):
        batch_path = next(self._iterator, None)
        return None if batch_path is None else self.load(batch_path)

    def get_first(self):
        return self.load(self.batch_paths[0])

    def rewind(self):
        self._iterator = iter(self.batch_paths)

    def load(self, batch_path):
        raise NotImplementedError


class EncoderCalibrationReader(FileCalibrationReader):
    def __init__(self, calibration_dir):
        super().__init__(calibration_dir, "*.npy")

    def load(self, batch_path):
        return {"img": np.load(batch_path)}


class DecoderCalibrationReader(FileCalibrationReader):
    def __init__(self, calibration_dir, onnx_path):
        graph = onnx.load(onnx_path, load_external_data=False).graph
        self.input_dtypes = {
            value.name: onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            for value in graph.input
        }
        super().__init__(calibration_dir, "*.npz")

    def load(self, batch_path):
        with np.load(batch_path) as batch:
            missing = self.input_dtypes.keys() - batch.files
            if missing:
                raise ValueError(f"{batch_path} is missing decoder inputs: {sorted(missing)}")
            return {
                name: batch[name].astype(dtype, copy=False)
                for name, dtype in self.input_dtypes.items()
            }


def find_encoder_nodes_to_exclude(onnx_path):
    graph = onnx.load(onnx_path, load_external_data=False).graph
    topologically_sort_graph_nodes(graph)

    excluded = set()
    downstream_tensors = set()
    for node in graph.node:
        is_osa = "OSA4_5" in node.name
        is_downstream = any(name in downstream_tensors for name in node.input)
        if is_osa or is_downstream:
            excluded.add(node.name)
        if "lateral_convs" in node.name or (is_downstream and not is_osa):
            downstream_tensors.update(node.output)
    return sorted(excluded)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the FAR3D ONNX models")
    parser.add_argument("encoder_onnx", help="Path to far3d.encoder.onnx")
    parser.add_argument("decoder_onnx", help="Path to far3d.decoder.onnx")
    parser.add_argument("calibration_dir", help="Directory created by prepare_calibration.py")
    parser.add_argument("--quantization-mode", choices=("int8", "fp8"), default="int8")
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
    excluded_nodes = [
        rf"^{re.escape(name)}$" for name in find_encoder_nodes_to_exclude(args.encoder_onnx)
    ]
    print(f"Excluding {len(excluded_nodes)} accuracy-sensitive nodes from quantization")
    quantize(
        onnx_path=args.encoder_onnx,
        quantize_mode=args.quantization_mode,
        calibration_data_reader=EncoderCalibrationReader(encoder_dir),
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
        calibration_data_reader=DecoderCalibrationReader(decoder_dir, args.decoder_onnx),
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        high_precision_dtype="fp16" if args.quantization_mode == "fp8" else "fp32",
        output_path=args.decoder_output,
    )


def main():
    args = parse_args()
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
