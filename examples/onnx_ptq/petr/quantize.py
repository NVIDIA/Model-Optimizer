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
    def __init__(self, calibration_dir, onnx_path):
        graph = onnx.load(onnx_path, load_external_data=False).graph
        self.input_dtypes = {
            value.name: onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            for value in graph.input
        }
        self.batch_paths = sorted(Path(calibration_dir).glob("*.npz"))
        if not self.batch_paths:
            raise ValueError(f"No calibration batches found in {calibration_dir}")
        self.rewind()

    def get_next(self):
        batch_path = next(self._iterator, None)
        return None if batch_path is None else self.load(batch_path)

    def get_first(self):
        return self.load(self.batch_paths[0])

    def rewind(self):
        self._iterator = iter(self.batch_paths)

    def load(self, batch_path):
        with np.load(batch_path) as batch:
            missing = self.input_dtypes.keys() - batch.files
            if missing:
                raise ValueError(f"{batch_path} is missing inputs: {sorted(missing)}")
            return {
                name: batch[name].astype(dtype, copy=False)
                for name, dtype in self.input_dtypes.items()
            }


def find_backbone_nodes_to_exclude(onnx_path):
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
    return [rf"^{re.escape(name)}$" for name in sorted(excluded)]


def default_output(onnx_path, precision):
    path = Path(onnx_path)
    return str(path.with_name(f"{path.stem}.{precision}{path.suffix}"))


def quantize_model(onnx_path, calibration_dir, precision, output_path, nodes_to_exclude=()):
    quantize(
        onnx_path=onnx_path,
        quantize_mode=precision,
        calibration_data_reader=FileCalibrationReader(calibration_dir, onnx_path),
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
    parser.add_argument("--quantize-head", action="store_true")
    parser.add_argument("--backbone-output")
    parser.add_argument("--head-output")
    return parser.parse_args()


def main():
    args = parse_args()
    backbone_output = args.backbone_output or default_output(args.backbone_onnx, args.precision)
    excluded = find_backbone_nodes_to_exclude(args.backbone_onnx)
    print(f"Excluding {len(excluded)} accuracy-sensitive backbone nodes")
    quantize_model(
        args.backbone_onnx,
        args.calibration_dir / "backbone",
        args.precision,
        backbone_output,
        excluded,
    )
    if args.quantize_head:
        head_output = args.head_output or default_output(args.head_onnx, args.precision)
        quantize_model(
            args.head_onnx,
            args.calibration_dir / "head",
            args.precision,
            head_output,
        )
    else:
        print("Keeping the head in FP16; use --quantize-head to quantize it")


if __name__ == "__main__":
    main()
