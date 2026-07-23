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
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.quantization import quantize


class Far3DCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir):
        self.batch_paths = sorted(Path(calibration_dir).glob("*.npy"))
        if not self.batch_paths:
            raise ValueError(f"No .npy calibration batches found in {calibration_dir}")
        self.rewind()

    def get_next(self):
        batch_path = next(self._iterator, None)
        if batch_path is None:
            return None
        return {"img": np.load(batch_path)}

    def get_first(self):
        return {"img": np.load(self.batch_paths[0])}

    def rewind(self):
        self._iterator = iter(self.batch_paths)


def find_nodes_to_exclude(onnx_path):
    graph = onnx.load(onnx_path, load_external_data=False).graph
    consumers = defaultdict(list)
    nodes_by_name = {}
    for node in graph.node:
        nodes_by_name[node.name] = node
        for input_name in node.input:
            consumers[input_name].append(node.name)

    excluded = {name for name in nodes_by_name if "OSA4_5" in name}
    queue = deque()
    for name, node in nodes_by_name.items():
        if "lateral_convs" in name:
            for output_name in node.output:
                queue.extend(consumers[output_name])

    while queue:
        name = queue.popleft()
        if not name or name in excluded:
            continue
        excluded.add(name)
        for output_name in nodes_by_name[name].output:
            queue.extend(consumers[output_name])
    return sorted(excluded)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize the FAR3D ONNX encoder to INT8")
    parser.add_argument("onnx_path", help="Path to far3d.encoder.onnx")
    parser.add_argument("calibration_dir", help="Directory created by prepare_calibration.py")
    parser.add_argument("--output-path", default="far3d.encoder.int8.onnx")
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    return parser.parse_args()


def main():
    args = parse_args()
    excluded_nodes = find_nodes_to_exclude(args.onnx_path)
    print(f"Excluding {len(excluded_nodes)} accuracy-sensitive nodes from quantization")
    quantize(
        onnx_path=args.onnx_path,
        quantize_mode="int8",
        calibration_data_reader=Far3DCalibrationReader(args.calibration_dir),
        calibration_method=args.calibration_method,
        calibration_eps=["cuda:0", "cpu"],
        nodes_to_exclude=excluded_nodes,
        high_precision_dtype="fp16",
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
