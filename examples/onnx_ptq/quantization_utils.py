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

import re
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

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


class NpzCalibrationReader(FileCalibrationReader):
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
                raise ValueError(f"{batch_path} is missing inputs: {sorted(missing)}")
            return {
                name: batch[name].astype(dtype, copy=False)
                for name, dtype in self.input_dtypes.items()
            }


def find_vovnet_nodes_to_exclude(onnx_path):
    """Find the VoVNet OSA4_5 stage and nodes downstream of FPN lateral_convs."""
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

    if not excluded:
        raise ValueError(f"No accuracy-sensitive VoVNet nodes found in {onnx_path}")
    return [rf"^{re.escape(name)}$" for name in sorted(excluded)]
