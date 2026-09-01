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

__all__ = ["NpzCalibrationReader", "NpzCalibrationWriter", "find_vovnet_nodes_to_exclude"]


def _onnx_input_dtypes(onnx_path):
    graph = onnx.load(onnx_path, load_external_data=False).graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    return {
        value.name: np.dtype(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))
        for value in graph.input
        if value.name not in initializer_names
    }


class NpzCalibrationWriter:
    """Write calibration batches that match an ONNX model's inputs."""

    def __init__(self, output_dir, onnx_path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.glob("batch_*.npz")):
            raise FileExistsError(f"{self.output_dir} already contains calibration batches")
        self.input_dtypes = _onnx_input_dtypes(onnx_path)
        self.count = 0

    def write(self, values):
        missing = self.input_dtypes.keys() - values.keys()
        unexpected = values.keys() - self.input_dtypes.keys()
        if missing or unexpected:
            raise ValueError(
                f"Calibration input mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        batch = {}
        for name, dtype in self.input_dtypes.items():
            value = values[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            batch[name] = np.asarray(value).astype(dtype, copy=False)
        np.savez(self.output_dir / f"batch_{self.count:04d}.npz", **batch)
        self.count += 1


class NpzCalibrationReader(CalibrationDataReader):
    """Stream example-generated NPZ calibration batches."""

    def __init__(self, calibration_dir):
        self.batch_paths = sorted(Path(calibration_dir).glob("batch_*.npz"))
        if not self.batch_paths:
            raise ValueError(f"No calibration batches found in {calibration_dir}")
        self.rewind()

    @staticmethod
    def load(batch_path):
        with np.load(batch_path, allow_pickle=False) as batch:
            return {name: batch[name] for name in batch.files}

    def get_next(self):
        batch_path = next(self._iterator, None)
        return None if batch_path is None else self.load(batch_path)

    def get_first(self):
        return self.load(self.batch_paths[0])

    def rewind(self):
        self._iterator = iter(self.batch_paths)


def find_vovnet_nodes_to_exclude(onnx_path):
    """Find the VoVNet OSA4_5 stage and nodes downstream of FPN lateral_convs."""
    # The evaluator image uses the calibration writer without installing ModelOpt.
    from modelopt.onnx.utils import topologically_sort_graph_nodes

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
