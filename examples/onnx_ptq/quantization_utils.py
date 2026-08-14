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
import zipfile
from itertools import islice
from math import prod
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.utils import topologically_sort_graph_nodes

__all__ = ["FileCalibrationReader", "NpzCalibrationReader", "find_vovnet_nodes_to_exclude"]

MAX_CALIBRATION_BATCHES = 512
MAX_NPZ_BYTES = 128 << 20
MAX_TENSOR_ELEMENTS = 32_000_000


class FileCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, pattern, max_batches=MAX_CALIBRATION_BATCHES):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        batch_paths = list(islice(Path(calibration_dir).glob(pattern), max_batches + 1))
        if not batch_paths:
            raise ValueError(f"No {pattern} calibration batches found in {calibration_dir}")
        if len(batch_paths) > max_batches:
            raise ValueError(
                f"More than {max_batches} calibration batches found in {calibration_dir}"
            )
        self.batch_paths = sorted(batch_paths)
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
    def __init__(
        self,
        calibration_dir,
        onnx_path,
        max_batches=MAX_CALIBRATION_BATCHES,
        max_archive_bytes=MAX_NPZ_BYTES,
        max_tensor_elements=MAX_TENSOR_ELEMENTS,
    ):
        if max_archive_bytes < 1 or max_tensor_elements < 1:
            raise ValueError("NPZ byte and tensor-element limits must be positive")
        graph = onnx.load(onnx_path, load_external_data=False).graph
        self.input_dtypes = {
            value.name: onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            for value in graph.input
        }
        self.max_archive_bytes = max_archive_bytes
        self.max_tensor_elements = max_tensor_elements
        super().__init__(calibration_dir, "*.npz", max_batches)

    def validate_archive(self, batch_path):
        if batch_path.stat().st_size > self.max_archive_bytes:
            raise ValueError(f"{batch_path} exceeds the compressed NPZ byte limit")

        with zipfile.ZipFile(batch_path) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > self.max_archive_bytes:
                raise ValueError(f"{batch_path} exceeds the expanded NPZ byte limit")
            element_count = 0
            for member in members:
                if member.is_dir() or not member.filename.endswith(".npy"):
                    raise ValueError(f"{batch_path} contains an unexpected archive member")
                with archive.open(member) as array_file:
                    version = np.lib.format.read_magic(array_file)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array_file)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array_file)
                    else:
                        raise ValueError(f"{batch_path} uses unsupported NPY version {version}")
                if dtype.hasobject:
                    raise ValueError(f"{batch_path} contains an object array")
                element_count += prod(shape)
                if element_count > self.max_tensor_elements:
                    raise ValueError(f"{batch_path} exceeds the tensor-element limit")

    def load(self, batch_path):
        self.validate_archive(batch_path)
        with np.load(batch_path, allow_pickle=False) as batch:
            missing = self.input_dtypes.keys() - batch.files
            unexpected = set(batch.files) - self.input_dtypes.keys()
            if missing or unexpected:
                raise ValueError(
                    f"{batch_path} input mismatch; missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
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
