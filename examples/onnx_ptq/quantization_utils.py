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

import os
import re
import stat
import zipfile
from itertools import islice
from math import prod
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.utils import topologically_sort_graph_nodes

__all__ = [
    "FileCalibrationReader",
    "NpyCalibrationReader",
    "NpzCalibrationReader",
    "find_vovnet_nodes_to_exclude",
]

MAX_CALIBRATION_BATCHES = 512
MAX_CALIBRATION_BYTES = 128 << 20
MAX_ONNX_BYTES = 512 << 20
MAX_TENSOR_ELEMENTS = 32_000_000


def _open_regular_file(path, file_type, max_bytes):
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError(f"Unable to open {path} as a {file_type} file") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{path} is not a regular {file_type} file")
        if file_stat.st_size > max_bytes:
            raise ValueError(f"{path} exceeds the {file_type} byte limit")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _load_onnx_graph(onnx_path, max_onnx_bytes=MAX_ONNX_BYTES):
    if max_onnx_bytes < 1:
        raise ValueError("ONNX byte limit must be positive")
    with _open_regular_file(onnx_path, "ONNX", max_onnx_bytes) as onnx_file:
        model_bytes = onnx_file.read(max_onnx_bytes + 1)
        if len(model_bytes) > max_onnx_bytes:
            raise ValueError(f"{onnx_path} exceeds the ONNX byte limit")
        return onnx.load_model_from_string(model_bytes).graph


def _read_npy_header(array_file, batch_path):
    version = np.lib.format.read_magic(array_file)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(array_file)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(array_file)
    else:
        raise ValueError(f"{batch_path} uses unsupported NPY version {version}")
    if dtype.hasobject:
        raise ValueError(f"{batch_path} contains an object array")
    return tuple(shape), np.dtype(dtype)


def _validate_array(
    shape,
    dtype,
    expected_shape,
    expected_dtype,
    payload_bytes,
    max_elements,
    max_payload_bytes,
    path,
    allow_safe_cast=False,
):
    element_count = prod(shape)
    if element_count > max_elements:
        raise ValueError(f"{path} exceeds the tensor-element limit")
    expected_bytes = element_count * dtype.itemsize
    if expected_bytes > max_payload_bytes:
        raise ValueError(f"{path} exceeds the array byte limit")
    if expected_bytes != payload_bytes:
        raise ValueError(f"{path} array payload size does not match its header")
    converted_bytes = element_count * expected_dtype.itemsize
    if converted_bytes > max_payload_bytes:
        raise ValueError(f"{path} exceeds the converted array byte limit")
    if expected_shape is not None and (
        len(shape) != len(expected_shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(shape, expected_shape)
        )
    ):
        raise ValueError(f"{path} shape {shape} does not match ONNX input shape {expected_shape}")
    if dtype != expected_dtype and not (
        allow_safe_cast and np.can_cast(dtype, expected_dtype, casting="safe")
    ):
        raise ValueError(f"{path} dtype {dtype} does not match ONNX input dtype {expected_dtype}")
    return converted_bytes


def _onnx_inputs(graph):
    inputs = {}
    for value in graph.input:
        tensor_type = value.type.tensor_type
        inputs[value.name] = (
            (
                tuple(
                    dim.dim_value if dim.HasField("dim_value") else None
                    for dim in tensor_type.shape.dim
                )
                if tensor_type.HasField("shape")
                else None
            ),
            np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)),
        )
    return inputs


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


class NpyCalibrationReader(FileCalibrationReader):
    def __init__(
        self,
        calibration_dir,
        onnx_path,
        input_name,
        max_batches=MAX_CALIBRATION_BATCHES,
        max_batch_bytes=MAX_CALIBRATION_BYTES,
        max_tensor_elements=MAX_TENSOR_ELEMENTS,
        max_onnx_bytes=MAX_ONNX_BYTES,
    ):
        if max_batch_bytes < 1 or max_tensor_elements < 1:
            raise ValueError("NPY byte and tensor-element limits must be positive")
        graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
        inputs = _onnx_inputs(graph)
        if input_name not in inputs:
            raise ValueError(f"ONNX model has no input named {input_name}")
        self.input_name = input_name
        self.input_shape, self.input_dtype = inputs[input_name]
        self.max_batch_bytes = max_batch_bytes
        self.max_tensor_elements = max_tensor_elements
        super().__init__(calibration_dir, "*.npy", max_batches)

    def load(self, batch_path):
        with _open_regular_file(batch_path, "NPY", self.max_batch_bytes) as array_file:
            shape, dtype = _read_npy_header(array_file, batch_path)
            payload_bytes = os.fstat(array_file.fileno()).st_size - array_file.tell()
            _validate_array(
                shape,
                dtype,
                self.input_shape,
                self.input_dtype,
                payload_bytes,
                self.max_tensor_elements,
                self.max_batch_bytes,
                batch_path,
            )
            array_file.seek(0)
            return {self.input_name: np.load(array_file, allow_pickle=False)}


class NpzCalibrationReader(FileCalibrationReader):
    def __init__(
        self,
        calibration_dir,
        onnx_path,
        max_batches=MAX_CALIBRATION_BATCHES,
        max_archive_bytes=MAX_CALIBRATION_BYTES,
        max_tensor_elements=MAX_TENSOR_ELEMENTS,
        max_onnx_bytes=MAX_ONNX_BYTES,
        safe_cast_inputs=(),
    ):
        if max_archive_bytes < 1 or max_tensor_elements < 1:
            raise ValueError("NPZ byte and tensor-element limits must be positive")
        graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
        self.inputs = _onnx_inputs(graph)
        self.safe_cast_inputs = set(safe_cast_inputs)
        if not self.safe_cast_inputs <= self.inputs.keys():
            raise ValueError("Safe-cast inputs must be ONNX input names")
        self.max_archive_bytes = max_archive_bytes
        self.max_tensor_elements = max_tensor_elements
        super().__init__(calibration_dir, "*.npz", max_batches)

    def validate_archive(self, batch_file, batch_path):
        with zipfile.ZipFile(batch_file) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > self.max_archive_bytes:
                raise ValueError(f"{batch_path} exceeds the expanded NPZ byte limit")
            element_count = 0
            converted_bytes = 0
            names = set()
            for member in members:
                if member.is_dir() or not member.filename.endswith(".npy"):
                    raise ValueError(f"{batch_path} contains an unexpected archive member")
                name = member.filename[:-4]
                if name in names or name not in self.inputs:
                    raise ValueError(f"{batch_path} contains an unexpected archive member")
                names.add(name)
                with archive.open(member) as array_file:
                    shape, dtype = _read_npy_header(array_file, batch_path)
                    payload_bytes = member.file_size - array_file.tell()
                    input_shape, input_dtype = self.inputs[name]
                    converted_bytes += _validate_array(
                        shape,
                        dtype,
                        input_shape,
                        input_dtype,
                        payload_bytes,
                        self.max_tensor_elements - element_count,
                        self.max_archive_bytes - converted_bytes,
                        batch_path,
                        name in self.safe_cast_inputs,
                    )
                    element_count += prod(shape)
                    while array_file.read(1 << 20):
                        pass
            missing = self.inputs.keys() - names
            if missing:
                raise ValueError(f"{batch_path} input mismatch; missing={sorted(missing)}")

    def load(self, batch_path):
        with _open_regular_file(batch_path, "NPZ", self.max_archive_bytes) as batch_file:
            self.validate_archive(batch_file, batch_path)
            batch_file.seek(0)
            with np.load(batch_file, allow_pickle=False) as batch:
                return {
                    name: batch[name].astype(dtype, copy=False)
                    for name, (_, dtype) in self.inputs.items()
                }


def find_vovnet_nodes_to_exclude(onnx_path, max_onnx_bytes=MAX_ONNX_BYTES):
    """Find the VoVNet OSA4_5 stage and nodes downstream of FPN lateral_convs."""
    graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
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
