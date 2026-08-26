# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provides calibration data readers and utilities."""

import os
import stat
import struct
import zipfile
from heapq import nsmallest
from math import prod
from pathlib import Path
from typing import TypeAlias

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.logging_config import logger
from modelopt.onnx.utils import (
    gen_random_inputs,
    get_input_names,
    get_input_shapes,
    parse_shapes_spec,
)

__all__ = [
    "CalibrationDataProvider",
    "CalibrationDataType",
    "NpyCalibrationReader",
    "NpzCalibrationReader",
    "RandomDataProvider",
    "create_directory_calibration_reader",
    "import_scales_from_calib_cache",
]

CalibrationDataType: TypeAlias = np.ndarray | dict[str, np.ndarray]

_MAX_CALIBRATION_BATCHES = 512
_MAX_CALIBRATION_BYTES = 128 << 20
_MAX_ONNX_BYTES = 512 << 20
_MAX_TENSOR_ELEMENTS = 32_000_000


def _open_regular_file(path, file_type, max_bytes):
    path = Path(path)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
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


def _load_onnx_graph(onnx_path, max_onnx_bytes=_MAX_ONNX_BYTES):
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


def _onnx_inputs(graph, calibration_shapes=None):
    initializer_names = {initializer.name for initializer in graph.initializer}
    inputs = {}
    for value in graph.input:
        if value.name in initializer_names:
            continue
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

    shape_overrides = {} if calibration_shapes is None else parse_shapes_spec(calibration_shapes)
    unknown_inputs = shape_overrides.keys() - inputs.keys()
    if unknown_inputs:
        raise ValueError(f"Calibration shapes contain unknown inputs: {sorted(unknown_inputs)}")
    for name, shape in shape_overrides.items():
        expected_shape, dtype = inputs[name]
        if expected_shape is not None and len(shape) != len(expected_shape):
            raise ValueError(f"Calibration shape for {name} has the wrong rank")
        if any(dim < 1 for dim in shape):
            raise ValueError(f"Calibration shape for {name} must contain positive dimensions")
        inputs[name] = (tuple(shape), dtype)
    return inputs


class _FileCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, pattern, max_batches=_MAX_CALIBRATION_BATCHES):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        batch_paths = nsmallest(max_batches, Path(calibration_dir).glob(pattern))
        if not batch_paths:
            raise ValueError(f"No {pattern} calibration batches found in {calibration_dir}")
        self.batch_paths = batch_paths
        self.rewind()

    def get_next(self):
        """Return the next calibration batch."""
        batch_path = next(self._iterator, None)
        return None if batch_path is None else self.load(batch_path)

    def get_first(self):
        """Return the first calibration batch without advancing the iterator."""
        return self.load(self.batch_paths[0])

    def rewind(self):
        """Rewind the reader to the first batch."""
        self._iterator = iter(self.batch_paths)

    def load(self, batch_path):
        raise NotImplementedError


class NpyCalibrationReader(_FileCalibrationReader):
    """Stream single-input calibration batches from a directory of NPY files."""

    def __init__(
        self,
        calibration_dir,
        onnx_path,
        input_name=None,
        max_batches=_MAX_CALIBRATION_BATCHES,
        max_batch_bytes=_MAX_CALIBRATION_BYTES,
        max_tensor_elements=_MAX_TENSOR_ELEMENTS,
        max_onnx_bytes=_MAX_ONNX_BYTES,
        calibration_shapes=None,
    ):
        """Initialize an NPY reader from the model input metadata."""
        if max_batch_bytes < 1 or max_tensor_elements < 1:
            raise ValueError("NPY byte and tensor-element limits must be positive")
        graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
        inputs = _onnx_inputs(graph, calibration_shapes)
        if input_name is None:
            if len(inputs) != 1:
                raise ValueError("NPY calibration requires an ONNX model with exactly one input")
            input_name = next(iter(inputs))
        if input_name not in inputs:
            raise ValueError(f"ONNX model has no input named {input_name}")
        self.input_name = input_name
        self.input_shape, self.input_dtype = inputs[input_name]
        self.max_batch_bytes = max_batch_bytes
        self.max_tensor_elements = max_tensor_elements
        super().__init__(calibration_dir, "*.npy", max_batches)

    def load(self, batch_path):
        """Validate and load one NPY calibration batch."""
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


class NpzCalibrationReader(_FileCalibrationReader):
    """Stream single- or multi-input calibration batches from a directory of NPZ files."""

    def __init__(
        self,
        calibration_dir,
        onnx_path,
        max_batches=_MAX_CALIBRATION_BATCHES,
        max_archive_bytes=_MAX_CALIBRATION_BYTES,
        max_tensor_elements=_MAX_TENSOR_ELEMENTS,
        max_onnx_bytes=_MAX_ONNX_BYTES,
        safe_cast_inputs=(),
        calibration_shapes=None,
    ):
        """Initialize an NPZ reader from the model input metadata."""
        if max_archive_bytes < 1 or max_tensor_elements < 1:
            raise ValueError("NPZ byte and tensor-element limits must be positive")
        graph = _load_onnx_graph(onnx_path, max_onnx_bytes)
        self.inputs = _onnx_inputs(graph, calibration_shapes)
        self.safe_cast_inputs = set(safe_cast_inputs)
        if not self.safe_cast_inputs <= self.inputs.keys():
            raise ValueError("Safe-cast inputs must be ONNX input names")
        self.max_archive_bytes = max_archive_bytes
        self.max_tensor_elements = max_tensor_elements
        super().__init__(calibration_dir, "*.npz", max_batches)

    def _validate_archive(self, batch_file, batch_path):
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
        """Validate and load one NPZ calibration batch."""
        with _open_regular_file(batch_path, "NPZ", self.max_archive_bytes) as batch_file:
            self._validate_archive(batch_file, batch_path)
            batch_file.seek(0)
            with np.load(batch_file, allow_pickle=False) as batch:
                return {
                    name: batch[name].astype(dtype, copy=False)
                    for name, (_, dtype) in self.inputs.items()
                }


def create_directory_calibration_reader(
    calibration_dir,
    onnx_path,
    *,
    max_batches=_MAX_CALIBRATION_BATCHES,
    calibration_shapes=None,
    safe_cast_inputs=(),
):
    """Create a bounded calibration reader for a directory of NPY or NPZ batches."""
    calibration_dir = Path(calibration_dir)
    has_npy = next(calibration_dir.glob("*.npy"), None) is not None
    has_npz = next(calibration_dir.glob("*.npz"), None) is not None
    if has_npy and has_npz:
        raise ValueError(f"Calibration directory {calibration_dir} mixes NPY and NPZ batches")
    if has_npy:
        if safe_cast_inputs:
            raise ValueError("Safe-cast inputs are supported only for NPZ calibration batches")
        return NpyCalibrationReader(
            calibration_dir,
            onnx_path,
            max_batches=max_batches,
            calibration_shapes=calibration_shapes,
        )
    if has_npz:
        return NpzCalibrationReader(
            calibration_dir,
            onnx_path,
            max_batches=max_batches,
            calibration_shapes=calibration_shapes,
            safe_cast_inputs=safe_cast_inputs,
        )
    raise ValueError(f"No NPY or NPZ calibration batches found in {calibration_dir}")


class CalibrationDataProvider(CalibrationDataReader):
    """Calibration data provider class."""

    def __init__(
        self,
        onnx_path: str | onnx.ModelProto,
        calibration_data: CalibrationDataType,
        calibration_shapes: str | None = None,
    ):
        """Initializes the data provider class with the calibration data iterator.

        Args:
            onnx_path: Path to the ONNX model.
            calibration_data: Numpy data to calibrate the model.
                Ex. If a model has input shapes like {"sample": (2, 4, 64, 64), "timestep": (1,),
                "encoder_hidden_states": (2, 16, 768)}, the calibration data should have dictionary
                of tensors with shapes like {"sample": (1024, 4, 64, 64), "timestep": (512,),
                "encoder_hidden_states": (1024, 16, 768)} to calibrate with 512 samples.
            calibration_shapes: A string representing the shape of each input tensors for one calibration step.
                If the shape is not provided for an input tensor, the shape is inferred from the onnx model directly,
                with all the unknown dims filled with 1.
        """
        logger.info("Setting up CalibrationDataProvider for calibration")
        # Tensor data is not required to generate the calibration data
        # So even if the model has external data, we don't need to load them here
        onnx_model = onnx.load(onnx_path) if isinstance(onnx_path, str) else onnx_path
        input_names = get_input_names(onnx_model)
        input_shapes = {} if calibration_shapes is None else parse_shapes_spec(calibration_shapes)
        inferred_input_shapes = get_input_shapes(onnx_model)
        for name in input_names:
            if name not in input_shapes:
                input_shapes[name] = inferred_input_shapes[name]
                logger.debug(f"Inferred shape for {name}: {inferred_input_shapes[name]}")

        # Validate calibration data against expected inputs by the model
        if isinstance(calibration_data, np.ndarray):
            assert len(input_names) == 1, "Calibration data has only one tensor."
            calibration_data = {input_names[0]: calibration_data}
            logger.debug(
                f"Single tensor calibration data shape: {calibration_data[input_names[0]].shape}"
            )
        elif isinstance(calibration_data, dict):
            assert len(input_names) == len(calibration_data), (
                "Model input count and calibration data doesn't match."
            )
            for input_name in input_names:
                assert input_name in calibration_data
            logger.debug(f"Multi-tensor calibration data with {len(calibration_data)} inputs")
        else:
            raise ValueError(
                f"calibration data must be numpy array or dict, got {type(calibration_data)}"
            )

        # Create list of model inputs with appropriate batch size
        n_itr = int(calibration_data[input_names[0]].shape[0] / input_shapes[input_names[0]][0])
        logger.debug(f"Creating {n_itr} calibration iterations")
        self.calibration_data_list = [{} for _ in range(n_itr)]
        for input_name in input_names:
            for idx, calib_data in enumerate(
                np.array_split(calibration_data[input_name], n_itr, axis=0)
            ):
                self.calibration_data_list[idx][input_name] = calib_data

        self.calibration_data_reader = iter(self.calibration_data_list)

    def get_next(self):
        """Returns the next available calibration input from the reader."""
        return next(self.calibration_data_reader, None)

    def get_first(self):
        """Returns the first calibration input from the reader without incrementing the iterator.

        This is useful when doing a test run for the session.
        """
        assert len(self.calibration_data_list) > 0, "Calibration data list is empty!"
        return self.calibration_data_list[0]

    def rewind(self):
        """Rewinds the data reader to the first index."""
        self.calibration_data_reader = iter(self.calibration_data_list)


class RandomDataProvider(CalibrationDataReader):
    """Calibration data reader class with random data provider."""

    def __init__(self, onnx_model: str | onnx.ModelProto, calibration_shapes: str | None = None):
        """Initializes the data reader class with random calibration data."""
        logger.info("Initializing RandomDataProvider")
        if isinstance(onnx_model, str):
            onnx_path = onnx_model
            logger.debug(
                f"Loading ONNX model from: {onnx_path} to read the input shapes for RandomDataProvider"
            )
            # Tensor data is not required to generate the calibration data
            # So even if the model has external data, we don't need to load them here
            onnx_model = onnx.load(onnx_path)
        self.calibration_data_list: list[dict[str, np.ndarray]] = [
            gen_random_inputs(onnx_model, calibration_shapes)
        ]
        self.calibration_data_reader = iter(self.calibration_data_list)

    def get_next(self):
        """Returns the next available calibration input from the reader."""
        return next(self.calibration_data_reader, None)

    def get_first(self):
        """Returns the first calibration input from the reader without incrementing the iterator.

        This is useful when doing a test run for the session.
        """
        assert len(self.calibration_data_list) > 0, "Calibration data list is empty!"
        return self.calibration_data_list[0]

    def rewind(self):
        """Rewinds the data reader to the first index."""
        self.calibration_data_reader = iter(self.calibration_data_list)


def import_scales_from_calib_cache(cache_path: str) -> dict[str, float]:
    """Reads TensorRT calibration cache and returns as dictionary.

    Args:
        cache_path: Calibration cache path.

    Returns:
        Dictionary with scales in the format {tensor_name: float_scale}.
    """
    logger.info(f"Importing scales from calibration cache: {cache_path}")
    with open(cache_path) as f:
        scales_dict = {}
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i > 0:  # Skips the first line (i.e., TRT-8501-EntropyCalibration2)
                layer_name, hex_value = line.replace("\n", "").split(": ")
                try:
                    scale = struct.unpack("!f", bytes.fromhex(hex_value))[0]
                    scales_dict[layer_name + "_scale"] = scale
                    logger.debug(f"Imported scale for {layer_name}: {scale}")
                except Exception as e:
                    logger.error(f"Failed to parse scale for tensor {layer_name}: {e!s}")
                    raise ValueError(f"Scale value for tensor {layer_name} was not found!")

        return scales_dict
