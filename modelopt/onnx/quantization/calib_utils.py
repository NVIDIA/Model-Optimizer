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

"""Provides basic calibration utils."""

import struct
from typing import Union

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

CalibrationDataType = Union[np.ndarray, dict[str, np.ndarray]]  # noqa: UP007


class CalibrationDataProvider(CalibrationDataReader):
    """Calibration data provider class."""

    def __init__(
        self,
        onnx_path: str,
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
        onnx_model = onnx.load(onnx_path)
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
