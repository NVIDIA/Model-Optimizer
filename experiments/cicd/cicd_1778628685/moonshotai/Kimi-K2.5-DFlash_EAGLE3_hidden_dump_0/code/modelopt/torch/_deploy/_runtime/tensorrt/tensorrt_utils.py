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

import hashlib
import logging

import numpy as np
import onnx
import tensorrt as trt
import torch

from modelopt.onnx.utils import get_batch_size
from modelopt.onnx.utils import get_input_names as get_onnx_input_names

from .constants import TENSORRT_8_MAJOR_VERSION


def is_trt8():
    return int(trt.__version__.split(".", maxsplit=1)[0]) == TENSORRT_8_MAJOR_VERSION


class HostDeviceMem:
    """Simple helper data class to hold host and device memory pointers."""

    def __init__(self, host_mem, device_mem, dtype):
        self.host = host_mem
        self.device = device_mem
        self.dtype = dtype

    def __str__(self):
        return (
            "Host:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\nType:\n"
            + str(self.dtype)
        )

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.host
        del self.device


def get_engine_bytes(engine: trt.tensorrt.ICudaEngine) -> bytes:
    """Return serialized TensorRT engine bytes."""
    return bytearray(engine.serialize())  # type: ignore[return-value]


def load_engine(buffer: bytes, log_level: int = trt.Logger.ERROR) -> trt.tensorrt.ICudaEngine:
    """Load a TensorRT engine from engine data and return."""
    try:
        trt_logger = trt.Logger(log_level)
        with trt.Runtime(trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(buffer), ""
    except Exception as e:
        logging.exception(str(e))
        return None, str(e)


def get_input_names(engine: trt.tensorrt.ICudaEngine) -> list[str]:
    """Gather the input names from an ICudaEngine.

    Args:
        engine: TensorRT engine object.

    Returns:
        List of engine input names.
    """
    input_names = [
        engine.get_binding_name(binding_index)
        for binding_index in range(engine.num_bindings)
        if engine.binding_is_input(binding_index)
    ]
    return input_names


def get_output_names(engine: trt.tensorrt.ICudaEngine) -> list[str]:
    """Gather the output names from an ICudaEngine.

    Args:
        engine: TensorRT engine object.

    Returns:
        List of engine output names.
    """
    output_names = [
        engine.get_binding_name(binding_index)
        for binding_index in range(engine.num_bindings)
        if not engine.binding_is_input(binding_index)
    ]
    return output_names


def get_output_shapes(
    engine: trt.tensorrt.ICudaEngine,
    context: trt.tensorrt.IExecutionContext,
) -> list[list[int]]:
    """Gather the output shapes from an ICudaEngine.

    Args:
        engine: TensorRT engine object.
        context: Current execution context for the inference.

    Returns:
        List of shapes of outputs which are list of integers.
    """
    assert context.all_binding_shapes_specified
    assert context.all_shape_inputs_specified

    output_shapes = []
    for binding_index in range(engine.num_bindings):
        if not engine.binding_is_input(binding_index):
            shape = context.get_binding_shape(binding_index)
            output_shapes.append(shape)
    return output_shapes


def calib_data_generator(onnx_bytes: bytes, input_tensors: list[np.ndarray]):
    """The calibation data generator that yields calibration feed_dict to tensorrt."""
    input_names = get_onnx_input_names(onnx.load_from_string(onnx_bytes))

    batch_size = get_batch_size(onnx.load_from_string(onnx_bytes))
    if not batch_size or batch_size <= 0:
        batch_size = 1
    # If input tensor batch % batch_size != 0, we don't use all input tensors for calibration.
    num_batches = int(input_tensors[0].shape[0] / batch_size)

    for i in range(num_batches):
        feed_dict = {}
        tensor_batch_dim_index = i * batch_size
        for idx, input_name in enumerate(input_names):
            feed_dict[input_name] = input_tensors[idx][
                tensor_batch_dim_index : (tensor_batch_dim_index + batch_size)
            ]
        yield feed_dict


def convert_trt_dtype_to_torch(trt_dtype: trt.tensorrt.DataType) -> torch.dtype:
    """Convert TensorRT data type to torch data type."""
    trt_to_torch_dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }

    assert trt_dtype in trt_to_torch_dtype_map, f"Unsupported TensorRT data type: {trt_dtype}"
    return trt_to_torch_dtype_map[trt_dtype]


def prepend_hash_to_bytes(engine_bytes: bytes) -> bytes:
    """Prepend the engine bytes with the SHA256 hash of the engine bytes
    This has will serve as a unique identifier for the engine and will be used to manage
    TRTSessions in the TRTClient.
    """
    hash_object = hashlib.sha256(engine_bytes)
    hash_object.update(engine_bytes)
    hash_bytes = hash_object.digest()
    engine_bytes = hash_bytes + engine_bytes
    return engine_bytes


def convert_shape_to_string(shape: dict[str, list]) -> str:
    """Convert a shape dictionary to a string.
    For example, if the shape is:
        {
            "input": [1, 3, 224, 224],
            "output": [1, 1000]
        }.
    The output string will be:
        input:1x3x244x244,output:1x1000
    """
    result = ""
    for key, value in shape.items():
        result += f"{key}:{'x'.join(map(str, value))},"
    return result[:-1]
