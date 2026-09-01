# Adapted from https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/far3d-trt/tools/test_tensorrt.py
# which was modified from https://github.com/megvii-research/Far3D/blob/5efb9d73a246c39fac79b3cf8c20a8e059611c3f/tools/test.py.
# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Zhiqi Li.
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import tensorrt as trt
import torch

__all__ = ["TensorRTRunner"]

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}
if int(trt.__version__.split(".")[0]) >= 10:
    TRT_TO_TORCH[trt.DataType.INT64] = torch.int64

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def aligned_tensor(shape, dtype, device, alignment=256):
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = math.prod(shape)
    storage = torch.empty(element_count + alignment // element_size, dtype=dtype, device=device)
    offset_bytes = (-storage.data_ptr()) % alignment
    offset = offset_bytes // element_size
    return storage[offset : offset + element_count].view(shape)


def _base_tensor_name(name):
    return name.rsplit(".1", maxsplit=1)[0] if name.endswith(".1") else name


class TensorRTRunner:
    def __init__(self, engine_path, state_names=()):
        with open(engine_path, "rb") as engine_file:
            engine_bytes = engine_file.read()
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create an execution context for {engine_path}")
        self.tensor_names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
        self.input_shapes = {}
        self.output_shapes = {}
        self.tensor_dtypes = {}
        for name in self.tensor_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            self.tensor_dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shapes[name] = shape
            else:
                self.output_shapes[name] = shape

        self.state = {}
        for base_name in state_names:
            name = self.resolve_name(base_name)
            if name in self.input_shapes:
                tensor = aligned_tensor(self.input_shapes[name], self.tensor_dtypes[name], "cuda")
                tensor.zero_()
                self.state[name] = tensor
                self.context.set_tensor_address(name, tensor.data_ptr())
        if self.state:
            torch.cuda.synchronize()

    def resolve_name(self, base_name):
        if base_name in self.tensor_names:
            return base_name
        suffixed_name = f"{base_name}.1"
        return suffixed_name if suffixed_name in self.tensor_names else base_name

    def reset_state(self):
        for tensor in self.state.values():
            tensor.zero_()

    def prepare_input(self, name, inputs):
        input_key = name if name in inputs else _base_tensor_name(name)
        if input_key not in inputs:
            raise KeyError(f"Missing TensorRT input {name}")
        shape = self.input_shapes[name]
        value = inputs[input_key].to(device="cuda", dtype=self.tensor_dtypes[name])
        if tuple(value.shape) != shape:
            if tuple(value.shape[1:]) == shape:
                value = value.squeeze(0)
            elif tuple(shape[1:]) == tuple(value.shape):
                value = value.unsqueeze(0)
            else:
                raise ValueError(
                    f"Input {input_key} has shape {tuple(value.shape)}, expected {shape}"
                )
        return value

    def __call__(self, stream, **inputs):
        input_buffers = []
        for name, shape in self.input_shapes.items():
            if name in self.state:
                continue
            value = self.prepare_input(name, inputs)
            buffer = aligned_tensor(shape, value.dtype, value.device)
            buffer.copy_(value)
            input_buffers.append(buffer)
            self.context.set_tensor_address(name, buffer.data_ptr())

        outputs = {}
        for name, shape in self.output_shapes.items():
            output = aligned_tensor(shape, self.tensor_dtypes[name], "cuda")
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())

        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return outputs
