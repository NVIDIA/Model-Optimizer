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

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

tensorrt = ModuleType("tensorrt")
tensorrt.Logger = mock.Mock()
tensorrt.TensorIOMode = SimpleNamespace(INPUT=object())
tensorrt.tensorrt = mock.Mock()
tensorrt.__version__ = "11.0"
sys.modules["tensorrt"] = tensorrt

from modelopt.torch._deploy._runtime import trt_client


class _ExecutionContext:
    def __init__(self):
        self.infer_shapes_called = False

    def set_tensor_address(self, *_):
        return True

    def set_input_shape(self, *_):
        return True

    def set_optimization_profile_async(self, *_):
        return True

    def infer_shapes(self):
        self.infer_shapes_called = True
        return []


class _Engine:
    num_io_tensors = 1

    def get_tensor_name(self, _):
        return "input"

    def get_tensor_profile_shape(self, *_):
        return ((1,), (1,), (1,))

    def get_tensor_dtype(self, _):
        return None

    def get_tensor_mode(self, _):
        return trt_client.trt.TensorIOMode.INPUT


def test_initialize_io_tensors_without_deprecated_shape_property(monkeypatch):
    monkeypatch.setattr(
        trt_client.trt, "TensorIOMode", SimpleNamespace(INPUT=object()), raising=False
    )
    context = _ExecutionContext()
    session = trt_client.TRTLocalClient.TRTSession.__new__(trt_client.TRTLocalClient.TRTSession)
    session.execution_context = context
    session.stream = SimpleNamespace(cuda_stream=0)
    session.io_shapes = {}
    tensor = mock.Mock()
    tensor.data_ptr.return_value = 1
    monkeypatch.setattr(trt_client, "convert_trt_dtype_to_torch", lambda _: torch.float32)
    monkeypatch.setattr(trt_client.torch, "empty", lambda *_args, **_kwargs: tensor)

    inputs, outputs = session.initialize_input_output_tensors(_Engine())

    assert inputs == [tensor]
    assert outputs == []
    assert context.infer_shapes_called
