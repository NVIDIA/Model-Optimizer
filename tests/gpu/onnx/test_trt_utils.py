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

import ctypes.util
import logging

import onnx
import pytest

from modelopt.onnx import trt_utils

pytest.importorskip("tensorrt")


@pytest.mark.parametrize("log_level", [logging.WARNING, logging.DEBUG])
def test_get_custom_layers_with_plugin_library(caplog, log_level):
    """Loading a plugin library must preserve ONNX parsing at either log level."""
    plugin_library = ctypes.util.find_library("nvinfer_plugin")
    assert plugin_library is not None
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            "identity",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
        ir_version=10,
    )

    with caplog.at_level(log_level, logger=trt_utils.logger.name):
        custom_layers, tensor_info = trt_utils.get_custom_layers(model, [plugin_library])

    assert custom_layers == []
    assert tensor_info == {
        name: {"shape": [1], "dtype": trt_utils.trt.float32} for name in ["x", "y"]
    }
