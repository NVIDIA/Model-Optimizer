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

"""Base class for ONNX quantizer exporters."""

from abc import ABC, abstractmethod

import onnx


def _sync_initializer_metadata(graph: onnx.GraphProto, tensor: onnx.TensorProto) -> None:
    """Synchronize declarations for an initializer after changing its type or shape."""
    tensor_type = onnx.helper.make_tensor_value_info(
        tensor.name, tensor.data_type, tensor.dims
    ).type
    for value_info in (*graph.input, *graph.value_info, *graph.output):
        if value_info.name == tensor.name:
            value_info.type.CopyFrom(tensor_type)


def _replace_initializer(graph: onnx.GraphProto, tensor: onnx.TensorProto) -> None:
    """Replace an initializer and synchronize any existing type declarations."""
    existing = next((item for item in graph.initializer if item.name == tensor.name), None)
    if existing is None:
        graph.initializer.append(tensor)
    else:
        existing.CopyFrom(tensor)

    _sync_initializer_metadata(graph, tensor)


def _materialize_initializer_input(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    input_index: int,
    tensor: onnx.TensorProto,
) -> None:
    """Materialize a constant input, splitting shared values per quantized weight."""
    input_name = node.input[input_index]
    consumers = [candidate for candidate in graph.node if input_name in candidate.input]
    producer = next((candidate for candidate in graph.node if input_name in candidate.output), None)

    if len(consumers) > 1:
        tensor.name = f"{tensor.name}_{node.output[0]}"
    elif producer is not None and producer.op_type == "Constant":
        graph.node.remove(producer)

    node.input[input_index] = tensor.name
    _replace_initializer(graph, tensor)


def _single_consumer(consumers, name: str, weight_name: str) -> onnx.NodeProto:
    matches = consumers.get(name, [])
    if len(matches) != 1:
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight topology for '{weight_name}': "
            f"expected one consumer; found {len(matches)}."
        )
    return matches[0]


def _validate_linear_weight_path(consumers, marker: onnx.NodeProto) -> None:
    """Require a straight marker-to-MatMul/Gemm weight path."""
    _single_consumer(consumers, marker.input[0], marker.input[0])
    value_name = marker.output[0]
    consumer = _single_consumer(consumers, value_name, marker.input[0])
    while consumer.op_type in {"Cast", "Transpose"}:
        value_name = consumer.output[0]
        consumer = _single_consumer(consumers, value_name, marker.input[0])
    if consumer.op_type not in {"MatMul", "Gemm"} or consumer.input[1] != value_name:
        raise NotImplementedError(
            f"Unsupported Dynamo quantized weight topology for '{marker.input[0]}': "
            "expected terminal MatMul/Gemm at input 1."
        )


class ONNXQuantExporter(ABC):
    """Base class for ONNX quantizer exporters."""

    @classmethod
    def process_model(cls, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Processes the ONNX model."""
        onnx_model = cls.pre_process(onnx_model)
        onnx_model = cls.compute_scales(onnx_model)
        onnx_model = cls.compress_weights(onnx_model)
        onnx_model = cls.post_process(onnx_model)
        return onnx_model

    @staticmethod
    @abstractmethod
    def pre_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Pre-processes the ONNX model. Converts all DQ -> * -> op patterns to DQ -> op."""

    @staticmethod
    @abstractmethod
    def compute_scales(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Computes the scales for the weights in the ONNX model."""

    @staticmethod
    @abstractmethod
    def compress_weights(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Compresses the weights in the ONNX model."""

    @staticmethod
    @abstractmethod
    def post_process(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Post-processes the ONNX model."""
