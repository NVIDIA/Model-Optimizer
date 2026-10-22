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

"""Normalize the double-DQ contract for a preconverted FP16 NVFP4 graph."""

import copy

import onnx
from onnx import TensorProto

__all__ = ["canonicalize_nvfp4_fp16"]


def canonicalize_nvfp4_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Keep literal global scales in FP32 and produce FP16 from both DQ stages."""
    graph = model.graph
    source_nodes = {node.name: node for node in graph.node}
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    infos = {value.name: value for value in [*graph.input, *graph.output, *graph.value_info]}
    fp4 = {
        name for name, tensor in initializers.items() if tensor.data_type == TensorProto.FLOAT4E2M1
    }
    fp8 = {
        name
        for name, tensor in initializers.items()
        if tensor.data_type == TensorProto.FLOAT8E4M3FN
    }
    for node in graph.node:
        if node.op_type == "TRT_FP4DynamicQuantize":
            fp4.add(node.output[0])
            fp8.add(node.output[1])
    scale_outputs = {
        node.output[0]
        for node in graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] in fp8
    }

    def literal(name):
        if name in initializers:
            return copy.deepcopy(initializers[name])
        node = producers.get(name)
        if node is not None and node.op_type == "Constant":
            return copy.deepcopy(next(attr.t for attr in node.attribute if attr.name == "value"))
        if node is not None and node.op_type in {"Cast", "Identity"}:
            return literal(node.input[0])
        raise ValueError(f"NVFP4 global scale must be a literal: {name}")

    def preserve_scale(node):
        tensor = literal(source_nodes[node.name].input[1])
        if tensor.data_type != TensorProto.FLOAT:
            raise ValueError(f"NVFP4 global scale must originate in FP32: {node.name}")
        tensor.name = node.name + "_global_scale_fp32_literal"
        graph.initializer.append(tensor)
        node.input[1] = tensor.name

    for node in graph.node:
        if node.op_type == "TRT_FP4DynamicQuantize":
            preserve_scale(node)
        if node.op_type != "DequantizeLinear":
            continue
        is_scale = node.input[0] in fp8
        is_data = node.input[0] in fp4
        if not (is_scale or is_data):
            continue
        if is_scale:
            preserve_scale(node)
        else:
            scale = node.input[1]
            while scale not in scale_outputs:
                parent = producers.get(scale)
                if parent is None or parent.op_type not in {"Cast", "Identity"}:
                    break
                scale = parent.input[0]
            if scale in scale_outputs:
                node.input[1] = scale
        attrs = [attr for attr in node.attribute if attr.name != "output_dtype"]
        node.ClearField("attribute")
        node.attribute.extend(attrs)
        node.attribute.append(onnx.helper.make_attribute("output_dtype", TensorProto.FLOAT16))
        if node.output[0] in infos:
            infos[node.output[0]].type.tensor_type.elem_type = TensorProto.FLOAT16

    # The Torch exporter can retain FP32 annotations after FP4 post-processing.
    for node in graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        input_types = []
        for name in node.input[:2]:
            producer = producers.get(name)
            if producer is not None and producer.op_type == "Cast":
                input_types.append(next(attr.i for attr in producer.attribute if attr.name == "to"))
            elif name in initializers:
                input_types.append(initializers[name].data_type)
            elif name in infos:
                input_types.append(infos[name].type.tensor_type.elem_type)
        if input_types == [TensorProto.FLOAT16, TensorProto.FLOAT16]:
            for name in node.output:
                if name in infos:
                    infos[name].type.tensor_type.elem_type = TensorProto.FLOAT16

    # Drop bypassed scale casts and order live nodes before TensorRT parsing.
    ordered, seen, needed = [], set(), set()

    def visit(name):
        needed.add(name)
        node = producers.get(name)
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        for input_name in node.input:
            visit(input_name)
        ordered.append(node)

    for output in graph.output:
        visit(output.name)
    graph.ClearField("node")
    graph.node.extend(ordered)
    for field in ("initializer", "input", "value_info"):
        values = [value for value in getattr(graph, field) if value.name in needed]
        graph.ClearField(field)
        getattr(graph, field).extend(values)
    return model
