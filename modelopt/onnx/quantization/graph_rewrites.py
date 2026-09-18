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

"""Shared ONNX graph rewrites for quantization."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.ir.tensor import Variable

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_indexing import (
    get_tensor_consumer_nodes,
    match_fp8_mha_pattern,
)

__all__ = [
    "cast_custom_ops",
    "convert_fp16_io",
    "insert_fp8_mha_casts",
    "insert_matmul_casts",
    "remove_output_initializers",
    "remove_redundant_cast_nodes",
]


def cast_custom_ops(onnx_model: onnx.ModelProto, ops_to_cast: dict) -> onnx.ModelProto:
    """Adds cast_to_fp16 nodes to the inputs and cast_to_fp32 to the outputs of a layer in the requested indices."""
    logger.info("Casting custom ops in the requested inputs and outputs")
    name_dict = {}

    def _get_unique_name(old_name):
        if old_name not in name_dict:
            name_dict[old_name] = 0
            return old_name
        name_dict[old_name] = name_dict[old_name] + 1
        return old_name + "_" + str(name_dict[old_name])

    def _is_castable_tensor(tensor) -> bool:
        castable_types = ["float16", "float32", "double"]
        return tensor.dtype and tensor.dtype in castable_types

    def _add_cast_node_inp(tensor, precision="fp16", suffix=""):
        if precision == "fp16":
            onnx_precision = int(onnx.TensorProto.FLOAT16)
            np_precision = "float16"
        else:
            onnx_precision = int(onnx.TensorProto.FLOAT)
            np_precision = "float32"

        cast_out = Variable(
            name=_get_unique_name(tensor.name + f"_{precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{precision}{suffix}"),
            attrs={"to": onnx_precision},
            inputs=[tensor],
            outputs=[cast_out],
        )
        graph.nodes.append(cast_node)
        return cast_out

    def _add_cast_node_out(tensor, inp_precision="fp16", out_precision="fp32", suffix=""):
        cast_precision = (
            int(onnx.TensorProto.FLOAT16)
            if out_precision == "fp16"
            else int(onnx.TensorProto.FLOAT)
        )
        np_precision = "float16" if inp_precision == "fp16" else "float32"

        cast_inp = gs.Variable(
            name=_get_unique_name(tensor.name + f"_{inp_precision}{suffix}"),
            dtype=np_precision,
            shape=tensor.shape,
        )
        cast_node = gs.Node(
            op="Cast",
            name=_get_unique_name(tensor.name + f"_cast_to_{out_precision}{suffix}"),
            attrs={"to": cast_precision},
            inputs=[cast_inp],
            outputs=[tensor],
        )
        graph.nodes.append(cast_node)
        return cast_inp

    graph = gs.import_onnx(onnx_model)
    castable_nodes = [n for n in graph.nodes if n.op in ops_to_cast]

    for node in castable_nodes:
        inp_idxs = ops_to_cast[node.op]["inp"]
        out_idxs = ops_to_cast[node.op]["out"]

        # Cast relevant inputs to FP16
        for inp_idx, inp in enumerate(node.inputs):
            if inp_idx in inp_idxs and _is_castable_tensor(inp):
                cast_out = _add_cast_node_inp(inp)
                node.inputs[inp_idx] = cast_out

        # Cast relevant outputs from FP16 back to FP32
        for out_idx, out in enumerate(node.outputs):
            if out_idx in out_idxs and _is_castable_tensor(out):
                cast_inp = _add_cast_node_out(out)
                node.outputs[out_idx] = cast_inp

    graph.cleanup().toposort()

    onnx_model = gs.export_onnx(graph)
    # TODO: remove manual ir_version change once ORT supports ir_version 11
    onnx_model.ir_version = 10

    return onnx_model


def insert_matmul_casts(graph, matmul_node):
    """Insert three cast nodes for MatMul's two inputs and output."""
    matmul_input0 = matmul_node.inputs[0]
    matmul_input0_cast_output = gs.Variable(
        name=f"{matmul_input0.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input0.name}/Cast",
        inputs=[matmul_input0],
        outputs=[matmul_input0_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[0] = matmul_input0_cast_output

    matmul_input1 = matmul_node.inputs[1]
    matmul_input1_cast_output = gs.Variable(
        name=f"{matmul_input1.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_input1.name}/Cast",
        inputs=[matmul_input1],
        outputs=[matmul_input1_cast_output],
        attrs={"to": np.float32},
    )
    matmul_node.inputs[1] = matmul_input1_cast_output

    matmul_output = matmul_node.outputs[0]
    matmul_output_cast_input = gs.Variable(
        name=f"{matmul_output.name}/Cast_output", dtype=np.float32
    )
    graph.layer(
        op="Cast",
        name=f"{matmul_output.name}/Cast",
        inputs=[matmul_output_cast_input],
        outputs=[matmul_output],
        attrs={"to": np.float16},
    )
    matmul_node.outputs[0] = matmul_output_cast_input


def insert_fp8_mha_casts(onnx_model):
    r"""Insert three cast ops.

    The first cast will be added before the input0 of MatMul to cast fp16 to fp32.
    The second cast will be added before the input1 of MatMul to cast fp16 to fp32.
    The third cast will be added after the output of MatMul to cast fp32 back to fp16.
    The insertion of Cast ops in the FP8 MHA part actually forbids the MHAs to run
    with FP16 accumulation because the compiler only has FP32 accumulation kernels for FP8 MHAs.
    """
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()

    # Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ
    for node in graph.nodes:
        if node.op == "Softmax":
            fp8_fmha_v2_pattern = match_fp8_mha_pattern(graph, node, True)
            # Insert cast nodes on BMM2's input and output tensors.
            if len(fp8_fmha_v2_pattern) == 3:
                insert_matmul_casts(graph, fp8_fmha_v2_pattern[0])
                insert_matmul_casts(graph, fp8_fmha_v2_pattern[2])

    graph.cleanup().toposort()

    return gs.export_onnx(graph)


def convert_fp16_io(graph):
    """Convert graph I/O to FP16."""
    convertible_dtypes = [
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    ]
    for input_tensor in graph.inputs:
        input_tensor.dtype = (
            onnx.TensorProto.FLOAT16
            if input_tensor.dtype in convertible_dtypes
            else input_tensor.dtype
        )
    for output_tensor in graph.outputs:
        output_tensor.dtype = (
            onnx.TensorProto.FLOAT16
            if output_tensor.dtype in convertible_dtypes
            else output_tensor.dtype
        )


def remove_output_initializers(graph: gs.Graph, graph_initializers: list):
    """Remove initializers that are also listed as graph outputs.

    Having initializers (constant tensors) that are also marked as outputs can lead to ONNX Runtime
    or conversion tool errors, particularly related to ambiguous 'dtype' or shape inference.
    This step ensures compatibility by detaching such initializers from the graph's outputs.
    """
    init_names = [init.name for init in graph_initializers]
    init_names_removed = []
    for output_tensor in graph.outputs:
        if output_tensor.name in init_names:
            graph.outputs.remove(output_tensor)
            init_names_removed.append(output_tensor.name)
    if init_names_removed:
        logger.info(f"Removed output initializers: {init_names_removed}")


def remove_redundant_cast_nodes(graph: onnx.GraphProto) -> None:
    """Remove redundant Cast nodes from the ONNX graph to optimize model performance.

    This function identifies and removes two types of redundant Cast nodes:

    1. Cast nodes where input and output types are identical
       - Before: t1 (dtype=fp16) -> cast (to=fp16) -> t2 -> Op
       - After:  t1 (dtype=fp16) -> Op

    2. Cast nodes that can be fused with initializers
       - Before: (initializer) t1 (dtype=fp32) -> cast (to=fp16) -> t2 -> Op
       - After:  (initializer) t1 (dtype=fp16) -> Op

    The function preserves Cast nodes that:
    - Have outputs that are graph outputs
    - Are necessary for type conversion
    - Have dynamic inputs (not initializers)

    Args:
        graph: ONNX graph to optimize. The graph will be modified in-place.

    Note:
        - This optimization is particularly useful for models with many Cast operations
        - The function modifies the graph in-place
        - All tensor consumers are updated to maintain graph connectivity
        - Initializer data types are converted when possible to eliminate Cast nodes
    """
    initializers = {init.name: init for init in graph.initializer}
    tensor_consumers = get_tensor_consumer_nodes(graph)
    value_info_map = {info.name: info for info in graph.value_info}
    cast_indices = []
    output_names = {output.name for output in graph.output}

    def _get_tensor_type(tensor_name: str) -> int | None:
        """Get the tensor type for a given tensor name."""
        if tensor_name in value_info_map:
            return value_info_map[tensor_name].type.tensor_type.elem_type
        if tensor_name in initializers:
            return initializers[tensor_name].data_type
        return None

    for node_idx, node in enumerate(graph.node):
        if node.op_type != "Cast":
            continue

        # Skip if output is a graph output
        if any(out_name in output_names for out_name in node.output):
            continue

        input_name = node.input[0]
        input_type = _get_tensor_type(input_name)
        if input_type is None:
            continue

        # Get target type from Cast node attributes
        attr = next((attr for attr in node.attribute if attr.name == "to"), None)
        if attr is None:
            continue

        # Pattern 1: Input and output types are the same
        if input_type == attr.i:
            cast_indices.append(node_idx)
        # Pattern 2: Convert and fuse Cast node for initializers
        elif input_name in initializers:
            cast_indices.append(node_idx)
            cast_input = onnx.numpy_helper.to_array(initializers[input_name])
            dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.i)
            converted_tensor = onnx.numpy_helper.from_array(cast_input.astype(dtype), input_name)
            initializers[input_name].CopyFrom(converted_tensor)
        else:
            continue

        # Update consumer nodes
        for consumer in tensor_consumers.get(node.output[0], []):
            for i, input_tensor in enumerate(consumer.input):
                if input_tensor == node.output[0]:
                    consumer.input[i] = input_name
                    break

    # Remove Cast nodes in reverse order
    logger.info(f"Removing {len(cast_indices)} redundant Cast nodes")
    for node_idx in sorted(cast_indices, reverse=True):
        del graph.node[node_idx]
