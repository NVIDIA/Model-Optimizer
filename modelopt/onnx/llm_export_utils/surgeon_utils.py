# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities to surgeon ONNX graph after export."""

import re
import time

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx_graphsurgeon.ir.tensor import LazyValues


def clear_inputs(node: gs.Node | gs.Tensor):
    """Clear all inputs for a node or tensor in ONNX."""
    for i in node.inputs:
        i.outputs.clear()
    node.inputs.clear()
    return node


def clear_outputs(node: gs.Node | gs.Tensor):
    """Clear all outputs for a node or tensor in ONNX."""
    for o in node.outputs:
        o.inputs.clear()
    node.outputs.clear()
    return node


def extract_layer_id(name: str):
    """Extract layer id from certain ONNX layer name.

    Parameters:
        name: str
            The name of ONNX layer. e.g. /model/layer.0/q_proj/...

    Returns:
        The layer id for the layer as int. In the example above, it returns 0
    """
    match = re.search(r"layers\.(\d+)", name)
    if match:
        return int(match.group(1))
    raise Exception(f"{name} does not contain layer info!")


def no_none_elements(elements: list):
    """Check if all elements in the list are not None."""
    return all(i is not None for i in elements)


def fold_fp8_qdq_to_dq(graph: gs.Graph):
    """Convert FP32/FP16 weights of the given ONNX model to FP8 weights.

    Even though modelopt supports FP8 onnx export, the weights are represented in fp32 + QDQ.
    The storage is therefore very bad. In this function,
    Q nodes will get removed from the weights and have only DQ nodes with those converted FP8
    weights in the output model.

    Parameters:
        graph: gs.Graph.

    Returns:
        gs.Graph with only DQ nodes for weights and same QDQ nodes for activations.
    """
    start_time = time.time()
    print("Replacing all (fp32 weights + fp8 QDQ) with (fp8 weights + DQ)...")
    # Fold constants is required since the scale is not constant yet.
    graph.cleanup().toposort().fold_constants().cleanup()

    for node in graph.nodes:
        if node.op == "TRT_FP8QuantizeLinear":
            # Should not remove input QDQ
            if not isinstance(node.inputs[0], gs.Constant):
                continue

            weights = node.inputs[0]
            scale = node.inputs[1]
            torch_weights = torch.from_numpy(weights.values)
            torch_scale = torch.from_numpy(scale.values)
            quantizer_name = scale.name.rsplit("/", 1)[0]
            dq_op = node.outputs[0].outputs[0]
            assert dq_op.op == "TRT_FP8DequantizeLinear", (
                f"QDQ does not occur in pairs. You reached {dq_op.op}"
            )

            # Replace it with Dequantize with FP8 weights. This is a WAR because numpy does not support fp8.
            numpy_weights = (
                (torch_weights / torch_scale).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            )
            tensor = onnx.TensorProto()
            tensor.data_type = onnx.TensorProto.FLOAT8E4M3FN
            tensor.dims.extend(numpy_weights.shape)
            tensor.raw_data = numpy_weights.tobytes()
            values = LazyValues(tensor)
            onnx_weights_fp8 = gs.Constant(quantizer_name + "/fp8_weights", values)

            node.outputs.clear()
            # DQ Op is separated out
            dq_op.inputs[0] = onnx_weights_fp8
            dq_op.op = "DequantizeLinear"
            dq_op.outputs[0].dtype = dq_op.inputs[1].dtype

    graph.cleanup().toposort()
    end_time = time.time()
    print(f"fp8 qdq replaced with only dq completed in {end_time - start_time}s.")

    return graph
