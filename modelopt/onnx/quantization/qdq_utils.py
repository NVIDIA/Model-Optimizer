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

"""Various utils to support inserting Q/DQ nodes."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import numpy_helper

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_utils import (
    get_tensor_consumer_nodes,
    get_tensor_from_name,
    get_tensor_producer_nodes,
    remove_redundant_cast_nodes,
)
from modelopt.onnx.quantization.quant_utils import get_num_bits

QUANTIZE_NODE_NAME = "QuantizeLinear"
DEQUANTIZE_NODE_NAME = "DequantizeLinear"

onnx_dtype_map = {
    "BFloat16": onnx.TensorProto.BFLOAT16,
    "Float": onnx.TensorProto.FLOAT,
    "Float4": onnx.TensorProto.FLOAT4E2M1,
    "Float8": onnx.TensorProto.FLOAT8E4M3FN,
    "Half": onnx.TensorProto.FLOAT16,
    "INT8": onnx.TensorProto.INT8,
    "UINT8": onnx.TensorProto.UINT8,
    "INT4": onnx.TensorProto.INT4,
    "UINT4": onnx.TensorProto.UINT4,
}
onnx_bit_dtype_signed_map = {4: "INT4", 8: "INT8"}
onnx_bit_dtype_unsigned_map = {4: "UINT4", 8: "UINT8"}

np_dtype_map = {
    "Float": np.float32,
    "Half": np.float16,
    "INT8": np.int8,
    "UINT8": np.uint8,
}


def use_trt_qdq_ops():
    """Globally set node names to TRT custom names."""
    logger.debug("Using TRT QDQ ops")
    global QUANTIZE_NODE_NAME
    QUANTIZE_NODE_NAME = "TRT_INT4QuantizeLinear"
    global DEQUANTIZE_NODE_NAME
    DEQUANTIZE_NODE_NAME = "TRT_INT4DequantizeLinear"


def _wq_name(name: str):
    return name + "_i4"


def _scale_name(name: str):
    return name + "_scale"


def _awq_scale_name(name: str):
    return name + "_awq_scale"


def _zp_name(name: str):
    return name + "_zp"


def _q_name(name: str):
    return name + "_QuantizeLinear"


def _q_out_name(name: str):
    return name + "_QuantizeLinear_Output"


def _dq_name(name: str):
    return name + "_DequantizeLinear"


def _pqs_name(name: str):
    return name + "_PQS"


def _dq_out_name(name: str):
    return name + "_DequantizeLinear_Output"


def _pqs_out_name(name: str):
    return name + "_PQS_Tensor"


def make_gs_quantized_weight(name: str, wq: np.ndarray, dtype) -> gs.Constant:
    """Create a GraphSurgeon tensor from a quantized weight tensor.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(_wq_name(name), np.asarray(wq), dtype)


def make_gs_zp(name: str, shape: Sequence[int], dtype) -> gs.Constant:
    """Create a GraphSurgeon zero-point tensor of all zeroes with the given shape.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.make_constant(
        _zp_name(name),
        np.zeros(shape, dtype=onnx.helper.tensor_dtype_to_np_dtype(dtype)),
        dtype,
    )


def make_gs_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_scale_name(name), np.asarray(scale))


def make_gs_awq_scale(name: str, scale: np.ndarray) -> gs.Constant:
    """Create a GraphSurgeon scale tensor from the given numpy array.

    `name` is the desired _basename_ of the tensor.
    """
    return gs.Constant(_awq_scale_name(name), np.asarray(scale))


def make_gs_quantize_output(
    name: str, shape: Sequence[int], dtype: onnx.TensorProto.DataType
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.make_variable(_q_out_name(name), dtype=dtype, shape=shape)


def make_gs_quantize_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        QUANTIZE_NODE_NAME,
        name=_q_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_pre_quant_scale_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_pqs_out_name(name), dtype=dtype, shape=shape)


def make_gs_dequantize_output(
    name: str,
    shape: Sequence[int],
    dtype: np.dtype,
) -> gs.Variable:
    """Create a GraphSurgeon variable representing the output of a quantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Variable(_dq_out_name(name), dtype=dtype, shape=shape)


def make_gs_pre_quant_scale_node(
    name: str, inputs: Sequence[gs.Tensor], outputs: Sequence[gs.Tensor]
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        "Mul",
        name=_pqs_name(name),
        inputs=inputs,
        outputs=outputs,
    )


def make_gs_dequantize_node(
    name: str,
    inputs: Sequence[gs.Tensor],
    outputs: Sequence[gs.Tensor],
    attributes: dict[str, Any] | None = None,
) -> gs.Node:
    """Create a GraphSurgeon Dequantize node.

    `name` is the desired _basename_ of the node.
    """
    return gs.Node(
        DEQUANTIZE_NODE_NAME,
        name=_dq_name(name),
        inputs=inputs,
        outputs=outputs,
        attrs=attributes,
    )


def _postprocess_qdq(
    graph: gs.Graph,
    orig_weight_names: set[str],
    q_nodes: dict[str, gs.Node] = {},
    dq_nodes: dict[str, gs.Node] = {},
):
    # Inserts all newly created nodes to graph.
    # Update all consumers of original initializers to point to the DQ nodes.
    logger.debug(f"Postprocessing QDQ nodes for {len(orig_weight_names)} weights")
    for node in graph.nodes:
        for i in range(len(node.inputs)):
            key = node.inputs[i].name
            if key not in orig_weight_names:
                continue
            node.inputs[i] = dq_nodes[key].outputs[0]

    # Insert new nodes.
    graph.nodes.extend(q_nodes.values())
    graph.nodes.extend(dq_nodes.values())

    graph.cleanup()
    graph.toposort()
    logger.debug(f"Added {len(q_nodes)} Q nodes and {len(dq_nodes)} DQ nodes")


def insert_pre_quant_scale_nodes(
    graph: gs.Graph, input_tensors: dict[str, str], pre_quant_scale: dict[str, np.ndarray]
):
    """Insert new mul nodes into graph.

    Args:
        graph: The graph to modify.
        input_tensors: A dictionary of weight tensor names mapped to corresponding input tensor names
        pre_quant_scale: A map from ONNX input tensor name to corresponding pre-quant scale.
    """
    logger.debug(f"Inserting pre-quant scale nodes for {len(pre_quant_scale)} tensors")

    def _insert_helper(
        weight_tensor_name: str,
        input_tensor_name: str,
        scale: np.ndarray,
        mul_nodes: dict[str, gs.Node],
    ):
        pre_quant_scale_tensor = make_gs_awq_scale(weight_tensor_name, scale)
        # TODO: Study effects of caching Gemm/Matmul nodes on perf and mem usage.
        gemm_nodes = [node for node in graph.nodes if node.op in ["Gemm", "MatMul"]]
        for node in gemm_nodes:
            input_set = {input.name for input in node.inputs}
            input_idxs = {input.name: idx for idx, input in enumerate(node.inputs)}
            if _dq_out_name(weight_tensor_name) in input_set and input_tensor_name in input_set:
                pqs_in = node.inputs[input_idxs[input_tensor_name]]
                pqs_out = make_gs_pre_quant_scale_output(
                    weight_tensor_name, shape=pqs_in.shape, dtype=scale.dtype
                )
                mul_node = make_gs_pre_quant_scale_node(
                    weight_tensor_name, inputs=[pqs_in, pre_quant_scale_tensor], outputs=[pqs_out]
                )
                node.inputs[input_idxs[input_tensor_name]] = mul_node.outputs[0]
                mul_nodes[weight_tensor_name] = mul_node

    mul_nodes = {}
    for w_name, scale in pre_quant_scale.items():
        inv_scale = 1.0 / scale
        _insert_helper(w_name, input_tensors[w_name], inv_scale, mul_nodes)

    graph.nodes.extend(mul_nodes.values())

    graph.cleanup()
    graph.toposort()


def get_tensor_dtype(num_bits: int = 4, has_zero_point: bool = False) -> int:
    """Get the appropriate tensor dtype based on precision info and zero point presence.

    Args:
        num_bits: Number of bits for quantization
        has_zero_point: Whether the tensor has a zero point
    Returns:
        ONNX tensor data type constant
    """
    if has_zero_point:
        dtype_str = onnx_bit_dtype_unsigned_map[num_bits]
    else:
        dtype_str = onnx_bit_dtype_signed_map[num_bits]
    return onnx_dtype_map[dtype_str]


def update_attributes_for_per_channel_nodes(
    attributes: dict[str, Any] | None = None, num_bits: int = 4
) -> dict[str, Any] | None:
    """Get the attributes for per-channel nodes."""
    attrs = attributes.copy() if attributes is not None else None
    if ((attrs is not None) and (attrs.get("block_size", None) == -1)) or (num_bits == 8):
        if attrs is not None:
            attrs["axis"] = 1
            if "block_size" in attrs:
                del attrs["block_size"]
    return attrs


def validate_scale_shape_for_per_channel_nodes(
    scale: np.ndarray, attrs: dict[str, Any] | None = None, num_bits: int = 4
):
    """Validate the shape of the scale tensor for per-channel nodes."""
    if attrs is not None:
        if ("block_size" not in attrs) or (num_bits == 8):
            assert scale.ndim == 1, "Scale shape is not valid for per-channel nodes"


def insert_dq_nodes(
    graph: gs.Graph,
    scales: dict[str, np.ndarray],
    quantized_weights: dict[str, np.ndarray],
    attributes: dict[str, Any] | None = None,
    zero_points: dict[str, np.ndarray] | None = None,
    layer_info: dict[str, dict] | None = None,
):
    """Insert new initializers and DQ nodes into graph.

    Args:
        graph: The graph to modify.
        weights: A map from ONNX initializer name to tensor.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        dq_only: Whether to only insert dq nodes.
        layer_info: Optional dictionary mapping tensor names to precision (old format) or
            to layer configuration dict (new format with precision, block_size, axis).
    """
    logger.debug(f"Inserting DQ nodes for {len(scales)} weights")

    def _insert_helper(
        name: str,
        wq: np.ndarray,
        scale: np.ndarray,
        dq_nodes: dict[str, gs.Node],
        zp: np.ndarray,
        attrs: dict[str, Any] | None = None,
        num_bits: int = 4,
    ):
        tensor_dtype = get_tensor_dtype(num_bits, zp is not None)

        wq_tensor = make_gs_quantized_weight(name, wq, tensor_dtype)
        scale_tensor = make_gs_scale(name, scale)
        dq_out = make_gs_dequantize_output(name, shape=wq.shape, dtype=scale.dtype)
        inputs = [wq_tensor, scale_tensor]
        if zp is not None:
            zp_tensor = gs.make_constant(_zp_name(name), zp, tensor_dtype)
            inputs.append(zp_tensor)
        dq_node = make_gs_dequantize_node(
            name,
            inputs=inputs,
            outputs=[dq_out],
            attributes=attrs,
        )
        dq_nodes[name] = dq_node

    dq_nodes = {}
    for name, scale in scales.items():
        zp = None
        if zero_points is not None:
            zp = zero_points.get(name)
            assert zp is not None, "zero-point is enabled but zero-point values not found"

        num_bits = get_num_bits(layer_info, name)
        # Updating the attributes for per-channel nodes.
        attrs = attributes.copy() if attributes is not None else None
        attrs = update_attributes_for_per_channel_nodes(attrs, num_bits)
        validate_scale_shape_for_per_channel_nodes(scale, attrs, num_bits)
        _insert_helper(
            name,
            quantized_weights[name],
            scale,
            dq_nodes,
            zp,
            attrs,
            num_bits=num_bits,
        )

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        dq_nodes=dq_nodes,
    )


def insert_qdq_nodes(
    graph: gs.Graph,
    scales: dict[str, np.ndarray],
    weight_map: dict[str, gs.Tensor],
    layer_info: dict[str, dict] | None = None,
):
    """Insert scales and QDQ nodes into graph.

    Args:
        graph: The graph to modify.
        scales: A map from ONNX initializer name to desired scale factor for that initializer.
        weight_map: A map from ONNX initializer name to graphsurgeon tensor.
        layer_info: Optional dictionary mapping tensor names to precision (old format) or
            to layer configuration dict (new format with precision, block_size, axis).
    """
    logger.debug(f"Inserting QDQ nodes for {len(scales)} weights")

    def _insert_helper(
        name: str,
        weight_to_quantize: gs.Tensor,
        scale: np.ndarray,
        q_nodes: dict[str, gs.Node],
        dq_nodes: dict[str, gs.Node],
        num_bits: int = 4,
    ):
        tensor_dtype = get_tensor_dtype(num_bits)

        scale_tensor = make_gs_scale(name, scale)
        zp_tensor = make_gs_zp(name, scale.shape, tensor_dtype)
        q_out = make_gs_quantize_output(name, weight_to_quantize.shape, tensor_dtype)
        q_node = make_gs_quantize_node(
            name, inputs=[weight_to_quantize, scale_tensor, zp_tensor], outputs=[q_out]
        )
        dq_out = make_gs_dequantize_output(name, shape=weight_to_quantize.shape, dtype=scale.dtype)
        dq_node = make_gs_dequantize_node(
            name, inputs=[q_out, scale_tensor, zp_tensor], outputs=[dq_out]
        )
        q_nodes[name] = q_node
        dq_nodes[name] = dq_node

    q_nodes, dq_nodes = {}, {}
    for name, scale in scales.items():
        _insert_helper(
            name,
            weight_map[name],
            scale,
            q_nodes,
            dq_nodes,
            num_bits=get_num_bits(layer_info, name),
        )

    _postprocess_qdq(
        graph,
        orig_weight_names=set(scales.keys()),
        q_nodes=q_nodes,
        dq_nodes=dq_nodes,
    )


def replace_scale_values(graph: onnx.GraphProto, act_scales_dict: dict[str, float]) -> None:
    """Replace scale values in the graph with values from calibration cache.

    Args:
        graph: ONNX graph to modify
        act_scales_dict: Dictionary mapping scale tensor names to their new values
    """
    logger.debug(f"Replacing scale values for {len(act_scales_dict)} tensors")
    initializer_indices = {init.name: idx for idx, init in enumerate(graph.initializer)}

    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue

        scale_name = node.input[1]
        if scale_name in act_scales_dict:
            if scale_name not in initializer_indices:
                raise ValueError(f"Scale tensor '{scale_name}' not found in graph initializers")

            scale = onnx.numpy_helper.from_array(
                np.float32(act_scales_dict[scale_name]), scale_name
            )
            graph.initializer[initializer_indices[scale_name]].CopyFrom(scale)
            logger.debug(f"Updated scale value for {scale_name}")
        else:
            # For weight quantizers, verify the weight tensor exists
            weight_name = node.input[0]
            if weight_name not in initializer_indices:
                raise ValueError(f"Weight tensor '{weight_name}' not found in graph initializers")


def has_qdq_nodes(onnx_model: onnx.ModelProto):
    """Check if the onnx graph already has QDQ nodes."""
    qdq_ops = {QUANTIZE_NODE_NAME, DEQUANTIZE_NODE_NAME}
    return any(node.op_type in qdq_ops for node in onnx_model.graph.node)


def _get_graph_metadata(
    graph: onnx.GraphProto,
) -> tuple[dict[str, onnx.TensorProto], dict[str, onnx.NodeProto], dict[str, list[onnx.NodeProto]]]:
    """Get helper dictionaries for efficient graph traversal and node analysis.

    Args:
        graph: ONNX graph to analyze

    Returns:
        Tuple containing:
            - initializers: Maps initializer names to their TensorProto objects
            - tensor_producers: Maps tensor names to their producer nodes
            - tensor_consumers: Maps tensor names to their consumer nodes
    """
    initializers = {init.name: init for init in graph.initializer}
    tensor_producers = get_tensor_producer_nodes(graph)
    tensor_consumers = get_tensor_consumer_nodes(graph)
    return initializers, tensor_producers, tensor_consumers


def _get_scale_and_zp(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    tensor_producers: dict[str, onnx.NodeProto],
) -> tuple[onnx.TensorProto, onnx.TensorProto]:
    """Get scale and zero point tensors for a node.

    Args:
        node: ONNX node to get scale and zero point for
        initializers: Dictionary of initializers
        tensor_producers: Dictionary of tensor producers

    Returns:
        Tuple of (scale_tensor, zero_point_tensor)

    Raises:
        ValueError: If scale or zero point cannot be found
    """
    # Get scale tensor
    scale_name = node.input[1]
    if scale_name in initializers:
        scale = initializers[scale_name]
    else:
        producer = tensor_producers.get(scale_name)
        if not producer or not producer.attribute:
            raise ValueError(f"Invalid scale producer for {scale_name}")
        scale = producer.attribute[0].t

    # Get zero point tensor
    zp_name = node.input[2]
    if zp_name in initializers:
        zp = initializers[zp_name]
    else:
        producer = tensor_producers.get(zp_name)
        if not producer or not producer.attribute:
            raise ValueError(f"Invalid zero point producer for {zp_name}")
        zp = producer.attribute[0].t

    return scale, zp


def _get_successive_consumers(
    node: onnx.NodeProto, tensor_consumers: dict[str, list[onnx.NodeProto]]
) -> tuple[onnx.NodeProto, onnx.NodeProto]:
    """Get the DequantizeLinear node and its consumer node for a given QuantizeLinear node.

    This function validates and retrieves the next two nodes in the quantization chain:
    QuantizeLinear -> DequantizeLinear -> Operation

    Args:
        node: The QuantizeLinear node to find consumers for
        tensor_consumers: Dictionary mapping tensor names to their consumer nodes

    Returns:
        Tuple containing:
            - dq_node: The DequantizeLinear node that consumes the QuantizeLinear output
            - quantized_node: The operation node that consumes the DequantizeLinear output
    """
    dq_node = tensor_consumers.get(node.output[0], [None])[0]
    if not dq_node or dq_node.op_type != "DequantizeLinear":
        raise ValueError(f"Invalid consumer for {node.name}")

    quantized_node = tensor_consumers.get(dq_node.output[0], [None])[0]
    if not quantized_node:
        raise ValueError(f"No consumer found for {dq_node.name}")
    if quantized_node.op_type == "Cast":
        next_node = tensor_consumers.get(quantized_node.output[0], [None])[0]
        if not next_node:
            raise ValueError(f"No consumer found after Cast for {quantized_node.name}")
        quantized_node = next_node

    return dq_node, quantized_node


def _convert_weight(
    weight_array: np.ndarray,
    scale: onnx.TensorProto,
    zp: onnx.TensorProto,
    quantized_node: onnx.NodeProto,
) -> np.ndarray:
    """Convert a weight tensor to INT8/FP8 format based on scale and zero point.

    Args:
        weight_array: The weight tensor to convert
        scale: The scale tensor for quantization
        zp: The zero point tensor for quantization
        quantized_node: The operation node that will use the converted weight

    Returns:
        The converted weight tensor as a numpy array

    Raises:
        ValueError: If scale shape doesn't match weight shape for the operation

    Note:
        - INT8 weights are clipped to [-128, 127]
        - FP8 weights use float8e4m3fn format
    """
    # Per-op quantization axis mapping (must match ORT config)
    weight_shape = weight_array.shape
    op_type = quantized_node.op_type

    # Convert onnx tensors to numpy array
    scale_array = onnx.numpy_helper.to_array(scale)
    zp_array = onnx.numpy_helper.to_array(zp)

    # Dynamically determine transB for Gemm
    trans_b = 0
    if op_type == "Gemm":
        for attr in quantized_node.attribute:
            if attr.name == "transB":
                trans_b = attr.i
                break

    axis_map = {
        "Conv": 0,
        "ConvTranspose": 1,
        "Gemm": 0 if trans_b else 1,
        "MatMul": 1,
    }

    if op_type not in axis_map:
        raise ValueError(f"Unsupported op_type for real weight quantization: {op_type}")

    axis = axis_map[op_type]

    if scale_array.shape and scale_array.shape[0] != weight_shape[axis]:
        raise ValueError(
            f"Scale shape {scale_array.shape} does not match weight shape {weight_shape} along axis {axis}"
        )

    reshape_dims = [1] * len(weight_shape)
    reshape_dims[axis] = scale_array.shape[0]
    scale_array = scale_array.reshape(*reshape_dims)
    zp_array = zp_array.reshape(*reshape_dims)

    # Convert to INT8/FP8
    if zp.data_type == onnx_dtype_map["Float8"]:
        scaled = np.asarray(weight_array / scale_array) + zp_array
    else:
        scaled = np.asarray((weight_array / scale_array).round())
        np.clip(scaled + zp_array, -128, 127, out=scaled)

    return scaled


def _cast_fp8(array: np.ndarray) -> np.ndarray:
    """Cast a numpy array to FLOAT8E4M3FN using PyTorch."""
    array_f32_t = torch.from_numpy(array)
    if torch.cuda.is_available():
        array_f32_t = array_f32_t.cuda()
    array_f8_t = array_f32_t.clamp(min=-448, max=448).to(torch.float8_e4m3fn).view(torch.uint8)
    array_f8 = array_f8_t.cpu().numpy().astype(np.uint8)
    return array_f8


def _create_fp8_tensor(scaled: np.ndarray, weight_name: str) -> onnx.TensorProto:
    """Create a FLOAT8E4M3FN tensor directly from numpy array."""
    fp8_data = _cast_fp8(scaled)
    tensor = onnx.numpy_helper.from_array(fp8_data, weight_name)
    tensor.data_type = onnx_dtype_map["Float8"]
    return tensor


def qdq_to_dq(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32/FP16 weights of the given ONNX model to INT8/FP8 weights.

    This function converts a model with QDQ (QuantizeLinear-DequantizeLinear) nodes to a model
    with only DQ nodes for weights. It:
    1. Converts FP32/FP16 weights to INT8/FP8
    2. Updates the graph to maintain proper connections
    3. Removes redundant cast nodes in the quantized model (additional optimization for diffusers)

    Args:
        onnx_model: ONNX model protobuf to convert

    Returns:
        ONNX model protobuf with only DQ nodes for weights

    Raises:
        ValueError: If the model is invalid or conversion fails
        RuntimeError: If graph operations fail
    """
    logger.info("Converting model with QDQ nodes to DQ only model")
    if not isinstance(onnx_model, onnx.ModelProto):
        raise ValueError("Input must be an ONNX model protobuf")

    graph = onnx_model.graph
    if not graph.node:
        raise ValueError("Model graph is empty")

    initializers, tensor_producers, tensor_consumers = _get_graph_metadata(graph)
    q_nodes = [
        (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "QuantizeLinear"
    ]
    q_indices = []

    for node_idx, node in q_nodes:
        weight_name = node.input[0]
        logger.debug(f"Processing QDQ node for weight {weight_name}")

        # Nothing to do for non-const weight inputs
        if weight_name in tensor_producers:
            continue

        try:
            # Get weight tensor
            if weight_name not in initializers:
                raise ValueError(f"Weight {weight_name} not found in initializers")
            weight = initializers[weight_name]
            weight_array = onnx.numpy_helper.to_array(weight)

            # Get scale and zero point
            scale, zp = _get_scale_and_zp(node, initializers, tensor_producers)

            # Validate Q->DQ->Op pattern and get consumers
            dq_node, quantized_node = _get_successive_consumers(node, tensor_consumers)

            # Convert weight
            scaled = _convert_weight(weight_array, scale, zp, quantized_node)

            # Create and update new weight tensor
            if zp.data_type == onnx_dtype_map["Float8"]:
                new_weight = _create_fp8_tensor(scaled, weight_name)
                logger.debug(f"Converted {weight_name} to FP8")
            else:
                new_weight = onnx.numpy_helper.from_array(scaled.astype("int8"), weight_name)
                logger.debug(f"Converted {weight_name} to INT8")
            weight.CopyFrom(new_weight)

            # Track QuantizeLinear node indices for cleanup
            # Note. Scale and zero point tensors are shared between Q and DQ nodes and should not be deleted
            q_indices.append(node_idx)

            # Update following DQ nodes input name, each q should only have one dq consumer
            consumers = tensor_consumers[node.output[0]]
            assert len(consumers) == 1, f"Expected exactly one consumer for {node.name}"
            dq_node = consumers[0]
            assert dq_node.op_type == "DequantizeLinear", (
                f"Expected DequantizeLinear consumer for {node.name}"
            )
            dq_node.input[0] = weight_name

        except Exception as e:
            raise RuntimeError(f"Failed to convert node {node.name}: {e!s}")

    # Remove processed nodes
    for node_idx in sorted(q_indices, reverse=True):
        del graph.node[node_idx]

    # Remove redundant cast nodes in the quantized model
    # Note. This optimization is used by diffusers through --dq_only option, so keeping it here as well
    remove_redundant_cast_nodes(graph)
    logger.info(f"Removed {len(q_indices)} Q nodes and redundant cast nodes")

    return onnx_model


def remove_input_dq_and_output_q(
    onnx_model: onnx.ModelProto, quantizable_custom_ops: dict
) -> onnx.ModelProto:
    """Remove DQ nodes from the input and Q from the output of quantized custom ops for TensorRT compatibility.

    TensorRT requires only Q nodes in the inputs and only DQ nodes in the outputs of custom ops.
    For more information, see https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html#q-dq-interaction-with-plugins

    Args:
        onnx_model: ONNX model protobuf to convert
        quantizable_custom_ops: dictionary of custom ops and I/O indices to perform Q and DQ deletions as needed.

    Returns:
        ONNX model protobuf with only Q in the inputs and only DQ in the outputs of custom ops.

    Raises:
        ValueError: If the model is invalid or removal fails
        RuntimeError: If graph operations fail
    """
    logger.info("Deleting DQ nodes in the input and Q nodes in the output of custom ops.")
    if not isinstance(onnx_model, onnx.ModelProto):
        raise ValueError("Input must be an ONNX model protobuf")

    graph = onnx_model.graph
    if not graph.node:
        raise ValueError("Model graph is empty")

    initializers, tensor_producers, tensor_consumers = _get_graph_metadata(graph)
    q_nodes = [
        (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "QuantizeLinear"
    ]
    dq_nodes = [
        (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "DequantizeLinear"
    ]
    q_indices = []
    dq_indices = []

    # Remove DQ nodes in the input of custom ops
    for node_idx, node in dq_nodes:
        consumers = tensor_consumers[node.output[0]]
        for inp_name in node.input:
            logger.debug(f"Processing QDQ node for input {inp_name}")

            # Ignore initializers (scale, zero_point)
            if inp_name in initializers:
                continue

            try:
                # Update the previous Q node output name, each DQ should only have one Q producer
                q_node = tensor_producers[inp_name]
                assert isinstance(q_node, onnx.NodeProto), (
                    f"Expected producer {node.name} to be of type NodeProto"
                )
                assert q_node.op_type == "QuantizeLinear", (
                    f"Expected QuantizeLinear producer for {node.name}"
                )

                # Only remove DQs from the inputs of custom ops
                has_cast = consumers[0].op_type == "Cast"
                consumers_2 = tensor_consumers[consumers[0].output[0]] if has_cast else consumers
                if consumers_2[0].op_type not in quantizable_custom_ops:
                    continue

                if has_cast:
                    # Assume that this input tensor is not meant to be quantized as there's a Cast node between DQ
                    # and the custom op. Keep the Cast node and delete both Q/DQ nodes.
                    q_node_prev = tensor_producers.get(q_node.input[0], None)
                    consumers[0].input[0] = (
                        q_node_prev.output[0] if q_node_prev else q_node.input[0]
                    )
                else:
                    # Rewire graph to connect Q with the node after DQ (skip DQ)
                    for consumer in consumers:
                        for cons_idx, cons_inp in enumerate(consumer.input):
                            if cons_inp == node.output[0]:
                                # If the input tensor is meant to be quantized, delete DQ. Otherwise, delete both Q/DQ.
                                if cons_idx in quantizable_custom_ops[consumer.op_type]["inp"]:
                                    consumer.input[cons_idx] = q_node.output[0]
                                else:
                                    q_node_prev = tensor_producers.get(q_node.input[0], None)
                                    consumer.input[cons_idx] = (
                                        q_node_prev.output[0] if q_node_prev else q_node.input[0]
                                    )
                                break

                # Track DequantizeLinear node indices for cleanup
                dq_indices.append(node_idx)

            except Exception as e:
                raise RuntimeError(f"Failed to convert node {node.name}: {e!s}")

    # Remove Q nodes in the output of custom ops
    for node_idx, node in q_nodes:
        for out_name in node.output:
            logger.debug(f"Processing QDQ node for output {out_name}")

            try:
                # Update the Q node output name, each Q should only have one DQ consumer
                dq_node = tensor_consumers[out_name]
                assert len(dq_node) == 1, f"Expected single consumer for {node.name}"
                assert dq_node[0].op_type == "DequantizeLinear", (
                    f"Expected DequantizeLinear producer for {node.name}"
                )

                # Only remove Qs from the output of custom ops
                if (
                    node.input[0] in initializers
                    or get_tensor_from_name(graph, node.input[0]) in graph.input
                ):
                    continue
                producer = tensor_producers[node.input[0]]
                if producer.op_type not in quantizable_custom_ops:
                    continue

                # Rewire graph to connect the output of custom op to the input of DQ (skip Q)
                # If the output tensor is meant to be quantized, delete Q. Otherwise, delete both Q/DQ.
                if quantizable_custom_ops[producer.op_type]["out"]:
                    dq_node[0].input[0] = producer.output[0]
                else:
                    dq_node_next = tensor_consumers.get(dq_node[0].output[0], None)
                    if dq_node_next:
                        dq_node_next[0].input[0] = producer.output[0]
                    else:
                        dq_node[0].input[0] = producer.output[0]

                # Track QuantizeLinear node indices for cleanup
                q_indices.append(node_idx)

            except Exception as e:
                raise RuntimeError(f"Failed to convert node {node.name}: {e!s}")

    # Remove processed nodes
    for node_idx in sorted(q_indices + dq_indices, reverse=True):
        del graph.node[node_idx]

    logger.info(
        f"Removed {len(q_indices)} Q node{'' if len(q_indices) == 1 else 's'} and"
        f" {len(dq_indices)} DQ node{'' if len(dq_indices) == 1 else 's'}"
    )

    # Cleanup graph to remove any dangling Q/DQ nodes
    graph = gs.import_onnx(onnx_model)
    graph.cleanup()
    onnx_model = gs.export_onnx(graph)

    # TODO: remove manual ir_version change once ORT supports ir_version 11
    onnx_model.ir_version = 10

    return onnx_model


def remove_graph_input_q(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove Q nodes from the inputs of a quantized ONNX model.

    This supports generating quantized models with low-precision graph I/O.

    Args:
        onnx_model: ONNX model protobuf to convert

    Returns:
        ONNX model protobuf with only DQ in the inputs whenever possible.

    Raises:
        ValueError: If the model is invalid or removal fails
        RuntimeError: If graph operations fail
    """
    logger.info("Deleting Q nodes in the input of a quantized ONNX model.")
    if not isinstance(onnx_model, onnx.ModelProto):
        raise ValueError("Input must be an ONNX model protobuf")

    graph = onnx_model.graph
    if not graph.node:
        raise ValueError("Model graph is empty")

    initializers, _, tensor_consumers = _get_graph_metadata(graph)
    q_nodes = [
        (idx, node) for idx, node in enumerate(graph.node) if node.op_type == "QuantizeLinear"
    ]
    q_indices = []
    graph_input_names = {inp.name: inp for inp in graph.input}

    # Remove Q nodes in the graph inputs
    for node_idx, node in q_nodes:
        if not any(inp in graph_input_names for inp in node.input):
            continue

        inp = node.input[0]
        for out_name in node.output:
            logger.debug(f"Processing QDQ node for output {out_name}")

            try:
                # Update the Q node output name, each Q should only have one DQ consumer
                dq_node = tensor_consumers[out_name]
                assert len(dq_node) == 1, f"Expected single consumer for {node.name}"
                assert dq_node[0].op_type == "DequantizeLinear", (
                    f"Expected DequantizeLinear producer for {node.name}"
                )

                # Rewire graph to connect the graph input to the output of the Q node
                dq_node[0].input[0] = inp

                # Set the input precision to match the zero-point precision in the DQ node
                inp_tensor = graph_input_names[inp]
                inp_tensor.type.tensor_type.elem_type = initializers[dq_node[0].input[2]].data_type

                # Track QuantizeLinear node indices for cleanup
                q_indices.append(node_idx)

            except Exception as e:
                raise RuntimeError(f"Failed to convert node {node.name}: {e!s}")

    # Remove processed nodes
    for node_idx in sorted(q_indices, reverse=True):
        del graph.node[node_idx]

    logger.info(f"Removed {len(q_indices)} Q node{'' if len(q_indices) == 1 else 's'}")

    # TODO: remove manual ir_version change once ORT supports ir_version 11
    onnx_model.ir_version = 10

    return onnx_model


def replace_zero_scale_with_smallest_nonzero(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace zero scale values with smallest nonzero fp16 value in the ONNX model."""
    graph = onnx_model.graph
    fp16_smallest_nonzero = np.float16(6e-08)
    scale_nodes = [node.input[1] for node in graph.node if node.op_type == "QuantizeLinear"]
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] in scale_nodes:
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = numpy_helper.to_array(attr.t)
                    new_tensor = np.where(tensor == 0, fp16_smallest_nonzero, tensor)
                    attr.t.CopyFrom(numpy_helper.from_array(new_tensor, attr.t.name))
    return onnx_model


def cast_initializer_to_dtype(
    node: onnx.NodeProto, dtype: str, initializer_map: dict[str, onnx.TensorProto]
):
    """Casts the initializer to the given dtype."""
    for id, input_name in enumerate(node.input):
        if input_name in initializer_map:
            input_id = id
    input_name = node.input[input_id]
    input = numpy_helper.to_array(initializer_map[input_name])
    input = input.astype(np_dtype_map[dtype])
    input_onnx = onnx.numpy_helper.from_array(input, input_name)
    input_onnx.data_type = onnx_dtype_map[dtype]
    initializer_map[input_name].CopyFrom(input_onnx)


def get_quantized_tensors(onnx_model: onnx.ModelProto) -> set[str]:
    """Get the names of all quantized tensors from an ONNX model.

    This function identifies all DequantizeLinear nodes in the ONNX model
    and extracts the names of tensors being dequantized (the first input of
    each DequantizeLinear node, excluding scale and zero-point inputs).

    Args:
        onnx_model: ONNX model protobuf to analyze

    Returns:
        Set of tensor names that are inputs to DequantizeLinear nodes
        (i.e., the tensors being dequantized)
    """
    quantized_tensors = set()

    for node in onnx_model.graph.node:
        if node.op_type == "DequantizeLinear":
            # First input is the tensor being dequantized
            # (inputs[1] is scale, inputs[2] is zero-point)
            if node.input and len(node.input) > 0:
                quantized_tensors.add(node.input[0])

    logger.debug(f"Found {len(quantized_tensors)} dequantized tensors in ONNX model")
    return quantized_tensors
