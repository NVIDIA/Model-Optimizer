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

"""Utility functions related to onnx."""

import copy
import io
import os
import tempfile
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx.helper import get_attribute_value
from onnx_graphsurgeon import Constant, Node, Variable

from modelopt.onnx.logging_config import logger

# Base minimum opset for quantization (opset 19 is the first to support fp16 scales)
BASE_MIN_OPSET = 19


def get_input_names_from_bytes(model_bytes: bytes, external_inputs_only: bool = True) -> list[str]:
    """This function returns the inputs names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of input names of the model.
    """
    logger.debug("Getting input names from model bytes")
    model = onnx.load_from_string(model_bytes)
    return get_input_names(model, external_inputs_only)


def get_all_input_names(model: onnx.ModelProto) -> list[str]:
    """This function returns the inputs names of the given onnx model."""
    return [graph_input.name for graph_input in model.graph.input]


def _get_initializer_names(model: onnx.ModelProto) -> list[str]:
    return [initializer.name for initializer in model.graph.initializer]


def get_input_names(model: onnx.ModelProto, external_inputs_only: bool = True) -> list[str]:
    """This function returns the external inputs names of the given onnx model.

    Note: external_input_names = input_names - initializer_names

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of external input names of the model.
    """
    input_names = get_all_input_names(model)
    if not external_inputs_only:
        return input_names

    initializer_names = _get_initializer_names(model)
    external_input_names = list(np.setdiff1d(input_names, initializer_names))
    return external_input_names


def get_output_names_from_bytes(model_bytes: bytes) -> list[str]:
    """This function returns the output names of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        List of output names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_output_names(model)


def get_output_names(model: onnx.ModelProto) -> list[str]:
    """This function returns the output names of the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of output names of the model.
    """
    return [output.name for output in model.graph.output]


def get_node_names_from_bytes(model_bytes: bytes) -> list[str]:
    """This function returns all node names from the given onnx model in bytes.

    Args:
        model: onnx model in bytes.

    Returns:
        List of node names of the model.
    """
    model = onnx.load_from_string(model_bytes)
    return get_node_names(model)


def get_node_names(model: onnx.ModelProto) -> list[str]:
    """This function returns all node names from the given onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        List of node names of the model.
    """
    return [node.name for node in model.graph.node]


def _get_tensor_shape(tensor: onnx.ValueInfoProto) -> list[int]:
    """This function returns the shape of the input onnx tensor.

    Onnx tensors are of type ValueInfoProto and their dimensions are stored in a
    RepeatedCompositeContainer. Each of these dimensions is of type onnx.Dimension.
    In a loop we access each of the Dimension object and create a shape list to return.
    Dynamic dimensions (i.e. with "dim_param" field) are replaced with 1.

    Args:
        tensor: Onnx tensor object for which the shape needs to be computed.

    Returns:
        Shape of the input tensor.
    """
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")

    dimensions = tensor.type.tensor_type.shape.dim
    shape = []
    for dim in dimensions:
        if dim.HasField("dim_param"):
            shape.append(1)
        if dim.HasField("dim_value"):
            if dim.dim_value == -1:
                shape.append(1)
            else:
                shape.append(dim.dim_value)

    return shape


def get_dynamic_graph_inputs(onnx_model: onnx.ModelProto):
    """This function returns the dynamic inputs of an ONNX model.

    Args:
        onnx_model: ONNX model to obtain dynamic inputs from.

    Returns:
        List of dynamic inputs.
    """
    graph = gs.import_onnx(onnx_model)
    return [inp for inp in graph.inputs if any(isinstance(s, str) or s <= 0 for s in inp.shape)]


def _get_all_shapes(container: Any) -> dict[str, list[int]]:
    """This method returns the shape of tensors within a RepeatedCompositeContainer.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names and shape of the tensors within the container.
    """
    results = {}
    for tensor in container:
        results[tensor.name] = _get_tensor_shape(tensor)
    return results


def _get_selected_shapes(container: Any, inputs_to_include: list[str]) -> dict[str, list[int]]:
    """This method returns the shape tensors within a RepeatedCompositeContainer.

    It only computes the shape of the tensors with name containing in `inputs_to_include` list.

    Args:
        container: Model graph input/output container.

    Returns:
        Dictionary of tensor names in inputs_to_include and their shapes.
    """
    results = {}
    for tensor in container:
        if tensor.name in inputs_to_include:
            results[tensor.name] = _get_tensor_shape(tensor)
    return results


def get_input_shapes_from_bytes(model_bytes: bytes) -> dict[str, list[int]]:
    """This function returns the input shapes of the given onnx model in bytes.

    Args:
        model_bytes: Onnx model in bytes.

    Returns:
        Dictionary of inputs names and shapes.
    """
    model = onnx.load_from_string(model_bytes)
    return get_input_shapes(model)


def get_input_shapes(
    model: onnx.ModelProto, external_inputs_only: bool = True
) -> dict[str, list[int]]:
    """This function returns the inputs shapes for the given onnx model."""
    logger.debug("Getting input shapes from model")
    if external_inputs_only:
        return _get_selected_shapes(model.graph.input, get_input_names(model))
    return _get_all_shapes(model.graph.input)


def get_output_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    """This function returns the output shapes for the given onnx model."""
    return _get_all_shapes(model.graph.output)


def parse_shapes_spec(shapes_spec: str) -> dict[str, list[int]]:
    """Parse shapes spec and returns them in a dictionary.

    Example shapes spec: input0:1x3x256x256,input1:1x3x128x128
    """
    shapes = shapes_spec.split(",")
    shape_dict = {}

    for shape_def in shapes:
        tensor_name, shape_str = shape_def.rsplit(":", 1)
        shape_list = list(map(int, shape_str.split("x")))
        shape_dict[tensor_name.strip()] = shape_list

    return shape_dict


def _get_tensor_type(tensor: onnx.ValueInfoProto) -> int:
    if not hasattr(tensor.type, "tensor_type"):
        raise NotImplementedError("Only tensor type inputs are supported.")
    type_ = tensor.type.tensor_type.elem_type
    return type_


def _get_container_types(container, inputs_to_include: list[str] | None = None) -> dict[str, int]:
    results = {}
    for tensor in container:
        if inputs_to_include is not None and tensor.name not in inputs_to_include:
            continue
        t_type = _get_tensor_type(tensor)
        results[tensor.name] = t_type
    return results


def _get_input_types(model: onnx.ModelProto, external_inputs_only: bool = True) -> dict[str, int]:
    inputs_to_include = get_input_names(model, external_inputs_only)
    return _get_container_types(model.graph.input, inputs_to_include)


def _get_output_types(model: onnx.ModelProto) -> dict[str, int]:
    results = _get_container_types(model.graph.output)
    return results


def _convert_types_to_np(types: dict[str, int] | list[int] | int) -> Any:
    if isinstance(types, dict):
        types_np = {}
        for name in types:
            types_np[name] = onnx.helper.tensor_dtype_to_np_dtype(types[name])
        return types_np
    elif isinstance(types, list):
        return [onnx.helper.tensor_dtype_to_np_dtype(type_) for type_ in types]
    else:
        return onnx.helper.tensor_dtype_to_np_dtype(types)


def get_tensor_by_name(
    onnx_model: onnx.ModelProto, tensor_name: str
) -> onnx.ValueInfoProto | onnx.TensorProto | None:
    """This function returns a tensor from its name.

    This function searches for a tensor in the model's:
    1. Value info (shape/type info, no data)
    2. Initializers (TensorProto, contains actual data)
    3. Inputs and outputs

    Args:
        onnx_model: ONNX model.
        tensor_name: tensor name.

    Returns:
        tensor
    """
    tensor_val = next(
        (tens for tens in onnx_model.graph.value_info if tens.name == tensor_name), None
    )
    tensor_init = next(
        (tens for tens in onnx_model.graph.initializer if tens.name == tensor_name), None
    )
    tensor_inp = next((tens for tens in onnx_model.graph.input if tens.name == tensor_name), None)
    tensor_out = next((tens for tens in onnx_model.graph.output if tens.name == tensor_name), None)
    return tensor_val or tensor_init or tensor_inp or tensor_out


def gen_random_inputs(
    model: onnx.ModelProto, shapes_spec: str | None = None
) -> dict[str, np.ndarray]:
    """This function generates random inputs for an onnx model.

    Args:
        model: Loaded in-memory onnx ModelProto.
        shapes_spec: A string representing the shape of each input tensors. The format is
        "<tensor1>:<d1>x<d2>,<tensor2>:<d1>,...". If the shape is not provided for an input tensor, the shape is
        inferred from the onnx model directly, with all the unknown dims filled with 1.

    Returns:
        Dictionary of numpy tensors.
    """
    logger.info("Generating random inputs for model")
    input_dict = {}
    types = _get_input_types(model)
    types_np = _convert_types_to_np(types)
    input_shapes = {} if shapes_spec is None else parse_shapes_spec(shapes_spec)
    logger.debug(f"Using input shapes: {input_shapes}")

    for graph_input in model.graph.input:
        # Generate tensors for external inputs only
        if graph_input.name not in types_np:
            continue

        shape_arr = (
            input_shapes[graph_input.name]
            if graph_input.name in input_shapes
            else _get_tensor_shape(graph_input)
        )

        target_np_type = types_np[graph_input.name]
        if np.issubdtype(target_np_type, np.integer):
            # For integer types, generate random integers in a representative range
            # Example: if it's int32, maybe -1000 to 1000, or 0 to 1000 if non-negative.
            # This needs to be context-aware or have a sensible default.
            # For token IDs (e.g. input_ids), a typical vocab size might be ~50000
            if "ids" in graph_input.name:  # Heuristic for tokenizers
                min_val, max_val = 0, 50000
            elif "mask" in graph_input.name:  # Heuristic for attention masks (0 or 1)
                min_val, max_val = 0, 2  # np.random.randint excludes high, so use 2 for 0,1
            else:  # General integer case, small range
                min_val, max_val = 0, 100
            input_dict[graph_input.name] = np.random.randint(
                min_val, max_val, size=shape_arr
            ).astype(target_np_type)
        elif np.issubdtype(target_np_type, np.floating):
            # For float types, np.random.uniform() is fine, but ensure a decent range
            # if default (0,1) is not good enough. E.g., np.random.uniform(-1.0, 1.0, size=shape_arr)
            input_dict[graph_input.name] = np.random.uniform(
                low=0.0, high=1.0, size=shape_arr
            ).astype(target_np_type)
        else:  # Fallback for other types (e.g. bool)
            input_dict[graph_input.name] = np.random.uniform(size=shape_arr).astype(target_np_type)

    return input_dict


def remove_weights_data(onnx_bytes: bytes) -> bytes:
    """Removes raw weight data from the onnx model."""
    logger.info("Removing weights data from ONNX model")
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer
    weights_removed = 0

    for idx, init in enumerate(inits):
        # Only remove arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                # Setting up some metadata to randomize the weights later
                np_tensor = np.frombuffer(init.raw_data, dtype=dtype)
                meta = model.metadata_props.add()
                meta.key = init.name + "_avg"
                meta.value = str(np.average(np_tensor))

                meta = model.metadata_props.add()
                meta.key = init.name + "_var"
                meta.value = str(np.var(np_tensor))

                # Note that, onnx.checker will fail due to data cleaning
                # We should not check the model till weights are reassigned
                model.graph.initializer[idx].raw_data = b""
                weights_removed += 1

    logger.debug(f"Removed weights data from {weights_removed} tensors")
    buffer = io.BytesIO()
    onnx.save_model(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def randomize_weights(onnx_path: str) -> None:
    """Assigns random values to the onnx model weights."""
    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()
        onnx_bytes = randomize_weights_onnx_bytes(onnx_bytes)

    with open(onnx_path, "wb") as f:
        # Write the modified onnx model to the same path
        f.write(onnx_bytes)


def randomize_weights_onnx_bytes(onnx_bytes: bytes, seed: int = 0) -> bytes:
    """Assigns random values to the onnx model weights."""
    model = onnx.load_from_string(onnx_bytes)
    inits = model.graph.initializer
    np.random.seed(seed)
    weight_metadata = {item.key: item.value for item in model.metadata_props}

    for idx, init in enumerate(inits):
        # Randomize only the arrays with dimension larger than 1
        if len(init.dims) > 1:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            if dtype in ["float16", "float32", "float64"]:
                avg = weight_metadata.get(init.name + "_avg", None)
                var = weight_metadata.get(init.name + "_var", None)
                if avg and var:
                    numpy_array = np.random.normal(float(avg), float(var), size=init.dims).astype(
                        dtype
                    )
                    tensor = onnx.numpy_helper.from_array(numpy_array, init.name)
                    model.graph.initializer[idx].CopyFrom(tensor)

    buffer = io.BytesIO()
    onnx.save_model(model, buffer)
    buffer.seek(0, 0)

    return buffer.read()


def validate_onnx(onnx_bytes: bytes) -> bool:
    """Returns True if the onnx_bytes is valid, else False."""
    logger.info("Validating ONNX model")
    if not onnx_bytes:
        logger.error("Empty ONNX bytes provided")
        return False

    try:
        onnx_model = onnx.load_from_string(onnx_bytes)
        return onnx_model is not None
    except Exception:
        return False


def validate_batch_size(onnx_bytes: bytes, batch_size: int) -> bool:
    """Returns True if all the model inputs has batch dimension equal to batch_size."""
    input_shapes = list(get_input_shapes_from_bytes(onnx_bytes).values())
    return all(shape[0] == batch_size for shape in input_shapes)


def get_batch_size(model: onnx.ModelProto) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    input_shapes = list(get_input_shapes(model).values())
    batch_size = input_shapes[0][0]
    for shape in input_shapes:
        if batch_size != shape[0]:
            # The model does not have the batch dimension
            return 1

    return batch_size


def get_batch_size_from_bytes(onnx_bytes: bytes) -> int:
    """Returns the batch size of the given onnx model.

    Assertion will fail if batch size is not same over all the inputs.
    """
    model = onnx.load_from_string(onnx_bytes)
    return get_batch_size(model)


def save_onnx_bytes_to_dir(onnx_bytes: bytes, onnx_dir: str, onnx_name: str) -> None:
    """Saves the onnx bytes to a directory with specified file name."""
    os.makedirs(onnx_dir, exist_ok=True)
    file_path = os.path.join(onnx_dir, onnx_name + ".onnx")

    try:
        with open(file_path, "wb") as f:
            f.write(onnx_bytes)
        logger.info(f"Onnx model saved as {file_path}")
    except Exception as e:
        logger.error(f"Onnx model exporting as {file_path} failed, error {e!s}")


def name_onnx_nodes(graph: onnx.GraphProto) -> bool:
    """Assigns name to the onnx nodes if not present and return the modified status."""
    is_modified = False
    node_names = {node.name for node in graph.node}
    start_id = len(node_names)
    for node in graph.node:
        if not node.name:
            new_name = f"{node.op_type}_{start_id}"
            while new_name in node_names:
                start_id += 1
                new_name = f"{node.op_type}_{start_id}"

            node.name = new_name
            node_names.add(new_name)
            is_modified = True

    return is_modified


def duplicate_shared_constants(onnx_model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    """Duplicate constant tensors if they are shared."""
    graph = gs.import_onnx(onnx_model)
    name_dict = defaultdict(lambda: 0)

    def _get_unique_name(old_name):
        name_dict[old_name] += 1
        return old_name + "_" + str(name_dict[old_name])

    # Get tensors with shared constant inputs
    tensors = []
    for node in graph.nodes:
        for inp_idx, tensor in enumerate(node.inputs):
            # constant is shared across multiple nodes
            if isinstance(tensor, Constant) and len(tensor.outputs) > 1:
                tensors.append({"tensor": tensor, "inp_node": node, "inp_idx": inp_idx})

    # Duplicate shared tensors
    for tensor_dict in tensors:
        tensor = tensor_dict["tensor"]
        new_tensor = Constant(
            name=_get_unique_name(tensor.name),
            values=tensor.values,
        )
        tensor_dict["inp_node"].inputs[tensor_dict["inp_idx"]] = new_tensor

    onnx_model = gs.export_onnx(graph)
    is_modified = bool(tensors)
    return onnx_model, is_modified


def check_model(model: onnx.ModelProto) -> None:
    """Checks if the given model is valid."""
    if model.ByteSize() > (2 * (1024**3)):  # 2GB limit
        with tempfile.TemporaryDirectory() as temp_dir:
            # ONNX also looks in CWD, so we need to use a unique id
            unique_id = str(uuid.uuid4())[:8]
            onnx_tmp_path = os.path.join(temp_dir, f"model_{unique_id}.onnx")
            save_onnx(model, onnx_tmp_path, save_as_external_data=True)
            onnx.checker.check_model(onnx_tmp_path)
    else:
        onnx.checker.check_model(model)


def find_lowest_common_ancestor(node1: Node, node2: Node) -> tuple[str | None, int, int]:
    """Function to find the lowest common ancestor of two nodes.

    Args:
        node1: First node name.
        node2: Second node name.

    Returns:
        LCA node.
        Distance from first node.
        Distance from second node.
    """

    def _find_ancestors(node: Node):
        ancestors = {node.name: 0}
        stack = [(node, 0)]
        while stack:
            cur_node, distance = stack.pop()
            for parent_node in get_parent_nodes(cur_node):
                if parent_node.name not in ancestors:
                    ancestors[parent_node.name] = distance + 1
                    stack.append((parent_node, distance + 1))

        return ancestors

    ancestors1 = _find_ancestors(node1)
    ancestors2 = _find_ancestors(node2)

    # Find the lowest common ancestor
    common_ancestors = set(ancestors1.keys()).intersection(ancestors2.keys())
    if common_ancestors:
        lowest_common_ancestor = common_ancestors.pop()
        distance = ancestors1[lowest_common_ancestor]
        for t in common_ancestors:
            if ancestors1[t] < distance:
                distance = ancestors1[t]
                lowest_common_ancestor = t
        distance1 = ancestors1[lowest_common_ancestor]
        distance2 = ancestors2[lowest_common_ancestor]
        return lowest_common_ancestor, distance1, distance2
    else:
        return None, -1, -1  # No common ancestor found


def get_parent_nodes(node: Node) -> list[Node]:
    """Returns list of input producer nodes for the given node."""
    # If the tensor is not a constant or graph input and has a producer,
    # the producer is a parent of node `node`
    parents = [tensor.inputs[0] for tensor in node.inputs if len(tensor.inputs) == 1]

    return parents


def get_child_nodes(node: Node) -> list[Node]:
    """Returns list of output consumer nodes for the given node."""
    children = [consumer for tensor in node.outputs for consumer in tensor.outputs]
    return children


def get_variable_inputs(node: Node) -> list[Variable]:
    """Returns the variable inputs of the given Node."""
    var_inputs = [
        tensor
        for tensor in node.inputs
        if isinstance(tensor, Variable)
        and (not tensor.inputs or (tensor.inputs and tensor.inputs[0].op != "Constant"))
    ]
    return var_inputs


def save_onnx(model: onnx.ModelProto, onnx_path: str, save_as_external_data: bool = False):
    """Save an ONNX model to given path. If a model is larger than 2GB, will save with external data."""
    size_threshold = 2 * (1024**3)  # 2GB
    try:
        model_proto = model.SerializeToString()
        model_size = len(model_proto)
        save_as_external_data = save_as_external_data or model_size > size_threshold
        logger.debug(
            f"Model size: {model_size} bytes, using external data: {save_as_external_data}"
        )

    except ValueError as e:
        if "Message onnx.ModelProto exceeds maximum protobuf size of 2GB" in str(e):
            logger.warning("Model exceeds 2GB limit, switching to external data storage")
            save_as_external_data = True
        else:
            logger.error(f"Failed to serialize model: {e!s}")
            raise

    # Set ir_version to 10, remove it once ORT supports ir_version 11
    model.ir_version = 10
    if save_as_external_data:
        external_data_path = os.path.basename(onnx_path) + "_data"
        if os.path.exists(external_data_path):
            logger.warning(f"Removing existing external data file: {external_data_path}")
            os.remove(external_data_path)

        # Copy so the onnx.ModelProto object will not be modified
        model_copy = copy.deepcopy(model)
        onnx.save_model(
            model_copy,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
            size_threshold=1024,
        )
    else:
        onnx.save(model, onnx_path)


def update_domain(onnx_model: onnx.ModelProto, op_type: str, domain: str) -> onnx.ModelProto:
    """Updates the domain of all the nodes of the specified op_type to the specified domain."""
    for node in onnx_model.graph.node:
        if node.op_type == op_type:
            node.domain = domain

    return onnx_model


def get_opset_version(model: onnx.ModelProto) -> int:
    """Returns the opset version of the given model."""
    ai_onnx_domain = [
        opset
        for opset in model.opset_import
        if not opset.domain or opset.domain in ["ai.onnx", "ai.onnx.contrib", "trt.plugins"]
    ]
    return ai_onnx_domain[0].version


def check_model_uses_external_data(model: onnx.ModelProto) -> bool:
    """Checks if the model uses external data. True if any initializer tensor has data_location set to EXTERNAL."""
    return any(
        init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL
        for init in model.graph.initializer
    )


def get_qdq_precisions(model: onnx.ModelProto) -> set:
    """Gets the Q/DQ precision types present in the model.

    Args:
        model: Loaded in-memory onnx ModelProto.

    Returns:
        set: Set of Q/DQ precision types present in the model (e.g., 'float8_e4m3fn', 'int8',
             'int4', 'float4_e2m1fn').
    """
    graph = gs.import_onnx(model)
    precisions = set()

    # Check for custom 'NVFP4' nodes
    custom_fp4_q_nodes = [node for node in graph.nodes if node.op == "TRT_FP4DynamicQuantize"]
    if custom_fp4_q_nodes:
        precisions.add("float4_e2m1fn")

    # Check for precision in DQ nodes
    dq_nodes = [node for node in graph.nodes if node.op == "DequantizeLinear"]
    for dq_node in dq_nodes:
        if len(dq_node.inputs) >= 3 and dq_node.inputs[2] is not None:
            # If zero-point is set, return that as the quantization mode
            if isinstance(dq_node.inputs[2], Constant) and dq_node.inputs[2].values is not None:
                precisions.add(dq_node.inputs[2].values.dtype.name)
        elif isinstance(dq_node.inputs[0], Constant) and dq_node.inputs[0].values is not None:
            # Else, return the node's input precision (ex: 'NVFP4' weight quantization)
            precisions.add(dq_node.inputs[0].values.dtype.name)

    return precisions


# Minimum opset requirements by quantization mode/precision
# Base minimum is 19 (first opset that allows fp16 scales in Q/DQ nodes)
# Supports both quantize modes (e.g., "fp8") and dtype prefixes (e.g., "float8" for "float8_e4m3fn")
QDQ_PRECISION_MIN_OPSET = {
    "int8": BASE_MIN_OPSET,
    "float8_e4m3fn": BASE_MIN_OPSET,
    "int4": 21,
    "uint4": 21,
    "float4_e2m1fn": 23,
}


def get_min_opset_for_precisions(precisions: set) -> int:
    """Gets the minimum required opset version for a set of Q/DQ precision types.

    Args:
        precisions: Set of precision type strings (e.g., 'float8_e4m3fn', 'int4').

    Returns:
        int: Minimum required opset version for the given precisions.
    """
    min_opset = BASE_MIN_OPSET  # Base minimum for fp16 scales support
    for precision in precisions:
        # Direct lookup first
        if precision in QDQ_PRECISION_MIN_OPSET:
            min_opset = max(min_opset, QDQ_PRECISION_MIN_OPSET[precision])
    return min_opset


def bfloat16_to_float32(bf16_array):
    """Converts a bfloat16 array (as raw data) to a float32 array."""
    uint32_array = bf16_array.astype(np.uint32) << 16
    return uint32_array.view(np.float32)


def read_f16_tensor_as_fp32(tensor):
    """Reads a float16 or bfloat16 tensor as a float32 numpy ndarray."""
    if tensor.data_type == onnx.TensorProto.BFLOAT16:
        raw_data = tensor.raw_data
        uint16_array = np.frombuffer(raw_data, dtype=np.uint16)
        float32_array = bfloat16_to_float32(uint16_array)
        tensor_shape = tuple(dim for dim in tensor.dims)
        return float32_array.reshape(tensor_shape)

    # Read FLOAT16 tensor and return
    return onnx.numpy_helper.to_array(tensor).astype(np.float32)


def has_attribute(node: onnx.NodeProto, attr_name: str) -> bool:
    """Checks if the given node has the specified attribute."""
    return any(attr.name == attr_name for attr in node.attribute)


def get_attribute(node: onnx.NodeProto, attr_name: str) -> Any:
    """Returns the value of the specified attribute."""
    for attr in node.attribute:
        if attr.name == attr_name:
            return get_attribute_value(attr)
    raise ValueError(f"Attribute {attr_name} not found in node {node.name}")


def _infer_types_only(model: onnx.ModelProto) -> onnx.ModelProto:
    """Infers types (but not shapes) of the onnx graph using local implementation.

    This is an internal function. Use infer_types() as the public API.

    This is a workaround for cases when ONNX's shape inference fails.
    ONNX's infer_shapes performs both shape and type inference together, but for AutoCast, we only
    need type inference.

    Args:
        model: ONNX model to infer types for.

    Returns:
        onnx.ModelProto: Model with inferred types updated in value_info and outputs.
    """
    from modelopt.onnx.autocast import utils as autocast_utils

    # Get opset version
    opset = get_opset_version(model)

    # Process each graph (main graph and all subgraphs) recursively
    def infer_types_for_graph(
        graph: onnx.GraphProto, parent_node: onnx.NodeProto = None, is_subgraph: bool = False
    ) -> None:
        """Infer types for a single graph (main or subgraph).

        Args:
            graph: The graph to infer types for.
            parent_node: The parent node containing this subgraph (None for main graph).
            is_subgraph: Whether this is a subgraph (True) or the main graph (False).
        """
        # Use graphsurgeon to topologically sort nodes for efficient single-pass traversal
        # Create a temporary model with just this graph for graphsurgeon
        temp_model = onnx.ModelProto()
        temp_model.graph.CopyFrom(graph)
        temp_model.opset_import.add().version = opset
        temp_model.ir_version = model.ir_version

        try:
            gs_graph = gs.import_onnx(temp_model)
            gs_graph.toposort()
            # Convert back to ONNX to get topologically sorted nodes
            sorted_model = gs.export_onnx(gs_graph)
            sorted_graph = sorted_model.graph
        except Exception as e:
            logger.debug(
                f"Graphsurgeon toposort failed for {'subgraph' if is_subgraph else 'main graph'},"
                f"using original order: {e!s}"
            )
            # Fallback: process nodes in original order
            sorted_graph = graph

        # Create mappings for quick lookup for this graph
        initializer_map = {init.name: init for init in graph.initializer}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        output_names = {out.name for out in graph.output}

        # Map tensor names to their inferred types (scoped to this graph)
        tensor_types = {}

        # Initialize types from inputs and initializers
        for inp in graph.input:
            if inp.type.HasField("tensor_type"):
                tensor_types[inp.name] = inp.type.tensor_type.elem_type

        for init_name, init in initializer_map.items():
            tensor_types[init_name] = init.data_type

        # Helper function to get tensor type
        def get_tensor_type_from_name(tensor_name: str) -> int | None:
            if tensor_name in tensor_types:
                return tensor_types[tensor_name]
            if tensor_name in value_info_map:
                vi = value_info_map[tensor_name]
                return _get_tensor_type(vi)
            return None

        # Process nodes in topological order (single pass)
        for node in sorted_graph.node:
            # Get input types for this node
            input_types = []
            for inp_name in node.input:
                # an empty tensor name is typically a sign of an optional input, skip it
                if not inp_name:
                    continue
                inp_type = get_tensor_type_from_name(inp_name)
                if inp_type is None:
                    raise ValueError(f"Input {inp_name} of node {node.name} has unknown type")
                input_types.append(inp_type)

            # Infer output types for this node
            output_types = []

            if node.op_type == "Cast":
                # Cast node: output type is the 'to' attribute
                cast_to_type = None
                for attr in node.attribute:
                    if attr.name == "to":
                        cast_to_type = attr.i
                        break
                if cast_to_type is None:
                    raise ValueError(f"Cast node {node.name} has unknown target type")
                output_types = [cast_to_type]
            elif node.op_type == "DequantizeLinear":
                # DequantizeLinear: output type is determined by output_dtype attribute if present,
                # otherwise use the scale type (input[1])
                # inputs: [data, scale, zero_point (optional)]
                output_dtype = None
                for attr in node.attribute:
                    if attr.name == "output_dtype":
                        output_dtype = attr.i
                        break

                if output_dtype is not None:
                    output_types = [output_dtype]
                elif len(node.input) >= 2 and node.input[1]:
                    scale_type = get_tensor_type_from_name(node.input[1])
                    if scale_type is not None:
                        output_types = [scale_type]
                    else:
                        # Fallback: use first input type or FLOAT
                        output_types = [input_types[0] if input_types else onnx.TensorProto.FLOAT]
                else:
                    # Fallback: use first input type or FLOAT
                    output_types = [input_types[0] if input_types else onnx.TensorProto.FLOAT]
            elif node.op_type == "QuantizeLinear":
                # QuantizeLinear: output type is determined by output_dtype attribute if present,
                # otherwise use the zero_point type (input[2])
                # inputs: [data, scale, zero_point]
                output_dtype = None
                for attr in node.attribute:
                    if attr.name == "output_dtype":
                        output_dtype = attr.i
                        break

                if output_dtype is not None:
                    output_types = [output_dtype] * len(node.output)
                elif len(node.input) >= 3 and node.input[2]:
                    zero_point_type = get_tensor_type_from_name(node.input[2])
                    if zero_point_type is not None:
                        output_types = [zero_point_type]
                    else:
                        # Fallback: use INT8 as fallback, since TRT doesn't support UINT8
                        output_types = [onnx.TensorProto.INT8]
                else:
                    # Fallback: use INT8 as fallback, since TRT doesn't support UINT8
                    output_types = [onnx.TensorProto.INT8]
            elif node.op_type == "Constant":
                # Constant: output type is from the value attribute's tensor data_type
                const_type = None
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        if attr.t.HasField("data_type"):
                            const_type = attr.t.data_type
                            break
                assert const_type is not None
                output_types = [const_type]
            elif node.op_type == "ConstantOfShape":
                # ConstantOfShape: output type is from the value attribute's tensor data_type
                # If no value attribute, defaults to FLOAT
                # Note: Schema allows multiple types, so we need to check the value attribute
                const_type = None
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        if attr.t.HasField("data_type"):
                            const_type = attr.t.data_type
                            break
                assert const_type is not None
                output_types = [const_type]
            elif node.op_type == "Split":
                # Split schema allows multiple outputs, but the schema only specifies one output type
                output_types = [input_types[0]] * len(node.output)
            else:
                # Check if this node has subgraphs (GRAPH or GRAPHS attributes)
                # Common nodes with subgraphs: If, Loop, Scan
                subgraphs = []
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        subgraphs.append(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        subgraphs.extend(attr.graphs)

                # If node has subgraphs, infer types for them first
                if subgraphs:
                    for subgraph in subgraphs:
                        infer_types_for_graph(subgraph, parent_node=node, is_subgraph=True)

                    # For nodes with subgraphs, try to infer output types from subgraph outputs
                    # This avoids incorrectly matching to control inputs (e.g., condition for If, trip_count for Loop)
                    output_types = []
                    if len(node.output) > 0:
                        # Use the first subgraph as reference (works for If, Loop, Scan)
                        first_subgraph = subgraphs[0]
                        for out_idx, out_name in enumerate(node.output):
                            if out_idx < len(first_subgraph.output):
                                subgraph_out = first_subgraph.output[out_idx]
                                # Typically we only have one subgraph, but If nodes have two subgraphs
                                # (then_branch and else_branch). In any case, the output types of the
                                # subgraphs must be identical, so we check just the first one
                                if (
                                    subgraph_out.type.HasField("tensor_type")
                                    and subgraph_out.type.tensor_type.elem_type
                                    != onnx.TensorProto.UNDEFINED
                                ):
                                    output_types.append(subgraph_out.type.tensor_type.elem_type)
                            else:
                                output_types.append(onnx.TensorProto.FLOAT)
                    else:
                        # Fallback if we can't infer from subgraphs
                        output_types = None

                    # If we couldn't infer from subgraphs, fall through to schema-based inference
                    if output_types is None or len(output_types) != len(node.output):
                        output_types = None
                else:
                    # No subgraphs, proceed with normal inference
                    output_types = None

                # If output_types not set yet, use schema-based inference
                if output_types is None:
                    default_type = input_types[0] if input_types else onnx.TensorProto.FLOAT
                    # Use ONNX operator schema to determine output types
                    try:
                        schema = onnx.defs.get_schema(node.op_type, opset, domain=node.domain or "")
                        assert schema.outputs and len(schema.outputs) >= len(node.output)
                    except Exception as e:
                        # Fallback: if schema lookup fails, propagate first input type
                        logger.debug(
                            f"Node {node.name}: Failed to get schema for {node.op_type}: {e}, "
                            "propagate first input type"
                        )
                        default_type = input_types[0] if input_types else onnx.TensorProto.FLOAT
                        output_types = [default_type] * len(node.output)
                    else:
                        # Try to infer from schema
                        input_schemas = [
                            schema.inputs[i].type_str for i in range(len(schema.inputs))
                        ]
                        output_schemas = [
                            schema.outputs[i].type_str for i in range(len(schema.outputs))
                        ]
                        output_types = [None] * len(node.output)

                        for output_idx in range(len(node.output)):
                            # explicit type is set in schema, use it
                            if "tensor" in output_schemas[output_idx]:
                                found_type = onnx_type_str_to_enum(output_schemas[output_idx])
                                output_types[output_idx] = found_type
                                continue
                            # sometimes output type is set with a placeholder name despite supporting a single type
                            # e.g. Shape operator is constrained to int64, but the type_str is "T1"
                            for constraint in schema.type_constraints:
                                # If output type constraint has only one allowed type, use it directly
                                if constraint.type_param_str == output_schemas[output_idx]:
                                    if len(constraint.allowed_type_strs) == 1:
                                        found_type = onnx_type_str_to_enum(
                                            constraint.allowed_type_strs[0]
                                        )
                                        output_types[output_idx] = found_type
                                        break
                            else:
                                # We have a placeholder name "T", "T1", "T2", etc that should
                                # match one of the input types
                                try:
                                    input_match_idx = input_schemas.index(
                                        output_schemas[output_idx]
                                    )
                                except ValueError:
                                    input_match_idx = None
                                if input_match_idx is not None:
                                    found_type = input_types[input_match_idx]
                                else:
                                    found_type = default_type
                                    logger.debug(
                                        f"Node {node.name}: Failed to infer type for output "
                                        f"#{output_idx}, propagate first input type"
                                    )
                                output_types[output_idx] = found_type

            # Update output tensor types
            for out_idx, out_name in enumerate(node.output):
                if not out_name or out_idx >= len(output_types):
                    continue

                output_type = output_types[out_idx]
                tensor_types[out_name] = output_type

                # Update value_info if it exists
                if out_name in value_info_map:
                    value_info_map[out_name].type.tensor_type.elem_type = output_type
                elif out_name not in output_names:
                    # Create new value_info for intermediate tensor
                    new_vi = graph.value_info.add()
                    new_vi.name = out_name
                    new_vi.type.tensor_type.elem_type = output_type
                    value_info_map[out_name] = new_vi

        # Update output types for this graph
        for out in graph.output:
            if out.name in tensor_types:
                out.type.tensor_type.elem_type = tensor_types[out.name]

    # Process main graph and all subgraphs recursively
    autocast_utils.walk_subgraphs_recursive(model.graph, infer_types_for_graph, is_subgraph=False)
    infer_types_verification(model)
    return model


def infer_types_verification(model: onnx.ModelProto) -> onnx.ModelProto:
    """Verify that all reachable tensors have a defined type.

    This is necessary because some nodes may be removed during the inference process,
    leaving unreachable value_info entries.
    """
    reachable_tensors = set()

    # Add graph inputs as reachable
    for inp in model.graph.input:
        reachable_tensors.add(inp.name)

    # Add initializers as reachable
    for init in model.graph.initializer:
        reachable_tensors.add(init.name)

    # Traverse nodes to find all reachable tensor outputs
    for node in model.graph.node:
        # A node is reachable if any of its inputs are reachable
        # (or if it has no inputs - rare but possible)
        node_is_reachable = not node.input or any(
            inp in reachable_tensors for inp in node.input if inp
        )

        if node_is_reachable:
            # All outputs of a reachable node are reachable
            for out in node.output:
                if out:  # Skip empty output names
                    reachable_tensors.add(out)

    is_undefined = False
    # Check value_info for reachable tensors
    for vi in model.graph.value_info:
        if vi.name in reachable_tensors:
            if vi.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
                logger.error(
                    f"Infer types verification failed. Value info {vi.name} has undefined type"
                )
                is_undefined = True

    # Graph outputs should always be reachable
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
            logger.error(f"Infer types verification failed. Output {out.name} has undefined type")
            is_undefined = True
    if is_undefined:
        raise ValueError(
            "Infer types verification failed. Undefined types found in the model - see logs for details."
        )
    return model


def infer_shapes(model: onnx.ModelProto, **kwargs):
    """Infers shapes of the onnx graph, handles large models."""
    if model.ByteSize() > (2 * (1024**3)):  # 2GB limit
        with tempfile.TemporaryDirectory() as temp_dir:
            # ONNX also looks in CWD, so we need to use a unique id
            unique_id = str(uuid.uuid4())[:8]
            onnx_orig_path = os.path.join(temp_dir, f"model_{unique_id}.onnx")
            onnx_inferred_path = os.path.join(temp_dir, f"inferred_{unique_id}.onnx")
            save_onnx(model, onnx_orig_path, save_as_external_data=True)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path, **kwargs)
            model = onnx.load(onnx_inferred_path)
        return model
    else:
        return onnx.shape_inference.infer_shapes(model, **kwargs)


def infer_types(
    model: onnx.ModelProto, use_standalone_type_inference: bool = False, **kwargs
) -> onnx.ModelProto:
    """Infers types (and optionally shapes) based on the use_standalone_type_inference flag.

    When use_standalone_type_inference is True, uses a standalone type inference implementation
    that only infers types. Otherwise, uses ONNX's infer_shapes which infers both types and shapes.

    Args:
        model: ONNX model to infer types/shapes for.
        use_standalone_type_inference: If True, use standalone type inference (_infer_types_only).
                                       If False, use ONNX's shape inference (infer_shapes).
        **kwargs: Additional arguments passed to infer_shapes when not using standalone type inference.

    Returns:
        onnx.ModelProto: Model with inferred types (and shapes if not using standalone type inference).
    """
    if use_standalone_type_inference:
        return _infer_types_only(model)
    else:
        return infer_shapes(model, **kwargs)


def onnx_type_str_to_enum(dtype: str) -> int:
    """Converts ONNX type in string format to onnx.TensorProto format.

    Example: 'tensor(float16)' becomes onnx.TensorProto.FLOAT16

    Args:
        dtype: ONNX type in string format.

    Returns:
        int: ONNX type in enum format.
    """
    dtype = dtype.split("tensor(")[-1].split(")")[0]
    dtype = "FLOAT" if dtype == "float32" else dtype.upper()
    return getattr(onnx.TensorProto, dtype)


def remove_node_training_mode(onnx_model: onnx.ModelProto, node_op_type: str) -> onnx.ModelProto:
    """Remove `training_mode` attribute and extra training outputs from nodes of a given op type.

    This also removes the unused outputs from the training_mode nodes.

    Args:
        onnx_model: The onnx model.
        node_op_type: The node type to remove training_mode attribute from.

    Returns:
        The onnx model with the training_mode attribute removed.
    """
    removed_output_names = set()
    all_inputs = {inp for n in onnx_model.graph.node for inp in n.input}
    graph_outputs = {o.name for o in onnx_model.graph.output}
    keep = all_inputs | graph_outputs

    for node in onnx_model.graph.node:
        if node.op_type != node_op_type:
            continue

        is_training_mode = False
        # Drop the 'training_mode' attribute if present
        for idx, attr in enumerate(list(node.attribute)):
            if attr.name == "training_mode":
                del node.attribute[idx]
                if attr.i == 1:
                    is_training_mode = True
                break

        # If the node has extra outputs, remove them all including the training outputs
        if is_training_mode:
            to_remove = []
            for name in node.output:
                if name not in keep:
                    removed_output_names.add(name)
                    to_remove.append(name)

            for name in to_remove:
                node.output.remove(name)

    if removed_output_names:
        # Clean up corresponding value_info entries
        keep = [vi for vi in onnx_model.graph.value_info if vi.name not in removed_output_names]
        del onnx_model.graph.value_info[:]
        onnx_model.graph.value_info.extend(keep)

    return onnx_model
