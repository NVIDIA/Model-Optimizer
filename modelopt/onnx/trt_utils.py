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

"""This module contains TensorRT utils."""

import ctypes
import platform

import lief
import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.utils import (
    get_dynamic_graph_inputs,
    get_tensor_by_name,
    parse_shapes_spec,
    save_onnx,
)

try:
    import tensorrt as trt

    TRT_PYTHON_AVAILABLE = True
except ImportError:
    TRT_PYTHON_AVAILABLE = False

MAX_IR_VERSION = 10


def _is_static_plugin(plugin_path: str) -> bool:
    """Check if a plugin is statically linked against 'libnvinfer'.

    Args:
        plugin_path: Path to the plugin .so file.

    Returns:
        True if plugin is statically linked, False otherwise.
    """
    try:
        return any(dep.startswith("libnvinfer.so") for dep in lief.parse(plugin_path).libraries)
    except Exception as e:
        logger.debug(
            f"Could not find 'libnvinfer.so' link in {plugin_path}: {e}. Assuming it's dynamically linked."
        )
        return False


def _load_trt_plugin(plugin_path: str, registry) -> None:
    """Load a TensorRT plugin using the appropriate method.

    Static plugins (linked against 'libnvinfer') are loaded with ctypes.CDLL to avoid double-loading TensorRT and
    thus avoid SegFault error. Dynamic plugins use registry.load_library for proper TensorRT registration.

    Args:
        plugin_path: Path to the plugin .so file.
        registry: TensorRT plugin registry.
    """
    is_static = _is_static_plugin(plugin_path)
    logger.debug(f"Loading {'static' if is_static else 'dynamic'} plugin: {plugin_path}")

    if is_static:
        # Static plugins: use ctypes.CDLL to avoid double-loading 'libnvinfer'.
        ctypes.CDLL(plugin_path)
    else:
        # Dynamic plugins: use registry.load_library for proper TensorRT registration, falling back to ctypes.CDLL
        # as needed.
        try:
            registry.load_library(plugin_path)
        except Exception as e:
            logger.warning(f"registry.load_library() failed: {e}, falling back to ctypes.CDLL()")
            ctypes.CDLL(plugin_path)


def get_custom_layers(
    onnx_path: str | onnx.ModelProto,
    trt_plugins: list[str] | None,
    strongly_typed: bool = False,
) -> tuple[list[str], dict]:
    """Gets custom layers in ONNX model.

    Args:
        onnx_path: Path or ModelProto of the input ONNX model.
        trt_plugins: list with paths to custom TensorRT plugins.
        strongly_typed: Boolean indicating whether to run TensorRT inference in stronglyTyped mode or not.

    Returns:
        List of custom layers.
        Dictionary containing tensors information: {'tensor_name': {'shape': tensor.shape, 'dtype': tensor.dtype}}
    """
    logger.debug("Checking for custom TensorRT ops")

    # Initialize TensorRT logger and plugins
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")

    # Load custom TensorRT plugins
    if trt_plugins:
        logger.debug(f"Loading {len(trt_plugins)} TensorRT plugin(s)")
        registry = trt.get_plugin_registry()

        for plugin in trt_plugins:
            _load_trt_plugin(plugin, registry)

        logger.debug(f"Total registered plugin creators: {len(registry.plugin_creator_list)}")
    builder = trt.Builder(trt_logger)
    network = (
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        if strongly_typed
        else builder.create_network()
    )
    logger.debug("Created TensorRT builder and network")

    # Parse ONNX file
    parser = trt.OnnxParser(network, trt_logger)
    parser_func = parser.parse_from_file if isinstance(onnx_path, str) else parser.parse
    onnx_path = onnx_path if isinstance(onnx_path, str) else onnx_path.SerializeToString()
    if not parser_func(onnx_path):
        error_str = [str(parser.get_error(error)) for error in range(parser.num_errors)]
        raise Exception(f"Failed to parse ONNX file: {''.join(error_str)}")

    # Obtain layer info
    custom_layers = []
    all_tensor_info = {}
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)

        # Obtain plugin layer names
        if "PLUGIN" in str(layer.type):
            custom_layers.append(layer.name)
            logger.debug(f"Found custom layer: {layer.name}")

        # Collect all tensors' type and shape.
        # Replace dynamic axis representation from -1 to 'unk' for 'onnxsim' support if enabled.
        for i in range(layer.num_inputs):
            input_tensor = layer.get_input(i)
            if input_tensor:
                all_tensor_info[input_tensor.name] = {
                    "shape": ["unk" if (s == -1) else s for s in input_tensor.shape],
                    "dtype": input_tensor.dtype,
                }

        for i in range(layer.num_outputs):
            output_tensor = layer.get_output(i)
            if output_tensor and not all_tensor_info.get(output_tensor.name, None):
                all_tensor_info[output_tensor.name] = {
                    "shape": ["unk" if (s == -1) else s for s in output_tensor.shape],
                    "dtype": output_tensor.dtype,
                }

    logger.info(f"Found {len(custom_layers)} custom layers and {len(all_tensor_info)} tensors")
    return custom_layers, all_tensor_info


def infer_types_shapes(model: onnx.ModelProto, all_tensor_info: dict) -> onnx.ModelProto:
    """Updates tensor shapes in ONNX graph.

    Args:
        model: ONNX model.
        all_tensor_info: Dictionary containing tensors information.

    Returns:
        onnx.ModelProto: ONNX model with inferred types and shapes.
    """
    logger.debug("Inferring types and shapes for graph tensors")

    def _map_trt_to_onnx_type(trt_type: trt.DataType):
        trt_to_onnx_dtype_mapping = {
            trt.float32: onnx.TensorProto.FLOAT,
            trt.float16: onnx.TensorProto.FLOAT16,
            trt.bfloat16: onnx.TensorProto.BFLOAT16,
            trt.int4: onnx.TensorProto.INT4,
            trt.int8: onnx.TensorProto.INT8,
            trt.uint8: onnx.TensorProto.UINT8,
            trt.int32: onnx.TensorProto.INT32,
            trt.int64: onnx.TensorProto.INT64,
            trt.bool: onnx.TensorProto.BOOL,
            trt.fp8: onnx.TensorProto.FLOAT8E4M3FN,
            trt.fp4: onnx.TensorProto.FLOAT4E2M1,
        }
        try:
            return trt_to_onnx_dtype_mapping[trt_type]
        except TypeError as e:
            logger.warning(f"{e}. TRT datatype: {trt_type}. Setting to None")
            return None

    def _create_tensor_shape_proto_from_np_arr(np_arr):
        new_shape_proto = onnx.TensorShapeProto()
        for dim_val in np_arr:
            dim = onnx.TensorShapeProto.Dimension()
            setattr(dim, "dim_param" if isinstance(dim_val, str) else "dim_value", dim_val)
            new_shape_proto.dim.append(dim)
        return new_shape_proto

    for node in model.graph.node:
        for out in node.output:
            if out not in all_tensor_info:
                continue

            tensor = get_tensor_by_name(model, out)
            if isinstance(tensor, onnx.ValueInfoProto):
                if not tensor.type.tensor_type.elem_type:
                    tensor.type.tensor_type.elem_type = _map_trt_to_onnx_type(
                        all_tensor_info[tensor.name]["dtype"]
                    )
                if all_tensor_info[tensor.name]["shape"]:
                    tensor.type.tensor_type.shape.CopyFrom(
                        _create_tensor_shape_proto_from_np_arr(
                            all_tensor_info[tensor.name]["shape"]
                        )
                    )
            elif tensor is None:
                tensor = onnx.helper.make_tensor_value_info(
                    name=out,
                    elem_type=_map_trt_to_onnx_type(all_tensor_info[out]["dtype"]),
                    shape=all_tensor_info[out]["shape"],
                )
                model.graph.value_info.append(tensor)

    logger.info("Updated tensors with type and shape information")

    # Topologically sort graph
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)

    return model


def set_trt_plugin_domain(model: onnx.ModelProto, custom_ops: list[str]) -> onnx.ModelProto:
    """Set TensorRT plugin domain info in the graph.

    Args:
        model: ONNX model to set custom op domain.
        custom_ops: list of custom ops.

    Returns:
        onnx.ModelProto: ONNX model with domain set in custom ops.
    """
    logger.info(f"Found custom operators: {custom_ops}")
    trt_plugin_domain = "trt.plugins"
    trt_plugin_version = 1

    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op in custom_ops:
            # Add TRT domain to each custom node
            node.domain = trt_plugin_domain

    # Add TRT domain and version to the graph
    model = gs.export_onnx(graph)
    model.opset_import.append(onnx.helper.make_opsetid(trt_plugin_domain, trt_plugin_version))
    logger.info(f"Added TRT plugin domain {trt_plugin_domain} version {trt_plugin_version}")
    return model


def infer_types_shapes_tensorrt(
    model: onnx.ModelProto,
    trt_plugins: list[str] = [],
    all_tensor_info: dict = {},
    strongly_typed: bool = False,
) -> onnx.ModelProto:
    """Update tensor types and shapes from TensorRT inference data.

    Args:
        model: ONNX model to infer types and shapes.
        trt_plugins: list of TensorRT plugin library paths in .so format (compiled shared library).
        all_tensor_info: dictionary with tensor data from TensorRT run.
        strongly_typed: boolean indicating if the TensorRT run should be stronglyTyped or not.

    Returns:
        onnx.ModelProto: ONNX model with inferred types and shapes.
    """
    # Obtain tensor data if not given
    if not all_tensor_info:
        _, all_tensor_info = get_custom_layers(model, trt_plugins, strongly_typed)

    # Ensure that all tensors in the graph have type and shape info
    return infer_types_shapes(model, all_tensor_info)


def load_onnx_model(
    onnx_path: str,
    trt_plugins: list[str] | None = None,
    override_shapes: str | None = None,
    use_external_data_format: bool = False,
    intermediate_generated_files: list[str] | None = None,
) -> tuple[onnx.ModelProto, bool, list[str], str, bool]:
    """Load ONNX model. If 'tensorrt' is installed, check if the model has custom ops and ensure it's supported by ORT.

    Args:
        onnx_path: Path to the input ONNX model.
        trt_plugins: List with paths to custom TensorRT plugins.
        override_shapes: Override model input shapes with static shapes.
        use_external_data_format: If True, separate data path will be used to store the weights of the quantized model.
        intermediate_generated_files: List of paths of intermediate ONNX files, generated during quantization.

    Returns:
        Loaded ONNX model supported by ORT.
        Boolean indicating whether the model has custom ops or not.
        List of custom ops in the ONNX model.
        Path to new intermediary ONNX model.
        Boolean indicating whether we should use external data format for the intermediate and quantized models.
    """
    custom_ops = []
    has_custom_op = False

    # Infer shapes
    onnx.shape_inference.infer_shapes_path(onnx_path)

    # Load the model and weights
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    size_threshold = 2 * (1024**3)  # 2GB
    use_external_data_format = onnx_model.ByteSize() > size_threshold or use_external_data_format

    # If inputs are dynamic and override shapes are given, set them as static
    dynamic_inputs = get_dynamic_graph_inputs(onnx_model)
    static_shaped_onnx_path = None
    if len(dynamic_inputs) > 0:
        input_names = [inp.name for inp in dynamic_inputs]
        logger.info(f"Model has dynamic inputs: {input_names}")

        if override_shapes:
            override_shapes_arr = parse_shapes_spec(override_shapes)
            for graph_input in onnx_model.graph.input:
                if graph_input.name not in input_names:
                    continue
                inp_shapes = override_shapes_arr[graph_input.name]
                logger.info(f"Setting '{graph_input.name}' shape to {inp_shapes}")
                for idx, (d, s) in enumerate(
                    zip(graph_input.type.tensor_type.shape.dim, inp_shapes)
                ):
                    if not d.dim_value:
                        graph_input.type.tensor_type.shape.dim[idx].dim_value = s

            static_shaped_onnx_path = onnx_path.replace(".onnx", "_static.onnx")
            save_onnx(onnx_model, static_shaped_onnx_path, use_external_data_format)
            if intermediate_generated_files is not None:
                intermediate_generated_files.append(static_shaped_onnx_path)

    if TRT_PYTHON_AVAILABLE and platform.system() != "Windows":
        # Check if there's a custom TensorRT op in the ONNX model. If so, make it ORT compatible by adding
        # `trt.plugins to the ONNX graph.
        custom_layers, all_tensor_info = get_custom_layers(
            static_shaped_onnx_path or onnx_path, trt_plugins
        )
        has_custom_op = bool(custom_layers)

        if has_custom_op:
            logger.debug(f"Found custom layers: {custom_layers}")
            custom_ops = {
                node.op_type for node in onnx_model.graph.node if node.name in custom_layers
            }

            # Set TensorRT plugin domain info in the graph for ORT compatibility
            onnx_model = set_trt_plugin_domain(onnx_model, custom_ops)

            # Infer types and shapes in the graph for ORT compatibility
            onnx_model = infer_types_shapes_tensorrt(onnx_model, trt_plugins or [], all_tensor_info)

    # Ensure nodes are topologically sorted
    graph = gs.import_onnx(onnx_model)
    graph.toposort()
    onnx_model = gs.export_onnx(graph)

    # Enforce IR version = 10
    ir_version_onnx_path = None
    if onnx_model.ir_version > MAX_IR_VERSION:
        onnx_model.ir_version = MAX_IR_VERSION
        ir_version_onnx_path = (
            static_shaped_onnx_path.replace(".onnx", f"_ir{MAX_IR_VERSION}.onnx")
            if static_shaped_onnx_path
            else onnx_path.replace(".onnx", f"_ir{MAX_IR_VERSION}.onnx")
        )
        save_onnx(onnx_model, ir_version_onnx_path, use_external_data_format)
        if intermediate_generated_files is not None:
            intermediate_generated_files.append(ir_version_onnx_path)

    # Check that the model is valid
    if use_external_data_format:
        # For large models, use the file path to avoid protobuf size limitation
        model_path_to_check = ir_version_onnx_path or static_shaped_onnx_path or onnx_path
        onnx.checker.check_model(model_path_to_check)
    else:
        # For smaller models, checking the model object is fine
        onnx.checker.check_model(onnx_model)

    return (
        onnx_model,
        has_custom_op,
        custom_ops,
        ir_version_onnx_path or static_shaped_onnx_path or onnx_path,
        use_external_data_format,
    )


def interpret_trt_plugins_precision_flag(
    onnx_model: onnx.ModelProto,
    trt_plugins_precision: list[str],
    quantize_mode: str = "int8",
) -> tuple[dict, dict]:
    """Convert custom ops precision flag to dictionaries with custom op and I/O indices to be cast/quantized.

    Args:
        onnx_model: ONNX model to detect with nodes need to be cast/quantized.
        trt_plugins_precision: List indicating the precision for each custom op.
        quantize_mode: String indicating the quantization mode.

    Returns:
        Dictionary with custom ops to cast containing the I/O indices to cast.
        Dictionary with custom ops to quantize containing the I/O indices to quantize.
    """
    # If custom op precisions are given, check if they're supported (fp32, fp16, int8, fp8)
    custom_ops_to_cast = {}
    custom_ops_to_quantize = {}
    supported_precisions = ["fp32", "fp16", "int8", "fp8"]
    logger.debug("Processing custom op precisions")

    graph = gs.import_onnx(onnx_model)

    for trt_plugin_precision in trt_plugins_precision:
        assert trt_plugin_precision.count(":") in [1, 2], (
            "Plugin precision is incorrectly formatted."
            " Please check that it's in the format <op_type>:<precision> or"
            " <op_type>:[<inp1_precision>,<inp2_precision>,...]:[<out1_precision>,<out2_precision>,...]."
        )
        # Split only on the first ":" to get 'op_type'
        op_type, precision = trt_plugin_precision.split(":", 1)
        custom_op_nodes = [node for node in graph.nodes if node.op == op_type]
        if not custom_op_nodes:
            logger.warning(f"No nodes of type {op_type} were found. Skipping.")
            continue
        num_inps = max([len(node.inputs) for node in custom_op_nodes])
        num_outs = max([len(node.outputs) for node in custom_op_nodes])

        # Now split the remainder of the string to get the I/O precisions
        if trt_plugin_precision.count(":") == 1:
            if precision not in supported_precisions:
                logger.warning(f"Precision {precision} is not supported. Skipping.")
            if precision in ["fp16", "fp32"]:
                custom_ops_to_cast[precision] = {
                    op_type: {
                        "inp": list(range(num_inps)),
                        "out": list(range(num_outs)),
                    }
                }
            if precision in ["int8", "fp8"]:
                if precision != quantize_mode:
                    precision = quantize_mode
                    logger.warning(
                        f"Requested custom op precision ({precision}) is different than quantize mode: "
                        f"{quantize_mode}. Mixed {precision}+{quantize_mode} precision is not yet supported. "
                        f"Setting the custom op precision to be the same as quantize mode."
                    )
                custom_ops_to_quantize[op_type] = {
                    "inp": list(range(num_inps)),
                    "out": list(range(num_outs)),
                }
        else:
            inp_precision, out_precision = precision.split(":")
            inp_precision = inp_precision.strip("[]").split(",")
            out_precision = out_precision.strip("[]").split(",")
            if not all(p in supported_precisions for p in inp_precision + out_precision):
                logger.warning(
                    f"One or more precisions in {inp_precision + out_precision} are not supported. Skipping those."
                )
            assert len(inp_precision) == num_inps, (
                f"Number of inputs doesn't match expectation: {len(inp_precision)} vs {num_inps}."
            )
            assert len(out_precision) == num_outs, (
                f"Number of outputs doesn't match expectation: {len(out_precision)} vs {num_outs}."
            )

            if any(
                p in ["int8", "fp8"] and p != quantize_mode for p in inp_precision + out_precision
            ):
                logger.warning(
                    f"Requested custom op precision ('inp': {inp_precision}, 'out': {out_precision}) is different "
                    f"than quantize mode: {quantize_mode}. Such mixed precision is not yet supported. "
                    f"Setting the custom op precision to be the same as quantize mode."
                )

            # Will cast the inputs to FP16/FP32 and the outputs back to FP32
            for precision in ["fp16", "fp32"]:
                inp_precision_cast = [i for i, p in enumerate(inp_precision) if p == precision]
                out_precision_cast = [i for i, p in enumerate(out_precision) if p == precision]
                if inp_precision_cast:
                    custom_ops_to_cast[precision] = {
                        op_type: {"inp": inp_precision_cast, "out": out_precision_cast}
                    }

            # Will add Q/DQ nodes in the requested I/O indices
            inp_precision_quant = [i for i, p in enumerate(inp_precision) if p in ["int8", "fp8"]]
            out_precision_quant = [i for i, p in enumerate(out_precision) if p in ["int8", "fp8"]]
            if inp_precision_quant or out_precision_quant:
                custom_ops_to_quantize[op_type] = {
                    "inp": inp_precision_quant,
                    "out": out_precision_quant,
                }

    return custom_ops_to_cast, custom_ops_to_quantize
