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

"""Node selection and runtime probes for calibrated ONNX quantization."""

from collections.abc import Sequence
from functools import reduce

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.tensor import Variable
from onnxruntime.quantization.calibrate import CalibrationDataReader

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.graph_indexing import (
    expand_node_names_from_patterns,
    find_mha_partitions,
    match_fp8_mha_pattern,
)
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.utils import infer_shapes, parse_shapes_spec, save_onnx

__all__ = [
    "find_nodes_from_convs_to_exclude",
    "find_nodes_from_matmul_to_exclude",
    "find_nodes_from_mha_to_exclude",
    "find_nodes_to_exclude",
    "get_extended_model_outputs",
    "get_input_shapes",
    "validate_op_types_spelling",
]


def get_input_shapes(onnx_path: str) -> dict[str, list[int]]:
    """Returns the input shapes of the given ONNX model."""
    onnx_model = onnx.load(onnx_path)
    input_shape_dict = {}
    for input in onnx_model.graph.input:
        input_shape_dict[input.name] = [x.dim_value for x in input.type.tensor_type.shape.dim]
    return input_shape_dict


def _find_nodes_from_op_types_to_exclude(graph: Graph, op_types_to_exclude=None) -> list[str]:
    nodes_to_exclude = []
    if op_types_to_exclude:
        nodes_to_exclude = [node.name for node in graph.nodes if node.op in op_types_to_exclude]
    return nodes_to_exclude


def find_nodes_to_exclude(
    graph: Graph, nodes_to_exclude: list[str], op_types_to_exclude: list[str]
):
    """Find the node names from the ONNX graph which matches user's exclusion patterns."""
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_exclude = expand_node_names_from_patterns(graph, nodes_to_exclude)
    nodes_to_exclude.extend(_find_nodes_from_op_types_to_exclude(graph, op_types_to_exclude))

    # Remove duplicates from the exclusion list
    return [*set(nodes_to_exclude)]


def get_extended_model_outputs(
    onnx_path: str,
    extended_model: onnx.ModelProto,
    use_external_data_format: bool,
    intermediate_generated_files: list[str],
    calibration_data_reader: CalibrationDataReader,
    calibration_eps: list[str],
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    trt_rtx_backend: str = "legacy",
) -> dict[str, np.ndarray]:
    """Run one inference step on an onnx model which has some intermediate tensor marked as model outputs.

    The first calibration data is used for the dummy inference. This is useful when we want to know the shape of an
    intermediate tensor given the calibration data.

    Args:
        onnx_path:
            Path to the original onnx model, used for saving the extended model nearby if it is larger than 2GB.
        extended_model:
            The onnx model with some intermediate tensors marked as model outputs.
        use_external_data_format:
            If True, external data path will be used to store the weights of the intermediate model.
        intermediate_generated_files:
            List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader:
            Calibration data reader for running inference.
        calibration_eps:
            Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'cpu', 'trt'], where 'x' is the device id.

    Returns: a map with each output name pointed to the corresponding output numpy ndarray.
    """
    # Get the first calibration input data.
    inputs = calibration_data_reader.get_first()

    # Initialize ORT session.
    if use_external_data_format:
        extended_onnx_path = onnx_path.replace(".onnx", "_extended.onnx")
        save_onnx(extended_model, extended_onnx_path, save_as_external_data=True)
        intermediate_generated_files.append(extended_onnx_path)
        session = create_inference_session(
            extended_onnx_path, calibration_eps, input_shapes_profile, trt_rtx_backend
        )
    else:
        session = create_inference_session(
            extended_model.SerializeToString(),
            calibration_eps,
            input_shapes_profile,
            trt_rtx_backend,
        )

    # Run extended model's inference.
    extended_model_output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(extended_model_output_names, inputs)
    output_map = dict(zip(extended_model_output_names, outputs))

    return output_map


def find_nodes_from_matmul_to_exclude(
    onnx_path: str,
    use_external_data_format: bool = False,
    intermediate_generated_files: list[str] | None = None,
    calibration_data_reader: CalibrationDataReader = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    calibration_shapes: str | dict | None = None,
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    trt_rtx_backend: str = "legacy",
) -> list[str]:
    """Find MatMul nodes that meet gemv or small-gemm conditions and should be excluded.

    A MatMul is excluded if either:

    - m or n in the output is 1 (GEMV): cannot utilize TensorCores; or
    - K or N is smaller than ``_MIN_MATMUL_DIM`` (16): both INT8 and FP8 Tensor Core
      kernels need K/N >= 16 to be efficient, and adding Q/DQ layers on such small
      GEMMs causes TRT perf regressions.

    Args:
        onnx_path: Path to the onnx model.
        use_external_data_format: If True, external data path will be used to store the
            weights of the intermediate model.
        intermediate_generated_files: List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader: Calibration data reader for running inference.
        calibration_shapes: Model input shapes for inference. If provided, symbolic shape inference will be used
            instead of calibration_data_reader.
        calibration_eps: Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'cpu', 'trt'], where 'x' is the device id.

    Returns:
        List of Nodes to exclude from quantization.
    """
    model = onnx.load(onnx_path, load_external_data=True)
    graph = gs.import_onnx(model)

    matmul_nodes = [node for node in graph.nodes if node.op in {"MatMul", "Gemm"}]
    if not matmul_nodes:
        logger.debug("No MatMul nodes found in the model")
        return []

    logger.debug(f"Found {len(matmul_nodes)} MatMul nodes to analyze")

    if calibration_shapes:
        nodes_to_exclude = _exclude_matmuls_by_shape_inference(
            model, matmul_nodes, calibration_shapes
        )
    else:
        if calibration_data_reader is None:
            raise ValueError(
                "Either calibration_shapes or calibration_data_reader must be provided"
            )
        nodes_to_exclude = _exclude_matmuls_by_inference(
            onnx_path,
            model,
            matmul_nodes,
            use_external_data_format,
            intermediate_generated_files or [],
            calibration_data_reader,
            calibration_eps,
            input_shapes_profile,
            trt_rtx_backend,
        )

    logger.debug(f"Matmul nodes to exclude: {nodes_to_exclude}")
    return [*set(nodes_to_exclude)]


_MIN_CHANNELS_FP8 = 16
# Minimum K/N dim for MatMul/Gemm under INT8 or FP8 quantization. Both INT8 and FP8
# Tensor Core kernels need K/N >= 16 to be efficient; adding Q/DQ layers on smaller
# GEMMs causes TRT perf regressions.
_MIN_MATMUL_DIM = 16


def find_nodes_from_convs_to_exclude(graph: Graph, quantize_mode: str = "int8"):
    """Find unsupported Conv nodes to exclude from quantization.

    - The input and output channels should be >= 16. The exception is for Conv layers in INT8 quantization mode,
      which supports it if the input or output channel % 8.
    - The filter size for FP8 conv kernels should be less than 32.
    - For FP8 mode, Conv nodes with input or output channels <= _MIN_CHANNELS_FP8 are excluded.
      Small-channel convolutions do not benefit from FP8 quantization.

    Args:
        graph: Onnx model graph.
        quantize_mode: Quantize mode (int8 or fp8).

    Returns:
        List of Conv nodes.
    """
    unsupported_conv_nodes = []
    logger.info("Scanning for unsupported Conv nodes for quantization")
    for node in graph.nodes:
        if node.op == "Conv":
            weight = node.inputs[1]

            # If weight.shape is None, it means the weight is not a constant tensor.
            # Skip the convs with non-constant weights.
            if weight.shape is None:
                logger.debug(f"Skipped quantizing conv: {node.name} due to non-constant weight")
                unsupported_conv_nodes.append(node.name)
                continue

            assert 3 <= len(weight.shape) <= 5, (
                f"Invalid weight shape {weight.shape}. Only 1D, 2D, and 3D convolutions are supported"
            )
            output_channel = weight.shape[0]
            input_channel = weight.shape[1]
            group = node.attrs.get("group", 1)
            kernel_shape = node.attrs.get("kernel_shape", [1, 1])
            if output_channel < 16 or input_channel < 16:
                if quantize_mode == "int8":
                    # If standard Conv (group == 1), check for supported configs:
                    # (1) OC or IC % 8
                    # (2) Conv is the 1st layer of the graph or OC or IC >= 16
                    # (3) Kernel shape is the same in all dimensions (i.e, 3x3)
                    if (
                        group == 1
                        and (output_channel % 8 == 0 or input_channel % 8 == 0)
                        and (
                            any(inp in graph.inputs for inp in node.inputs)
                            or (output_channel >= 16 or input_channel >= 16)
                        )
                        and len(set(kernel_shape)) == 1
                    ):
                        continue

                    # If grouped Conv (group > 1), check for supported configs:
                    # (1) OC % 8
                    # (2) Kernel shape is >= 3 across all dimensions
                    if group > 1 and output_channel % 8 == 0 and all(k >= 3 for k in kernel_shape):
                        continue

                logger.debug(f"Found Conv with I/O channel size less than 16: {node.name}")
                unsupported_conv_nodes.append(node.name)
                continue

            filter_size = reduce(lambda x, y: x * y, weight.shape[2:])
            if quantize_mode == "fp8" and filter_size > 32:
                logger.debug(f"Found large filter conv for FP8: {node.name}")
                unsupported_conv_nodes.append(node.name)
                # skip the small-channel check below; already excluded
                continue

            # For FP8, exclude small-channel convolutions. These layers do not benefit from
            # FP8 quantization and cause perf regressions on GPUs where the FP8 conv kernels
            # are slower than FP16 CASK kernels for small channels.
            if quantize_mode == "fp8" and (
                output_channel <= _MIN_CHANNELS_FP8 or input_channel <= _MIN_CHANNELS_FP8
            ):
                logger.debug(
                    f"Excluding small-channel Conv from FP8 quantization: {node.name} "
                    f"(IC={input_channel}, OC={output_channel}, threshold={_MIN_CHANNELS_FP8})"
                )
                unsupported_conv_nodes.append(node.name)

    logger.info(f"Found {len(unsupported_conv_nodes)} unsupported Conv nodes for quantization")
    return unsupported_conv_nodes


def _get_inp_b_k_dim(
    matmul_node, value_info_map: dict | None = None, output_map: dict | None = None
):
    """Get the K dimension from the second input of a MatMul/Gemm node.

    Tries Constant shape first, then falls back to shape inference (value_info_map)
    or runtime inference (output_map). For Gemm nodes, honors the ``transB`` attribute:
    when ``transB=1``, B has shape ``[N, K]`` so K lives at axis -1; otherwise B is
    ``[..., K, N]`` and K is at axis -2.

    Returns:
        The K dimension value, or None if it cannot be determined.
    """
    # For Gemm, transB=1 means B is [N, K] (K is last axis); default/MatMul is [K, N].
    trans_b = bool(matmul_node.attrs.get("transB", 0)) if matmul_node.op == "Gemm" else False
    k_axis = -1 if trans_b else -2

    inp_b = matmul_node.inputs[1]
    if hasattr(inp_b, "values") and inp_b.values is not None:
        inp_b_shape = inp_b.values.shape
        if len(inp_b_shape) >= 2:
            return inp_b_shape[k_axis]
    if value_info_map is not None:
        inp_b_info = value_info_map.get(inp_b.name)
        if inp_b_info:
            inp_b_dims = inp_b_info.type.tensor_type.shape.dim
            if len(inp_b_dims) >= 2:
                return inp_b_dims[k_axis].dim_value
    if output_map is not None and inp_b.name in output_map:
        inp_b_out = output_map[inp_b.name]
        if len(inp_b_out.shape) >= 2:
            return inp_b_out.shape[k_axis]
    return None


def _exclude_matmuls_by_shape_inference(
    model: onnx.ModelProto,
    matmul_nodes: list,
    calibration_shapes: str | dict | None = None,
) -> list[str]:
    """Use shape inference to find MatMuls with dimension 1 or small K/N."""
    # Prepare model for symbolic inference
    for graph_input in model.graph.input:
        for dim in graph_input.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                dim.Clear()
                dim.dim_value = 1

    # Apply calibration shapes if provided
    input_shapes = {}
    if calibration_shapes:
        input_shapes = (
            parse_shapes_spec(calibration_shapes)
            if isinstance(calibration_shapes, str)
            else calibration_shapes
        )
    for graph_input in model.graph.input:
        if graph_input.name in input_shapes:
            input_shape = input_shapes[graph_input.name]
            tensor_shape = graph_input.type.tensor_type.shape.dim
            if len(tensor_shape) != len(input_shape):
                raise ValueError(
                    f"{graph_input.name} expects shape of rank {len(tensor_shape)}, "
                    f"but calibration shape of rank {len(input_shape)} was passed."
                )
            for dim, new_dim_value in zip(tensor_shape, input_shape):
                dim.dim_value = new_dim_value

    model = infer_shapes(model)
    # Include graph inputs, value_info, and outputs so B that comes from a graph input
    # is visible when deriving K.
    value_info_map = {vi.name: vi for vi in model.graph.input}
    value_info_map.update({vi.name: vi for vi in model.graph.value_info})
    value_info_map.update({vi.name: vi for vi in model.graph.output})

    nodes_to_exclude = []
    for matmul_node in matmul_nodes:
        output_name = matmul_node.outputs[0].name
        value_info = value_info_map.get(output_name)
        if not value_info:
            raise RuntimeError(f"Shape inference did not find shape for {output_name}.")

        dims = value_info.type.tensor_type.shape.dim
        if all(isinstance(inp, Variable) for inp in matmul_node.inputs):
            if len(dims) < 2:
                raise RuntimeError(f"Shape for {output_name} is incorrect.")

            if dims[-1].dim_value == 1 or dims[-2].dim_value == 1:
                nodes_to_exclude.append(matmul_node.name)
                continue
        elif len(dims) < 3 and any(out.dim_value == 1 for out in dims):
            nodes_to_exclude.append(matmul_node.name)
            continue

        # Small-gemm check: applies to both INT8 and FP8 quantization.
        n_dim = dims[-1].dim_value if len(dims) >= 2 else 0
        k_dim = _get_inp_b_k_dim(matmul_node, value_info_map=value_info_map)
        small_n = 0 < n_dim < _MIN_MATMUL_DIM
        small_k = k_dim is not None and 0 < k_dim < _MIN_MATMUL_DIM

        if small_n or small_k:
            logger.debug(
                f"Excluding small-dim MatMul from quantization: {matmul_node.name} "
                f"(N={n_dim}, K={k_dim}, threshold={_MIN_MATMUL_DIM})"
            )
            nodes_to_exclude.append(matmul_node.name)

    return nodes_to_exclude


def _exclude_matmuls_by_inference(
    onnx_path: str,
    model: onnx.ModelProto,
    matmul_nodes: list,
    use_external_data_format: bool,
    intermediate_generated_files: list[str],
    calibration_data_reader: CalibrationDataReader,
    calibration_eps: list[str],
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    trt_rtx_backend: str = "legacy",
) -> list[str]:
    """Use actual inference to find MatMuls with dimension 1 or small K/N."""
    # Add matmul outputs and second-input outputs to model outputs
    existing_output_names = {out.name for out in model.graph.output}
    for matmul_node in matmul_nodes:
        out_name = matmul_node.outputs[0].name
        if out_name not in existing_output_names:
            model.graph.output.extend([onnx.ValueInfoProto(name=out_name)])
            existing_output_names.add(out_name)
        # Also add second input for K-dimension check (only if it's a Variable, not a Constant)
        if isinstance(matmul_node.inputs[1], Variable):
            inp_b_name = matmul_node.inputs[1].name
            if inp_b_name not in existing_output_names:
                model.graph.output.extend([onnx.ValueInfoProto(name=inp_b_name)])
                existing_output_names.add(inp_b_name)

    output_map = get_extended_model_outputs(
        onnx_path,
        model,
        use_external_data_format,
        intermediate_generated_files,
        calibration_data_reader,
        calibration_eps,
        input_shapes_profile,
        trt_rtx_backend,
    )

    nodes_to_exclude = []
    for matmul_node in matmul_nodes:
        matmul_output = output_map[matmul_node.outputs[0].name]
        if all(isinstance(inp, Variable) for inp in matmul_node.inputs):
            if (
                len(matmul_output.shape) < 2
                or matmul_output.shape[-1] == 1
                or matmul_output.shape[-2] == 1
            ):
                nodes_to_exclude.append(matmul_node.name)
                continue
        elif len(matmul_output.shape) < 3 and any(out == 1 for out in matmul_output.shape):
            nodes_to_exclude.append(matmul_node.name)
            continue

        # Small-gemm check: applies to both INT8 and FP8 quantization.
        n_dim = matmul_output.shape[-1] if len(matmul_output.shape) >= 2 else 0
        k_dim = _get_inp_b_k_dim(matmul_node, output_map=output_map)
        small_n = 0 < n_dim < _MIN_MATMUL_DIM
        small_k = k_dim is not None and 0 < k_dim < _MIN_MATMUL_DIM

        if small_n or small_k:
            logger.debug(
                f"Excluding small-dim MatMul from quantization: {matmul_node.name} "
                f"(N={n_dim}, K={k_dim}, threshold={_MIN_MATMUL_DIM})"
            )
            nodes_to_exclude.append(matmul_node.name)

    return nodes_to_exclude


def find_nodes_from_mha_to_exclude(
    onnx_path: str,
    use_external_data_format: bool = False,
    nodes_to_exclude: list[str] | None = None,
    disable_mha_qdq: bool = False,
    quantize_mode: str = "int8",
    intermediate_generated_files: list[str] | None = None,
    calibration_data_reader: CalibrationDataReader = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    trt_rtx_backend: str = "legacy",
) -> list[str]:
    """Find MatMul nodes in MHA pattern to exclude.

    If disable_mha_qdq is set, don't add Q/DQ layers to MatMuls in MHA pattern.
    else when quantize_mode == "fp8", if head_size > 256 or head_size <= 8 or
    mha doesn't meet fp8 fMHA v2 pattern, don't add Q/DQ layers to MatMuls in MHA pattern.
    else when quantize_mode == "int8", if seq_len > 512, don't add Q/DQ layers
    to MatMuls in MHA pattern.

    Args:
        onnx_path:
            Path to the onnx model.
        use_external_data_format:
            If True, external data path will be used to store the weights of the intermediate model.
        nodes_to_exclude:
            List of Nodes to exclude from quantization.
        disable_mha_qdq:
            If True, all MHA's BMM1 and BMM2 will be added to nodes_to_exclude.
            Else, each MHA will be checked whether to enable QDQ or not when is_fp8fp16 is True.
        quantize_mode:
            Quantization mode. One of 'int8' (default), 'int4' and 'fp8'.
        intermediate_generated_files:
            List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader:
            Calibration data reader for running inference.
        calibration_eps:
            Priority list of execution providers (EP) for calibration.

    Returns:
        List of Nodes to exclude from quantization.
    """
    logger.info(f"Analyzing MHA nodes for {quantize_mode} quantization")
    model = onnx.load(onnx_path, load_external_data=True)
    graph = gs.import_onnx(model)

    mha_partitions = find_mha_partitions(graph)
    if len(mha_partitions) == 0:
        logger.info("No MHA partitions found in the model")
        return nodes_to_exclude  # type: ignore[return-value]

    matmul_nodes_to_exclude = []
    if disable_mha_qdq:
        logger.info("Disabling QDQ for all MHA nodes")
        for mha_partition in mha_partitions:
            matmul_nodes_to_exclude.append(mha_partition[0].name)
            matmul_nodes_to_exclude.append(mha_partition[2].name)
    elif quantize_mode in {"fp8", "int8"}:
        # Add each BMM1's second input as BS1 model's extended outputs.
        for mha_partition in mha_partitions:
            bmm1_node = mha_partition[0]
            model.graph.output.extend([onnx.ValueInfoProto(name=bmm1_node.inputs[1].name)])

        # To get head_size and seq_len of MHA of the model, we run extended model inference once to get the shape info.
        output_map = get_extended_model_outputs(
            onnx_path,
            model,
            use_external_data_format,
            intermediate_generated_files,  # type: ignore[arg-type]
            calibration_data_reader,
            calibration_eps,
            input_shapes_profile,
            trt_rtx_backend,
        )

        # For each MHA block,
        # In quantize_mode == int8, if seq_len > 512, add bmm to nodes_to_exclude.
        # In quantize_mode == fp8, if head_size > 256 or head_size <= 8, add its bmm to nodes_to_exclude.
        for mha_partition in mha_partitions:
            bmm1_node = mha_partition[0]
            softmax_node = mha_partition[1]
            bmm1_input_name = bmm1_node.inputs[1].name
            bmm1_input = output_map[bmm1_input_name]
            seq_len = bmm1_input.shape[-1]
            head_size = bmm1_input.shape[-2]
            enable_mha_qdq = True
            if quantize_mode == "int8":
                if seq_len > 512:
                    enable_mha_qdq = False
                    logger.debug(
                        f"Disabling QDQ for MHA node {bmm1_node.name} due to seq_len {seq_len} > 512"
                    )
            elif quantize_mode == "fp8":
                if head_size > 256 or head_size <= 8:
                    enable_mha_qdq = False
                    logger.debug(
                        f"Disabling QDQ for MHA node {bmm1_node.name} due to head_size {head_size}"
                    )
                else:
                    fp8_fmha_v2_pattern = match_fp8_mha_pattern(graph, softmax_node, False)
                    if len(fp8_fmha_v2_pattern) == 0:
                        enable_mha_qdq = False
                        logger.debug(
                            f"Disabling QDQ for MHA node {bmm1_node.name} due to non-matching FP8 pattern"
                        )
            if not enable_mha_qdq:
                matmul_nodes_to_exclude.append(mha_partition[0].name)
                matmul_nodes_to_exclude.append(mha_partition[2].name)

    logger.debug(f"Matmul nodes From MHA to exclude: {matmul_nodes_to_exclude}")

    nodes_to_exclude.extend(matmul_nodes_to_exclude)  # type: ignore[union-attr]
    # Remove duplicates from the exclusion list
    return [*set(nodes_to_exclude)]  # type: ignore[arg-type]


def validate_op_types_spelling(onnx_path, op_types_to_quantize, op_types_to_exclude) -> None:
    """Validate spelling in op types."""

    def find_item_ignore_case(target, arr):
        target_lower = target.lower()
        for item in arr:
            if item.lower() == target_lower:
                return item
        return None

    model = onnx.load(onnx_path, load_external_data=True)
    op_types = {node.op_type for node in model.graph.node}
    for op_type in op_types:
        if op_types_to_quantize:
            op_to_quant = find_item_ignore_case(op_type, op_types_to_quantize)
            if op_type not in op_types_to_quantize and op_to_quant is not None:
                logger.warning(
                    f"Model contains '{op_type}' ops, but you're requesting '{op_to_quant}' "
                    f"to be quantized, which is not a match. Please ensure that the lower/uppercasing is correct."
                )
        if op_types_to_exclude:
            op_to_exclude = find_item_ignore_case(op_type, op_types_to_exclude)
            if op_type not in op_types_to_exclude and op_to_exclude is not None:
                logger.warning(
                    f"Model contains '{op_type}' ops, but you're requesting '{op_to_exclude}' "
                    f"to be excluded from quantization, which is not a match. "
                    f"Please ensure that the lower/uppercasing is correct."
                )
