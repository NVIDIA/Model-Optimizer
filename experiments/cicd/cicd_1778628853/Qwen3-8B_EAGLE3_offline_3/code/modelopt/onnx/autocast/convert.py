# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""AutoCast module for converting ONNX models to mixed precision.

AutoCast is a tool for converting FP32 ONNX models to mixed precision FP32-FP16 or FP32-BF16 models.
While casting FP32 to FP16/BF16, some nodes might be more sensitive to affecting accuracy.
AutoCast intelligently selects nodes to keep in FP32 precision to maintain model accuracy while benefiting from
reduced precision on the rest of the nodes. AutoCast automatically injects cast operations around the selected
nodes.
"""

import numpy as np
import onnx

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.graphsanitizer import GraphSanitizer
from modelopt.onnx.autocast.logging_config import logger
from modelopt.onnx.autocast.nodeclassifier import NodeClassifier, NodeRuleBase
from modelopt.onnx.autocast.precisionconverter import PrecisionConverter
from modelopt.onnx.autocast.referencerunner import ReferenceRunner
from modelopt.onnx.utils import get_min_opset_for_precisions, get_qdq_precisions

"""
FP16 accuracy decreases in accordance with the data's magnitude.
For 512, the unit in last place (ULP) is 0.5, for 1024 it is 1.0, etc.
"""
DEFAULT_DATA_MAX = 512
DEFAULT_INIT_MAX = np.finfo(np.float16).max
LATEST_IR_VERSION_SUPPORTED_BY_ORT = 10


def convert_to_mixed_precision(
    onnx_path: str,
    low_precision_type: str = "fp16",
    nodes_to_exclude: list[str] | None = None,
    op_types_to_exclude: list[str] | None = None,
    nodes_to_include: list[str] | None = None,
    op_types_to_include: list[str] | None = None,
    data_max: float = DEFAULT_DATA_MAX,
    init_max: float = DEFAULT_INIT_MAX,
    keep_io_types: bool = False,
    calibration_data: str | None = None,
    custom_rule: NodeRuleBase | None = None,
    init_conversion_max_bytes: int | None = None,
    providers: list[str] = ["cpu"],
    trt_plugins: list[str] = [],
    trt_plugins_precision: list[str] = [],
    max_depth_of_reduction: int | None = None,
    opset: int | None = None,
    use_standalone_type_inference: bool = False,
) -> onnx.ModelProto:
    """Convert model to mixed precision.

    Args:
        onnx_path: Path to the input ONNX model.
        low_precision_type: Target precision to reduce to ('fp16' or 'bf16').
        nodes_to_exclude: List of regex patterns to match node names that should remain in FP32.
        op_types_to_exclude: List of operation types that should remain in FP32.
        nodes_to_include: List of regex patterns to match node names that should be included in low precision.
        op_types_to_include: List of operation types that should be included in low precision.
        data_max: Maximum absolute value for node input and output values.
        init_max: Maximum absolute value for initializers.
        keep_io_types: Whether to preserve input/output types.
        calibration_data: Path to input data file for reference runner.
        custom_rule: Optional custom rule for node classification (inherits from NodeRuleBase).
        init_conversion_max_bytes: Maximum size in bytes for initializer conversion. Larger initializers will be cast at
                                   runtime.
        providers: List of ORT execution providers.
        trt_plugins: List of TensorRT plugin library paths in .so format (compiled shared library).
        trt_plugins_precision: List indicating the precision for each custom op.
        max_depth_of_reduction: Maximum depth of reduction for node classification.
        opset: Target ONNX opset version. If None, uses default minimum opset based on low_precision_type
               (22 for bf16, 19 for fp16). The opset may be automatically increased if certain operations
               require a higher version.
        use_standalone_type_inference: If True, use standalone type inference implementation instead of ONNX's
                                  infer_shapes. This is a workaround (WAR) when only type inference is
                                  needed without shape inference. Default: False.

    Returns:
        onnx.ModelProto: The converted mixed precision model.
    """
    # Load and process model
    model = onnx.load(onnx_path, load_external_data=True)
    assert low_precision_type in ["fp16", "bf16"], "low_precision_type must be either fp16 or bf16"

    # Get original model's opset version
    original_opset = onnx_utils.get_opset_version(model)

    # Apply graph sanitization and optimizations
    # Opsets < 22 have a very limited support for bfloat16
    # Otherwise, prefer to keep the original opset version unless it's very old
    if opset is not None:
        min_opset = opset
        # Validate opset compatibility
        if low_precision_type == "bf16" and opset < 22:
            logger.warning(
                f"Opset {opset} has limited BF16 support. Recommended minimum opset is 22. "
                "The conversion may fail or produce unexpected results."
            )
        elif low_precision_type == "fp16" and opset < 13:
            logger.warning(
                f"Opset {opset} has limited FP16 support. Recommended minimum opset is 13. "
                "The conversion may fail or produce unexpected results."
            )
        # Warn if user-specified opset is lower than original
        if opset < original_opset:
            logger.warning(
                f"Specified opset {opset} is lower than the original model's opset {original_opset}. "
                "Downgrading opset version may cause compatibility issues or conversion failures."
            )
    else:
        min_opset = 22 if low_precision_type == "bf16" else 13

    graph_sanitizer = GraphSanitizer(
        model,
        min_opset,
        trt_plugins=trt_plugins,
        trt_plugins_precision=trt_plugins_precision,
        max_ir_version=LATEST_IR_VERSION_SUPPORTED_BY_ORT,
    )
    graph_sanitizer.sanitize()
    model = graph_sanitizer.model

    # Setup internal mappings
    model = onnx_utils.infer_types(model, use_standalone_type_inference)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    # Automatically add 'trt' to list of providers if custom ops are detected
    if "trt" not in providers and graph_sanitizer.custom_ops:
        providers.insert(0, "trt")

    # Initialize classifiers and converters
    node_classifier = NodeClassifier(
        model,
        node_to_init_map,
        initializer_map,
        nodes_to_exclude=nodes_to_exclude or [],
        op_types_to_exclude=op_types_to_exclude or [],
        nodes_to_include=nodes_to_include or [],
        op_types_to_include=op_types_to_include or [],
        data_max=data_max,
        init_max=init_max,
        custom_rule=custom_rule,
        max_depth_of_reduction=max_depth_of_reduction,
        custom_ops_low_precision_nodes=graph_sanitizer.custom_ops_low_precision_nodes or [],
    )

    precision_converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
        init_conversion_max_bytes=init_conversion_max_bytes,
        custom_ops=graph_sanitizer.custom_ops,
        use_standalone_type_inference=use_standalone_type_inference,
    )

    # Obtain reference data
    ref_outputs_dict = None
    if (data_max is not None and data_max != np.inf) or graph_sanitizer.custom_ops:
        ref_runner = ReferenceRunner(model, providers, trt_plugins)
        ref_outputs_dict = ref_runner.run(calibration_data)

    # Run conversion
    low_precision_nodes, high_precision_nodes = node_classifier.run(ref_outputs_dict)
    model_mod = precision_converter.convert(high_precision_nodes, low_precision_nodes)

    # Log results
    total_nodes = len(low_precision_nodes) + len(high_precision_nodes)
    low_precision_percentage = (
        100 * len(low_precision_nodes) / total_nodes if total_nodes > 0 else 0
    )
    logger.info(
        f"Converted {len(low_precision_nodes)}/{total_nodes} nodes "
        f"({low_precision_percentage:.2f}%) to {low_precision_type}"
    )

    return model_mod


def convert_to_f16(
    model: onnx.ModelProto,
    low_precision_type: str = "fp16",
    keep_io_types: bool = True,
    op_block_list: list[str] = [],
    tensor_block_dict: dict[str, dict[str, list[int]]] = {},
    trt_plugins: list[str] | None = [],
    use_standalone_type_inference: bool = False,
    opset: int | None = None,
) -> onnx.ModelProto:
    """Convert model to mixed precision, using PrecisionConverter.

    This method bypasses NodeClassifier, and uses a simple op_block_list.

    Args:
        model: ONNX model to convert.
        low_precision_type: Target precision to reduce to ('fp16' or 'bf16').
        keep_io_types: Whether to preserve input/output types.
        op_block_list: List of operation types that should remain in FP32.
        tensor_block_dict: Dictionary of tensors (operation type and I/O indices) that should remain in FP32.
        trt_plugins: List of TensorRT plugin library paths in .so format (compiled shared library).
        use_standalone_type_inference: If True, use standalone type inference implementation instead of ONNX's
                                  infer_shapes. This is a workaround (WAR) when only type inference is
                                  needed without shape inference. Default: False.
        opset: Target ONNX opset version. If None, uses default minimum opset based on precision type
               (22 for bf16, 19 for fp16) and Q/DQ node requirements. The opset may be automatically
               increased if Q/DQ nodes in the model require a higher version (e.g., FP8 requires 19,
               INT4 requires 21, NVFP4 requires 23).
    """
    assert low_precision_type in ["fp16", "bf16"], "low_precision_type must be either fp16 or bf16"

    # Check Q/DQ precision types in the model and determine required opset
    qdq_precisions = get_qdq_precisions(model)
    qdq_min_opset = get_min_opset_for_precisions(qdq_precisions)

    # Base minimum opset for FP16/BF16 conversion
    # Opset 19 is the first to support fp16 scales in Q/DQ nodes
    base_min_opset = 22 if low_precision_type == "bf16" else 19

    # Determine target opset version
    if opset is not None:
        min_opset = opset
        # Check if Q/DQ nodes require a higher opset
        if qdq_precisions and qdq_min_opset > min_opset:
            logger.warning(
                f"Model contains Q/DQ nodes with precisions {qdq_precisions} that require "
                f"opset >= {qdq_min_opset}. Upgrading from specified opset {opset} to {qdq_min_opset}."
            )
            min_opset = qdq_min_opset
        # Also ensure we meet base minimum for precision type
        if min_opset < base_min_opset:
            logger.warning(
                f"Opset {min_opset} is below minimum opset {base_min_opset} for {low_precision_type}. "
                f"Upgrading to opset {base_min_opset}."
            )
            min_opset = base_min_opset
    else:
        # Use the highest required opset between base and Q/DQ requirements
        min_opset = max(base_min_opset, qdq_min_opset)

    sanitizer = GraphSanitizer(
        model,
        min_opset=min_opset,
        trt_plugins=trt_plugins,
        max_ir_version=LATEST_IR_VERSION_SUPPORTED_BY_ORT,
    )
    sanitizer.find_custom_nodes()
    sanitizer.convert_opset()
    sanitizer.ensure_graph_name_exists()
    sanitizer.convert_fp64_to_fp32()
    model = sanitizer.model

    # Setup internal mappings
    model = onnx_utils.infer_types(model, use_standalone_type_inference)
    value_info_map, initializer_map, node_to_init_map = utils.setup_mappings(model)

    precision_converter = PrecisionConverter(
        model,
        value_info_map,
        initializer_map,
        node_to_init_map,
        keep_io_types=keep_io_types,
        low_precision_type=low_precision_type,
        custom_ops=sanitizer.custom_ops,
        tensor_block_dict=tensor_block_dict,
        use_standalone_type_inference=use_standalone_type_inference,
    )
    high_precision_nodes = [node.name for node in model.graph.node if node.op_type in op_block_list]
    low_precision_nodes = [
        node.name for node in model.graph.node if node.op_type not in op_block_list
    ]
    model_mod = precision_converter.convert(high_precision_nodes, low_precision_nodes)
    return model_mod
