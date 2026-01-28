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

"""Graph sanitization and optimization for ONNX models."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxscript
from onnx import helper, numpy_helper

import modelopt.onnx.autocast.utils as utils
import modelopt.onnx.utils as onnx_utils
from modelopt.onnx.autocast.logging_config import logger
from modelopt.onnx.quantization.graph_utils import cast_custom_ops
from modelopt.onnx.trt_utils import interpret_trt_plugins_precision_flag


class GraphSanitizer:
    """A class for sanitizing ONNX model graphs, a part of the AutoCast tool."""

    def __init__(
        self,
        model: onnx.ModelProto,
        min_opset: int = 13,
        max_ir_version: int | None = None,
        trt_plugins: list[str] | None = [],
        trt_plugins_precision: list[str] | None = [],
    ) -> None:
        """Initialize GraphSanitizer.

        Args:
            model: ONNX model to sanitize
            min_opset: minimum opset version to use
            max_ir_version: maximum IR version supported by ORT
            trt_plugins: list of TensorRT plugin library paths in .so format (compiled shared library).
        """
        self.model = model
        self.min_opset = min_opset
        self.max_ir_version = max_ir_version
        self.standard_ops = {schema.name for schema in onnx.defs.get_all_schemas()}
        self.custom_ops = None
        self.custom_ops_low_precision_nodes = []
        self.trt_plugins = trt_plugins
        self.trt_plugins_precision = trt_plugins_precision or []

    def sanitize(self) -> None:
        """Sanitize the model graph.

        Currently, this finds decomposed LayerNorm patterns and replaces them with a single LayerNormalization operator.
        Additional functionality may be added in the future.
        """
        self.find_custom_nodes()
        self.remove_disconnected_outputs()
        self.convert_opset()
        self.replace_layernorm_pattern()
        self.ensure_graph_name_exists()
        self.duplicate_shared_constants()
        onnx_utils.name_onnx_nodes(self.model.graph)
        self.replace_custom_domain_nodes()
        self.sanitize_io_casts()
        self.cleanup_model()
        self.set_ir_version(self.max_ir_version)
        self.convert_fp64_to_fp32()
        self.ensure_custom_ops_precision()

    def convert_fp64_to_fp32(self) -> None:
        """Convert FP64 initializers, I/O types, and specific nodes to FP32."""
        modified = False

        # Convert initializers
        if self._convert_fp64_initializers():
            modified = True

        # Convert input/output types
        if self._convert_fp64_io_types():
            modified = True

        # Convert specific node types: Cast, ConstantOfShape, Constant
        if self._convert_fp64_nodes():
            modified = True

        if modified:
            logger.info("Converted FP64 initializers, I/O types, and nodes to FP32")

    def ensure_custom_ops_precision(self) -> None:
        """Ensure that custom ops run in the requested precision."""
        custom_ops_to_cast, _ = interpret_trt_plugins_precision_flag(
            self.model,
            self.trt_plugins_precision,
        )
        if custom_ops_to_cast.get("fp16", {}):
            self.model = cast_custom_ops(self.model, custom_ops_to_cast["fp16"])
            self.custom_ops_low_precision_nodes = [
                n.name for n in self.model.graph.node if n.op_type in custom_ops_to_cast["fp16"]
            ]
            logger.info("Ensured custom ops precision")

    def find_custom_nodes(self) -> None:
        """Find custom nodes in the model.

        Scans through all nodes in the graph and logs any nodes that use custom operators
        that are not part of the standard ONNX operator set.
        """
        self.custom_ops = {
            node.op_type for node in self.model.graph.node if node.op_type not in self.standard_ops
        }
        if self.custom_ops:
            from modelopt.onnx.trt_utils import infer_types_shapes_tensorrt, set_trt_plugin_domain

            # Set TensorRT plugin domain info in the graph for ORT compatibility
            self.model = set_trt_plugin_domain(self.model, self.custom_ops)

            # Infer types and shapes in the graph for ORT compatibility
            self.model = infer_types_shapes_tensorrt(self.model, self.trt_plugins)

    def remove_disconnected_outputs(self) -> None:
        """Remove disconnected outputs from the model."""
        tensors_to_remove = []
        for tensor in self.model.graph.output:
            if not utils.get_producer_nodes(self.model, tensor.name):
                tensors_to_remove.append(tensor)
                logger.debug(f"Found disconnected output: {tensor.name}")

        if tensors_to_remove:
            logger.warning(
                f"Found {len(tensors_to_remove)} disconnected outputs. Removing disconnected outputs from the graph."
            )

        for tensor in tensors_to_remove:
            self.model.graph.output.remove(tensor)

    def convert_opset(self) -> None:
        """Convert the model to the given opset version.

        The method checks all opset imports and converts the model if any are below the minimum version.
        Uses onnxscript for conversion when available, which handles large models (>2GB) better.
        """
        # Check all opset imports
        default_opsets = list(self.model.opset_import)

        # Check for quantization nodes and update min_opset if needed
        # Before opset 19, QuantizeLinear and DequantizeLinear data and scale must be fp32
        has_quant_nodes = any(
            node.op_type in ["QuantizeLinear", "DequantizeLinear"] for node in self.model.graph.node
        )
        if has_quant_nodes and self.min_opset < 19:
            logger.warning(
                f"Found QuantizeLinear/DequantizeLinear nodes. Updating minimum opset from {self.min_opset} to 19."
            )
            self.min_opset = 19

        # Convert if the default domain opset is below minimum
        if onnx_utils.get_opset_version(self.model) < self.min_opset:
            invalid_opsets = [op.version for op in default_opsets if op.version < self.min_opset]
            try:
                logger.info(
                    f"Converting model from opset {invalid_opsets} to {self.min_opset} using onnxscript..."
                )

                # Convert to onnxscript IR
                model_ir = onnxscript.ir.serde.deserialize_model(self.model)

                # onnxscript handles conversion of large models better than the standard ONNX version_converter
                # Convert opset with fallback=True (automatically falls back to C API if needed)
                onnxscript.version_converter.convert_version(
                    model_ir, target_version=self.min_opset, fallback=True
                )

                # Convert back to ONNX proto
                self.model = onnxscript.ir.serde.serialize_model(model_ir)
                logger.info(f"Successfully converted model to opset {self.min_opset}")
            except Exception as e:
                logger.warning(f"Failed to convert model to opset {self.min_opset}: {e!s}")
                logger.warning(f"Attempting to continue with the original opsets: {invalid_opsets}")
        else:
            logger.debug(
                f"No opset conversion needed. Current opset {[op.version for op in default_opsets]} >= min_opset "
                "{self.min_opset}"
            )

    def set_ir_version(self, max_ir_version: int | None) -> None:
        """Set the model's IR version to the maximum supported version.

        Args:
            max_ir_version: maximum IR version to use.

        The method checks the IR version and cuts it off at the maximum supported version.
        See https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support
        """
        if self.custom_ops:
            # Set ir_version to 10, remove it once ORT supports ir_version 11
            self.max_ir_version = 10
            max_ir_version = max_ir_version or self.max_ir_version
        if max_ir_version and self.model.ir_version > max_ir_version:
            try:
                self.model.ir_version = max_ir_version
            except Exception as e:
                logger.warning(f"Failed to set IR version to {max_ir_version}: {e!s}")
                logger.warning(
                    f"Attempting to continue with the original IR version: {self.model.ir_version}"
                )

    def replace_layernorm_pattern(self) -> None:
        """Detects and replaces LayerNorm operation patterns.

        This method scans through the graph looking for sequences of operations that implement LayerNorm functionality
        and replaces them with the more efficient LayerNormalization operator.
        """
        nodes_to_remove = []
        modified = False

        for node in self.model.graph.node:
            if node.op_type != "ReduceMean":
                continue

            try:
                pattern = self._match_layernorm_pattern(node)
                if not pattern:
                    continue

                ln_node = self._create_layernorm_node(pattern)
                insert_idx = self._find_insertion_point(pattern["input_name"])

                # Insert LayerNorm node and update graph
                self.model.graph.node.insert(insert_idx, ln_node)
                nodes_to_remove.extend(pattern["nodes_to_remove"])
                modified = True

                logger.info(f"Replaced LayerNorm pattern starting at {node.name}")

            except Exception as e:
                logger.debug(f"Failed to match LayerNorm pattern at {node.name}: {e!s}")
                continue
            self._remove_nodes(nodes_to_remove)

        if modified:
            self._update_opset_version()
            self.model = onnx_utils.infer_shapes(self.model, strict_mode=True)

    def ensure_graph_name_exists(self) -> None:
        """Ensures that the model's name exists."""
        if not self.model.graph.name:
            self.model.graph.name = "model"

    def duplicate_shared_constants(self) -> None:
        """Duplicate constant tensors if they are shared."""
        self.model, is_duplicated_constant = onnx_utils.duplicate_shared_constants(self.model)
        if is_duplicated_constant:
            logger.warning("Shared constants were detected and duplicated accordingly.")

    def _match_layernorm_pattern(self, mean_node: onnx.NodeProto) -> dict | None:
        """Match the sequence of operations that constitute a LayerNorm.

        Args:
            mean_node: The ReduceMean node to start pattern matching from.

        Returns:
            Dict | None: Pattern information if matched, None otherwise.
        """
        try:
            axis = mean_node.attribute[0].ints[0]
            sub_node = utils.get_unique_consumer_node(self.model, mean_node.output[0])
            if sub_node.op_type != "Sub":
                return None

            # Find variance computation branch
            pow_nodes = [
                n
                for n in utils.get_consumer_nodes(self.model, sub_node.output[0])
                if n.op_type == "Pow"
            ]
            if len(pow_nodes) != 1:
                return None
            pow_node = pow_nodes[0]
            pow_of_value = self._get_initializer_value(pow_node.input[1])
            if pow_of_value != 2:
                return None

            var_mean_node = utils.get_unique_consumer_node(self.model, pow_node.output[0])
            if var_mean_node.op_type != "ReduceMean":
                return None

            add_eps_node = utils.get_unique_consumer_node(self.model, var_mean_node.output[0])
            if add_eps_node.op_type != "Add":
                return None

            sqrt_node = utils.get_unique_consumer_node(self.model, add_eps_node.output[0])
            if sqrt_node.op_type != "Sqrt":
                return None

            # Find Div node
            # Find the Div node that consumes both sqrt and sub outputs
            sqrt_consumers = utils.get_consumer_nodes(self.model, sqrt_node.output[0])
            sub_consumers = utils.get_consumer_nodes(self.model, sub_node.output[0])

            div_nodes = [n for n in sqrt_consumers if n in sub_consumers and n.op_type == "Div"]
            if len(div_nodes) != 1:
                div_node = None
            else:
                div_node = div_nodes[0]
                if (
                    div_node.input[0] != sub_node.output[0]
                    or div_node.input[1] != sqrt_node.output[0]
                ):
                    div_node = None
            if not div_node:
                return None

            # Get epsilon value
            epsilon = self._get_initializer_value(add_eps_node.input[1])
            if epsilon is None:
                logger.warning(
                    f"Could not find epsilon value for LayerNorm pattern starting at {mean_node.name}"
                )
                return None

            # Check for scale/bias
            # Find and extract scale and bias nodes if present
            scale = None
            bias = None
            scale_dimension = None
            final_node = div_node
            nodes_to_remove = [
                mean_node,
                sub_node,
                pow_node,
                var_mean_node,
                add_eps_node,
                sqrt_node,
                div_node,
            ]

            consumers = utils.get_consumer_nodes(self.model, div_node.output[0])
            if len(consumers) == 1 and consumers[0].op_type == "Mul":
                mul_node = consumers[0]
                scale = self._get_initializer_value(mul_node.input[1], return_array=True)
                final_node = mul_node
                nodes_to_remove.append(mul_node)

                consumers = utils.get_consumer_nodes(self.model, mul_node.output[0])
                if len(consumers) == 1 and consumers[0].op_type == "Add":
                    add_node = consumers[0]
                    bias = self._get_initializer_value(add_node.input[1], return_array=True)
                    final_node = add_node
                    nodes_to_remove.append(add_node)
            elif len(consumers) == 1 and consumers[0].op_type == "Add":
                # just bias, no scale
                add_node = consumers[0]
                bias = self._get_initializer_value(add_node.input[1], return_array=True)
                final_node = add_node
                nodes_to_remove.append(add_node)

            if scale is not None:
                scale_dimension = scale.shape

            # Skip pattern if we can't determine the scale dimension
            if scale_dimension is None:
                logger.debug(
                    f"Could not determine scale dimension for LayerNorm pattern at {mean_node.name}"
                )
                return None

            return {
                "mean_node": mean_node,
                "input_name": sub_node.input[0],
                "scale_dimension": scale_dimension,
                "output_name": final_node.output[0],
                "epsilon": epsilon,
                "scale": scale,
                "bias": bias,
                "axis": axis,
                "nodes_to_remove": nodes_to_remove,
            }

        except Exception as e:
            logger.debug(f"Failed to match LayerNorm pattern at {mean_node.name}: {e!s}")
            return None

    def sanitize_io_casts(self) -> None:
        """Handle the special case where an input is casted directly to an output.

        Inject an identity node after the cast node.
        """
        model_input_names = {input.name for input in self.model.graph.input}
        model_output_names = {output.name for output in self.model.graph.output}
        insertions: list[tuple[int, onnx.NodeProto]] = []

        for idx, node in enumerate(self.model.graph.node):
            if (
                node.op_type == "Cast"
                and node.input
                and node.output
                and node.input[0] in model_input_names
                and node.output[0] in model_output_names
            ):
                # Unique per graph output to avoid collisions when multiple outputs are cast from the same input
                cast_output_name = node.output[0]
                cast_new_output_name = f"{cast_output_name}__io_cast_src"
                identity_node = helper.make_node(
                    "Identity",
                    inputs=[cast_new_output_name],
                    outputs=[cast_output_name],
                    name=f"{node.name}__io_cast_identity",
                )
                # Rewire Cast to produce the new intermediate
                node.output[0] = cast_new_output_name
                insertions.append((idx + 1, identity_node))

        # Insert Identities in-order right after their corresponding Casts
        for offset, (pos, id_node) in enumerate(insertions):
            self.model.graph.node.insert(pos + offset, id_node)

    def _create_layernorm_node(self, pattern: dict) -> onnx.NodeProto:
        """Create a LayerNormalization node with optional bias."""
        ln_name = f"LayerNorm_{pattern['mean_node'].name}"
        scale_name = f"{ln_name}_scale"
        bias_name = f"{ln_name}_bias" if pattern["bias"] is not None else ""
        axis = pattern["axis"]

        # Always create scale tensor, default to ones if not provided
        scale_tensor = (
            pattern["scale"]
            if pattern["scale"] is not None
            else np.ones(pattern["scale_dimension"], dtype=np.float32)
        )
        self.model.graph.initializer.append(numpy_helper.from_array(scale_tensor, name=scale_name))

        if pattern["bias"] is not None:
            self.model.graph.initializer.append(
                numpy_helper.from_array(pattern["bias"], name=bias_name)
            )

        inputs = [pattern["input_name"], scale_name]
        if pattern["bias"] is not None:
            inputs.append(bias_name)

        return helper.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=[pattern["output_name"]],
            name=ln_name,
            epsilon=pattern["epsilon"],
            axis=axis,
        )

    def _find_insertion_point(self, input_name: str) -> int:
        """Find the correct insertion point for the new LayerNorm node."""
        producer_nodes = utils.get_producer_nodes(self.model, input_name)
        if not producer_nodes:
            return 0

        producer_indices = [i for i, n in enumerate(self.model.graph.node) if n in producer_nodes]
        return max(producer_indices) + 1

    def _remove_nodes(self, nodes_to_remove: list[onnx.NodeProto]) -> None:
        """Remove replaced nodes and their corresponding value_info entries."""
        tensors_to_remove = set()
        for node in nodes_to_remove:
            tensors_to_remove.update(node.output)
            if node in self.model.graph.node:
                self.model.graph.node.remove(node)

        value_info_to_remove = [
            vi for vi in self.model.graph.value_info if vi.name in tensors_to_remove
        ]
        for vi in value_info_to_remove:
            self.model.graph.value_info.remove(vi)

    def _update_opset_version(self) -> None:
        """Update the model's opset version to support LayerNormalization."""
        current_opset = None
        for opset in self.model.opset_import:
            if opset.domain in {"", "ai.onnx"}:
                current_opset = opset.version
                break

        if current_opset is None or current_opset < 17:
            # Remove existing default domain opsets
            default_opsets = [op for op in self.model.opset_import if op.domain in ("", "ai.onnx")]
            for op in default_opsets:
                self.model.opset_import.remove(op)
            # Add opset 17
            self.model.opset_import.append(helper.make_opsetid("", 17))
            logger.info(
                f"Updated model opset to 17 for LayerNormalization support (was {current_opset})"
            )

    def _get_initializer_value(self, name: str, return_array: bool = False) -> np.ndarray | None:
        """Get value from an initializer by name."""
        for init in self.model.graph.initializer:
            if init.name == name:
                value = numpy_helper.to_array(init)
                return value if return_array else value.item()
        return None

    def _convert_fp64_initializers(self) -> bool:
        """Convert FP64 initializers to FP32.

        Returns:
            bool: True if any initializers were modified, False otherwise.
        """
        modified = False

        for initializer in self.model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.DOUBLE:
                # Convert the data to FP32
                fp64_data = numpy_helper.to_array(initializer)
                fp32_data = fp64_data.astype(np.float32)

                # Create new initializer with FP32 data
                new_initializer = numpy_helper.from_array(fp32_data, name=initializer.name)

                # Replace the old initializer
                initializer.CopyFrom(new_initializer)
                modified = True
                logger.debug(f"Converted initializer {initializer.name} from FP64 to FP32")

        return modified

    def _convert_fp64_io_types(self) -> bool:
        """Convert FP64 input/output types to FP32.

        Returns:
            bool: True if any I/O types were modified, False otherwise.
        """
        modified = False

        def convert_tensor_list(tensors, tensor_type):
            nonlocal modified
            for tensor in tensors:
                if tensor.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
                    tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                    modified = True
                    logger.debug(f"Converted {tensor_type} {tensor.name} from FP64 to FP32")

        convert_tensor_list(self.model.graph.input, "input")
        convert_tensor_list(self.model.graph.output, "output")
        convert_tensor_list(self.model.graph.value_info, "value_info")

        return modified

    def _convert_fp64_nodes(self) -> bool:
        """Convert specific node types from FP64 to FP32.

        Handles Cast, ConstantOfShape, and Constant nodes that use FP64.

        Returns:
            bool: True if any nodes were modified, False otherwise.
        """
        modified = False

        for node in self.model.graph.node:
            if node.op_type == "Cast":
                # Check if casting to FP64, change to FP32
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == onnx.TensorProto.DOUBLE:
                        attr.i = onnx.TensorProto.FLOAT
                        modified = True
                        logger.debug(f"Converted Cast node {node.name} from FP64 to FP32")

            elif node.op_type in ["ConstantOfShape", "Constant"]:
                # Check if the value attribute uses FP64
                for attr in node.attribute:
                    if attr.name == "value" and attr.t.data_type == onnx.TensorProto.DOUBLE:
                        # Convert the tensor value to FP32
                        fp64_data = numpy_helper.to_array(attr.t)
                        fp32_data = fp64_data.astype(np.float32)
                        new_tensor = numpy_helper.from_array(fp32_data)
                        attr.t.CopyFrom(new_tensor)
                        modified = True
                        logger.debug(f"Converted {node.op_type} node {node.name} from FP64 to FP32")

        return modified

    def cleanup_model(self) -> None:
        """Use GraphSurgeon to cleanup unused nodes, tensors and initializers."""
        gs_graph = gs.import_onnx(self.model)
        gs_graph.cleanup()
        self.model = gs.export_onnx(gs_graph)

    def replace_custom_domain_nodes(self):
        """Replace custom domain nodes with standard ONNX nodes."""
        for node in self.model.graph.node:
            if node.domain.startswith("com.microsoft") and node.op_type in self.standard_ops:
                logger.warning(
                    f"Replacing custom domain node {node.name}, domain {node.domain} with standard ONNX "
                    f"node {node.op_type}"
                )
                node.domain = ""
