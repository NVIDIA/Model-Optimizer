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

"""Node classification module for AutoCast.

This module provides classes for classifying ONNX nodes based on various rules
to determine which nodes should be converted to lower precision and which should
remain in high precision. It includes rules for handling node names, operation types,
initializer ranges, and I/O value ranges.
"""

import abc
import re

import numpy as np
import onnx

from modelopt.onnx.autocast.logging_config import configure_logging, logger

configure_logging()


class NodeRuleBase:
    """Base class for node classification rules.

    This class defines the interface for rules that determine whether a node
    should be kept in high precision or converted to low precision.
    """

    @abc.abstractmethod
    def _check_inner(self, node):
        """Implement this method to check if node conversion should be skipped based on rule criteria."""

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes."""
        logger.info(f"Skipping node {node.name}: {self.__class__.__name__}")

    def check(self, node):
        """Check if a node should be skipped based on the rule.

        Args:
            node: The ONNX node to check.

        Returns:
            bool: True if the node should be kept in high precision, False otherwise.
        """
        result = self._check_inner(node)
        if result:
            self._log_skipped(node)
            return True
        return False


class DisabledNodeNameRegexRule(NodeRuleBase):
    """Rule for keeping nodes with matching names in high precision."""

    def __init__(self, disabled_node_name_regex):
        """Initialize the rule.

        Args:
            disabled_node_name_regex: List of regex patterns for node names to keep in high precision.
        """
        self.disabled_node_name_regex = disabled_node_name_regex

    def _check_inner(self, node):
        return any(re.match(regex, node.name) for regex in self.disabled_node_name_regex)


class DisabledOpTypes(NodeRuleBase):
    """Rule for keeping nodes with specific operation types in high precision."""

    def __init__(self, op_types_to_exclude):
        """Initialize the rule.

        Args:
            op_types_to_exclude: List of operation types to keep in high precision.
        """
        self.op_types_to_exclude = op_types_to_exclude

    def _check_inner(self, node):
        return node.op_type in self.op_types_to_exclude


class IncludeNodeNameRegexRule(DisabledNodeNameRegexRule):
    """Rule for force-including nodes with matching names in low precision.

    Inherits matching behavior from DisabledNodeNameRegexRule but overrides logging.
    """

    def _log_skipped(self, node, **kwargs):
        # For include rules, a positive match means we will force-include the node in low precision
        logger.info(f"Force-including node {node.name}: {self.__class__.__name__}")


class IncludeOpTypes(DisabledOpTypes):
    """Rule for force-including specific operation types in low precision.

    Inherits matching behavior from DisabledOpTypes but overrides logging.
    """

    def _log_skipped(self, node, **kwargs):
        # For include rules, a positive match means we will force-include the node in low precision
        logger.info(f"Force-including node {node.name}: {self.__class__.__name__}")


class InitializerRangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range initializers in high precision."""

    def __init__(self, init_max, node_to_init_map):
        """Initialize the rule.

        Args:
            init_max: Maximum absolute value allowed for initializers.
            node_to_init_map: Mapping from node names to their initializers.
        """
        self.init_max = init_max
        self.node_to_init_map = node_to_init_map
        self.init_data = (None, None)

    def _check_inner(self, node):
        for init in self.node_to_init_map[node.name]:
            np_array = onnx.numpy_helper.to_array(init)
            if np_array.dtype == np.float32 and np.any(np.abs(np_array) > self.init_max):
                self.init_data = init.name, np_array
                return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with initializer range violations."""
        if self.init_data[1] is not None:
            logger.info(
                f"Skipping node {node.name}: initializer {self.init_data[0]} out of range: "
                f"min={self.init_data[1].min()}, max={self.init_data[1].max()}, range=[{-self.init_max},"
                f"{self.init_max}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class IORangeRule(NodeRuleBase):
    """Rule for keeping nodes with out-of-range inputs/outputs in high precision."""

    def __init__(self, data_max, reference_data, node_to_init_map):
        """Initialize the rule.

        Args:
            data_max: Maximum absolute value allowed for node I/O.
            reference_data: Reference data for checking I/O ranges.
            node_to_init_map: Mapping from node names to their initializers.
        """
        self.data_max = data_max
        self.reference_data = reference_data
        self.node_to_init_map = node_to_init_map
        self.output_data = None

    def _check_inner(self, node):
        def is_io_out_of_range(node, tensor_name):
            if tensor_name not in self.reference_data:
                # Issue a warning only if the tensor is not an initializer/network input
                init_names = [init.name for init in self.node_to_init_map.get(node.name, [])]
                init_names.extend(node.input)
                if tensor_name not in init_names:
                    logger.warning(
                        f"Node {node.name}: Tensor {tensor_name} not found in reference data."
                    )
                return False
            ref_data = self.reference_data[tensor_name]
            if ref_data.size == 0:
                logger.debug(
                    f"Node {node.name}: Tensor {tensor_name} has size 0. Skipping I/O range check."
                )
                return False
            logger.debug(
                f"Node {node.name}: reference data: min={np.min(ref_data)}, max={np.max(ref_data)}"
            )
            if np.any(np.abs(ref_data) > self.data_max):
                self.output_data = ref_data
                return True

        if node.op_type == "Constant":
            return False
        if self.reference_data:
            for in_name in node.input:
                if is_io_out_of_range(node, in_name):
                    return True
            for out_name in node.output:
                if is_io_out_of_range(node, out_name):
                    return True
        return False

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with I/O range violations."""
        if self.output_data is not None:
            logger.info(
                f"Skipping node {node.name}: reference IO out of range: min={np.min(self.output_data)}, "
                f"max={np.max(self.output_data)}, range=[{-self.data_max}, {self.data_max}]"
            )
        else:
            super()._log_skipped(node, **kwargs)


class DepthOfReductionRule(NodeRuleBase):
    """Rule for keeping nodes with high depth of reduction in high precision."""

    def __init__(self, max_depth_of_reduction, reference_data, node_to_init_map, initializer_map):
        """Initialize the rule.

        Args:
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
            reference_data: Reference data for checking I/O ranges.
            node_to_init_map: Mapping from node names to their initializers.
            initializer_map: Mapping from initializer names to initializers.
        """
        self.max_depth_of_reduction = max_depth_of_reduction
        self.reference_data = reference_data
        self.node_to_init_map = node_to_init_map
        self.initializer_map = initializer_map
        self.reduction_depth = 0

    def _get_tensor_shape(self, tensor_name):
        """Get tensor shape from reference data."""
        if tensor_name in self.reference_data:
            return self.reference_data[tensor_name].shape
        if tensor_name in self.initializer_map:
            return self.initializer_map[tensor_name].dims
        return None

    def _log_skipped(self, node, **kwargs):
        """Log information about skipped nodes with depth of reduction violations."""
        if self.reduction_depth > 0:
            logger.info(
                f"Skipping node {node.name}: depth of reduction {self.reduction_depth} exceeds "
                f"{self.max_depth_of_reduction}."
            )
        else:
            super()._log_skipped(node, **kwargs)

    def _check_inner(self, node):
        # All reduction ops rely on shape of input[0]
        input_0_dims = self._get_tensor_shape(node.input[0]) if len(node.input) > 0 else None
        if input_0_dims is None:
            return False
        self.reduction_depth = 0
        if node.op_type == "Attention":
            # Attention: input (batch_size, sequence_length, hidden_size)
            # or (batch_size, kv_num_heads, total_sequence_length, head_size)
            assert len(input_0_dims) == 3 or len(input_0_dims) == 4
            hidden_size = (
                input_0_dims[2] if len(input_0_dims) == 3 else input_0_dims[1] * input_0_dims[3]
            )
            self.reduction_depth = hidden_size
        elif node.op_type == "Conv":
            # Conv: input (N x C x D1 x D2 ... x Dn)
            # weight (out_channels, in_channels, kD1, kD2, ... kDn)
            # Reduction depth = in_channels * kernel_volume
            weight_shape = self._get_tensor_shape(node.input[1]) if len(node.input) > 1 else None
            if weight_shape is None:
                return False
            in_channels = weight_shape[1]
            kernel_volume = np.prod(weight_shape[2:])
            self.reduction_depth = in_channels * kernel_volume
        elif node.op_type == "CumSum":
            axis_name = node.input[1] if len(node.input) > 1 else None
            if axis_name is None:
                return False
            # Find the axis initializer
            axis_init = None
            for init in self.node_to_init_map.get(node.name, []):
                if init.name == axis_name:
                    axis_init = init
                    break
            if axis_init is None:
                return False
            axis_array = onnx.numpy_helper.to_array(axis_init)
            assert axis_array.ndim == 0 or (axis_array.ndim == 1 and axis_array.size == 1)
            axis = int(axis_array.item())
            # Normalize negative axis and check bounds
            if axis < 0:
                axis = len(input_0_dims) + axis
            if axis < 0 or axis >= len(input_0_dims):
                return False
            if input_0_dims[axis] > self.max_depth_of_reduction:
                self.reduction_depth = input_0_dims[axis]

        elif node.op_type == "Gemm":
            # GEMM: A (M, K) @ B (K, N) = C (M, N)
            # Check for transpose attributes
            trans_a = False
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = bool(attr.i)

            # Get K dimension based on transpose flag
            self.reduction_depth = (
                input_0_dims[0] if trans_a else input_0_dims[1]
            )  # A is (K, M) when transposed

        elif node.op_type == "MatMul":
            # MatMul: (..., M, K) @ (..., K, N) = (..., M, N)
            # K is the last dimension of first input
            if len(input_0_dims) >= 2:
                self.reduction_depth = input_0_dims[-1]
        elif node.op_type == "Mean":
            self.reduction_depth = len(input_0_dims)
        elif node.op_type in [
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMean",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        ]:
            if len(node.input) > 1:
                axes_name = node.input[1]
                # Find the axes initializer
                axes_init = None
                for init in self.node_to_init_map.get(node.name, []):
                    if init.name == axes_name:
                        axes_init = init
                        break
                if axes_init is None:
                    return False
                axes_array = onnx.numpy_helper.to_array(axes_init)
                if axes_array.ndim == 0:
                    axes_array = [int(axes_array.item())]
                else:
                    assert axes_array.ndim == 1
                    axes_array = axes_array.astype(np.int64)
            else:
                axes_array = range(len(input_0_dims))
            for axis in axes_array:
                if input_0_dims[axis] > self.max_depth_of_reduction:
                    self.reduction_depth = input_0_dims[axis]
                    return True
        return self.reduction_depth > self.max_depth_of_reduction


class NodeClassifier:
    """Main class for classifying nodes into high and low precision groups."""

    def __init__(
        self,
        model: onnx.ModelProto,
        node_to_init_map: dict[str, list[onnx.TensorProto]] | None = None,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
        nodes_to_exclude: list[str] | None = None,
        op_types_to_exclude: list[str] | None = None,
        nodes_to_include: list[str] | None = None,
        op_types_to_include: list[str] | None = None,
        custom_rule: NodeRuleBase | None = None,
        data_max: float | None = 1000.0,
        init_max: float | None = np.finfo(np.float16).max,
        max_depth_of_reduction: int | None = None,
        custom_ops_low_precision_nodes: list[str] | None = None,
    ):
        """Initialize the node classifier.

        Args:
            model: The ONNX model to classify nodes for.
            node_to_init_map: Mapping from node names to their initializers.
            initializer_map: Mapping from initializer names to their tensors.
            nodes_to_exclude: List of regex patterns for node names to keep in high precision.
            op_types_to_exclude: List of operation types to keep in high precision.
            nodes_to_include: List of regex patterns for node names to force-include in low precision.
            op_types_to_include: List of operation types to force-include in low precision.
            custom_rule: Optional custom classification rule.
            data_max: Maximum absolute value allowed for node I/O.
            init_max: Maximum absolute value allowed for initializers.
            max_depth_of_reduction: Maximum depth of reduction allowed in low precision.
            custom_ops_low_precision_nodes: List of custom op node names to convert to low precision.
        """
        self.model = model
        self.node_to_init_map = node_to_init_map
        self.initializer_map = initializer_map
        self.nodes_to_exclude = nodes_to_exclude
        self.op_types_to_exclude = op_types_to_exclude
        self.nodes_to_include = nodes_to_include
        self.op_types_to_include = op_types_to_include
        self.custom_rule = custom_rule
        self.data_max = data_max
        self.init_max = init_max
        self.max_depth_of_reduction = max_depth_of_reduction
        self.custom_ops_low_precision_nodes = custom_ops_low_precision_nodes

    def _gen_exclude_node_rules(self, reference_data):
        """Generate list of rules for blocking nodes from precision conversion.

        Args:
            reference_data: Reference data for checking I/O ranges.

        Returns:
            list[NodeRuleBase]: List of rules to apply.
        """
        block_node_rules: list[NodeRuleBase] = []
        if self.nodes_to_exclude:
            block_node_rules.append(DisabledNodeNameRegexRule(self.nodes_to_exclude))
        if self.op_types_to_exclude:
            block_node_rules.append(DisabledOpTypes(self.op_types_to_exclude))
        if self.init_max is not None:
            block_node_rules.append(InitializerRangeRule(self.init_max, self.node_to_init_map))
        if reference_data:
            block_node_rules.append(
                IORangeRule(self.data_max, reference_data, self.node_to_init_map)
            )
        if self.max_depth_of_reduction is not None:
            block_node_rules.append(
                DepthOfReductionRule(
                    self.max_depth_of_reduction,
                    reference_data,
                    self.node_to_init_map,
                    self.initializer_map,
                )
            )
        if self.custom_rule:
            block_node_rules.append(self.custom_rule)
        return block_node_rules

    def _gen_include_node_rules(self):
        """Generate list of rules for force-including nodes in low precision.

        Returns:
            list[NodeRuleBase]: List of rules to apply.
        """
        include_node_rules: list[NodeRuleBase] = []
        if self.nodes_to_include:
            include_node_rules.append(IncludeNodeNameRegexRule(self.nodes_to_include))
        if self.op_types_to_include:
            include_node_rules.append(IncludeOpTypes(self.op_types_to_include))

        return include_node_rules

    def run(self, ref_outputs_dict=None):
        """Run node classification.

        Args:
            ref_outputs_dict: Optional tensors' reference data.

        Returns:
            tuple: Lists of node names (low_precision_nodes, high_precision_nodes).
        """
        exclude_node_rules = self._gen_exclude_node_rules(ref_outputs_dict)
        include_node_rules = self._gen_include_node_rules()
        # Use a set to avoid duplicates
        low_precision_nodes_set = set(self.custom_ops_low_precision_nodes or [])
        high_precision_nodes = []
        for node in self.model.graph.node:
            # Skip if already classified as low precision
            if node.name in low_precision_nodes_set:
                continue
            # If any condition is met - node will be executed in high precision
            if any(rule.check(node) for rule in exclude_node_rules) and not any(
                rule.check(node) for rule in include_node_rules
            ):
                high_precision_nodes.append(node.name)
            else:
                low_precision_nodes_set.add(node.name)
        # Convert back to list for return value
        low_precision_nodes = list(low_precision_nodes_set)
        logger.debug(f"Low Precision Nodes: {low_precision_nodes}")
        logger.debug(f"High Precision Nodes: {high_precision_nodes}")
        return low_precision_nodes, high_precision_nodes
