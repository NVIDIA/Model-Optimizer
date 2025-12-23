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

"""Q/DQ Insertion Point Management for ONNX Quantization.

This module provides data structures and utilities for managing Quantization/Dequantization (Q/DQ)
insertion points in ONNX computational graphs during autotune optimization. It enables pattern-based
Q/DQ insertion that can be reused across multiple matching regions in a model.

Core Concepts:
--------------
1. **Pattern-Relative Insertion Points**: Insertion points are defined relative to region patterns
   rather than absolute node IDs, enabling scheme reuse across all matching regions.

2. **Resolution Process**: Pattern-relative indices are resolved to actual tensor names for each
   specific region instance, then Q/DQ pairs are inserted at the resolved locations.

3. **Hierarchical Support**: Supports Q/DQ insertion at multiple levels:
   - Node inputs within regions
   - Child region boundaries (inputs/outputs)
   - Region outputs

Classes:
--------
- ResolvedInsertionPoint: Resolved Q/DQ insertion point with actual tensor name
- NodeInputInsertionPoint: Pattern-relative insertion point at node inputs
- ChildRegionInputInsertionPoint: Pattern-relative insertion point at child region inputs
- RegionOutputInsertionPoint: Pattern-relative insertion point at region/node outputs

Utilities:
----------
- skip_invalid_insertion_points(): Filter out non-quantizable tensors
- has_quantizable_operations(): Check if region contains major quantizable ops
- resolve_region_io_insertion_points(): Resolve region I/O to actual insertion points
- merge_resolved_insertion_points(): Merge insertion points when all users are quantized

Constants:
----------
- BOOL_OPERATIONS: Boolean/comparison operations (not quantizable)
- SHAPE_OPERATIONS: Shape manipulation operations (not quantizable)
- MAJOR_QUANTIZABLE_OPERATIONS: Key operations that benefit from quantization
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx_graphsurgeon as gs

if TYPE_CHECKING:
    from modelopt.onnx.quantization.autotune.common import Region

from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

BOOL_OPERATIONS = {
    "Not",
    "And",
    "Or",
    "Xor",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
    "IsNaN",
    "IsInf",
    "Sign",
    "Abs",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Where",
    "Max",
    "Min",
    "Mean",
    "Median",
    "ArgMax",
    "ArgMin",
    "ReduceMax",
    "ReduceMin",
    "ReduceSum",
    "ReduceMean",
    "All",
    "Any",
    "Unique",
    "NonZero",
    "TopK",
}

SHAPE_OPERATIONS = {
    "Cast",
    "Ceil",
    "Clip",
    "Compress",
    "Concat",
    "ExpandDims",
    "Flatten",
    "Gather",
    "GatherElements",
    "GatherND",
    "Identity",
    "Pad",
    "Range",
    "Scatter",
    "ScatterND",
    "Shape",
    "Slice",
    "Split",
    "Squeeze",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "View",
}

MAJOR_QUANTIZABLE_OPERATIONS = {
    "Conv",
    "ConvTranspose",
    "Gemm",
    "MatMul",
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Resize",
    "Add",
    "Sum",
    "Mul",
    "Relu",
}


@dataclass(frozen=True)
class ResolvedInsertionPoint:
    """Resolved Q/DQ insertion point with actual tensor name and optional node context.

    After resolving pattern-relative insertion points, this class represents the
    actual location where Q/DQ pairs should be inserted in the graph.

    **Insertion Modes:**
    1. Node-specific insertion (node_index and input_index are set):
       - Inserts Q/DQ at a specific input of a specific node
       - More precise control over where quantization happens
    2. Tensor-level insertion (node_index and input_index are None):
       - Inserts Q/DQ for all users of the tensor
       - Used when all consumers of a tensor should be quantized together

    **Attributes:**
    - tensor_name: Name of the tensor where Q/DQ should be inserted
    - node_index: Absolute graph node index (not pattern-relative), or None for tensor-level insertion
    - input_index: Input tensor index of that node, or None for tensor-level insertion

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    tensor_name: str
    # Absolute graph node index (or None for tensor-level insertion)
    node_index: int | None = None
    # Input tensor index of that node (or None)
    input_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tensor_name": self.tensor_name,
            "node_index": self.node_index,
            "input_index": self.input_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedInsertionPoint":
        """Create from dictionary."""
        return cls(
            tensor_name=data["tensor_name"],
            node_index=data["node_index"],
            input_index=data.get("input_index"),
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"ResolvedInsertionPoint(tensor_name={self.tensor_name}, "
            f"node={self.node_index}, input={self.input_index})"
        )


@dataclass(frozen=True)
class NodeInputInsertionPoint:
    """Pattern-relative Q/DQ insertion point at a node's input.

    Specifies where to insert a Q/DQ pair within a region pattern using
    pattern-relative indices rather than absolute node IDs. This enables
    insertion scheme reuse across all regions matching the same pattern.

    **Resolution Process:**
    1. Pattern-relative indices (node_index, input_index) are defined once
    2. For each matching region, indices are resolved to actual tensor names
    3. Q/DQ pairs are inserted at the resolved tensor locations

    **Example:**
    - NodeInputInsertionPoint(node_index=0, input_index=1)
    - Resolves to: the second input (index 1) of the first node (index 0) in the pattern
    - Actual tensor name depends on the specific region instance

    **Attributes:**
    - node_index: Index of the node within the pattern's sorted node list (0-based)
    - input_index: Index of the input tensor for that node (0-based)

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    # Pattern-relative node index
    node_index: int
    # Input tensor index of that node
    input_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"node_index": self.node_index, "input_index": self.input_index}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeInputInsertionPoint":
        """Create from dictionary."""
        return cls(node_index=data["node_index"], input_index=data["input_index"])

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"NodeInputInsertionPoint(node={self.node_index}, input={self.input_index})"

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a node input insertion point to actual tensor names for a matching region.

        Converts pattern-relative node/input indices to absolute node indices and actual
        tensor names in the graph. Special handling for Conv/ConvTranspose operations
        automatically includes weight quantization when input is quantized.

        Args:
            region: The region instance matching this pattern
            graph: The ONNX graph containing the nodes

        Returns:
            Set of ResolvedInsertionPoint objects with actual tensor names
        """
        nodes_list = list(graph.nodes)
        node_indices = sorted(region.get_nodes())
        resolved_ips = set()

        # Map from pattern-relative node index to absolute graph node index
        assert self.node_index < len(node_indices), "Node index out of range"
        actual_node_idx = node_indices[self.node_index]
        assert actual_node_idx < len(nodes_list), "Node index out of range"
        node = nodes_list[actual_node_idx]
        assert self.input_index < len(node.inputs), "Input index out of range"

        # Resolve the input tensor name using input_index
        inp = node.inputs[self.input_index]
        if hasattr(inp, "name") and inp.name:
            ip = ResolvedInsertionPoint(
                tensor_name=inp.name, node_index=actual_node_idx, input_index=self.input_index
            )
            resolved_ips.add(ip)

        if node.op in ["Conv", "ConvTranspose"]:
            assert self.input_index == 0, (
                "Conv and ConvTranspose inputs and weights should be quantized at same time"
            )
            assert len(node.inputs) >= 2, "Conv and ConvTranspose should have at least 2 inputs"
            inp = node.inputs[1]
            if hasattr(inp, "name") and inp.name:
                ip = ResolvedInsertionPoint(
                    tensor_name=inp.name, node_index=actual_node_idx, input_index=1
                )
                resolved_ips.add(ip)

        return resolved_ips

    @staticmethod
    def collect_from_region(region: "Region", graph: gs.Graph) -> list["NodeInputInsertionPoint"]:
        """Collect all valid node input insertion points from a region.

        Analyzes each node in the region and identifies all valid input tensors
        where Q/DQ pairs could be inserted. Filters out invalid insertion points
        using skip_invalid_insertion_points().

        Args:
            region: The region to collect insertion points from
            graph: The ONNX graph containing the nodes

        Returns:
            List of NodeInputInsertionPoint objects representing valid insertion locations
        """
        nodes_list = list(graph.nodes)
        node_indices = sorted(region.get_nodes())

        node_input_insertion_points = []
        for local_idx, node_idx in enumerate(node_indices):
            assert node_idx < len(nodes_list), "Node index out of range"
            node = nodes_list[node_idx]
            # Analyze each input of the node
            for input_idx, inp in enumerate(node.inputs):
                # Skip if tensor doesn't have a valid name
                if not (hasattr(inp, "name") and inp.name):
                    continue
                # Skip if insertion point is invalid (wrong dtype, small size, special input, etc.)
                if skip_invalid_insertion_points(graph, inp.name, node):
                    continue
                # Create insertion point for valid tensor
                ip = NodeInputInsertionPoint(
                    # Pattern-relative node index
                    node_index=local_idx,
                    input_index=input_idx,
                )
                node_input_insertion_points.append(ip)

        return node_input_insertion_points


@dataclass(frozen=True)
class ChildRegionInputInsertionPoint:
    """Pattern-relative Q/DQ insertion point at a child region's input boundary.

    Specifies where to insert Q/DQ pairs at the input boundaries of child regions
    within COMPOSITE regions. This allows parent regions to control quantization
    at child boundaries, potentially overriding or complementing child region
    optimizations.

    **Use Case:**
    Parent regions can insert Q/DQ pairs at child region inputs to:
    - Add quantization at child boundaries even if the child has no internal Q/DQ
    - Override or supplement the child's own boundary Q/DQ decisions
    - Apply different quantization schemes based on the parent context

    **Resolution Process:**
    1. Pattern-relative indices (region_index, input_index) are defined once
    2. For each matching parent region, indices resolve to actual child boundaries:
       - region_index identifies which child region (in parent's sorted child list)
       - input_index identifies which input tensor of that child region
    3. Q/DQ pairs are inserted at the resolved child input tensor locations

    **Example:**
    - ChildRegionInputInsertionPoint(region_index=0, input_index=1)
    - Resolves to: the second input tensor (index 1) of the first child region (index 0)
    - Actual tensor name depends on the specific parent/child region instances

    **Note:** Only applies to COMPOSITE regions. LEAF regions have no children,
    so child region insertion points have no effect there.

    **Attributes:**
    - region_index: Index of the child region within the parent pattern's sorted child list (0-based)
    - input_index: Index of the input tensor for that child region (0-based)

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    # Index of the child region within the parent pattern's sorted child list (0-based)
    region_index: int
    # Index of the input tensor for that child region (0-based)
    input_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"region_index": self.region_index, "input_index": self.input_index}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChildRegionInputInsertionPoint":
        """Create from dictionary.

        Backward compatible: Ignores obsolete fields like 'child_region_id'
        from older serialization formats.

        Args:
            data: Dictionary with 'region_index' and 'input_index' keys

        Returns:
            ChildRegionInputInsertionPoint instance
        """
        # Ignore child_region_id if present in old data
        return cls(region_index=data["region_index"], input_index=data["input_index"])

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"ChildRegionInputInsertionPoint(region={self.region_index}, input={self.input_index})"
        )

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a child region input insertion point to actual tensor names for a matching region.

        Converts pattern-relative child region index and input index to the actual tensor
        name at that child region's input boundary, then resolves to all node inputs that
        consume that tensor.

        Args:
            region: The parent region instance matching this pattern
            graph: The ONNX graph containing the nodes

        Returns:
            Set of ResolvedInsertionPoint objects with actual tensor names.
            Returns empty set for LEAF regions (no children).
        """
        from modelopt.onnx.quantization.autotune.common import RegionType

        if graph is None:
            raise ValueError("graph parameter is required")

        # LEAF regions have no child boundaries
        if region.get_type() == RegionType.LEAF:
            return set()

        # Get sorted child regions (must match order in RegionPattern._compute_signature_recursive)
        children_regions = region.get_children()
        children_regions = sorted(
            children_regions, key=lambda r: (-r.get_level(), r.get_total_size())
        )
        # Map from pattern-relative child index to actual child region
        resolved_ips = set()
        assert self.region_index < len(children_regions), "Child region index out of range"
        child_region = children_regions[self.region_index]
        assert self.input_index < len(child_region.get_inputs()), "Input index out of range"
        # Resolve the input tensor name using input_index
        tensor_name = child_region.get_inputs()[self.input_index]
        assert tensor_name is not None, "Tensor name is required"
        resolved_ips.update(resolve_region_io_insertion_points(child_region, graph, tensor_name))

        return resolved_ips

    @staticmethod
    def collect_from_region(
        region: "Region", graph: gs.Graph
    ) -> list["ChildRegionInputInsertionPoint"]:
        """Collect all valid child region input insertion points from a region.

        For COMPOSITE regions, analyzes each child region and identifies all valid
        input tensors where Q/DQ pairs could be inserted at child boundaries.
        Returns empty list for LEAF regions (no children).

        Args:
            region: The parent region to collect insertion points from
            graph: The ONNX graph containing the nodes

        Returns:
            List of ChildRegionInputInsertionPoint objects representing valid insertion locations
        """
        from modelopt.onnx.quantization.autotune.common import RegionType

        child_region_input_insertion_points = []

        # Only COMPOSITE regions have child boundaries for Q/DQ insertion
        if region.get_type() != RegionType.LEAF:
            # Get all child regions, sorted for deterministic ordering
            # Must match sorting in _compute_signature_recursive to ensure
            # insertion point indices align with pattern structure
            children_regions = region.get_children()
            children_regions = sorted(
                children_regions, key=lambda r: (-r.get_level(), r.get_total_size())
            )

            for local_idx, child_region in enumerate(children_regions):
                # Create insertion point for each input tensor of the child region
                for input_idx, inp in enumerate(child_region.get_inputs()):
                    if skip_invalid_insertion_points(graph, inp, child_region):
                        continue
                    point = ChildRegionInputInsertionPoint(
                        # Child region index within parent pattern
                        region_index=local_idx,
                        # Input index within child region
                        input_index=input_idx,
                    )
                    child_region_input_insertion_points.append(point)

        return child_region_input_insertion_points


@dataclass(frozen=True)
class RegionOutputInsertionPoint:
    """Pattern-relative Q/DQ insertion point at an output location.

    Specifies where to insert Q/DQ pairs at output boundaries. This can be either:
    1. Output from a child region (in COMPOSITE regions)
    2. Output from a node within the region

    **Use Case:**
    Parent regions can:
    - Add Q/DQ at child region output boundaries
    - Add Q/DQ at node outputs within the region
    - Control quantization precision as data flows through the region hierarchy

    **Resolution Process:**
    1. Pattern-relative indices are defined once
    2. If output is from a child region: use region_index (node_index is None)
       - region_index identifies which child region (in sorted order)
       - output_index identifies which output tensor of that child region
    3. If output is from a node: use node_index (region_index is None)
       - node_index identifies which node (in sorted order)
       - output_index identifies which output tensor of that node
    4. Resolves to the actual tensor name at that output location

    **Examples:**
    - RegionOutputInsertionPoint(region_index=0, node_index=None, output_index=0)
      → First output of the first child region
    - RegionOutputInsertionPoint(region_index=None, node_index=2, output_index=1)
      → Second output of the third node

    **Note:** Exactly one of region_index or node_index must be set (the other must be None).

    **Attributes:**
    - region_index: Index of child region within parent pattern (0-based), or None
    - node_index: Index of node within the region (0-based), or None
    - output_index: Index of the output tensor (0-based)

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    # Index of child region within parent pattern (0-based), or None
    region_index: int | None
    # Index of node within the region (0-based), or None
    node_index: int | None
    # Index of the output tensor (0-based)
    output_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region_index": self.region_index,
            "node_index": self.node_index,
            "output_index": self.output_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionOutputInsertionPoint":
        """Create from dictionary.

        Args:
            data: Dictionary with 'region_index', 'node_index', and 'output_index' keys

        Returns:
            RegionOutputInsertionPoint instance
        """
        return cls(
            region_index=data.get("region_index"),
            node_index=data.get("node_index"),
            output_index=data["output_index"],
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.region_index is not None:
            return f"RegionOutputInsertionPoint(region={self.region_index}, output={self.output_index})"
        else:
            return f"RegionOutputInsertionPoint(node={self.node_index}, output={self.output_index})"

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a region output insertion point to actual tensor names for a matching region.

        Converts pattern-relative indices to the actual tensor name at an output location:
        - If region_index is set: Resolves to a child region's output tensor
        - If node_index is set: Resolves to a node's output tensor

        Then identifies all node inputs that consume that output tensor.

        Args:
            region: The region instance matching this pattern
            graph: The ONNX graph containing the nodes

        Returns:
            Set of ResolvedInsertionPoint objects with actual tensor names
        """
        if graph is None:
            raise ValueError("graph parameter is required")

        # Get sorted nodes for node output resolution
        nodes_list = list(graph.nodes)
        node_indices = sorted(region.get_nodes())
        children_regions = region.get_children()
        children_regions = sorted(
            children_regions, key=lambda r: (-r.get_level(), r.get_total_size())
        )

        # Resolve each region output insertion point from the scheme to actual tensor names
        resolved_ips = set()
        # Handle child region outputs (region_index is set)
        if self.region_index is not None:
            assert self.region_index < len(children_regions), "Region index out of range"
            child_region = children_regions[self.region_index]
            assert self.output_index < len(child_region.get_outputs()), "Output index out of range"
            tensor_name = child_region.get_outputs()[self.output_index]
            assert tensor_name is not None, "Invalid tensor name"
            resolved_ips.update(
                resolve_region_io_insertion_points(child_region, graph, tensor_name)
            )
        # Handle node outputs (node_index is set)
        elif self.node_index is not None:
            assert self.node_index < len(node_indices), "Node index out of range"
            node_idx = node_indices[self.node_index]
            assert node_idx < len(nodes_list), "Node index out of range"
            node = nodes_list[node_idx]
            assert self.output_index < len(node.outputs), "Output index out of range"
            tensor = node.outputs[self.output_index]
            assert tensor is not None, "Invalid tensor name"
            assert hasattr(tensor, "name") and tensor.name, "Tensor name is required"
            resolved_ips.update(resolve_region_io_insertion_points(None, graph, tensor.name))
        return resolved_ips

    @staticmethod
    def collect_from_region(
        region: "Region", graph: gs.Graph
    ) -> list["RegionOutputInsertionPoint"]:
        """Collect all valid region output insertion points from a region.

        Identifies all valid output tensors (from child regions or nodes) that leave
        the region boundary and could have Q/DQ pairs inserted. Only includes outputs
        that are actual region outputs (not consumed internally).

        For COMPOSITE regions:
        - Collects child region outputs that are also region outputs
        - Collects node outputs that are region outputs

        For LEAF regions:
        - Only collects node outputs that are region outputs

        Args:
            region: The region to collect insertion points from
            graph: The ONNX graph containing the nodes

        Returns:
            List of RegionOutputInsertionPoint objects representing valid insertion locations
        """
        from modelopt.onnx.quantization.autotune.common import RegionType

        nodes_list = list(graph.nodes)
        node_indices = sorted(region.get_nodes())
        region_outputs_set = set(region.get_outputs())

        # Only include outputs that are actual region outputs (leave the region)
        region_output_insertion_points = []
        if region.get_type() != RegionType.LEAF:
            # For COMPOSITE regions: check if child region output is a region output
            children_regions = region.get_children()
            children_regions = sorted(
                children_regions, key=lambda r: (-r.get_level(), r.get_total_size())
            )
            for local_idx, child_region in enumerate(children_regions):
                for output_idx, out in enumerate(child_region.get_outputs()):
                    if out not in region_outputs_set:
                        continue
                    if skip_invalid_insertion_points(graph, out, child_region):
                        continue
                    point = RegionOutputInsertionPoint(
                        region_index=local_idx,
                        node_index=None,
                        output_index=output_idx,
                    )
                    region_output_insertion_points.append(point)
        # For all regions: check if node output is a region output
        for local_idx, node_idx in enumerate(node_indices):
            assert node_idx < len(nodes_list), "Node index out of range"
            node = nodes_list[node_idx]
            for output_idx, out in enumerate(node.outputs):
                # Skip if tensor doesn't have a valid name
                if not (hasattr(out, "name") and out.name):
                    continue
                # Skip if this output is not a region output (i.e., it's consumed internally)
                if out.name not in region_outputs_set:
                    continue
                # Skip if insertion point is invalid (wrong dtype, small size, etc.)
                if skip_invalid_insertion_points(graph, out.name, node):
                    continue
                # Create insertion point for valid output tensor
                point = RegionOutputInsertionPoint(
                    region_index=None,
                    node_index=local_idx,
                    output_index=output_idx,
                )
                region_output_insertion_points.append(point)

        return region_output_insertion_points


InsertionPointType = (
    NodeInputInsertionPoint | ChildRegionInputInsertionPoint | RegionOutputInsertionPoint
)


def skip_invalid_insertion_points(
    graph: gs.Graph, tensor_name: str, region_or_node: "Region | gs.Node"
) -> bool:
    """Determine if a tensor should be skipped for Q/DQ insertion.

    Filters out tensors that are not suitable for quantization based on various criteria:
    - Boolean and shape operations (not quantizable)
    - Fused operation patterns (Conv->BatchNorm->ReLU)
    - Operation-specific non-quantizable inputs (weights, biases, BN parameters)
    - Non-floating-point tensors (indices, masks)
    - Small tensors (scalars, small vectors with < 8 elements)

    Args:
        graph: The ONNX graph containing the nodes
        tensor_name: Name of the tensor to evaluate
        region_or_node: Either a Region or a Node to check for usage of this tensor

    Returns:
        True if the insertion point should be skipped, False if it's valid for quantization
    """
    from modelopt.onnx.quantization.autotune.common import Region

    if isinstance(region_or_node, Region):
        node_indices = region_or_node.get_all_nodes_recursive()
        nodes: list[gs.Node] = [graph.nodes[node_idx] for node_idx in node_indices]
    else:
        assert isinstance(region_or_node, gs.Node)
        nodes = [region_or_node]

    for node in nodes:
        for input_idx, inp in enumerate(node.inputs):
            if hasattr(inp, "name") and inp.name == tensor_name:
                # Skip weights of Conv and ConvTranspose, they should be quantized with inputs at same time
                if node.op in ["Conv", "ConvTranspose"] and input_idx >= 1:
                    return True
                if node.op in ["Relu", "LeakyRelu", "Softmax"]:
                    # Conv -> ReLU/LeakyRelu/Softmax
                    if len(node.inputs) == 1 and len(node.inputs[0].inputs) == 1:
                        producer = node.inputs[0].inputs[0]
                        if producer.op in ["Conv", "ConvTranspose"]:
                            return True
                    # Conv -> BatchNormalization -> ReLU/LeakyRelu/Softmax
                    if len(node.inputs) == 1 and len(node.inputs[0].inputs) == 1:
                        producer = node.inputs[0].inputs[0]
                        if producer.op == "BatchNormalization":
                            assert len(producer.inputs) >= 1, (
                                "BN node should have more than one inputs"
                            )
                            if len(producer.inputs[0].inputs) == 1:
                                producer = producer.inputs[0].inputs[0]
                                if producer.op in ["Conv", "ConvTranspose"]:
                                    return True
                # Conv -> BatchNormalization -> ReLU/LeakyRelu/Softmax
                if node.op == "BatchNormalization":
                    assert len(node.inputs) >= 1, "BN node should have more than one inputs"
                    if len(node.inputs[0].inputs) == 1:
                        producer = node.inputs[0].inputs[0]
                        if producer.op in ["Conv", "ConvTranspose"]:
                            return True
                # Filter 1: out boolean operations
                if node.op in BOOL_OPERATIONS:
                    return True
                # Filter 2: out shape operations
                if node.op in SHAPE_OPERATIONS:
                    return True
                # Filter 3: Skip operation-specific non-quantizable inputs
                if node.op in ["BatchNormalization", "Resize"] and input_idx >= 1:
                    return True
                if node.op in ["Conv", "Gemm"] and input_idx >= 2:
                    return True
                # Filter 4: Skip non-floating-point tensors (int/bool indices, masks, etc.)
                if hasattr(inp, "dtype") and inp.dtype not in [
                    None,
                    np.float32,
                    np.float16,
                    np.float64,
                ]:
                    return True
                # Filter 5: Skip small tensors (scalars, small vectors)
                if hasattr(inp, "shape") and inp.shape is not None:
                    if all(isinstance(s, int) for s in inp.shape):
                        if np.prod(inp.shape) < 8:
                            return True
    return False


def has_quantizable_operations(region: "Region", graph: gs.Graph) -> bool:
    """Check if a region contains major quantizable operations.

    Args:
        region: The region to check
        graph: The ONNX graph containing the nodes

    Returns:
        True if the region contains major quantizable operations, False otherwise
    """
    from modelopt.onnx.quantization.autotune.common import RegionType

    # only check leaf regions for quantizable operations
    if region.get_type() == RegionType.LEAF:
        region_ops = {graph.nodes[idx].op for idx in region.get_nodes()}
        return bool(region_ops.intersection(MAJOR_QUANTIZABLE_OPERATIONS))
    return True


def resolve_region_io_insertion_points(
    region: "Region | None", graph: gs.Graph, tensor_name: str
) -> set[ResolvedInsertionPoint]:
    """Resolve region input/output boundaries to actual Q/DQ insertion points.

    For a given tensor at a region boundary (input or output), this function
    identifies all the actual node inputs where Q/DQ pairs should be inserted.
    It considers both nodes within the region (if provided) and all users of
    the tensor in the graph.

    **Use Cases:**
    - Child region inputs: Find all nodes inside the child that consume the input tensor
    - Child region outputs: Find all nodes outside the child that consume the output tensor
    - Node outputs: Find all nodes that consume the tensor (region can be None)

    Args:
        region: The region to search within (or None to search entire graph)
        graph: The ONNX graph containing the nodes
        tensor_name: Name of the tensor at the region boundary

    Returns:
        Set of ResolvedInsertionPoint objects specifying where to insert Q/DQ pairs
    """
    resolved_insertion_points = set()
    tensor_users_map: dict[str, list[int]] = {}
    if hasattr(graph, "tensor_users_map"):
        tensor_users_map = graph.tensor_users_map
    if not tensor_users_map:
        tensor_users_map = get_tensor_consumer_node_indices(graph)

    if region is not None:
        for node_idx in region.get_all_nodes_recursive():
            assert node_idx < len(graph.nodes), "Node index out of range"
            node = graph.nodes[node_idx]
            for input_idx, inp in enumerate(node.inputs):
                if inp.name == tensor_name:
                    ip = ResolvedInsertionPoint(
                        tensor_name=tensor_name, node_index=node_idx, input_index=input_idx
                    )
                    resolved_insertion_points.add(ip)

    if tensor_name in tensor_users_map:
        for node_idx in tensor_users_map[tensor_name]:
            node = graph.nodes[node_idx]
            for input_idx, inp in enumerate(node.inputs):
                if inp.name == tensor_name:
                    ip = ResolvedInsertionPoint(
                        tensor_name=tensor_name, node_index=node_idx, input_index=input_idx
                    )
                    resolved_insertion_points.add(ip)

    return resolved_insertion_points


def merge_resolved_insertion_points(
    graph: gs.Graph, resolved_insertion_points: set[ResolvedInsertionPoint]
) -> set[ResolvedInsertionPoint]:
    """Optimize insertion points by merging node-specific insertions into tensor-level insertions.

    When all consumers (users) of a tensor have Q/DQ insertion points, it's more efficient
    to insert Q/DQ once at the tensor level rather than at each individual node input.
    This reduces the number of Q/DQ nodes in the graph and simplifies the quantization scheme.

    **Optimization Logic:**
    - For each tensor with multiple node-specific insertion points:
      - If ALL users of the tensor have insertion points → merge to tensor-level insertion
      - If SOME users have insertion points → keep node-specific insertions

    Args:
        graph: The ONNX graph containing the nodes
        resolved_insertion_points: Set of resolved insertion points to optimize

    Returns:
        Optimized set of insertion points with merged tensor-level insertions where possible
    """
    tensor_users_map = get_tensor_consumer_node_indices(graph)
    node_input_insertion_points = {
        ip for ip in resolved_insertion_points if ip.node_index is not None
    }
    tensor_names = {ip.tensor_name for ip in node_input_insertion_points}

    results = resolved_insertion_points.difference(node_input_insertion_points)
    for tensor_name in tensor_names:
        all_users = set(tensor_users_map[tensor_name])
        qdq_users = {
            user for user in node_input_insertion_points if user.tensor_name == tensor_name
        }
        qdq_user_ids = set({user.node_index for user in qdq_users})
        if all_users == qdq_user_ids:
            results.add(
                ResolvedInsertionPoint(tensor_name=tensor_name, node_index=None, input_index=None)
            )
        else:
            results.update(qdq_users)

    return results
