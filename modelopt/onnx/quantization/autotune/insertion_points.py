# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Q/DQ insertion point management for ONNX quantization autotune.

This module provides data structures and utilities for managing Quantization/Dequantization (Q/DQ)
insertion points in ONNX computational graphs during autotune optimization. It enables pattern-based
Q/DQ insertion that can be reused across multiple matching regions in a model.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx_graphsurgeon as gs

if TYPE_CHECKING:
    from modelopt.onnx.quantization.autotune.common import Region

from modelopt.onnx.op_types import (
    get_aggregation_ops,
    get_bitwise_ops,
    get_bool_ops,
    get_comparison_ops,
    get_conditional_ops,
    get_copy_ops,
    get_set_ops,
    get_value_check_ops,
    is_fusible_reduction_op,
)
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices


class InsertionPoint(ABC):
    """Abstract base class for pattern-relative Q/DQ insertion points."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsertionPoint":
        """Create from dictionary."""
        ...

    @abstractmethod
    def resolve(self, region: "Region", graph: gs.Graph) -> set["ResolvedInsertionPoint"]:
        """Resolve pattern-relative insertion point to actual tensor names."""
        ...

    @staticmethod
    @abstractmethod
    def collect_from_region(region: "Region", graph: gs.Graph) -> list["InsertionPoint"]:
        """Collect all valid insertion points of this type from a region."""
        ...


@dataclass(frozen=True)
class ResolvedInsertionPoint:
    """Resolved Q/DQ insertion point with actual tensor name and optional node context.

    After resolving pattern-relative insertion points, this class represents the
    actual location where Q/DQ pairs should be inserted in the graph. It contains the
    tensor name and the node index (if applicable) and input index (if applicable).

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    tensor_name: str
    node_index: int | None = None  # Absolute graph node index (or None for tensor-level insertion)
    input_index: int | None = None  # Input tensor index of that node (or None)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedInsertionPoint":
        """Create from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class NodeInputInsertionPoint(InsertionPoint):
    """Pattern-relative Q/DQ insertion point at a node's input (frozen/hashable).

    Specifies where to insert a Q/DQ pair within a region pattern using
    pattern-relative indices rather than absolute node IDs. This enables
    insertion scheme reuse across all regions matching the same pattern.

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    node_index: int  # Pattern-relative node index
    input_index: int  # Input tensor index of that node

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeInputInsertionPoint":
        """Create from dictionary."""
        return cls(node_index=data["node_index"], input_index=data["input_index"])

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a node input insertion point to actual tensor names for a matching region."""
        node_indices = region.get_nodes(sort=True)
        assert self.node_index < len(node_indices), "Node index out of range"
        actual_node_idx = node_indices[self.node_index]
        node = graph.nodes[actual_node_idx]
        assert self.input_index < len(node.inputs), "Input index out of range"

        resolved_ips = set()
        # Determine which input indices to resolve (include weights for Conv/ConvTranspose)
        input_indices = [self.input_index]
        if node.op in ["Conv", "ConvTranspose"]:
            assert self.input_index == 0, (
                "Conv/ConvTranspose inputs and weights must be quantized together"
            )
            assert len(node.inputs) >= 2, "Conv/ConvTranspose should have at least 2 inputs"
            input_indices.append(1)

        for idx in input_indices:
            inp = node.inputs[idx]
            if hasattr(inp, "name") and inp.name:
                resolved_ips.add(
                    ResolvedInsertionPoint(
                        tensor_name=inp.name, node_index=actual_node_idx, input_index=idx
                    )
                )
        return resolved_ips

    @staticmethod
    def collect_from_region(region: "Region", graph: gs.Graph) -> list["NodeInputInsertionPoint"]:
        """Collect all valid node input insertion points from a region."""
        node_indices = region.get_nodes(sort=True)
        insertion_points = []
        for local_idx, node_idx in enumerate(node_indices):
            node = graph.nodes[node_idx]
            for input_idx, inp in enumerate(node.inputs):
                name = getattr(inp, "name", None)
                if not name or skip_invalid_insertion_points(graph, name, node):
                    continue
                insertion_points.append(
                    NodeInputInsertionPoint(node_index=local_idx, input_index=input_idx)
                )
        return insertion_points


@dataclass(frozen=True)
class ChildRegionInputInsertionPoint(InsertionPoint):
    """Pattern-relative Q/DQ insertion point at a child region's input boundary (frozen/hashable).

    Specifies where to insert Q/DQ pairs at the input boundaries of child regions
    within COMPOSITE regions. This allows parent regions to control quantization
    at child boundaries, potentially overriding or complementing child region
    optimizations.

    Only applies to COMPOSITE regions; LEAF regions have no children.

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    # Pattern-relative child region index
    region_index: int
    # Input tensor index of that child region
    input_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChildRegionInputInsertionPoint":
        """Create from dictionary."""
        return cls(**data)

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a child region input insertion point to actual tensor names."""
        from modelopt.onnx.quantization.autotune.common import RegionType

        if region.type == RegionType.LEAF:
            return set()

        children_regions = region.get_children(sort=True)
        assert self.region_index < len(children_regions), "Child region index out of range"
        child_region = children_regions[self.region_index]
        assert self.input_index < len(child_region.inputs), "Input index out of range"
        tensor_name = child_region.inputs[self.input_index]
        return resolve_region_io_insertion_points(child_region, graph, tensor_name)

    @staticmethod
    def collect_from_region(
        region: "Region", graph: gs.Graph
    ) -> list["ChildRegionInputInsertionPoint"]:
        """Collect all valid child region input insertion points from a region."""
        from modelopt.onnx.quantization.autotune.common import RegionType

        if region.type == RegionType.LEAF:
            return []

        insertion_points = []
        for local_idx, child_region in enumerate(region.get_children(sort=True)):
            for input_idx, inp in enumerate(child_region.inputs):
                if skip_invalid_insertion_points(graph, inp, child_region):
                    continue
                insertion_points.append(
                    ChildRegionInputInsertionPoint(region_index=local_idx, input_index=input_idx)
                )
        return insertion_points


@dataclass(frozen=True)
class ChildRegionOutputInsertionPoint(InsertionPoint):
    """Pattern-relative Q/DQ insertion point at a child region or node output (frozen/hashable).

    Specifies where to insert Q/DQ pairs at output boundaries. This can be either:
    1. Output from a child region (in COMPOSITE regions)
    2. Output from a node within the region

    This class is immutable (frozen) to allow safe use in sets and as dict keys.
    """

    region_index: int | None  # Pattern-relative child region index (or None)
    node_index: int | None  # Pattern-relative node index (or None)
    output_index: int  # Output tensor index

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChildRegionOutputInsertionPoint":
        """Create from dictionary."""
        return cls(**data)

    def resolve(self, region: "Region", graph: gs.Graph) -> set[ResolvedInsertionPoint]:
        """Resolve a region output insertion point to actual tensor names."""
        if self.region_index is not None:
            children_regions = region.get_children(sort=True)
            assert self.region_index < len(children_regions), "Region index out of range"
            child_region = children_regions[self.region_index]
            assert self.output_index < len(child_region.outputs), "Output index out of range"
            tensor_name = child_region.outputs[self.output_index]
            return resolve_region_io_insertion_points(child_region, graph, tensor_name)

        if self.node_index is not None:
            node_indices = region.get_nodes(sort=True)
            assert self.node_index < len(node_indices), "Node index out of range"
            node = graph.nodes[node_indices[self.node_index]]
            assert self.output_index < len(node.outputs), "Output index out of range"
            tensor = node.outputs[self.output_index]
            assert hasattr(tensor, "name") and tensor.name, "Tensor name is required"
            return resolve_region_io_insertion_points(None, graph, tensor.name)

        return set()

    @staticmethod
    def collect_from_region(
        region: "Region", graph: gs.Graph
    ) -> list["ChildRegionOutputInsertionPoint"]:
        """Collect all valid region output insertion points from a region."""
        from modelopt.onnx.quantization.autotune.common import RegionType

        region_outputs_set = set(region.outputs)
        insertion_points = []

        # For COMPOSITE regions: collect child region outputs
        if region.type != RegionType.LEAF:
            for local_idx, child_region in enumerate(region.get_children(sort=True)):
                for output_idx, out in enumerate(child_region.outputs):
                    if out in region_outputs_set and not skip_invalid_insertion_points(
                        graph, out, child_region
                    ):
                        insertion_points.append(
                            ChildRegionOutputInsertionPoint(
                                region_index=local_idx, node_index=None, output_index=output_idx
                            )
                        )

        # For all regions: collect node outputs
        for local_idx, node_idx in enumerate(region.get_nodes(sort=True)):
            node = graph.nodes[node_idx]
            for output_idx, out in enumerate(node.outputs):
                if not (hasattr(out, "name") and out.name):
                    continue
                if out.name in region_outputs_set and not skip_invalid_insertion_points(
                    graph, out.name, node
                ):
                    insertion_points.append(
                        ChildRegionOutputInsertionPoint(
                            region_index=None, node_index=local_idx, output_index=output_idx
                        )
                    )

        return insertion_points


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
        node_indices = region_or_node.get_region_nodes_and_descendants()
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
                # Conv -> ReLU/Softmax or Conv -> BatchNormalization -> ReLU/Softmax
                if node.op in ["Relu", "Softmax"]:
                    if len(node.inputs) == 1 and len(node.inputs[0].inputs) == 1:
                        producer = node.inputs[0].inputs[0]
                        if producer.op in ["Conv", "ConvTranspose"]:
                            return True
                        if (
                            producer.op == "BatchNormalization"
                            and len(producer.inputs[0].inputs) == 1
                            and producer.inputs[0].inputs[0].op in ["Conv", "ConvTranspose"]
                        ):
                            return True
                # Conv -> BatchNormalization
                if node.op == "BatchNormalization":
                    assert len(node.inputs) >= 1, "BN node should have more than one inputs"
                    if len(node.inputs[0].inputs) == 1:
                        producer = node.inputs[0].inputs[0]
                        if producer.op in ["Conv", "ConvTranspose"]:
                            return True
                # Filter 1: out boolean operations
                if node.op in (
                    get_bool_ops()
                    | get_bitwise_ops()
                    | get_value_check_ops()
                    | get_comparison_ops()
                    | get_conditional_ops()
                    | get_aggregation_ops()
                    | get_set_ops()
                ) or is_fusible_reduction_op(node.op):
                    return True
                # Filter 2: out shape operations
                if node.op in get_autotuner_skip_ops():
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
    """Check if a region contains major quantizable operations (only checks LEAF regions).

    Args:
        region: The region to check
        graph: The ONNX graph containing the nodes

    Returns:
        True if the region contains major quantizable operations, False otherwise
    """
    from modelopt.onnx.quantization.autotune.common import RegionType

    if region.type != RegionType.LEAF:
        return True
    region_ops = {graph.nodes[idx].op for idx in region.get_nodes()}
    return bool(region_ops & get_autotuner_quantizable_ops())


def resolve_region_io_insertion_points(
    region: "Region | None", graph: gs.Graph, tensor_name: str
) -> set[ResolvedInsertionPoint]:
    """Resolve region input/output boundaries to actual Q/DQ insertion points.

    For a given tensor at a region boundary (input or output), this function
    identifies all the actual node inputs where Q/DQ pairs should be inserted.
    It considers both nodes within the region (if provided) and all users of
    the tensor in the graph.

    Args:
        region: The region to search within (or None to search entire graph)
        graph: The ONNX graph containing the nodes
        tensor_name: Name of the tensor at the region boundary

    Returns:
        Set of ResolvedInsertionPoint objects specifying where to insert Q/DQ pairs
    """
    tensor_users_map = getattr(graph, "tensor_users_map", None) or get_tensor_consumer_node_indices(
        graph
    )

    node_indices: set[int] = set()
    if region is not None:
        node_indices.update(region.get_region_nodes_and_descendants())
    node_indices.update(tensor_users_map.get(tensor_name, []))

    resolved = set()
    for node_idx in node_indices:
        node = graph.nodes[node_idx]
        for input_idx, inp in enumerate(node.inputs):
            if hasattr(inp, "name") and inp.name == tensor_name:
                if not skip_invalid_insertion_points(graph, tensor_name, node):
                    resolved.add(
                        ResolvedInsertionPoint(
                            tensor_name=tensor_name, node_index=node_idx, input_index=input_idx
                        )
                    )
    return resolved


def merge_resolved_insertion_points(
    graph: gs.Graph, resolved_insertion_points: set[ResolvedInsertionPoint]
) -> set[ResolvedInsertionPoint]:
    """Optimize insertion points by merging node-specific insertions into tensor-level insertions.

    When all consumers (users) of a tensor have Q/DQ insertion points, it's more efficient
    to insert Q/DQ once at the tensor level rather than at each individual node input.
    This reduces the number of Q/DQ nodes in the graph and simplifies the quantization scheme.

    Args:
        graph: The ONNX graph containing the nodes
        resolved_insertion_points: Set of resolved insertion points to optimize

    Returns:
        Optimized set of insertion points with merged tensor-level insertions where possible
    """
    tensor_users_map = get_tensor_consumer_node_indices(graph)
    node_ips = {ip for ip in resolved_insertion_points if ip.node_index is not None}

    results = resolved_insertion_points - node_ips
    for tensor_name in {ip.tensor_name for ip in node_ips}:
        all_users = set(tensor_users_map.get(tensor_name, []))
        qdq_users = {ip for ip in node_ips if ip.tensor_name == tensor_name}
        if all_users == {ip.node_index for ip in qdq_users}:
            results.add(
                ResolvedInsertionPoint(tensor_name=tensor_name, node_index=None, input_index=None)
            )
        else:
            results.update(qdq_users)
    return results


def get_autotuner_skip_ops():
    """Returns set of shape/structural operations that are not quantizable."""
    return set(get_copy_ops()) | {
        # Additional indexing/scatter/reshape ops
        "Compress",
        "Scatter",
        "ExpandDims",
        "Unsqueeze",
        "View",
        "Pad",
        # Utility ops
        "Cast",
        "Ceil",
        "Clip",
        "Identity",
        "Range",
        "Shape",
    }


def get_autotuner_quantizable_ops():
    """Returns set of key operations that benefit from quantization."""
    return {
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
