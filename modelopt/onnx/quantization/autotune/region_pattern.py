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

"""Region pattern signature generator for grouping structurally similar regions."""

import hashlib
from typing import Union, overload

import onnx_graphsurgeon as gs

from modelopt.onnx.op_types import get_symmetric_ops
from modelopt.onnx.quantization.autotune.common import InsertionScheme, Region
from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    ChildRegionOutputInsertionPoint,
    NodeInputInsertionPoint,
    ResolvedInsertionPoint,
)


class RegionPattern:
    """Represents a structural pattern of a region.

    The pattern captures the topology and operation types in a region,
    enabling pattern matching and region comparison. Patterns are hashable
    and can be used as dictionary keys for efficient grouping and lookup.
    """

    def __init__(self, signature: str, size: int):
        """Initialize a region pattern.

        Args:
            signature: The structural signature of the region.
            size: The number of nodes in the region.
        """
        self.signature = signature
        self.size = size

    @property
    def is_empty(self) -> bool:
        """Check if the pattern represents an empty region."""
        return self.size == 0

    @property
    def is_composite(self) -> bool:
        """Check if the pattern represents a composite region."""
        return self.signature.startswith("COMPOSITE(")

    @property
    def is_leaf(self) -> bool:
        """Check if the pattern represents a leaf region (no composite structure)."""
        return not self.is_composite and not self.is_empty

    def __str__(self) -> str:
        """String representation of the pattern."""
        return self.signature

    def __repr__(self) -> str:
        """Developer-friendly representation with signature and size."""
        return f"RegionPattern('{self.signature}', size={self.size})"

    def __eq__(self, other) -> bool:
        """Check equality based on signature only."""
        if not isinstance(other, RegionPattern):
            return False
        return self.signature == other.signature

    def __hash__(self) -> int:
        """Hash based on signature for use as dict key."""
        return hash(self.signature)

    def get_hash(self) -> str:
        """Get a 128-bit cryptographic hash of the pattern signature."""
        return hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:32]

    def get_short_signature(self, max_length: int = 80) -> str:
        """Get a truncated version of the signature for display purposes."""
        if len(self.signature) <= max_length or max_length > len(self.signature):
            return self.signature
        return self.signature[: max_length - 3] + "..."

    @classmethod
    def from_region(cls, region: Region, graph: gs.Graph) -> "RegionPattern":
        """Compute a structural pattern for a region.

        The pattern captures:
        - Direct node operations in the region
        - Structure of sub-regions (recursively)
        - Handles symmetric operations consistently
        - Sorts sub-regions by size for determinism

        Args:
            region: The region to compute pattern for
            graph: The ONNX graph containing the nodes

        Returns:
            RegionPattern object containing the signature and metadata
        """
        signature_str = cls._compute_signature_recursive(region, graph)
        total_size = len(region.get_region_nodes_and_descendants())
        return cls(signature_str, total_size)

    @overload
    def matches(self, other: "RegionPattern") -> bool: ...
    @overload
    def matches(self, other: Region, graph: gs.Graph, scheme: None = None) -> list[int] | None: ...
    @overload
    def matches(
        self, other: Region, graph: gs.Graph, scheme: InsertionScheme
    ) -> set[ResolvedInsertionPoint]: ...

    def matches(
        self,
        other: Union["RegionPattern", Region],
        graph: gs.Graph | None = None,
        scheme: InsertionScheme | None = None,
    ) -> bool | list[int] | set[ResolvedInsertionPoint] | None:
        """Check if this pattern matches another pattern or region.

        This method provides three distinct behaviors depending on the arguments:

        1. **Pattern-to-pattern comparison** (other is RegionPattern, scheme is None):
           Returns bool indicating structural equivalence.

        2. **Pattern-to-region matching** (other is Region, scheme is None):
           Returns list of node IDs in pattern order if match succeeds, None otherwise.

        3. **Pattern-to-region with insertion scheme** (other is Region, scheme provided):
           Returns set of resolved insertion points where Q/DQ should be inserted, considering:
           - NodeInputInsertionPoints from the scheme (node-level Q/DQ)
           - ChildRegionInputInsertionPoints from the scheme (child region input Q/DQ)
           - RegionOutputInsertionPoints from the scheme (region output Q/DQ)
           Returns empty set if pattern doesn't match.

        Args:
            other: Either a RegionPattern or Region to compare with
            graph: Required when other is a Region (for computing its pattern)
            scheme: Optional InsertionScheme containing node_inputs,
                   child_region_inputs, and region_outputs
                   to resolve to tensor names

        Returns:
            - True if other is RegionPattern and patterns match
            - List of node IDs in pattern order if other is Region and scheme is None, None if no match
            - Set of resolved insertion points for Q/DQ insertion if other is Region and scheme is provided

        Raises:
            ValueError: If other is Region but graph is not provided, or if scheme
                       is provided but other is not a Region
            TypeError: If other is neither RegionPattern nor Region
        """
        if isinstance(other, RegionPattern):
            if scheme is not None:
                raise ValueError("scheme parameter can only be used when matching against a Region")
            return self._matches_pattern(other)
        elif isinstance(other, Region) and scheme is None:
            return self._matches_region(other, graph)
        elif isinstance(other, Region) and scheme is not None:
            if graph is None:
                raise ValueError("graph parameter is required")

            region_pattern = RegionPattern.from_region(other, graph)
            if self != region_pattern:
                return set()

            resolved_ips = set()
            for ip in scheme.node_inputs:
                resolved_ips.update(ip.resolve(other, graph))
            for ip in scheme.child_region_inputs:
                resolved_ips.update(ip.resolve(other, graph))
            for ip in scheme.region_outputs:
                resolved_ips.update(ip.resolve(other, graph))
            return resolved_ips
        else:
            raise TypeError(f"Expected RegionPattern or Region, got {type(other).__name__}")

    def _matches_pattern(self, other: "RegionPattern") -> bool:
        """Internal function: Match this pattern against another pattern.

        Args:
            other: Another RegionPattern to compare with

        Returns:
            True if patterns are structurally equivalent, False otherwise
        """
        return self == other

    def _matches_region(self, region: Region, graph: gs.Graph | None) -> list[int] | None:
        """Internal function: Match this pattern against a region.

        Args:
            region: The region to match against
            graph: The ONNX graph containing the nodes

        Returns:
            List of node IDs in match order if pattern matches, None otherwise.
            Match order follows the pattern computation order:
            - Direct nodes of the region (sorted)
            - Then recursively, nodes from child regions (in child sort order)

        Raises:
            ValueError: If graph is not provided
        """
        if graph is None:
            raise ValueError("graph parameter is required when matching against a Region")

        region_pattern = RegionPattern.from_region(region, graph)

        if self == region_pattern:
            return self._collect_nodes_in_match_order(region)
        else:
            return None

    def get_full_insertion_scheme(self, region: Region, graph: gs.Graph) -> InsertionScheme:
        """Collect all possible insertion points for quantization in a region.

        This method gathers all locations where Q/DQ  nodes could be inserted within a region's
        computational graph. These insertion points are organized into three categories:
        - node_inputs: Inputs to individual nodes within the region
        - child_region_inputs: Inputs to child regions within composite regions
        - region_outputs: Outputs from the region or its child regions

        Args:
            region: The region to collect insertion points for
            graph: The ONNX graph containing the nodes

        Returns:
            InsertionScheme object containing the insertion points
        """
        region_pattern = RegionPattern.from_region(region, graph)

        if self != region_pattern:
            raise ValueError("Region pattern mismatch")

        scheme = InsertionScheme()
        scheme.node_inputs = NodeInputInsertionPoint.collect_from_region(region, graph)
        scheme.child_region_inputs = ChildRegionInputInsertionPoint.collect_from_region(
            region, graph
        )
        scheme.region_outputs = ChildRegionOutputInsertionPoint.collect_from_region(region, graph)

        return scheme

    def format_tree(self, region: Region, graph: gs.Graph, indent: int = 0) -> str:
        """Format this pattern and region as a human-readable tree.

        Useful for debugging and visualization.

        Args:
            region: The region associated with this pattern
            graph: The ONNX graph
            indent: Indentation level

        Returns:
            Formatted string representation
        """
        prefix = "  " * indent
        result = f"{prefix}Region {region.id}: {self.signature} (size={self.size})\n"

        for child in region.get_children():
            child_pattern = RegionPattern.from_region(child, graph)
            result += child_pattern.format_tree(child, graph, indent + 1)

        return result

    @staticmethod
    def _collect_nodes_in_match_order(region: Region) -> list[int]:
        """Collect node IDs in the same order as signature computation.

        This follows the traversal order used by _compute_signature_recursive:
        1. Direct nodes of the region (sorted by node index)
        2. Recursively, nodes from child regions (children sorted by -level, then size)

        The child sorting order MUST match _compute_signature_recursive and
        insertion_points.py for correct pattern-relative index alignment.

        Args:
            region: The region to collect nodes from

        Returns:
            List of node IDs in match order
        """
        node_ids = []

        node_ids.extend(region.get_nodes(sort=True))
        sorted_children = region.get_children(sort=True)

        for child in sorted_children:
            node_ids.extend(RegionPattern._collect_nodes_in_match_order(child))

        return node_ids

    @staticmethod
    def _compute_signature_recursive(region: Region, graph: gs.Graph) -> str:
        """Recursively compute structural signature for a region.

        The signature captures:
        - Node operations and their key parameters (for LEAF regions)
        - Hierarchical structure with child patterns (for COMPOSITE regions)
        - Deterministic ordering (sorted nodes and children)
        - Normalized handling of symmetric/commutative operations

        Signature formats:
        - Empty region: "EMPTY"
        - Leaf region: "Op1->Op2->Op3" or "Op1[params]->Op2[params]"
        - Composite with nodes: "COMPOSITE(nodes|child1+child2)"
        - Composite without nodes: "COMPOSITE(child1+child2)"

        Child Sorting:
        - Children are sorted by (-level, size) for deterministic signatures
        - This order MUST match insertion_points.py for correct pattern-relative indexing
        - Higher-level (more abstract) children come first
        - Within same level, smaller children come first

        Args:
            region: The region to process
            graph: The ONNX graph containing the nodes

        Returns:
            Deterministic signature string representing the region structure
        """
        nodes_list = list(graph.nodes)
        node_indices_set = set(region.get_nodes())

        if node_indices_set and max(node_indices_set) >= len(nodes_list):
            raise ValueError("Region contains node indices outside the graph")

        node_ops = [
            RegionPattern._make_node_with_params_signature(nodes_list[idx], graph, node_indices_set)
            for idx in sorted(node_indices_set)
        ]

        sorted_children = region.get_children(sort=True)

        if not sorted_children:
            return "->".join(node_ops) if node_ops else "EMPTY"

        child_sigs = "+".join(
            [RegionPattern._compute_signature_recursive(child, graph) for child in sorted_children]
        )

        if node_ops:
            node_sig = "->".join(node_ops)
            return f"COMPOSITE({node_sig}|{child_sigs})"
        return f"COMPOSITE({child_sigs})"

    @staticmethod
    def _get_symmetric_input_signature(
        node: gs.Node, graph: gs.Graph, region_node_indices: set
    ) -> str | None:
        """Compute normalized input source signature for symmetric operations."""
        if node.op not in get_symmetric_ops() or len(node.inputs) <= 1:
            return None

        nodes_list = list(graph.nodes)
        node_to_idx = {id(n): idx for idx, n in enumerate(nodes_list)}

        input_sources = []
        for inp in node.inputs:
            if inp is None or not hasattr(inp, "inputs") or not inp.inputs:
                input_sources.append(("external", "input-or-constant"))
            else:
                producer_node = inp.inputs[0] if inp.inputs else None
                if producer_node and id(producer_node) in node_to_idx:
                    producer_idx = node_to_idx[id(producer_node)]
                    location = "internal" if producer_idx in region_node_indices else "external"
                    input_sources.append((location, producer_node.op))
                else:
                    input_sources.append(("external", "unknown"))

        sorted_sources = sorted(input_sources)
        return ",".join(f"{loc}:{op}" for loc, op in sorted_sources)

    @staticmethod
    def _format_attr_value(value: object) -> str:
        """Format an attribute value for inclusion in a signature."""
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and all(isinstance(v, (int, float)) for v in value):
                if all(isinstance(v, int) for v in value):
                    return "x".join(str(v) for v in value)
                return "x".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in value)
            return ",".join(str(v) for v in value)
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, bytes):
            hex_str = value.hex()
            return hex_str if len(hex_str) <= 16 else f"{hex_str[:16]}..."
        return str(value)

    @staticmethod
    def _make_node_with_params_signature(
        node: gs.Node, graph: gs.Graph, region_node_indices: set
    ) -> str:
        """Create signature for a single node including its parameters.

        Includes operation type and key attributes that affect behavior.
        For symmetric/commutative operations (Add, Mul, etc.), normalizes
        input order to ensure consistent signatures regardless of operand order.
        Ensures deterministic ordering by sorting attributes by key name.

        Args:
            node: The ONNX node
            graph: The ONNX graph containing all nodes
            region_node_indices: Set of node indices in the current region

        Returns:
            Signature string examples:
            - "Relu" - Simple operation without attributes
            - "Conv[dilations=1x1,kernel_shape=3x3]" - Operation with attributes
            - "Add<external:Conv,internal:Mul>" - Symmetric op with sorted input sources
            - "Mul[axis=1]<external:unknown,internal:Add>" - Symmetric op with both
        """
        op = node.op
        sym_sig = RegionPattern._get_symmetric_input_signature(node, graph, region_node_indices)

        attr_sig = ""
        if node.attrs:
            attr_parts = [
                f"{key}={RegionPattern._format_attr_value(node.attrs[key])}"
                for key in sorted(node.attrs.keys())
            ]
            attr_sig = f"[{','.join(attr_parts)}]"

        if attr_sig and sym_sig:
            return f"{op}{attr_sig}<{sym_sig}>"
        if sym_sig:
            return f"{op}<{sym_sig}>"
        if attr_sig:
            return f"{op}{attr_sig}"
        return op
