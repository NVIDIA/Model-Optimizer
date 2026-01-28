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
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
    ResolvedInsertionPoint,
)


class RegionPattern:
    """Represents a structural pattern of a region."""

    def __init__(self, signature: str, size: int):
        """Initialize a region pattern."""
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
        if len(self.signature) <= max_length:
            return self.signature
        return self.signature[: max_length - 3] + "..."

    @classmethod
    def from_region(cls, region: Region, graph: gs.Graph) -> "RegionPattern":
        """Compute a structural pattern for a region."""
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
        """Internal function: Match this pattern against another pattern."""
        return self == other

    def _matches_region(self, region: Region, graph: gs.Graph | None) -> list[int] | None:
        """Internal function: Match this pattern against a region."""
        if graph is None:
            raise ValueError("graph parameter is required when matching against a Region")

        region_pattern = RegionPattern.from_region(region, graph)

        if self == region_pattern:
            return self._collect_nodes_in_match_order(region)
        else:
            return None

    def get_full_insertion_scheme(self, region: Region, graph: gs.Graph) -> InsertionScheme:
        """Get all possible insertion points for a region in a single InsertionScheme."""
        region_pattern = RegionPattern.from_region(region, graph)
        assert self == region_pattern, "Region pattern mismatch"

        scheme = InsertionScheme()
        scheme.node_inputs = NodeInputInsertionPoint.collect_from_region(region, graph)
        scheme.child_region_inputs = ChildRegionInputInsertionPoint.collect_from_region(
            region, graph
        )
        scheme.region_outputs = RegionOutputInsertionPoint.collect_from_region(region, graph)

        return scheme

    def format_tree(self, region: Region, graph: gs.Graph, indent: int = 0) -> str:
        """Format this pattern and region as a human-readable tree."""
        prefix = "  " * indent
        result = f"{prefix}Region {region.id}: {self.signature} (size={self.size})\n"

        for child in region.get_children():
            child_pattern = RegionPattern.from_region(child, graph)
            result += child_pattern.format_tree(child, graph, indent + 1)

        return result

    @staticmethod
    def _collect_nodes_in_match_order(region: Region) -> list[int]:
        """Collect node IDs in the same order as signature computation."""
        node_ids = []

        node_ids.extend(region.get_nodes(sort=True))
        sorted_children = region.get_children(sort=True)

        for child in sorted_children:
            node_ids.extend(RegionPattern._collect_nodes_in_match_order(child))

        return node_ids

    @staticmethod
    def _compute_signature_recursive(region: Region, graph: gs.Graph) -> str:
        """Recursively compute structural signature for a region."""
        nodes_list = list(graph.nodes)
        node_indices_set = set(region.get_nodes())

        node_ops = [
            RegionPattern._make_node_with_params_signature(nodes_list[idx], graph, node_indices_set)
            for idx in sorted(node_indices_set)
            if idx < len(nodes_list)
        ]

        sorted_children = region.get_children(sort=True)

        if not sorted_children and not node_ops:
            return "EMPTY"

        if not sorted_children:
            return "->".join(node_ops)

        child_sigs = "+".join(
            [RegionPattern._compute_signature_recursive(child, graph) for child in sorted_children]
        )

        if node_ops:
            node_sig = "->".join(node_ops)
            return f"COMPOSITE({node_sig}|{child_sigs})"
        return f"COMPOSITE({'+'.join(child_sigs)})"

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
        """Create signature for a single node including its parameters."""
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
