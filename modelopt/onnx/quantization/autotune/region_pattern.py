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

"""Region Pattern Signature Generator.

Provides structural pattern analysis for regions in ONNX computation graphs.
This module enables:
- Pattern-based region grouping by structural similarity
- Deterministic signature generation for pattern matching
- Resolution of insertion points to actual tensor names
- Support for both node-level and region-level Q/DQ insertion

Key concepts:
- NodeInputInsertionPoint: Specifies Q/DQ insertion at a node's input
- ChildRegionInputInsertionPoint: Specifies Q/DQ insertion at a child region's input boundary
- RegionOutputInsertionPoint: Specifies Q/DQ insertion at a region output (child or node)
- Pattern matching: Groups regions with identical structure for shared optimization
"""

import hashlib
import logging
from typing import Union

import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.common import InsertionScheme, Region
from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
    ResolvedInsertionPoint,
)

# Module logger
logger = logging.getLogger(__name__)

# Commutative/symmetric operations where operand order doesn't matter
SYMMETRIC_OPERATIONS = {
    "Add",
    "Mul",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Max",
    "Min",
    "Sum",
    "Mean",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
}


class RegionPattern:
    """Represents a structural pattern of a region.

    The pattern captures the topology and operation types in a region,
    enabling pattern matching and region comparison. Patterns are hashable
    and can be used as dictionary keys for efficient grouping and lookup.

    Two RegionPattern objects are considered equal if they have the same
    signature string, regardless of their size (which represents instance-specific
    node count).

    Attributes:
        signature: The unique signature string identifying the pattern
        size: Total node count for this pattern instance
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, signature: str, size: int):
        """Initialize a region pattern.

        Args:
            signature: The signature string representing the pattern structure
            size: Total size (node count) of the region
        """
        self.signature = signature
        self.size = size

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_empty(self) -> bool:
        """Check if pattern represents an empty region."""
        return self.signature == "EMPTY" or self.size == 0

    @property
    def is_composite(self) -> bool:
        """Check if pattern represents a composite region."""
        return self.signature.startswith("COMPOSITE(")

    @property
    def is_leaf(self) -> bool:
        """Check if pattern represents a leaf region (no composite structure)."""
        return not self.is_composite and not self.is_empty

    # =========================================================================
    # Special Methods (Python Protocol)
    # =========================================================================

    def __str__(self) -> str:
        """String representation showing just the signature."""
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

    # =========================================================================
    # Public Query Methods
    # =========================================================================

    def get_hash(self) -> str:
        """Get a 128-bit cryptographic hash of the pattern signature.

        Uses SHA-256 (truncated to 128 bits) to generate a compact, deterministic
        hash for efficient pattern comparison and storage. This hash is more
        compact than the full signature for storage and comparison purposes.

        Returns:
            Hexadecimal string representation of the hash (32 characters)

        Example:
            >>> pattern = RegionPattern.from_region(region, graph)
            >>> hash_val = pattern.get_hash()  # Returns 32 hex characters
            >>> print(f"Pattern hash: {hash_val}")
        """
        # SHA-256 truncated to 128 bits = 32 hex characters
        return hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:32]

    def get_short_signature(self, max_length: int = 80) -> str:
        """Get a truncated version of the signature for display purposes.

        Args:
            max_length: Maximum length of the returned string (default: 80)

        Returns:
            Truncated signature with '...' suffix if needed
        """
        if len(self.signature) <= max_length:
            return self.signature
        return self.signature[: max_length - 3] + "..."

    # =========================================================================
    # Public Pattern Matching and Construction
    # =========================================================================

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
        total_size = region.get_total_size()
        return cls(signature_str, total_size)

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
            - bool: If other is RegionPattern, True if patterns match
            - List[int]: If other is Region and scheme is None, list of node IDs
              in pattern order (None if no match)
            - Set[ResolvedInsertionPoint]: If other is Region and scheme is provided,
              set of resolved insertion points for Q/DQ insertion (empty set if no match)

        Raises:
            ValueError: If other is Region but graph is not provided, or if scheme
                       is provided but other is not a Region
            TypeError: If other is neither RegionPattern nor Region
        """
        if isinstance(other, RegionPattern):
            # Behavior 1: Pattern-to-pattern comparison
            if scheme is not None:
                raise ValueError("scheme parameter can only be used when matching against a Region")
            return self._matches_pattern(other)
        elif isinstance(other, Region) and scheme is None:
            # Behavior 2: Pattern-to-region matching (returns node IDs)
            return self._matches_region(other, graph)
        elif isinstance(other, Region) and scheme is not None:
            if graph is None:
                raise ValueError("graph parameter is required")
            # Verify the region matches this pattern
            region_pattern = RegionPattern.from_region(other, graph)
            if self != region_pattern:
                return set()

            resolved_ips = set()
            # Resolve NodeInputInsertionPoints to tensor names
            for ip in scheme.node_inputs:
                resolved_ips.update(ip.resolve(other, graph))
            # Resolve ChildRegionInputInsertionPoints to tensor names
            for ip in scheme.child_region_inputs:
                resolved_ips.update(ip.resolve(other, graph))
            # Resolve RegionOutputInsertionPoints to tensor names
            for ip in scheme.region_outputs:
                resolved_ips.update(ip.resolve(other, graph))
            return resolved_ips
        else:
            raise TypeError(f"Expected RegionPattern or Region, got {type(other).__name__}")

    # =========================================================================
    # Private Pattern Matching Helpers
    # =========================================================================

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

        # Compute pattern for the region
        region_pattern = RegionPattern.from_region(region, graph)

        # Check if patterns match
        if self == region_pattern:
            # Return node IDs in match order (same as signature computation order)
            return self._collect_nodes_in_match_order(region)
        else:
            return None

    def get_full_insertion_scheme(self, region: Region, graph: gs.Graph) -> InsertionScheme:
        """Get all possible insertion points for a region in a single InsertionScheme.

        This method first verifies that the region matches this pattern (raises if not).
        It then collects all three types of insertion points:
        1. Node input insertion points (Q/DQ at node inputs within the region)
        2. Child region input insertion points (Q/DQ at child region input boundaries)
        3. Region output insertion points (Q/DQ at region output boundaries)

        The returned InsertionScheme contains all possible Q/DQ insertion
        locations for this region pattern. This can be used as:
        - A baseline scheme with all possible insertions
        - A starting point for optimization algorithms
        - A comprehensive view of all insertion opportunities

        Important: Pattern-relative indices in the returned scheme are based on
        sorted child/node ordering. The sorting order (-level, size) MUST match
        insertion_points.py for correct resolution.

        Note: The returned scheme has no child region schemes specified,
        latency is set to infinity (unmeasured), and error flag is False.

        Args:
            region: The region to analyze
            graph: The ONNX graph containing the nodes

        Returns:
            InsertionScheme containing all possible insertion points for this region

        Raises:
            AssertionError: If the region doesn't match this pattern
        """
        # Verify that the region matches this pattern
        region_pattern = RegionPattern.from_region(region, graph)
        assert self == region_pattern, "Region pattern mismatch"

        scheme = InsertionScheme()
        # Collect all node input insertion points
        scheme.node_inputs = NodeInputInsertionPoint.collect_from_region(region, graph)
        # Collect all child region input insertion points (at child boundaries)
        scheme.child_region_inputs = ChildRegionInputInsertionPoint.collect_from_region(
            region, graph
        )
        # Collect all region output insertion points
        scheme.region_outputs = RegionOutputInsertionPoint.collect_from_region(region, graph)

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
        result = f"{prefix}Region {region.get_id()}: {self.signature} (size={self.size})\n"

        for child in region.get_children():
            child_pattern = RegionPattern.from_region(child, graph)
            result += child_pattern.format_tree(child, graph, indent + 1)

        return result

    # =========================================================================
    # Static Utility Methods
    # =========================================================================

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

        # Add direct nodes of this region (sorted)
        node_ids.extend(sorted(region.get_nodes()))

        # Get children and sort them the same way as signature computation
        # CRITICAL: This sorting must match _compute_signature_recursive and insertion_points.py
        # Sort by: 1) level (descending - higher level first), 2) size (ascending)
        children = region.get_children()
        sorted_children = sorted(children, key=lambda r: (-r.get_level(), r.get_total_size()))

        # Recursively collect nodes from children in order
        for child in sorted_children:
            node_ids.extend(RegionPattern._collect_nodes_in_match_order(child))

        return node_ids

    # --- Signature Computation ---

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
        # Collect direct node operations in this region
        node_ops = []
        nodes_list = list(graph.nodes)
        node_indices_set = region.get_nodes()

        for node_idx in sorted(node_indices_set):
            if node_idx < len(nodes_list):
                node = nodes_list[node_idx]
                # Include operation type and key parameters
                # Pass region node indices for symmetric operation handling
                node_sig = RegionPattern._make_node_with_params_signature(
                    node, graph, node_indices_set
                )
                node_ops.append(node_sig)

        # Get child regions
        children = region.get_children()

        if not children and not node_ops:
            # Empty region (edge case)
            return "EMPTY"

        if not children:
            # LEAF region - only direct nodes, no hierarchical structure
            return RegionPattern._make_node_signature(node_ops)

        # COMPOSITE region - has hierarchical structure with children
        # Sort children deterministically for consistent signatures
        # CRITICAL: This sorting must match insertion_points.py for pattern-relative index alignment
        # Sort by: 1) level (descending - higher level first), 2) size (ascending)
        sorted_children = sorted(children, key=lambda r: (-r.get_level(), r.get_total_size()))

        # Recursively compute child signatures
        child_signatures = []
        for child in sorted_children:
            child_sig = RegionPattern._compute_signature_recursive(child, graph)
            child_signatures.append(child_sig)

        # Combine node operations and child signatures
        if node_ops:
            # Has both direct nodes and hierarchical children
            node_sig = RegionPattern._make_node_signature(node_ops)
            return f"COMPOSITE({node_sig}|{RegionPattern._join_signatures(child_signatures)})"
        else:
            # Only children, no direct nodes in this region
            return f"COMPOSITE({RegionPattern._join_signatures(child_signatures)})"

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

        # Handle symmetric operations - normalize input order
        if op in SYMMETRIC_OPERATIONS and len(node.inputs) > 1:
            # Get input source information for normalization
            input_sources = []
            nodes_list = list(graph.nodes)

            # Build node index lookup for efficient producer finding
            node_to_idx = {id(n): idx for idx, n in enumerate(nodes_list)}

            for inp in node.inputs:
                if inp is None or not hasattr(inp, "inputs") or not inp.inputs:
                    # Input from graph input or constant
                    input_sources.append(("external", "input-or-constant"))
                else:
                    # Input from another node's output
                    producer_node = inp.inputs[0] if inp.inputs else None
                    if producer_node and id(producer_node) in node_to_idx:
                        producer_idx = node_to_idx[id(producer_node)]
                        # Check if producer is in the same region
                        if producer_idx in region_node_indices:
                            # Use relative position: 'internal' + producer op type
                            input_sources.append(("internal", producer_node.op))
                        else:
                            # Producer outside region
                            input_sources.append(("external", producer_node.op))
                    else:
                        # Unknown producer
                        input_sources.append(("external", "unknown"))

            # Sort input sources for deterministic ordering
            # This ensures Add(A,B) and Add(B,A) have the same signature
            sorted_sources = sorted(input_sources)

            # Create source signature
            source_sig = ",".join(f"{src[0]}:{src[1]}" for src in sorted_sources)

            # If node has no attributes, return op with input signature
            if not node.attrs:
                return f"{op}<{source_sig}>"

            # Otherwise, will add input signature after attributes
            has_symmetric_inputs = True
        else:
            has_symmetric_inputs = False

        # Handle non-symmetric operations or symmetric ops without multiple inputs
        if not node.attrs and not has_symmetric_inputs:
            return op

        # Extract and format key attributes (only if node has attributes)
        if node.attrs:
            # Sort attributes alphabetically for deterministic ordering
            attr_parts = []
            for key in sorted(node.attrs.keys()):
                value = node.attrs[key]

                # Format different attribute types deterministically
                if isinstance(value, (list, tuple)):
                    # Format lists/tuples compactly
                    # Use 'x' separator for numeric arrays (common in ONNX)
                    if len(value) > 0 and all(isinstance(v, (int, float)) for v in value):
                        # Format each element consistently
                        if all(isinstance(v, int) for v in value):
                            value_str = "x".join(str(v) for v in value)
                        else:
                            # Mixed int/float - format floats with limited precision
                            value_str = "x".join(
                                f"{v:.4g}" if isinstance(v, float) else str(v) for v in value
                            )
                    else:
                        # Non-numeric or mixed types - use comma separator
                        value_str = ",".join(str(v) for v in value)
                elif isinstance(value, float):
                    # Format floats with limited precision to avoid floating point noise
                    value_str = f"{value:.4g}"
                elif isinstance(value, bool):
                    # Format booleans as 0/1 for compactness
                    value_str = "1" if value else "0"
                elif isinstance(value, bytes):
                    # Format bytes as hex string (truncated for long values)
                    hex_str = value.hex()
                    value_str = hex_str if len(hex_str) <= 16 else f"{hex_str[:16]}..."
                else:
                    # Default: convert to string
                    value_str = str(value)

                attr_parts.append(f"{key}={value_str}")

            # Build final signature with attributes
            attr_sig = f"[{','.join(attr_parts)}]"

            # Add symmetric input signature if applicable
            if has_symmetric_inputs:
                return f"{op}{attr_sig}<{source_sig}>"
            else:
                return f"{op}{attr_sig}"
        else:
            # No attributes - already handled above for symmetric ops
            return op

    @staticmethod
    def _make_node_signature(ops: list[str]) -> str:
        """Create signature from list of node operations.

        Handles single and multiple operations, including symmetric operations.

        Args:
            ops: List of operation signatures (may include parameters)

        Returns:
            Signature string for the operations
        """
        if not ops:
            return ""

        if len(ops) == 1:
            return ops[0]

        # Multiple operations - create sequential signature
        return "->".join(ops)

    @staticmethod
    def _join_signatures(signatures: list[str]) -> str:
        """Join multiple child signatures.

        Sorts signatures alphabetically to ensure deterministic ordering.
        This is critical for pattern matching and comparison.

        Args:
            signatures: List of child signatures

        Returns:
            Combined signature string with deterministic ordering
        """
        if not signatures:
            return ""

        if len(signatures) == 1:
            return signatures[0]

        # Sort signatures alphabetically for deterministic ordering
        # This ensures that parallel/sibling regions always produce
        # the same combined signature regardless of traversal order
        sorted_sigs = sorted(signatures)
        return "+".join(sorted_sigs)
