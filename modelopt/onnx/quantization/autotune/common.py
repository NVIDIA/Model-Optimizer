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

"""Common data structures and types for the QDQ Autotuner.

This module provides the foundational classes used throughout the autotuner:

**Exceptions:**
- Region-related: RegionError
- Autotuner-related: AutotunerError, AutotunerNotInitializedError, InvalidSchemeError

**Region Hierarchy:**
- Region: Hierarchical subgraph representation with parent/child relationships
- RegionType: Enumeration for LEAF, COMPOSITE, and ROOT regions

**Q/DQ Insertion Specifications:**
- InsertionScheme: Collection of insertion points with performance metrics

**Scheme Management:**
- PatternSchemes: Multiple insertion schemes for a pattern (applies to all matching regions)
- PatternCache: Collection of top schemes for multiple patterns, used as autotuning seeds

**Configuration:**
- Config: Autotuning parameters and Q/DQ default values
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
)

# Module logger
logger = logging.getLogger(__name__)


# Region-related Exceptions
class RegionError(Exception):
    """Base exception for region-related errors."""


# Autotuner-related Exceptions
class AutotunerError(Exception):
    """Base exception for autotuner-related errors."""


class AutotunerNotInitializedError(AutotunerError):
    """Exception raised when autotuner is used without initialization."""


class InvalidSchemeError(AutotunerError):
    """Exception raised when an invalid scheme is referenced."""


class RegionType(Enum):
    """Region type enumeration for hierarchical graph structure.

    - LEAF: Atomic region containing direct nodes with no child regions
    - COMPOSITE: Hierarchical region containing child regions (and optionally direct nodes)
    - ROOT: Top-level region encompassing the entire computation graph
    """

    LEAF = "LEAF"
    COMPOSITE = "COMPOSITE"
    ROOT = "ROOT"


class Region:
    """Hierarchical subgraph region in an ONNX computation graph.

    A Region represents a cohesive subgraph with well-defined boundaries, supporting:

    **Hierarchical Structure:**
    - Parent/child relationships forming a multi-level hierarchy
    - LEAF regions contain only direct nodes
    - COMPOSITE regions contain child regions (and optionally direct nodes)
    - ROOT regions encompass the entire graph

    **Node Management:**
    - Direct nodes: Operations directly in this region (not in children)
    - Recursive nodes: All operations including those in descendant regions

    **Boundary Tracking:**
    - Input tensors: Data entering the region from outside
    - Output tensors: Data leaving the region to outside consumers

    **Pattern Matching:**
    - Regions with identical structure share the same pattern signature
    - Pattern-based optimization applies schemes to all matching regions

    Regions are the fundamental unit for Q/DQ insertion and optimization.
    """

    def __init__(self, region_id: int, level: int, region_type: RegionType):
        """Initialize a new region.

        Args:
            region_id: Unique identifier within the region hierarchy
            level: Hierarchical level (0 = leaf, higher = more composite)
            region_type: Type classification (LEAF, COMPOSITE, or ROOT)
        """
        self.id = region_id
        self.level = level
        self.type = region_type
        self.parent: Region | None = None
        self.children: list[Region] = []
        self.nodes: set[int] = set()
        self.inputs: list[str] = []
        self.outputs: list[str] = []

    # =========================================================================
    # Basic Accessors
    # =========================================================================

    def get_id(self) -> int:
        """Get region ID."""
        return self.id

    def set_id(self, region_id: int) -> None:
        """Set region ID (for RegionBuilder use)."""
        self.id = region_id

    def get_level(self) -> int:
        """Get region level in hierarchy."""
        return self.level

    def set_level(self, level: int) -> None:
        """Set region level in hierarchy (for RegionBuilder use)."""
        self.level = level

    def get_type(self) -> RegionType:
        """Get region type."""
        return self.type

    def set_type(self, region_type: RegionType) -> None:
        """Set region type (for RegionBuilder use)."""
        self.type = region_type

    # =========================================================================
    # Hierarchy Management
    # =========================================================================

    def get_parent(self) -> Optional["Region"]:
        """Get parent region."""
        return self.parent

    def set_parent(self, parent: Optional["Region"]) -> None:
        """Set parent region."""
        self.parent = parent

    def get_children(self) -> list["Region"]:
        """Get all child regions."""
        return self.children

    def remove_child(self, child: "Region") -> bool:
        """Remove a child region from this region's children list.

        Args:
            child: The child region to remove

        Returns:
            True if child was found and removed, False otherwise
        """
        child_id = child.get_id()
        initial_count = len(self.children)
        self.children = [c for c in self.children if c.get_id() != child_id]
        removed = len(self.children) < initial_count

        if removed and child.parent and child.parent.get_id() == self.id:
            child.set_parent(None)

        return removed

    def add_child(self, child: "Region") -> None:
        """Add a child sub-region."""
        # Prevent adding self as child
        if child.get_id() == self.id:
            logger.warning(f"Cannot add region {self.id} as its own child")
            return

        # Prevent creating cycles: check if self is already a descendant of child
        if self._is_descendant_of(child):
            logger.warning(
                f"Cycle detected: region {self.id} is already a descendant of region {child.get_id()}"
            )
            return

        # Check if child already has a different parent
        if child.parent is not None and child.parent.get_id() != self.id:
            old_parent_id = child.parent.get_id()
            logger.debug(
                f"Re-parenting region {child.get_id()}: moving from parent {old_parent_id} to {self.id}"
            )
            # Remove from old parent to maintain tree structure
            child.parent.remove_child(child)

        # Check if child is already in children list
        if any(c.get_id() == child.get_id() for c in self.children):
            logger.debug(f"Region {child.get_id()} already child of {self.id}")
            return

        self.children.append(child)
        child.set_parent(self)

    def _is_descendant_of(self, potential_ancestor: "Region") -> bool:
        """Check if this region is a descendant of potential_ancestor."""
        visited = set()
        current = self.parent
        while current:
            if current.get_id() in visited:
                # Already visited, there's a cycle in parents
                return False
            visited.add(current.get_id())
            if current.get_id() == potential_ancestor.get_id():
                return True
            current = current.parent
        return False

    # =========================================================================
    # Node Management
    # =========================================================================

    def add_node(self, node_index: int) -> None:
        """Add a node index to this region."""
        self.nodes.add(node_index)

    def add_nodes(self, node_indices: list[int]) -> None:
        """Add multiple node indices to this region."""
        self.nodes.update(node_indices)

    def get_nodes(self) -> set[int]:
        """Get direct node indices in this region only.

        Returns only nodes directly owned by this region, excluding nodes
        in child regions. Use get_all_nodes_recursive() for complete coverage.

        Returns:
            Set of node indices (absolute positions in the graph)
        """
        return self.nodes

    def get_all_nodes_recursive(self, _visited: set[int] | None = None) -> set[int]:
        """Get all node indices recursively, including descendants.

        Traverses the entire subtree rooted at this region, collecting nodes
        from this region and all child regions recursively.

        Args:
            _visited: Internal parameter for cycle detection (do not use)

        Returns:
            Set of all node indices in this region and its descendants
        """
        if _visited is None:
            _visited = set()

        # Detect cycles
        if self.id in _visited:
            logger.warning(f"Cycle detected in region {self.id} during node traversal")
            return set()

        _visited.add(self.id)
        all_nodes = set(self.nodes)
        for child in self.children:
            all_nodes.update(child.get_all_nodes_recursive(_visited))
        return all_nodes

    def contains_node(self, node_index: int) -> bool:
        """Check if region contains a specific node (direct only)."""
        return node_index in self.nodes

    def contains_node_recursive(self, node_index: int, _visited: set[int] | None = None) -> bool:
        """Check if region contains a node recursively."""
        if _visited is None:
            _visited = set()

        # Detect cycles
        if self.id in _visited:
            return False

        _visited.add(self.id)

        if self.contains_node(node_index):
            return True
        return any(child.contains_node_recursive(node_index, _visited) for child in self.children)

    # =========================================================================
    # Input/Output Management
    # =========================================================================

    def add_input(self, tensor_name: str) -> None:
        """Add an input tensor name."""
        if tensor_name not in self.inputs:
            self.inputs.append(tensor_name)

    def add_output(self, tensor_name: str) -> None:
        """Add an output tensor name."""
        if tensor_name not in self.outputs:
            self.outputs.append(tensor_name)

    def get_inputs(self) -> list[str]:
        """Get region input tensors."""
        return self.inputs

    def get_outputs(self) -> list[str]:
        """Get region output tensors."""
        return self.outputs

    # =========================================================================
    # Size and Query Methods
    # =========================================================================

    def get_size(self) -> int:
        """Get the number of direct nodes in this region.

        Returns:
            Count of nodes directly in this region (excludes child regions)
        """
        return len(self.nodes)

    def get_total_size(self, _visited: set[int] | None = None) -> int:
        """Get total node count recursively including all descendants.

        Computes the sum of nodes in this region and all child regions,
        providing the total footprint of the region subtree.

        Args:
            _visited: Internal parameter for cycle detection (do not use)

        Returns:
            Total number of nodes in this region and all descendants
        """
        if _visited is None:
            _visited = set()

        # Detect cycles
        if self.id in _visited:
            logger.warning(f"Cycle detected in region {self.id} during size calculation")
            return len(self.nodes)

        _visited.add(self.id)
        total = len(self.nodes)
        for child in self.children:
            total += child.get_total_size(_visited)
        return total

    # =========================================================================
    # Region Operations
    # =========================================================================

    def merge(self, other: "Region") -> None:
        """Merge another region into this one.

        Combines the nodes and children from the other region into this region.
        The other region's children become children of this region, updating
        their parent references accordingly.

        Args:
            other: Region to merge into this one
        """
        if not other:
            return
        # Merge direct nodes
        self.nodes.update(other.nodes)
        # Merge children (updates their parent references)
        for child in other.children:
            self.add_child(child)

    # =========================================================================
    # String Representation
    # =========================================================================

    def to_string(self) -> str:
        """Print region information for debugging."""
        type_str = self.type.value
        return (
            f"Region[id={self.id}, level={self.level}, type={type_str}, "
            f"nodes={len(self.nodes)}, children={len(self.children)}, "
            f"inputs={len(self.inputs)}, outputs={len(self.outputs)}]"
        )

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def compute_structural_signature(self, graph: gs.Graph) -> str:
        """Compute deterministic structural signature for pattern matching.

        Creates a signature that uniquely identifies the region's topology,
        node operations, and hierarchical structure. Regions with identical
        signatures can share Q/DQ insertion schemes.

        The signature captures:
        - Node operation types and key parameters
        - Hierarchical structure (child regions)
        - Deterministic ordering (sorted for consistency)

        Args:
            graph: The ONNX graph containing the region's nodes

        Returns:
            Signature string (e.g., "Conv->BatchNorm->Relu" or "COMPOSITE(...)")
        """
        raise NotImplementedError("Not implemented")


# =============================================================================
# Autotuner Q/DQ Insertion Specifications
# =============================================================================


@dataclass
class InsertionScheme:
    """Complete Q/DQ insertion specification for a region pattern.

    An InsertionScheme defines a complete Q/DQ configuration for a pattern,
    combining both node-level and region-level insertion points. The scheme
    is applied to all regions matching the pattern.

    **Scheme Identity:**
    - Uniquely identified by the combination of insertion points (computed hash)
    - latency_ms is a measured performance metric, not part of identity
    - Two schemes with same insertion points but different latencies are considered identical

    **Application:**
    - Node insertion points: Q/DQ at node inputs within the pattern
    - Region insertion points: Q/DQ at child region boundaries (COMPOSITE only)
    - All are resolved to actual configurations for each matching region

    **Performance Tracking:**
    - latency_ms: Measured performance (inf = not yet measured)
    - error: Whether this scheme encountered an error during measurement
    - Used to select the best scheme for each pattern

    **Attributes:**
        node_inputs: Q/DQ insertions at node inputs (list of NodeInputInsertionPoint)
        child_region_inputs: Q/DQ insertions at child boundaries (list of ChildRegionInputInsertionPoint)
        region_outputs: Q/DQ insertions at region outputs (list of RegionOutputInsertionPoint)
        latency_ms: Measured latency in milliseconds (inf if not measured)
        error: True if scheme measurement failed, False otherwise
        profile_timestamp: ISO format timestamp when this scheme was profiled (None if not yet profiled)
    """

    node_inputs: list[NodeInputInsertionPoint] = field(default_factory=list)
    child_region_inputs: list[ChildRegionInputInsertionPoint] = field(default_factory=list)
    region_outputs: list[RegionOutputInsertionPoint] = field(default_factory=list)
    latency_ms: float = float("inf")
    error: bool = False
    profile_timestamp: str | None = None

    @property
    def hash(self) -> str:
        """Compute deterministic hash for scheme identity.

        The hash uniquely identifies this scheme configuration based on its
        insertion points. Two schemes with identical insertion points produce
        the same hash, regardless of their measured latencies.

        **Hash Input:**
        - Sorted node_inputs (for deterministic ordering)
        - Sorted child_region_inputs (for deterministic ordering)
        - Sorted region_outputs (for deterministic ordering)
        - latency_ms is EXCLUDED (performance metric, not identity)

        **Use Cases:**
        - Detect duplicate schemes before measurement
        - Group schemes by configuration
        - Efficient scheme comparison

        Returns:
            32-character hexadecimal string (SHA-256 truncated to 128 bits)
        """
        # Sort points for deterministic hashing
        sorted_nodes = sorted([(pt.node_index, pt.input_index) for pt in self.node_inputs])
        sorted_regions = sorted(
            [(pt.region_index, pt.input_index) for pt in self.child_region_inputs]
        )
        sorted_region_outputs = sorted(
            [(pt.region_index, pt.node_index, pt.output_index) for pt in self.region_outputs]
        )

        # Create hash input string
        hash_input = f"{sorted_nodes}|{sorted_regions}|{sorted_region_outputs}"

        # Compute SHA-256 hash (128 bits)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:32]

    @property
    def is_empty(self) -> bool:
        """Check if this is a baseline scheme with no Q/DQ insertions.

        Returns:
            True if scheme has no node/region insertion points
        """
        return (
            len(self.node_inputs) == 0
            and len(self.child_region_inputs) == 0
            and len(self.region_outputs) == 0
        )

    @property
    def has_error(self) -> bool:
        """Check if this scheme encountered an error during measurement.

        Returns:
            True if scheme has error=True, False otherwise
        """
        return self.error

    @property
    def is_profiled(self) -> bool:
        """Check if this scheme has been profiled (measured).

        A scheme is considered profiled if it has been measured (has non-infinite latency)
        or has encountered an error during measurement.

        Returns:
            True if scheme has been measured (latency_ms != inf) or has error,
            False if scheme is waiting to be profiled (error=False and latency_ms=inf)
        """
        return self.error or self.latency_ms != float("inf")

    @property
    def num_node_insertions(self) -> int:
        """Get count of node-level Q/DQ insertion points.

        Returns:
            Number of NodeInputInsertionPoint entries
        """
        return len(self.node_inputs)

    @property
    def num_region_insertions(self) -> int:
        """Get count of region-level Q/DQ insertion points.

        These specify Q/DQ insertions at child region boundaries within
        COMPOSITE regions.

        Returns:
            Number of ChildRegionInputInsertionPoint entries
        """
        return len(self.child_region_inputs)

    @property
    def num_region_output_insertions(self) -> int:
        """Get count of region output insertion points.

        These specify Q/DQ insertions at outputs from child regions or nodes.

        Returns:
            Number of RegionOutputInsertionPoint entries
        """
        return len(self.region_outputs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "latency_ms": self.latency_ms,
            "error": self.error,
            "profile_timestamp": self.profile_timestamp,
            "nodes_insertion_points": [pt.to_dict() for pt in self.node_inputs],
            "child_region_inputs": [pt.to_dict() for pt in self.child_region_inputs],
            "region_outputs": [pt.to_dict() for pt in self.region_outputs],
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsertionScheme":
        """Create InsertionScheme from serialized dictionary.

        Reconstructs the insertion scheme from saved data, including node and
        region insertion points. The hash is automatically recomputed from all
        components to ensure consistency.

        Args:
            data: Dictionary containing 'latency_ms', 'nodes_insertion_points',
                  'child_region_inputs', and 'region_outputs' keys

        Returns:
            Reconstructed InsertionScheme instance
        """
        scheme = cls()
        scheme.latency_ms = data.get("latency_ms", float("inf"))
        scheme.error = data.get("error", False)
        scheme.profile_timestamp = data.get("profile_timestamp")

        scheme.node_inputs = [
            NodeInputInsertionPoint.from_dict(pt) for pt in data.get("nodes_insertion_points", [])
        ]
        scheme.child_region_inputs = [
            ChildRegionInputInsertionPoint.from_dict(pt)
            for pt in data.get("child_region_inputs", [])
        ]
        scheme.region_outputs = [
            RegionOutputInsertionPoint.from_dict(pt) for pt in data.get("region_outputs", [])
        ]

        # Note: hash is computed from points, so we don't load it from dict
        # This ensures consistency even if stored hash differs

        return scheme

    def distance(self, other: "InsertionScheme") -> int:
        """Compute edit distance between this scheme and another scheme.

        The edit distance is the minimum number of add/remove operations needed
        to transform this scheme into the other scheme. This is computed as the
        symmetric difference between the insertion point sets.

        **Distance Calculation:**
        - Counts insertion points in self but not in other (need to be removed)
        - Counts insertion points in other but not in self (need to be added)
        - Considers all three types of insertion points:
          * node_inputs
          * child_region_inputs
          * region_outputs

        Args:
            other: InsertionScheme to compare against

        Returns:
            Total edit distance (number of add + remove operations)

        Example:
            >>> scheme1 = InsertionScheme(
            ...     node_inputs=[
            ...         NodeInputInsertionPoint(0, 0),
            ...         NodeInputInsertionPoint(1, 0),
            ...     ]
            ... )
            >>> scheme2 = InsertionScheme(
            ...     node_inputs=[
            ...         NodeInputInsertionPoint(0, 0),
            ...         NodeInputInsertionPoint(2, 0),
            ...     ]
            ... )
            >>> scheme1.distance(scheme2)  # 2 (remove (1,0), add (2,0))
            2
        """
        # Convert insertion points to sets for efficient set operations
        self_nodes = set(self.node_inputs)
        other_nodes = set(other.node_inputs)

        self_regions = set(self.child_region_inputs)
        other_regions = set(other.child_region_inputs)

        self_region_outputs = set(self.region_outputs)
        other_region_outputs = set(other.region_outputs)

        # Compute symmetric difference (elements in either set but not both)
        # This gives us the total number of add + remove operations
        node_distance = len(self_nodes.symmetric_difference(other_nodes))
        region_distance = len(self_regions.symmetric_difference(other_regions))
        region_output_distance = len(self_region_outputs.symmetric_difference(other_region_outputs))

        return node_distance + region_distance + region_output_distance

    def __str__(self) -> str:
        """String representation for debugging."""
        error_str = ", error=True" if self.error else ""
        return (
            f"InsertionScheme(node_insertions={self.num_node_insertions}, "
            f"region_insertions={self.num_region_insertions}, "
            f"region_output_insertions={self.num_region_output_insertions}, "
            f"latency={self.latency_ms:.3f}ms{error_str})"
        )
