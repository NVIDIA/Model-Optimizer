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
from typing import TYPE_CHECKING, Any, Optional

import onnx_graphsurgeon as gs
import yaml

from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
    ResolvedInsertionPoint,
)

if TYPE_CHECKING:
    from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

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
        # Import here to avoid circular dependency at runtime
        from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

        return RegionPattern.from_region(self, graph).signature


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


@dataclass
class PatternSchemes:
    """Collection of Q/DQ insertion schemes for a single pattern.

    Manages multiple InsertionScheme candidates for a region pattern, tracking
    their performance and identifying the best-performing configuration. This
    enables pattern-based optimization where all regions with the same structure
    use the same Q/DQ insertion strategy.

    **Workflow:**
    1. Pattern is identified from region structure
    2. Multiple schemes are generated and tested
    3. Each scheme is measured (latency_ms)
    4. Best scheme is selected (lowest latency)
    5. Best scheme is applied to all matching regions

    **Best Scheme Selection:**
    - Automatically identifies scheme with lowest latency
    - Excludes schemes with errors (error=True)
    - Schemes with latency_ms = inf are considered unmeasured
    - best_scheme property provides easy access to optimal configuration

    **Attributes:**
        pattern: RegionPattern defining the structural signature
        schemes: List of InsertionScheme candidates with measurements
    """

    pattern: Optional["RegionPattern"] = None  # Structural pattern signature
    schemes: list[InsertionScheme] = field(default_factory=list)  # Candidate schemes

    @property
    def pattern_signature(self) -> str:
        """Get the pattern signature string."""
        return self.pattern.signature if self.pattern else ""

    @property
    def pattern_size(self) -> int:
        """Get the pattern size (total node count)."""
        return self.pattern.size if self.pattern else 0

    @property
    def best_scheme_index(self) -> int:
        """Get index of the best performing scheme (lowest latency).

        Scans all schemes to find the one with minimum latency_ms,
        excluding schemes with errors.
        If no schemes exist or all have errors, returns -1.

        Returns:
            Index of best scheme (without errors), or -1 if no valid schemes available
        """
        if len(self.schemes) == 0:
            return -1
        min_idx, min_latency = -1, float("inf")
        for idx, scheme in enumerate(self.schemes):
            if not scheme.has_error and scheme.latency_ms < min_latency:
                min_idx = idx
                min_latency = scheme.latency_ms
        return min_idx

    @property
    def best_scheme(self) -> InsertionScheme | None:
        """Get the best performing scheme (lowest latency).

        Convenience property for accessing the optimal scheme directly
        without needing to look up by index. Excludes schemes with errors.

        Returns:
            InsertionScheme with lowest latency (excluding error schemes),
            or None if no valid schemes exist
        """
        index = self.best_scheme_index
        if index < 0 or index >= len(self.schemes):
            return None
        return self.schemes[index]

    @property
    def num_schemes(self) -> int:
        """Get total number of schemes."""
        return len(self.schemes)

    @property
    def has_schemes(self) -> bool:
        """Check if any schemes have been added."""
        return len(self.schemes) > 0

    def add_scheme(self, scheme: InsertionScheme) -> None:
        """Add a scheme to the collection.

        Args:
            scheme: InsertionScheme to add
        """
        self.schemes.append(scheme)

    def get_measured_schemes(self) -> list[InsertionScheme]:
        """Get schemes that have been measured (finite latency).

        Returns:
            List of schemes with performance measurements (excludes unmeasured schemes with inf latency)
        """
        return [s for s in self.schemes if s.latency_ms != float("inf")]

    def get_valid_schemes(self) -> list[InsertionScheme]:
        """Get schemes without errors.

        Returns:
            List of schemes that completed successfully without errors
        """
        return [s for s in self.schemes if not s.has_error]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Note: Excludes runtime objects like pattern (RegionPattern).
        Only serializes metadata and schemes.
        """
        return {
            "pattern_signature": self.pattern_signature,
            "pattern_size": self.pattern_size,
            "schemes": [scheme.to_dict() for scheme in self.schemes],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], pattern: Optional["RegionPattern"] = None
    ) -> "PatternSchemes":
        """Create PatternSchemes from serialized dictionary.

        Reconstructs the pattern schemes collection from saved data. The
        RegionPattern object must be provided separately since it's not
        serialized (it's a runtime object computed from the graph).

        If no pattern is provided, creates a minimal RegionPattern from the
        saved signature and size for signature matching purposes.

        Args:
            data: Dictionary containing 'pattern_signature', 'pattern_size',
                  and 'schemes' keys
            pattern: RegionPattern object to associate (must match signature).
                    If None, creates minimal pattern from saved data.

        Returns:
            Reconstructed PatternSchemes instance
        """
        # Import here to avoid circular dependency at runtime
        from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

        ps = cls()

        # If no pattern provided, create minimal one from saved data
        if pattern is None and "pattern_signature" in data:
            pattern = RegionPattern(
                signature=data["pattern_signature"], size=data.get("pattern_size", 0)
            )

        ps.pattern = pattern

        ps.schemes = [
            InsertionScheme.from_dict(scheme_data) for scheme_data in data.get("schemes", [])
        ]

        return ps

    def __str__(self) -> str:
        """String representation for debugging."""
        best_latency = self.best_scheme.latency_ms if self.best_scheme else 0.0
        return (
            f"PatternSchemes(pattern='{self.pattern_signature[:40]}...', "
            f"schemes={self.num_schemes}, best_latency={best_latency:.3f}ms)"
        )


@dataclass
class PatternCache:
    """Pattern cache containing best-performing schemes for patterns with automatic eviction.

    Stores a collection of PatternSchemes that can be used as seeds for autotuning.
    Each PatternSchemes contains high-performing insertion schemes for a specific
    pattern signature. The cache automatically evicts non-performant schemes based on:
    - Error status (schemes with errors are evicted)
    - Duplicate schemes (only better-performing duplicate is kept)
    - Similarity (similar schemes where only better-performing one is kept)
    - Count limit (only top N best schemes are kept per pattern)

    **Seeded Autotuning:**
    - Use previous autotuning results as starting points
    - Skip redundant measurements for known patterns
    - Transfer learned schemes across models or runs

    **Use Cases:**
    - Load pattern cache from previous run to warm-start autotuning
    - Share pattern cache data across similar models
    - Store best-known schemes for common patterns

    **Workflow:**
    1. After autotuning, add schemes to PatternCache (non-performant entries auto-evicted)
    2. Serialize PatternCache to file (YAML)
    3. Load PatternCache in future runs as seeds
    4. Autotuner uses seeds to initialize pattern schemes

    **Attributes:**
        pattern_schemes: List of PatternSchemes, one per pattern
        minimum_distance: Minimum edit distance required between schemes in cache.
            When adding new schemes, if a scheme is too similar (distance < minimum_distance)
            to an existing scheme, only the better-performing one is kept (default: 4)
        max_entries_per_pattern: Maximum number of schemes to keep per pattern.
            Only the top N best-performing schemes are kept for each pattern.
            Use 0 to keep all schemes (default: 32)

    Example:
        >>> # Save pattern cache after autotuning
        >>> cache = PatternCache(minimum_distance=4, max_entries_per_pattern=32)
        >>> for schemes in autotuner.pattern_schemes.values():
        ...     cache.add_pattern_schemes(schemes)  # Auto-eviction happens here
        >>> cache.save("pattern_cache.yaml")
        >>>
        >>> # Load pattern cache for next run
        >>> cache = PatternCache.load("pattern_cache.yaml")
        >>> autotuner.initialize(config, pattern_cache=cache)
    """

    pattern_schemes: list[PatternSchemes] = field(default_factory=list)
    # Minimum distance between schemes in cache.
    minimum_distance: int = 4
    # Maximum number of schemes per pattern.
    max_entries_per_pattern: int = 32

    def add_pattern_schemes(self, pattern_schemes: PatternSchemes) -> None:
        """Add PatternSchemes to pattern cache with automatic eviction of non-performant entries.

        Merges new schemes with existing schemes for the same pattern, automatically
        evicting schemes that are non-performant based on multiple criteria.

        **Automatic Eviction Strategy:**

        1. **Error Eviction**: Schemes with errors are automatically excluded

        2. **Duplicate Eviction**: When schemes have identical configurations (same hash),
           only the one with better latency is kept

        3. **Similarity Eviction**: When minimum_distance > 0, schemes that are too similar
           to better-performing schemes are evicted

        4. **Count Eviction**: When max_entries_per_pattern > 0, only the top N
           best-performing schemes are kept per pattern

        Args:
            pattern_schemes: PatternSchemes to add to the cache
        """
        if not pattern_schemes or not pattern_schemes.pattern:
            return

        pattern_sig = pattern_schemes.pattern_signature

        # Find existing PatternSchemes for this pattern
        existing_idx = None
        for idx, ps in enumerate(self.pattern_schemes):
            if ps.pattern_signature == pattern_sig:
                existing_idx = idx
                break

        # Collect all schemes (existing + new)
        all_schemes = list(pattern_schemes.schemes)
        if existing_idx is not None:
            all_schemes.extend(self.pattern_schemes[existing_idx].schemes)

        # Filter out schemes with errors and deduplicate by hash
        valid_schemes = [s for s in all_schemes if not s.has_error]
        unique_schemes = {}
        for scheme in valid_schemes:
            scheme_hash = scheme.hash
            if (
                scheme_hash not in unique_schemes
                or scheme.latency_ms < unique_schemes[scheme_hash].latency_ms
            ):
                unique_schemes[scheme_hash] = scheme

        # Sort by latency to get best schemes
        sorted_schemes = sorted(unique_schemes.values(), key=lambda s: s.latency_ms)

        # Apply distance-based filtering if minimum_distance > 0
        if self.minimum_distance > 0:
            filtered_schemes = []
            for scheme in sorted_schemes:
                # Check if this scheme is too similar to any already-filtered scheme
                too_similar = False
                for existing_scheme in filtered_schemes:
                    distance = scheme.distance(existing_scheme)
                    if distance < self.minimum_distance:
                        # Schemes are too similar, keep the better one
                        if scheme.latency_ms < existing_scheme.latency_ms:
                            # New scheme is better, remove existing and add new
                            filtered_schemes.remove(existing_scheme)
                            break
                        else:
                            # Existing scheme is better, skip new one
                            too_similar = True
                            break

                if not too_similar:
                    filtered_schemes.append(scheme)

            sorted_schemes = filtered_schemes

        # Apply count limit if max_entries_per_pattern > 0
        # Keep only the top N best-performing schemes per pattern
        if self.max_entries_per_pattern > 0:
            sorted_schemes = sorted_schemes[: self.max_entries_per_pattern]

        # Create PatternSchemes with all schemes that passed the eviction criteria
        result = PatternSchemes(pattern=pattern_schemes.pattern)
        result.schemes = sorted_schemes

        # Replace existing or append new
        if existing_idx is not None:
            self.pattern_schemes[existing_idx] = result
        else:
            self.pattern_schemes.append(result)

    def get_pattern_schemes(self, pattern_signature: str) -> PatternSchemes | None:
        """Get PatternSchemes for a specific pattern signature.

        Args:
            pattern_signature: Pattern signature to lookup

        Returns:
            PatternSchemes if found, None otherwise
        """
        for ps in self.pattern_schemes:
            if ps.pattern_signature == pattern_signature:
                return ps
        return None

    def has_pattern(self, pattern_signature: str) -> bool:
        """Check if pattern cache contains a specific pattern.

        Args:
            pattern_signature: Pattern signature to check

        Returns:
            True if pattern exists in pattern cache
        """
        return any(ps.pattern_signature == pattern_signature for ps in self.pattern_schemes)

    def add_pattern_from_region(
        self, region: Region, graph: gs.Graph, quantized_tensors: set[str]
    ) -> None:
        """Build and add a pattern cache entry from a region in a quantized model.

        Analyzes a region from an already-quantized model to extract its Q/DQ
        insertion scheme. This allows capturing known-good quantization strategies
        from existing models and using them as seeds for autotuning.

        **Workflow:**
        1. Create RegionPattern from the region structure
        2. Identify which tensors in the region are quantized
        3. Map quantized tensors to pattern-relative insertion points:
           - Node input tensors → NodeInputInsertionPoint
           - Child region input tensors → ChildRegionInputInsertionPoint
           - Region output tensors → RegionOutputInsertionPoint
        4. Create InsertionScheme with identified insertion points
        5. Add to pattern cache (or merge if pattern already exists)

        Args:
            region: Region from the quantized model to analyze
            graph: ONNX graph containing the region
            quantized_tensors: Set of tensor names that have Q/DQ nodes

        Example:
            >>> cache = PatternCache()
            >>> for region in all_regions:
            ...     cache.add_pattern_from_region(region, graph, quantized_tensors)
            >>> cache.save("learned_patterns.yaml")
        """
        # Import here to avoid circular dependency at runtime
        from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

        # Create pattern from region
        pattern = RegionPattern.from_region(region, graph)
        # Track insertion points
        scheme = InsertionScheme(
            node_inputs=[],
            child_region_inputs=[],
            region_outputs=[],
            latency_ms=float("inf"),
            error=False,
        )
        # Analyze node inputs
        full_insertion_scheme = pattern.get_full_insertion_scheme(region, graph)
        for point in full_insertion_scheme.node_inputs:
            temp_scheme = InsertionScheme(
                node_inputs=[point],
                child_region_inputs=[],
                region_outputs=[],
                latency_ms=float("inf"),
                error=False,
            )
            temp_ips: list[ResolvedInsertionPoint] = pattern.matches(region, graph, temp_scheme)
            temp_tensor_names = {tensor.tensor_name for tensor in temp_ips}
            if len(temp_tensor_names.intersection(quantized_tensors)) > 0:
                scheme.node_inputs.append(point)
        # Analyze region boundaries (for COMPOSITE regions)
        if region.type == RegionType.COMPOSITE:
            for child_point in full_insertion_scheme.child_region_inputs:
                temp_scheme = InsertionScheme(
                    node_inputs=[],
                    child_region_inputs=[child_point],
                    region_outputs=[],
                    latency_ms=float("inf"),
                    error=False,
                )
                temp_ips = pattern.matches(region, graph, temp_scheme)
                temp_tensor_names = {tensor.tensor_name for tensor in temp_ips}
                if len(temp_tensor_names.intersection(quantized_tensors)) > 0:
                    scheme.child_region_inputs.append(child_point)
        # Analyze region outputs
        for output_point in full_insertion_scheme.region_outputs:
            temp_scheme = InsertionScheme(
                node_inputs=[],
                child_region_inputs=[],
                region_outputs=[output_point],
                latency_ms=float("inf"),
                error=False,
            )
            temp_ips = pattern.matches(region, graph, temp_scheme)
            temp_tensor_names = {tensor.tensor_name for tensor in temp_ips}
            if len(temp_tensor_names.intersection(quantized_tensors)) > 0:
                scheme.region_outputs.append(output_point)
        # Add pattern and scheme to pattern cache
        pattern_schemes = PatternSchemes(pattern=pattern, schemes=[scheme])
        self.add_pattern_schemes(pattern_schemes)
        num_points = (
            len(scheme.node_inputs) + len(scheme.child_region_inputs) + len(scheme.region_outputs)
        )
        logger.debug(
            f"Added pattern from region {region.get_id()} with {num_points} insertion points"
        )
        # Add patterns from child regions
        if region.type == RegionType.COMPOSITE:
            for child_region in region.get_children():
                self.add_pattern_from_region(child_region, graph, quantized_tensors)

    @property
    def num_patterns(self) -> int:
        """Get number of patterns in pattern cache."""
        return len(self.pattern_schemes)

    @property
    def total_schemes(self) -> int:
        """Get total number of schemes across all patterns."""
        return sum(ps.num_schemes for ps in self.pattern_schemes)

    def get_all_pattern_signatures(self) -> list[str]:
        """Get list of all pattern signatures in pattern cache.

        Returns:
            List of pattern signature strings
        """
        return [ps.pattern_signature for ps in self.pattern_schemes]

    def clear(self) -> None:
        """Clear all pattern cache data."""
        self.pattern_schemes.clear()

    def merge(self, other: "PatternCache", prefer_existing: bool = True) -> None:
        """Merge another PatternCache into this one.

        Args:
            other: PatternCache to merge
            prefer_existing: If True, keep existing patterns when there's a conflict.
                           If False, overwrite with other's patterns.
        """
        for schemes in other.pattern_schemes:
            if not self.has_pattern(schemes.pattern_signature) or not prefer_existing:
                self.add_pattern_schemes(schemes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with 'minimum_distance', 'max_entries_per_pattern', and 'pattern_schemes' keys
        """
        return {
            "minimum_distance": self.minimum_distance,
            "max_entries_per_pattern": self.max_entries_per_pattern,
            "pattern_schemes": [ps.to_dict() for ps in self.pattern_schemes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternCache":
        """Create PatternCache from serialized dictionary.

        Note: RegionPattern objects are not restored (they're runtime objects).
        Only pattern signatures and scheme data are loaded.

        Args:
            data: Dictionary containing pattern cache data

        Returns:
            Reconstructed PatternCache instance
        """
        cache = cls(
            minimum_distance=data.get("minimum_distance", 4),
            max_entries_per_pattern=data.get("max_entries_per_pattern", 32),
        )

        for ps_data in data.get("pattern_schemes", []):
            # Create PatternSchemes without pattern object (pattern=None)
            ps = PatternSchemes.from_dict(ps_data, pattern=None)
            cache.pattern_schemes.append(ps)

        return cache

    def save(self, output_path: str) -> None:
        """Save pattern cache to a YAML file.

        Serializes all pattern schemes and their insertion points to a YAML file
        that can be loaded later for seeded autotuning. The format matches the
        autotuner state file format for consistency.

        **Contents:**
        - minimum_distance: Minimum distance between schemes
        - max_entries_per_pattern: Maximum number of schemes per pattern
        - pattern_schemes: List of all PatternSchemes with their insertion points

        Args:
            output_path: File path where the YAML pattern cache file will be written

        Example:
            >>> cache = PatternCache(minimum_distance=1, max_entries_per_pattern=16)
            >>> for schemes in autotuner.pattern_schemes.values():
            ...     cache.add_pattern_schemes(schemes)
            >>> cache.save("pattern_cache.yaml")
        """
        state = self.to_dict()

        with open(output_path, "w") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)

        logger.info(
            f"Saved pattern cache → {output_path} ({self.num_patterns} patterns, "
            f"{self.total_schemes} schemes)"
        )
        logger.debug(
            f"Cache settings: min_distance={self.minimum_distance}, "
            f"max_per_pattern={self.max_entries_per_pattern}"
        )

    @classmethod
    def load(cls, input_path: str) -> "PatternCache":
        """Load pattern cache from a YAML file.

        Reads a previously saved pattern cache file and reconstructs all pattern
        schemes. The loaded pattern cache can be used to seed autotuning with
        known-good insertion schemes.

        **Note:** RegionPattern objects are not restored since they depend on
        the actual model structure. Only pattern signatures and scheme data
        are loaded.

        Args:
            input_path: File path to the YAML pattern cache file to load

        Returns:
            PatternCache instance with all pattern schemes loaded

        Raises:
            FileNotFoundError: If the input_path doesn't exist

        Example:
            >>> cache = PatternCache.load("pattern_cache.yaml")
            >>> autotuner.initialize(config, pattern_cache=cache)
        """
        with open(input_path) as f:
            state = yaml.safe_load(f)

        cache = cls.from_dict(state)

        logger.info(
            f"Loaded pattern cache from {input_path} ({cache.num_patterns} patterns, "
            f"{cache.total_schemes} schemes)"
        )
        logger.debug(
            f"Cache settings: min_distance={cache.minimum_distance}, "
            f"max_per_pattern={cache.max_entries_per_pattern}"
        )

        return cache

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"PatternCache(patterns={self.num_patterns}, "
            f"schemes={self.total_schemes}, "
            f"minimum_distance={self.minimum_distance}, "
            f"max_entries_per_pattern={self.max_entries_per_pattern})"
        )


@dataclass
class Config:
    """Configuration parameters for QDQ autotuning.

    Controls the autotuning process including performance requirements, quantization
    parameters, region building, scheme generation, and finetuning behavior.

    Attributes:
        # Logging
        verbose: Enable detailed logging of autotuning progress (default: False)

        # Quantization Parameters
        default_q_scale: Default scale parameter for Q/DQ nodes. Controls quantization
            granularity. Typical range: 0.01-0.1 (default: 0.1)
        default_q_zero_point: Default zero-point for Q/DQ nodes. Use 0 for signed int8,
            128 for unsigned uint8 (default: 0)
        default_quant_type: Quantization type for Q/DQ nodes. Options: "int8" (default), "fp8"

        # Region Builder Settings
        maximum_sequence_region_size: Maximum number of nodes in a sequence region during
            top-down refinement. Prevents overly large merged regions (default: 10)
        minimum_topdown_search_size: Minimum number of nodes in a region to trigger
            top-down search during region building (default: 10)

    # Scheme Generation Settings
    top_percent_to_mutate: Top percentage of best schemes to use as mutation seeds
        during scheme generation. Range: 0.0-1.0 (default: 0.1 = top 10%)
    minimum_schemes_to_mutate: Minimum number of schemes to keep as mutation seeds,
        even if top_percent_to_mutate results in fewer (default: 10)
    maximum_mutations: Maximum number of mutations to apply to a single scheme
        during generation (default: 3)
    maximum_generation_attempts: Maximum attempts to generate a unique new scheme
        before giving up (default: 100)

    # Pattern Cache Settings
    pattern_cache_minimum_distance: Minimum edit distance required between schemes in cache.
        When adding schemes, if a scheme is too similar (distance < minimum_distance)
        to an existing scheme, only the better-performing one is kept (default: 4)
    pattern_cache_max_entries_per_pattern: Maximum number of schemes to keep per pattern
        in pattern cache. Only the top N best-performing schemes are kept for each pattern.
        Use 0 to keep all schemes (default: 32)

    Example:
        >>> config = Config(
        ...     verbose=True,  # Enable detailed logging
        ...     top_percent_to_mutate=0.2,  # Use top 20% schemes as seeds
        ...     pattern_cache_minimum_distance=2,  # Require more diversity in cache
        ... )
        >>> autotuner = QDQAutotuner(model)
        >>> autotuner.initialize(config)
    """

    # Logging
    verbose: bool = False

    # Quantization Parameters
    default_q_scale: float = 0.1
    default_q_zero_point: int = 0
    default_quant_type: str = "int8"
    default_dq_dtype: str = "float32"

    # Region Builder Settings
    maximum_sequence_region_size: int = 10
    minimum_topdown_search_size: int = 10

    # Scheme Generation Settings
    top_percent_to_mutate: float = 0.1
    minimum_schemes_to_mutate: int = 10
    maximum_mutations: int = 3
    maximum_generation_attempts: int = 100

    # Pattern Cache Settings
    pattern_cache_minimum_distance: int = 4
    pattern_cache_max_entries_per_pattern: int = 32
