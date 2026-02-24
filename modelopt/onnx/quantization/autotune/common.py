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

"""Common data structures and types for the QDQ Autotuner."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    ChildRegionOutputInsertionPoint,
    NodeInputInsertionPoint,
)


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
    """A subgraph region in an ONNX graph, used as the unit for Q/DQ insertion.

    Regions form a hierarchy: ROOT contains the entire graph, COMPOSITE regions
    contain child regions, and LEAF regions contain only nodes. Each region tracks
    its direct nodes, input/output tensors, and a pattern signature for matching
    regions with identical structure.
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
        self.metadata: dict[str, str] = {}

    def get_children(self, *, sort: bool = False) -> list["Region"]:
        """Get all child regions. If sort is True, sort the children by level and size.

        Args:
            sort: Whether to sort the children by level and size

        Returns:
            List of child regions
        """
        if sort:
            return sorted(
                self.children, key=lambda r: (-r.level, r.get_size_of_region_and_descendants())
            )
        return self.children

    def remove_child(self, child: "Region") -> bool:
        """Remove a child region from this region's children list."""
        if child not in self.children:
            return False
        self.children.remove(child)
        if child.parent and child.parent.id == self.id:
            child.parent = None
        return True

    def add_child(self, child: "Region") -> None:
        """Add a child sub-region."""
        if child.id == self.id:
            logger.warning(f"Cannot add region {self.id} as its own child")
            return

        if self.is_descendant_of(child):
            logger.warning(
                f"Cycle detected: region {self.id} is already a descendant of region {child.id}"
            )
            return

        if child.parent is not None and child.parent.id != self.id:
            old_parent_id = child.parent.id
            logger.debug(
                f"Re-parenting region {child.id}: moving from parent {old_parent_id} to {self.id}"
            )
            child.parent.remove_child(child)

        if any(c.id == child.id for c in self.children):
            logger.debug(f"Region {child.id} already child of {self.id}")
            return

        self.children.append(child)
        child.parent = self

    def is_descendant_of(self, potential_ancestor: "Region") -> bool:
        """Check if this region is a descendant of potential_ancestor."""
        visited = set()
        current = self.parent
        while current:
            if current.id in visited:
                return False
            visited.add(current.id)
            if current.id == potential_ancestor.id:
                return True
            current = current.parent
        return False

    def get_nodes(self, *, sort: bool = False) -> list[int]:
        """Get direct node indices in this region only."""
        if sort:
            return sorted(self.nodes)
        return list(self.nodes)

    def get_region_nodes_and_descendants(self, _visited: set[int] | None = None) -> set[int]:
        """Get all node indices recursively, including descendants."""
        if _visited is None:
            _visited = set()

        # Detect cycles
        assert self.id not in _visited, f"Cycle detected in region {self.id} during node traversal"

        _visited.add(self.id)
        all_nodes = set(self.nodes)
        for child in self.children:
            all_nodes.update(child.get_region_nodes_and_descendants(_visited))
        return all_nodes

    def contains_node(self, node_index: int) -> bool:
        """Check if region contains a specific node (direct only)."""
        return node_index in self.nodes

    def contains_node_within_region_and_descendants(self, node_index: int) -> bool:
        """Check if region contains a node recursively."""
        return node_index in self.get_region_nodes_and_descendants()

    def get_size_of_region_and_descendants(self, _visited: set[int] | None = None) -> int:
        """Get total node count recursively including all descendants."""
        if _visited is None:
            _visited = set()

        # Detect cycles
        assert self.id not in _visited, (
            f"Cycle detected in region {self.id} during size calculation"
        )

        _visited.add(self.id)
        total = len(self.nodes)
        for child in self.children:
            total += child.get_size_of_region_and_descendants(_visited)
        return total

    def merge(self, other: "Region") -> None:
        """Merge another region into this one."""
        if not other:
            return
        self.nodes.update(other.nodes)
        for child in other.children:
            self.add_child(child)

    def __repr__(self) -> str:
        type_str = self.type.value
        return (
            f"Region[id={self.id}, level={self.level}, type={type_str}, "
            f"nodes={len(self.nodes)}, children={len(self.children)}, "
            f"inputs={len(self.inputs)}, outputs={len(self.outputs)}]"
        )


@dataclass
class InsertionScheme:
    """Complete Q/DQ insertion specification for a region pattern.

    An InsertionScheme defines a complete Q/DQ configuration for a pattern,
    combining both node-level and region-level insertion points. The scheme
    is applied to all regions matching the pattern.
    """

    node_inputs: list[NodeInputInsertionPoint] = field(default_factory=list)
    child_region_inputs: list[ChildRegionInputInsertionPoint] = field(default_factory=list)
    region_outputs: list[ChildRegionOutputInsertionPoint] = field(default_factory=list)
    latency_ms: float = float("inf")
    error: bool = False
    profile_timestamp: str | None = None

    @property
    def hash(self) -> str:
        """Compute deterministic hash for scheme identity.

        The hash uniquely identifies this scheme configuration based on its
        insertion points. Two schemes with identical insertion points produce
        the same hash, regardless of their measured latencies.
        """
        sorted_nodes = sorted([(pt.node_index, pt.input_index) for pt in self.node_inputs])
        sorted_regions = sorted(
            [(pt.region_index, pt.input_index) for pt in self.child_region_inputs]
        )
        sorted_region_outputs = sorted(
            [(pt.region_index, pt.node_index, pt.output_index) for pt in self.region_outputs]
        )

        hash_input = f"{sorted_nodes}|{sorted_regions}|{sorted_region_outputs}"

        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:32]

    @property
    def is_empty(self) -> bool:
        """Check if this is a baseline scheme with no Q/DQ insertions."""
        return not self.node_inputs and not self.child_region_inputs and not self.region_outputs

    @property
    def is_profiled(self) -> bool:
        """Check if this scheme has been profiled (measured).

        A scheme is considered profiled if it has been measured (has non-infinite latency)
        or has encountered an error during measurement.
        """
        return self.error or self.latency_ms != float("inf")

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
        """Create InsertionScheme from serialized dictionary."""
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
            ChildRegionOutputInsertionPoint.from_dict(pt) for pt in data.get("region_outputs", [])
        ]

        return scheme

    def distance(self, other: "InsertionScheme") -> int:
        """Compute edit distance between this scheme and another scheme.

        The edit distance is the minimum number of add/remove operations needed
        to transform this scheme into the other scheme. This is computed as the
        symmetric difference between the insertion point sets.

        Args:
            other: InsertionScheme to compare against

        Returns:
            Total edit distance (number of add + remove operations)
        """
        return (
            len(set(self.node_inputs).symmetric_difference(other.node_inputs))
            + len(set(self.child_region_inputs).symmetric_difference(other.child_region_inputs))
            + len(set(self.region_outputs).symmetric_difference(other.region_outputs))
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        error_str = ", error=True" if self.error else ""
        return (
            f"InsertionScheme(node_insertions={len(self.node_inputs)}, "
            f"region_insertions={len(self.child_region_inputs)}, "
            f"region_output_insertions={len(self.region_outputs)}, "
            f"latency={self.latency_ms:.3f}ms{error_str})"
        )
