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
from typing import TYPE_CHECKING, Any, Optional

import onnx_graphsurgeon as gs
import yaml

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.insertion_points import (
    ChildRegionInputInsertionPoint,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
    ResolvedInsertionPoint,
)

if TYPE_CHECKING:
    from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern


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
        """Get all child regions."""
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

    def add_node(self, node_index: int) -> None:
        """Add a node index to this region."""
        self.nodes.add(node_index)

    def add_nodes(self, node_indices: list[int]) -> None:
        """Add multiple node indices to this region."""
        self.nodes.update(node_indices)

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

    def add_input(self, tensor_name: str) -> None:
        """Add an input tensor name."""
        if tensor_name not in self.inputs:
            self.inputs.append(tensor_name)

    def add_output(self, tensor_name: str) -> None:
        """Add an output tensor name."""
        if tensor_name not in self.outputs:
            self.outputs.append(tensor_name)

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


@dataclass
class InsertionScheme:
    """Q/DQ insertion specification applied to all regions matching a pattern."""

    node_inputs: list[NodeInputInsertionPoint] = field(default_factory=list)
    child_region_inputs: list[ChildRegionInputInsertionPoint] = field(default_factory=list)
    region_outputs: list[RegionOutputInsertionPoint] = field(default_factory=list)
    latency_ms: float = float("inf")
    error: bool = False
    profile_timestamp: str | None = None

    @property
    def hash(self) -> str:
        """Compute deterministic hash for scheme identity."""
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
        """Check if this scheme has been profiled (measured)."""
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
            RegionOutputInsertionPoint.from_dict(pt) for pt in data.get("region_outputs", [])
        ]

        return scheme

    def distance(self, other: "InsertionScheme") -> int:
        """Compute edit distance between this scheme and another scheme."""
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


@dataclass
class PatternSchemes:
    """Collection of Q/DQ insertion schemes for a single pattern.

    Manages multiple InsertionScheme candidates for a region pattern, tracking
    their performance and identifying the best-performing configuration. This
    enables pattern-based optimization where all regions with the same structure
    use the same Q/DQ insertion strategy.

    **Attributes:**
        pattern: RegionPattern defining the structural signature
        schemes: List of InsertionScheme candidates with measurements
    """

    pattern: Optional["RegionPattern"] = None
    schemes: list[InsertionScheme] = field(default_factory=list)

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
            if not scheme.error and scheme.latency_ms < min_latency:
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
        return [s for s in self.schemes if not s.error]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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
        from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

        ps = cls()

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

    **Attributes:**
        pattern_schemes: List of PatternSchemes, one per pattern
        minimum_distance: Minimum edit distance required between schemes in cache.
            When adding new schemes, if a scheme is too similar (distance < minimum_distance)
            to an existing scheme, only the better-performing one is kept (default: 4)
        max_entries_per_pattern: Maximum number of schemes to keep per pattern.
            Only the top N best-performing schemes are kept for each pattern.
            Use 0 to keep all schemes (default: 32)
    """

    # List of PatternSchemes in the cache.
    pattern_schemes: list[PatternSchemes] = field(default_factory=list)
    # Minimum distance between schemes in cache.
    minimum_distance: int = 4
    # Maximum number of schemes per pattern.
    max_entries_per_pattern: int = 32

    def add_pattern_schemes(self, pattern_schemes: PatternSchemes) -> None:
        """Add PatternSchemes to pattern cache with automatic eviction of non-performant entries.

        Merges new schemes with existing schemes for the same pattern, automatically
        evicting schemes that are non-performant based on multiple criteria.

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
        valid_schemes = [s for s in all_schemes if not s.error]
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
                schemes_to_replace = []
                for existing_scheme in filtered_schemes:
                    distance = scheme.distance(existing_scheme)
                    if distance < self.minimum_distance:
                        # Schemes are too similar, keep the better one
                        too_similar = True
                        if scheme.latency_ms < existing_scheme.latency_ms:
                            # New scheme is better, mark existing for replacement
                            schemes_to_replace.append(existing_scheme)

                if not too_similar:
                    filtered_schemes.append(scheme)
                elif schemes_to_replace:
                    for scheme_to_replace in schemes_to_replace:
                        filtered_schemes.remove(scheme_to_replace)
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

        Args:
            region: Region from the quantized model to analyze
            graph: ONNX graph containing the region
            quantized_tensors: Set of tensor names that have Q/DQ nodes

        """
        from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern

        pattern = RegionPattern.from_region(region, graph)
        scheme = InsertionScheme(
            node_inputs=[],
            child_region_inputs=[],
            region_outputs=[],
            latency_ms=float("inf"),
            error=False,
        )
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
        pattern_schemes = PatternSchemes(pattern=pattern, schemes=[scheme])
        self.add_pattern_schemes(pattern_schemes)
        num_points = (
            len(scheme.node_inputs) + len(scheme.child_region_inputs) + len(scheme.region_outputs)
        )
        logger.debug(f"Added pattern from region {region.id} with {num_points} insertion points")
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
        """Get list of all pattern signatures in pattern cache."""
        return [ps.pattern_signature for ps in self.pattern_schemes]

    def clear(self) -> None:
        """Clear all pattern cache data."""
        self.pattern_schemes.clear()

    def merge(self, other: "PatternCache", prefer_existing: bool = True) -> None:
        """Merge another PatternCache into this one."""
        for schemes in other.pattern_schemes:
            if not self.has_pattern(schemes.pattern_signature) or not prefer_existing:
                self.add_pattern_schemes(schemes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "minimum_distance": self.minimum_distance,
            "max_entries_per_pattern": self.max_entries_per_pattern,
            "pattern_schemes": [ps.to_dict() for ps in self.pattern_schemes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternCache":
        """Create PatternCache from serialized dictionary."""
        cache = cls(
            minimum_distance=data.get("minimum_distance", 4),
            max_entries_per_pattern=data.get("max_entries_per_pattern", 32),
        )

        for ps_data in data.get("pattern_schemes", []):
            ps = PatternSchemes.from_dict(ps_data, pattern=None)
            cache.pattern_schemes.append(ps)

        return cache

    def save(self, output_path: str) -> None:
        """Save pattern cache to a YAML file."""
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
        """Load pattern cache from a YAML file."""
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
    """Configuration parameters for QDQ autotuning."""

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
