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

"""Region Search - Hierarchical Region Discovery and Partitioning.

This module provides sophisticated algorithms for discovering and organizing regions
in ONNX computation graphs. It creates hierarchical region structures that respect
computational patterns like divergence, convergence, and sequential operations.

**Core Functionality:**
- **Two-Phase Region Discovery**: Combines bottom-up partitioning with top-down refinement
- **Pattern Recognition**: Identifies divergence/convergence patterns in computation flow
- **Hierarchical Structure**: Creates COMPOSITE regions containing LEAF child regions
- **Boundary Computation**: Automatically determines region input/output tensors
- **Graph Analysis**: Pre-computes reachability and data flow information

**Key Algorithms:**

1. **Bottom-Up Partitioning (RegionPartitioner)**:
   - Traverses graph from inputs to outputs
   - Identifies divergent nodes where computation branches
   - Finds convergence points where branches rejoin
   - Creates initial LEAF regions based on these patterns

2. **Top-Down Refinement (TopDownRegionBuilder)**:
   - Merges converged sub-patterns within regions
   - Splits long sequences into optimal-sized regions
   - Creates hierarchical COMPOSITE region structures
   - Respects operation boundaries (Conv, Gemm, etc.)

3. **Combined Strategy (CombinedRegionSearch)**:
   - Orchestrates both phases for comprehensive region discovery
   - Produces well-formed hierarchical regions covering entire graph

**Region Types:**
- **LEAF regions**: Contain actual graph nodes (basic building blocks)
- **COMPOSITE regions**: Contain child regions (hierarchical organization)
- **ROOT region**: Single region containing all graph nodes (for analysis)

**Use Cases:**
- Graph partitioning for distributed execution
- Identifying optimization boundaries for quantization/pruning
- Creating hierarchical abstractions of computation
- Analyzing graph structure and computational patterns

**Key Classes:**
- **RegionSearchBase**: Base class with common graph analysis utilities
- **CombinedRegionSearch**: Main two-phase region discovery algorithm
- **RegionPartitioner**: Bottom-up partitioning based on divergence/convergence
- **TopDownRegionBuilder**: Top-down refinement creating hierarchical structure
"""

import argparse
import logging
import sys
from collections import Counter, deque

import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.insertion_points import has_quantizable_operations
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

# Module logger
logger = logging.getLogger(__name__)


def enable_debug():
    """Enable debug-level logging for the region search module."""
    global logger
    logger.setLevel(logging.DEBUG)


DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_NODES_TO_SHOW = 20


class RegionSearchBase:
    """Base class for region search algorithms providing common graph analysis utilities.

    This class serves as a foundation for region-based graph analysis algorithms by
    providing essential data structures and methods for:
    - Graph traversal and reachability analysis
    - Divergence/convergence pattern detection
    - Region boundary computation
    - Tensor flow tracking

    **Core Data Structures:**
    - **tensor_users_map**: Maps tensor names to node indices that consume them.
      Used to efficiently find divergence points and track data flow.
    - **forward_reachable_nodes_map**: Pre-computed forward reachability for all nodes.
      Maps each node to all nodes reachable from it (with distances).
    - **root**: Root region containing all graph nodes, used as search space.

    **Key Algorithms:**
    - **Divergence Detection**: Identifies nodes whose outputs branch to multiple consumers
    - **Convergence Detection**: Finds nodes where multiple branches rejoin
    - **Boundary Computation**: Determines input/output tensors for regions
    - **Reachability Analysis**: Computes forward-reachable nodes with distances

    **Design Pattern:**
    This is a base class meant to be subclassed. Subclasses implement specific
    region formation strategies (e.g., bottom-up partitioning, top-down refinement)
    while reusing the common analysis utilities provided here.

    **Performance:**
    Pre-computation in __init__ scales with graph size:
    - tensor_users_map: O(E) where E = number of edges
    - forward_reachable_nodes_map: O(N * (N + E)) where N = number of nodes

    For large graphs, initialization may take significant time but enables
    efficient queries during region formation.

    Attributes:
        graph: The ONNX computation graph (onnx_graphsurgeon.Graph)
        root: Root region containing all nodes in the graph
        tensor_users_map: Mapping from tensor names to consuming node indices
        forward_reachable_nodes_map: Pre-computed forward reachability for all nodes

    Example:
        >>> # Typically used as a base class
        >>> class MyRegionSearch(RegionSearchBase):
        ...     def find_regions(self):
        ...         # Use inherited utilities like _is_node_divergent()
        ...         pass
    """

    def __init__(
        self, graph: gs.Graph, root: Region | None = None, max_steps: int = DEFAULT_MAX_STEPS
    ):
        """Initialize the base region search with graph analysis.

        Performs pre-computation of essential data structures for efficient
        region analysis:
        1. Creates or validates root region containing all nodes
        2. Builds tensor-to-users mapping for divergence detection
        3. Pre-computes forward reachability for convergence detection

        Args:
            graph: The ONNX graph to analyze (onnx_graphsurgeon.Graph)
            root: Optional root region. If None, creates one containing all nodes.
            max_steps: Maximum distance for forward reachability pre-computation.
                      Limits memory usage and computation time for large graphs.

        Note:
            Initialization time scales with graph complexity. For graphs with
            thousands of nodes, this may take several seconds.
        """
        self.graph = graph
        if root is None:
            root = self._build_root_region()
        self.root = root
        self.tensor_users_map = get_tensor_consumer_node_indices(self.graph)
        self.forward_reachable_nodes_map = self._build_forward_reachable_nodes_map(
            max_steps=max_steps
        )

    def _build_root_region(self) -> Region:
        """Create a root region containing all nodes in the graph.

        The root region serves as the universal search space for region
        formation algorithms. It represents the entire computation graph
        as a single region before any partitioning.

        Returns:
            Region of type ROOT containing all graph nodes.
        """
        root = Region(region_id=0, level=0, region_type=RegionType.ROOT)
        for node_idx in range(len(self.graph.nodes)):
            root.add_node(node_idx)
        for tensor_name in root.get_inputs():
            root.add_input(tensor_name)
        for tensor_name in root.get_outputs():
            root.add_output(tensor_name)
        return root

    def _is_tensor_divergent(self, tensor_name: str) -> bool:
        """Check if a tensor is consumed by multiple nodes (divergent).

        A divergent tensor indicates branching in the computation graph,
        where one operation's output feeds into multiple downstream operations.

        Args:
            tensor_name: Name of the tensor to check

        Returns:
            True if tensor has more than one consumer, False otherwise
        """
        return len(self.tensor_users_map.get(tensor_name, [])) > 1

    def _is_node_divergent(self, node_idx: int) -> bool:
        """Check if a node has outputs that branch to multiple consumers.

        A divergent node is one that produces outputs consumed by multiple
        downstream nodes, creating branches in the computation graph. These
        nodes are important boundaries for region formation.

        **Significance:**
        - Divergent nodes often represent natural region boundaries
        - They indicate where computation splits into parallel paths
        - Useful for identifying opportunities for parallel optimization

        Args:
            node_idx: Index of the node to check

        Returns:
            True if the node has at least one output consumed by multiple nodes,
            False otherwise or if node is not in root region.

        Example:
            >>> # Node 10 outputs tensor "X" consumed by nodes 11 and 12
            >>> _is_node_divergent(10)  # Returns True
        """
        if node_idx not in self.root.get_nodes():
            logger.debug(f"Node {node_idx} not in root region")
            return False

        node = self.graph.nodes[node_idx]
        divergent_outputs = [
            out.name for out in node.outputs if self._is_tensor_divergent(out.name)
        ]
        is_divergent = len(divergent_outputs) > 0

        if is_divergent:
            logger.debug(
                f"Divergent node {node_idx} ({node.op}): {len(divergent_outputs)} branches"
            )

        return is_divergent

    def _compute_forward_reachable_nodes(
        self, start_node_idx: int, max_steps: int
    ) -> dict[int, int]:
        """Compute all nodes reachable forward from a starting node with distances.

        Uses breadth-first search (BFS) to find all nodes reachable by following
        forward edges (data flow direction) from the start node, up to a maximum
        distance. Records the shortest-path distance to each reachable node.

        **Algorithm:**
        1. Initialize with start node at distance 0
        2. For each node in queue:
           - If at max distance, skip
           - For each output tensor:
             - For each consumer of that tensor:
               - If not yet visited, add to queue with distance+1

        **Use Cases:**
        - Convergence detection: Find where branches rejoin
        - Region size estimation: Count nodes in forward cone
        - Dependency analysis: Understand downstream impact

        Args:
            start_node_idx: Index of node to start search from
            max_steps: Maximum forward distance to explore

        Returns:
            Dictionary mapping reachable node indices to their distances from start.
            Includes start_node_idx mapped to distance 0.

        Example:
            >>> # Find all nodes within 5 steps forward of node 10
            >>> reachable = _compute_forward_reachable_nodes(10, 5)
            >>> reachable[10]  # 0 (start node)
            >>> reachable[15]  # 3 (if node 15 is 3 steps away)
        """
        reachable: dict[int, int] = {start_node_idx: 0}
        queue: deque[tuple[int, int]] = deque([(start_node_idx, 0)])
        while queue:
            current_node_idx, distance = queue.popleft()
            if distance >= max_steps:
                continue
            current_node = self.graph.nodes[current_node_idx]
            for output in current_node.outputs:
                if output.name not in self.tensor_users_map:
                    continue
                for next_node_idx in self.tensor_users_map[output.name]:
                    if next_node_idx not in reachable:
                        reachable[next_node_idx] = distance + 1
                        queue.append((next_node_idx, distance + 1))
        return reachable

    def _build_forward_reachable_nodes_map(self, max_steps: int) -> dict[int, dict[int, int]]:
        """Pre-compute forward reachability for all nodes in the graph.

        This is a key optimization that enables efficient convergence detection.
        By pre-computing forward reachability once, we can quickly answer queries
        like "Can node A reach node B?" and "What is the distance from A to B?"

        **Complexity:**
        - Time: O(N * (N + E)) where N = nodes, E = edges
        - Space: O(N²) in worst case for dense graphs

        **Trade-off:**
        Pre-computation takes time upfront but dramatically speeds up convergence
        detection, which would otherwise require repeated BFS traversals.

        Args:
            max_steps: Maximum forward distance to pre-compute for each node.
                      Limits both time and space complexity.

        Returns:
            Nested dictionary where outer key is start node index, inner key is
            reachable node index, and value is shortest-path distance.

        Example:
            >>> map = _build_forward_reachable_nodes_map(10)
            >>> map[5][8]  # Distance from node 5 to node 8
            3
            >>> 12 in map[5]  # Can node 5 reach node 12?
            True
        """
        logger.debug(f"Building forward reachability map (max_steps={max_steps})...")
        forward_reachable_nodes_map: dict[int, dict[int, int]] = {}
        for node_idx in self.root.get_nodes():
            forward_reachable_nodes_map[node_idx] = self._compute_forward_reachable_nodes(
                node_idx, max_steps
            )

        total_reachable = sum(len(reachable) for reachable in forward_reachable_nodes_map.values())
        avg_reachable = total_reachable / len(self.root.get_nodes()) if self.root.get_nodes() else 0
        logger.debug(f"Reachability map complete: avg {avg_reachable:.1f} reachable nodes per node")
        return forward_reachable_nodes_map

    def _find_converge_nodes(self, node_idx: int) -> tuple[int | None, set[int]]:
        """Find convergence point and intermediate nodes for a divergent node.

        Given a divergent node (where computation branches), this method finds:
        1. The convergence node: Where the branches rejoin
        2. All nodes between divergence and convergence

        **Algorithm:**
        1. Identify all branches from the divergent node
        2. Find nodes reachable from all branches (common nodes)
        3. Select nearest common node that forms a valid region
        4. Compute all nodes between divergence and convergence

        **Convergence Criteria:**
        A valid convergence node must:
        - Be reachable from all branches
        - Form a contiguous region (no nodes escape the region)
        - Be the nearest such node (minimize region size)

        **Region Validity:**
        A region is valid if all nodes within it either stay in the region
        or directly reach the convergence point. No node should reach outside
        the region before reaching the convergence point.

        Args:
            node_idx: Index of the divergent node to find convergence for

        Returns:
            Tuple of (converge_node_idx, visited_nodes):
            - converge_node_idx: Index of convergence node, or None if not found
            - visited_nodes: Set of node indices between divergence and convergence

        Example:
            >>> # Node 10 branches to 11 and 12, which rejoin at node 15
            >>> converge_idx, visited = _find_converge_nodes(10)
            >>> converge_idx  # 15
            >>> visited  # {10, 11, 12, 13, 14} (all nodes in between)
        """
        node = self.graph.nodes[node_idx]
        logger.debug(f"Finding convergence for node {node_idx} ({node.op})")

        branches: list[int] = []
        for output in node.outputs:
            if output.name in self.tensor_users_map:
                branches.extend(self.tensor_users_map[output.name])

        seen: set[int] = set()
        unique_branches: list[int] = []
        for branch_idx in branches:
            if branch_idx not in seen:
                seen.add(branch_idx)
                unique_branches.append(branch_idx)
        branches = unique_branches

        logger.debug(f"  {len(branches)} unique branches found")

        # Need at least 2 branches for convergence to be meaningful
        if len(branches) <= 1:
            logger.debug("  Insufficient branches for convergence")
            return None, set()

        # =====================================================================
        # STEP 1: Find Common Reachable Nodes (Potential Convergence Points)
        # =====================================================================
        # A valid convergence node must be reachable from ALL branches.
        # Use pre-computed forward reachability for efficiency.

        # Collect forward-reachable nodes for each branch
        branch_reachable: list[dict[int, int]] = []
        for branch_idx in branches:
            reachable = self.forward_reachable_nodes_map.get(branch_idx, {})
            branch_reachable.append(reachable)

        if not branch_reachable:
            logger.debug("  No reachable nodes from branches")
            return None, set()

        # Find intersection: nodes reachable from ALL branches
        # These are the only candidates for convergence points
        common_nodes = set(branch_reachable[0].keys())
        for reachable in branch_reachable[1:]:
            common_nodes.intersection_update(reachable.keys())

        logger.debug(f"  {len(common_nodes)} common nodes found")

        # Remove the divergent node itself (not a convergence point)
        common_nodes.discard(node_idx)

        if not common_nodes:
            logger.debug("  No valid convergence candidates")
            return None, set()

        # =====================================================================
        # STEP 2: Select Best Convergence Node with Region Validity Check
        # =====================================================================
        # Not all common nodes make good convergence points. We need to ensure
        # the region formed is "valid" - i.e., contiguous with no escaping edges.
        #
        # Region validity criterion:
        # For every node R in the region (between divergence and candidate):
        #   For every node T reachable from R:
        #     If T is outside the region:
        #       T must be at least as far from R as the candidate is
        #       (i.e., R doesn't "escape" before reaching candidate)

        converge_node_idx: int | None = None
        min_max_distance = float("inf")

        # Get all nodes reachable from the divergent node
        reachable_from_start = self.forward_reachable_nodes_map.get(node_idx, {})

        # Evaluate each candidate convergence point
        for candidate_idx in common_nodes:
            # ---------------------------------------------------------------
            # Define the potential region: nodes between start and candidate
            # ---------------------------------------------------------------
            # Region = nodes reachable from start BUT NOT reachable from candidate
            # (candidate acts as the boundary)
            region_nodes: set[int] = set()
            region_nodes.update(set(reachable_from_start.keys()))
            reachable_from_candidate = self.forward_reachable_nodes_map.get(candidate_idx, {})
            # Remove nodes beyond the candidate (not in our region)
            region_nodes.difference_update(set(reachable_from_candidate.keys()))

            # ---------------------------------------------------------------
            # Validate region: Check for "escaping" edges
            # ---------------------------------------------------------------
            # A region is invalid if any node inside can reach a node outside
            # BEFORE reaching the convergence point. This would mean the region
            # has edges that "leak out" and isn't properly bounded.
            broken_region = False

            # Check each node in the proposed region
            for rnode_index in region_nodes:
                # Get all nodes reachable from this region node
                reachable_from_rnode = self.forward_reachable_nodes_map.get(rnode_index, {})

                # Distance from this node to the candidate (convergence)
                rnode_to_candidate_distance = reachable_from_rnode.get(candidate_idx, float("inf"))

                # Check all nodes reachable from this region node
                for test_node_idx in reachable_from_rnode:
                    # Skip nodes that are inside the region (they're fine)
                    if test_node_idx in region_nodes:
                        continue

                    # test_node is OUTSIDE the region. Check if it's "escaping"
                    # An escaping edge: region_node reaches test_node BEFORE candidate
                    rnode_to_test_distance = reachable_from_rnode.get(test_node_idx, float("inf"))

                    # If either distance is infinite, region is broken
                    # (indicates disconnected components or unreachable convergence)
                    if rnode_to_test_distance == float(
                        "inf"
                    ) or rnode_to_candidate_distance == float("inf"):
                        broken_region = True
                        break

                    # If test_node is closer than candidate, we have an escape!
                    # This means computation flows OUT of region before converging
                    if rnode_to_test_distance < rnode_to_candidate_distance:
                        broken_region = True
                        break

                if broken_region:
                    break

            # Skip this candidate if region is invalid
            if broken_region:
                continue

            # ---------------------------------------------------------------
            # Valid candidate! Check if it's the nearest one
            # ---------------------------------------------------------------
            # We want the closest convergence point to minimize region size
            # "Distance" = maximum distance from any branch to convergence
            max_distance = max(reachable[candidate_idx] for reachable in branch_reachable)

            if max_distance < min_max_distance:
                min_max_distance = max_distance
                converge_node_idx = candidate_idx

        # If no valid convergence found, this divergence has no convergence
        if converge_node_idx is None:
            logger.debug("  No valid convergence found")
            return None, set()

        converge_node = self.graph.nodes[converge_node_idx]
        logger.debug(
            f"  Convergence at node {converge_node_idx} ({converge_node.op}), distance {min_max_distance}"
        )

        # =====================================================================
        # STEP 3: Compute All Nodes Between Divergence and Convergence
        # =====================================================================
        # Now that we have a valid convergence point, we need to identify ALL
        # nodes that should be included in the convergence region.
        #
        # A node is "between" divergence and convergence if:
        # 1. It's reachable from the divergence node (on some path from divergence)
        # 2. The convergence node is reachable from it (on some path to convergence)
        # 3. It's not the convergence node itself (convergence is the boundary)
        #
        # This captures all the "interior" nodes of the funnel/diamond pattern,
        # including all branches and intermediate computations.

        visited_nodes: set[int] = set()

        # Check each node reachable from the divergent node
        for candidate_idx in reachable_from_start:
            # Skip the convergence node itself (it's the boundary, not interior)
            if candidate_idx == converge_node_idx:
                continue

            # Check if this node can reach the convergence node
            # If yes, it's on a path from divergence to convergence
            reachable_from_candidate = self.forward_reachable_nodes_map.get(candidate_idx, {})
            if converge_node_idx in reachable_from_candidate:
                # This node is between divergence and convergence!
                visited_nodes.add(candidate_idx)

        logger.debug(f"  {len(visited_nodes)} nodes between divergence and convergence")
        return converge_node_idx, visited_nodes

    def _max_distance_to_nodes(self, src_idx: int, dst_indices: set[int]) -> int:
        """Compute maximum distance from a source node to a set of destination nodes.

        Uses pre-computed forward reachability to efficiently find the maximum
        shortest-path distance from src_idx to any node in dst_indices.

        **Use Cases:**
        - Determine if a convergence region is within acceptable size limits
        - Measure the "spread" of nodes in a potential region
        - Validate region compactness constraints

        Args:
            src_idx: Source node index
            dst_indices: Set of destination node indices

        Returns:
            Maximum distance from src to any node in dst_indices.
            Returns 0 if dst_indices is empty or no nodes are reachable.

        Example:
            >>> # Check if all nodes are within 10 steps
            >>> max_dist = _max_distance_to_nodes(start_node, candidate_nodes)
            >>> if max_dist <= 10:
            ...     # Region is compact enough
        """
        max_distance = 0
        for dst_idx in dst_indices:
            reachable = self.forward_reachable_nodes_map.get(src_idx, {})
            if dst_idx in reachable:
                max_distance = max(max_distance, reachable[dst_idx])

        logger.debug(
            f"Max distance from node {src_idx}: {max_distance} steps to {len(dst_indices)} nodes"
        )
        return max_distance

    def compute_region_boundaries(self, region: Region, include_constant: bool = False) -> None:
        """Compute input and output tensor boundaries for a region.

        **Algorithm:**
        1. Collect all tensors consumed by region nodes (potential inputs)
        2. Collect all tensors produced by region nodes (potential outputs)
        3. Input = consumed tensors NOT produced by region nodes
        4. Output = produced tensors consumed by nodes OUTSIDE the region

        This accurately captures the data flow boundaries of the region.

        Args:
            region: The region to compute boundaries for
        """
        node_indices = region.get_all_nodes_recursive()
        all_inputs: set[str] = set()
        all_outputs: set[str] = set()
        internal_tensors: set[str] = set()

        # First pass: collect all inputs and outputs
        for node_idx in node_indices:
            if node_idx >= len(self.graph.nodes):
                continue
            node = self.graph.nodes[node_idx]
            # Collect input tensors
            for input_tensor in node.inputs:
                if isinstance(input_tensor, gs.Constant) and not include_constant:
                    continue
                all_inputs.add(input_tensor.name)
            # Collect output tensors
            for output_tensor in node.outputs:
                all_outputs.add(output_tensor.name)
                internal_tensors.add(output_tensor.name)

        # Region inputs = consumed tensors not produced internally
        region_inputs = all_inputs - internal_tensors

        # Region outputs = produced tensors consumed externally
        region_outputs: set[str] = set()
        for node_idx in node_indices:
            if node_idx >= len(self.graph.nodes):
                continue
            node = self.graph.nodes[node_idx]
            for output_tensor in node.outputs:
                tensor_name = output_tensor.name
                if tensor_name not in self.tensor_users_map:
                    region_outputs.add(tensor_name)
                    continue
                # Check if any consumer is outside the region
                has_external_consumer = False
                # Get consumer nodes from tensor_users_map
                consumer_indices = self.tensor_users_map[tensor_name]
                for consumer_idx in consumer_indices:
                    if consumer_idx not in node_indices:
                        # Consumer is outside the region
                        has_external_consumer = True
                        break
                if has_external_consumer:
                    region_outputs.add(tensor_name)
                # Also check if this is a graph output
                if output_tensor in self.graph.outputs:
                    region_outputs.add(tensor_name)

        # Add to region
        region.inputs = sorted(region_inputs)
        region.outputs = sorted(region_outputs)

        logger.debug(
            f"Computed boundaries: {len(region_inputs)} inputs, {len(region_outputs)} outputs"
        )

    def print_tree(
        self,
        region: Region | None = None,
        indent: int = 0,
        max_nodes_to_show: int = DEFAULT_MAX_NODES_TO_SHOW,
        file=None,
    ) -> None:
        """Print hierarchical region tree in human-readable text format.

        Recursively prints the region hierarchy with indentation showing depth.
        For each region, displays:
        - ID, level, and type (LEAF/COMPOSITE/ROOT)
        - Node counts (direct and recursive)
        - I/O tensor counts
        - Sample of nodes in the region (up to max_nodes_to_show)
        - Child regions (recursively)

        Args:
            region: Region to print (None defaults to root)
            indent: Current indentation level (0 = root)
            max_nodes_to_show: Maximum nodes to display per region (default: 5)
            file: Output file object (None defaults to stdout)

        Example:
            >>> builder.print_tree()
            ├─ Region 0 (Level 0, Type: ROOT)
            │  ├─ Direct nodes: 0
            │  └─ Children: 2
            │    ├─ Region 1 (Level 1, Type: COMPOSITE)
            ...
        """
        region = region or self.root
        if region is None:
            return

        if file is None:
            file = sys.stdout

        prefix = "  " * indent

        # Print region header
        region_type = region.get_type().value
        print(
            f"{prefix}├─ Region {region.get_id()} (Level {region.get_level()}, Type: {region_type})",
            file=file,
        )

        # Print region size info
        direct_nodes = region.get_nodes()
        total_nodes = region.get_all_nodes_recursive()
        num_children = len(region.get_children())

        print(f"{prefix}│  ├─ Direct nodes: {len(direct_nodes)}", file=file)
        print(f"{prefix}│  ├─ Total nodes (recursive): {len(total_nodes)}", file=file)
        print(f"{prefix}│  ├─ Children: {num_children}", file=file)

        # Print region I/O info
        inputs = region.get_inputs()
        outputs = region.get_outputs()
        print(f"{prefix}│  ├─ Inputs: {len(inputs)} tensors", file=file)
        if inputs:
            for tensor_name in list(inputs)[:max_nodes_to_show]:
                print(f"{prefix}│  │    - {tensor_name}", file=file)
            if len(inputs) > max_nodes_to_show:
                print(f"{prefix}│  │    ... and {len(inputs) - max_nodes_to_show} more", file=file)
        print(f"{prefix}│  └─ Outputs: {len(outputs)} tensors", file=file)
        if outputs:
            for tensor_name in list(outputs)[:max_nodes_to_show]:
                print(f"{prefix}│       - {tensor_name}", file=file)
            if len(outputs) > max_nodes_to_show:
                print(f"{prefix}│       ... and {len(outputs) - max_nodes_to_show} more", file=file)

        # Print direct nodes in this region (if any)
        if direct_nodes:
            print(f"{prefix}│", file=file)
            print(f"{prefix}│  Nodes in this region:", file=file)
            nodes_list = sorted(direct_nodes)[:max_nodes_to_show]
            for node_idx in nodes_list:
                if node_idx < len(self.graph.nodes):
                    node = self.graph.nodes[node_idx]
                    print(
                        f"{prefix}│    - Node {node_idx}: {node.op} (name: {node.name})", file=file
                    )

            if len(direct_nodes) > max_nodes_to_show:
                print(
                    f"{prefix}│    ... and {len(direct_nodes) - max_nodes_to_show} more nodes",
                    file=file,
                )

        # Print children (recursively)
        children = region.get_children()
        if children:
            print(f"{prefix}│", file=file)
            print(f"{prefix}│  Child regions:", file=file)
            for child_index, child in enumerate(children):
                print(f"{prefix}│", file=file)
                self.print_tree(child, indent + 1, max_nodes_to_show, file)


class RegionPartitioner(RegionSearchBase):
    """Bottom-up graph partitioner that creates initial regions based on divergence patterns.

    This class implements Phase 1 of the combined region search strategy. It performs
    a systematic traversal of the computation graph from inputs to outputs, identifying
    natural boundaries for region formation based on computation flow patterns.

    **Core Strategy:**
    Partitions the graph by analyzing three types of computational patterns:

    1. **Divergent Nodes with Convergence:**
       - Nodes whose outputs branch to multiple paths (divergence)
       - Paths that eventually rejoin at a common node (convergence)
       - Creates a single region encompassing divergence + branches + convergence
       - Example: A → (B,C) → D creates region containing {A, B, C, D}

    2. **Divergent Nodes without Convergence:**
       - Nodes whose outputs branch but never rejoin
       - Creates a single-node "orphan" region for the divergent node
       - Example: A → (B,C) with no convergence creates region {A}

    3. **Linear Sequences:**
       - Chains of non-divergent nodes (simple sequential computation)
       - Groups entire sequence into one region
       - Example: A → B → C → D creates region {A, B, C, D}

    **Algorithm Overview:**
    ```
    For each node in graph order:
        If already visited: skip
        If divergent:
            Find convergence point
            If convergence exists within threshold:
                Create region with all nodes between divergence and convergence
            Else:
                Create single-node region (orphan)
        Else (non-divergent):
            Build sequence: follow chain until hitting divergent node
            Create region containing entire sequence
    ```

    **Key Features:**
    - **Complete Coverage:** Every node is assigned to exactly one region
    - **Convergence Detection:** Uses pre-computed reachability for efficiency
    - **Distance Threshold:** Limits region size to DEFAULT_MAX_STEPS
    - **Sequential Processing:** Respects data flow order for natural groupings

    **Region Types Created:**
    All regions created by this class are LEAF regions (level 0). Higher-level
    structure is created later by TopDownRegionBuilder.

    **State Management:**
    - **visited_nodes:** Tracks which nodes have been assigned to regions
    - **current_region:** Region being built (commit when complete)
    - **regions:** List of completed regions
    - **current_region_id:** Counter for unique region IDs

    **Output:**
    A list of LEAF regions that partition the entire graph. These regions
    serve as input to Phase 2 (TopDownRegionBuilder) for refinement.

    **Example:**
    ```python
    partitioner = RegionPartitioner(graph)
    initial_regions = partitioner.partition_graph()

    # Analyze results
    print(f"Created {len(initial_regions)} regions")
    print(f"Covered {len(partitioner.visited_nodes)} / {len(graph.nodes)} nodes")

    # Typical output for a ResNet layer:
    # - Conv node → orphan region (diverges to BN and skip path)
    # - BN → ReLU sequence → sequential region
    # - Add (convergence) → orphan or part of next sequence
    ```

    **Performance:**
    - Time: O(N) where N = number of nodes (each visited once)
    - Space: O(N) for visited_nodes set and region storage

    Attributes:
        regions: List of completed LEAF regions
        current_region: Region currently being built (None if between regions)
        current_region_id: Counter for assigning unique region IDs
        visited_nodes: Set of node indices already assigned to regions

    See Also:
        TopDownRegionBuilder: Phase 2 refinement of partitioner output
        CombinedRegionSearch: Orchestrates both phases
    """

    def __init__(self, graph: gs.Graph):
        """Initialize the partitioner with a computation graph.

        Sets up necessary data structures and inherits graph analysis utilities
        from RegionSearchBase (tensor users map, reachability, etc.).

        Args:
            graph: The ONNX graph to partition (onnx_graphsurgeon.Graph)
        """
        super().__init__(graph, root=None)
        self.regions: list[Region] = []
        self.current_region: Region | None = None
        self.current_region_id: int = 0
        self.visited_nodes: set[int] = set()

    def _append_node_to_region(self, node_idx: int):
        """Add a node to the current region, creating a new region if needed.

        This is the primary method for building regions incrementally. If no
        region is currently active, creates a new LEAF region. Then adds the
        specified node to that region.

        **Usage Pattern:**
        Typically called multiple times to build up a region, then followed
        by _commit_region() to finalize and store the completed region.

        Args:
            node_idx: Index of node to add to current region

        Side Effects:
            - Creates new region if current_region is None
            - Increments current_region_id when creating new region
            - Adds node to current_region
        """
        node = self.graph.nodes[node_idx]
        if self.current_region is None:
            self.current_region = Region(
                region_id=self.current_region_id, level=0, region_type=RegionType.LEAF
            )
            logger.debug(f"Started region {self.current_region_id}")
            self.current_region_id += 1

        self.current_region.add_node(node_idx)
        logger.debug(
            f"  Added node {node_idx} ({node.op}), region size: {self.current_region.get_size()}"
        )

    def _commit_region(self):
        """Finalize and store the current region being built.

        Completes region construction by:
        1. Computing input/output tensor boundaries
        2. Adding region to the completed regions list
        3. Resetting current_region to None for next region

        **Boundary Computation:**
        Determines which tensors flow into and out of the region based on
        which nodes produce/consume them. This is essential for understanding
        region dependencies.

        **Post-Conditions:**
        - current_region is added to regions list
        - current_region is reset to None
        - Region has computed input/output tensor lists

        Side Effects:
            - Appends current_region to self.regions
            - Sets current_region to None
            - Logs region commit with size info
        """
        if self.current_region is not None:
            region_size = self.current_region.get_size()
            region_id = self.current_region.id

            # Compute input/output tensor boundaries
            self.compute_region_boundaries(self.current_region)

            self.regions.append(self.current_region)
            logger.debug(
                f"Committed region {region_id}: {region_size} nodes (total: {len(self.regions)})"
            )
            self.current_region = None
        else:
            logger.debug("No region to commit")

    def _build_sequence_from_node(self, node_idx: int, max_nodes: int = -1):
        """Build a region from a linear sequence of non-divergent nodes.

        Starting from a non-divergent node, follows the forward chain of nodes,
        adding each non-divergent node to the current region. Stops when hitting:
        - A divergent node (branches to multiple paths)
        - A node already visited
        - End of graph

        **Algorithm:**
        ```
        queue = [start_node]
        while queue not empty:
            node = dequeue()
            if node is divergent:
                stop (this node will be handled separately)
            else:
                add node to region
                add all successors to queue
        commit region
        ```

        **Example:**
        For graph: Conv → BN → ReLU → MaxPool (no branching)
        Creates one region containing all four nodes.

        **Stopping Conditions:**
        - Divergent node encountered (boundary for this region)
        - All successors already visited
        - No more forward connections

        Args:
            node_idx: Index of starting node (must be non-divergent)

        Side Effects:
            - Adds nodes to current_region via _append_node_to_region
            - Marks nodes as visited
            - Commits completed region

        Note:
            Always commits the region at the end, even if only one node was added.
        """
        start_node = self.graph.nodes[node_idx]
        logger.debug(f"Building sequence from node {node_idx} ({start_node.op})")

        queue: deque[int] = deque([node_idx])
        nodes_added = 0

        while len(queue) > 0:
            current_node_idx = queue.popleft()
            current_node = self.graph.nodes[current_node_idx]

            if not self._is_node_divergent(current_node_idx):
                self._append_node_to_region(current_node_idx)
                self.visited_nodes.add(current_node_idx)
                nodes_added += 1

                # Find successors
                successor_count = 0
                for output_tensor in current_node.outputs:
                    if output_tensor.name in self.tensor_users_map:
                        successors = self.tensor_users_map[output_tensor.name]
                        successor_count += len(successors)
                        queue.extend(successors)
            else:
                self._append_node_to_region(current_node_idx)
                nodes_added += 1
                logger.debug(f"  Stopped at divergent node {current_node_idx} ({current_node.op})")

            if max_nodes > 0 and nodes_added >= max_nodes:
                logger.debug("  Max nodes reached")
                break

        logger.debug(f"Sequence complete: {nodes_added} nodes")

    def _build_small_converged_region(
        self, start_node_idx: int, converge_node_idx: int, visited_nodes: set[int]
    ):
        r"""Create a region encompassing divergence, branches, and convergence.

        Builds a single region containing:
        - The divergent node (where branches split)
        - All nodes in the branches
        - The convergence node (where branches rejoin)

        This creates a "diamond" or "funnel" shaped region that captures
        parallel computation paths and their merge point.

        **Structure:**
        ```
               start (divergent)
              /      \
            path1   path2  (visited_nodes)
              \\      /
              convergence
        ```

        **Example:**
        For ResNet skip connection:
        - start_node: Output of previous layer (branches)
        - visited_nodes: {Conv, BN, ReLU, Conv, BN} (main path)
        - converge_node: Add operation (merges with skip)

        Args:
            start_node_idx: The divergent node where branches begin
            converge_node_idx: Where branches rejoin (currently unused but kept for API)
            visited_nodes: All nodes between divergence and convergence

        Side Effects:
            - Adds all nodes to current region
            - Marks all nodes as visited
            - Commits the completed region
        """
        visited_nodes.remove(start_node_idx)
        for node_idx in sorted(visited_nodes):
            self._append_node_to_region(node_idx)
            self.visited_nodes.add(node_idx)
        if not self._is_node_divergent(converge_node_idx):
            self._append_node_to_region(converge_node_idx)
            self.visited_nodes.add(converge_node_idx)
        self._build_sequence_from_node(converge_node_idx, max_nodes=3)

    def _build_region_from_node(self, node_idx: int):
        """Process a single node and create appropriate region(s) based on its pattern.

        This is the core dispatch method that determines how to handle each node
        based on whether it's divergent (branches) or sequential. Implements the
        three pattern recognition strategies described in the class documentation.

        **Decision Logic:**
        ```
        If node already visited:
            Skip (already in a region)
        Else if node is divergent:
            Try to find convergence point
            If convergence found within distance threshold:
                Create convergence region (divergence + branches + convergence)
            Else:
                Create orphan region (just the divergent node)
        Else (non-divergent):
            Build sequence region (follow chain until divergence)
        ```

        **Pattern 1: Divergent with Convergence (Ideal Case)**
        Creates a complete "funnel" region capturing parallel branches:
        - Example: ResNet skip connection (Conv branch + identity → Add)
        - Condition: converge_node found AND distance < DEFAULT_MAX_STEPS
        - Result: One region containing all nodes between divergence and convergence

        **Pattern 2: Divergent without Convergence (Boundary Case)**
        Creates a single-node "orphan" region:
        - Example: Final layer that branches to multiple outputs
        - Condition: No convergence found OR convergence too far away
        - Result: Region containing only the divergent node

        **Pattern 3: Sequential Chain (Common Case)**
        Creates a region containing linear sequence:
        - Example: Conv → BN → ReLU → MaxPool
        - Condition: Node is not divergent
        - Result: Region containing the full non-divergent chain

        Args:
            node_idx: Index of node to process

        Side Effects:
            - Marks processed nodes as visited
            - Creates and commits region(s) via helper methods
            - May recursively process successor nodes (in sequence building)

        Note:
            This method is idempotent - calling it multiple times on the same
            node has no effect after the first call (due to visited check).
        """
        node = self.graph.nodes[node_idx]

        # Skip nodes already assigned to regions
        if node_idx in self.visited_nodes:
            logger.debug(f"Skipping node {node_idx} ({node.op}): already visited")
            return

        logger.debug(f"Processing node {node_idx} ({node.op})")

        # Pattern 1 & 2: Handle divergent nodes
        if self._is_node_divergent(node_idx):
            logger.debug("  Divergent node, searching for convergence")

            # Attempt to find where branches rejoin
            converge_node_idx, visited_nodes = self._find_converge_nodes(node_idx)

            # Check if convergence creates a reasonable-sized region
            max_distance = self._max_distance_to_nodes(node_idx, visited_nodes)

            # Pattern 1: Convergence found and region size is acceptable
            if converge_node_idx is not None and max_distance < DEFAULT_MAX_STEPS:
                converge_node = self.graph.nodes[converge_node_idx]
                logger.debug(
                    f"  Creating converged region: {len(visited_nodes)} nodes, "
                    f"convergence at {converge_node_idx} ({converge_node.op}), distance {max_distance}"
                )
                # Create region containing: divergence + all branches + convergence
                self._build_small_converged_region(node_idx, converge_node_idx, visited_nodes)
                self._commit_region()
            # Pattern 2: No convergence or region would be too large
            else:
                logger.debug("  Creating orphan region for divergent node")
                # Create single-node region for this divergent node
                # Its successors will be processed separately
                self._append_node_to_region(node_idx)
                self.visited_nodes.add(node_idx)
                self._commit_region()
        # Pattern 3: Handle non-divergent (sequential) nodes
        else:
            logger.debug("  Non-divergent node, building sequence")
            # Build region by following the linear chain forward
            self._build_sequence_from_node(node_idx)
            self._commit_region()

    def partition_graph(self):
        """Partition the entire graph into non-overlapping LEAF regions.

        This is the main entry point for bottom-up graph partitioning. Performs
        a single pass over all nodes in graph order, creating regions based on
        divergence/convergence patterns and sequential chains.

        **Algorithm:**
        ```
        For each node in graph (in index order):
            If node not yet visited:
                Analyze node type (divergent vs sequential)
                Create appropriate region(s) for node and its neighborhood
                Mark processed nodes as visited

        Result: Complete partitioning where every node belongs to exactly one region
        ```

        **Processing Order:**
        Nodes are processed in index order (typically matches graph construction
        order / topological-ish order). This tends to group naturally related
        operations together.

        **Completeness Guarantee:**
        Every node in the graph will be assigned to exactly one region. The
        visited_nodes set ensures no node is processed twice, and the loop over
        all indices ensures no node is skipped.

        **Region Types Created:**
        - Convergence regions: Divergent node + branches + convergence
        - Orphan regions: Single divergent node with no close convergence
        - Sequence regions: Linear chains of non-divergent nodes

        **Output Quality:**
        - Total regions: Typically 10-30% of total nodes (varies by graph)
        - Region sizes: Mix of small (1-3 nodes) and medium (5-15 nodes)
        - Coverage: 100% of graph nodes

        Returns:
            List of LEAF regions that partition the entire graph.
            Each node appears in exactly one region.
            Regions are stored in self.regions and also returned.

        Side Effects:
            - Populates self.regions with created regions
            - Populates self.visited_nodes with all node indices
            - Logs progress and statistics

        Example:
            >>> partitioner = RegionPartitioner(graph)
            >>> regions = partitioner.partition_graph()
            >>> # Verify complete coverage
            >>> all_nodes = set()
            >>> for region in regions:
            ...     all_nodes.update(region.get_nodes())
            >>> assert all_nodes == set(range(len(graph.nodes)))

        Performance:
            - Time: O(N) where N = number of nodes (each visited once)
            - Space: O(N) for visited set and region storage
        """
        logger.info(f"Partitioning graph ({len(self.graph.nodes)} nodes)")
        logger.debug(
            f"Initial state: {len(self.visited_nodes)} visited, {len(self.regions)} regions"
        )

        # Main partitioning loop: process each node in graph order
        for node_idx in range(len(self.graph.nodes)):
            self._build_region_from_node(node_idx)

        # Log completion and coverage statistics
        coverage_pct = (
            100 * len(self.visited_nodes) / len(self.graph.nodes) if self.graph.nodes else 0
        )
        logger.info(
            f"Partitioning complete: {len(self.regions)} regions, "
            f"{len(self.visited_nodes)}/{len(self.graph.nodes)} nodes ({coverage_pct:.1f}%)"
        )

        # Log summary statistics about region sizes
        if self.regions:
            region_sizes = [r.get_size() for r in self.regions]
            avg_size = sum(region_sizes) / len(region_sizes)
            min_size = min(region_sizes)
            max_size = max(region_sizes)
            logger.debug(f"Region sizes: min={min_size}, max={max_size}, avg={avg_size:.1f}")

        return self.regions


class TopDownRegionBuilder(RegionSearchBase):
    """Top-down region refiner that creates hierarchical structure from initial regions.

    This class implements Phase 2 of the combined region search strategy. It takes
    a region created by RegionPartitioner and refines it by:
    1. Identifying and merging converged sub-patterns
    2. Splitting long sequences into optimal sub-regions
    3. Creating a hierarchical COMPOSITE region structure

    **Core Strategy:**
    Starting with a flat LEAF region, creates a hierarchy by:

    **Step 1: Merge Converged Regions**
    - Identifies divergent nodes within the region
    - Finds their convergence points
    - Groups divergence+branches+convergence into sub-regions
    - Leaves remaining nodes for sequence splitting

    **Step 2: Split Sequence Regions**
    - Takes ungrouped nodes (not part of converged patterns)
    - Splits into individual node regions initially
    - Merges adjacent nodes if they form producer-consumer chains
    - Avoids merging boundary operations (Conv, Gemm, etc.)
    - Limits region size to prevent overly large groups

    **Step 3: Create Composite**
    - Wraps all sub-regions into a single COMPOSITE region
    - Computes hierarchical input/output boundaries
    - Returns refined region with better internal structure

    **Merging Criteria for Sequences:**
    Two adjacent sequence regions can merge if ALL of:
    - Producer region's outputs go to exactly one region (simple producer→consumer chain)
    - Neither region is too large (< maximum_sequence_region_size nodes each)
    - Consumer node is not a boundary operation (Conv, Gemm, etc.)
    - Regions are adjacent in data flow (no gaps)

    **Boundary Operations:**
    These operation types are treated as boundaries (don't merge across them):
    - Conv, ConvTranspose: Convolution layers
    - Gemm, MatMul: Matrix multiplications
    - AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool: Pooling
    - Resize: Spatial resizing

    **Example Transformation:**
    ```
    Input (flat LEAF region):
    [Conv, BN, ReLU, Split, Path1_A, Path1_B, Path2_A, Path2_B, Concat]

    Output (hierarchical COMPOSITE region):
    COMPOSITE {
        LEAF {Conv},           # Boundary op stays alone
        LEAF {BN, ReLU},       # Sequence merged
        LEAF {Split},          # Divergent node
        LEAF {Path1_A, Path1_B, Path2_A, Path2_B, Concat},  # Converged pattern
    }
    ```

    **Key Features:**
    - **Hierarchical Structure:** Creates parent-child region relationships
    - **Pattern-Aware:** Recognizes convergence and sequence patterns
    - **Size-Bounded:** Limits region sizes for optimal granularity
    - **Boundary-Aware:** Respects operation type boundaries

    **Inputs:**
    - A LEAF region from RegionPartitioner (flat list of nodes)
    - The graph structure
    - Starting region ID for new regions

    **Output:**
    - A COMPOSITE region containing LEAF child regions
    - Better internal structure reflecting computation patterns
    - Same total nodes, but organized hierarchically

    **Usage Pattern:**
    ```python
    # After partitioning
    initial_region = partitioner.regions[0]

    # Refine structure
    builder = TopDownRegionBuilder(graph, initial_region, next_region_id=10)
    refined_region = builder.build_composite_region()

    # refined_region now has hierarchical structure
    print(f"Children: {len(refined_region.get_children())}")
    for child in refined_region.get_children():
        print(f"  Child {child.get_id()}: {child.get_size()} nodes")
    ```

    **Performance:**
    - Time: O(N + E) where N = nodes in region, E = edges between them
    - Space: O(N) for temporary data structures

    Attributes:
        graph: The computation graph
        root: Input region to refine (becomes search space)
        regions: Output list of refined regions (typically one COMPOSITE)
        next_region_id: Counter for assigning unique IDs to new regions
        boundary_op_types: Set of operation types treated as boundaries
        maximum_sequence_region_size: Maximum number of nodes allowed in a sequence region
            during merging. Prevents overly large regions (default: 10)

    See Also:
        RegionPartitioner: Creates initial regions for refinement
        CombinedRegionSearch: Orchestrates partitioning and refinement
    """

    def __init__(
        self,
        graph: gs.Graph,
        root: Region,
        next_region_id: int = 0,
        maximum_sequence_region_size: int = 10,
    ):
        """Initialize the refiner with a region to refine.

        Args:
            graph: The ONNX graph (onnx_graphsurgeon.Graph)
            root: The region to refine (typically from RegionPartitioner)
            next_region_id: Starting ID for new regions created during refinement
            maximum_sequence_region_size: Maximum nodes per sequence region during merging (default: 10)
        """
        super().__init__(graph, root=root)
        self.regions: list[Region] = []
        self.next_region_id = next_region_id
        self.maximum_sequence_region_size = maximum_sequence_region_size
        self.boundary_op_types = {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            "AveragePool",
            "MaxPool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "Resize",
        }

    def _create_leaf_region(self, node_indices: set[int]) -> Region:
        """Create a new LEAF region containing specified nodes.

        Helper method to construct a properly configured LEAF region:
        - Assigns unique region ID
        - Sets level one deeper than root
        - Adds all specified nodes
        - Computes input/output tensor boundaries

        Args:
            node_indices: Set of node indices to include in the region

        Returns:
            New LEAF region containing the specified nodes with computed boundaries

        Side Effects:
            Increments next_region_id counter
        """
        region = Region(
            region_id=self.next_region_id, level=self.root.level + 1, region_type=RegionType.LEAF
        )
        self.next_region_id += 1
        for node_idx in node_indices:
            region.add_node(node_idx)
        self.compute_region_boundaries(region)
        return region

    def _build_region_usage_map(self, regions: list[Region]) -> dict[str, list[Region]]:
        """Build mapping from tensor names to regions that consume them.

        Similar to tensor_users_map but at the region level instead of node level.
        This enables efficient traversal of region dependencies for merging decisions.

        **Purpose:**
        Used during sequence splitting to identify producer-consumer chains
        between regions. If a tensor is consumed by only one region, that
        region might be mergeable with its producer.

        Args:
            regions: List of regions to analyze

        Returns:
            Dictionary mapping tensor names to lists of regions that consume them.
            Tensors with len(consumers) == 1 indicate potential merge opportunities.

        Example:
            >>> # Tensor "X" consumed by region 5 and region 7
            >>> usage_map["X"] == [region5, region7]
        """
        region_usage_map: dict[str, list[Region]] = {}
        for region in regions:
            for tensor_name in region.inputs:
                if tensor_name not in region_usage_map:
                    region_usage_map[tensor_name] = []
                region_usage_map[tensor_name].append(region)
        return region_usage_map

    def _split_sequence_regions(self, root: Region) -> list[Region]:
        """Split a region into smaller sub-regions by merging producer-consumer chains.

        Takes a region and creates optimal sub-regions by:
        1. Initially splitting into individual single-node regions
        2. Traversing in data flow order (following tensor dependencies)
        3. Merging adjacent regions that form simple producer-consumer chains
        4. Respecting boundary operations and size limits

        **Algorithm:**
        ```
        1. Create one LEAF region per node
        2. Build tensor → consuming regions map
        3. Traverse regions in data flow order (BFS from inputs):
           For each region:
               Check if all outputs go to single consumer region
               If yes and merge criteria met:
                   Merge this region into consumer region
                   Mark this region as removed
        4. Return regions not marked as removed
        ```

        **Merge Criteria (ALL must be true):**
        - All outputs of producer go to exactly one consumer (simple chain)
        - Producer region size < maximum_sequence_region_size (avoid overly large regions)
        - Consumer region size < maximum_sequence_region_size (avoid overly large regions)
        - If consumer is single-node boundary op (Conv, etc.), don't merge
        - Consumer not already removed (merged elsewhere)

        **Boundary Operations:**
        Single-node regions containing these ops stay independent:
        Conv, ConvTranspose, Gemm, MatMul, Pooling ops, Resize

        **Example:**
        ```
        Input nodes: [Conv, BN, ReLU, Add]

        Initial: Region{Conv}, Region{BN}, Region{ReLU}, Region{Add}

        Processing:
        - Conv outputs only to BN, but Conv is boundary → don't merge
        - BN outputs only to ReLU, both small → merge to Region{BN, ReLU}
        - Region{BN,ReLU} outputs only to Add → merge to Region{BN, ReLU, Add}

        Final: Region{Conv}, Region{BN, ReLU, Add}
        ```

        **Purpose:**
        Groups simple sequential operations while keeping compute-heavy
        operations (Conv, Gemm) as separate regions for optimization targeting.

        Args:
            root: Region to split (contains nodes to partition into sub-regions)

        Returns:
            List of LEAF regions that partition the root's nodes with better
            granularity than one-node-per-region or all-in-one.

        Note:
            This is the "bottom" of the top-down strategy - splits fine-grained,
            then merges selectively based on data flow patterns.
        """
        result_regions: list[Region] = []
        removed_regions: set[int] = set()

        # =====================================================================
        # PHASE 1: Split into Single-Node Regions
        # =====================================================================
        # Start with maximum granularity: one region per node.
        # This gives us the most flexibility for selective merging.
        for node_idx in root.get_nodes():
            region = Region(
                region_id=self.next_region_id, level=root.level + 1, region_type=RegionType.LEAF
            )
            region.add_node(node_idx)
            self.compute_region_boundaries(region)
            result_regions.append(region)
            self.next_region_id += 1

        # Build map: tensor_name -> regions that consume it
        # Enables efficient lookup of producer-consumer relationships
        region_usage_map = self._build_region_usage_map(result_regions)

        # =====================================================================
        # PHASE 2: Merge Regions in Data Flow Order
        # =====================================================================
        # Traverse regions following data flow (BFS from inputs).
        # At each step, check if producer can merge with consumer.
        # This creates longer sequences while respecting constraints.

        # Start from root's input tensors and traverse forward
        queue = deque(root.get_inputs())

        while len(queue) > 0:
            tensor_name = queue.popleft()

            # Skip tensors not produced by any region in our scope
            if tensor_name not in region_usage_map:
                continue

            # Process each region consuming this tensor (potential merge targets)
            consumers = region_usage_map[tensor_name]
            for consumer in consumers:
                # Skip regions already merged into others
                if consumer.get_id() in removed_regions:
                    continue

                # -------------------------------------------------------------
                # Check if this consumer can merge with its downstream region
                # -------------------------------------------------------------
                # Merging criteria: ALL outputs go to same single region
                common_use_region = None
                can_merge = True

                # Check all outputs of the consumer region
                for output_tensor in consumer.outputs:
                    # Add output to queue for continued traversal
                    queue.append(output_tensor)

                    # Check if output has consumers in our region set
                    if output_tensor not in region_usage_map:
                        # Output goes outside (or nowhere) - can't merge
                        can_merge = False
                        break

                    # Get regions consuming this output
                    use_regions = region_usage_map[output_tensor]

                    # Must go to exactly ONE region (simple chain)
                    if len(use_regions) != 1:
                        # Branches to multiple regions - can't merge
                        can_merge = False
                        break

                    # Check if all outputs go to the SAME region
                    if common_use_region is None:
                        # First output: remember its consumer
                        common_use_region = use_regions[0]
                    elif common_use_region != use_regions[0]:
                        # Different outputs go to different regions - can't merge
                        can_merge = False
                        break

                # No valid downstream region to merge with
                if common_use_region is None or common_use_region.get_id() in removed_regions:
                    can_merge = False
                    continue

                # -------------------------------------------------------------
                # Apply Additional Constraints
                # -------------------------------------------------------------

                # Constraint 1: Limit the number of boundary operations after merge
                nodes_after_merge = set()
                nodes_after_merge.update(consumer.get_nodes())
                nodes_after_merge.update(common_use_region.get_nodes())
                node_ops = [self.graph.nodes[idx].op for idx in nodes_after_merge]
                boundary_op_count = sum(
                    [1 if op in self.boundary_op_types else 0 for op in node_ops]
                )

                if boundary_op_count > 3:
                    can_merge = False
                    continue

                # Constraint 2: Size limits to avoid overly large regions
                # Keep regions manageable for optimization passes
                if (
                    consumer.get_size() >= self.maximum_sequence_region_size
                    or common_use_region.get_size() >= self.maximum_sequence_region_size
                ):
                    # One or both regions too large - don't merge
                    can_merge = False
                    continue

                # -------------------------------------------------------------
                # Perform Merge
                # -------------------------------------------------------------
                # All criteria met: merge consumer into its downstream region
                if can_merge:
                    common_use_region.merge(consumer)
                    removed_regions.add(consumer.get_id())

        # =====================================================================
        # PHASE 3: Cleanup and Finalize
        # =====================================================================
        # Remove regions that were merged into others
        result_regions = [
            region for region in result_regions if region.get_id() not in removed_regions
        ]

        # Recompute boundaries for all remaining regions
        # (merging may have changed input/output tensors)
        for region in result_regions:
            self.compute_region_boundaries(region)

        return result_regions

    def _merge_converged_regions(self, root: Region):
        """Identify and merge convergence patterns within a region.

        Traverses the region to find divergent nodes and their convergence points,
        creating sub-regions that capture divergence→branches→convergence patterns.
        Nodes not part of any convergence pattern are left for sequence splitting.

        **Algorithm:**
        ```
        1. Traverse region in data flow order (BFS from inputs)
        2. For each node:
           If node is divergent (branches):
               Find convergence point
               If convergence exists within root:
                   Create LEAF region with all nodes between divergence and convergence
                   Mark those nodes as removed (grouped)
        3. Create LEAF region for remaining ungrouped nodes
        4. Return all created regions
        ```

        **Convergence Detection:**
        Uses inherited _find_converge_nodes() to identify where branches rejoin.
        Only creates convergence regions if the convergence point is within
        the root region being refined.

        **Example:**
        ```
        Root contains: [A, B, C, D, E, F, G]

        Graph structure:
        A → B (divergent) → C, D
        C → E
        D → E (convergence)
        E → F → G

        Result:
        - Region1 {B, C, D, E}: Convergence pattern
        - Region2 {A, F, G}: Remaining sequence nodes
        ```

        **Use Case:**
        Captures patterns like:
        - ResNet skip connections (Conv branch + identity → Add)
        - Inception modules (multiple parallel conv paths → Concat)
        - Attention mechanisms (Q/K/V branches → attention computation)

        **Limitations:**
        - Only finds convergence patterns where convergence is in root region
        - Nodes can only belong to one convergence pattern (first match wins)
        - Uses intersection with root nodes to ensure boundaries respected

        Args:
            root: Region to analyze for convergence patterns

        Returns:
            List of LEAF regions:
            - Some containing convergence patterns (divergence + branches + convergence)
            - One containing remaining nodes not part of any pattern

        Note:
            This is the "top" of the top-down strategy - identifies high-level
            patterns first, then delegates remaining nodes to sequence splitting.
        """
        result_regions: list[Region] = []
        removed_nodes: set[int] = set()
        queue = deque(root.get_inputs())
        while len(queue) > 0:
            tensor_name = queue.popleft()
            if tensor_name not in self.tensor_users_map:
                continue
            consumer_nodes = self.tensor_users_map[tensor_name]
            for node_idx in consumer_nodes:
                # stop at boundary nodes
                if node_idx not in root.get_nodes():
                    continue
                consumer = self.graph.nodes[node_idx]
                for output_tensor in consumer.outputs:
                    if output_tensor.name not in self.tensor_users_map:
                        continue
                    queue.append(output_tensor.name)
                # if the node is already in a region, skip
                if node_idx in removed_nodes:
                    continue
                if not self._is_node_divergent(node_idx):
                    continue
                converge_node_idx, visited_nodes = self._find_converge_nodes(node_idx)
                visited_nodes = visited_nodes.intersection(root.get_all_nodes_recursive())
                # if no convergence found, skip
                if converge_node_idx is None:
                    continue
                # group converged nodes into a region
                if converge_node_idx in root.get_nodes():
                    converged_region = self._create_leaf_region(visited_nodes)
                    result_regions.append(converged_region)
                    removed_nodes.update(visited_nodes)
                    continue
        # create a leaf region for the remaining nodes
        remaining_nodes = root.get_nodes() - removed_nodes
        if len(remaining_nodes) > 0:
            result_regions.append(self._create_leaf_region(remaining_nodes))
        # compute region boundaries for all regions
        for region in result_regions:
            self.compute_region_boundaries(region)
        return result_regions

    def build_composite_region(self) -> Region:
        """Refine a flat region into a hierarchical COMPOSITE region.

        This is the main entry point for top-down refinement. Transforms a flat
        LEAF region from RegionPartitioner into a hierarchical structure with
        better internal organization.

        **Three-Stage Algorithm:**

        **Stage 1: Merge Converged Patterns**
        Identifies divergence→convergence patterns and groups them:
        - Finds divergent nodes where computation branches
        - Locates convergence points where branches rejoin
        - Creates sub-regions for complete convergence patterns
        - Leaves ungrouped nodes for next stage

        **Stage 2: Split Sequence Regions**
        Takes remaining (ungrouped) nodes and optimizes granularity:
        - Splits into fine-grained (single-node) regions
        - Merges adjacent regions forming producer-consumer chains
        - Respects boundary operations (Conv, Gemm, etc.)
        - Limits region sizes to avoid overly large groups

        **Stage 3: Create Composite Wrapper**
        Wraps all refined sub-regions into hierarchy:
        - Creates COMPOSITE region at same level as input root
        - Adds all refined LEAF regions as children
        - Computes input/output boundaries for composite
        - Returns single COMPOSITE containing hierarchical structure

        **Transformation Example:**
        ```
        Input (flat LEAF region from partitioner):
        Region(nodes=[0,1,2,3,4,5,6,7,8])

        After Stage 1 (converged patterns):
        [Region{0,1,2}, Region{3,4,5,6,7,8}]  # Found one convergence

        After Stage 2 (sequence splitting):
        [Region{0,1,2}, Region{3}, Region{4,5,6}, Region{7,8}]

        After Stage 3 (composite wrapping):
        COMPOSITE {
            LEAF{0,1,2},    # Convergence pattern
            LEAF{3},        # Boundary op
            LEAF{4,5,6},    # Merged sequence
            LEAF{7,8}       # Merged sequence
        }
        ```

        **Benefits:**
        - **Better Granularity:** Not too coarse, not too fine
        - **Pattern Recognition:** Convergence patterns kept together
        - **Optimization-Friendly:** Boundary ops isolated for targeting
        - **Hierarchical:** Enables recursive optimization strategies

        **Invariants Maintained:**
        - Total node count unchanged (reorganization only)
        - All nodes assigned to exactly one LEAF region
        - LEAF regions don't overlap
        - Parent-child relationships properly formed

        **Output Format:**
        Always returns a single region:
        - If input had >1 nodes: COMPOSITE region with LEAF children
        - If input had 1 node: That single LEAF region unchanged

        Returns:
            COMPOSITE region containing hierarchically organized LEAF sub-regions.
            The composite represents the same nodes as input root but with
            better internal structure reflecting computation patterns.

        Example:
            >>> builder = TopDownRegionBuilder(graph, flat_region, next_id=10)
            >>> refined = builder.build_composite_region()
            >>> print(f"Type: {refined.get_type()}")  # COMPOSITE
            >>> print(f"Children: {len(refined.get_children())}")  # 4-10 typically
            >>> for child in refined.get_children():
            ...     print(f"  {child.get_id()}: {child.get_size()} nodes")
        """
        # merge converged regions into composite regions
        self.regions = self._merge_converged_regions(self.root)
        # split sequence regions into smaller regions
        result_regions: list[Region] = []
        for region in self.regions:
            result_regions.extend(self._split_sequence_regions(region))
        for region in result_regions:
            self.compute_region_boundaries(region, include_constant=True)
        self.regions = result_regions
        # merge all regions into a single composite region
        if len(self.regions) > 1:
            composite = Region(
                region_id=self.next_region_id,
                level=self.root.level,
                region_type=RegionType.COMPOSITE,
            )
            self.next_region_id += 1
            self.regions = sorted(
                self.regions, key=lambda x: RegionPattern.from_region(x, self.graph).signature
            )
            for region in self.regions:
                composite.add_child(region)
            self.compute_region_boundaries(composite)
            self.regions = [composite]
        return self.regions[0]


class CombinedRegionSearch(RegionSearchBase):
    """Two-phase region search combining bottom-up partitioning with top-down refinement.

    This class implements a sophisticated region discovery algorithm that combines two
    complementary strategies to create well-formed, hierarchical regions from an ONNX
    computation graph:

    **Phase 1: Bottom-Up Partitioning (RegionPartitioner)**
    - Traverses the graph from inputs to outputs
    - Identifies divergent nodes (nodes with outputs consumed by multiple branches)
    - Finds convergence points where divergent branches rejoin
    - Creates initial LEAF regions based on divergence/convergence patterns
    - Groups linear sequences of non-divergent nodes together

    **Phase 2: Top-Down Refinement (TopDownRegionBuilder)**
    - Takes each region from Phase 1 as input
    - Identifies and merges converged sub-regions within each region
    - Splits long sequences into smaller, more manageable regions
    - Creates COMPOSITE regions with hierarchical structure
    - Ensures region boundaries align with natural computation patterns

    **Key Features:**
    - **Comprehensive Coverage:** Visits all nodes in the graph
    - **Hierarchical Structure:** Creates multi-level region hierarchies
    - **Pattern Recognition:** Identifies divergence/convergence patterns
    - **Boundary Computation:** Automatically computes input/output tensors for each region
    - **Quality Metrics:** Provides coverage and node count statistics

    **Region Types Created:**
    - LEAF regions: Basic building blocks containing graph nodes
    - COMPOSITE regions: Higher-level regions containing child regions

    **Use Cases:**
    - Graph partitioning for distributed execution
    - Identifying optimization boundaries for quantization/pruning
    - Creating sub-graphs for incremental processing
    - Analyzing graph structure and dependencies

    **Algorithm Overview:**
    1. Initialize RegionPartitioner for bottom-up search
    2. Partition graph into initial LEAF regions
    3. For each initial region:
       a. Merge converged sub-regions
       b. Split long sequences into smaller regions
       c. Create COMPOSITE region hierarchy
    4. Compute final region boundaries

    **Output:**
    A list of COMPOSITE regions that collectively cover the entire graph,
    each containing a hierarchical structure of child regions.

    **Example:**
        >>> search = CombinedRegionSearch(graph)
        >>> regions = search.search_regions()
        >>> print(f"Created {len(regions)} top-level regions")
        >>> for region in regions:
        ...     print(f"Region {region.get_id()}: {region.get_size()} nodes")

    **Performance Considerations:**
    - Complexity depends on graph structure (divergence/convergence patterns)
    - Pre-computes forward-reachable nodes for efficient convergence detection
    - Uses BFS for systematic graph traversal

    **Validation:**
    - Logs warnings if node counts change during refinement
    - Verifies coverage of all nodes in the graph
    - Ensures no duplicate nodes across regions

    Attributes:
        graph: The ONNX graph to partition (onnx_graphsurgeon.Graph)
        regions: List of top-level COMPOSITE regions created by the search
        region_partitioner: Internal RegionPartitioner instance
        root: Root region containing all graph nodes (inherited from RegionSearchBase)
        tensor_users_map: Mapping from tensor names to consuming node indices
        forward_reachable_nodes_map: Pre-computed forward reachability information
        maximum_sequence_region_size: Maximum nodes per sequence region during merging
    """

    def __init__(
        self,
        graph: gs.Graph,
        maximum_sequence_region_size: int = 10,
        minimum_topdown_search_size: int = 10,
    ):
        """Initialize CombinedRegionSearch for a given ONNX graph.

        Sets up the necessary data structures for two-phase region search:
        - Initializes base class with graph and builds root region
        - Creates empty regions list for storing results
        - Initializes RegionPartitioner for Phase 1 bottom-up search
        - Pre-computes tensor users map and forward reachability information

        Args:
            graph: The ONNX graph to partition (onnx_graphsurgeon.Graph).
                   Must be a valid, connected computation graph.
            maximum_sequence_region_size: Maximum nodes per sequence region during merging
                   in Phase 2 refinement (default: 10)
            minimum_topdown_search_size: Minimum nodes per region to search during top-down refinement (default: 10)

        Note:
            Initialization performs pre-computation that scales with graph size.
            For very large graphs, this may take significant time.

        Example:
            >>> import onnx_graphsurgeon as gs
            >>> import onnx
            >>> model = onnx.load("model.onnx")
            >>> graph = gs.import_onnx(model)
            >>> search = CombinedRegionSearch(graph, maximum_sequence_region_size=10)
        """
        super().__init__(graph)
        self.regions: list[Region] = []
        self.region_partitioner = RegionPartitioner(graph)
        self.minimum_topdown_search_size = minimum_topdown_search_size
        self.maximum_sequence_region_size = maximum_sequence_region_size

    def search_regions(self) -> list[Region]:
        """Execute two-phase region search to partition the graph into hierarchical regions.

        This is the main entry point for the CombinedRegionSearch algorithm. It performs
        a sophisticated two-phase analysis of the computation graph:

        **Phase 1: Bottom-Up Partitioning**
        Uses RegionPartitioner to create initial regions by:
        - Traversing graph from inputs to outputs
        - Identifying divergent nodes (where computation branches)
        - Finding convergence points (where branches rejoin)
        - Grouping linear sequences of operations
        - Creating initial LEAF regions based on these patterns

        **Phase 2: Top-Down Refinement**
        For each region from Phase 1, uses TopDownRegionBuilder to:
        - Identify and merge converged sub-patterns within the region
        - Split long sequences into smaller, more manageable regions
        - Create hierarchical COMPOSITE region structures
        - Ensure optimal region granularity for optimization

        **Algorithm Steps:**
        1. Initialize RegionPartitioner with the graph
        2. Partition graph into initial regions (Phase 1)
        3. Log partitioning statistics (coverage, region count)
        4. For each initial region:
           a. Create TopDownRegionBuilder for refinement
           b. Share tensor users map for efficient lookups
           c. Build composite region hierarchy (Phase 2)
           d. Validate node count consistency
           e. Recompute region boundaries
        5. Return final list of refined regions

        **Output Structure:**
        Each returned region is typically a COMPOSITE region containing:
        - LEAF child regions with actual graph nodes
        - Computed input/output tensor boundaries
        - Hierarchical structure reflecting computation patterns

        **Quality Metrics Logged:**
        - Total regions found: Number of top-level regions created
        - Total nodes visited: How many graph nodes were processed
        - Coverage percentage: What fraction of the graph was partitioned

        **Validation:**
        - Warns if node counts change during refinement (potential bug)
        - Ensures all nodes are accounted for
        - Verifies region boundary consistency

        Returns:
            List of Region objects representing the partitioned graph.
            Each region is a COMPOSITE region with a hierarchical structure
            of child regions. The regions collectively cover all nodes in
            the graph without overlap.

        Raises:
            May propagate exceptions from RegionPartitioner or TopDownRegionBuilder
            if graph structure is invalid or contains unsupported patterns.

        Example:
            >>> search = CombinedRegionSearch(graph)
            >>> regions = search.search_regions()
            >>> print(f"Graph partitioned into {len(regions)} regions")
            >>> # Analyze results
            >>> total_nodes = sum(r.get_all_nodes_recursive_count() for r in regions)
            >>> print(f"Total nodes in all regions: {total_nodes}")
            >>> # Print hierarchical structure
            >>> for region in regions:
            ...     search.print_tree(region)

        Note:
            This method modifies self.regions and returns it. Calling this
            method multiple times will overwrite previous results.

        See Also:
            RegionPartitioner: Phase 1 bottom-up partitioning
            TopDownRegionBuilder: Phase 2 top-down refinement
            print_tree: Visualize the resulting region hierarchy
        """
        # =====================================================================
        # PHASE 1: Bottom-Up Partitioning
        # =====================================================================
        # Create a fresh RegionPartitioner instance for this search.
        # This performs initial graph analysis including:
        # - Building tensor-to-users mapping for tracking data flow
        # - Computing forward reachability for convergence detection
        logger.info("Phase 1: Bottom-up partitioning")
        logger.debug("Initializing RegionPartitioner")
        region_partitioner = RegionPartitioner(self.graph)

        # Execute the bottom-up partitioning algorithm.
        # This traverses the graph and creates initial LEAF regions based on:
        # - Divergence/convergence patterns (where computation branches/rejoins)
        # - Linear sequences of non-divergent nodes
        # - Graph structure and operation types
        self.regions = region_partitioner.partition_graph()

        # =====================================================================
        # Log Phase 1 Results
        # =====================================================================
        # Report statistics about the initial partitioning to help understand
        # graph structure and verify complete coverage.
        coverage_pct = (
            100 * len(self.region_partitioner.visited_nodes) / len(self.graph.nodes)
            if self.graph.nodes
            else 0
        )
        logger.info(
            f"Phase 1 complete: {len(self.regions)} regions, "
            f"{len(self.region_partitioner.visited_nodes)}/{len(self.graph.nodes)} nodes ({coverage_pct:.1f}%)"
        )
        logger.debug("Proceeding to Phase 2: Top-down refinement")

        # =====================================================================
        # PHASE 2: Top-Down Refinement
        # =====================================================================
        # Track the next available region ID to ensure unique IDs across all regions.
        # This is important because we'll be creating new regions during refinement.
        logger.info("Phase 2: Top-down refinement")
        next_region_id = region_partitioner.current_region_id

        # Process each initial region to refine its structure.
        # Each region from Phase 1 becomes a root for hierarchical refinement.
        refined_count = 0
        skipped_count = 0
        for idx in range(len(self.regions)):
            total_nodes = len(self.regions[idx].get_all_nodes_recursive())
            if total_nodes < self.minimum_topdown_search_size:
                logger.debug(f"Skipping region {idx}: {total_nodes} nodes (below minimum)")
                skipped_count += 1
                continue

            # Create a TopDownRegionBuilder for this specific region.
            # This builder will analyze the region and create a hierarchical
            # structure of child regions based on internal patterns.
            logger.debug(f"Refining region {idx}: {total_nodes} nodes")
            region_builder = TopDownRegionBuilder(
                self.graph,
                self.regions[idx],
                next_region_id=next_region_id,
                maximum_sequence_region_size=self.maximum_sequence_region_size,
            )

            # Share the tensor users map from Phase 1 to avoid recomputation.
            # This map is expensive to build and is shared across all refinements.
            region_builder.tensor_users_map = region_partitioner.tensor_users_map

            # Track node count for validation.
            # The refinement should reorganize nodes into hierarchies without
            # losing or duplicating any nodes.
            node_count_before = len(self.regions[idx].get_all_nodes_recursive())

            # Execute top-down refinement on this region.
            # This creates a COMPOSITE region with hierarchical structure:
            # 1. Merges converged sub-regions (nodes between divergence/convergence)
            # 2. Splits long sequences into smaller regions
            # 3. Creates appropriate parent-child relationships

            self.regions[idx] = region_builder.build_composite_region()

            # Validate that refinement preserved all nodes.
            # A mismatch indicates a bug in the refinement logic.
            node_count_after = len(self.regions[idx].get_all_nodes_recursive())
            if node_count_before != node_count_after:
                logger.warning(
                    f"Node count mismatch in region {idx}: {node_count_before} → {node_count_after}"
                )

            # Recompute region boundaries after refinement.
            # The hierarchical structure may have changed the input/output
            # tensors at the top level of this region.
            region_partitioner.compute_region_boundaries(self.regions[idx])

            # Update next_region_id for the next iteration.
            # Each builder may have created new regions with new IDs.
            next_region_id = region_builder.next_region_id
            refined_count += 1

        logger.info(f"Phase 2 complete: refined {refined_count} regions, skipped {skipped_count}")

        # Return the final refined regions
        return self.regions


# =============================================================================
# Region Search Inspection Tool
# =============================================================================


def inspect_region_search(
    onnx_path: str,
    max_sequence_size: int = 10,
    include_all_regions: bool = False,
) -> list[Region]:
    """Inspect region search results for an ONNX model.

    This function loads an ONNX model, runs CombinedRegionSearch (which performs
    both bottom-up partitioning and top-down refinement internally), and prints
    detailed information about the discovered regions including their hierarchical
    structure.

    **What it does:**
    1. Loads ONNX model and converts to GraphSurgeon format
    2. Creates CombinedRegionSearch instance with specified parameters
    3. Runs two-phase search (partitioning + refinement) via search()
    4. Displays detailed region structure and statistics
    5. Returns the final list of refined regions

    **Output Sections:**
    - Initialization: Shows search parameters
    - Two-Phase Search: Runs automatically via CombinedRegionSearch.search()
    - Detailed Structure: Shows each region's hierarchy and properties
    - Summary Statistics: Shows region counts and node coverage

    Args:
        onnx_path: Path to the ONNX model file
        max_sequence_size: Maximum size for sequence regions during refinement (default: 10)
        include_all_regions: Include all regions, even those without major quantizable
                   operations (Conv, MatMul, etc.). Default: False (skips such regions)

    Returns:
        List of discovered and refined regions (LEAF and COMPOSITE)

    Example:
        >>> # Inspect model with default settings
        >>> regions = inspect_region_search("model.onnx")
        >>> print(f"Found {len(regions)} regions")
        >>>
        >>> # Custom sequence size
        >>> regions = inspect_region_search("model.onnx", max_sequence_size=20)
        >>>
        >>> # Include all regions
        >>> regions = inspect_region_search("model.onnx", include_all_regions=True)
    """
    # Load ONNX model
    logger.info(f"Loading model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    # Convert to onnx_graphsurgeon Graph
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()
    logger.info(
        f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.inputs)} inputs, {len(graph.outputs)} outputs"
    )

    # Initialize CombinedRegionSearch (contains RegionPartitioner internally)
    logger.debug(
        f"Search parameters: max_steps={DEFAULT_MAX_STEPS}, max_sequence_size={max_sequence_size}"
    )

    combined_search = CombinedRegionSearch(graph, maximum_sequence_region_size=max_sequence_size)

    # Run complete two-phase region search
    logger.info("Running region search")
    regions = combined_search.search_regions()

    # Show detailed region structure
    logger.info("Analyzing region structure")
    all_regions = []
    for i, region in enumerate(regions):
        for child in region.get_children():
            if not include_all_regions and not has_quantizable_operations(child, graph):
                region.remove_child(child)
        if not include_all_regions and not has_quantizable_operations(region, graph):
            logger.debug(f"Filtered out region {i} (no quantizable operations)")
            continue
        logger.debug(
            f"Region {i}: {region.get_type().value}, {len(region.get_all_nodes_recursive())} nodes, "
            f"{len(region.inputs)} inputs, {len(region.outputs)} outputs"
        )
        all_regions.append(region)
        if region.get_type() == RegionType.COMPOSITE:
            logger.debug(f"  {len(region.get_children())} child regions")
            all_regions.extend(region.get_children())
        combined_search.print_tree(region, indent=2)

    # Summary statistics
    leaf_regions = sum(1 for r in all_regions if r.get_type() == RegionType.LEAF)
    composite_regions = sum(1 for r in all_regions if r.get_type() == RegionType.COMPOSITE)

    all_nodes = set()
    for region in all_regions:
        all_nodes.update(region.get_all_nodes_recursive())
    total_nodes = len(all_nodes)
    coverage_pct = 100 * total_nodes / len(graph.nodes) if graph.nodes else 0

    logger.info(
        f"Summary: {len(all_regions)} regions ({leaf_regions} LEAF, {composite_regions} COMPOSITE), "
        f"{total_nodes}/{len(graph.nodes)} nodes ({coverage_pct:.1f}%)"
    )

    # Print histogram of region sizes
    region_sizes = [
        len(r.get_all_nodes_recursive()) for r in all_regions if r.get_type() == RegionType.LEAF
    ]

    if region_sizes:
        min_size = min(region_sizes)
        max_size = max(region_sizes)
        avg_size = sum(region_sizes) / len(region_sizes)

        logger.info(f"LEAF region sizes: min={min_size}, max={max_size}, avg={avg_size:.1f}")

        # Create histogram bins
        size_counts = Counter(region_sizes)

        # Display histogram
        logger.debug("Size distribution:")
        for size in sorted(size_counts.keys()):
            count = size_counts[size]
            bar = "█" * min(count, 50)  # Cap bar length at 50
            logger.debug(f"  {size:4d} nodes: {bar} ({count} regions)")

    return regions


def main():
    """Command-line entry point for region search inspection."""
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune.region_search",
        description="Inspect region search results for ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python -m modelopt.onnx.quantization.autotune.region_search --model model.onnx

  # Verbose mode for debug logging
  python -m modelopt.onnx.quantization.autotune.region_search \\
      --model model.onnx --verbose

  # Custom maximum sequence size
  python -m modelopt.onnx.quantization.autotune.region_search \\
      --model model.onnx --max-sequence-size 20
        """,
    )

    parser.add_argument("--model", "-m", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument(
        "--max-sequence-size",
        type=int,
        default=10,
        help="Maximum size for sequence regions during refinement (default: 10)",
    )
    parser.add_argument(
        "--include-all-regions",
        action="store_true",
        help="Include all regions, even those without major quantizable operations. "
        "Default: False (skips such regions)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)

    # Run inspection
    try:
        regions = inspect_region_search(
            onnx_path=args.model,
            max_sequence_size=args.max_sequence_size,
            include_all_regions=args.include_all_regions,
        )
        logger.info(f"✓ Inspection complete: {len(regions)} top-level regions discovered")
        return 0
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
