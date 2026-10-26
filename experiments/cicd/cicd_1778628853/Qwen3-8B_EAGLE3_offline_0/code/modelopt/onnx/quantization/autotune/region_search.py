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

"""Hierarchical region discovery and partitioning for ONNX graphs."""

import sys
from collections import defaultdict, deque

import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_NODES_TO_SHOW = 20
MAX_PROBE_STEPS_AFTER_CONVERGE = 3


class RegionSearchBase:
    """Base class for region search algorithms providing common graph analysis utilities.

    This class serves as a foundation for region-based graph analysis algorithms by
    providing essential data structures and methods for:
    - Graph traversal and reachability analysis
    - Divergence/convergence pattern detection
    - Region boundary computation
    - Tensor flow tracking

    For large graphs, initialization may take significant time but enables
    efficient queries during region formation.
    """

    def __init__(
        self,
        graph: gs.Graph,
        root: Region | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        tensor_users_map: dict[str, list[int]] | None = None,
        forward_reachable_nodes_map: dict[int, dict[int, int]] | None = None,
    ):
        """Initialize the base region search with graph analysis.

        Performs pre-computation of essential data structures for efficient
        region analysis:
        1. Creates or validates root region containing all nodes
        2. Builds tensor-to-users mapping for divergence detection
        3. Pre-computes forward reachability for convergence detection
        """
        self.graph = graph
        if tensor_users_map is None:
            tensor_users_map = get_tensor_consumer_node_indices(self.graph)
        self.tensor_users_map = tensor_users_map
        if root is None:
            root = self._build_root_region()
        self.root = root
        if forward_reachable_nodes_map is None:
            forward_reachable_nodes_map = self._build_forward_reachable_nodes_map(
                max_steps=max_steps
            )
        self.forward_reachable_nodes_map = forward_reachable_nodes_map

    def _build_root_region(self) -> Region:
        """Create a root region containing all nodes in the graph.

        The root region serves as the universal search space for region
        formation algorithms. It represents the entire computation graph
        as a single region before any partitioning.

        Returns:
            Region of type ROOT containing all graph nodes.
        """
        root = Region(region_id=0, level=0, region_type=RegionType.ROOT)
        root.nodes.update(range(len(self.graph.nodes)))
        self.compute_region_boundaries(root)
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

        Args:
            node_idx: Index of the node to check

        Returns:
            True if the node has at least one output consumed by multiple nodes,
            False otherwise or if node is not in root region.
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

        Args:
            start_node_idx: Index of node to start search from
            max_steps: Maximum forward distance to explore

        Returns:
            Dictionary mapping reachable node indices to their distances from start.
            Includes start_node_idx mapped to distance 0.
        """
        reachable: dict[int, int] = {start_node_idx: 0}
        queue: deque[tuple[int, int]] = deque([(start_node_idx, 0)])
        while queue:
            current_node_idx, distance = queue.popleft()
            if distance >= max_steps:
                continue
            for output in self.graph.nodes[current_node_idx].outputs:
                for next_node_idx in self.tensor_users_map.get(output.name, ()):
                    if next_node_idx not in reachable:
                        reachable[next_node_idx] = distance + 1
                        queue.append((next_node_idx, distance + 1))
        return reachable

    def _build_forward_reachable_nodes_map(self, max_steps: int) -> dict[int, dict[int, int]]:
        """Pre-compute forward reachability for all nodes in the graph.

        This is a key optimization that enables efficient convergence detection.
        By pre-computing forward reachability once, we can quickly answer queries
        like "Can node A reach node B?" and "What is the distance from A to B?"

        Args:
            max_steps: Maximum forward distance to pre-compute for each node.
                      Limits both time and space complexity.

        Returns:
            Nested dictionary where outer key is start node index, inner key is
            reachable node index, and value is shortest-path distance.
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

    def _find_common_reachable_nodes(
        self, node_idx: int, branches: list[int]
    ) -> tuple[list[dict], set[int]]:
        """Find common reachable nodes from all branches (potential convergence points).

        Used as STEP 1 of convergence detection in _find_converge_nodes.

        Args:
            node_idx: Index of the divergent node (excluded from common_nodes).
            branches: List of branch head node indices.

        Returns:
            (branch_reachable, common_nodes)
        """
        branch_reachable = [self.forward_reachable_nodes_map.get(b, {}) for b in branches]

        if not branch_reachable:
            logger.debug("  No reachable nodes from branches")
            return [], set()

        common_nodes = set.intersection(*[set(r.keys()) for r in branch_reachable])
        logger.debug(f"  {len(common_nodes)} common nodes found")
        common_nodes.discard(node_idx)

        if not common_nodes:
            logger.debug("  No valid convergence candidates")
            return [], set()

        return branch_reachable, common_nodes

    def _evaluate_convergence_candidate(
        self,
        candidate_idx: int,
        reachable_from_start: dict,
        branch_reachable: list,
    ) -> tuple[bool, int]:
        r"""Check if a candidate convergence node forms a valid region and return its max distance.

        A valid region has no \"escaping\" edges: no node inside the region may reach a node
        outside the region before reaching the candidate convergence point.

        Args:
            candidate_idx: Candidate convergence node index.
            reachable_from_start: Forward reachability from the divergent node.
            branch_reachable: Per-branch reachability dicts (for max distance).

        Returns:
            (is_valid, max_distance). max_distance is only meaningful when is_valid is True.
        """
        region_nodes: set[int] = set(reachable_from_start.keys())
        reachable_from_candidate = self.forward_reachable_nodes_map.get(candidate_idx, {})
        region_nodes = region_nodes - set(reachable_from_candidate.keys())

        for rnode_index in region_nodes:
            reachable_from_rnode = self.forward_reachable_nodes_map.get(rnode_index, {})
            rnode_to_candidate_distance = reachable_from_rnode.get(candidate_idx, float("inf"))
            for test_node_idx in reachable_from_rnode:
                if test_node_idx in region_nodes:
                    continue
                rnode_to_test_distance = reachable_from_rnode.get(test_node_idx, float("inf"))
                if any(
                    d == float("inf") for d in (rnode_to_test_distance, rnode_to_candidate_distance)
                ):
                    return False, 0

        max_distance = max(reachable[candidate_idx] for reachable in branch_reachable)
        return True, max_distance

    def _find_converge_nodes(self, node_idx: int) -> tuple[int | None, set[int]]:
        """Find convergence point and intermediate nodes for a divergent node.

        Given a divergent node (where computation branches), this method finds:
        1. The convergence node: Where the branches rejoin
        2. All nodes between divergence and convergence

        Args:
            node_idx: Index of the divergent node to find convergence for

        Returns:
            Tuple containing:
            - Convergence node index (None if no convergence found)
            - Set of nodes between divergence and convergence
        """
        node = self.graph.nodes[node_idx]
        logger.debug(f"Finding convergence for node {node_idx} ({node.op})")

        branches: list[int] = []
        for output in node.outputs:
            branches.extend(self.tensor_users_map.get(output.name, []))

        branches = list(dict.fromkeys(branches))

        logger.debug(f"  {len(branches)} unique branches found")

        if len(branches) <= 1:
            logger.debug("  Insufficient branches for convergence")
            return None, set()

        branch_reachable, common_nodes = self._find_common_reachable_nodes(node_idx, branches)
        if not branch_reachable or not common_nodes:
            return None, set()

        # Select Best Convergence Node with Region Validity Check
        converge_node_idx: int | None = None
        min_max_distance = float("inf")

        reachable_from_start = self.forward_reachable_nodes_map.get(node_idx, {})

        for candidate_idx in common_nodes:
            valid, max_distance = self._evaluate_convergence_candidate(
                candidate_idx, reachable_from_start, branch_reachable
            )
            if not valid:
                continue
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

        # Compute All Nodes Between Divergence and Convergence
        visited_nodes: set[int] = set()
        for candidate_idx in reachable_from_start:
            if candidate_idx == converge_node_idx:
                continue
            reachable_from_candidate = self.forward_reachable_nodes_map.get(candidate_idx, {})
            if converge_node_idx in reachable_from_candidate:
                visited_nodes.add(candidate_idx)
        logger.debug(f"  {len(visited_nodes)} nodes between divergence and convergence")
        return converge_node_idx, visited_nodes

    def _max_distance_to_nodes(self, src_idx: int, dst_indices: set[int]) -> int:
        """Compute maximum distance from a source node to a set of destination nodes.

        Uses pre-computed forward reachability to efficiently find the maximum
        shortest-path distance from src_idx to any node in dst_indices.

        Args:
            src_idx: Index of the source node
            dst_indices: Set of destination node indices

        Returns:
            Maximum distance from src_idx to any node in dst_indices
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

        Args:
            region: The region to compute boundaries for
            include_constant: Whether to include constant tensors in inputs
        """
        node_indices = region.get_region_nodes_and_descendants()

        consumed_tensors: set[str] = set()
        produced_tensors: set[str] = set()
        region_outputs: set[str] = set()

        for node_idx in node_indices:
            if node_idx >= len(self.graph.nodes):
                continue
            node = self.graph.nodes[node_idx]

            # Collect consumed tensors (potential inputs)
            for input_tensor in node.inputs:
                if isinstance(input_tensor, gs.Constant) and not include_constant:
                    continue
                consumed_tensors.add(input_tensor.name)

            # Collect produced tensors and determine outputs
            for output_tensor in node.outputs:
                tensor_name = output_tensor.name
                produced_tensors.add(tensor_name)

                consumer_indices = self.tensor_users_map.get(tensor_name, [])
                has_external_consumer = any(idx not in node_indices for idx in consumer_indices)
                is_graph_output = output_tensor in self.graph.outputs

                if has_external_consumer or is_graph_output or not consumer_indices:
                    region_outputs.add(tensor_name)

        # Region inputs = consumed tensors not produced internally
        region.inputs = sorted(consumed_tensors - produced_tensors)
        region.outputs = sorted(region_outputs)

        logger.debug(
            f"Computed boundaries: {len(region.inputs)} inputs, {len(region.outputs)} outputs"
        )

    def print_tree(
        self,
        region: Region | None = None,
        indent: int = 0,
        max_items: int = DEFAULT_MAX_NODES_TO_SHOW,
        file=None,
    ) -> None:
        """Print hierarchical region tree in human-readable text format."""
        region = region or self.root
        file = file or sys.stdout
        p = "  " * indent

        def truncated(items, fmt=str):
            """Yield formatted items, truncating with count if needed."""
            items = list(items)
            yield from (fmt(x) for x in items[:max_items])
            if len(items) > max_items:
                yield f"... and {len(items) - max_items} more"

        direct_nodes = region.get_nodes()
        children = region.get_children()
        # Header + counts
        print(
            f"{p}├─ Region {region.id} (Level {region.level}, Type: {region.type.value})", file=file
        )
        print(f"{p}│  ├─ Direct nodes: {len(direct_nodes)}", file=file)
        print(f"{p}│  ├─ Total nodes: {len(region.get_region_nodes_and_descendants())}", file=file)
        print(f"{p}│  ├─ Children: {len(children)}", file=file)
        # I/O
        for label, items in [("Inputs", region.inputs), ("Outputs", region.outputs)]:
            print(f"{p}│  ├─ {label}: {len(items)}", file=file)
            for line in truncated(items):
                print(f"{p}│  │    - {line}", file=file)
        # Direct nodes
        if direct_nodes:
            print(f"{p}│\n{p}│  Nodes in this region:", file=file)

            def node_fmt(i: int) -> str:
                return f"Node {i}: {self.graph.nodes[i].op} ({self.graph.nodes[i].name})"

            for line in truncated(sorted(direct_nodes), node_fmt):
                print(f"{p}│    - {line}", file=file)
        # Children
        if children:
            print(f"{p}│\n{p}│  Child regions:", file=file)
            for child in children:
                print(f"{p}│", file=file)
                self.print_tree(child, indent + 1, max_items, file)


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
    """

    def __init__(
        self,
        graph: gs.Graph,
        tensor_users_map: dict[str, list[int]] | None = None,
        forward_reachable_nodes_map: dict[int, dict[int, int]] | None = None,
    ):
        """Initialize the partitioner with a computation graph.

        Sets up necessary data structures and inherits graph analysis utilities
        from RegionSearchBase (tensor users map, reachability, etc.).

        Args:
            graph: The ONNX computation graph (onnx_graphsurgeon.Graph)
            tensor_users_map: Mapping from tensor names to consuming node indices
            forward_reachable_nodes_map: Pre-computed forward reachability for all nodes
        """
        super().__init__(
            graph,
            root=None,
            tensor_users_map=tensor_users_map,
            forward_reachable_nodes_map=forward_reachable_nodes_map,
        )
        self.regions: list[Region] = []
        self.current_region: Region | None = None
        self.current_region_id: int = 0
        self.visited_nodes: set[int] = set()

    def _append_node_to_region(self, node_idx: int):
        """Add a node to the current region, creating a new region if needed.

        This is the primary method for building regions incrementally. If no
        region is currently active, creates a new LEAF region. Then adds the
        specified node to that region.

        Args:
            node_idx: Index of the node to add to the current region

        Returns:
            None
        """
        node = self.graph.nodes[node_idx]
        if self.current_region is None:
            self.current_region = Region(
                region_id=self.current_region_id, level=0, region_type=RegionType.LEAF
            )
            logger.debug(f"Started region {self.current_region_id}")
            self.current_region_id += 1

        self.current_region.nodes.add(node_idx)
        logger.debug(
            f"  Added node {node_idx} ({node.op}), region size: {len(self.current_region.nodes)}"
        )

    def _commit_region(self):
        """Finalize and store the current region being built.

        Completes region construction by:
        1. Computing input/output tensor boundaries
        2. Adding region to the completed regions list
        3. Resetting current_region to None for next region

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
            region_size = len(self.current_region.nodes)
            region_id = self.current_region.id

            self.compute_region_boundaries(self.current_region)

            self.regions.append(self.current_region)
            logger.debug(
                f"Committed region {region_id}: {region_size} nodes (total: {len(self.regions)})"
            )
            self.current_region = None
        else:
            logger.debug("No region to commit")

    def _build_sequence_from_node(self, node_idx: int, max_nodes: int = -1):
        """Build a region from a linear sequence of nodes.

        Starting from a non-divergent node, follows the forward chain of nodes,
        adding each non-divergent node to the current region. Stops when hitting:
        - A divergent node (branches to multiple paths)
        - A node already visited
        - End of graph

        Args:
            node_idx: Index of the starting node
            max_nodes: Maximum number of nodes to add to the region (-1 for no limit)

        Returns:
            None
        """
        logger.debug(f"Building sequence from node {node_idx} ({self.graph.nodes[node_idx].op})")

        queue: deque[int] = deque([node_idx])
        nodes_added = 0

        while queue:
            current_idx = queue.popleft()
            node = self.graph.nodes[current_idx]

            self._append_node_to_region(current_idx)
            self.visited_nodes.add(current_idx)
            nodes_added += 1

            if self._is_node_divergent(current_idx):
                logger.debug(f"  Stopped at divergent node {current_idx} ({node.op})")
            else:
                # Queue successors for non-divergent nodes
                for output in node.outputs:
                    if output.name in self.tensor_users_map:
                        queue.extend(self.tensor_users_map[output.name])

            if 0 < max_nodes <= nodes_added:
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
        """
        visited_nodes.remove(start_node_idx)
        for node_idx in sorted(visited_nodes):
            self._append_node_to_region(node_idx)
            self.visited_nodes.add(node_idx)
        if not self._is_node_divergent(converge_node_idx):
            self._append_node_to_region(converge_node_idx)
            self.visited_nodes.add(converge_node_idx)
        self._build_sequence_from_node(converge_node_idx, max_nodes=MAX_PROBE_STEPS_AFTER_CONVERGE)

    def _build_region_from_node(self, node_idx: int):
        """Process a single node and create appropriate region(s) based on its pattern.

        This is the core dispatch method that determines how to handle each node based on whether
        it's divergent (branches) or sequential.

        - Pattern 1: Divergent with Convergence (Ideal Case)
        - Pattern 2: Divergent without Convergence (Boundary Case)
        - Pattern 3: Sequential Chain (Common Case)

        Args:
            node_idx: Index of node to process

        Side Effects:
            - Marks processed nodes as visited
            - Creates and commits region(s) via helper methods
            - May recursively process successor nodes (in sequence building)
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
        else:
            # Pattern 3: Handle non-divergent (sequential) nodes
            logger.debug("  Non-divergent node, building sequence")
            # Build region by following the linear chain forward
            self._build_sequence_from_node(node_idx)
            self._commit_region()

    def partition_graph(self):
        """Partition the entire graph into non-overlapping LEAF regions.

        This is the main entry point for bottom-up graph partitioning. Performs
        a single pass over all nodes in graph order, creating regions based on
        divergence/convergence patterns and sequential chains.

        Returns:
            List of non-overlapping LEAF regions created from the graph.

        """
        logger.info(f"Partitioning graph ({len(self.graph.nodes)} nodes)")
        logger.debug(
            f"Initial state: {len(self.visited_nodes)} visited, {len(self.regions)} regions"
        )

        for node_idx in range(len(self.graph.nodes)):
            self._build_region_from_node(node_idx)

        coverage_pct = (
            100 * len(self.visited_nodes) / len(self.graph.nodes) if self.graph.nodes else 0
        )
        logger.info(
            f"Partitioning complete: {len(self.regions)} regions, "
            f"{len(self.visited_nodes)}/{len(self.graph.nodes)} nodes ({coverage_pct:.1f}%)"
        )

        if self.regions:
            region_sizes = [len(r.nodes) for r in self.regions]
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
    """

    def __init__(
        self,
        graph: gs.Graph,
        root: Region,
        next_region_id: int = 0,
        maximum_sequence_region_size: int = 10,
        tensor_users_map: dict[str, list[int]] | None = None,
        forward_reachable_nodes_map: dict[int, dict[int, int]] | None = None,
    ):
        """Initialize the refiner with a region to refine.

        Args:
            graph: The ONNX graph (onnx_graphsurgeon.Graph)
            root: The region to refine (typically from RegionPartitioner)
            next_region_id: Starting ID for new regions created during refinement
            maximum_sequence_region_size: Maximum nodes per sequence region during merging (default: 10)
        """
        super().__init__(
            graph,
            root=root,
            tensor_users_map=tensor_users_map,
            forward_reachable_nodes_map=forward_reachable_nodes_map,
        )
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

        Args:
            node_indices: Set of node indices to add to the region

        Returns:
            New LEAF region containing the specified nodes
        """
        region = Region(
            region_id=self.next_region_id, level=self.root.level + 1, region_type=RegionType.LEAF
        )
        self.next_region_id += 1
        for node_idx in node_indices:
            region.nodes.add(node_idx)
        self.compute_region_boundaries(region)
        return region

    def _build_region_usage_map(self, regions: list[Region]) -> dict[str, list[Region]]:
        """Build mapping from tensor names to regions that consume them.

        Similar to tensor_users_map but at the region level instead of node level.
        This enables efficient traversal of region dependencies for merging decisions.

        Args:
            regions: List of regions to build the usage map for

        Returns:
            Mapping from tensor names to regions that consume them
        """
        region_usage_map: dict[str, list[Region]] = defaultdict(list)
        for region in regions:
            for input_tensor in region.inputs:
                region_usage_map[input_tensor].append(region)
        return region_usage_map

    def _split_sequence_regions(self, root: Region) -> list[Region]:
        """Split a region into smaller sub-regions by merging producer-consumer chains.

        Takes a region and creates optimal sub-regions by:
        1. Initially splitting into individual single-node regions
        2. Traversing in data flow order (following tensor dependencies)
        3. Merging adjacent regions that form simple producer-consumer chains
        4. Respecting boundary operations and size limits

        Args:
            root: The region to split

        Returns:
            List of smaller sub-regions
        """
        result_regions: list[Region] = []
        removed_regions: set[int] = set()

        # PHASE 1: Split into Single-Node Regions
        for node_idx in root.get_nodes():
            region = Region(
                region_id=self.next_region_id, level=root.level + 1, region_type=RegionType.LEAF
            )
            region.nodes.add(node_idx)
            self.compute_region_boundaries(region)
            result_regions.append(region)
            self.next_region_id += 1

        region_usage_map = self._build_region_usage_map(result_regions)

        # PHASE 2: Merge Regions in Data Flow Order
        queue = deque(root.inputs)

        while len(queue) > 0:
            tensor_name = queue.popleft()
            # Skip tensors not produced by any region in our scope
            if tensor_name not in region_usage_map:
                continue
            # Process each region consuming this tensor (potential merge targets)
            consumers = region_usage_map[tensor_name]
            for consumer in consumers:
                # Skip regions already merged into others
                if consumer.id in removed_regions:
                    continue
                # Merging criteria: ALL outputs go to same single region
                common_use_region = None
                can_merge = True
                # Check all outputs of the consumer region
                for output_tensor in consumer.outputs:
                    queue.append(output_tensor)
                    if output_tensor not in region_usage_map:
                        can_merge = False
                        break
                    use_regions = region_usage_map[output_tensor]
                    if len(use_regions) != 1:
                        can_merge = False
                        break
                    if common_use_region is None:
                        common_use_region = use_regions[0]
                    elif common_use_region != use_regions[0]:
                        can_merge = False
                        break
                # No valid downstream region to merge with
                if common_use_region is None or common_use_region.id in removed_regions:
                    can_merge = False
                    continue
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
                    len(consumer.nodes) >= self.maximum_sequence_region_size
                    or len(common_use_region.nodes) >= self.maximum_sequence_region_size
                ):
                    # One or both regions too large - don't merge
                    can_merge = False
                    continue
                # All criteria met: merge consumer into its downstream region
                if can_merge:
                    common_use_region.merge(consumer)
                    removed_regions.add(consumer.id)
        # Remove regions that were merged into others
        result_regions = [region for region in result_regions if region.id not in removed_regions]
        # Recompute boundaries for all remaining regions
        for region in result_regions:
            self.compute_region_boundaries(region)

        return result_regions

    def _merge_converged_regions(self, root: Region):
        """Identify and merge convergence patterns within a region.

        Traverses the region to find divergent nodes and their convergence points,
        creating sub-regions that capture divergence→branches→convergence patterns.
        Nodes not part of any convergence pattern are left for sequence splitting.

        Args:
            root: The region to merge

        Returns:
            List of merged regions
        """
        result_regions: list[Region] = []
        removed_nodes: set[int] = set()
        queue = deque(root.inputs)
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
                visited_nodes = visited_nodes.intersection(root.get_region_nodes_and_descendants())
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
        remaining_nodes = set(root.get_nodes()) - removed_nodes
        if len(remaining_nodes) > 0:
            result_regions.append(self._create_leaf_region(remaining_nodes))
        # compute region boundaries for all regions
        for region in result_regions:
            self.compute_region_boundaries(region)
        return result_regions

    def build_composite_region(self) -> Region:
        """Refine a flat region into a hierarchical COMPOSITE region."""
        # merge converged regions into composite regions
        regions = self._merge_converged_regions(self.root)
        # split sequence regions into smaller regions
        result_regions: list[Region] = []
        for region in regions:
            result_regions.extend(self._split_sequence_regions(region))
        for region in result_regions:
            self.compute_region_boundaries(region, include_constant=True)
        regions = result_regions
        # merge all regions into a single composite region
        if len(regions) > 1:
            composite = Region(
                region_id=self.next_region_id,
                level=self.root.level,
                region_type=RegionType.COMPOSITE,
            )
            self.next_region_id += 1
            regions = sorted(
                regions, key=lambda x: RegionPattern.from_region(x, self.graph).signature
            )
            for region in regions:
                composite.add_child(region)
            self.compute_region_boundaries(composite)
            regions = [composite]
        self.regions = regions
        return self.regions[0]


class CombinedRegionSearch(RegionSearchBase):
    """Two-phase region search combining bottom-up partitioning with top-down refinement.

    This class implements a sophisticated region discovery algorithm that combines two
    complementary strategies to create well-formed, hierarchical regions from an ONNX
    computation graph.

    """

    def __init__(
        self,
        graph: gs.Graph,
        maximum_sequence_region_size: int = 10,
        minimum_topdown_search_size: int = 10,
    ):
        """Initialize CombinedRegionSearch for a given ONNX graph."""
        super().__init__(graph)
        self.regions: list[Region] = []
        self.minimum_topdown_search_size = minimum_topdown_search_size
        self.maximum_sequence_region_size = maximum_sequence_region_size

    def search_regions(self) -> list[Region]:
        """Execute two-phase region search to partition the graph into hierarchical regions.

        1. Bottom-up partitioning
        2. Top-down refinement

        Args:
            None

        Returns:
            List of hierarchical regions created from the graph
        """
        logger.info("Phase 1: Bottom-up partitioning")
        logger.debug("Initializing RegionPartitioner")
        region_partitioner = RegionPartitioner(self.graph)

        # Execute the bottom-up partitioning algorithm.
        self.regions = region_partitioner.partition_graph()

        coverage_pct = (
            100 * len(region_partitioner.visited_nodes) / len(self.graph.nodes)
            if self.graph.nodes
            else 0
        )
        logger.info(
            f"Phase 1 complete: {len(self.regions)} regions, "
            f"{len(region_partitioner.visited_nodes)}/{len(self.graph.nodes)} nodes ({coverage_pct:.1f}%)"
        )
        logger.debug("Proceeding to Phase 2: Top-down refinement")

        logger.info("Phase 2: Top-down refinement")
        next_region_id = region_partitioner.current_region_id

        refined_count = 0
        for idx, region in enumerate(self.regions):
            node_count = len(region.get_region_nodes_and_descendants())
            if node_count < self.minimum_topdown_search_size:
                logger.debug(f"Skipping region {idx}: {node_count} nodes (below minimum)")
                continue

            logger.debug(f"Refining region {idx}: {node_count} nodes")
            region_builder = TopDownRegionBuilder(
                self.graph,
                region,
                next_region_id=next_region_id,
                maximum_sequence_region_size=self.maximum_sequence_region_size,
                tensor_users_map=region_partitioner.tensor_users_map,
                forward_reachable_nodes_map=region_partitioner.forward_reachable_nodes_map,
            )

            self.regions[idx] = region_builder.build_composite_region()
            node_count_after = len(self.regions[idx].get_region_nodes_and_descendants())
            if node_count != node_count_after:
                logger.warning(
                    f"Node count mismatch in region {idx}: {node_count} → {node_count_after}"
                )

            region_partitioner.compute_region_boundaries(self.regions[idx])
            next_region_id = region_builder.next_region_id
            refined_count += 1

        logger.info(f"Phase 2 complete: refined {refined_count}/{len(self.regions)} regions")

        return self.regions
