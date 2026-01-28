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
from collections import deque

import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_NODES_TO_SHOW = 20
MAX_PROBE_STEPS_AFTER_CONVERGE = 3


class RegionSearchBase:
    """Base class for region search algorithms providing common graph analysis utilities."""

    def __init__(
        self,
        graph: gs.Graph,
        root: Region | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        tensor_users_map: dict[str, list[int]] | None = None,
        forward_reachable_nodes_map: dict[int, dict[int, int]] | None = None,
    ):
        """Initialize the base region search with graph analysis."""
        self.graph = graph
        if root is None:
            root = self._build_root_region()
        self.root = root
        if tensor_users_map is None:
            tensor_users_map = get_tensor_consumer_node_indices(self.graph)
        self.tensor_users_map = tensor_users_map
        if forward_reachable_nodes_map is None:
            forward_reachable_nodes_map = self._build_forward_reachable_nodes_map(
                max_steps=max_steps
            )
        self.forward_reachable_nodes_map = forward_reachable_nodes_map

    def _build_root_region(self) -> Region:
        """Create a root region containing all nodes in the graph."""
        root = Region(region_id=0, level=0, region_type=RegionType.ROOT)
        for node_idx in range(len(self.graph.nodes)):
            root.add_node(node_idx)
        for tensor_name in root.inputs:
            root.add_input(tensor_name)
        for tensor_name in root.outputs:
            root.add_output(tensor_name)
        return root

    def _is_tensor_divergent(self, tensor_name: str) -> bool:
        """Check if a tensor is consumed by multiple nodes (divergent)."""
        return len(self.tensor_users_map.get(tensor_name, [])) > 1

    def _is_node_divergent(self, node_idx: int) -> bool:
        """Check if a node has outputs that branch to multiple consumers."""
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
        """Compute all nodes reachable forward from a starting node with distances."""
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
        """Pre-compute forward reachability for all nodes in the graph."""
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
        """Find convergence point and intermediate nodes for a divergent node."""
        node = self.graph.nodes[node_idx]
        logger.debug(f"Finding convergence for node {node_idx} ({node.op})")

        branches: list[int] = []
        for output in node.outputs:
            if output.name in self.tensor_users_map:
                branches.extend(self.tensor_users_map[output.name])

        branches = list(dict.fromkeys(branches))

        logger.debug(f"  {len(branches)} unique branches found")

        # Need at least 2 branches for convergence to be meaningful
        if len(branches) <= 1:
            logger.debug("  Insufficient branches for convergence")
            return None, set()

        # STEP 1: Find Common Reachable Nodes (Potential Convergence Points)
        branch_reachable = [self.forward_reachable_nodes_map.get(b, {}) for b in branches]

        if not branch_reachable:
            logger.debug("  No reachable nodes from branches")
            return None, set()

        common_nodes = set.intersection(*[set(r.keys()) for r in branch_reachable])
        logger.debug(f"  {len(common_nodes)} common nodes found")
        # Remove the divergent node itself (not a convergence point)
        common_nodes.discard(node_idx)

        if not common_nodes:
            logger.debug("  No valid convergence candidates")
            return None, set()

        # STEP 2: Select Best Convergence Node with Region Validity Check
        converge_node_idx: int | None = None
        min_max_distance = float("inf")

        reachable_from_start = self.forward_reachable_nodes_map.get(node_idx, {})

        # Evaluate each candidate convergence point
        for candidate_idx in common_nodes:
            # Define the potential region: nodes between start and candidate
            region_nodes: set[int] = set()
            region_nodes.update(set(reachable_from_start.keys()))
            reachable_from_candidate = self.forward_reachable_nodes_map.get(candidate_idx, {})
            region_nodes.difference_update(set(reachable_from_candidate.keys()))

            broken_region = False
            for rnode_index in region_nodes:
                reachable_from_rnode = self.forward_reachable_nodes_map.get(rnode_index, {})
                rnode_to_candidate_distance = reachable_from_rnode.get(candidate_idx, float("inf"))
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
            # Valid candidate! Check if it's the nearest one
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
        # STEP 3: Compute All Nodes Between Divergence and Convergence
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
        """Compute maximum distance from a source node to a set of destination nodes."""
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

        def print_items(items, label, formatter=str):
            """Print a truncated list of items."""
            items = list(items)
            print(f"{p}│  ├─ {label}: {len(items)}", file=file)
            for item in items[:max_items]:
                print(f"{p}│  │    - {formatter(item)}", file=file)
            if len(items) > max_items:
                print(f"{p}│  │    ... and {len(items) - max_items} more", file=file)

        # Header
        print(
            f"{p}├─ Region {region.id} (Level {region.level}, Type: {region.type.value})",
            file=file,
        )

        # Counts
        direct_nodes = region.get_nodes()
        children = region.get_children()
        print(f"{p}│  ├─ Direct nodes: {len(direct_nodes)}", file=file)
        print(f"{p}│  ├─ Total nodes: {len(region.get_region_nodes_and_descendants())}", file=file)
        print(f"{p}│  ├─ Children: {len(children)}", file=file)

        # I/O
        print_items(region.inputs, "Inputs")
        print_items(region.outputs, "Outputs")

        # Direct nodes
        if direct_nodes:
            print(f"{p}│\n{p}│  Nodes in this region:", file=file)
            for node_idx in sorted(direct_nodes)[:max_items]:
                if node_idx < len(self.graph.nodes):
                    node = self.graph.nodes[node_idx]
                    print(f"{p}│    - Node {node_idx}: {node.op} ({node.name})", file=file)
            if len(direct_nodes) > max_items:
                print(f"{p}│    ... and {len(direct_nodes) - max_items} more", file=file)

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
    """

    def __init__(
        self,
        graph: gs.Graph,
        tensor_users_map: dict[str, list[int]] | None = None,
        forward_reachable_nodes_map: dict[int, dict[int, int]] | None = None,
    ):
        """Initialize the partitioner with a computation graph."""
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
        """Add a node to the current region, creating a new region if needed."""
        node = self.graph.nodes[node_idx]
        if self.current_region is None:
            self.current_region = Region(
                region_id=self.current_region_id, level=0, region_type=RegionType.LEAF
            )
            logger.debug(f"Started region {self.current_region_id}")
            self.current_region_id += 1

        self.current_region.add_node(node_idx)
        logger.debug(
            f"  Added node {node_idx} ({node.op}), region size: {len(self.current_region.nodes)}"
        )

    def _commit_region(self):
        """Finalize and store the current region being built."""
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
        """Build a region from a linear sequence of nodes."""
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
        """Process a single node and create appropriate region(s) based on its pattern."""
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
        """Partition the entire graph into non-overlapping LEAF regions."""
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
        """Initialize the refiner with a region to refine."""
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
        """Create a new LEAF region containing specified nodes."""
        region = Region(
            region_id=self.next_region_id, level=self.root.level + 1, region_type=RegionType.LEAF
        )
        self.next_region_id += 1
        for node_idx in node_indices:
            region.add_node(node_idx)
        self.compute_region_boundaries(region)
        return region

    def _build_region_usage_map(self, regions: list[Region]) -> dict[str, list[Region]]:
        """Build mapping from tensor names to regions that consume them."""
        region_usage_map: dict[str, list[Region]] = {}
        for region in regions:
            for tensor_name in region.inputs:
                if tensor_name not in region_usage_map:
                    region_usage_map[tensor_name] = []
                region_usage_map[tensor_name].append(region)
        return region_usage_map

    def _split_sequence_regions(self, root: Region) -> list[Region]:
        """Split a region into smaller sub-regions by merging producer-consumer chains."""
        result_regions: list[Region] = []
        removed_regions: set[int] = set()

        # PHASE 1: Split into Single-Node Regions
        for node_idx in root.get_nodes():
            region = Region(
                region_id=self.next_region_id, level=root.level + 1, region_type=RegionType.LEAF
            )
            region.add_node(node_idx)
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
        """Identify and merge convergence patterns within a region."""
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
        """Execute two-phase region search to partition the graph into hierarchical regions."""
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
