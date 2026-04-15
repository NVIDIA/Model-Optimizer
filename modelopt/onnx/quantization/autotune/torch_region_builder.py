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

"""Torch Region Builder - Hierarchical Region Discovery from PyTorch-exported ONNX Models.

This module provides region building capabilities specifically designed for ONNX models
exported from PyTorch using torch.onnx.export(). It leverages the hierarchical naming
convention in PyTorch-exported node names to create multi-level region structures.
"""

import fnmatch
import logging
from collections import Counter

import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.insertion_points import has_quantizable_operations
from modelopt.onnx.quantization.autotune.region_search import RegionSearchBase

# Module logger
logger = logging.getLogger(__name__)


def check_torch_naming_convention(graph: gs.Graph, threshold: float = 0.8) -> bool:
    """Check if an ONNX graph follows PyTorch's node naming convention.

    PyTorch-exported ONNX models have node names starting with "/" in a
    hierarchical structure like "/module/submodule/operation".

    Args:
        graph: The ONNX graph to check
        threshold: Minimum ratio of nodes with "/" prefix (default: 0.8 = 80%)

    Returns:
        True if the graph follows PyTorch naming conventions
    """
    non_constant_nodes = [n for n in graph.nodes if n.op != "Constant"]
    total = len(non_constant_nodes)
    if total == 0:
        return False

    slash_count = sum(1 for n in non_constant_nodes if n.name and n.name.startswith("/"))
    return (slash_count / total) >= threshold


class TorchRegionBuilder(RegionSearchBase):
    """Region builder that creates hierarchical regions from PyTorch-exported ONNX node names."""

    def __init__(self, graph: gs.Graph):
        """Initialize the TorchRegionBuilder with a computation graph."""
        super().__init__(graph, root=None)
        self.graph.toposort()
        self.regions: list[Region] = []
        self.next_region_id = 0
        self.min_depth = 1
        self.max_depth = None
        self.min_region_size = 1

        self.path_to_nodes: dict[str, list[int]] = {}
        self.path_trie: dict[str, set[str]] = {}
        self.constant_tensor_names: set[str] = self._build_constant_tensor_set()

    def _build_constant_tensor_set(self) -> set[str]:
        """Build a set of tensor names that are produced by Constant nodes."""
        return {
            output.name
            for node in self.graph.nodes
            if node.op == "Constant"
            for output in node.outputs
        }

    def _parse_node_path(self, node_name: str) -> list[str]:
        """Parse a PyTorch-style node name into path components."""
        if not node_name:
            return []
        return [p for p in node_name.split("/") if p]

    def _get_path_at_depth(self, path_parts: list[str], depth: int) -> str:
        """Get the path string at a specific depth."""
        if depth <= 0 or depth > len(path_parts):
            return ""
        return "/" + "/".join(path_parts[:depth])

    def _build_path_trie(self) -> None:
        """Build a trie structure from all node paths for hierarchical region discovery."""
        self.path_to_nodes = {}
        self.path_trie = {"": set()}

        for node_idx, node in enumerate(self.graph.nodes):
            if node.op == "Constant":
                continue
            path_parts = self._parse_node_path(node.name)
            if not path_parts:
                self.path_to_nodes.setdefault("/_misc_", []).append(node_idx)
                self.path_trie[""].add("_misc_")
                continue

            full_depth = len(path_parts)
            if self.max_depth is not None:
                full_depth = min(full_depth, self.max_depth)
            register_depth = max(full_depth - 1, 1)

            register_path = self._get_path_at_depth(path_parts, register_depth)
            self.path_to_nodes.setdefault(register_path, []).append(node_idx)

            for depth in range(1, register_depth + 1):
                parent_path = self._get_path_at_depth(path_parts, depth - 1)
                self.path_trie.setdefault(parent_path, set()).add(path_parts[depth - 1])
                current_path = self._get_path_at_depth(path_parts, depth)
                self.path_trie.setdefault(current_path, set())

    def _collect_nodes_recursive(self, path: str) -> set[int]:
        """Recursively collect all node indices under a path prefix."""
        nodes: set[int] = set()

        if path in self.path_to_nodes:
            nodes.update(self.path_to_nodes[path])

        if path in self.path_trie:
            for child_component in self.path_trie[path]:
                child_path = f"{path}/{child_component}" if path else f"/{child_component}"
                nodes.update(self._collect_nodes_recursive(child_path))

        return nodes

    def _create_region_for_path(
        self, path: str, level: int, parent: Region | None = None
    ) -> Region | None:
        """Create a region for a specific path in the hierarchy."""
        all_nodes = self._collect_nodes_recursive(path)
        if not all_nodes:
            return None

        direct_nodes = set(self.path_to_nodes.get(path, []))

        # Collect child paths and their node sets in one pass
        child_entries: list[tuple[str, set[int]]] = []
        if path in self.path_trie:
            for child_component in sorted(self.path_trie[path]):
                child_path = f"{path}/{child_component}" if path else f"/{child_component}"
                child_entries.append((child_path, self._collect_nodes_recursive(child_path)))

        has_significant_children = any(
            len(nodes) >= self.min_region_size for _, nodes in child_entries
        )

        region = Region(
            region_id=self.next_region_id,
            level=level,
            region_type=RegionType.COMPOSITE if has_significant_children else RegionType.LEAF,
        )
        self.next_region_id += 1
        region.metadata["path"] = path if path else "/"
        region.nodes.update(direct_nodes)

        if parent is not None:
            parent.add_child(region)

        if has_significant_children:
            for child_path, child_nodes in child_entries:
                if len(child_nodes) >= self.min_region_size:
                    self._create_region_for_path(child_path, level + 1, parent=region)
                else:
                    region.nodes.update(child_nodes)
        else:
            region.nodes.update(all_nodes - direct_nodes)

        return region

    def _count_regions(self, region: Region) -> int:
        """Count total regions in hierarchy."""
        count = 1
        for child in region.get_children():
            count += self._count_regions(child)
        return count

    def _compute_all_boundaries(self, region: Region) -> None:
        """Recursively compute boundaries for a region and all its descendants."""
        for child in region.get_children():
            self._compute_all_boundaries(child)

        self._compute_region_boundaries_no_constants(region)

    def _compute_region_boundaries_no_constants(self, region: Region) -> None:
        """Compute input and output tensor boundaries for a region, excluding constant tensors."""
        node_indices = region.get_region_nodes_and_descendants()
        all_inputs: set[str] = set()
        internal_tensors: set[str] = set()
        region_outputs: set[str] = set()
        graph_output_names = {t.name for t in self.graph.outputs}

        for node_idx in node_indices:
            if node_idx >= len(self.graph.nodes):
                continue
            node = self.graph.nodes[node_idx]
            for input_tensor in node.inputs:
                if not isinstance(input_tensor, gs.Constant):
                    all_inputs.add(input_tensor.name)
            for output_tensor in node.outputs:
                name = output_tensor.name
                internal_tensors.add(name)
                consumers = self.tensor_users_map.get(name, [])
                if (
                    not consumers
                    or any(c not in node_indices for c in consumers)
                    or name in graph_output_names
                ):
                    region_outputs.add(name)

        region.inputs = sorted(all_inputs - internal_tensors - self.constant_tensor_names)
        region.outputs = sorted(region_outputs)

    def _sort_regions(self, region: Region) -> None:
        """Sort regions by topological order."""
        region.children = sorted(
            region.children, key=lambda r: max(r.get_region_nodes_and_descendants())
        )
        for child in region.get_children():
            self._sort_regions(child)

    def _build_id_to_region_map(
        self, region: Region, id_to_region_map: dict[int, Region] | None = None
    ) -> dict[int, Region]:
        """Build a map from region ids to regions."""
        if id_to_region_map is None:
            id_to_region_map = {}
        id_to_region_map[region.id] = region
        for child in region.get_children():
            self._build_id_to_region_map(child, id_to_region_map)
        return id_to_region_map

    def _build_tensor_to_regions_map(
        self, region: Region, tensor_to_regions_map: dict[str, set[int]] | None = None
    ) -> dict[str, set[int]]:
        """Build a map from tensor names to regions."""
        if tensor_to_regions_map is None:
            tensor_to_regions_map = {}
        for input in region.inputs:
            if input not in tensor_to_regions_map:
                tensor_to_regions_map[input] = set()
            tensor_to_regions_map[input].add(region.id)

        for child in region.get_children():
            self._build_tensor_to_regions_map(child, tensor_to_regions_map)
        return tensor_to_regions_map

    def _merge_neighboring_regions(self, region: Region, to_remove: set[int] | None = None) -> None:
        if to_remove is None:
            to_remove = set()
        self._compute_all_boundaries(region)
        id_to_region_map = self._build_id_to_region_map(region)
        tensor_to_regions_map = self._build_tensor_to_regions_map(region)
        for child in region.get_children():
            if child.id in to_remove:
                continue
            if child.type == RegionType.COMPOSITE:
                self._merge_neighboring_regions(child, to_remove)
                continue
            outputs = child.outputs
            if len(outputs) != 1:
                continue
            output = outputs[0]
            if output not in tensor_to_regions_map:
                continue
            users_ids = tensor_to_regions_map[output]
            users = [id_to_region_map[user_id] for user_id in users_ids]
            if len(users) != 1:
                continue
            user = users[0]
            if user.type == RegionType.COMPOSITE:
                continue
            if user.id in to_remove:
                continue
            child.merge(user)
            to_remove.add(user.id)
        region.children = [child for child in region.get_children() if child.id not in to_remove]
        self._compute_all_boundaries(region)

    def _flatten_leaf_regions(self, region: Region) -> None:
        """Ensure LEAF regions have no children by absorbing descendant nodes.

        After merging, a LEAF region may end up with orphaned children
        (e.g., when a COMPOSITE parent was dissolved and its children
        were re-parented under a LEAF sibling). This method absorbs all
        descendant nodes into the LEAF and removes the children.
        """
        for child in list(region.get_children()):
            self._flatten_leaf_regions(child)

        if region.type == RegionType.LEAF and region.get_children():
            region.nodes.update(region.get_region_nodes_and_descendants())
            for child in list(region.get_children()):
                region.remove_child(child)

    def _merge_small_composite_regions(self, region: Region, target_region_size: int) -> None:
        """Merge small composite regions into their parent regions."""
        all_nodes = region.get_region_nodes_and_descendants()
        if region.type == RegionType.LEAF:
            return
        elif len(all_nodes) < target_region_size:
            for node_idx in all_nodes:
                region.nodes.add(node_idx)
            for child_to_remove in region.get_children():
                region.remove_child(child_to_remove)
            region.type = RegionType.LEAF
            self._compute_all_boundaries(region)
            return
        for child in region.get_children():
            self._merge_small_composite_regions(child, target_region_size)

    def _move_direct_nodes_to_children(self, region: Region) -> None:
        """Move direct nodes in COMPOSITE regions into new child regions.

        For each COMPOSITE region that has direct nodes, this method:
        1. Creates a new LEAF child region
        2. Moves all direct nodes to the new child
        3. Recursively processes all children

        Args:
            region: The region (or region hierarchy) to process
        """
        for child in region.get_children():
            self._move_direct_nodes_to_children(child)

        if region.type != RegionType.COMPOSITE:
            return

        direct_nodes = region.get_nodes()
        if not direct_nodes:
            return

        logger.debug(
            f"Moving {len(direct_nodes)} direct nodes from COMPOSITE region {region.id} to new child"
        )

        new_child = Region(
            region_id=self.next_region_id,
            level=region.level + 1,
            region_type=RegionType.LEAF,
        )
        self.next_region_id += 1

        parent_path = region.metadata.get("path", "")
        new_child.metadata["path"] = f"{parent_path}/__direct__"
        for node_idx in direct_nodes:
            new_child.nodes.add(node_idx)

        region.nodes.clear()
        region.add_child(new_child)

        logger.debug(f"Created new LEAF child region {new_child.id} with {len(direct_nodes)} nodes")

    @staticmethod
    def is_quantizable_node(op_type: str) -> bool:
        """Check if a node is quantizable."""
        return op_type in {
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

    _NON_FUSIBLE_OPS = frozenset(
        {
            # Math / activation
            "Div",
            "Sqrt",
            "Pow",
            "Neg",
            "Log",
            "Exp",
            "Erf",
            "Softmax",
            "Clip",
            # Normalization (not fused with quantized kernels by TRT)
            "LayerNormalization",
            "BatchNormalization",
            "InstanceNormalization",
            "GroupNormalization",
            # Type / constant
            "Cast",
            "Constant",
            # Layout / shape
            "Transpose",
            "Reshape",
            "Squeeze",
            "Unsqueeze",
            "Split",
            "Expand",
            "Slice",
            "Concat",
            "Shape",
            "Flatten",
            # Gather / scatter
            "Gather",
            "GatherND",
            "GatherElements",
            "Scatter",
            "ScatterND",
            "GridSample",
            # Reduction
            "ReduceMean",
            "ReduceMax",
            "ReduceSum",
            "ArgMax",
            "ArgMin",
            # Comparison / logic
            "Equal",
            "Greater",
            "GreaterOrEqual",
            "Less",
            "LessOrEqual",
            "Where",
            "And",
            "Or",
            "Xor",
            "Not",
        }
    )

    @staticmethod
    def is_fusible_node(op_type: str) -> bool:
        """Check if a node is fusible (not in the non-fusible op set)."""
        return op_type not in TorchRegionBuilder._NON_FUSIBLE_OPS

    def _has_quantizable_upstream(self, node: gs.Node, max_steps: int = 5) -> bool:
        """Check if a node has a quantizable upstream within max_steps hops."""
        if max_steps <= 0:
            return False
        if self.is_quantizable_node(node.op):
            return True
        for input_tensor in node.inputs:
            if hasattr(input_tensor, "inputs") and input_tensor.inputs:
                if self._has_quantizable_upstream(input_tensor.inputs[0], max_steps - 1):
                    return True
        return False

    def _probe_epilogues_recursive(
        self, node_idx: int, current_step: int, max_steps: int, epilogue_ops: list[int]
    ) -> None:
        """Recursively probe forward to find fusible non-divergent epilogue nodes."""
        if current_step >= max_steps or node_idx >= len(self.graph.nodes):
            return
        if self._is_node_divergent(node_idx):
            return

        consumer_indices = [
            idx
            for output in self.graph.nodes[node_idx].outputs
            for idx in self.tensor_users_map.get(output.name, [])
        ]

        for consumer_idx in consumer_indices:
            if consumer_idx >= len(self.graph.nodes):
                continue
            epilogue_ops.append(consumer_idx)
            # Recurse into fusible consumers; non-fusible consumers are included
            # as boundary nodes (their inputs are Q/DQ insertion points) but not
            # recursed past.
            if self.is_fusible_node(self.graph.nodes[consumer_idx].op):
                self._probe_epilogues_recursive(
                    consumer_idx, current_step + 1, max_steps, epilogue_ops
                )

    def _probe_epilogues(self, region: Region, max_steps: int = 3) -> None:
        """Probe forward from leaf outputs to find fusible non-divergent epilogue nodes.

        Epilogue nodes are added to the leaf region to create better fusion patterns.
        Nodes can be included in multiple regions to optimize fusion opportunities.
        """
        for child in region.get_children():
            self._probe_epilogues(child, max_steps)
        if region.type != RegionType.LEAF:
            return
        region_nodes = region.get_nodes()
        if not region_nodes:
            return

        epilogue_ops: list[int] = []
        for node_idx in region_nodes:
            self._probe_epilogues_recursive(node_idx, 0, max_steps, epilogue_ops)

        if epilogue_ops:
            region.nodes.update(epilogue_ops)
            logger.debug(f"Region {region.id}: added {len(epilogue_ops)} epilogue nodes")

    def _filter_out_non_quantizable_nodes(self, region: Region) -> None:
        """Filter out non-quantizable nodes from regions recursively."""
        for child in region.get_children():
            self._filter_out_non_quantizable_nodes(child)
        nodes_to_remove = []
        for node_idx in region.get_nodes():
            if node_idx >= len(self.graph.nodes):
                nodes_to_remove.append(node_idx)
            elif not self.is_fusible_node(self.graph.nodes[node_idx].op):
                if not self._has_quantizable_upstream(self.graph.nodes[node_idx], max_steps=2):
                    nodes_to_remove.append(node_idx)
            elif not self._has_quantizable_upstream(self.graph.nodes[node_idx]):
                nodes_to_remove.append(node_idx)
        for node_idx in nodes_to_remove:
            region.nodes.remove(node_idx)

    def _add_boundary_consumers(self, region: Region) -> None:
        """Add immediate consumers of surviving nodes as Q/DQ insertion boundaries.

        After filtering removes non-quantizable nodes, the region's output edges
        lead to nodes outside the region. The autotuner searches node inputs for
        Q/DQ insertion points, so these immediate consumers must be in the region
        for their inputs to be visible as insertion candidates.

        E.g., MatMul → Add → Reshape: Reshape is non-fusible but its input
        (Add's output) is a natural Q/DQ insertion point.
        """
        for child in region.get_children():
            self._add_boundary_consumers(child)
        if region.type != RegionType.LEAF:
            return

        boundary = set()
        for node_idx in list(region.get_nodes()):
            if node_idx >= len(self.graph.nodes):
                continue
            for output in self.graph.nodes[node_idx].outputs:
                for consumer_idx in self.tensor_users_map.get(output.name, []):
                    if consumer_idx not in region.nodes and consumer_idx < len(self.graph.nodes):
                        boundary.add(consumer_idx)
        region.nodes.update(boundary)

    def _remove_empty_regions(self, region: Region) -> None:
        """Remove leaf regions that are empty or have no quantizable backbone ops.

        After node filtering, some regions may have zero nodes or only contain
        non-quantizable ops (e.g., a standalone LayerNormalization). These regions
        would never benefit from Q/DQ insertion, so remove them.
        Also removes COMPOSITE regions whose children were all removed.
        """
        for child in list(region.get_children()):
            self._remove_empty_regions(child)

        children_to_remove = []
        for child in region.get_children():
            child_nodes = child.get_region_nodes_and_descendants()
            if not child_nodes:
                children_to_remove.append(child)
                continue
            if child.type == RegionType.LEAF:
                has_backbone = any(
                    self.is_quantizable_node(self.graph.nodes[idx].op)
                    for idx in child_nodes
                    if idx < len(self.graph.nodes)
                )
                if not has_backbone:
                    children_to_remove.append(child)

        for child in children_to_remove:
            region.remove_child(child)

    def torch_node_ratio(self) -> float:
        """Return the fraction of non-Constant nodes with PyTorch-style '/' names."""
        non_constant_nodes = [n for n in self.graph.nodes if n.op != "Constant"]
        if not non_constant_nodes:
            return 0.0
        slash_count = sum(1 for n in non_constant_nodes if n.name and n.name.startswith("/"))
        return slash_count / len(non_constant_nodes)

    def _linearize_regions(self, region: Region) -> list[Region]:
        """Linearize the regions into a list using DFS post-order traversal.

        Visits regions in depth-first order, with leaf regions added before their parent
        composite regions. This ordering is used by the autotuner to tune QDQ insertion points.

        Args:
            region: The root region to linearize

        Returns:
            List of regions in post-order (leaves first, then composites)
        """
        result = []
        for child in region.get_children():
            result.extend(self._linearize_regions(child))
        # only keep leaf regions and innermost composite regions
        if region.type == RegionType.LEAF or all(r.type == RegionType.LEAF for r in result):
            result.append(region)
        return result

    def linearize_regions(self) -> list[Region]:
        """Linearize the regions into a list using DFS post-order traversal."""
        result = []
        for child in self.regions:
            result.extend(self._linearize_regions(child))
        return result

    def build_regions(self, linearize: bool = True, only_quantizable: bool = False) -> list[Region]:
        """Build hierarchical regions from PyTorch-style node names."""
        logger.info(f"Building regions from PyTorch node names ({len(self.graph.nodes)} nodes)")

        self._build_path_trie()

        root_region = self._create_region_for_path("", level=0)
        if root_region is None:
            self.regions = []
            return self.regions

        self._move_direct_nodes_to_children(root_region)
        self._sort_regions(root_region)

        for _ in range(10):
            self._merge_neighboring_regions(root_region)
            self._merge_small_composite_regions(root_region, target_region_size=12)

        self._flatten_leaf_regions(root_region)
        self._probe_epilogues(root_region)

        root_region.type = RegionType.ROOT
        self.regions = [root_region]
        self._compute_all_boundaries(root_region)
        self._sort_regions(root_region)
        if only_quantizable:
            self._filter_out_non_quantizable_nodes(root_region)
            self._add_boundary_consumers(root_region)
            self._remove_empty_regions(root_region)
            self._compute_all_boundaries(root_region)
        logger.info(f"Created region hierarchy: {self._count_regions(root_region)} total regions")

        return self.linearize_regions() if linearize else self.regions

    def search_regions_at_depth(self, depth: int) -> list[Region]:
        """Get all regions at a specific depth in the hierarchy."""
        result: list[Region] = []

        def collect_at_depth(region: Region, current_depth: int):
            if current_depth == depth:
                result.append(region)
            elif current_depth < depth:
                for child in region.get_children():
                    collect_at_depth(child, current_depth + 1)

        for region in self.regions:
            collect_at_depth(region, 0)

        return result

    def search_regions_by_path(self, pattern: str) -> list[Region]:
        """Search for regions matching a path pattern."""
        result: list[Region] = []

        def collect_matching(region: Region):
            path = region.metadata.get("path", "")
            if fnmatch.fnmatch(path, pattern):
                result.append(region)
            for child in region.get_children():
                collect_matching(child)

        for region in self.regions:
            collect_matching(region)

        return result


def inspect_torch_regions(
    onnx_path: str,
    include_all_regions: bool = False,
    only_quantizable: bool = False,
) -> list[Region]:
    """Inspect region discovery using PyTorch-style node naming for an ONNX model.

    Args:
        onnx_path: Path to the ONNX model file (should be exported from PyTorch)
        include_all_regions: Include all regions, even those without quantizable ops
        only_quantizable: Only include quantizable regions

    Returns:
        List of discovered regions with hierarchical structure
    """
    logger.info(f"Loading model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    graph = gs.import_onnx(onnx_model)
    graph.cleanup().toposort()

    builder = TorchRegionBuilder(graph)
    logger.info(f"PyTorch naming ratio: {builder.torch_node_ratio():.2f}")
    regions = builder.build_regions(only_quantizable=only_quantizable)

    if not include_all_regions:
        for region in regions:
            for child in [
                c for c in region.get_children() if not has_quantizable_operations(c, graph)
            ]:
                region.remove_child(child)

    all_nodes = set()
    for region in regions:
        all_nodes.update(region.get_region_nodes_and_descendants())
    coverage_pct = 100 * len(all_nodes) / len(graph.nodes) if graph.nodes else 0

    type_counts = Counter(r.type for r in regions)
    logger.info(
        f"{len(regions)} regions ({type_counts[RegionType.LEAF]} LEAF, "
        f"{type_counts[RegionType.COMPOSITE]} COMPOSITE), "
        f"coverage: {len(all_nodes)}/{len(graph.nodes)} ({coverage_pct:.1f}%)"
    )

    # Print region tree for the root (non-linearized hierarchy)
    if builder.regions:
        for root in builder.regions:
            builder.print_tree(root)

    return regions


def main():
    """Command-line entry point for TorchRegionBuilder inspection."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune.torch_region_builder",
        description="Build hierarchical regions from PyTorch-exported ONNX models",
    )
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--include-all-regions", action="store_true", help="Include all regions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--only-quantizable", action="store_true", help="Only include quantizable regions"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)

    try:
        regions = inspect_torch_regions(args.model, args.include_all_regions, args.only_quantizable)
        logger.info(f"✓ Inspection complete: {len(regions)} top-level regions")
        return 0
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
