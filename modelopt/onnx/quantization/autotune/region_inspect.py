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

"""Region search inspection tool for ONNX models."""

import argparse
import logging
import sys
from collections import Counter

import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.common import Region, RegionType
from modelopt.onnx.quantization.autotune.insertion_points import has_quantizable_operations
from modelopt.onnx.quantization.autotune.region_search import (
    DEFAULT_MAX_STEPS,
    CombinedRegionSearch,
)


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
        region.children = [
            c
            for c in region.get_children()
            if include_all_regions or has_quantizable_operations(c, graph)
        ]
        if not include_all_regions and not has_quantizable_operations(region, graph):
            logger.debug(f"Filtered out region {i} (no quantizable operations)")
            continue
        logger.debug(
            f"Region {i}: {region.type.value}, {len(region.get_region_nodes_and_descendants())} nodes, "
            f"{len(region.inputs)} inputs, {len(region.outputs)} outputs"
        )
        all_regions.append(region)
        if region.type == RegionType.COMPOSITE:
            logger.debug(f"  {len(region.get_children())} child regions")
            all_regions.extend(region.get_children())
        combined_search.print_tree(region, indent=2)

    # Summary statistics
    type_counts = Counter(r.type for r in all_regions)
    leaf_regions, composite_regions = (
        type_counts[RegionType.LEAF],
        type_counts[RegionType.COMPOSITE],
    )

    all_nodes = {n for r in all_regions for n in r.get_region_nodes_and_descendants()}
    total_nodes = len(all_nodes)
    coverage_pct = 100 * total_nodes / len(graph.nodes) if graph.nodes else 0

    logger.info(
        f"Summary: {len(all_regions)} regions ({leaf_regions} LEAF, {composite_regions} COMPOSITE), "
        f"{total_nodes}/{len(graph.nodes)} nodes ({coverage_pct:.1f}%)"
    )

    # Print histogram of region sizes
    region_sizes = [
        len(r.get_region_nodes_and_descendants()) for r in all_regions if r.type == RegionType.LEAF
    ]

    if region_sizes:
        min_size = min(region_sizes)
        max_size = max(region_sizes)
        avg_size = sum(region_sizes) / len(region_sizes)

        logger.info(f"LEAF region sizes: min={min_size}, max={max_size}, avg={avg_size:.1f}")
        size_counts = Counter(region_sizes)
        logger.debug("Size distribution:")
        for size in sorted(size_counts.keys()):
            count = size_counts[size]
            bar = "█" * min(count, 50)
            logger.debug(f"  {size:4d} nodes: {bar} ({count} regions)")

    return all_regions


def main():
    """Command-line entry point for region search inspection."""
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune.region_inspect",
        description="Inspect region search results for ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx

  # Verbose mode for debug logging
  python -m modelopt.onnx.quantization.autotune.region_inspect \\
      --model model.onnx --verbose

  # Custom maximum sequence size
  python -m modelopt.onnx.quantization.autotune.region_inspect \\
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

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)

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
