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

"""Automatic Q/DQ insertion optimization for ONNX models via pattern-based profiling."""

from collections import Counter, deque

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.autotuner_base import QDQAutotunerBase
from modelopt.onnx.quantization.autotune.common import Config, PatternCache, Region, RegionType
from modelopt.onnx.quantization.autotune.region_search import CombinedRegionSearch


class QDQAutotuner(QDQAutotunerBase):
    """Q/DQ autotuner with automatic region discovery around compute-intensive ops."""

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuner and discover optimization regions automatically.

        Extends base class initialization by automatically searching for regions
        after configuration is set up. Regions are discovered using pattern-based
        search around compute-intensive operations.
        """
        super().initialize(config, pattern_cache)
        self._search_regions()

    @staticmethod
    def _visit_region_recursively(region: Region) -> list[Region]:
        """Recursively traverse region hierarchy and collect all regions.

        Performs depth-first traversal of the region tree starting from a given
        region. Collects the root region and all descendant regions (children,
        grandchildren, etc.) into a flat list.

        Args:
            region: Root region to start traversal from

        Returns:
            List of all regions in the subtree (including root), in pre-order DFS.
        """
        regions = [region]

        for child in region.get_children():
            regions.extend(QDQAutotuner._visit_region_recursively(child))

        return regions

    def _reassign_region_ids(self, regions: list[Region]) -> None:
        """Reassign sequential IDs to regions in breadth-first order.

        Traverses the region hierarchy (including children) and assigns new
        sequential IDs starting from 0. This ensures clean, predictable region
        numbering after region discovery and manipulation.

        Args:
            regions: List of top-level regions (children will be processed too)
        """
        region_id = 0

        queue = deque(regions)

        while queue:
            region = queue.popleft()
            region.id = region_id
            region_id += 1
            queue.extend(region.get_children())

    def _search_regions(self) -> None:
        """Discover and organize optimization regions automatically.

        This is the core region discovery method that:
        1. Runs automatic region search to find optimization targets
        2. Flattens hierarchical structure into a list
        3. Prioritizes LEAF regions (contain actual nodes)
        4. Reassigns IDs for clean indexing

        **Search Strategy:**
        Uses CombinedRegionSearch which performs:
        - Phase 1: Bottom-up partitioning based on divergence/convergence
        - Phase 2: Top-down refinement creating hierarchical structure
        """
        logger.info("Discovering optimization regions")
        search = CombinedRegionSearch(
            self.graph,
            maximum_sequence_region_size=self.config.maximum_sequence_region_size,
            minimum_topdown_search_size=self.config.minimum_topdown_search_size,
        )
        self.regions = search.search_regions()
        self._reassign_region_ids(self.regions)
        logger.debug(f"Found {len(self.regions)} top-level regions")

        # Flatten the hierarchy into a list of all regions
        all_regions = []
        for region in self.regions:
            all_regions.extend(QDQAutotuner._visit_region_recursively(region))

        all_regions.sort(key=lambda r: r.type != RegionType.LEAF)
        self.regions = all_regions

        type_counts = Counter(r.type for r in self.regions)
        logger.info(
            f"Discovery complete: {len(self.regions)} regions "
            f"({type_counts[RegionType.LEAF]} LEAF, {type_counts[RegionType.COMPOSITE]} COMPOSITE, "
            f"{type_counts[RegionType.ROOT]} ROOT)"
        )
        logger.debug("Regions prioritized: LEAF regions first for profiling")
