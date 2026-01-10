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

"""QDQ Autotuner - Automatic Q/DQ Insertion Optimization for ONNX Models.

This module provides pattern-based automatic optimization of Quantize/Dequantize
(Q/DQ) node placement in ONNX computation graphs using iterative profiling and
performance measurement.

**Core Functionality:**
- Identifies regions around compute-intensive operations (Conv, MatMul, Gemm, etc.)
- Generates and tests multiple Q/DQ insertion schemes per region pattern
- Measures performance and selects optimal configurations
- Applies best schemes to all regions matching each pattern

**Pattern-Based Optimization:**
- Regions with identical structure share the same pattern signature
- Each pattern gets multiple InsertionScheme candidates tested
- Schemes use pattern-relative addressing (portable across matching regions)
- Best scheme per pattern applies to all regions with that structure

**Typical Workflow:**
1. Initialize autotuner with ONNX model → regions discovered automatically
2. Measure baseline performance (optional but recommended)
3. For each region: generate schemes → export → measure → submit results
4. Export optimized model with best schemes applied

**Classes:**
- QDQAutotuner: Default autotuner with automatic region discovery (use this)
- QDQAutotunerBase: Base class for custom region identification strategies
"""

import copy
import logging
import os
import random
from collections import deque
from datetime import datetime, timezone

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import yaml

from modelopt.onnx.quantization.autotune.common import (
    AutotunerNotInitializedError,
    Config,
    InsertionScheme,
    InvalidSchemeError,
    PatternCache,
    PatternSchemes,
    Region,
    RegionType,
)
from modelopt.onnx.quantization.autotune.insertion_points import (
    ResolvedInsertionPoint,
    merge_resolved_insertion_points,
)
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.autotune.region_search import CombinedRegionSearch
from modelopt.onnx.quantization.fp8 import int8_to_fp8
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

# Module logger
logger = logging.getLogger(__name__)


class QDQAutotunerBase:
    """Base class for pattern-based Q/DQ node insertion optimization in ONNX models.

    This base class provides core functionality for optimizing Quantize/Dequantize (Q/DQ)
    node placement in ONNX models. It orchestrates scheme generation, performance profiling,
    and model export, using pattern-based optimization where regions with identical structure
    share insertion schemes.

    **Design:**
    - Subclasses must populate `self.regions` with Region objects (e.g., via region search)
    - Pattern-relative addressing: Schemes use indices relative to region structure
    - Performance-driven selection: Measures and compares scheme latencies
    - Best scheme per pattern: Optimal configuration applies to all matching regions

    **Key Attributes:**
    - graph: ONNX GraphSurgeon representation of the model (clean copy)
    - onnx_model: Original ONNX protobuf model
    - regions: List of regions to optimize (populated by subclass)
    - profiled_patterns: Patterns with tested schemes and performance results
    - current_profile_region: Currently active region being tested
    - current_profile_pattern_schemes: Currently active pattern schemes for the region
    - config: Configuration for Q/DQ parameters and autotuning behavior
    - pattern_cache: Pattern cache data for seeding schemes and tracking best results

    **Public API:**
    - initialize(): Set up configuration and prepare for profiling
    - set_profile_region(): Select region to profile and generate schemes
    - generate(): Create new insertion scheme for current region
    - export_onnx(): Export model with Q/DQ nodes (test scheme or best schemes)
    - submit(): Record performance measurement for current scheme
    - save_state(): Persist profiling results to file
    - load_state(): Resume from previous session

    **Typical Workflow:**
    1. Subclass populates regions during/after initialization
    2. Measure baseline performance without Q/DQ
    3. For each region: set_profile_region() → generate() → export() → submit()
    4. Export final optimized model with best schemes

    **Note:**
    Most users should use QDQAutotuner (subclass) which automatically searches for
    regions based on common operations. Use QDQAutotunerBase directly only for
    custom region identification strategies.
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(self, model: onnx.ModelProto | gs.Graph):
        """Initialize the autotuner with an ONNX model.

        Creates a clean copy of the model graph and initializes internal state.
        After construction, call initialize() to configure the autotuner, then
        use a subclass strategy to populate regions (e.g., QDQAutotuner does this
        automatically during initialize()).

        Args:
            model: ONNX model (onnx.ModelProto) or graph (gs.Graph) to optimize.
                   A clean copy is created internally, leaving the original unchanged.

        Raises:
            TypeError: If model is neither onnx.ModelProto nor gs.Graph

        Example:
            >>> # Most users should use QDQAutotuner subclass
            >>> autotuner = QDQAutotuner(model)
            >>> autotuner.initialize()
        """
        # Store ONNX model representation (needed for graph copying)
        if isinstance(model, onnx.ModelProto):
            self.onnx_model = model
        elif isinstance(model, gs.Graph):
            self.onnx_model = gs.export_onnx(model)
        else:
            raise TypeError(f"Expected onnx.ModelProto or gs.Graph, got {type(model)}")

        # Create clean graph copy (modifications won't affect original)
        self.graph = self._copy_graph()
        self.graph.tensor_users_map = get_tensor_consumer_node_indices(self.graph)

        # Region state (populated by subclass during/after initialize)
        self.regions: list[Region] = []
        self.current_profile_region: Region | None = None

        # Pattern profiling state
        self.profiled_patterns: list[PatternSchemes] = []
        self.current_profile_pattern_schemes: PatternSchemes | None = None

        # Current insertion scheme index (for generating new schemes)
        self.current_insertion_scheme_index: int | None = None

        # Configuration (set properly in initialize())
        self.config = Config()

        # Session state
        self.initialized = False
        self.baseline_latency_ms: float | None = None

        # Pattern cache data (set in initialize())
        self.pattern_cache: PatternCache | None = None

        logger.debug(f"Initialized autotuner with model type: {type(model).__name__}")

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuning session with configuration and pattern cache.

        Prepares the autotuner for profiling by setting configuration parameters
        and optionally loading pattern cache data. This base method resets all profiling
        state and sets up the pattern cache storage.

        **Note:** This base class does NOT populate regions. Subclasses (e.g., QDQAutotuner)
        should override this method to add region discovery after calling super().initialize().

        After initialization, populate self.regions (in subclass), then use
        set_profile_region() to begin testing schemes.

        Args:
            config: Autotuning configuration parameters. If None, uses default Config().
                   Controls Q/DQ parameters, performance thresholds, and scheme generation.
            pattern_cache: Optional PatternCache object for seeding with known-good schemes.
                        If None, creates a new empty pattern cache for tracking best schemes.
                        If provided, uses existing schemes to warm-start optimization.

        Raises:
            None (safe to call multiple times - will reset state each time)

        Example:
            >>> # In subclass
            >>> def initialize(self, config=None, pattern_cache=None):
            >>>     super().initialize(config, pattern_cache)
            >>> # Add region discovery here
            >>>     self._search_regions()
        """
        # Apply user configuration
        if config is not None:
            self.config = config

        # Set up pattern cache (for seeding schemes and tracking best results)
        if pattern_cache is None:
            # Create empty pattern cache with config settings
            self.pattern_cache = PatternCache(
                minimum_distance=self.config.pattern_cache_minimum_distance,
                max_entries_per_pattern=self.config.pattern_cache_max_entries_per_pattern,
            )
        else:
            # Use provided pattern cache for warm-start
            self.pattern_cache = pattern_cache
            logger.debug(
                f"Loaded pattern cache with {pattern_cache.num_patterns} patterns and "
                f"{pattern_cache.total_schemes} schemes"
            )

        # Reset all profiling state (safe to call multiple times)
        self.initialized = False
        self.baseline_latency_ms = None
        self.profiled_patterns.clear()
        self.regions.clear()
        self.current_profile_region = None
        self.current_profile_pattern_schemes = None
        self.current_insertion_scheme_index = None

        logger.info("Initializing autotuner")
        logger.debug(
            f"Configuration: q_scale={self.config.default_q_scale}, "
            f"q_zero_point={self.config.default_q_zero_point}, quant_type={self.config.default_quant_type}"
        )

        # Mark as initialized
        self.initialized = True

    def set_profile_region(
        self, region: Region | None, commit: bool = True, per_region: bool = False
    ) -> None:
        """Set the target region for profiling and scheme generation.

        This method manages the profiling workflow:
        1. If commit=True: Saves current schemes to profiled_patterns
        2. Creates a RegionPattern from the new region's structure
        3. For pattern-based: tries to seed schemes from pattern cache if available
        4. Sets as current for generate() and submit() calls

        Pass region=None to clear the current profile target without setting a new one.

        **Workflow Pattern:**
        - Call with commit=True (default) when moving between regions
        - This commits previous region's results before starting new one
        - Call with commit=False during initialization to avoid empty commits

        **Pattern Cache:**
        - Automatically seeds from pattern cache if available for this pattern (pattern-based only)
        - Remove profile result of seeded schemes (need profiling)
        - If pattern already profiled, skips it

        Args:
            region: The region to profile next (None to clear current target)
            commit: If True, commit current schemes to profiled_patterns
                   before switching. Set to False during initialization.

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called

        Example:
            >>> # Pattern-based optimization (default)
            >>> region = autotuner.regions[0]
            >>> autotuner.set_profile_region(region)
            >>> autotuner.generate()  # Creates schemes for pattern
            >>> # Per-region optimization
            >>> region = autotuner.regions[1]
            >>> autotuner.set_profile_region(region, per_region=True)
            >>> autotuner.generate()  # Creates schemes for this region only
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        # Commit current pattern if requested
        if commit:
            if self.current_profile_pattern_schemes is not None:
                num_schemes = len(self.current_profile_pattern_schemes.schemes)
                best_scheme = self.current_profile_pattern_schemes.best_scheme
                best_latency = best_scheme.latency_ms if best_scheme else float("inf")

                # Compute convergence metrics
                samples_before_best, time_to_best = self._compute_convergence_metrics(
                    self.current_profile_pattern_schemes.schemes, best_scheme
                )

                logger.info(
                    f"Pattern complete: {num_schemes} schemes tested, best latency {best_latency:.3f} ms"
                )
                logger.debug(
                    f"Pattern signature: {self.current_profile_pattern_schemes.pattern_signature}"
                )
                if samples_before_best is not None:
                    logger.debug(f"Convergence: best found at sample {samples_before_best}")
                if time_to_best is not None:
                    logger.debug(f"Time to best: {time_to_best:.2f}s")
                self.profiled_patterns.append(self.current_profile_pattern_schemes)

        if commit or region is None:
            self.current_profile_region = None
            self.current_profile_pattern_schemes = None
            self.current_insertion_scheme_index = None
            if region is None:
                return

        # Validate region
        if region not in self.regions:
            raise ValueError(f"Region {region.id} not found in regions")

        # Create pattern for this region
        region_pattern = RegionPattern.from_region(region, self.graph)

        # Check if pattern is already profiled - skip if so
        if self._is_region_profiled(region):
            logger.info(f"Skipping region {region.id} (pattern already profiled)")
            logger.debug(f"Pattern signature: {region_pattern.signature}")
            return

        # Try to seed from pattern cache
        pattern_schemes = None
        num_seeded = 0

        if self.pattern_cache is not None:
            cache_schemes = self.pattern_cache.get_pattern_schemes(region_pattern.signature)

            if cache_schemes is not None and len(cache_schemes.schemes) > 0:
                # Create new PatternSchemes and seed it
                pattern_schemes = PatternSchemes()
                pattern_schemes.pattern = region_pattern

                # Copy schemes from pattern cache and erase profile data
                for cached_scheme in cache_schemes.schemes:
                    scheme_copy = copy.deepcopy(cached_scheme)
                    scheme_copy.latency_ms = float("inf")
                    scheme_copy.error = False
                    pattern_schemes.schemes.append(scheme_copy)
                    num_seeded += 1

                logger.debug(f"Seeded {num_seeded} scheme(s) from pattern cache")
            else:
                logger.debug("No pattern cache entries for this region")

        # Create empty PatternSchemes if not seeded from pattern cache
        if pattern_schemes is None:
            pattern_schemes = PatternSchemes()
            pattern_schemes.pattern = region_pattern
            logger.debug("Initialized with empty scheme collection")

        # Set current region
        self.current_profile_region = region

        # Set pattern schemes
        self.current_profile_pattern_schemes = pattern_schemes
        mode_info = f"seeded with {num_seeded} schemes" if num_seeded > 0 else "starting fresh"
        logger.info(
            f"Profiling region {region.id} [pattern mode, level {region.get_level()}, "
            f"size {region.get_size()}, {mode_info}]"
        )
        logger.debug(f"Pattern signature: {region_pattern.signature}")

    def generate(self) -> int:
        """Generate a new Q/DQ insertion scheme for the current pattern or region.

        Creates a new InsertionScheme by mutating the top-performing schemes:
        1. Checks if there are any cached schemes (error=False, latency_ms=inf)
        2. If cached schemes exist, picks one to re-profile
        3. Otherwise, generates a new scheme by mutation
        4. Selects a random scheme from the top 10 performers
        5. Mutates it by adding/removing insertion points
        6. Ensures the new scheme is unique (different from existing schemes)
        7. Adds the scheme to current_profile_pattern_schemes

        The generated scheme includes both:
        - node_inputs: Q/DQ at node inputs
        - child_region_inputs: Q/DQ at child region boundaries (COMPOSITE only)

        After calling generate(), use export_onnx() to create a test model and
        submit() to record its performance.

        Returns:
            Index of the newly generated scheme in the active schemes collection,
            or -1 if unable to generate a unique scheme after 100 attempts

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
            InvalidSchemeError: If no region is currently set for profiling
                               (call set_profile_region() first)

        Example:
            >>> autotuner.set_profile_region(region)
            >>> # Generate and test multiple schemes
            >>> for i in range(10):
            >>>     scheme_idx = autotuner.generate()
            >>>     if scheme_idx < 0:
            >>>         print("No more unique schemes")
            >>>         break
            >>>     autotuner.export_onnx(f"test_{i}.onnx")
            >>>     latency = benchmark(f"test_{i}.onnx")
            >>>     autotuner.submit(latency)
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        # Determine which schemes collection is active (mutually exclusive)
        if self.current_profile_pattern_schemes is not None:
            schemes_collection = self.current_profile_pattern_schemes
        else:
            raise InvalidSchemeError(
                "No pattern or region selected. Call set_profile_region() first."
            )

        pattern_schemes = schemes_collection

        # Check if there are any cached schemes (from pattern cache or previous runs)
        cached_schemes = [
            (idx, scheme)
            for idx, scheme in enumerate(pattern_schemes.schemes)
            if not scheme.is_profiled
        ]

        if cached_schemes:
            # Re-profile a cached scheme
            scheme_index, cached_scheme_data = cached_schemes[0]

            num_node_points = len(cached_scheme_data.node_inputs)
            num_region_composite_points = len(cached_scheme_data.child_region_inputs)
            num_region_output_points = len(cached_scheme_data.region_outputs)
            total_points = num_node_points + num_region_composite_points + num_region_output_points
            logger.info(
                f"Scheme #{scheme_index + 1}: profiling cached scheme ({total_points} Q/DQ points)"
            )
            logger.debug(
                f"Cached scheme breakdown: {num_node_points} node input, "
                f"{num_region_composite_points} region composite, "
                f"{num_region_output_points} region output points ({len(cached_schemes)} cached schemes remaining)"
            )

            self.current_insertion_scheme_index = scheme_index
            return self.current_insertion_scheme_index

        # Generate a new scheme by mutation
        # Collect known scheme hashes to avoid duplicates
        known_schemes = {scheme.hash for scheme in pattern_schemes.schemes}
        logger.debug(f"Generating new scheme ({len(pattern_schemes.schemes)} schemes exist)")

        max_attempts = getattr(self.config, "maximum_generation_attempts", 100)

        for attempts in range(max_attempts):
            new_scheme = self._generate_next_insertion_sample()

            if new_scheme.hash not in known_schemes and not new_scheme.error:
                # Found a unique, valid scheme
                pattern_schemes.schemes.append(new_scheme)
                scheme_index = len(pattern_schemes.schemes) - 1

                num_node_points = len(new_scheme.node_inputs)
                num_region_composite_points = len(new_scheme.child_region_inputs)
                num_region_output_points = len(new_scheme.region_outputs)
                total_points = (
                    num_node_points + num_region_composite_points + num_region_output_points
                )
                logger.info(
                    f"Scheme #{scheme_index + 1}: generated new scheme ({total_points} Q/DQ points)"
                )
                logger.debug(
                    f"Scheme breakdown: {num_node_points} node input, "
                    f"{num_region_composite_points} region composite, "
                    f"{num_region_output_points} region output points "
                    f"(hash: {new_scheme.hash[:16]}..., attempts: {attempts + 1})"
                )

                self.current_insertion_scheme_index = scheme_index
                return self.current_insertion_scheme_index

        # Failed to generate unique scheme
        logger.warning(f"Could not generate unique scheme after {max_attempts} attempts")
        return -1

    def export_onnx(
        self, output_path: str | None = None, insert_qdq: bool = True, best: bool = False
    ) -> bytes:
        """Export ONNX model with Q/DQ nodes inserted according to tested schemes.

        This method creates a modified version of the model by:
        1. For each region, finding the matching pattern
        2. Applying the best scheme for profiled patterns
        3. Applying the current scheme for the active profile pattern
        4. Resolving pattern-relative insertion points to actual tensor names
        5. Inserting Q/DQ pairs at the resolved locations
        6. Converting to FP8 if needed (always creates INT8 first, then converts)

        **Scheme Selection Logic:**
        - Profiled patterns: Uses best_scheme (lowest latency)
        - Current profile pattern: Uses most recently generated scheme
        - Unmatched patterns: No Q/DQ insertion

        **Parent-Child Coordination:**
        - If a region's parent is profiled, skip inserting Q/DQ at region inputs
        - Parent will handle boundary Q/DQ via CompositeRegionInsertionPoints
        - Prevents duplicate Q/DQ at region boundaries

        Args:
            output_path: Optional file path where the modified ONNX model will be saved.
                        If None, the model is not saved to disk and only bytes are returned.
            insert_qdq: If True, insert Q/DQ nodes. If False, export unmodified model
                       (useful for baseline measurements)

        Returns:
            bytes: Serialized ONNX model as bytes

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called

        Example:
            >>> # Export baseline (no Q/DQ) to file
            >>> model_bytes = autotuner.export_onnx("baseline.onnx", insert_qdq=False)
            >>> # Export with current test scheme to file
            >>> autotuner.generate()
            >>> model_bytes = autotuner.export_onnx("test.onnx", insert_qdq=True)
            >>> # Export only to bytes without saving to file
            >>> model_bytes = autotuner.export_onnx(None, insert_qdq=True)
            >>> # Export final optimized model (all best schemes)
            >>> model_bytes = autotuner.export_onnx("optimized.onnx", insert_qdq=True)
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        output_desc = output_path if output_path is not None else "<bytes>"
        logger.debug(
            f"Exporting model to {output_desc} (insert_qdq={insert_qdq}, "
            f"regions={len(self.regions)}, profiled_patterns={len(self.profiled_patterns)})"
        )

        # Save original quant type for potential FP8 conversion
        original_quant_type = self.config.default_quant_type
        needs_fp8_conversion = insert_qdq and original_quant_type == "fp8"

        # Temporarily set quant type to int8 if FP8 is requested
        if needs_fp8_conversion:
            logger.debug("FP8 conversion: creating INT8 model first")
            self.config.default_quant_type = "int8"

        # =====================================================================
        # Collect Q/DQ Insertion Points from Profiled Schemes
        # =====================================================================
        # For each region, find the matching pattern and apply its best scheme.
        # Pattern matching uses structural signatures, so the same pattern can
        # apply to multiple regions with identical structure.
        resolved_insertion_points = set()

        if insert_qdq:
            logger.debug(f"Resolving Q/DQ insertion points from {len(self.regions)} regions")
            matched_regions = 0

            for region in self.regions:
                # Create pattern signature for this region
                pattern = RegionPattern.from_region(region, self.graph)
                logger.debug(f"Region {region.id} (level {region.level})")
                logger.debug(f"  → Pattern signature: {pattern.signature}")

                current_scheme = None
                for pattern_index, pattern_schemes in enumerate(self.profiled_patterns):
                    if pattern_schemes.pattern == pattern:
                        current_scheme = pattern_schemes.best_scheme
                        if current_scheme:
                            logger.debug(
                                f"  → Matched profiled pattern #{pattern_index} "
                                f"(latency={current_scheme.latency_ms:.3f} ms)"
                            )
                        else:
                            logger.debug(
                                f"  → Matched profiled pattern #{pattern_index} but no valid schemes"
                            )
                        break

                if current_scheme is None:
                    if (
                        self.current_profile_pattern_schemes is None
                        or pattern != self.current_profile_pattern_schemes.pattern
                    ):
                        pass
                    elif best:
                        current_scheme = self.current_profile_pattern_schemes.best_scheme
                    else:
                        scheme_index = self.current_insertion_scheme_index
                        if scheme_index is None:
                            pass
                        else:
                            assert scheme_index < len(
                                self.current_profile_pattern_schemes.schemes
                            ), f"Invalid scheme index: {scheme_index}"
                            current_scheme = self.current_profile_pattern_schemes.schemes[
                                scheme_index
                            ]
                            logger.debug(f"  → Using current pattern scheme #{scheme_index}")

                if current_scheme is None and self.pattern_cache is not None:
                    pattern_schemes = self.pattern_cache.get_pattern_schemes(pattern.signature)
                    if pattern_schemes is not None:
                        schemes = pattern_schemes.schemes
                        if schemes is not None and len(schemes) == 1 and not schemes[0].is_profiled:
                            current_scheme = schemes[0]
                            logger.debug("  → Using imported pattern from cache")

                # -------------------------------------------------------------
                # No matching pattern: skip this region
                # -------------------------------------------------------------
                if current_scheme is None:
                    logger.debug("  → No scheme available, skipping")
                    continue

                # Remove these tensors if they were already added by profiled patterns/regions
                # Current profile pattern has higher priority (more recent results)
                full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)
                assert full_insertion_scheme is not None
                all_region_ips = pattern.matches(region, self.graph, full_insertion_scheme)
                assert isinstance(all_region_ips, set)  # matches returns set when scheme provided
                excluded_tensors = all_region_ips - resolved_insertion_points
                if excluded_tensors:
                    logger.debug(
                        f"  → Excluded {len(excluded_tensors)} overlapping insertion points"
                    )

                resolved_insertion_points.difference_update(all_region_ips)

                # -------------------------------------------------------------
                # Resolve pattern-relative insertion points to tensor names
                # -------------------------------------------------------------
                # Pattern insertion points are relative (e.g., "node 2, input 0").
                # Resolve them to actual tensor names for this specific region.
                new_ips = pattern.matches(region, self.graph, current_scheme)
                assert isinstance(new_ips, set)  # matches returns set when scheme provided
                if new_ips:
                    resolved_insertion_points.update(new_ips)
                    matched_regions += 1
                    logger.debug(f"  → Added {len(new_ips)} insertion points")

            logger.debug(
                f"Matched {matched_regions}/{len(self.regions)} regions, "
                f"total {len(resolved_insertion_points)} unique insertion points"
            )

        # =====================================================================
        # Create Modified Graph with Q/DQ Nodes
        # =====================================================================
        unique_tensors = len(resolved_insertion_points)
        logger.debug(f"Inserting {unique_tensors} Q/DQ pairs into graph")

        # Create fresh graph copy (preserves original)
        graph_copy = self._copy_graph()

        # Insert Q/DQ pairs at all collected tensor locations
        if insert_qdq and resolved_insertion_points:
            self._insert_qdq_at_tensors(graph_copy, resolved_insertion_points)

        # =====================================================================
        # Export to ONNX Format
        # =====================================================================
        logger.debug("Serializing to ONNX format")
        model = gs.export_onnx(graph_copy)

        # ---------------------------------------------------------------------
        # Fix INT8 Zero-Point Initializers
        # ---------------------------------------------------------------------
        # ONNX requires INT8 zero_point to use int32_data field (4-byte aligned)
        # instead of raw_data. This is a quirk of the ONNX format and required
        # for correct INT8 and FP8 conversion.
        if insert_qdq and resolved_insertion_points:
            self._fix_zero_point_initializers(model)

        # ---------------------------------------------------------------------
        # Convert INT8 to FP8 if Requested
        # ---------------------------------------------------------------------
        # FP8 quantization is a two-step process:
        # 1. Create INT8 Q/DQ model (all tools understand INT8)
        # 2. Convert INT8 to FP8 (specialized conversion utility)
        # This approach ensures compatibility with ONNX tooling that may not
        # natively support FP8 yet.
        if needs_fp8_conversion:
            logger.debug("Converting INT8 to FP8")
            model = int8_to_fp8(model)

        # Restore original quantization type in config
        self.config.default_quant_type = original_quant_type

        # Serialize to bytes
        model_bytes = model.SerializeToString()

        # Save to file if output_path is provided
        quant_type_str = "baseline"
        if insert_qdq:
            quant_type_str = f"{original_quant_type.upper()}" if needs_fp8_conversion else "INT8"
        if output_path is not None:
            onnx.save(model, output_path)
            logger.info(
                f"Exported {quant_type_str} model with {unique_tensors} Q/DQ pairs → {output_path}"
            )
        else:
            logger.info(f"Exported {quant_type_str} model with {unique_tensors} Q/DQ pairs")

        return model_bytes

    def submit(self, latency_ms: float, success: bool = True) -> None:
        """Submit performance measurement for the most recently generated scheme.

        This method records the measured latency and manages the optimization state:

        **First Submission (Baseline):**
        - Sets baseline_latency_ms for speedup calculations
        - Does not modify any schemes

        **Subsequent Submissions:**
        - Updates the most recently generated scheme's latency_ms
        - Sets scheme's error flag based on success parameter
        - Computes speedup relative to baseline (if successful)
        - Sorts all schemes by latency (best schemes first)
        - Logs results if config.verbose is True

        **Scheme Sorting:**
        - Schemes are sorted by latency_ms (ascending)
        - Unmeasured schemes (latency_ms = 0) go to the end
        - This ensures best_scheme property returns optimal configuration

        **Optimization Mode:**
        - Automatically detects whether pattern-based or per-region mode is active
        - Commits to the appropriate collection (profiled_patterns)
        - Mode is determined by set_profile_region(per_region=True/False)

        Args:
            latency_ms: Measured latency in milliseconds (must be > 0)
            success: Whether the measurement succeeded. If False, sets scheme.error=True,
                    logs a warning, and skips speedup calculation.

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
            InvalidSchemeError: If no pattern or region is set, or no schemes have been generated

        Example:
            >>> # Submit baseline
            >>> autotuner.export_onnx("baseline.onnx", insert_qdq=False)
            >>> autotuner.submit(benchmark("baseline.onnx"))  # Sets baseline
            >>> # Submit test measurements
            >>> autotuner.set_profile_region(region)
            >>> autotuner.generate()
            >>> autotuner.export_onnx("test.onnx")
            >>> latency = benchmark("test.onnx")
            >>> autotuner.submit(latency)  # Records to profiled_patterns
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        # Handle baseline (first measurement establishes baseline)
        if self.baseline_latency_ms is None:
            self.baseline_latency_ms = latency_ms
            logger.info(f"Baseline latency: {latency_ms:.3f} ms")
            logger.debug("Baseline set for speedup calculations")
            return

        # Determine which schemes collection is active (mutually exclusive)
        if self.current_profile_pattern_schemes is not None:
            schemes_collection = self.current_profile_pattern_schemes
        else:
            raise InvalidSchemeError(
                "No pattern or region selected. Call set_profile_region() first."
            )

        # Check if there are schemes
        if not schemes_collection.schemes:
            raise InvalidSchemeError("No schemes available. Call generate() first.")

        pattern_schemes = schemes_collection

        # Update the scheme's latency
        # Use current_insertion_scheme_index if set (handles both new and re-profiled schemes)
        if (
            hasattr(self, "current_insertion_scheme_index")
            and self.current_insertion_scheme_index is not None
        ):
            scheme_index = self.current_insertion_scheme_index
            if scheme_index >= len(pattern_schemes.schemes):
                raise InvalidSchemeError(f"Invalid scheme index: {scheme_index}")
            scheme = pattern_schemes.schemes[scheme_index]
        else:
            # Fallback: use the last scheme (for backward compatibility)
            scheme = pattern_schemes.schemes[-1]
            scheme_index = len(pattern_schemes.schemes) - 1

        scheme.latency_ms = latency_ms
        scheme.error = not success
        scheme.profile_timestamp = datetime.now(timezone.utc).isoformat()
        # Display index is 1-based
        display_index = scheme_index + 1

        if not success:
            logger.warning(
                f"Scheme #{display_index}: measurement failed (latency={latency_ms:.3f} ms)"
            )
            logger.debug("Marking scheme with error flag")
            return

        # Compute speedup
        speedup = self.baseline_latency_ms / latency_ms if latency_ms > 0 else 0.0

        # Log results
        logger.info(f"Scheme #{display_index}: {latency_ms:.3f} ms ({speedup:.2f}x speedup)")
        logger.debug(f"Compared to baseline: {self.baseline_latency_ms:.3f} ms")

        # Sort schemes by latency (best first)
        # Unmeasured schemes (latency_ms <= 0) go to the end
        old_best = (
            pattern_schemes.schemes[0].latency_ms if pattern_schemes.schemes else float("inf")
        )
        pattern_schemes.schemes.sort(
            key=lambda s: s.latency_ms if s.latency_ms > 0 else float("inf")
        )
        new_best = (
            pattern_schemes.schemes[0].latency_ms if pattern_schemes.schemes else float("inf")
        )

        if new_best < old_best:
            new_speedup = self.baseline_latency_ms / new_best if new_best > 0 else 0.0
            logger.info(f"  ★ New best: {new_best:.3f} ms ({new_speedup:.2f}x speedup)")
            logger.debug(f"Previous best: {old_best:.3f} ms")

        # Update pattern cache with best schemes (only for pattern-based mode)
        if self.current_profile_pattern_schemes is not None and self.pattern_cache is not None:
            self.pattern_cache.add_pattern_schemes(pattern_schemes)
            logger.debug(
                f"Pattern cache updated: {self.pattern_cache.num_patterns} patterns, "
                f"{self.pattern_cache.total_schemes} schemes"
            )

    # =========================================================================
    # State Management
    # =========================================================================

    def save_state(self, output_path: str) -> None:
        """Save complete autotuner state to a YAML file for later reuse.

        Serializes all optimization results including:
        - Baseline latency measurement
        - All profiled patterns with their signatures
        - All generated schemes with insertion points and latencies
        - Configuration parameters
        - Current profiling state

        Also saves pattern cache to a separate file with the suffix "_pattern_cache.yaml"
        containing only the best schemes per pattern (if any patterns were profiled).

        The saved state can be loaded with load_state() to:
        - Resume an interrupted optimization session
        - Reuse results on a similar model
        - Analyze optimization history

        **Note:** The state file contains pattern signatures and performance data,
        but not the actual ONNX model or graph structure.

        Args:
            output_path: File path where the YAML state file will be written.
                        Pattern cache will be saved to <base>_pattern_cache.yaml

        Example:
            >>> # Save after profiling some regions
            >>> autotuner.save_state("checkpoint.yaml")
            >>> # Creates: checkpoint.yaml and checkpoint_pattern_cache.yaml
            >>> # Save final results
            >>> autotuner.save_state("final_state.yaml")
            >>> # Creates: final_state.yaml and final_state_pattern_cache.yaml
        """
        # Save current_profile_pattern_schemes as pattern signature
        current_pattern_sig = None
        if self.current_profile_pattern_schemes is not None:
            current_pattern_sig = self.current_profile_pattern_schemes.pattern_signature

        state = {
            "baseline_latency_ms": self.baseline_latency_ms,
            "current_profile_pattern_schemes_signature": current_pattern_sig,
            "config": {
                "default_q_scale": self.config.default_q_scale,
                "default_q_zero_point": self.config.default_q_zero_point,
                "default_quant_type": self.config.default_quant_type,
                "verbose": self.config.verbose,
            },
            "patterns": [pattern_schemes.to_dict() for pattern_schemes in self.profiled_patterns],
        }

        with open(output_path, "w") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)

        num_patterns = len(self.profiled_patterns)
        total_schemes = sum(len(p.schemes) for p in self.profiled_patterns)
        logger.info(
            f"Saved state → {output_path} ({num_patterns} patterns, {total_schemes} schemes)"
        )
        logger.debug(f"State: baseline={self.baseline_latency_ms:.3f} ms")

        # Save pattern cache to separate file if it has patterns
        if self.pattern_cache is not None and self.pattern_cache.num_patterns > 0:
            # Generate pattern cache path: <base>_pattern_cache.yaml
            base_path, ext = os.path.splitext(output_path)
            cache_path = f"{base_path}_pattern_cache{ext}"

            self.pattern_cache.save(cache_path)
            logger.info(f"Saved pattern cache → {cache_path}")
            logger.debug(
                f"Cache: {self.pattern_cache.num_patterns} patterns, "
                f"{self.pattern_cache.total_schemes} schemes"
            )

    def load_state(self, input_path: str) -> None:
        """Load autotuner state from a previously saved YAML file.

        Restores optimization results from a previous session:
        1. Matches saved patterns to current model's patterns by signature
        2. Loads all schemes with their insertion points and latencies (including unmeasured ones)
        3. Restores baseline latency and configuration

        **Requirements:**
        - Autotuner must be initialized first (model loaded, regions built)
        - Saved patterns must match current model's structure
        - Pattern matching is done by signature, not by index

        **Use Cases:**
        - Resume interrupted optimization
        - Apply previous results to similar model
        - Start from checkpoint instead of scratch

        **Compatibility:**
        - Skips patterns from saved state that don't match current model
        - Warns about mismatched pattern sizes
        - Backward compatible with older state file formats

        Args:
            input_path: File path to the YAML state file to load

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
            FileNotFoundError: If the input_path doesn't exist

        Example:
            >>> # Load and resume from checkpoint
            >>> autotuner = QDQAutotunerBase(model)
            >>> autotuner.initialize()
            >>> autotuner.load_state("checkpoint.yaml")
            >>> # Continue profiling where you left off
            >>> # Reuse results on similar model
            >>> autotuner2 = QDQAutotunerBase(similar_model)
            >>> autotuner2.initialize()
            >>> autotuner2.load_state("final_state.yaml")
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        with open(input_path) as f:
            state = yaml.safe_load(f)

        # Load baseline latency
        if state.get("baseline_latency_ms") is not None:
            self.baseline_latency_ms = state["baseline_latency_ms"]
            logger.debug(f"Baseline latency: {self.baseline_latency_ms:.3f} ms")

        # Load config (optional, merge with existing)
        if "config" in state:
            config_data = state["config"]
            if "default_q_scale" in config_data:
                self.config.default_q_scale = config_data["default_q_scale"]
            if "default_q_zero_point" in config_data:
                self.config.default_q_zero_point = config_data["default_q_zero_point"]
            if "default_quant_type" in config_data:
                self.config.default_quant_type = config_data["default_quant_type"]
            if "verbose" in config_data:
                self.config.verbose = config_data["verbose"]
            logger.debug(f"Config merged: quant_type={self.config.default_quant_type}")

        # Load profiled patterns
        if "patterns" in state:
            num_loaded_patterns = 0
            num_loaded_schemes = 0

            for pattern_data in state["patterns"]:
                try:
                    pattern_schemes = PatternSchemes.from_dict(pattern_data)

                    if pattern_schemes.schemes:  # Only add if it has schemes
                        self.profiled_patterns.append(pattern_schemes)
                        num_loaded_patterns += 1
                        num_loaded_schemes += len(pattern_schemes.schemes)
                    else:
                        logger.debug(
                            f"Skipped empty pattern {pattern_schemes.pattern_signature[:16]}..."
                        )

                except Exception as e:  # noqa: PERF203
                    logger.warning(f"Failed to load pattern: {e}")
                    continue

            logger.info(
                f"Loaded state from {input_path} ({num_loaded_patterns} patterns, "
                f"{num_loaded_schemes} schemes)"
            )

        # Try to load pattern cache if it exists
        base_path, ext = os.path.splitext(input_path)
        cache_path = f"{base_path}_pattern_cache{ext}"

        if os.path.exists(cache_path):
            try:
                loaded_cache = PatternCache.load(cache_path)
                if self.pattern_cache is None:
                    self.pattern_cache = loaded_cache
                else:
                    # Merge with existing pattern cache
                    for pattern_schemes in loaded_cache.pattern_schemes:
                        self.pattern_cache.add_pattern_schemes(pattern_schemes)
                logger.info(
                    f"Loaded pattern cache from {cache_path} ({loaded_cache.num_patterns} patterns, "
                    f"{loaded_cache.total_schemes} schemes)"
                )
            except Exception as e:
                logger.warning(f"Failed to load pattern cache: {e}")
        else:
            logger.debug(f"No pattern cache file at {cache_path}")

    def import_insertion_points(self, quantized_tensors: set[str] | list[str]) -> None:
        """Import Q/DQ insertion points from a list of quantized tensors and update pattern cache.

        Analyzes the current model's regions against the provided quantized tensors
        to extract Q/DQ insertion patterns. For each region, creates a pattern cache
        entry that captures which insertion points correspond to the quantized tensors.
        These cached patterns can then be used as seeds for future autotuning sessions.

        **Use Cases:**
        - Import quantization strategy from an existing quantized model
        - Seed pattern cache with known-good configurations
        - Transfer quantization patterns across similar models
        - Bootstrap autotuning with expert knowledge

        **Workflow:**
        1. Convert input to set for efficient lookup
        2. Iterate through all discovered regions (both LEAF and COMPOSITE)
        3. For each region, call pattern_cache.add_pattern_from_region()
        4. Pattern cache automatically handles deduplication and merging

        **Requirements:**
        - Autotuner must be initialized first (regions must be discovered)
        - Quantized tensors should correspond to actual tensor names in the graph

        Args:
            quantized_tensors: Set or list of tensor names that are quantized
                              (i.e., tensors that have Q/DQ nodes applied to them)

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called

        Example:
            >>> # Import from an existing quantized model
            >>> import onnx
            >>> # Load quantized model and extract quantized tensor names
            >>> # (e.g., by finding first inputs of QuantizeLinear nodes)
            >>> quantized_model = onnx.load("quantized_model.onnx")
            >>> quantized_tensors = set()
            >>> for node in quantized_model.graph.node:
            ...     if node.op_type == "QuantizeLinear":
            ...         quantized_tensors.add(node.input[0])
            >>> # Initialize autotuner on new model and import patterns
            >>> autotuner = QDQAutotuner(new_model)
            >>> autotuner.initialize()
            >>> autotuner.import_insertion_points(quantized_tensors)
            >>> # Pattern cache now contains learned insertion patterns
            >>> print(f"Imported {autotuner.pattern_cache.num_patterns} patterns")
            >>> autotuner.pattern_cache.save("imported_patterns.yaml")
        """
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        # Convert to set for efficient lookup
        if isinstance(quantized_tensors, list):
            quantized_tensors = set(quantized_tensors)

        logger.info(f"Importing insertion points from {len(quantized_tensors)} quantized tensors")
        logger.debug(f"Processing {len(self.regions)} regions")

        if self.pattern_cache is None:
            logger.warning("Pattern cache not initialized, skipping import")
            return

        # Track statistics
        patterns_before = self.pattern_cache.num_patterns
        schemes_before = self.pattern_cache.total_schemes

        # Process all regions (both LEAF and COMPOSITE)
        for region in self.regions:
            self.pattern_cache.add_pattern_from_region(region, self.graph, quantized_tensors)

        # Log results
        patterns_added = self.pattern_cache.num_patterns - patterns_before
        schemes_added = self.pattern_cache.total_schemes - schemes_before

        logger.info(
            f"Import complete: {patterns_added} patterns, {schemes_added} schemes added to cache"
        )
        logger.debug(
            f"Total cache: {self.pattern_cache.num_patterns} patterns, "
            f"{self.pattern_cache.total_schemes} schemes"
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_convergence_metrics(
        self, schemes: list[InsertionScheme], best_scheme: InsertionScheme | None
    ) -> tuple[int | None, float | None]:
        """Compute convergence metrics for a collection of schemes.

        Analyzes when the best scheme was discovered during the profiling process
        by sorting schemes by their profile timestamps and finding the position
        of the best scheme.

        Args:
            schemes: List of insertion schemes with profile timestamps
            best_scheme: The best performing scheme (lowest latency)

        Returns:
            Tuple of (samples_before_best, time_to_best) where:
            - samples_before_best: Number of samples tested before finding best (0-based index)
            - time_to_best: Time in seconds from first sample to best sample
            Both values are None if metrics cannot be computed (e.g., missing timestamps)
        """
        samples_before_best = None
        time_to_best = None

        if not best_scheme or not best_scheme.profile_timestamp:
            return samples_before_best, time_to_best

        # Get schemes with timestamps, sorted by timestamp
        schemes_with_time = [s for s in schemes if s.profile_timestamp is not None]

        if not schemes_with_time:
            return samples_before_best, time_to_best

        from datetime import datetime

        schemes_with_time.sort(key=lambda s: s.profile_timestamp or "")

        # Find position of best scheme in time-sorted list
        try:
            best_position = next(
                i for i, s in enumerate(schemes_with_time) if s.hash == best_scheme.hash
            )
            samples_before_best = best_position

            # Compute time difference
            first_ts = schemes_with_time[0].profile_timestamp
            best_ts = best_scheme.profile_timestamp
            assert first_ts is not None and best_ts is not None
            first_timestamp = datetime.fromisoformat(first_ts)
            best_timestamp = datetime.fromisoformat(best_ts)
            time_to_best = (best_timestamp - first_timestamp).total_seconds()
        except (StopIteration, ValueError):
            pass

        return samples_before_best, time_to_best

    def _is_region_profiled(self, region: Region) -> bool:
        """Check if a region's pattern has already been fully profiled."""

        def match_pattern(pattern: PatternSchemes, region: Region) -> bool:
            """Check if a pattern matches a region."""
            if pattern.pattern is None or not pattern.pattern.matches(region, self.graph):
                return False
            return not any(not scheme.is_profiled for scheme in pattern.schemes)

        return any(match_pattern(pattern, region) for pattern in self.profiled_patterns)

    # --- Scheme Generation ---

    def _mutate_insertion_points(
        self, base_points, all_points, point_type: str, max_mutations: int
    ) -> list:
        """Mutate a set of insertion points by adding, removing, or both.

        Args:
            base_points: Set of tuples representing current insertion points
            all_points: List of all possible insertion point objects
            point_type: Type of insertion points (for logging)
            max_mutations: Maximum number of mutations per operation

        Returns:
            List of mutated insertion point objects
        """
        current_points = set(base_points)
        initial_count = len(current_points)

        # Randomly choose mutation type
        mutation_type = random.choice(["add", "remove", "both"])

        # Add points
        if mutation_type in ["add", "both"] and len(current_points) < len(all_points):
            # Get point keys based on type
            all_keys: set[tuple] = set()
            if point_type == "node input points":
                all_keys = {(p.node_index, p.input_index) for p in all_points}
            elif point_type == "region composite points":
                all_keys = {(p.region_index, p.input_index) for p in all_points}
            elif point_type == "region output points":
                all_keys = {(p.region_index, p.node_index, p.output_index) for p in all_points}

            available_keys = all_keys - current_points
            if available_keys:
                max_add = min(max_mutations, len(available_keys))
                num_to_add = random.randint(1, max_add)
                to_add = random.sample(list(available_keys), num_to_add)
                current_points.update(to_add)

        # Remove points
        if mutation_type in ["remove", "both"] and current_points:
            max_remove = min(max_mutations, len(current_points))
            num_to_remove = random.randint(1, max_remove) if len(current_points) > 1 else 1
            num_to_remove = min(num_to_remove, len(current_points))
            to_remove = random.sample(list(current_points), num_to_remove)
            for p in to_remove:
                current_points.discard(p)

        logger.debug(
            f"Mutated {point_type}: {initial_count} → {len(current_points)} ({mutation_type})"
        )

        # Convert back to insertion point objects based on type
        if point_type == "node input points":
            return [p for p in all_points if (p.node_index, p.input_index) in current_points]
        elif point_type == "region composite points":
            return [p for p in all_points if (p.region_index, p.input_index) in current_points]
        elif point_type == "region output points":
            return [
                p
                for p in all_points
                if (p.region_index, p.node_index, p.output_index) in current_points
            ]
        else:
            return []

    def _generate_next_insertion_sample(self) -> InsertionScheme:
        """Generate a new insertion scheme by mutating top performers.

        This is the core scheme generation algorithm:
        1. Identifies top schemes by latency
        2. Randomly selects one as the base
        3. Mutates node input insertion points (add, remove, or both)
        4. Mutates region composite insertion points (child boundaries)
        5. Mutates region output insertion points
        6. Returns new unique scheme

        **Mutation Strategy:**
        - Node input points: Add/remove 1-3 insertion points
        - Region composite points: Add/remove 1-3 boundary points
        - Region output points: Add/remove 1-3 output points
        - Mutation type chosen randomly: 'add', 'remove', or 'both'

        **Baseline Case:**
        If no schemes exist yet, returns an empty baseline scheme.

        Returns:
            New InsertionScheme with mutated insertion points.
            Returns empty scheme if no region is set or no candidates exist.
        """
        # Validate current profile region is set
        if self.current_profile_region is None:
            return InsertionScheme()

        # Determine which schemes collection is active (mutually exclusive)
        if self.current_profile_pattern_schemes is not None:
            schemes_collection = self.current_profile_pattern_schemes
        else:
            return InsertionScheme()

        region = self.current_profile_region
        pattern_schemes = schemes_collection

        # Get the pattern
        pattern = None
        if isinstance(schemes_collection, PatternSchemes):
            pattern = schemes_collection.pattern
        if pattern is None:
            return InsertionScheme()
        # Get all possible insertion points for this region
        full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)

        logger.debug(
            f"Available insertion points: {len(full_insertion_scheme.node_inputs)} node input, "
            f"{len(full_insertion_scheme.child_region_inputs)} region composite, "
            f"{len(full_insertion_scheme.region_outputs)} region output"
        )

        # Get top-performing schemes
        top_percent = getattr(self.config, "top_percent_to_mutate", 0.1)
        minimum_schemes = getattr(self.config, "minimum_schemes_to_mutate", 1)

        # Filter measured schemes
        measured_schemes = [s for s in pattern_schemes.schemes if s.latency_ms > 0 and not s.error]
        measured_schemes.sort(key=lambda s: s.latency_ms)

        num_top_schemes = max(
            int(len(measured_schemes) * top_percent), min(minimum_schemes, len(measured_schemes))
        )
        top_schemes = measured_schemes[:num_top_schemes]

        # Return empty baseline if no schemes exist
        if len(top_schemes) == 0:
            logger.debug("No measured schemes yet, generating baseline (empty) scheme")
            return InsertionScheme()

        # Select base scheme from top performers
        base_scheme = random.choice(top_schemes)
        total_base_points = (
            len(base_scheme.node_inputs)
            + len(base_scheme.child_region_inputs)
            + len(base_scheme.region_outputs)
        )
        logger.debug(
            f"Mutating from top {len(top_schemes)} schemes: "
            f"selected base with {total_base_points} points (latency={base_scheme.latency_ms:.3f} ms)"
        )

        # Create new scheme
        scheme = InsertionScheme()

        max_mutations = getattr(self.config, "maximum_mutations", 3)

        # Mutate node input insertion points
        base_node_points = {(p.node_index, p.input_index) for p in base_scheme.node_inputs}
        scheme.node_inputs = self._mutate_insertion_points(
            base_node_points, full_insertion_scheme.node_inputs, "node input points", max_mutations
        )

        # Mutate region composite insertion points
        base_region_composite_points = {
            (p.region_index, p.input_index) for p in base_scheme.child_region_inputs
        }
        scheme.child_region_inputs = self._mutate_insertion_points(
            base_region_composite_points,
            full_insertion_scheme.child_region_inputs,
            "region composite points",
            max_mutations,
        )

        # Mutate region output insertion points
        base_region_output_points = {
            (p.region_index, p.node_index, p.output_index) for p in base_scheme.region_outputs
        }
        scheme.region_outputs = self._mutate_insertion_points(
            base_region_output_points,
            full_insertion_scheme.region_outputs,
            "region output points",
            max_mutations,
        )

        return scheme

    # --- Graph Manipulation ---

    def _copy_graph(self) -> gs.Graph:
        """Create an independent copy of the computation graph.

        Exports the original model to ONNX and imports it back to create
        a fresh graph instance. This ensures modifications don't affect
        the original graph.

        Returns:
            New gs.Graph instance with identical structure to the original
        """
        new_graph = gs.import_onnx(self.onnx_model)
        new_graph.toposort()
        return new_graph

    def _get_quant_dtype(self, quant_type: str) -> np.dtype:
        """Get numpy dtype for quantization type.

        Args:
            quant_type: Quantization type string ("int8", "fp8")

        Returns:
            Numpy dtype for the quantization type

        Note:
            FP8 support requires numpy >= 2.0. If not available, falls back to a
            compatible representation.
        """
        # Handle FP8 with version check
        if quant_type == "fp8":
            try:
                # Try to get FP8 dtype (numpy >= 2.0)
                return np.dtype(np.float8_e4m3fn)
            except (AttributeError, TypeError):
                logger.warning(
                    "FP8 dtype not available (requires numpy >= 2.0), "
                    "using uint8 as placeholder. Note: This may not produce "
                    "correct results without proper FP8 support."
                )
                return np.uint8

        dtype_map = {
            "int8": np.int8,
            "uint8": np.uint8,
        }

        if quant_type not in dtype_map:
            logger.warning(f"Unknown quantization type '{quant_type}', defaulting to int8")
            return np.int8

        return dtype_map[quant_type]

    def _get_dq_output_dtype(self, dtype_str: str) -> np.dtype:
        """Convert DQ dtype string to numpy dtype.

        Args:
            dtype_str: Dtype string ("float16", "float32", "bfloat16")

        Returns:
            Numpy dtype for the DQ output type
        """
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
        }

        # Handle bfloat16 if available
        if hasattr(np, "bfloat16"):
            dtype_map["bfloat16"] = np.bfloat16

        if dtype_str not in dtype_map:
            logger.warning(f"Unknown DQ dtype '{dtype_str}', defaulting to float32")
            return np.float32

        return dtype_map[dtype_str]

    def _build_tensor_map(self, graph: gs.Graph) -> dict[str, gs.Tensor]:
        """Build mapping from tensor names to tensor objects.

        Args:
            graph: Graph to extract tensors from

        Returns:
            Dictionary mapping tensor names to tensor objects (Variables or Constants)
        """
        tensor_map = {}

        # Map node outputs (Variables)
        for node in graph.nodes:
            for output in node.outputs:
                if hasattr(output, "name") and output.name:
                    tensor_map[output.name] = output

        # Map graph inputs (Variables)
        for input_tensor in graph.inputs:
            if hasattr(input_tensor, "name") and input_tensor.name:
                tensor_map[input_tensor.name] = input_tensor

        # Map initializers/constants
        for node in graph.nodes:
            for input_tensor in node.inputs:
                if (
                    isinstance(input_tensor, gs.Constant)
                    and hasattr(input_tensor, "name")
                    and input_tensor.name
                ):
                    tensor_map[input_tensor.name] = input_tensor

        return tensor_map

    def _get_tensor_metadata(
        self, tensor: gs.Tensor, is_constant: bool
    ) -> tuple[tuple | None, np.dtype]:
        """Extract shape and dtype metadata from a tensor.

        Args:
            tensor: Tensor to extract metadata from
            is_constant: Whether the tensor is a Constant

        Returns:
            Tuple of (shape, dtype) where shape may be None if unknown
        """
        default_dtype = self._get_dq_output_dtype(self.config.default_dq_dtype)
        if is_constant and hasattr(tensor, "values") and tensor.values is not None:
            return tensor.values.shape, tensor.values.dtype
        elif hasattr(tensor, "shape"):
            dtype = (
                tensor.dtype
                if hasattr(tensor, "dtype") and tensor.dtype is not None
                else default_dtype
            )
            return tensor.shape, dtype
        else:
            return None, default_dtype

    def _fix_zero_point_initializers(self, model: onnx.ModelProto) -> None:
        """Fix INT8 zero_point initializers to use int32_data instead of raw_data.

        For INT8 tensors, ONNX stores the data in int32_data field with 4-byte alignment,
        not in raw_data. This is needed because int8_to_fp8 expects zero_point.int32_data
        to be populated.

        Args:
            model: ONNX model to fix
        """
        fixed_count = 0

        for initializer in model.graph.initializer:
            # Check if this is a zero_point tensor (q_zp_ or dq_zp_)
            if (
                "_zp_" in initializer.name
                and initializer.data_type == onnx.TensorProto.INT8
                and len(initializer.raw_data) > 0
                and len(initializer.int32_data) == 0
            ):
                # Convert raw_data to int32_data (4-byte aligned)
                np_array = onnx.numpy_helper.to_array(initializer)
                # Store INT8 values in int32_data field (4-byte aligned)
                int32_values = np_array.astype(np.int32).flatten().tolist()

                new_tensor = onnx.helper.make_tensor(
                    initializer.name,
                    onnx.TensorProto.INT8,
                    list(initializer.dims),
                    int32_values,  # This populates int32_data instead of raw_data
                )
                initializer.CopyFrom(new_tensor)
                fixed_count += 1

        if fixed_count > 0:
            logger.debug(f"Fixed {fixed_count} zero_point initializers (int32_data format)")

    def _create_qdq_nodes(
        self,
        tensor_name: str,
        qdq_input: gs.Tensor,
        output_shape: tuple | None,
        output_dtype: np.dtype,
        quant_dtype: np.dtype,
        quant_type: str,
        q_scale: float,
    ) -> tuple[gs.Node, gs.Node]:
        """Create QuantizeLinear and DequantizeLinear node pair.

        Args:
            tensor_name: Name of the tensor being quantized
            qdq_input: Input tensor to the Q node
            output_shape: Shape for Q/DQ outputs (may be None)
            output_dtype: Dtype for DQ output (also used for scale dtype)
            quant_dtype: Dtype for quantized values
            quant_type: Quantization type string
            q_scale: Quantization scale

        Returns:
            Tuple of (q_node, dq_node)
        """
        # Create unique names for Q/DQ nodes
        q_name = f"QDQ_Q_{tensor_name}".replace("/", "_").replace(":", "_")
        dq_name = f"QDQ_DQ_{tensor_name}".replace("/", "_").replace(":", "_")

        # Determine scale dtype from output_dtype (fp16/tf32/fp32)
        # Scale should match the precision of the original I/O tensor
        # Note: output_dtype can be either a numpy type class (np.float16) or dtype instance (dtype('float16'))
        # Use np.dtype().name for consistent comparison
        dtype_name = np.dtype(output_dtype).name
        if dtype_name == "float16":
            scale_dtype = np.float16
        elif dtype_name == "float32":
            scale_dtype = np.float32
        elif dtype_name == "bfloat16" and hasattr(np, "bfloat16"):
            scale_dtype = np.bfloat16
        else:
            scale_dtype = np.float32

        logger.debug(
            f"Creating Q/DQ pair for '{tensor_name}' (scale_dtype={np.dtype(scale_dtype).name})"
        )

        # Build QuantizeLinear inputs: [input, scale, zero_point]
        # Scale and zero_point must be proper ONNX initializers
        q_scale_values = np.array([q_scale], dtype=scale_dtype)
        q_zp_values = np.array([0], dtype=np.int8)

        q_inputs = [
            qdq_input,
            gs.Constant(f"q_scale_{tensor_name}", values=q_scale_values),
            gs.Constant(f"q_zp_{tensor_name}", values=q_zp_values),
        ]

        q_node = gs.Node(
            op="QuantizeLinear",
            name=q_name,
            inputs=q_inputs,
            outputs=[
                gs.Variable(f"{tensor_name}_quantized", dtype=quant_dtype, shape=output_shape)
            ],
        )

        # Build DequantizeLinear inputs: [quantized_input, scale, zero_point]
        # Scale and zero_point must be proper ONNX initializers
        dq_scale_values = np.array([q_scale], dtype=scale_dtype)
        dq_zp_values = np.array([0], dtype=np.int8)

        dq_inputs = [
            q_node.outputs[0],
            gs.Constant(f"dq_scale_{tensor_name}", values=dq_scale_values),
            gs.Constant(f"dq_zp_{tensor_name}", values=dq_zp_values),
        ]

        dq_node = gs.Node(
            op="DequantizeLinear",
            name=dq_name,
            inputs=dq_inputs,
            outputs=[
                gs.Variable(f"{tensor_name}_dequantized", dtype=output_dtype, shape=output_shape)
            ],
        )

        return q_node, dq_node

    def _insert_qdq_at_tensors(
        self, graph: gs.Graph, resolved_insertion_points: set[ResolvedInsertionPoint]
    ) -> None:
        """Insert Q/DQ (Quantize/Dequantize) node pairs at specified locations.

        This is the main entry point for Q/DQ insertion. It:
        1. Builds tensor map and tensor-to-users map for efficient lookup
        2. Processes each resolved insertion point to insert Q/DQ nodes
        3. Handles two insertion modes based on node_index

        **Insertion Modes:**

        1. **Node-level insertion** (node_index is set, input_index is set):
           - Inserts Q/DQ for a specific node's specific input connection
           - Only rewires that one node-tensor connection
           - Multiple Q/DQ pairs can be created for the same tensor at different nodes
           - Naming: `{tensor_name}_n{node_index}_i{input_index}`
           - Use case: Fine-grained control over quantization boundaries

        2. **Tensor-level insertion** (node_index=None, input_index=None):
           - Inserts one Q/DQ pair for the entire tensor
           - Rewires ALL users of the tensor to use the same DQ output
           - Only one Q/DQ pair created regardless of number of users
           - Naming: `{tensor_name}_qdq`
           - Use case: Quantize a tensor once when it feeds multiple nodes

        **Validation:**
        - When node_index is set: input_index must also be set
        - When node_index is None: input_index must be None
        - All validations use assertions (failures indicate programming errors)

        **Handling for Constants:**
        - Q/DQ nodes can be inserted directly on Constant tensors (weights, biases)
        - No conversion needed since QuantizeLinear accepts Constant inputs

        **Quantization Parameters:**
        - Uses config.default_quant_type for quantization type ("int8", "fp8")
        - Uses config.default_q_scale for quantization scale
        - Zero-point is always set to 0 (int8) for all quantization types
        - Creates separate constants for each Q/DQ pair

        Args:
            graph: Graph to modify in-place
            resolved_insertion_points: Set of ResolvedInsertionPoint objects specifying where to insert Q/DQ
        """
        # Extract quantization parameters
        q_scale = self.config.default_q_scale
        quant_type = self.config.default_quant_type
        quant_dtype = self._get_quant_dtype(quant_type)

        logger.debug(f"Q/DQ parameters: type={quant_type}, scale={q_scale}, zero_point=0")

        resolved_insertion_points = merge_resolved_insertion_points(
            graph, resolved_insertion_points
        )

        # Build tensor name → tensor object mapping
        tensor_map = self._build_tensor_map(graph)
        tensor_users_map = get_tensor_consumer_node_indices(graph)
        logger.debug(
            f"Built tensor maps: {len(tensor_map)} tensors, {len(tensor_users_map)} with users"
        )

        # Process each resolved insertion point
        for insertion_point in resolved_insertion_points:
            tensor_name = insertion_point.tensor_name
            node_index = insertion_point.node_index
            input_index = insertion_point.input_index

            original_tensor = tensor_map[tensor_name]
            # Validate input/output index
            if node_index is not None:
                assert node_index < len(graph.nodes), "Node index out of range"
                target_node = graph.nodes[node_index]
                assert input_index is not None, "Input index must be set when node index is set"
                assert input_index < len(target_node.inputs), (
                    f"Input index out of range for node {target_node.name}"
                )
                original_tensor = target_node.inputs[input_index]
                assert tensor_name == original_tensor.name, (
                    f"Tensor name mismatch for node {target_node.name} input {input_index}"
                )
            else:
                assert tensor_name in tensor_map, f"Tensor {tensor_name} not found in tensor map"
                assert input_index is None, "Input index must be None when node index is None"

            # Get node and tensor
            is_constant = isinstance(original_tensor, gs.Constant)

            # Extract tensor metadata (shape, dtype)
            output_shape, output_dtype = self._get_tensor_metadata(original_tensor, is_constant)

            # Create unique Q/DQ node pair for this specific insertion point
            unique_suffix = "qdq"
            if node_index is not None:
                unique_suffix = f"n{node_index}_i{input_index}"
            unique_tensor_name = f"{tensor_name}_{unique_suffix}"

            # Create Q/DQ node pair
            q_node, dq_node = self._create_qdq_nodes(
                unique_tensor_name,
                original_tensor,
                output_shape,
                output_dtype,
                quant_dtype,
                quant_type,
                q_scale,
            )

            # Add nodes to graph
            graph.nodes.extend([q_node, dq_node])

            # Rewire only the specific node-tensor connection
            if node_index is not None:
                # Insert QDQ between the producer and this specific input
                target_node.inputs[input_index] = dq_node.outputs[0]
                logger.debug(
                    f"  Q/DQ inserted: tensor '{tensor_name}' → node #{node_index} "
                    f"({target_node.name}) input #{input_index}"
                )
            else:
                users = tensor_users_map[tensor_name]
                for user_index in users:
                    user_node = graph.nodes[user_index]
                    for i, input_tensor in enumerate(user_node.inputs):
                        if hasattr(input_tensor, "name") and input_tensor.name == tensor_name:
                            user_node.inputs[i] = dq_node.outputs[0]
                            break
                logger.debug(f"  Q/DQ inserted: tensor '{tensor_name}' → {len(users)} users")

        # Cleanup and toposort
        logger.debug("Running graph cleanup and topological sort")
        try:
            graph.cleanup().toposort()
            logger.debug("Graph cleanup completed")
        except Exception as e:
            logger.warning(f"Graph cleanup failed: {e}")
            logger.debug("Continuing anyway")


class QDQAutotuner(QDQAutotunerBase):
    """Ready-to-use Q/DQ autotuner with automatic region discovery.

    This is the main class users should instantiate for Q/DQ optimization.
    It extends QDQAutotunerBase by automatically searching for optimization
    regions around compute-intensive operations during initialization.

    **Automatic Region Discovery:**
    - Uses CombinedRegionSearch to identify regions automatically
    - Focuses on Conv, MatMul, Gemm, and other compute-heavy operations
    - Creates hierarchical region structure (COMPOSITE with LEAF children)
    - Flattens hierarchy and prioritizes LEAF regions for profiling

    **Region Selection Strategy:**
    The discovered regions are organized to optimize profiling efficiency:
    1. LEAF regions: Contain actual nodes, profiled first (most specific)
    2. Non-COMPOSITE regions: Profiled second (intermediate level)
    3. COMPOSITE regions: Skipped (only containers, no direct nodes)

    This ensures we test the most granular patterns first, which provides
    better optimization opportunities and more reusable pattern cache entries.

    **Usage Pattern:**
    ```python
    # Load model
    model = onnx.load("model.onnx")

    # Create autotuner (regions discovered automatically)
    autotuner = QDQAutotuner(model)

    # Initialize with configuration
    config = Config(default_quant_type="fp8")
    autotuner.initialize(config)

    # Measure baseline (optional but recommended)
    baseline_bytes = autotuner.export_onnx("baseline.onnx", insert_qdq=False)
    baseline_latency = benchmark("baseline.onnx")
    autotuner.submit(baseline_latency)

    # Profile regions
    for region in autotuner.regions[:10]:  # Top 10 regions
        autotuner.set_profile_region(region)

        # Generate and test multiple schemes
        for i in range(5):
            scheme_idx = autotuner.generate()
            if scheme_idx < 0:
                break  # No more unique schemes

            # Export and benchmark
            test_bytes = autotuner.export_onnx(f"test_{i}.onnx")
            latency = benchmark(f"test_{i}.onnx")
            autotuner.submit(latency)

    # Export final optimized model
    autotuner.export_onnx("optimized.onnx")

    # Save results for reuse
    autotuner.save_state("results.yaml")
    ```

    **Key Differences from Base Class:**
    - Automatic region discovery (no manual region specification needed)
    - Hierarchical region structure flattened for efficient profiling
    - LEAF regions prioritized (contain actual nodes to optimize)
    - Ready to use out of the box (no custom region strategy needed)

    **Region Discovery Details:**
    Uses a two-phase search strategy:
    1. Bottom-up partitioning: Groups nodes by divergence/convergence patterns
    2. Top-down refinement: Creates hierarchical structure within regions

    See CombinedRegionSearch documentation for algorithm details.

    Attributes:
        regions: List of discovered regions, ordered by priority (LEAF first)
        graph: ONNX computation graph
        config: Configuration parameters
        profiled_patterns: Results from profiled regions

    Example:
        >>> # Simple usage
        >>> autotuner = QDQAutotuner(model)
        >>> autotuner.initialize()
        >>> print(f"Found {len(autotuner.regions)} regions to optimize")
        >>> # With custom config
        >>> config = Config(default_quant_type="fp8")
        >>> autotuner = QDQAutotuner(model)
        >>> autotuner.initialize(config)
    """

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuner and discover optimization regions automatically.

        Extends base class initialization by automatically searching for regions
        after configuration is set up. Regions are discovered using pattern-based
        search around compute-intensive operations.

        **Automatic Steps:**
        1. Calls base class initialize (sets up config, pattern cache)
        2. Runs region search (discovers optimization targets)
        3. Flattens region hierarchy and prioritizes LEAF regions
        4. Reassigns region IDs for clean indexing

        After this method completes, self.regions contains all discovered regions
        ready for profiling via set_profile_region().

        Args:
            config: Optional configuration for Q/DQ parameters and profiling behavior.
                   If None, uses default Config() settings.
            pattern_cache: Optional pattern cache for warm-starting with known schemes.
                          If None, creates empty cache.

        Raises:
            None (safe to call multiple times - resets state each time)

        Example:
            >>> autotuner = QDQAutotuner(model)
            >>> autotuner.initialize()
            >>> print(f"Ready to profile {len(autotuner.regions)} regions")
            >>> # With custom configuration
            >>> config = Config(default_quant_type="fp8")
            >>> autotuner.initialize(config)
        """
        # Initialize base class (config, pattern cache, reset state)
        super().initialize(config, pattern_cache)

        # Discover optimization regions automatically
        self._search_regions()

    def _visit_region_recursively(self, region: Region) -> list[Region]:
        """Recursively traverse region hierarchy and collect all regions.

        Performs depth-first traversal of the region tree starting from a given
        region. Collects the root region and all descendant regions (children,
        grandchildren, etc.) into a flat list.

        **Traversal Order:**
        - Pre-order: Parent added before children
        - Depth-first: Fully explores each branch before moving to next

        **Use Case:**
        Used to flatten hierarchical region structure (COMPOSITE → LEAF) into
        a single list for sequential profiling. This ensures all regions at all
        levels are available for optimization.

        Args:
            region: Root region to start traversal from

        Returns:
            List of all regions in the subtree (including root), in pre-order DFS.

        Example:
            >>> # Given hierarchy: COMPOSITE { LEAF{A, B}, COMPOSITE { LEAF{C} } }
            >>> regions = _visit_region_recursively(composite_root)
            >>> # Returns: [COMPOSITE_root, LEAF_AB, COMPOSITE_child, LEAF_C]
        """
        # Start with the current region
        regions = [region]

        # Recursively add all children and their descendants
        for child in region.get_children():
            regions.extend(self._visit_region_recursively(child))

        return regions

    def _reassign_region_ids(self, regions: list[Region]) -> None:
        """Reassign sequential IDs to regions in breadth-first order.

        Traverses the region hierarchy (including children) and assigns new
        sequential IDs starting from 0. This ensures clean, predictable region
        numbering after region discovery and manipulation.

        **Traversal Strategy:**
        - Breadth-first: Siblings get consecutive IDs before their children
        - Sequential: IDs are 0, 1, 2, ... with no gaps

        **Why This Matters:**
        - Clean logging: "Region 0", "Region 1", etc.
        - Pattern cache: Region IDs appear in insertion point references
        - Debugging: Predictable numbering aids in understanding results

        **Modifies:**
        Updates region.id for each region in-place (including all descendants).

        Args:
            regions: List of top-level regions (children will be processed too)

        Side Effects:
            Modifies the .id attribute of all regions and their descendants

        Example:
            >>> # Before: regions with IDs [5, 12, 8, ...]
            >>> _reassign_region_ids(regions)
            >>> # After: regions with IDs [0, 1, 2, ...]
        """
        region_id = 0

        # Use BFS to assign IDs level-by-level
        queue = deque(regions)

        while queue:
            region = queue.popleft()

            # Assign next sequential ID
            region.id = region_id
            region_id += 1

            # Add children to queue for processing
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

        **Region Organization:**
        After discovery, regions are reorganized for optimal profiling:
        ```
        Original: [COMPOSITE_1 { LEAF_A, LEAF_B }, COMPOSITE_2 { LEAF_C }]

        After flattening: [COMPOSITE_1, LEAF_A, LEAF_B, COMPOSITE_2, LEAF_C]

        After prioritization: [LEAF_A, LEAF_B, LEAF_C, COMPOSITE_1, COMPOSITE_2]
        ```

        **Why Prioritize LEAF Regions:**
        - LEAF regions contain actual nodes (direct optimization targets)
        - COMPOSITE regions are just containers (no direct nodes to optimize)
        - Profiling LEAF first gives more specific, reusable patterns
        - Pattern cache entries from LEAF regions apply to many models

        **Region Types:**
        - LEAF: Contains graph nodes, profiled first (highest priority)
        - COMPOSITE: Container for other regions, lower priority
        - ROOT: Special container (typically not profiled directly)

        Side Effects:
            Populates self.regions with discovered and organized regions

        Example:
            >>> # Automatically called during initialize()
            >>> _search_regions()
            >>> # self.regions now contains all discovered regions
            >>> leaf_count = sum(1 for r in self.regions if r.type == RegionType.LEAF)
            >>> print(f"Discovered {leaf_count} LEAF regions for profiling")
        """
        # =====================================================================
        # STEP 1: Run Automatic Region Discovery
        # =====================================================================
        # Use CombinedRegionSearch to find regions around compute-intensive ops
        # This creates a hierarchical structure: COMPOSITE → LEAF
        logger.info("Discovering optimization regions")
        search = CombinedRegionSearch(
            self.graph,
            maximum_sequence_region_size=self.config.maximum_sequence_region_size,
            minimum_topdown_search_size=self.config.minimum_topdown_search_size,
        )
        self.regions = search.search_regions()

        # Reassign IDs to top-level regions for clean indexing
        self._reassign_region_ids(self.regions)
        logger.debug(f"Found {len(self.regions)} top-level regions")

        # =====================================================================
        # STEP 2: Flatten Hierarchical Structure
        # =====================================================================
        # Traverse the region tree and collect all regions at all levels
        # This ensures we can profile both parent and child regions
        all_regions = []
        for region in self.regions:
            all_regions.extend(self._visit_region_recursively(region))

        logger.debug(f"Flattened hierarchy to {len(all_regions)} total regions")

        # =====================================================================
        # STEP 3: Prioritize LEAF Regions
        # =====================================================================
        # Organize regions to profile the most specific patterns first:
        # 1. LEAF regions: Contain actual nodes, most specific patterns
        # 2. Other non-COMPOSITE: Intermediate abstractions
        # 3. COMPOSITE regions excluded: Just containers, no direct nodes

        # Extract LEAF regions (highest priority)
        leaf_regions = [region for region in all_regions if region.type == RegionType.LEAF]
        other_regions = [region for region in all_regions if region.type != RegionType.LEAF]

        # Combine: LEAF first, then others
        # This ensures the most granular optimization targets are profiled first
        all_regions = leaf_regions + other_regions

        # Update self.regions with prioritized list
        self.regions = all_regions

        num_leaf = sum(1 for r in self.regions if r.type == RegionType.LEAF)
        num_composite = sum(1 for r in self.regions if r.type == RegionType.COMPOSITE)
        num_root = sum(1 for r in self.regions if r.type == RegionType.ROOT)

        logger.info(
            f"Discovery complete: {len(self.regions)} regions "
            f"({num_leaf} LEAF, {num_composite} COMPOSITE, {num_root} ROOT)"
        )
        logger.debug("Regions prioritized: LEAF regions first for profiling")
