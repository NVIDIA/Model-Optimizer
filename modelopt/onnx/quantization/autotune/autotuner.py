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

import copy
import os
import random
from collections import deque
from datetime import datetime, timezone

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import yaml

from modelopt.onnx.logging_config import logger
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


class QDQAutotunerBase:
    """Base class for pattern-based Q/DQ node insertion optimization in ONNX models."""

    def __init__(self, model: onnx.ModelProto | gs.Graph):
        """Initialize the autotuner with an ONNX model."""
        if isinstance(model, onnx.ModelProto):
            self.onnx_model = model
        elif isinstance(model, gs.Graph):
            self.onnx_model = gs.export_onnx(model)
        else:
            raise TypeError(f"Expected onnx.ModelProto or gs.Graph, got {type(model)}")

        self.graph = self._copy_graph()
        self.graph.tensor_users_map = get_tensor_consumer_node_indices(self.graph)

        self.regions: list[Region] = []
        self.current_profile_region: Region | None = None

        self.profiled_patterns: list[PatternSchemes] = []
        self.current_profile_pattern_schemes: PatternSchemes | None = None

        self.current_insertion_scheme_index: int | None = None

        self.config = Config()
        self.initialized = False
        self.baseline_latency_ms: float | None = None

        self.pattern_cache: PatternCache | None = None

        logger.debug(f"Initialized autotuner with model type: {type(model).__name__}")

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuning session with configuration and pattern cache."""
        if config is not None:
            self.config = config

        if pattern_cache is None:
            pattern_cache = PatternCache(
                minimum_distance=self.config.pattern_cache_minimum_distance,
                max_entries_per_pattern=self.config.pattern_cache_max_entries_per_pattern,
            )
        self.pattern_cache = pattern_cache

        logger.debug(
            f"Loaded pattern cache with {pattern_cache.num_patterns} patterns and "
            f"{pattern_cache.total_schemes} schemes"
        )

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

        self.initialized = True

    def set_profile_region(self, region: Region | None, commit: bool = True) -> None:
        """Set the target region for profiling and scheme generation."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        if commit:
            if self.current_profile_pattern_schemes is not None:
                num_schemes = len(self.current_profile_pattern_schemes.schemes)
                best_scheme = self.current_profile_pattern_schemes.best_scheme
                best_latency = best_scheme.latency_ms if best_scheme else float("inf")

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

        if region not in self.regions:
            raise ValueError(f"Region {region.id} not found in regions")

        region_pattern = RegionPattern.from_region(region, self.graph)

        if self._is_region_profiled(region):
            logger.info(f"Skipping region {region.id} (pattern already profiled)")
            logger.debug(f"Pattern signature: {region_pattern.signature}")
            return

        pattern_schemes = None
        num_seeded = 0

        if self.pattern_cache is not None:
            cache_schemes = self.pattern_cache.get_pattern_schemes(region_pattern.signature)

            if cache_schemes is not None and len(cache_schemes.schemes) > 0:
                pattern_schemes = PatternSchemes()
                pattern_schemes.pattern = region_pattern

                for cached_scheme in cache_schemes.schemes:
                    scheme_copy = copy.deepcopy(cached_scheme)
                    scheme_copy.latency_ms = float("inf")
                    scheme_copy.error = False
                    pattern_schemes.schemes.append(scheme_copy)
                    num_seeded += 1

                logger.debug(f"Seeded {num_seeded} scheme(s) from pattern cache")
            else:
                logger.debug("No pattern cache entries for this region")

        if pattern_schemes is None:
            pattern_schemes = PatternSchemes()
            pattern_schemes.pattern = region_pattern
            logger.debug("Initialized with empty scheme collection")

        self.current_profile_region = region
        self.current_profile_pattern_schemes = pattern_schemes

        mode_info = f"seeded with {num_seeded} schemes" if num_seeded > 0 else "starting fresh"
        logger.info(
            f"Profiling region {region.id} [pattern mode, level {region.get_level()}, "
            f"size {region.get_size()}, {mode_info}]"
        )
        logger.debug(f"Pattern signature: {region_pattern.signature}")

    def generate(self) -> int:
        """Generate a new Q/DQ insertion scheme for the current pattern or region."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )
        elif self.current_profile_pattern_schemes is None:
            raise InvalidSchemeError("No region selected. Call set_profile_region() first.")

        pattern_schemes = self.current_profile_pattern_schemes
        cached_schemes = [
            (idx, scheme)
            for idx, scheme in enumerate(pattern_schemes.schemes)
            if not scheme.is_profiled
        ]

        if cached_schemes:
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

        known_schemes = {scheme.hash for scheme in pattern_schemes.schemes}
        max_attempts = getattr(self.config, "maximum_generation_attempts", 100)

        logger.debug(f"Generating new scheme ({len(pattern_schemes.schemes)} schemes exist)")

        for attempts in range(max_attempts):
            new_scheme = self._generate_next_insertion_sample()
            if new_scheme.hash not in known_schemes and not new_scheme.error:
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

        logger.warning(f"Could not generate unique scheme after {max_attempts} attempts")
        return -1

    def export_onnx(
        self, output_path: str | None = None, insert_qdq: bool = True, best: bool = False
    ) -> bytes:
        """Export ONNX model with Q/DQ nodes inserted according to tested schemes."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        output_desc = output_path if output_path is not None else "<bytes>"
        original_quant_type = self.config.default_quant_type
        needs_fp8_conversion = insert_qdq and original_quant_type == "fp8"
        resolved_insertion_points = set()

        logger.debug(
            f"Exporting model to {output_desc} (insert_qdq={insert_qdq}, "
            f"regions={len(self.regions)}, profiled_patterns={len(self.profiled_patterns)})"
        )

        # Temporarily set quant type to int8 if FP8 is requested
        if needs_fp8_conversion:
            logger.debug("FP8 conversion: creating INT8 model first")
            self.config.default_quant_type = "int8"

        if insert_qdq:
            matched_regions = 0

            logger.debug(f"Resolving Q/DQ insertion points from {len(self.regions)} regions")

            for region in self.regions:
                pattern = RegionPattern.from_region(region, self.graph)
                logger.debug(f"Region {region.id} (level {region.level})")
                logger.debug(f"  → Pattern signature: {pattern.signature}")

                matched = next((ps for ps in self.profiled_patterns if ps.pattern == pattern), None)
                current_scheme = matched.best_scheme if matched else None

                if matched:
                    if current_scheme:
                        logger.debug(
                            f"  → Matched profiled pattern (latency={current_scheme.latency_ms:.3f} ms)"
                        )
                    else:
                        logger.debug("  → Matched profiled pattern but no valid schemes")

                if current_scheme is None:
                    current_scheme = self.current_profile_pattern_schemes
                    if current_scheme is None or pattern != current_scheme.pattern:
                        pass
                    elif best:
                        current_scheme = current_scheme.best_scheme
                    else:
                        scheme_index = self.current_insertion_scheme_index
                        if scheme_index is not None:
                            assert scheme_index < len(current_scheme.schemes), (
                                f"Invalid scheme index: {scheme_index}"
                            )
                            current_scheme = current_scheme.schemes[scheme_index]
                            logger.debug(f"  → Using current pattern scheme #{scheme_index}")

                if current_scheme is None and self.pattern_cache is not None:
                    pattern_schemes = self.pattern_cache.get_pattern_schemes(pattern.signature)
                    if pattern_schemes is not None:
                        schemes = pattern_schemes.schemes
                        if schemes is not None and len(schemes) == 1 and not schemes[0].is_profiled:
                            current_scheme = schemes[0]
                            logger.debug("  → Using imported pattern from cache")

                if current_scheme is None:
                    logger.debug("  → No scheme available, skipping")
                    continue

                full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)
                assert full_insertion_scheme is not None
                all_region_ips = pattern.matches(region, self.graph, full_insertion_scheme)
                assert isinstance(all_region_ips, set)
                resolved_insertion_points.difference_update(all_region_ips)
                excluded_tensors = all_region_ips - resolved_insertion_points
                if excluded_tensors:
                    logger.debug(
                        f"  → Excluded {len(excluded_tensors)} overlapping insertion points"
                    )

                new_ips = pattern.matches(region, self.graph, current_scheme)
                if new_ips:
                    resolved_insertion_points.update(new_ips)
                    matched_regions += 1
                    logger.debug(f"  → Added {len(new_ips)} insertion points")

            logger.debug(
                f"Matched {matched_regions}/{len(self.regions)} regions, "
                f"total {len(resolved_insertion_points)} unique insertion points"
            )

        graph_copy = self._copy_graph()
        unique_tensors = len(resolved_insertion_points)

        logger.debug(f"Inserting {unique_tensors} Q/DQ pairs into graph")

        if insert_qdq and resolved_insertion_points:
            self._insert_qdq_at_tensors(graph_copy, resolved_insertion_points)

        logger.debug("Serializing to ONNX format")
        model = gs.export_onnx(graph_copy)

        if insert_qdq and resolved_insertion_points:
            self._fix_zero_point_initializers(model)

        if needs_fp8_conversion:
            logger.debug("Converting INT8 to FP8")
            model = int8_to_fp8(model)

        self.config.default_quant_type = original_quant_type
        model_bytes = model.SerializeToString()
        quant_type_str = "baseline"
        output_dest = ""

        if insert_qdq:
            quant_type_str = f"{original_quant_type.upper()}" if needs_fp8_conversion else "INT8"

        if output_path is not None:
            onnx.save(model, output_path)
            output_dest = f" → {output_path}"

        logger.info(
            f"Exported {quant_type_str} model with {unique_tensors} Q/DQ pairs {output_dest}"
        )
        return model_bytes

    def submit(self, latency_ms: float, success: bool = True) -> None:
        """Submit performance measurement for the most recently generated scheme."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        if self.baseline_latency_ms is None:
            self.baseline_latency_ms = latency_ms
            logger.info(f"Baseline latency: {latency_ms:.3f} ms")
            return

        if self.current_profile_pattern_schemes is None:
            raise InvalidSchemeError(
                "No pattern or region selected. Call set_profile_region() first."
            )

        schemes_collection = self.current_profile_pattern_schemes
        if not schemes_collection.schemes:
            raise InvalidSchemeError("No schemes available. Call generate() first.")

        pattern_schemes = schemes_collection

        if self.current_insertion_scheme_index is not None:
            scheme_index = self.current_insertion_scheme_index
            if scheme_index >= len(pattern_schemes.schemes):
                raise InvalidSchemeError(f"Invalid scheme index: {scheme_index}")
            scheme = pattern_schemes.schemes[scheme_index]
        else:
            scheme = pattern_schemes.schemes[-1]
            scheme_index = len(pattern_schemes.schemes) - 1

        scheme.latency_ms = latency_ms
        scheme.error = not success
        scheme.profile_timestamp = datetime.now(timezone.utc).isoformat()
        display_index = scheme_index + 1

        if not success:
            logger.warning(
                f"Scheme #{display_index}: measurement failed (latency={latency_ms:.3f} ms)"
            )
            logger.debug("Marking scheme with error flag")
            return

        speedup = self.baseline_latency_ms / latency_ms if latency_ms > 0 else 0.0

        logger.info(f"Scheme #{display_index}: {latency_ms:.3f} ms ({speedup:.2f}x speedup)")
        logger.debug(f"Compared to baseline: {self.baseline_latency_ms:.3f} ms")

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

        if self.current_profile_pattern_schemes is not None and self.pattern_cache is not None:
            self.pattern_cache.add_pattern_schemes(pattern_schemes)
            logger.debug(
                f"Pattern cache updated: {self.pattern_cache.num_patterns} patterns, "
                f"{self.pattern_cache.total_schemes} schemes"
            )

    def save_state(self, output_path: str) -> None:
        """Save complete autotuner state to a YAML file for later reuse."""
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

        if self.pattern_cache is not None and self.pattern_cache.num_patterns > 0:
            base_path, ext = os.path.splitext(output_path)
            cache_path = f"{base_path}_pattern_cache{ext}"
            self.pattern_cache.save(cache_path)

            logger.info(f"Saved pattern cache → {cache_path}")
            logger.debug(
                f"Cache: {self.pattern_cache.num_patterns} patterns, "
                f"{self.pattern_cache.total_schemes} schemes"
            )

    def load_state(self, input_path: str) -> None:
        """Load autotuner state from a previously saved YAML file."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        with open(input_path) as f:
            state = yaml.safe_load(f)

        if state.get("baseline_latency_ms") is not None:
            self.baseline_latency_ms = state["baseline_latency_ms"]
            logger.debug(f"Baseline latency: {self.baseline_latency_ms:.3f} ms")

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

        if "patterns" in state:
            num_loaded_patterns = 0
            num_loaded_schemes = 0

            for pattern_data in state["patterns"]:
                try:
                    pattern_schemes = PatternSchemes.from_dict(pattern_data)

                    if pattern_schemes.schemes:
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

        base_path, ext = os.path.splitext(input_path)
        cache_path = f"{base_path}_pattern_cache{ext}"

        if os.path.exists(cache_path):
            try:
                loaded_cache = PatternCache.load(cache_path)

                if self.pattern_cache is not None:
                    for pattern_schemes in loaded_cache.pattern_schemes:
                        self.pattern_cache.add_pattern_schemes(pattern_schemes)
                else:
                    self.pattern_cache = loaded_cache
                logger.info(
                    f"Loaded pattern cache from {cache_path} ({loaded_cache.num_patterns} patterns, "
                    f"{loaded_cache.total_schemes} schemes)"
                )
            except Exception as e:
                logger.warning(f"Failed to load pattern cache: {e}")
        else:
            logger.debug(f"No pattern cache file at {cache_path}")

    def import_insertion_points(self, quantized_tensors: set[str] | list[str]) -> None:
        """Import Q/DQ insertion points from a list of quantized tensors and update pattern cache."""
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )

        if isinstance(quantized_tensors, list):
            quantized_tensors = set(quantized_tensors)

        logger.info(f"Importing insertion points from {len(quantized_tensors)} quantized tensors")
        logger.debug(f"Processing {len(self.regions)} regions")

        if self.pattern_cache is None:
            logger.warning("Pattern cache not initialized, skipping import")
            return

        patterns_before = self.pattern_cache.num_patterns
        schemes_before = self.pattern_cache.total_schemes

        for region in self.regions:
            self.pattern_cache.add_pattern_from_region(region, self.graph, quantized_tensors)

        patterns_added = self.pattern_cache.num_patterns - patterns_before
        schemes_added = self.pattern_cache.total_schemes - schemes_before

        logger.info(
            f"Import complete: {patterns_added} patterns, {schemes_added} schemes added to cache"
        )
        logger.debug(
            f"Total cache: {self.pattern_cache.num_patterns} patterns, "
            f"{self.pattern_cache.total_schemes} schemes"
        )

    def _compute_convergence_metrics(
        self, schemes: list[InsertionScheme], best_scheme: InsertionScheme | None
    ) -> tuple[int | None, float | None]:
        """Compute convergence metrics for a collection of schemes."""
        samples_before_best = None
        time_to_best = None

        if not best_scheme or not best_scheme.profile_timestamp:
            return samples_before_best, time_to_best

        schemes_with_time = [s for s in schemes if s.profile_timestamp is not None]

        if not schemes_with_time:
            return samples_before_best, time_to_best

        from datetime import datetime

        schemes_with_time.sort(key=lambda s: s.profile_timestamp or "")

        try:
            best_position = next(
                i for i, s in enumerate(schemes_with_time) if s.hash == best_scheme.hash
            )
            samples_before_best = best_position

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

    def _mutate_insertion_points(
        self, base_points, all_points, point_type: str, max_mutations: int
    ) -> list:
        """Mutate a set of insertion points by adding, removing, or both."""
        key_fn = {
            "node input points": lambda p: (p.node_index, p.input_index),
            "region composite points": lambda p: (p.region_index, p.input_index),
            "region output points": lambda p: (p.region_index, p.node_index, p.output_index),
        }.get(point_type)

        if not key_fn:
            return []

        current_points = set(base_points)
        initial_count = len(current_points)
        mutation_type = random.choice(["add", "remove", "both"])

        if mutation_type in ["add", "both"] and len(current_points) < len(all_points):
            all_keys = {key_fn(p) for p in all_points}
            available_keys = all_keys - current_points
            if available_keys:
                max_add = min(max_mutations, len(available_keys))
                num_to_add = random.randint(1, max_add)
                to_add = random.sample(list(available_keys), num_to_add)
                current_points.update(to_add)

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

        return [p for p in all_points if key_fn(p) in current_points]

    def _generate_next_insertion_sample(self) -> InsertionScheme:
        """Generate a new insertion scheme by mutating top performers."""
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

        if not isinstance(schemes_collection, PatternSchemes) or schemes_collection.pattern is None:
            return InsertionScheme()
        pattern = schemes_collection.pattern
        full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)

        logger.debug(
            f"Available insertion points: {len(full_insertion_scheme.node_inputs)} node input, "
            f"{len(full_insertion_scheme.child_region_inputs)} region composite, "
            f"{len(full_insertion_scheme.region_outputs)} region output"
        )

        top_percent = getattr(self.config, "top_percent_to_mutate", 0.1)
        minimum_schemes = getattr(self.config, "minimum_schemes_to_mutate", 1)

        measured_schemes = [s for s in pattern_schemes.schemes if s.latency_ms > 0 and not s.error]
        measured_schemes.sort(key=lambda s: s.latency_ms)

        num_top_schemes = max(
            int(len(measured_schemes) * top_percent), min(minimum_schemes, len(measured_schemes))
        )
        top_schemes = measured_schemes[:num_top_schemes]

        if len(top_schemes) == 0:
            logger.debug("No measured schemes yet, generating baseline (empty) scheme")
            return InsertionScheme()

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

        max_mutations = getattr(self.config, "maximum_mutations", 3)

        scheme = InsertionScheme()
        base_node_points = {(p.node_index, p.input_index) for p in base_scheme.node_inputs}
        scheme.node_inputs = self._mutate_insertion_points(
            base_node_points, full_insertion_scheme.node_inputs, "node input points", max_mutations
        )

        base_region_composite_points = {
            (p.region_index, p.input_index) for p in base_scheme.child_region_inputs
        }
        scheme.child_region_inputs = self._mutate_insertion_points(
            base_region_composite_points,
            full_insertion_scheme.child_region_inputs,
            "region composite points",
            max_mutations,
        )

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

    def _copy_graph(self) -> gs.Graph:
        """Create an independent copy of the computation graph."""
        new_graph = gs.import_onnx(self.onnx_model)
        new_graph.toposort()
        return new_graph

    def _get_quant_dtype(self, quant_type: str) -> np.dtype:
        """Get numpy dtype for quantization type."""
        if quant_type == "fp8":
            try:
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
        """Convert DQ dtype string to numpy dtype."""
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
        }

        if hasattr(np, "bfloat16"):
            dtype_map["bfloat16"] = np.bfloat16

        if dtype_str not in dtype_map:
            logger.warning(f"Unknown DQ dtype '{dtype_str}', defaulting to float32")
            return np.float32

        return dtype_map[dtype_str]

    def _build_tensor_map(self, graph: gs.Graph) -> dict[str, gs.Tensor]:
        """Build mapping from tensor names to tensor objects."""
        tensor_map = {}

        for node in graph.nodes:
            for output in node.outputs:
                if hasattr(output, "name") and output.name:
                    tensor_map[output.name] = output

        for input_tensor in graph.inputs:
            if hasattr(input_tensor, "name") and input_tensor.name:
                tensor_map[input_tensor.name] = input_tensor

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
        """Extract shape and dtype metadata from a tensor."""
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
        return None, default_dtype

    def _fix_zero_point_initializers(self, model: onnx.ModelProto) -> None:
        """Fix INT8 zero_point initializers to use int32_data instead of raw_data."""
        fixed_count = 0

        for initializer in model.graph.initializer:
            if (
                "_zp_" in initializer.name
                and initializer.data_type == onnx.TensorProto.INT8
                and len(initializer.raw_data) > 0
                and len(initializer.int32_data) == 0
            ):
                np_array = onnx.numpy_helper.to_array(initializer)
                int32_values = np_array.astype(np.int32).flatten().tolist()

                new_tensor = onnx.helper.make_tensor(
                    initializer.name,
                    onnx.TensorProto.INT8,
                    list(initializer.dims),
                    int32_values,
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
        """Create QuantizeLinear and DequantizeLinear node pair."""
        # Create unique names for Q/DQ nodes
        q_name = f"QDQ_Q_{tensor_name}".replace("/", "_").replace(":", "_")
        dq_name = f"QDQ_DQ_{tensor_name}".replace("/", "_").replace(":", "_")

        # Determine scale dtype from output_dtype (fp16/tf32/fp32)
        # Scale should match the precision of the original I/O tensor
        dtype_map = {"float16": np.float16, "float32": np.float32}
        if hasattr(np, "bfloat16"):
            dtype_map["bfloat16"] = np.bfloat16
        scale_dtype = dtype_map.get(np.dtype(output_dtype).name, np.float32)

        logger.debug(
            f"Creating Q/DQ pair for '{tensor_name}' (scale_dtype={np.dtype(scale_dtype).name})"
        )

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
        """Insert Q/DQ (Quantize/Dequantize) node pairs at specified locations."""
        q_scale = self.config.default_q_scale
        quant_type = self.config.default_quant_type
        quant_dtype = self._get_quant_dtype(quant_type)

        logger.debug(f"Q/DQ parameters: type={quant_type}, scale={q_scale}, zero_point=0")

        resolved_insertion_points = merge_resolved_insertion_points(
            graph, resolved_insertion_points
        )

        tensor_map = self._build_tensor_map(graph)
        tensor_users_map = get_tensor_consumer_node_indices(graph)
        logger.debug(
            f"Built tensor maps: {len(tensor_map)} tensors, {len(tensor_users_map)} with users"
        )

        for insertion_point in resolved_insertion_points:
            tensor_name = insertion_point.tensor_name
            node_index = insertion_point.node_index
            input_index = insertion_point.input_index

            original_tensor = tensor_map[tensor_name]
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

            is_constant = isinstance(original_tensor, gs.Constant)
            output_shape, output_dtype = self._get_tensor_metadata(original_tensor, is_constant)

            unique_suffix = "qdq"
            if node_index is not None:
                unique_suffix = f"n{node_index}_i{input_index}"
            unique_tensor_name = f"{tensor_name}_{unique_suffix}"

            q_node, dq_node = self._create_qdq_nodes(
                unique_tensor_name,
                original_tensor,
                output_shape,
                output_dtype,
                quant_dtype,
                quant_type,
                q_scale,
            )

            graph.nodes.extend([q_node, dq_node])

            if node_index is not None:
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

        logger.debug("Running graph cleanup and topological sort")
        try:
            graph.cleanup().toposort()
            logger.debug("Graph cleanup completed")
        except Exception as e:
            logger.warning(f"Graph cleanup failed: {e}")
            logger.debug("Continuing anyway")


class QDQAutotuner(QDQAutotunerBase):
    """Q/DQ autotuner with automatic region discovery around compute-intensive ops."""

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuner and discover optimization regions automatically."""
        super().initialize(config, pattern_cache)
        self._search_regions()

    def _visit_region_recursively(self, region: Region) -> list[Region]:
        """Recursively traverse region hierarchy and collect all regions."""
        regions = [region]

        for child in region.get_children():
            regions.extend(self._visit_region_recursively(child))

        return regions

    def _reassign_region_ids(self, regions: list[Region]) -> None:
        """Reassign sequential IDs to regions in breadth-first order."""
        region_id = 0

        queue = deque(regions)

        while queue:
            region = queue.popleft()
            region.id = region_id
            region_id += 1
            queue.extend(region.get_children())

    def _search_regions(self) -> None:
        """Discover and organize optimization regions automatically."""
        logger.info("Discovering optimization regions")
        search = CombinedRegionSearch(
            self.graph,
            maximum_sequence_region_size=self.config.maximum_sequence_region_size,
            minimum_topdown_search_size=self.config.minimum_topdown_search_size,
        )
        self.regions = search.search_regions()

        self._reassign_region_ids(self.regions)
        logger.debug(f"Found {len(self.regions)} top-level regions")

        all_regions = []
        for region in self.regions:
            all_regions.extend(self._visit_region_recursively(region))

        logger.debug(f"Flattened hierarchy to {len(all_regions)} total regions")

        leaf_regions = [region for region in all_regions if region.type == RegionType.LEAF]
        other_regions = [region for region in all_regions if region.type != RegionType.LEAF]

        all_regions = leaf_regions + other_regions
        self.regions = all_regions

        num_leaf = sum(1 for r in self.regions if r.type == RegionType.LEAF)
        num_composite = sum(1 for r in self.regions if r.type == RegionType.COMPOSITE)
        num_root = sum(1 for r in self.regions if r.type == RegionType.ROOT)

        logger.info(
            f"Discovery complete: {len(self.regions)} regions "
            f"({num_leaf} LEAF, {num_composite} COMPOSITE, {num_root} ROOT)"
        )
        logger.debug("Regions prioritized: LEAF regions first for profiling")
