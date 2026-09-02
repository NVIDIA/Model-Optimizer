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

"""Base implementation for pattern-based Q/DQ insertion optimization in ONNX models.

This module defines QDQAutotunerBase, which implements the core autotuning workflow:
region-aware scheme resolution, Q/DQ insertion point matching, scheme generation via
mutation, and export (delegating to export_utils for actual Q/DQ insertion and ONNX
serialization). Subclasses such as QDQAutotuner add region discovery (e.g., automatic
search around compute-intensive ops); this base does not populate regions itself and
expects them to be set by a subclass or caller before profiling and export.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import os
import random
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import onnx
import onnx_graphsurgeon as gs
import yaml

from modelopt.onnx.logging_config import logger
from modelopt.onnx.op_types import get_activation_ops, is_linear_op
from modelopt.onnx.quantization.autotune.common import (
    AutotunerNotInitializedError,
    Config,
    InsertionScheme,
    InvalidSchemeError,
    PatternCache,
    PatternSchemes,
    Region,
    SchemeAction,
    _atomic_yaml_dump,
    is_valid_latency,
)
from modelopt.onnx.quantization.autotune.export_utils import export_qdq_onnx
from modelopt.onnx.quantization.autotune.insertion_points import (
    ResolvedInsertionPoint,
    get_autotuner_quantizable_ops,
)
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.graph_utils import get_tensor_consumer_node_indices

_MUTATION_SPECS = [
    ("node_inputs", "node input points", lambda p: (p.node_index, p.input_index)),
    (
        "child_region_inputs",
        "region composite points",
        lambda p: (p.region_index, p.input_index),
    ),
    (
        "region_outputs",
        "region output points",
        lambda p: (p.region_index, p.node_index, p.output_index),
    ),
]

_FP8_AUTOTUNER_QUANTIZABLE_OPS = ("Conv", "Gemm", "MatMul", "Add")


def _requires_init(method):
    """Decorator that raises AutotunerNotInitializedError if initialize() has not been called."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            raise AutotunerNotInitializedError(
                "QDQAutotunerBase not initialized. Call initialize() first."
            )
        return method(self, *args, **kwargs)

    return wrapper


class QDQAutotunerBase:
    """Base class for pattern-based Q/DQ node insertion optimization in ONNX models."""

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
        """
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
        self.model_sha256 = hashlib.sha256(
            self.onnx_model.SerializeToString(deterministic=True)
        ).hexdigest()
        self.resume_fingerprint: dict[str, Any] = {}
        self.force_no_qdq = False
        self.proxy_decision: dict[str, Any] | None = None
        self.proxy_selection: str | None = None
        self.proxy_baseline_latency_ms: float | None = None
        self.proxy_candidate_latency_ms: float | None = None
        self.proxy_candidate_quantization_site_count: int | None = None
        self.proxy_candidate_model_sha256: str | None = None
        self.proxy_selected_model_sha256: str | None = None
        self.final_baseline_measurement: dict[str, Any] | None = None
        self.final_decision: dict[str, Any] | None = None
        self.decision_stage: str | None = None
        self.baseline_final_latency_ms: float | None = None
        self.baseline_model_sha256: str | None = None
        self.final_selection: str | None = None
        self.candidate_final_latency_ms: float | None = None
        self.final_latency_ms: float | None = None
        self.candidate_quantization_site_count: int | None = None
        self.candidate_model_sha256: str | None = None
        self.selected_model_sha256: str | None = None

        logger.debug(f"Initialized autotuner with model type: {type(model).__name__}")

    requires_init = _requires_init

    def initialize(
        self, config: Config | None = None, pattern_cache: PatternCache | None = None
    ) -> None:
        """Initialize autotuning session with configuration and pattern cache.

        Prepares the autotuner for profiling by setting configuration parameters
        and optionally loading pattern cache data. This base method resets all profiling
        state and sets up the pattern cache storage.

        Args:
            config: Autotuning configuration parameters. If None, uses default Config().
                   Controls Q/DQ parameters, performance thresholds, and scheme generation.
            pattern_cache: Optional PatternCache object for seeding with known-good schemes.
                        If None, creates a new empty pattern cache for tracking best schemes.
                        If provided, uses existing schemes to warm-start optimization.

        Raises:
            None (safe to call multiple times - will reset state each time)
        """
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
        self.resume_fingerprint = {}
        self.force_no_qdq = False
        self._clear_proxy_decision()
        self._clear_final_decision()

        logger.info("Initializing autotuner")
        logger.debug(
            f"Configuration: q_scale={self.config.default_q_scale}, "
            f"q_zero_point={self.config.default_q_zero_point}, quant_type={self.config.default_quant_type}"
        )

        self.initialized = True

    def _clear_proxy_decision(self) -> None:
        self.proxy_decision = None
        self.proxy_selection = None
        self.proxy_baseline_latency_ms = None
        self.proxy_candidate_latency_ms = None
        self.proxy_candidate_quantization_site_count = None
        self.proxy_candidate_model_sha256 = None
        self.proxy_selected_model_sha256 = None

    def _clear_final_decision(self) -> None:
        self.final_baseline_measurement = None
        self.final_decision = None
        self.decision_stage = None
        self.baseline_final_latency_ms = None
        self.baseline_model_sha256 = None
        self.final_selection = None
        self.candidate_final_latency_ms = None
        self.final_latency_ms = None
        self.candidate_quantization_site_count = None
        self.candidate_model_sha256 = None
        self.selected_model_sha256 = None

    @property
    def candidate_qdq_count(self) -> int | None:
        """Backward-compatible alias for the quantization-site count."""
        return self.candidate_quantization_site_count

    @candidate_qdq_count.setter
    def candidate_qdq_count(self, count: int | None) -> None:
        self.candidate_quantization_site_count = count

    @property
    def proxy_candidate_qdq_count(self) -> int | None:
        """Backward-compatible alias for the proxy quantization-site count."""
        return self.proxy_candidate_quantization_site_count

    @staticmethod
    def _resolve_quantization_site_count(
        candidate_quantization_site_count: int | None,
        candidate_qdq_count: int | None,
    ) -> int:
        if (
            candidate_quantization_site_count is not None
            and candidate_qdq_count is not None
            and candidate_quantization_site_count != candidate_qdq_count
        ):
            raise ValueError("quantization-site count aliases must match")
        count = (
            candidate_quantization_site_count
            if candidate_quantization_site_count is not None
            else candidate_qdq_count
        )
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("candidate_quantization_site_count must be a non-negative integer")
        return count

    def set_resume_fingerprint(self, **components: Any) -> None:
        """Set normalized benchmark/runtime identity used to validate resumed measurements."""
        self.resume_fingerprint = json.loads(json.dumps(components, sort_keys=True, default=str))

    def set_force_no_qdq(self, enabled: bool = True) -> None:
        """Force final resolution to the high-precision model without Q/DQ."""
        self.force_no_qdq = bool(enabled)

    def record_proxy_decision(
        self,
        *,
        proxy_selection: str,
        baseline_latency_ms: float,
        candidate_latency_ms: float | None,
        candidate_model_sha256: str,
        selected_model_sha256: str,
        candidate_quantization_site_count: int | None = None,
        candidate_qdq_count: int | None = None,
    ) -> None:
        """Record the uncalibrated workflow decision as proxy telemetry."""
        if proxy_selection not in {SchemeAction.QDQ.value, SchemeAction.NO_QDQ.value}:
            raise ValueError("proxy_selection must be 'qdq' or 'no_qdq'")
        if not is_valid_latency(baseline_latency_ms):
            raise ValueError("baseline_latency_ms must be finite and positive")
        if candidate_latency_ms is not None and not is_valid_latency(candidate_latency_ms):
            raise ValueError("candidate_latency_ms must be finite and positive")
        site_count = self._resolve_quantization_site_count(
            candidate_quantization_site_count, candidate_qdq_count
        )
        if not candidate_model_sha256 or not selected_model_sha256:
            raise ValueError("model SHA256 values must be non-empty")
        if proxy_selection == SchemeAction.QDQ.value and (
            candidate_latency_ms is None or site_count == 0
        ):
            raise ValueError("a Q/DQ proxy selection requires a measured quantized candidate")

        self.proxy_selection = proxy_selection
        self.proxy_baseline_latency_ms = baseline_latency_ms
        self.proxy_candidate_latency_ms = candidate_latency_ms
        self.proxy_candidate_quantization_site_count = site_count
        self.proxy_candidate_model_sha256 = candidate_model_sha256
        self.proxy_selected_model_sha256 = selected_model_sha256
        self.proxy_decision = {
            "proxy_selection": proxy_selection,
            "baseline_latency_ms": baseline_latency_ms,
            "candidate_latency_ms": candidate_latency_ms,
            "candidate_quantization_site_count": site_count,
            "candidate_model_sha256": candidate_model_sha256,
            "selected_model_sha256": selected_model_sha256,
        }

    def record_final_baseline_measurement(
        self,
        *,
        decision_stage: str,
        baseline_final_latency_ms: float,
        baseline_model_sha256: str,
    ) -> None:
        """Record the calibrated no-Q/DQ reference before measuring the candidate."""
        if decision_stage != "calibrated_baseline":
            raise ValueError("baseline decision_stage must be 'calibrated_baseline'")
        if not is_valid_latency(baseline_final_latency_ms):
            raise ValueError("baseline_final_latency_ms must be finite and positive")
        if not baseline_model_sha256:
            raise ValueError("baseline_model_sha256 must be non-empty")

        self._clear_final_decision()
        self.decision_stage = decision_stage
        self.baseline_final_latency_ms = baseline_final_latency_ms
        self.baseline_model_sha256 = baseline_model_sha256
        self.final_baseline_measurement = {
            "decision_stage": decision_stage,
            "baseline_final_latency_ms": baseline_final_latency_ms,
            "baseline_model_sha256": baseline_model_sha256,
        }

    def record_final_decision(
        self,
        *,
        decision_stage: str,
        final_selection: str,
        candidate_final_latency_ms: float | None,
        final_latency_ms: float,
        candidate_model_sha256: str,
        selected_model_sha256: str,
        candidate_quantization_site_count: int | None = None,
        candidate_qdq_count: int | None = None,
    ) -> None:
        """Record the calibrated final artifact decision for checkpointing."""
        if decision_stage != "calibrated_final":
            raise ValueError("final decision_stage must be 'calibrated_final'")
        if final_selection not in {SchemeAction.QDQ.value, SchemeAction.NO_QDQ.value}:
            raise ValueError("final_selection must be 'qdq' or 'no_qdq'")
        if candidate_final_latency_ms is not None and not is_valid_latency(
            candidate_final_latency_ms
        ):
            raise ValueError("candidate_final_latency_ms must be finite and positive")
        if not is_valid_latency(final_latency_ms):
            raise ValueError("final_latency_ms must be finite and positive")
        site_count = self._resolve_quantization_site_count(
            candidate_quantization_site_count, candidate_qdq_count
        )
        if not candidate_model_sha256 or not selected_model_sha256:
            raise ValueError("model SHA256 values must be non-empty")
        if not is_valid_latency(self.baseline_final_latency_ms) or not self.baseline_model_sha256:
            raise ValueError("record the calibrated baseline before the final decision")
        if final_selection == SchemeAction.QDQ.value and (
            candidate_final_latency_ms is None or site_count == 0
        ):
            raise ValueError("a Q/DQ final selection requires a measured quantized candidate")

        self.decision_stage = decision_stage
        self.final_selection = final_selection
        self.candidate_final_latency_ms = candidate_final_latency_ms
        self.final_latency_ms = final_latency_ms
        self.candidate_quantization_site_count = site_count
        self.candidate_model_sha256 = candidate_model_sha256
        self.selected_model_sha256 = selected_model_sha256
        self.final_decision = {
            "decision_stage": decision_stage,
            "baseline_final_latency_ms": self.baseline_final_latency_ms,
            "baseline_model_sha256": self.baseline_model_sha256,
            "final_selection": final_selection,
            "candidate_final_latency_ms": candidate_final_latency_ms,
            "final_latency_ms": final_latency_ms,
            "candidate_quantization_site_count": site_count,
            "candidate_model_sha256": candidate_model_sha256,
            "selected_model_sha256": selected_model_sha256,
        }

    def _commit_current_pattern(self, save: bool = True) -> None:
        """Save current pattern schemes to profiled_patterns (if save) and clear current state."""
        if save and self.current_profile_pattern_schemes is not None:
            pattern_schemes = self.current_profile_pattern_schemes
            pattern_schemes.select_best(self.config.performance_threshold)
            pattern_schemes.completed = True
            num_schemes = len(pattern_schemes.schemes)
            selected_scheme = pattern_schemes.selected_scheme
            selected_latency = (
                selected_scheme.latency_ms if selected_scheme is not None else float("inf")
            )

            samples_before_best, time_to_best = self._compute_convergence_metrics(
                pattern_schemes.schemes, selected_scheme
            )

            logger.info(
                f"Pattern complete: {num_schemes} schemes tested, "
                f"selected latency {selected_latency:.3f} ms"
            )
            logger.debug(f"Pattern signature: {pattern_schemes.pattern_signature}")
            if samples_before_best is not None:
                logger.debug(f"Convergence: best found at sample {samples_before_best}")
            if time_to_best is not None:
                logger.debug(f"Time to best: {time_to_best:.2f}s")
            self.profiled_patterns.append(pattern_schemes)
            if self.pattern_cache is not None:
                self.pattern_cache.add_pattern_schemes(pattern_schemes)

        self.current_profile_region = None
        self.current_profile_pattern_schemes = None
        self.current_insertion_scheme_index = None

    def _seed_from_cache(self, pattern: RegionPattern) -> tuple[PatternSchemes | None, int]:
        """Seed PatternSchemes from pattern cache for the given pattern. Returns (schemes, num_seeded)."""
        if self.pattern_cache is None:
            return None, 0
        cache_schemes = self.pattern_cache.get_pattern_schemes(pattern.signature)
        if cache_schemes is None or len(cache_schemes.schemes) == 0:
            logger.debug("No pattern cache entries for this region")
            return None, 0
        pattern_schemes = PatternSchemes()
        pattern_schemes.pattern = pattern
        num_seeded = 0
        for cached_scheme in cache_schemes.schemes:
            if not cached_scheme.is_qdq:
                continue
            scheme_copy = copy.deepcopy(cached_scheme)
            scheme_copy.action = SchemeAction.QDQ
            scheme_copy.latency_ms = float("inf")
            scheme_copy.error = False
            if hasattr(scheme_copy, "profile_timestamp"):
                scheme_copy.profile_timestamp = None
            pattern_schemes.schemes.append(scheme_copy)
            num_seeded += 1
        pattern_schemes.ensure_control_schemes()
        logger.debug(f"Seeded {num_seeded} scheme(s) from pattern cache")
        return pattern_schemes, num_seeded

    @_requires_init
    def set_profile_region(self, region: Region | None, commit: bool = True) -> None:
        """Set the target region for profiling and scheme generation.

        This method manages the profiling workflow:
        1. If commit=True: Saves current schemes to profiled_patterns
        2. Creates a RegionPattern from the new region's structure
        3. For pattern-based: tries to seed schemes from pattern cache if available
        4. Sets as current for generate() and submit() calls

        Pass region=None to clear the current profile target without setting a new one.

        Args:
            region: The region to profile next (None to clear current target)
            commit: If True, commit current schemes to profiled_patterns
                   before switching. Set to False during initialization.

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
        """
        if region is None:
            self._commit_current_pattern(save=commit)
            return

        if region not in self.regions:
            raise ValueError(f"Region {region.id} not found in regions")

        region_pattern = RegionPattern.from_region(region, self.graph)

        if self.current_profile_region is not None and self.current_profile_region.id == region.id:
            if self.current_profile_pattern_schemes is not None:
                self.current_profile_pattern_schemes.ensure_control_schemes()
            return

        if self.current_profile_pattern_schemes is not None:
            if not commit:
                raise ValueError("Cannot replace an active profile region without committing it")
            self._commit_current_pattern()

        if self._is_region_profiled(region):
            logger.info(f"Skipping region {region.id} (pattern already profiled)")
            logger.debug(f"Pattern signature: {region_pattern.signature}")
            return

        pattern_schemes, num_seeded = self._seed_from_cache(region_pattern)
        if pattern_schemes is None:
            pattern_schemes = PatternSchemes()
            pattern_schemes.pattern = region_pattern
            logger.debug("Initialized with empty scheme collection")
        pattern_schemes.ensure_control_schemes()

        self.current_profile_region = region
        self.current_profile_pattern_schemes = pattern_schemes

        mode_info = f"seeded with {num_seeded} schemes" if num_seeded > 0 else "starting fresh"
        logger.info(
            f"Profiling region {region.id} [level {region.level}, size"
            f"{region.get_size_of_region_and_descendants()}, {mode_info}]"
        )
        logger.debug(f"Pattern signature: {region_pattern.signature}")

    @_requires_init
    def begin_inherit_profile(self) -> int:
        """Select the mandatory INHERIT control, or return -1 if it is already measured."""
        if self.current_profile_pattern_schemes is None:
            raise InvalidSchemeError("No region selected. Call set_profile_region() first.")
        self.current_profile_pattern_schemes.ensure_control_schemes()
        scheme_index = next(
            idx
            for idx, scheme in enumerate(self.current_profile_pattern_schemes.schemes)
            if scheme.is_inherit
        )
        if self.current_profile_pattern_schemes.schemes[scheme_index].is_profiled:
            return -1
        self.current_insertion_scheme_index = scheme_index
        return scheme_index

    @_requires_init
    def submit_inherit(self, latency_ms: float, success: bool = True) -> None:
        """Submit the mandatory INHERIT control measurement."""
        if self.baseline_latency_ms is None:
            raise InvalidSchemeError("Measure the global baseline before the INHERIT control")
        if self.current_profile_pattern_schemes is None:
            raise InvalidSchemeError("No region selected. Call set_profile_region() first.")
        scheme_index = self.current_insertion_scheme_index
        if (
            scheme_index is None
            or not self.current_profile_pattern_schemes.schemes[scheme_index].is_inherit
        ):
            raise InvalidSchemeError("Call begin_inherit_profile() before submit_inherit()")
        self.submit(latency_ms, success=success and is_valid_latency(latency_ms))

    @_requires_init
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

        """
        if self.current_profile_pattern_schemes is None:
            raise InvalidSchemeError("No region selected. Call set_profile_region() first.")

        pattern_schemes = self.current_profile_pattern_schemes
        pattern_schemes.ensure_control_schemes()
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

            action = cached_scheme_data.action.value
            logger.info(
                f"Scheme #{scheme_index + 1}: profiling {action} scheme "
                f"({total_points} Q/DQ points)"
            )
            logger.debug(
                f"Cached scheme breakdown: {num_node_points} node input, "
                f"{num_region_composite_points} region composite, "
                f"{num_region_output_points} region output points ({len(cached_schemes)} cached schemes remaining)"
            )

            self.current_insertion_scheme_index = scheme_index
            return self.current_insertion_scheme_index

        if pattern_schemes.search_exhausted:
            return -1

        known_schemes = {scheme.hash for scheme in pattern_schemes.schemes}
        max_attempts = self.config.maximum_generation_attempts
        rng = self._get_generation_rng(pattern_schemes)

        logger.debug(f"Generating new scheme ({len(pattern_schemes.schemes)} schemes exist)")

        for attempts in range(max_attempts):
            new_scheme = self._generate_next_insertion_sample(rng)
            if new_scheme.is_qdq and new_scheme.hash not in known_schemes and not new_scheme.error:
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
        pattern_schemes.search_exhausted = True
        return -1

    def _get_generation_rng(self, pattern_schemes: PatternSchemes) -> random.Random:
        state = {
            "model_sha256": self.model_sha256,
            "pattern_signature": pattern_schemes.pattern_signature,
            "schemes": [
                {
                    "hash": scheme.hash,
                    "latency_ms": str(scheme.latency_ms),
                    "error": scheme.error,
                }
                for scheme in pattern_schemes.schemes
            ],
        }
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest, byteorder="big"))

    def _resolve_scheme_for_region(
        self, region: Region, best: bool
    ) -> tuple[InsertionScheme | None, RegionPattern]:
        """Resolve the insertion scheme to use for a region from profiled/current/cache.

        Args:
            region: The region to resolve the scheme for
            best: If True, return the best scheme for the region

        Returns:
            tuple[InsertionScheme | None, RegionPattern]: The scheme and pattern for the region
        """
        pattern = RegionPattern.from_region(region, self.graph)
        logger.debug(f"Region {region.id} (level {region.level})")
        logger.debug(f"  → Pattern signature: {pattern.signature}")

        matched = next((ps for ps in self.profiled_patterns if ps.pattern == pattern), None)
        current_scheme = matched.selected_scheme if matched else None

        if matched:
            if current_scheme:
                logger.debug(
                    f"  → Matched profiled pattern (latency={current_scheme.latency_ms:.3f} ms)"
                )
            else:
                logger.debug("  → Matched profiled pattern but no valid schemes")

        if current_scheme is None:
            pattern_schemes = self.current_profile_pattern_schemes
            if pattern_schemes is None or pattern != pattern_schemes.pattern:
                pass
            elif best:
                current_scheme = pattern_schemes.selected_scheme
            else:
                scheme_index = self.current_insertion_scheme_index
                if scheme_index is not None:
                    if scheme_index < 0 or scheme_index >= len(pattern_schemes.schemes):
                        raise IndexError(
                            f"Invalid scheme index: {scheme_index} "
                            f"(pattern has {len(pattern_schemes.schemes)} schemes)"
                        )
                    current_scheme = pattern_schemes.schemes[scheme_index]
                    logger.debug(f"  → Using current pattern scheme #{scheme_index}")

        if current_scheme is None:
            logger.debug("  → No scheme available, skipping")

        return current_scheme, pattern

    def _exclude_overlapping_insertion_points(
        self,
        resolved_insertion_points: set[ResolvedInsertionPoint],
        region: Region,
        pattern: RegionPattern,
    ) -> None:
        """Remove this region's full insertion points from resolved set so they can be replaced."""
        full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)
        if full_insertion_scheme is None:
            raise ValueError("get_full_insertion_scheme returned None")
        all_region_ips = pattern.matches(region, self.graph, full_insertion_scheme)
        for ip in all_region_ips:
            node = self.graph.nodes[ip.node_index]
            # Conv/ConvTranspose/Gemm/MatMul inputs and weights must be excluded together
            if is_linear_op(node.op) and ip.input_index == 0 and len(node.inputs) >= 2:
                resolved_insertion_points.discard(ip)
                resolved_insertion_points.discard(
                    ResolvedInsertionPoint(
                        tensor_name=node.inputs[1].name,
                        node_index=ip.node_index,
                        input_index=1,
                    )
                )
        if not isinstance(all_region_ips, set):
            raise TypeError(
                f"pattern.matches must return a set, got {type(all_region_ips).__name__}"
            )
        resolved_insertion_points.difference_update(all_region_ips)
        if all_region_ips:
            logger.debug(f"  → Excluded {len(all_region_ips)} overlapping insertion points")

    @_requires_init
    def get_resolved_insertion_points(
        self, best: bool = True, verbose: bool = False
    ) -> set[ResolvedInsertionPoint]:
        """Compute Q/DQ insertion points for the best schemes (assuming best=True).

        Args:
            best: If True, use the best scheme for each region. If False, use the current scheme.
            verbose: If True, log matched-region counts and per-region insertion point details.

        Returns:
            Set of ResolvedInsertionPoint objects representing where Q/DQ pairs should be inserted.

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
        """
        resolved_insertion_points: set[ResolvedInsertionPoint] = set()
        matched_regions = 0

        if self.force_no_qdq:
            return resolved_insertion_points

        if verbose:
            logger.debug(f"Resolving Q/DQ insertion points from {len(self.regions)} regions")

        for region in self.regions:
            current_scheme, pattern = self._resolve_scheme_for_region(region, best)
            if current_scheme is None:
                continue
            if current_scheme.is_inherit:
                continue
            self._exclude_overlapping_insertion_points(resolved_insertion_points, region, pattern)
            if current_scheme.is_no_qdq:
                matched_regions += 1
                continue
            new_insertion_points = pattern.matches(region, self.graph, current_scheme)
            if new_insertion_points:
                resolved_insertion_points.update(new_insertion_points)
                matched_regions += 1
                if verbose:
                    logger.debug(f"  → Added {len(new_insertion_points)} insertion points")
        if verbose:
            logger.debug(
                f"Matched {matched_regions}/{len(self.regions)} regions, "
                f"total {len(resolved_insertion_points)} unique insertion points"
            )
        return resolved_insertion_points

    @_requires_init
    def get_ort_quantization_config(
        self,
    ) -> tuple[list[str], list[str], list[tuple[gs.Node, gs.Node, str]], list[str]]:
        """Derive ORT quantization configuration from resolved insertion points.

        Returns the four parameters consumed by INT8 and FP8 quantize() to replicate the autotuner's
        Q/DQ placement decisions without exporting any intermediate ONNX file to disk.

        Returns:
            nodes_to_quantize: Node names that have at least one covered Q/DQ input.
            op_types_to_quantize: Op types eligible for quantization.
            no_quantize_inputs: List of (src_node, dst_node, tensor_name) tuples for inputs
              of quantized nodes that should NOT receive Q/DQ.
            op_types_needing_output_quant: Producer op types whose output feeds a covered
              activation-op input (needed so ORT inserts Q/DQ between e.g. Add and Relu).

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called.
        """
        resolved_ips = self.get_resolved_insertion_points(best=True)
        if not resolved_ips:
            return [], [], [], []

        graph = self.graph

        # Build (node_index, input_index) pairs that have Q/DQ
        covered: set[tuple[int, int]] = set()
        for ip in resolved_ips:
            if ip.node_index is not None and ip.input_index is not None:
                covered.add((ip.node_index, ip.input_index))
            else:
                # Tensor-level insertion point: expand to all consumer (node, input) pairs
                for consumer_idx in graph.tensor_users_map.get(ip.tensor_name, []):
                    node = graph.nodes[consumer_idx]
                    for inp_idx, inp in enumerate(node.inputs):
                        if getattr(inp, "name", None) == ip.tensor_name:
                            covered.add((consumer_idx, inp_idx))

        # Nodes that consume a covered (DQ-fed) input
        quantized_node_indices: set[int] = {node_idx for node_idx, _ in covered}

        # Also include producer nodes of covered inputs: a producer whose output feeds a
        # covered slot needs to be in nodes_to_quantize so ORT can place Q on its output
        # (e.g., Add must be included when Q/DQ sits between Add and Relu).
        node_name_to_idx = {node.name: i for i, node in enumerate(graph.nodes)}
        for node_idx, inp_idx in covered:
            tensor = graph.nodes[node_idx].inputs[inp_idx]
            if tensor.inputs:
                producer_idx = node_name_to_idx.get(tensor.inputs[0].name)
                if producer_idx is not None:
                    quantized_node_indices.add(producer_idx)

        sorted_quantized_node_indices = sorted(quantized_node_indices)
        nodes_to_quantize = [graph.nodes[i].name for i in sorted_quantized_node_indices]
        op_types_to_quantize = sorted(
            _FP8_AUTOTUNER_QUANTIZABLE_OPS
            if self.config.default_quant_type == "fp8"
            else get_autotuner_quantizable_ops()
        )

        # Inputs of quantized nodes NOT covered by Q/DQ (only non-constant producer inputs)
        no_quantize_inputs: list[tuple[gs.Node, gs.Node, str]] = []
        for node_idx in sorted_quantized_node_indices:
            node = graph.nodes[node_idx]
            for inp_idx, inp in enumerate(node.inputs):
                if (node_idx, inp_idx) not in covered and getattr(inp, "name", None):
                    if inp.inputs:
                        no_quantize_inputs.append((inp.inputs[0], node, inp.name))

        # Producer op types whose output feeds a covered activation-op input
        # (e.g., to support Add->Q/DQ->Relu patterns)
        op_types_needing_output_quant: set[str] = set()
        for node_idx, inp_idx in covered:
            node = graph.nodes[node_idx]
            if node.op in get_activation_ops():
                tensor = node.inputs[inp_idx]
                if tensor.inputs:
                    op_types_needing_output_quant.add(tensor.inputs[0].op)

        return (
            nodes_to_quantize,
            op_types_to_quantize,
            no_quantize_inputs,
            sorted(op_types_needing_output_quant),
        )

    @_requires_init
    def export_onnx(
        self,
        output_path: str | None = None,
        insert_qdq: bool = True,
        best: bool = False,
        model_transform: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
    ) -> bytes:
        """Export ONNX model with Q/DQ nodes inserted according to tested schemes.

        This method creates a modified version of the model by:
        1. For each region, finding the matching pattern
        2. Applying the best scheme for profiled patterns
        3. Applying the current scheme for the active profile pattern
        4. Resolving pattern-relative insertion points to actual tensor names
        5. Inserting Q/DQ pairs at the resolved locations
        6. Converting to FP8 if needed (always creates INT8 first, then converts)

        Args:
            output_path: Optional file path where the modified ONNX model will be saved.
                        If None, the model is not saved to disk and only bytes are returned.
            insert_qdq: If True, insert Q/DQ nodes. If False, export unmodified model
                       (useful for baseline measurements)

        Returns:
            bytes: Serialized ONNX model as bytes

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
        """
        output_desc = output_path if output_path is not None else "<bytes>"
        resolved_insertion_points = set()

        logger.debug(
            f"Exporting model to {output_desc} (insert_qdq={insert_qdq}, "
            f"regions={len(self.regions)}, profiled_patterns={len(self.profiled_patterns)})"
        )

        if insert_qdq:
            resolved_insertion_points = self.get_resolved_insertion_points(best=best, verbose=True)

        unique_tensors = len(resolved_insertion_points)

        logger.debug(f"Inserting {unique_tensors} Q/DQ pairs into graph")

        original_quant_type = self.config.default_quant_type
        needs_fp8_conversion = insert_qdq and original_quant_type == "fp8"

        model = export_qdq_onnx(
            self.onnx_model,
            resolved_insertion_points,
            self.config,
            insert_qdq=insert_qdq and bool(resolved_insertion_points),
            needs_fp8_conversion=needs_fp8_conversion,
            model_transform=model_transform,
        )

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

    @_requires_init
    def submit(self, latency_ms: float, success: bool = True) -> None:
        """Submit performance measurement for the most recently generated scheme.

        This method records the measured latency and manages the optimization state:

        Args:
            latency_ms: Measured latency in milliseconds (must be > 0)
            success: Whether the measurement succeeded. If False, sets scheme.error=True,
                    logs a warning, and skips speedup calculation.

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
            InvalidSchemeError: If no pattern or region is set, or no schemes have been generated
        """
        if self.baseline_latency_ms is None:
            if not success or not is_valid_latency(latency_ms):
                raise ValueError("Baseline latency must be finite and positive")
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
            if scheme_index < 0 or scheme_index >= len(pattern_schemes.schemes):
                raise InvalidSchemeError(f"Invalid scheme index: {scheme_index}")
            scheme = pattern_schemes.schemes[scheme_index]
        else:
            scheme = pattern_schemes.schemes[-1]
            scheme_index = len(pattern_schemes.schemes) - 1

        valid_measurement = success and is_valid_latency(latency_ms)
        scheme.latency_ms = latency_ms if is_valid_latency(latency_ms) else float("inf")
        scheme.error = not valid_measurement
        scheme.profile_timestamp = datetime.now(timezone.utc).isoformat()
        display_index = scheme_index + 1

        if not valid_measurement:
            logger.warning(f"Scheme #{display_index}: measurement failed")
            logger.debug("Marking scheme with error flag")
            pattern_schemes.select_best(self.config.performance_threshold)
            return

        speedup = self.baseline_latency_ms / latency_ms if latency_ms > 0 else 0.0

        logger.info(f"Scheme #{display_index}: {latency_ms:.3f} ms ({speedup:.2f}x speedup)")
        logger.debug(f"Compared to baseline: {self.baseline_latency_ms:.3f} ms")

        selected = pattern_schemes.select_best(self.config.performance_threshold)
        if selected is not None:
            logger.debug(f"Selected {selected.action.value} scheme at {selected.latency_ms:.3f} ms")

        if self.current_profile_pattern_schemes is not None and self.pattern_cache is not None:
            self.pattern_cache.add_pattern_schemes(pattern_schemes)
            logger.debug(
                f"Pattern cache updated: {self.pattern_cache.num_patterns} patterns, "
                f"{self.pattern_cache.total_schemes} schemes"
            )

    def save_state(self, output_path: str) -> None:
        """Atomically save complete autotuner state for later reuse."""
        current_pattern = self.current_profile_pattern_schemes

        state = {
            "state_version": 2,
            "model_sha256": self.model_sha256,
            "resume_fingerprint": self.resume_fingerprint,
            "baseline_latency_ms": self.baseline_latency_ms,
            "force_no_qdq": self.force_no_qdq,
            "proxy_decision": self.proxy_decision,
            "final_baseline_measurement": self.final_baseline_measurement,
            "final_decision": self.final_decision,
            "config": dataclasses.asdict(self.config),
            "patterns": [pattern_schemes.to_dict() for pattern_schemes in self.profiled_patterns],
            "current_pattern": current_pattern.to_dict() if current_pattern is not None else None,
            "current_region_id": (
                self.current_profile_region.id if self.current_profile_region is not None else None
            ),
            "current_insertion_scheme_index": self.current_insertion_scheme_index,
        }

        _atomic_yaml_dump(state, output_path)

        num_patterns = len(self.profiled_patterns)
        total_schemes = sum(len(p.schemes) for p in self.profiled_patterns)

        logger.info(
            f"Saved state → {output_path} ({num_patterns} patterns, {total_schemes} schemes)"
        )
        if self.baseline_latency_ms is not None:
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

    @_requires_init
    def load_state(self, input_path: str) -> None:
        """Load a checkpoint, rejecting model drift and remeasuring environment drift."""
        with open(input_path) as f:
            state = yaml.safe_load(f)
        if not isinstance(state, dict):
            raise ValueError("Autotuner state must contain a YAML mapping")

        self._load_pattern_cache_sidecar(input_path)

        state_version = state.get("state_version", 1)
        if state_version >= 2:
            saved_model_sha256 = state.get("model_sha256")
            if saved_model_sha256 != self.model_sha256:
                raise ValueError("Autotuner state model fingerprint does not match current model")

            saved_fingerprint = state.get("resume_fingerprint") or {}
            if self.resume_fingerprint and saved_fingerprint != self.resume_fingerprint:
                self._cache_state_qdq_candidates(state)
                self.baseline_latency_ms = None
                self.profiled_patterns.clear()
                self.current_profile_region = None
                self.current_profile_pattern_schemes = None
                self.current_insertion_scheme_index = None
                self.force_no_qdq = False
                self._clear_proxy_decision()
                self._clear_final_decision()
                logger.warning(
                    "Autotuner measurement environment changed; reusing Q/DQ candidates "
                    "but discarding saved latencies and selections"
                )
                return
            if not self.resume_fingerprint:
                self.resume_fingerprint = saved_fingerprint
        else:
            self._cache_state_qdq_candidates(state)
            self.baseline_latency_ms = None
            self.profiled_patterns.clear()
            self.current_profile_region = None
            self.current_profile_pattern_schemes = None
            self.current_insertion_scheme_index = None
            self.force_no_qdq = False
            self._clear_proxy_decision()
            self._clear_final_decision()
            logger.warning(
                "Legacy autotuner checkpoint loaded as unmeasured Q/DQ candidates; "
                "all performance measurements will be repeated"
            )
            return

        self.profiled_patterns.clear()
        self.current_profile_region = None
        self.current_profile_pattern_schemes = None
        self.current_insertion_scheme_index = None

        saved_baseline = state.get("baseline_latency_ms")
        if is_valid_latency(saved_baseline):
            self.baseline_latency_ms = saved_baseline
            logger.debug(f"Baseline latency: {self.baseline_latency_ms:.3f} ms")
        else:
            self.baseline_latency_ms = None

        num_loaded_schemes = 0
        for pattern_data in state.get("patterns", []):
            try:
                pattern_schemes = PatternSchemes.from_dict(pattern_data)
                if state_version >= 2:
                    pattern_schemes.ensure_control_schemes()
                    if "completed" not in pattern_data:
                        pattern_schemes.completed = True
                pattern_schemes.select_best(self.config.performance_threshold)
                if pattern_schemes.schemes:
                    self.profiled_patterns.append(pattern_schemes)
                    num_loaded_schemes += len(pattern_schemes.schemes)
            except (KeyError, TypeError, ValueError) as exc:  # noqa: PERF203
                logger.warning("Failed to load pattern: %s", exc)

        if state_version >= 2:
            self._restore_current_pattern(state)
            self.force_no_qdq = bool(state.get("force_no_qdq", False))
            self._restore_proxy_decision(state.get("proxy_decision"))
            self._restore_final_decision(
                state.get("final_baseline_measurement"), state.get("final_decision")
            )

        logger.info(
            f"Loaded state from {input_path} ({len(self.profiled_patterns)} patterns, "
            f"{num_loaded_schemes} schemes)"
        )

    def _load_pattern_cache_sidecar(self, input_path: str) -> None:
        base_path, ext = os.path.splitext(input_path)
        cache_path = f"{base_path}_pattern_cache{ext}"
        if not os.path.exists(cache_path):
            return
        try:
            loaded_cache = PatternCache.load(cache_path)
            if self.pattern_cache is None:
                self.pattern_cache = loaded_cache
            else:
                for pattern_schemes in loaded_cache.pattern_schemes:
                    self.pattern_cache.add_pattern_schemes(pattern_schemes)
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load pattern cache: %s", exc)

    def _cache_state_qdq_candidates(self, state: dict[str, Any]) -> None:
        if self.pattern_cache is None:
            return
        pattern_data_items = list(state.get("patterns", []))
        current_pattern = state.get("current_pattern")
        if isinstance(current_pattern, dict):
            pattern_data_items.append(current_pattern)
        for pattern_data in pattern_data_items:
            try:
                pattern_schemes = PatternSchemes.from_dict(pattern_data)
            except (KeyError, TypeError, ValueError):
                continue
            for scheme in pattern_schemes.schemes:
                if scheme.is_qdq:
                    scheme.latency_ms = float("inf")
                    scheme.error = False
                    scheme.profile_timestamp = None
            pattern_schemes.schemes = [
                scheme for scheme in pattern_schemes.schemes if scheme.is_qdq
            ]
            pattern_schemes.selected_scheme_hash = None
            self.pattern_cache.add_pattern_schemes(pattern_schemes)

    def _restore_current_pattern(self, state: dict[str, Any]) -> None:
        current_data = state.get("current_pattern")
        if not isinstance(current_data, dict):
            return
        pattern_schemes = PatternSchemes.from_dict(current_data)
        current_region_id = state.get("current_region_id")
        region = next(
            (candidate for candidate in self.regions if candidate.id == current_region_id), None
        )
        if (
            region is None
            or RegionPattern.from_region(region, self.graph) != pattern_schemes.pattern
        ):
            region = next(
                (
                    candidate
                    for candidate in self.regions
                    if RegionPattern.from_region(candidate, self.graph) == pattern_schemes.pattern
                ),
                None,
            )
        if region is None:
            logger.warning("Could not restore the current profile region from checkpoint")
            return

        pattern_schemes.pattern = RegionPattern.from_region(region, self.graph)
        pattern_schemes.ensure_control_schemes()
        pattern_schemes.select_best(self.config.performance_threshold)
        self.current_profile_region = region
        self.current_profile_pattern_schemes = pattern_schemes
        scheme_index = state.get("current_insertion_scheme_index")
        if isinstance(scheme_index, int) and 0 <= scheme_index < len(pattern_schemes.schemes):
            self.current_insertion_scheme_index = scheme_index

    def _restore_proxy_decision(self, proxy_decision: Any) -> None:
        self._clear_proxy_decision()
        if not isinstance(proxy_decision, dict):
            return
        try:
            self.record_proxy_decision(**proxy_decision)
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid proxy decision in checkpoint: %s", exc)

    def _restore_final_decision(self, final_baseline_measurement: Any, final_decision: Any) -> None:
        self._clear_final_decision()
        if isinstance(final_baseline_measurement, dict):
            try:
                self.record_final_baseline_measurement(**final_baseline_measurement)
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid calibrated baseline in checkpoint: %s", exc)

        if not isinstance(final_decision, dict) or final_decision.get("final_selection") is None:
            return
        if final_decision.get("decision_stage") in {None, "proxy_final"}:
            self._restore_legacy_proxy_decision(final_decision)
            return
        try:
            decision = dict(final_decision)
            baseline_latency = decision.pop("baseline_final_latency_ms", None)
            baseline_sha256 = decision.pop("baseline_model_sha256", None)
            if self.final_baseline_measurement is None:
                self.record_final_baseline_measurement(
                    decision_stage="calibrated_baseline",
                    baseline_final_latency_ms=baseline_latency,
                    baseline_model_sha256=baseline_sha256,
                )
            elif (baseline_latency is not None or baseline_sha256 is not None) and (
                baseline_latency != self.baseline_final_latency_ms
                or baseline_sha256 != self.baseline_model_sha256
            ):
                raise ValueError("calibrated baseline evidence does not match final decision")
            self.record_final_decision(**decision)
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid final decision in checkpoint: %s", exc)

    def _restore_legacy_proxy_decision(self, final_decision: dict[str, Any]) -> None:
        if self.proxy_decision is not None:
            return
        try:
            baseline_latency = self.baseline_latency_ms
            if baseline_latency is None or not is_valid_latency(baseline_latency):
                raise ValueError("legacy proxy decision requires a valid baseline latency")
            self.record_proxy_decision(
                proxy_selection=final_decision["final_selection"],
                baseline_latency_ms=baseline_latency,
                candidate_latency_ms=final_decision.get("candidate_final_latency_ms"),
                candidate_quantization_site_count=final_decision.get(
                    "candidate_quantization_site_count",
                    final_decision.get("candidate_qdq_count"),
                ),
                candidate_model_sha256=final_decision["candidate_model_sha256"],
                selected_model_sha256=final_decision["selected_model_sha256"],
            )
            logger.info("Loaded legacy final decision as proxy telemetry")
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid legacy proxy decision in checkpoint: %s", exc)

    @_requires_init
    def import_insertion_points(self, quantized_tensors: set[str] | list[str]) -> None:
        """Import Q/DQ insertion points from a list of quantized tensors and update pattern cache.

        Analyzes the current model's regions against the provided quantized tensors
        to extract Q/DQ insertion patterns. For each region, creates a pattern cache
        entry that captures which insertion points correspond to the quantized tensors.
        These cached patterns can then be used as seeds for future autotuning sessions.

        Args:
            quantized_tensors: Set or list of tensor names that are quantized
                              (i.e., tensors that have Q/DQ nodes applied to them)

        Raises:
            AutotunerNotInitializedError: If initialize() hasn't been called
        """
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

        schemes_with_time = [s for s in schemes if s.profile_timestamp is not None]

        if not schemes_with_time:
            return samples_before_best, time_to_best

        schemes_with_time.sort(key=lambda s: s.profile_timestamp or "")

        try:
            best_position = next(
                i for i, s in enumerate(schemes_with_time) if s.hash == best_scheme.hash
            )
            samples_before_best = best_position

            first_ts = schemes_with_time[0].profile_timestamp
            best_ts = best_scheme.profile_timestamp
            if first_ts is not None and best_ts is not None:
                first_timestamp = datetime.fromisoformat(first_ts)
                best_timestamp = datetime.fromisoformat(best_ts)
                time_to_best = (best_timestamp - first_timestamp).total_seconds()
        except (StopIteration, ValueError):
            pass

        return samples_before_best, time_to_best

    def _is_region_profiled(self, region: Region) -> bool:
        """Check if a region's pattern has already been fully profiled."""
        return any(
            p.pattern is not None and p.pattern.matches(region, self.graph) and p.completed
            for p in self.profiled_patterns
        )

    def _mutate_insertion_points(
        self,
        base_points,
        all_points,
        point_type: str,
        max_mutations: int,
        rng: random.Random,
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
        mutation_type = rng.choice(["add", "remove", "both"])

        if mutation_type in ["add", "both"] and len(current_points) < len(all_points):
            all_keys = {key_fn(p) for p in all_points}
            available_keys = all_keys - current_points
            if available_keys:
                max_add = min(max_mutations, len(available_keys))
                num_to_add = rng.randint(1, max_add)
                to_add = rng.sample(sorted(available_keys, key=repr), num_to_add)
                current_points.update(to_add)

        if mutation_type in ["remove", "both"] and current_points:
            max_remove = min(max_mutations, len(current_points))
            num_to_remove = rng.randint(1, max_remove) if len(current_points) > 1 else 1
            num_to_remove = min(num_to_remove, len(current_points))
            to_remove = rng.sample(sorted(current_points, key=repr), num_to_remove)
            for p in to_remove:
                current_points.discard(p)

        logger.debug(
            f"Mutated {point_type}: {initial_count} → {len(current_points)} ({mutation_type})"
        )

        return [p for p in all_points if key_fn(p) in current_points]

    def _generate_next_insertion_sample(self, rng: random.Random) -> InsertionScheme:
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

        **Seed Case:**
        If no valid measured override exists yet, returns the full Q/DQ scheme.

        Returns:
            New InsertionScheme with mutated insertion points.
            Returns empty scheme if no region is set or no candidates exist.
        """
        if self.current_profile_region is None:
            return InsertionScheme(action=SchemeAction.QDQ)

        if self.current_profile_pattern_schemes is not None:
            schemes_collection = self.current_profile_pattern_schemes
        else:
            return InsertionScheme(action=SchemeAction.QDQ)

        region = self.current_profile_region
        pattern_schemes = schemes_collection

        if not isinstance(schemes_collection, PatternSchemes) or schemes_collection.pattern is None:
            return InsertionScheme(action=SchemeAction.QDQ)
        pattern = schemes_collection.pattern
        full_insertion_scheme = pattern.get_full_insertion_scheme(region, self.graph)

        logger.debug(
            f"Available insertion points: {len(full_insertion_scheme.node_inputs)} node input, "
            f"{len(full_insertion_scheme.child_region_inputs)} region composite, "
            f"{len(full_insertion_scheme.region_outputs)} region output"
        )

        top_percent = self.config.top_percent_to_mutate
        minimum_schemes = self.config.minimum_schemes_to_mutate

        measured_schemes = [
            scheme
            for scheme in pattern_schemes.schemes
            if not scheme.is_inherit and not scheme.error and is_valid_latency(scheme.latency_ms)
        ]
        measured_schemes.sort(key=lambda s: s.latency_ms)

        num_top_schemes = max(
            int(len(measured_schemes) * top_percent), min(minimum_schemes, len(measured_schemes))
        )
        top_schemes = measured_schemes[:num_top_schemes]

        if len(top_schemes) == 0:
            logger.debug("No valid measured override, seeding from the full Q/DQ scheme")
            full_insertion_scheme.action = SchemeAction.QDQ
            return full_insertion_scheme

        base_scheme = rng.choice(top_schemes)
        total_base_points = (
            len(base_scheme.node_inputs)
            + len(base_scheme.child_region_inputs)
            + len(base_scheme.region_outputs)
        )
        logger.debug(
            f"Mutating from top {len(top_schemes)} schemes: "
            f"selected base with {total_base_points} points (latency={base_scheme.latency_ms:.3f} ms)"
        )

        max_mutations = self.config.maximum_mutations
        scheme = InsertionScheme(action=SchemeAction.QDQ)

        for attr, point_type, key_fn in _MUTATION_SPECS:
            base_points = {key_fn(p) for p in getattr(base_scheme, attr)}
            setattr(
                scheme,
                attr,
                self._mutate_insertion_points(
                    base_points,
                    getattr(full_insertion_scheme, attr),
                    point_type,
                    max_mutations,
                    rng,
                ),
            )

        return scheme

    def _copy_graph(self) -> gs.Graph:
        """Create an independent copy of the computation graph."""
        new_graph = gs.import_onnx(self.onnx_model)
        new_graph.toposort()
        return new_graph
