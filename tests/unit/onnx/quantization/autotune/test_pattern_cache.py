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

"""
Tests for PatternCache in the autotuner.

Covers pattern cache creation, serialization, YAML round-trip, and scheme management.
"""

import os
import tempfile

import pytest

from modelopt.onnx.quantization.autotune.common import (
    ChildRegionInputInsertionPoint,
    ChildRegionOutputInsertionPoint,
    InsertionScheme,
    NodeInputInsertionPoint,
    PatternCache,
    PatternSchemes,
    SchemeAction,
)
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern


class TestPatternCache:
    """Test PatternCache functionality."""

    @staticmethod
    def _create_test_pattern(signature: str, size: int = 2):
        """Create a test RegionPattern."""
        return RegionPattern(signature=signature, size=size)

    def test_empty_cache_creation(self):
        """Test creating an empty PatternCache."""
        cache = PatternCache()
        assert len(cache.pattern_schemes) == 0
        assert cache.pattern_schemes is not None

    def test_insertion_scheme_preserves_positional_argument_order(self):
        node_inputs = [NodeInputInsertionPoint(1, 2)]
        child_region_inputs = [ChildRegionInputInsertionPoint(3, 4)]
        region_outputs = [ChildRegionOutputInsertionPoint(5, None, 6)]

        scheme = InsertionScheme(
            node_inputs,
            child_region_inputs,
            region_outputs,
            7.5,
            True,
            "timestamp",
            action=SchemeAction.QDQ,
        )

        assert scheme.node_inputs == node_inputs
        assert scheme.child_region_inputs == child_region_inputs
        assert scheme.region_outputs == region_outputs
        assert scheme.latency_ms == 7.5
        assert scheme.error is True
        assert scheme.profile_timestamp == "timestamp"
        assert scheme.action == SchemeAction.QDQ

    def test_add_pattern_schemes(self):
        """Test adding pattern schemes to cache."""
        cache = PatternCache()
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme = InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=10.0)
        ps.schemes.append(scheme)
        cache.add_pattern_schemes(ps)
        assert len(cache.pattern_schemes) == 1
        assert cache.pattern_schemes[0].pattern_signature == "Conv->Relu"

    def test_multiple_patterns(self):
        """Test cache with multiple pattern schemes."""
        cache = PatternCache()
        pattern_sigs = ["Conv->Relu", "Gemm->Relu", "Conv->Add->Relu"]
        for pattern_sig in pattern_sigs:
            pattern = self._create_test_pattern(pattern_sig)
            ps = PatternSchemes(pattern=pattern)
            scheme = InsertionScheme(
                node_inputs=[NodeInputInsertionPoint(0, 0)],
                latency_ms=10.0 + len(pattern_sig),
            )
            ps.schemes.append(scheme)
            cache.add_pattern_schemes(ps)
        assert len(cache.pattern_schemes) == 3
        found_patterns = [ps.pattern_signature for ps in cache.pattern_schemes]
        for pattern_sig in pattern_sigs:
            assert pattern_sig in found_patterns

    def test_serialization_empty(self):
        """Test serialization of empty cache."""
        cache = PatternCache()
        data = cache.to_dict()
        assert "pattern_schemes" in data
        assert len(data["pattern_schemes"]) == 0
        restored = PatternCache.from_dict(data)
        assert len(restored.pattern_schemes) == 0

    def test_serialization_with_data(self):
        """Test serialization with pattern schemes."""
        cache = PatternCache(minimum_distance=0)
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme1 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=10.0)
        ps.schemes.append(scheme1)
        scheme2 = InsertionScheme(
            node_inputs=[
                NodeInputInsertionPoint(0, 0),
                NodeInputInsertionPoint(1, 0),
                NodeInputInsertionPoint(2, 0),
                NodeInputInsertionPoint(3, 0),
                NodeInputInsertionPoint(4, 0),
            ],
            latency_ms=12.0,
        )
        ps.schemes.append(scheme2)
        cache.add_pattern_schemes(ps)
        data = cache.to_dict()
        restored = PatternCache.from_dict(data)
        assert len(restored.pattern_schemes) == 1
        restored_ps = restored.pattern_schemes[0]
        assert restored_ps.pattern_signature == "Conv->Relu"
        assert len(restored_ps.schemes) == 2
        assert restored_ps.best_scheme is not None
        assert restored_ps.best_scheme.latency_ms == 10.0
        assert restored_ps.schemes[0].latency_ms == 10.0

    def test_yaml_round_trip(self):
        """Test saving and loading cache as YAML."""
        cache = PatternCache()
        pattern = self._create_test_pattern("Gemm->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme = InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=15.0)
        ps.schemes.append(scheme)
        cache.add_pattern_schemes(ps)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name
        try:
            cache.save(yaml_path)
            restored = PatternCache.load(yaml_path)
            assert len(restored.pattern_schemes) == 1
            assert restored.pattern_schemes[0].pattern_signature == "Gemm->Relu"
            assert restored.pattern_schemes[0].schemes[0].latency_ms == 15.0
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_update_cache(self):
        """Test updating existing pattern in cache (merges schemes)."""
        cache = PatternCache(minimum_distance=0)
        pattern1 = self._create_test_pattern("Conv->Relu")
        ps1 = PatternSchemes(pattern=pattern1)
        scheme1 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(1, 0)], latency_ms=10.0)
        ps1.schemes.append(scheme1)
        cache.add_pattern_schemes(ps1)
        pattern2 = self._create_test_pattern("Conv->Relu")
        ps2 = PatternSchemes(pattern=pattern2)
        scheme2 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=8.0)
        ps2.schemes.append(scheme2)
        cache.add_pattern_schemes(ps2)
        assert len(cache.pattern_schemes) == 1
        conv_relu_ps = cache.pattern_schemes[0]
        assert conv_relu_ps.pattern_signature == "Conv->Relu"
        assert len(conv_relu_ps.schemes) == 2
        assert conv_relu_ps.best_scheme is not None
        assert conv_relu_ps.best_scheme.latency_ms == 8.0

    def test_get_best_scheme(self):
        """Test retrieving best scheme for a pattern."""
        cache = PatternCache(minimum_distance=0)
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme1 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=12.0)
        ps.schemes.append(scheme1)
        scheme2 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(1, 0)], latency_ms=8.0)
        ps.schemes.append(scheme2)
        scheme3 = InsertionScheme(node_inputs=[NodeInputInsertionPoint(2, 0)], latency_ms=10.0)
        ps.schemes.append(scheme3)
        cache.add_pattern_schemes(ps)
        conv_relu_ps = cache.pattern_schemes[0]
        assert conv_relu_ps.pattern_signature == "Conv->Relu"
        assert len(conv_relu_ps.schemes) == 3
        best = conv_relu_ps.best_scheme
        assert best is not None
        assert best.latency_ms == 8.0
        latencies = sorted([s.latency_ms for s in conv_relu_ps.schemes])
        assert latencies == [8.0, 10.0, 12.0]

    def test_cache_keeps_only_qdq_candidates(self):
        cache = PatternCache(minimum_distance=0)
        pattern = self._create_test_pattern("Conv")
        schemes = PatternSchemes(
            pattern=pattern,
            schemes=[
                InsertionScheme(action=SchemeAction.INHERIT, latency_ms=10.0),
                InsertionScheme(action=SchemeAction.NO_QDQ, latency_ms=9.0),
                InsertionScheme(node_inputs=[NodeInputInsertionPoint(0, 0)], latency_ms=8.0),
            ],
        )

        cache.add_pattern_schemes(schemes)

        cached = cache.get_pattern_schemes(pattern.signature)
        assert cached is not None
        assert len(cached.schemes) == 1
        assert cached.schemes[0].action == SchemeAction.QDQ

    def test_mutating_empty_scheme_normalizes_action(self):
        scheme = InsertionScheme()
        scheme.node_inputs.append(NodeInputInsertionPoint(0, 0))

        assert scheme.action == SchemeAction.QDQ

        scheme.action = SchemeAction.NO_QDQ
        restored = InsertionScheme.from_dict(scheme.to_dict())
        assert scheme.action == SchemeAction.QDQ
        assert restored.action == SchemeAction.QDQ

    def test_completed_pattern_round_trip(self):
        schemes = PatternSchemes(
            pattern=self._create_test_pattern("Conv"),
            completed=True,
            search_exhausted=True,
        )

        restored = PatternSchemes.from_dict(schemes.to_dict())

        assert restored.completed
        assert restored.search_exhausted

    @pytest.mark.parametrize(
        ("candidate_latency", "expected_action"),
        [(50.0, SchemeAction.QDQ), (50.1, SchemeAction.INHERIT)],
    )
    def test_selection_requires_threshold(self, candidate_latency, expected_action):
        schemes = PatternSchemes(
            schemes=[
                InsertionScheme(action=SchemeAction.INHERIT, latency_ms=51.0),
                InsertionScheme(action=SchemeAction.NO_QDQ, latency_ms=52.0),
                InsertionScheme(
                    node_inputs=[NodeInputInsertionPoint(0, 0)],
                    latency_ms=candidate_latency,
                ),
            ]
        )

        assert schemes.select_best(1.02).action == expected_action

    def test_invalid_inherit_conservatively_selects_inherit(self):
        inherit = InsertionScheme(action=SchemeAction.INHERIT, error=True)
        schemes = PatternSchemes(
            schemes=[
                inherit,
                InsertionScheme(action=SchemeAction.NO_QDQ, latency_ms=1.0),
            ]
        )

        assert schemes.select_best(1.02) is inherit
