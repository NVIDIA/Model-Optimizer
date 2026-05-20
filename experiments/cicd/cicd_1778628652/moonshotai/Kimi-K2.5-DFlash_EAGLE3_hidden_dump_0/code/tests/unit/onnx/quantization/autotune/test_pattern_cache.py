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

from modelopt.onnx.quantization.autotune.common import (
    InsertionScheme,
    NodeInputInsertionPoint,
    PatternCache,
    PatternSchemes,
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

    def test_add_pattern_schemes(self):
        """Test adding pattern schemes to cache."""
        cache = PatternCache()
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme = InsertionScheme()
        scheme.latency_ms = 10.0
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
            scheme = InsertionScheme()
            scheme.latency_ms = 10.0 + len(pattern_sig)
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
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0)]
        scheme1.latency_ms = 10.0
        ps.schemes.append(scheme1)
        scheme2 = InsertionScheme()
        scheme2.node_inputs = [
            NodeInputInsertionPoint(0, 0),
            NodeInputInsertionPoint(1, 0),
            NodeInputInsertionPoint(2, 0),
            NodeInputInsertionPoint(3, 0),
            NodeInputInsertionPoint(4, 0),
        ]
        scheme2.latency_ms = 12.0
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
        scheme = InsertionScheme()
        scheme.latency_ms = 15.0
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
        scheme1 = InsertionScheme()
        scheme1.latency_ms = 10.0
        ps1.schemes.append(scheme1)
        cache.add_pattern_schemes(ps1)
        pattern2 = self._create_test_pattern("Conv->Relu")
        ps2 = PatternSchemes(pattern=pattern2)
        scheme2 = InsertionScheme()
        scheme2.latency_ms = 8.0
        scheme2.node_inputs = [NodeInputInsertionPoint(0, 0)]
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
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0)]
        scheme1.latency_ms = 12.0
        ps.schemes.append(scheme1)
        scheme2 = InsertionScheme()
        scheme2.node_inputs = [NodeInputInsertionPoint(1, 0)]
        scheme2.latency_ms = 8.0
        ps.schemes.append(scheme2)
        scheme3 = InsertionScheme()
        scheme3.node_inputs = [NodeInputInsertionPoint(2, 0)]
        scheme3.latency_ms = 10.0
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
