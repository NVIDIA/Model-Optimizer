#!/usr/bin/env python3
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

"""
Tests for PatternCache functionality in the autotuner.

Tests pattern cache creation, serialization, and scheme management.
"""

import os
import sys
import tempfile
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelopt.onnx.quantization.autotune.common import (
    InsertionScheme,
    NodeInputInsertionPoint,
    PatternCache,
    PatternSchemes,
)
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern


class TestPatternCache(unittest.TestCase):
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
        print("✓ Empty PatternCache creation")

    def test_add_pattern_schemes(self):
        """Test adding pattern schemes to cache."""
        cache = PatternCache()

        # Create a pattern scheme
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme = InsertionScheme()
        scheme.latency_ms = 10.0
        ps.schemes.append(scheme)

        cache.add_pattern_schemes(ps)

        assert len(cache.pattern_schemes) == 1
        assert cache.pattern_schemes[0].pattern_signature == "Conv->Relu"
        print("✓ PatternCache add pattern schemes")

    def test_multiple_patterns(self):
        """Test cache with multiple pattern schemes."""
        cache = PatternCache()

        # Add multiple patterns
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
        print("✓ PatternCache multiple patterns")

    def test_serialization_empty(self):
        """Test serialization of empty cache."""
        cache = PatternCache()

        data = cache.to_dict()
        assert "pattern_schemes" in data
        assert len(data["pattern_schemes"]) == 0

        restored = PatternCache.from_dict(data)
        assert len(restored.pattern_schemes) == 0
        print("✓ Empty PatternCache serialization")

    def test_serialization_with_data(self):
        """Test serialization with pattern schemes."""
        # Create cache with minimum_distance=0 to keep both schemes
        cache = PatternCache(minimum_distance=0)

        # Add a pattern scheme
        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)

        # Create schemes that are sufficiently different (distance >= 4)
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
            NodeInputInsertionPoint(4, 0),  # 5 total points, diff = 4 from scheme1
        ]
        scheme2.latency_ms = 12.0
        ps.schemes.append(scheme2)

        cache.add_pattern_schemes(ps)

        # Serialize and restore
        data = cache.to_dict()
        restored = PatternCache.from_dict(data)

        assert len(restored.pattern_schemes) == 1

        restored_ps = restored.pattern_schemes[0]
        assert restored_ps.pattern_signature == "Conv->Relu"
        assert len(restored_ps.schemes) == 2
        assert restored_ps.best_scheme_index == 0
        assert restored_ps.schemes[0].latency_ms == 10.0
        print("✓ PatternCache serialization with data")

    def test_yaml_round_trip(self):
        """Test saving and loading cache as YAML."""
        cache = PatternCache()

        # Add a pattern scheme
        pattern = self._create_test_pattern("Gemm->Relu")
        ps = PatternSchemes(pattern=pattern)
        scheme = InsertionScheme()
        scheme.latency_ms = 15.0
        ps.schemes.append(scheme)
        cache.add_pattern_schemes(ps)

        # Save to YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            cache.save(yaml_path)

            # Load from YAML
            restored = PatternCache.load(yaml_path)

            assert len(restored.pattern_schemes) == 1
            assert restored.pattern_schemes[0].pattern_signature == "Gemm->Relu"
            assert restored.pattern_schemes[0].schemes[0].latency_ms == 15.0
            print("✓ PatternCache YAML round trip")
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_update_cache(self):
        """Test updating existing pattern in cache (merges schemes)."""
        # Use minimum_distance=0 to keep all schemes
        cache = PatternCache(minimum_distance=0)

        # Add initial pattern
        pattern1 = self._create_test_pattern("Conv->Relu")
        ps1 = PatternSchemes(pattern=pattern1)
        scheme1 = InsertionScheme()
        scheme1.latency_ms = 10.0
        ps1.schemes.append(scheme1)
        cache.add_pattern_schemes(ps1)

        # Update with new scheme for same pattern
        pattern2 = self._create_test_pattern("Conv->Relu")
        ps2 = PatternSchemes(pattern=pattern2)
        scheme2 = InsertionScheme()
        scheme2.latency_ms = 8.0  # Better performance
        scheme2.node_inputs = [NodeInputInsertionPoint(0, 0)]  # Make it different
        ps2.schemes.append(scheme2)
        cache.add_pattern_schemes(ps2)

        # Verify merge (should have both schemes now)
        assert len(cache.pattern_schemes) == 1
        conv_relu_ps = cache.pattern_schemes[0]
        assert conv_relu_ps.pattern_signature == "Conv->Relu"
        assert len(conv_relu_ps.schemes) == 2  # Merged
        # Best scheme should be the one with lowest latency
        assert conv_relu_ps.best_scheme.latency_ms == 8.0
        print("✓ PatternCache update")

    def test_get_best_scheme(self):
        """Test retrieving best scheme for a pattern."""
        # Use minimum_distance=0 to keep all different schemes
        cache = PatternCache(minimum_distance=0)

        pattern = self._create_test_pattern("Conv->Relu")
        ps = PatternSchemes(pattern=pattern)

        # Add multiple schemes with different insertion points
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0)]
        scheme1.latency_ms = 12.0
        ps.schemes.append(scheme1)

        scheme2 = InsertionScheme()
        scheme2.node_inputs = [NodeInputInsertionPoint(1, 0)]  # Different node
        scheme2.latency_ms = 8.0
        ps.schemes.append(scheme2)

        scheme3 = InsertionScheme()
        scheme3.node_inputs = [NodeInputInsertionPoint(2, 0)]  # Different node
        scheme3.latency_ms = 10.0
        ps.schemes.append(scheme3)

        cache.add_pattern_schemes(ps)

        # Verify best scheme retrieval (automatically computed)
        conv_relu_ps = cache.pattern_schemes[0]
        assert conv_relu_ps.pattern_signature == "Conv->Relu"
        assert len(conv_relu_ps.schemes) == 3  # All 3 kept

        # Verify best scheme has lowest latency (cache may reorder schemes)
        best = conv_relu_ps.best_scheme
        assert best is not None
        assert best.latency_ms == 8.0

        # Verify all three latencies are present
        latencies = sorted([s.latency_ms for s in conv_relu_ps.schemes])
        assert latencies == [8.0, 10.0, 12.0]
        print("✓ PatternCache get best scheme")


def run_tests():
    """Run all PatternCache tests."""
    print("=" * 70)
    print("PatternCache Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPatternCache))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All PatternCache tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
