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
Comprehensive tests for common data structures in the autotuner.

Tests:
1. InsertionPoint classes (NodeInputInsertionPoint, RegionOutputInsertionPoint, ChildRegionInputInsertionPoint)
2. InsertionScheme serialization/deserialization
3. InsertionScheme hashing and equality
4. InsertionScheme properties and methods
5. PatternSchemes management
"""

import os
import sys
import unittest

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelopt.onnx.quantization.autotune.common import (
    ChildRegionInputInsertionPoint,
    InsertionScheme,
    NodeInputInsertionPoint,
    RegionOutputInsertionPoint,
)


class TestNodeInputInsertionPoint(unittest.TestCase):
    """Test NodeInputInsertionPoint functionality."""

    def test_creation(self):
        """Test creating NodeInputInsertionPoint."""
        point = NodeInputInsertionPoint(node_index=5, input_index=2)
        assert point.node_index == 5
        assert point.input_index == 2
        print("✓ NodeInputInsertionPoint creation")

    def test_immutability(self):
        """Test that NodeInputInsertionPoint is immutable (frozen)."""
        point = NodeInputInsertionPoint(node_index=1, input_index=0)
        with pytest.raises(AttributeError):
            point.node_index = 2
        print("✓ NodeInputInsertionPoint is immutable")

    def test_equality(self):
        """Test equality comparison."""
        point1 = NodeInputInsertionPoint(node_index=3, input_index=1)
        point2 = NodeInputInsertionPoint(node_index=3, input_index=1)
        point3 = NodeInputInsertionPoint(node_index=3, input_index=2)

        assert point1 == point2
        assert point1 != point3
        print("✓ NodeInputInsertionPoint equality")

    def test_hashable(self):
        """Test that points can be used in sets and dicts."""
        point1 = NodeInputInsertionPoint(node_index=1, input_index=0)
        point2 = NodeInputInsertionPoint(node_index=1, input_index=0)
        point3 = NodeInputInsertionPoint(node_index=2, input_index=0)

        point_set = {point1, point2, point3}
        assert len(point_set) == 2  # point1 and point2 are the same
        print("✓ NodeInputInsertionPoint is hashable")

    def test_serialization(self):
        """Test to_dict and from_dict."""
        point = NodeInputInsertionPoint(node_index=7, input_index=3)

        data = point.to_dict()
        assert data["node_index"] == 7
        assert data["input_index"] == 3

        restored = NodeInputInsertionPoint.from_dict(data)
        assert point == restored
        print("✓ NodeInputInsertionPoint serialization")

    def test_string_representation(self):
        """Test __str__ method."""
        point = NodeInputInsertionPoint(node_index=2, input_index=1)
        s = str(point)
        assert "2" in s
        assert "1" in s
        print("✓ NodeInputInsertionPoint string representation")


class TestRegionOutputInsertionPoint(unittest.TestCase):
    """Test RegionOutputInsertionPoint functionality."""

    def test_creation_with_region_index(self):
        """Test creating with region_index (child region output)."""
        point = RegionOutputInsertionPoint(region_index=2, node_index=None, output_index=1)
        assert point.region_index == 2
        assert point.node_index is None
        assert point.output_index == 1
        print("✓ RegionOutputInsertionPoint with region_index")

    def test_creation_with_node_index(self):
        """Test creating with node_index (node output)."""
        point = RegionOutputInsertionPoint(region_index=None, node_index=5, output_index=0)
        assert point.region_index is None
        assert point.node_index == 5
        assert point.output_index == 0
        print("✓ RegionOutputInsertionPoint with node_index")

    def test_equality(self):
        """Test equality comparison."""
        point1 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point2 = RegionOutputInsertionPoint(region_index=1, node_index=None, output_index=0)
        point3 = RegionOutputInsertionPoint(region_index=None, node_index=1, output_index=0)

        assert point1 == point2
        assert point1 != point3
        print("✓ RegionOutputInsertionPoint equality")

    def test_serialization_region_index(self):
        """Test serialization with region_index."""
        point = RegionOutputInsertionPoint(region_index=3, node_index=None, output_index=2)

        data = point.to_dict()
        assert data["region_index"] == 3
        assert data["node_index"] is None
        assert data["output_index"] == 2

        restored = RegionOutputInsertionPoint.from_dict(data)
        assert point == restored
        print("✓ RegionOutputInsertionPoint serialization (region_index)")

    def test_serialization_node_index(self):
        """Test serialization with node_index."""
        point = RegionOutputInsertionPoint(region_index=None, node_index=7, output_index=1)

        data = point.to_dict()
        assert data["region_index"] is None
        assert data["node_index"] == 7
        assert data["output_index"] == 1

        restored = RegionOutputInsertionPoint.from_dict(data)
        assert point == restored
        print("✓ RegionOutputInsertionPoint serialization (node_index)")

    def test_string_representation(self):
        """Test __str__ method."""
        point1 = RegionOutputInsertionPoint(region_index=2, node_index=None, output_index=1)
        s1 = str(point1)
        assert "region" in s1.lower()
        assert "2" in s1

        point2 = RegionOutputInsertionPoint(region_index=None, node_index=5, output_index=0)
        s2 = str(point2)
        assert "node" in s2.lower()
        assert "5" in s2
        print("✓ RegionOutputInsertionPoint string representation")


class TestChildRegionInputInsertionPoint(unittest.TestCase):
    """Test ChildRegionInputInsertionPoint functionality."""

    def test_creation(self):
        """Test creating ChildRegionInputInsertionPoint."""
        point = ChildRegionInputInsertionPoint(region_index=3, input_index=1)
        assert point.region_index == 3
        assert point.input_index == 1
        print("✓ ChildRegionInputInsertionPoint creation")

    def test_equality(self):
        """Test equality comparison."""
        point1 = ChildRegionInputInsertionPoint(region_index=2, input_index=0)
        point2 = ChildRegionInputInsertionPoint(region_index=2, input_index=0)
        point3 = ChildRegionInputInsertionPoint(region_index=2, input_index=1)

        assert point1 == point2
        assert point1 != point3
        print("✓ ChildRegionInputInsertionPoint equality")

    def test_serialization(self):
        """Test to_dict and from_dict."""
        point = ChildRegionInputInsertionPoint(region_index=5, input_index=2)

        data = point.to_dict()
        assert data["region_index"] == 5
        assert data["input_index"] == 2

        restored = ChildRegionInputInsertionPoint.from_dict(data)
        assert point == restored
        print("✓ ChildRegionInputInsertionPoint serialization")


class TestInsertionScheme(unittest.TestCase):
    """Test InsertionScheme functionality."""

    def test_empty_scheme(self):
        """Test empty InsertionScheme."""
        scheme = InsertionScheme()

        assert scheme.is_empty
        assert scheme.num_node_insertions == 0
        assert scheme.num_region_insertions == 0
        assert scheme.num_region_output_insertions == 0
        assert not scheme.error
        print("✓ Empty InsertionScheme")

    def test_scheme_with_node_inputs(self):
        """Test scheme with node input insertion points."""
        scheme = InsertionScheme()
        scheme.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        assert not scheme.is_empty
        assert scheme.num_node_insertions == 2
        print("✓ InsertionScheme with node inputs")

    def test_scheme_with_region_outputs(self):
        """Test scheme with region output insertion points."""
        scheme = InsertionScheme()
        scheme.region_outputs = [
            RegionOutputInsertionPoint(None, 0, 0),
            RegionOutputInsertionPoint(1, None, 0),
        ]

        assert not scheme.is_empty
        assert scheme.num_region_output_insertions == 2
        print("✓ InsertionScheme with region outputs")

    def test_scheme_with_composite_regions(self):
        """Test scheme with composite region insertion points."""
        scheme = InsertionScheme()
        scheme.child_region_inputs = [
            ChildRegionInputInsertionPoint(0, 0),
            ChildRegionInputInsertionPoint(1, 0),
        ]

        assert not scheme.is_empty
        assert scheme.num_region_insertions == 2
        print("✓ InsertionScheme with composite regions")

    def test_scheme_hash_empty(self):
        """Test hash of empty scheme."""
        scheme1 = InsertionScheme()
        scheme2 = InsertionScheme()

        assert scheme1.hash == scheme2.hash
        print("✓ Empty scheme hash consistency")

    def test_scheme_hash_with_points(self):
        """Test hash with insertion points."""
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme2 = InsertionScheme()
        scheme2.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme3 = InsertionScheme()
        scheme3.node_inputs = [
            NodeInputInsertionPoint(0, 0),
            NodeInputInsertionPoint(2, 0),  # Different
        ]

        assert scheme1.hash == scheme2.hash
        assert scheme1.hash != scheme3.hash
        print("✓ Scheme hash with points")

    def test_scheme_hash_order_independent(self):
        """Test that hash is independent of insertion point order."""
        scheme1 = InsertionScheme()
        scheme1.node_inputs = [NodeInputInsertionPoint(0, 0), NodeInputInsertionPoint(1, 0)]

        scheme2 = InsertionScheme()
        scheme2.node_inputs = [
            NodeInputInsertionPoint(1, 0),
            NodeInputInsertionPoint(0, 0),  # Reversed order
        ]

        # Hash should be the same regardless of order
        assert scheme1.hash == scheme2.hash
        print("✓ Scheme hash is order-independent")

    def test_serialization_empty(self):
        """Test serialization of empty scheme."""
        scheme = InsertionScheme()

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert restored.is_empty
        assert restored.latency_ms == float("inf")
        assert not restored.error
        print("✓ Empty scheme serialization")

    def test_serialization_full(self):
        """Test serialization with all types of insertion points."""
        scheme = InsertionScheme()
        scheme.node_inputs = [NodeInputInsertionPoint(0, 0)]
        scheme.child_region_inputs = [ChildRegionInputInsertionPoint(0, 0)]
        scheme.region_outputs = [RegionOutputInsertionPoint(None, 0, 0)]
        scheme.latency_ms = 12.5
        scheme.error = False

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert len(restored.node_inputs) == 1
        assert len(restored.child_region_inputs) == 1
        assert len(restored.region_outputs) == 1
        assert restored.latency_ms == 12.5
        assert not restored.error
        print("✓ Full scheme serialization")

    def test_serialization_with_error(self):
        """Test serialization with error flag."""
        scheme = InsertionScheme()
        scheme.error = True
        scheme.latency_ms = float("inf")

        data = scheme.to_dict()
        restored = InsertionScheme.from_dict(data)

        assert restored.error
        assert restored.latency_ms == float("inf")
        print("✓ Scheme serialization with error")


def run_tests():
    """Run all insertion point and scheme tests."""
    print("=" * 70)
    print("Autotuner Insertion Points & Schemes Test Suite")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNodeInputInsertionPoint))
    suite.addTests(loader.loadTestsFromTestCase(TestRegionOutputInsertionPoint))
    suite.addTests(loader.loadTestsFromTestCase(TestChildRegionInputInsertionPoint))
    suite.addTests(loader.loadTestsFromTestCase(TestInsertionScheme))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All insertion point and scheme tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
