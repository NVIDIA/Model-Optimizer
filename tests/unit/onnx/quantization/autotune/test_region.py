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
Tests for the Region class in the autotuner.

Tests region creation, hierarchy, and boundary management.
"""

import sys
import unittest

from modelopt.onnx.quantization.autotune.common import Region, RegionType


class TestRegion(unittest.TestCase):
    """Test Region class functionality."""

    def test_leaf_region_creation(self):
        """Test creating a LEAF region."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        assert region.get_id() == 1
        assert region.get_level() == 0
        assert region.get_type() == RegionType.LEAF
        assert region.get_parent() is None
        assert len(region.get_children()) == 0
        print("✓ LEAF region creation")

    def test_composite_region_creation(self):
        """Test creating a COMPOSITE region."""
        region = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)

        assert region.get_id() == 2
        assert region.get_level() == 1
        assert region.get_type() == RegionType.COMPOSITE
        print("✓ COMPOSITE region creation")

    def test_root_region_creation(self):
        """Test creating a ROOT region."""
        region = Region(region_id=0, level=2, region_type=RegionType.ROOT)

        assert region.get_id() == 0
        assert region.get_level() == 2
        assert region.get_type() == RegionType.ROOT
        print("✓ ROOT region creation")

    def test_parent_child_relationship(self):
        """Test parent-child relationships."""
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.get_children()) == 2
        assert child1.get_parent() == parent
        assert child2.get_parent() == parent
        assert child1 in parent.get_children()
        assert child2 in parent.get_children()
        print("✓ Parent-child relationships")

    def test_add_nodes(self):
        """Test adding nodes to a region."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        region.add_node(0)
        region.add_node(1)
        region.add_node(2)

        assert region.get_size() == 3
        assert 0 in region.get_nodes()
        assert 1 in region.get_nodes()
        assert 2 in region.get_nodes()
        print("✓ Add nodes to region")

    def test_input_output_tensors(self):
        """Test setting input and output tensors."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        # Directly assign to inputs/outputs attributes
        region.inputs = ["input_tensor_1", "input_tensor_2"]
        region.outputs = ["output_tensor_1"]

        assert len(region.get_inputs()) == 2
        assert len(region.get_outputs()) == 1
        assert "input_tensor_1" in region.get_inputs()
        assert "output_tensor_1" in region.get_outputs()
        print("✓ Input/output tensors")

    def test_region_size_recursive(self):
        """Test recursive size calculation."""
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)

        # Add nodes to children
        child1.add_node(0)
        child1.add_node(1)
        child2.add_node(2)
        child2.add_node(3)
        child2.add_node(4)

        # Add children to parent
        parent.add_child(child1)
        parent.add_child(child2)

        # Parent itself might have direct nodes
        parent.add_node(5)

        # Recursive count should include all nodes
        assert len(parent.get_all_nodes_recursive()) == 6
        print("✓ Recursive size calculation")

    def test_is_leaf(self):
        """Test checking if region is LEAF type."""
        leaf = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        composite = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)

        assert leaf.get_type() == RegionType.LEAF
        assert composite.get_type() != RegionType.LEAF
        print("✓ Region LEAF type check")

    def test_is_composite(self):
        """Test checking if region is COMPOSITE type."""
        leaf = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        composite = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)

        assert leaf.get_type() != RegionType.COMPOSITE
        assert composite.get_type() == RegionType.COMPOSITE
        print("✓ Region COMPOSITE type check")

    def test_hierarchical_structure(self):
        """Test complex hierarchical structure."""
        root = Region(region_id=0, level=2, region_type=RegionType.ROOT)
        composite1 = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        composite2 = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)
        leaf1 = Region(region_id=3, level=0, region_type=RegionType.LEAF)
        leaf2 = Region(region_id=4, level=0, region_type=RegionType.LEAF)
        leaf3 = Region(region_id=5, level=0, region_type=RegionType.LEAF)

        # Build hierarchy
        root.add_child(composite1)
        root.add_child(composite2)
        composite1.add_child(leaf1)
        composite1.add_child(leaf2)
        composite2.add_child(leaf3)

        # Add some nodes
        leaf1.add_node(0)
        leaf2.add_node(1)
        leaf3.add_node(2)

        # Verify structure
        assert len(root.get_children()) == 2
        assert len(composite1.get_children()) == 2
        assert len(composite2.get_children()) == 1
        assert len(root.get_all_nodes_recursive()) == 3
        print("✓ Complex hierarchical structure")

    def test_remove_child(self):
        """Test removing a child region."""
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)

        parent.add_child(child)
        assert len(parent.get_children()) == 1

        parent.remove_child(child)
        assert len(parent.get_children()) == 0
        assert child.get_parent() is None
        print("✓ Remove child region")


def run_tests():
    """Run all Region tests."""
    print("=" * 70)
    print("Region Class Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRegion))

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
        print("\n✓ All Region tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
