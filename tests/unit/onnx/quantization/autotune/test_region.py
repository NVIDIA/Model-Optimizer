#!/usr/bin/env python3
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
Tests for the Region class in the autotuner.

Tests region creation, hierarchy, and boundary management.
"""

import unittest

from modelopt.onnx.quantization.autotune.common import Region, RegionType


class TestRegion(unittest.TestCase):
    """Test Region class functionality."""

    def test_region_creation(self):
        """Test creating regions of all types."""
        test_cases = [
            {"region_id": 1, "level": 0, "region_type": RegionType.LEAF},
            {"region_id": 2, "level": 1, "region_type": RegionType.COMPOSITE},
            {"region_id": 0, "level": 2, "region_type": RegionType.ROOT},
        ]

        for params in test_cases:
            with self.subTest(**params):
                region = Region(**params)
                assert region.id == params["region_id"]
                assert region.level == params["level"]
                assert region.type == params["region_type"]

    def test_parent_child_relationship(self):
        """Test parent-child relationships."""
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
        child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.get_children()) == 2
        assert child1.parent == parent
        assert child2.parent == parent
        assert child1 in parent.get_children()
        assert child2 in parent.get_children()

    def test_add_nodes(self):
        """Test adding nodes to a region."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        region.add_node(0)
        region.add_node(1)
        region.add_node(2)

        assert len(region.nodes) == 3
        assert 0 in region.get_nodes()
        assert 1 in region.get_nodes()
        assert 2 in region.get_nodes()

    def test_input_output_tensors(self):
        """Test setting input and output tensors."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        # Directly assign to inputs/outputs attributes
        region.inputs = ["input_tensor_1", "input_tensor_2"]
        region.outputs = ["output_tensor_1"]

        assert len(region.inputs) == 2
        assert len(region.outputs) == 1
        assert "input_tensor_1" in region.inputs
        assert "output_tensor_1" in region.outputs

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
        assert len(parent.get_region_nodes_and_descendants()) == 6

    def test_metadata(self):
        """Test region metadata storage."""
        region = Region(region_id=1, level=0, region_type=RegionType.LEAF)

        region.metadata["pattern"] = "Conv->Relu"
        region.metadata["quantizable"] = "true"

        assert region.metadata["pattern"] == "Conv->Relu"
        assert region.metadata["quantizable"] == "true"

    def test_region_type_checks(self):
        """Test checking region types (LEAF and COMPOSITE)."""
        leaf = Region(region_id=1, level=0, region_type=RegionType.LEAF)
        composite = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)

        assert leaf.type == RegionType.LEAF
        assert leaf.type != RegionType.COMPOSITE
        assert composite.type == RegionType.COMPOSITE
        assert composite.type != RegionType.LEAF

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
        assert len(root.get_region_nodes_and_descendants()) == 3

    def test_remove_child(self):
        """Test removing a child region."""
        parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
        child = Region(region_id=2, level=0, region_type=RegionType.LEAF)

        parent.add_child(child)
        assert len(parent.get_children()) == 1

        parent.remove_child(child)
        assert len(parent.get_children()) == 0
        assert child.parent is None
