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

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Region class in the autotuner."""

import pytest

from modelopt.onnx.quantization.autotune.common import Region, RegionType


@pytest.fixture
def leaf():
    return Region(region_id=1, level=0, region_type=RegionType.LEAF)


@pytest.fixture
def parent_with_children():
    parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
    child1 = Region(region_id=2, level=0, region_type=RegionType.LEAF)
    child2 = Region(region_id=3, level=0, region_type=RegionType.LEAF)
    parent.add_child(child1)
    parent.add_child(child2)
    return parent, child1, child2


@pytest.mark.parametrize(
    ("region_id", "level", "region_type"),
    [
        (1, 0, RegionType.LEAF),
        (2, 1, RegionType.COMPOSITE),
        (0, 2, RegionType.ROOT),
    ],
)
def test_region_creation(region_id, level, region_type):
    region = Region(region_id=region_id, level=level, region_type=region_type)
    assert (region.id, region.level, region.type) == (region_id, level, region_type)


def test_parent_child_relationship(parent_with_children):
    parent, child1, child2 = parent_with_children
    assert parent.get_children() == [child1, child2]
    assert child1.parent == child2.parent == parent


def test_add_and_get_nodes(leaf):
    leaf.nodes.update([0, 1, 2])
    assert set(leaf.get_nodes()) == {0, 1, 2}


def test_input_output_tensors(leaf):
    leaf.inputs = ["in1", "in2"]
    leaf.outputs = ["out1"]
    assert leaf.inputs == ["in1", "in2"]
    assert leaf.outputs == ["out1"]


def test_region_size_recursive(parent_with_children):
    parent, child1, child2 = parent_with_children
    child1.nodes.update([0, 1])
    child2.nodes.update([2, 3, 4])
    parent.nodes.add(5)
    assert len(parent.get_region_nodes_and_descendants()) == 6


def test_metadata(leaf):
    leaf.metadata.update({"pattern": "Conv->Relu", "quantizable": "true"})
    assert leaf.metadata == {"pattern": "Conv->Relu", "quantizable": "true"}


def test_hierarchical_structure():
    root = Region(region_id=0, level=2, region_type=RegionType.ROOT)
    comp1 = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
    comp2 = Region(region_id=2, level=1, region_type=RegionType.COMPOSITE)
    leaves = [Region(region_id=i, level=0, region_type=RegionType.LEAF) for i in range(3, 6)]
    root.add_child(comp1)
    root.add_child(comp2)
    comp1.add_child(leaves[0])
    comp1.add_child(leaves[1])
    comp2.add_child(leaves[2])
    for i, leaf in enumerate(leaves):
        leaf.nodes.add(i)
    assert len(root.get_children()) == 2
    assert len(comp1.get_children()) == 2
    assert len(comp2.get_children()) == 1
    assert len(root.get_region_nodes_and_descendants()) == 3


def test_remove_child():
    parent = Region(region_id=1, level=1, region_type=RegionType.COMPOSITE)
    child = Region(region_id=2, level=0, region_type=RegionType.LEAF)
    parent.add_child(child)
    parent.remove_child(child)
    assert parent.get_children() == []
    assert child.parent is None
