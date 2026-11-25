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

import copy

import pytest

from modelopt.torch.utils._pytree import flatten_tree, unflatten_tree


@pytest.mark.parametrize(
    "data",
    [
        (5.0,),
        (5,),
        (True,),
        ([5],),
        ({"a": 5},),
        ((5,),),
        ((5, 100, True, False, "hello"),),
        ({"a": (1, True), "b": (3, False), "d": [3, 4, 5, (6, {"c": 5})]},),
    ],
)
def test_unflatten_tree(data):
    """A simple test to check if we can fill all kinds of nested data structures."""
    # get the values serialized
    data_serialized, tree_spec = copy.deepcopy(flatten_tree(data))

    # check that no values survived
    data_filled_flat, _ = flatten_tree(tree_spec.spec)
    assert data_filled_flat == [None] * len(data_filled_flat)

    # refill data structure and see if it is the same as before
    data_refilled = unflatten_tree(data_serialized, tree_spec)
    assert data_refilled == data


@pytest.mark.parametrize(
    ("data", "expected_keys"),
    [
        (5, [""]),
        ([5], ["0"]),
        ({"a": 5}, ["a"]),
        ((5,), ["0"]),
        ((5, 100, True, False, "hello"), ["0", "1", "2", "3", "4"]),
        (
            {"a": (1, True), "b": (3, False), "d": [3, 4, 5, (6, {"c": 5})]},
            ["a.0", "a.1", "b.0", "b.1", "d.0", "d.1", "d.2", "d.3.0", "d.3.1.c"],
        ),
    ],
)
def test_flatten_tree(data, expected_keys):
    """A simple test to check if we can flatten nested data structures as expected."""
    # flatten the data
    values, tree_spec = flatten_tree(data)

    # check that we get the expected keys
    assert expected_keys == tree_spec.names

    # try re-building to see if we get same structure
    tree_rebuilt = unflatten_tree(copy.deepcopy(values), tree_spec)
    assert data == tree_rebuilt
