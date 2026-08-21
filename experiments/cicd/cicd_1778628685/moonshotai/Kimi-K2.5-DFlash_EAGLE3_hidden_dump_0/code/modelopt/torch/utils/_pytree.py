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

"""Utility functions for working with nested python data structure.

This utility is inspired by the PyTree utility from PyTorch:
https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py

From the PyTree documentation:

    A *pytree* is Python nested data structure. It is a tree in the sense that
    nodes are Python collections (e.g., list, tuple, dict) and the leaves are
    Python values. Furthermore, a pytree should not contain reference cycles.

    pytrees are useful for working with nested collections of Tensors. For example,
    one can use `tree_map` to map a function over all Tensors inside some nested
    collection of Tensors and `tree_unflatten` to get a flat list of all Tensors
    inside some nested collection. pytrees are helpful for implementing nested
    collection support for PyTorch APIs.

We use the same terminology for our pytrees but implement a simpler version. Specifically, our tree
spec is simply the original data structure with the values eliminated to None instead of storing
a class-based, nested tree spec object.
"""

from collections import deque
from typing import Any


class TreeSpec:
    """A simple class to hold a tree spec."""

    def __init__(self, pytree: Any, names: list[str]):
        self.spec = self._fill_spec(None, pytree)
        self.names = names

    @staticmethod
    def _fill_spec(values: list | tuple | Any, spec: Any) -> Any:
        """Fill the pytree spec with values."""
        # put fill_values in a deque container to allow for popping it while traversing data
        values = deque(values) if isinstance(values, (list, tuple)) else values

        def fill(spec):
            """Eliminate values from output structure from keep structure."""
            if isinstance(spec, (tuple, list)):
                return type(spec)([fill(val) for val in spec])
            elif isinstance(spec, dict):
                _check_serializable_keys(spec)
                return {k: fill(val) for k, val in spec.items()}
            return values.popleft() if isinstance(values, deque) else values

        # eliminate values from output structure
        data_structure = fill(spec)

        # return output structure
        return data_structure

    def generate_pytree(self, values: list | tuple | Any) -> Any:
        """Fill the pytree spec with values (non-static version)."""
        return self._fill_spec(values, self.spec)

    def __eq__(self, other: Any) -> bool:
        """Compare two tree specs."""
        return self.spec == other.spec and self.names == other.names

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


def _check_serializable_keys(data: dict):
    """Check if all keys in the data structure are serializable."""
    allowed_key_types = (float, int, str, type(None), bool)
    assert all(isinstance(k, allowed_key_types) for k in data), "Keys must be serializable!"


def unflatten_tree(values: list | tuple | Any, tree_spec: TreeSpec) -> Any:
    """Return a pytree according to the tree_spec and values filled according to the fillers.

    Args:
        values: A list/tuple of values or a single value to fill the pytree with.

            * If ``values`` are a list/tuple, then the values in the data structure will be filled
              in a sequential fashion while traversing the data structure in a DFS manner.

            * Otherwise, ``values`` will be used for every value field in the pytree.

        tree_spec: A pytree spec describing the pytree.

    Returns:
        A python object structured according to the tree_spec filled with the provided values.
    """
    return tree_spec.generate_pytree(values)


def flatten_tree(pytree: Any, prefix: str = "") -> tuple[list[Any], TreeSpec]:
    """Flatten given pytree with depth-first search.

    Args:
        pytree: Data structure to flatten.
        prefix: Prefix for the flattened keys. Defaults to "".

    Returns: A tuple (values, pytree) where
        values is a list of values flattened from the provided pytree, and
        tree_spec is the pytree spec describing the structure of the pytree.
    """

    def collect_spec(pytree, prefix):
        if isinstance(pytree, dict):
            _check_serializable_keys(pytree)
            for key, value in pytree.items():
                yield from collect_spec(value, prefix + ("." if prefix else "") + str(key))
        elif isinstance(pytree, (tuple, list)):
            for i, value in enumerate(pytree):
                yield from collect_spec(value, prefix + ("." if prefix else "") + str(i))
        else:
            yield prefix, pytree

    # retrieve flattened values and names. Then initialize tree_spec with the flattened names.
    flattened = dict(collect_spec(pytree, prefix))

    return list(flattened.values()), TreeSpec(pytree, list(flattened.keys()))
