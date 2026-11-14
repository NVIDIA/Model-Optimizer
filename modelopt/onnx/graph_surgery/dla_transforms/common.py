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

"""Small shared IR queries and typed initializer construction for DLA rules."""

from collections.abc import Callable
from typing import Any

import numpy as np
from onnxscript import ir

__all__ = []


def static_shape(value: ir.Value | None) -> tuple[int, ...] | None:
    """Return concrete dimensions, retaining the empty tuple for scalar tensors."""
    if value is None or value.shape is None:
        return None
    dims = tuple(value.shape)
    return dims if all(isinstance(dim, int) and dim >= 0 for dim in dims) else None


def constant_array(value: ir.Value | None) -> np.ndarray | None:
    """Read a fixed tensor without treating an overridable initializer as constant."""
    if value is None or value.const_value is None or value.is_graph_input():
        return None
    return np.asarray(value.const_value.numpy())


def initializer_factory(graph: ir.Graph) -> Callable[[Any, np.ndarray, str], ir.Value]:
    """Reserve names before rule insertion; callers own payload typing and reuse."""
    reserved = set(graph.initializers)
    reserved.update(value.name for value in graph.inputs)
    reserved.update(value.name for value in graph.outputs)
    reserved.update(value.name for node in graph for value in node.outputs)

    def add(builder: Any, array: np.ndarray, label: str) -> ir.Value:
        """Add one initializer with the requested NumPy dtype and a fresh name."""
        name, suffix = label, 0
        while name in reserved:
            suffix += 1
            name = f"{label}_{suffix}"
        reserved.add(name)
        return builder.initializer(ir.tensor(np.asarray(array)), name=name)

    return add
