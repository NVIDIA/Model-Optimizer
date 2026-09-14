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

"""Shared helpers for DLA graph transforms."""

from __future__ import annotations

import gc
import inspect
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import onnx
from onnx import numpy_helper

from ...logging_config import logger


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run ORT symbolic shape inference and update ``value_info`` in-place.

    ORT's ``SymbolicShapeInference.infer_shapes`` creates a full deep copy of
    the model (including all weight tensors).  To avoid holding two large copies
    in memory simultaneously, this function:

    1. Runs ORT inference to get the inferred model.
    2. Copies **only** the shape metadata (``value_info`` and graph output types)
       back into the original model object.
    3. Explicitly deletes the ORT copy and calls ``gc.collect()`` so Python
       reclaims the memory immediately rather than waiting for the next GC cycle.

    The caller's ``model`` object is mutated in-place and also returned, so the
    existing call pattern ``model = infer_shapes(model)`` continues to work.

    Raises ``RuntimeError`` if ORT inference fails.
    """
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        inferred = SymbolicShapeInference.infer_shapes(model)

        # Copy back only shape metadata — weights stay in the original object.
        del model.graph.value_info[:]
        model.graph.value_info.extend(inferred.graph.value_info)

        inferred_out = {o.name: o.type for o in inferred.graph.output}
        for o in model.graph.output:
            if o.name in inferred_out:
                o.type.CopyFrom(inferred_out[o.name])

        del inferred  # release the ORT deep copy immediately
        gc.collect()  # force CPython to reclaim the freed memory now
        return model
    except Exception as exc:
        raise RuntimeError(f"ORT symbolic shape inference failed: {exc}") from exc


def resolve_reshape_shape(
    reshape_shape: np.ndarray,
    input_shape: list[int],
) -> list[int]:
    """Resolve a Reshape target shape containing ``0`` and ``-1`` into concrete dims.

    Rules (same as ONNX Reshape semantics):
    * ``0`` — copy the corresponding dim from ``input_shape``.
    * ``-1`` — infer this dim so that the total element count is preserved.
    * All other values are used as-is.

    Args:
        reshape_shape: 1-D int64 array of the Reshape shape initializer.
        input_shape:   Concrete input tensor shape (list of ints, no zeros).

    Returns:
        List of resolved concrete dimensions.

    Raises:
        ValueError: if more than one ``-1`` is present or element counts are inconsistent.
    """
    shape = [int(d) for d in reshape_shape.tolist()]
    total_elems = int(np.prod(input_shape))

    minus_one_indices = [i for i, d in enumerate(shape) if d == -1]
    if len(minus_one_indices) > 1:
        raise ValueError(f"resolve_reshape_shape: more than one -1 in shape {shape}")

    resolved = []
    for i, d in enumerate(shape):
        if d == 0:
            if i >= len(input_shape):
                raise ValueError(
                    f"resolve_reshape_shape: 0 at index {i} but input rank is {len(input_shape)}"
                )
            resolved.append(input_shape[i])
        elif d == -1:
            resolved.append(-1)  # placeholder
        else:
            resolved.append(d)

    if minus_one_indices:
        known_prod = 1
        for d in resolved:
            if d != -1:
                known_prod *= d
        if total_elems % known_prod != 0:
            raise ValueError(
                f"resolve_reshape_shape: cannot infer -1 dim: "
                f"total={total_elems}, known_prod={known_prod}"
            )
        resolved[minus_one_indices[0]] = total_elems // known_prod

    return resolved


def pad4d(shape: list) -> list:
    """Pad *shape* to 4-D by prepending 1-s."""
    return [1] * (4 - len(shape)) + list(shape)


# ── GraphCache: pre-built lookup dicts for O(1) access ──────────────────────


class GraphCache:
    """Pre-built index of an ONNX graph for O(1) lookups.

    Build once at the start of a transform via ``GraphCache(model.graph)``
    instead of scanning nodes/initializers linearly on every query.

    Attributes:
        init_map:      ``{init_name: TensorProto}`` — initializer lookup.
        init_names:    ``set[str]`` — names of all initializers.
        producer_map:  ``{tensor_name: NodeProto}`` — node that produces each tensor.
        consumer_map:  ``{tensor_name: [(NodeProto, input_index), ...]}`` — consumers.
        shape_map:     ``{tensor_name: list[int]}`` — concrete shapes from
                       value_info + inputs + outputs + initializers.
        op_index:      ``{op_type: [NodeProto, ...]}`` — nodes grouped by op type.
        graph_input_names:  ``set[str]`` — names of graph inputs.
        graph_output_names: ``set[str]`` — names of graph outputs.
    """

    __slots__ = (
        "_arr_cache",
        "_graph",
        "consumer_map",
        "graph_input_names",
        "graph_output_names",
        "init_map",
        "init_names",
        "op_index",
        "producer_map",
        "vi_map",
    )

    def __init__(self, graph: onnx.GraphProto) -> None:
        self._graph = graph
        self._arr_cache: dict[str, np.ndarray] = {}
        self.rebuild()

    # ── Build / rebuild all indices from the current graph state ──────────

    def rebuild(self) -> None:
        """(Re)build every index from scratch.  Call after bulk graph mutations."""
        graph = self._graph

        # Initializer map
        self.init_map: dict[str, onnx.TensorProto] = {init.name: init for init in graph.initializer}
        self.init_names: set[str] = set(self.init_map)

        # Producer map: tensor_name → node that produces it
        self.producer_map: dict[str, onnx.NodeProto] = {}
        for node in graph.node:
            for out in node.output:
                if out:
                    self.producer_map[out] = node

        # Consumer map: tensor_name → [(consumer_node, input_index), ...]
        self.consumer_map: dict[str, list[tuple]] = {}
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp:
                    self.consumer_map.setdefault(inp, []).append((node, i))

        # Value-info map: tensor_name → ValueInfoProto (single source of truth)
        self.vi_map: dict[str, onnx.ValueInfoProto] = {}
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            self.vi_map[vi.name] = vi

        # Op-type index: op_type → [node, ...]
        self.op_index: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            self.op_index.setdefault(node.op_type, []).append(node)

        # Graph boundary names
        self.graph_input_names: set[str] = {inp.name for inp in graph.input}
        self.graph_output_names: set[str] = {out.name for out in graph.output}

    # ── Convenience methods matching old helper signatures ────────────────

    def get_consumers(self, tensor_name: str) -> list[tuple]:
        """Return ``[(consumer_node, input_index), ...]`` for *tensor_name*."""
        return self.consumer_map.get(tensor_name, [])

    def get_producer(self, tensor_name: str) -> onnx.NodeProto | None:
        """Return the node that produces *tensor_name*, or ``None``."""
        return self.producer_map.get(tensor_name)

    def get_shape(self, tensor_name: str) -> list[int] | None:
        """Return the concrete shape of *tensor_name*, or ``None``."""
        vi = self.vi_map.get(tensor_name)
        if vi is not None:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                shape: list[int] = []
                for d in tt.shape.dim:
                    if d.HasField("dim_value"):
                        shape.append(d.dim_value)
                    else:
                        shape.append(0)
                        break
                return shape
        init = self.init_map.get(tensor_name)
        if init is not None:
            return list(init.dims)
        return None

    def get_init(self, name: str) -> onnx.TensorProto | None:
        """Return the initializer ``TensorProto`` for *name*, or ``None``."""
        return self.init_map.get(name)

    def is_init(self, name: str) -> bool:
        """Return whether *name* is an initializer."""
        return name in self.init_names

    def nodes_by_op(self, op_type: str) -> list[onnx.NodeProto]:
        """Return all nodes with the given *op_type*."""
        return self.op_index.get(op_type, [])

    def get_dtype(self, tensor_name: str) -> int | None:
        """Return the ONNX elem_type int for *tensor_name*, or ``None``."""
        vi = self.vi_map.get(tensor_name)
        if vi is not None:
            return vi.type.tensor_type.elem_type
        init = self.init_map.get(tensor_name)
        if init is not None:
            return init.data_type
        return None

    def update_shape(self, name: str, new_shape: list) -> bool:
        """Update the shape of tensor *name* in-place via ``vi_map``.  O(1) lookup."""
        vi = self.vi_map.get(name)
        if vi is None:
            return False
        tt = vi.type.tensor_type
        tt.shape.dim.clear()
        for d in new_shape:
            dim = tt.shape.dim.add()
            if d is None:
                dim.dim_param = ""
            else:
                dim.dim_value = int(d)
        return True

    # ── Cached numpy array deserialization ────────────────────────────────

    def get_init_array(self, name: str) -> np.ndarray | None:
        """Return deserialized numpy array for initializer *name*, cached.

        Avoids repeated ``numpy_helper.to_array()`` calls on the same
        initializer — each deserialization copies the full ``raw_data``
        bytes field into a new numpy array.
        """
        if name in self._arr_cache:
            return self._arr_cache[name]
        init = self.init_map.get(name)
        if init is None:
            return None
        arr = numpy_helper.to_array(init)
        self._arr_cache[name] = arr
        return arr

    def clear_array_cache(self) -> None:
        """Drop all cached numpy arrays (call after modifying initializers)."""
        self._arr_cache = {}


def insert_nodes_at_position(graph, nodes_to_add, reference_node):
    """Insert new nodes at the position of a reference node to maintain topological order."""
    if not nodes_to_add:
        return

    node_list = list(graph.node)
    try:
        insert_position = node_list.index(reference_node)
    except ValueError:
        insert_position = len(node_list)

    for i, new_node in enumerate(nodes_to_add):
        graph.node.insert(insert_position + i, new_node)


def batch_replace_nodes(
    graph: onnx.GraphProto,
    replacements: dict[int, tuple[list, list]],
) -> None:
    """Batch node removal + insertion, processed in reverse index order.

    ``replacements`` maps the index of the first node to remove to a
    ``(nodes_to_remove, nodes_to_add)`` tuple.  Processing in reverse
    ensures earlier indices stay valid after later removals/insertions.

    For each entry the nodes in ``nodes_to_remove`` are deleted (scanning
    forward from *idx*) and ``nodes_to_add`` are inserted at *idx*.
    If ``nodes_to_remove`` is empty, ``nodes_to_add`` are simply inserted.
    If ``nodes_to_add`` is empty, nodes are only removed.
    """
    for idx in sorted(replacements.keys(), reverse=True):
        nodes_to_remove, nodes_to_add = replacements[idx]
        for node in reversed(nodes_to_remove):
            if node in graph.node:
                graph.node.remove(node)
        for i, new_node in enumerate(nodes_to_add):
            graph.node.insert(idx + i, new_node)


def get_initializer_by_name(model: onnx.ModelProto, init_name: str) -> np.ndarray | None:
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None


def get_constant_by_name(model: onnx.ModelProto, const_name: str) -> np.ndarray | None:
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == const_name:
            value = None
            for attr in node.attribute:
                if attr.name == "value":
                    value = attr.t
                    break
            if value is not None:
                return numpy_helper.to_array(value)
    return None


def get_tensor_dtype_by_name(model: onnx.ModelProto, name: str) -> int | None:
    """Return ONNX TensorProto data type enum for a tensor name, or None."""
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    for vi in model.graph.value_info:
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    return attr.t.data_type
    for graph_input in model.graph.input:
        if graph_input.name == name:
            return graph_input.type.tensor_type.elem_type
    for output in model.graph.output:
        if output.name == name:
            return output.type.tensor_type.elem_type
    return None


def get_node_attr_i(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    """Return integer attribute *name* from *node*, or *default* if absent."""
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def set_node_attr_i(node: onnx.NodeProto, name: str, value: int) -> None:
    """Set integer attribute *name* on *node*, creating it if absent."""
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return
    node.attribute.append(onnx.helper.make_attribute(name, int(value)))


def axes_for_rank(rank: int) -> list[int]:
    """Leading axes to prepend when unsqueezing rank-R → 4D."""
    return {1: [0, 1, 2], 2: [0, 1], 3: [0], 4: []}.get(rank, [])


def calculate_clip_range(node, model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray]:
    x_scale = get_initializer_by_name(model, node.input[1])
    if x_scale is None:
        x_scale = get_constant_by_name(model, node.input[1])
    x_zero_point = get_initializer_by_name(model, node.input[2])
    if x_scale is None:
        raise ValueError(f"{node.name} should have x_scale value")
    if x_zero_point is None:
        logger.info("x_zero_point is None; assuming int8 symmetric range for clip")
        x_zero_point = np.array(0, dtype=np.int32)
        int_max = np.int32(127)
        int_min = np.int32(-128)
    else:
        int_max = np.int32(
            65535
            if x_zero_point.dtype == np.uint16
            else 255
            if x_zero_point.dtype == np.uint8
            else 127
        )
        int_min = np.int32(0 if x_zero_point.dtype in (np.uint16, np.uint8) else -128)
        x_zero_point = x_zero_point.astype(np.int32)
    clip_min = ((int_min - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return clip_min, clip_max


def add_value_info(
    graph: onnx.GraphProto,
    name: str,
    dtype: int = onnx.TensorProto.FLOAT,
    shape: list | tuple | None = None,
) -> None:
    """Append a ``value_info`` entry for tensor *name* with *dtype* and *shape*."""
    graph.value_info.append(onnx.helper.make_tensor_value_info(name, dtype, shape))


def add_unique_initializers(graph, unique_set: set[str], initializers_list) -> int:
    """Add initializers whose names are not already in unique_set; update unique_set in place."""
    added_count = 0
    for init in initializers_list:
        if init.name not in unique_set:
            graph.initializer.append(init)
            unique_set.add(init.name)
            added_count += 1
    return added_count


def save_model(
    model: onnx.ModelProto,
    output_path: str,
    *,
    use_external_data: bool,
    external_data_name: str | None,
    verbose: bool,
) -> None:
    """Save ONNX model with optional external data (same convention as other graph surgeries)."""
    if verbose:
        logger.info("Saving modified model to: %s", output_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if use_external_data:
        if external_data_name is None:
            external_data_name = os.path.basename(output_path) + "_data"
        if verbose:
            logger.info("Saving weights to external file: %s", external_data_name)
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=1024,
        )
    else:
        onnx.save(model, output_path)


_SAVE_KW = frozenset({"use_external_data", "external_data_name", "verbose"})


def run_onnx_file_transform(
    model_path: str,
    output_path: str,
    transform_fn: Callable[..., onnx.ModelProto | None],
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX, apply ``transform_fn`` (``ModelProto`` in → optional ``ModelProto`` out), save, return model."""
    transform_kw = {k: v for k, v in kwargs.items() if k not in _SAVE_KW}
    if verbose:
        logger.info("Loading model from: %s", model_path)
    model = onnx.load(model_path, load_external_data=True)
    model = infer_shapes(model)
    sig = inspect.signature(transform_fn)
    try:
        bound = sig.bind_partial(model, **transform_kw)
    except TypeError:
        # Legacy transforms that only accept ``model`` (no keyword args from callers).
        bound = sig.bind_partial(model)
    bound.apply_defaults()
    result = transform_fn(*bound.args, **bound.kwargs)
    if result is not None:
        model = result
    model = infer_shapes(model)
    save_model(
        model,
        output_path,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
    )
    if verbose:
        logger.info("Saved: %s", output_path)
    return model
