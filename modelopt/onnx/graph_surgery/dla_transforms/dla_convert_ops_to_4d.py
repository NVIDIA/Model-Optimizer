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

"""Convert non-4D operations to work with 4D tensors in a single graph pass.

Each eligible op gets:
  - Unsqueeze node(s) inserted before non-4D data inputs
  - Axis / perm attribute adjustments to match the promoted 4D inputs
  - Squeeze node(s) inserted after outputs to restore the original rank

Constant initializer inputs are expanded in-place (prepend 1s) instead of
adding Unsqueeze nodes.

A subsequent ``dla-remove-intermediary-squeeze-and-unsqueeze`` pass removes
adjacent Squeeze→Unsqueeze pairs, yielding fully-4D intermediate tensors.

Ops **skipped** (handled by separate transforms or structurally special):
  LSTM, MatMul, Gemm  — dedicated transforms exist
  Squeeze, Unsqueeze  — structural; modifying them would corrupt the graph
  Constant, Identity, Shape — no axes / passthrough
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    batch_replace_nodes,
    get_node_attr_i,
    pad4d,
    resolve_reshape_shape,
    set_node_attr_i,
)
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures
from .dla_indexing import _operator_nodes
from .dla_rank_conversion import _runtime_input_names


class _InitializerNames(set[str]):
    """Track inserted controls while reusing the shared graph name allocator."""

    __slots__ = ("_cache",)

    def __init__(self, cache: GraphCache) -> None:
        """Seed existing initializer names and retain the shared allocator."""
        super().__init__(cache.init_names)
        self._cache = cache

    def unique_name(self, preferred: str) -> str:
        """Reserve a name across values, controls and nodes."""
        return self._cache.unique_name(preferred)


class _CacheShapeProxy:
    """Dict-like wrapper around ``GraphCache.get_shape`` for handler compatibility."""

    __slots__ = ("_cache", "_captured", "_index_views", "_runtime_inputs", "_standard_nodes")

    def __init__(self, cache: GraphCache, standard_nodes, captured, runtime_inputs) -> None:
        """Retain native shape lookups and precomputed operator ownership guards."""
        self._cache = cache
        self._standard_nodes = {id(node) for node in standard_nodes}
        self._captured = captured
        self._runtime_inputs = runtime_inputs
        self._index_views = {}

    def get(self, name, default=None):
        """Return a native shape or the caller default."""
        r = self._cache.get_shape(name)
        return r if r is not None else default

    def __contains__(self, name):
        """Check whether native shape metadata is available."""
        return self._cache.get_shape(name) is not None

    def __getitem__(self, name):
        """Return the known native shape or raise KeyError."""
        r = self._cache.get_shape(name)
        if r is None:
            raise KeyError(name)
        return r


# ---------------------------------------------------------------------------
# Ops that this transform must not touch
# ---------------------------------------------------------------------------
_OP_SKIP = frozenset(
    {
        "LSTM",
        "MatMul",
        "Gemm",
        "Squeeze",
        "Unsqueeze",
        "Constant",
        "Identity",
        "Shape",
        "GroupQueryAttention",
        "Resize",
    }
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _find_squeeze_axes(shape_4d: list, target_shape: list) -> list[int]:
    """Return the axes to remove from ``shape_4d`` to obtain ``target_shape``.

    Handles cases where the extra unary dim is at a non-leading position.
    Example: shape_4d=[1,1,3,1], target=[1,1,3] → need axis=3, not 0.

    Falls back to leading axes when no exact match is found.
    """
    n_remove = len(shape_4d) - len(target_shape)
    if n_remove <= 0:
        return []
    target = list(target_shape)
    for axes_combo in combinations(range(len(shape_4d)), n_remove):
        remaining = [shape_4d[i] for i in range(len(shape_4d)) if i not in set(axes_combo)]
        if remaining == target:
            return sorted(axes_combo)
    return list(range(n_remove))  # fallback: leading axes


def _axis_delta(rank: int) -> int:
    """Amount to add to a non-negative axis index when promoting rank-R → 4D."""
    return max(0, 4 - rank)


_get_attr_i = get_node_attr_i
_set_attr_i = set_node_attr_i


def _make_unsqueeze_to_4d(
    input_name: str,
    rank: int,
    prefix: str,
    unique_init: _InitializerNames,
    graph,
    c: int,
):
    """Return (out_name, unsq_node).  If rank==4 return (input_name, None)."""
    if rank == 4:
        return input_name, None
    axes = list(range(4 - rank))
    axes_init = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=unique_init.unique_name(f"{prefix}_unsq_axes_{c}")
    )
    add_unique_initializers(graph, unique_init, [axes_init])
    out_name = unique_init.unique_name(f"{prefix}_unsq4d_{c}")
    node = helper.make_node(
        "Unsqueeze",
        inputs=[input_name, axes_init.name],
        outputs=[out_name],
        name=unique_init.unique_name(f"{prefix}_unsq4d_node_{c}"),
    )
    return out_name, node


def _make_squeeze_to_rank(
    input_name: str,
    orig_rank: int,
    output_name: str,
    prefix: str,
    unique_init: _InitializerNames,
    graph,
    c: int,
    *,
    shape_4d: list | None = None,
    orig_out_shape: list | None = None,
):
    """Return sq_node.  If orig_rank==4 return None.

    When ``shape_4d`` and ``orig_out_shape`` are both supplied the squeeze axes
    are determined by :func:`_find_squeeze_axes`, which handles the case where
    the extra unary dim is at a non-leading position (e.g. shape_4d=[1,1,3,1]
    with orig_out_shape=[1,1,3] → squeeze axis=3, not 0).
    """
    if orig_rank == 4:
        return None
    if shape_4d is not None and orig_out_shape is not None:
        axes = _find_squeeze_axes(shape_4d, orig_out_shape)
    else:
        axes = list(range(4 - orig_rank))
    axes_init = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=unique_init.unique_name(f"{prefix}_sq_axes_{c}")
    )
    add_unique_initializers(graph, unique_init, [axes_init])
    node = helper.make_node(
        "Squeeze",
        inputs=[input_name, axes_init.name],
        outputs=[output_name],
        name=unique_init.unique_name(f"{prefix}_sq_node_{c}"),
    )
    return node


def _expand_init_dims(init, cache) -> None:
    """Prepend 1-dims to a static initializer to make it 4D. No-op if already ≥ 4D."""
    if len(init.dims) < 4:
        shape = pad4d(list(init.dims))
        del init.dims[:]
        init.dims.extend(shape)
        info = cache.vi_map.get(init.name)
        if info is not None and info.type.tensor_type.HasField("shape"):
            info.type.tensor_type.shape.CopyFrom(
                helper.make_tensor_value_info(
                    init.name, init.data_type, shape
                ).type.tensor_type.shape
            )


# ---------------------------------------------------------------------------
# Per-op handlers
#   Signature:
#     (node, graph, shape_map, init_names, unique_init,
#      inits_to_add, cnt_ref) -> list[NodeProto] | None
#
#   Return value:
#     None  - node needs no change (skip)
#     list  - full replacement chain: [pre_nodes..., node_or_new_node, post_nodes...]
#             The original `node` object (possibly modified in-place) is included
#             when it should be kept; Flatten replaces it with a new Reshape node.
#
#   cnt_ref is a list[int] used as a mutable counter (avoids nonlocal boilerplate).
# ---------------------------------------------------------------------------


def _handle_transpose(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Adapt supported permutations and restore the original output rank."""
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4 or (rank == 5 and 1 not in orig_shape):
        return None

    pfx = node.name or f"tr_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    # ── Adjust perm ──────────────────────────────────────────────────────────
    perm_updated = False
    for attr in node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
            if rank == 2:
                new_perm = [0, 1, perm[0] + 2, perm[1] + 2]
            elif rank == 3:
                new_perm = [0, perm[0] + 1, perm[1] + 1, perm[2] + 1]
            elif rank == 5:
                # General algorithm: find rightmost unary dim in the OUTPUT shape,
                # drop that permutation entry, shift remaining values.
                current_shape = [orig_shape[i] for i in perm]
                idx_to_remove = len(current_shape) - current_shape[::-1].index(1) - 1
                removed_orig_dim = perm[idx_to_remove]
                new_perm_raw = [p for i, p in enumerate(perm) if i != idx_to_remove]
                new_perm = [p if p < removed_orig_dim else p - 1 for p in new_perm_raw]
            else:
                return None  # unsupported rank
            del attr.ints[:]
            attr.ints.extend(int(x) for x in new_perm)
            perm_updated = True
            break

    if not perm_updated:
        return None

    if rank == 5:
        # 5D collapse: squeeze the unary input dim → 4D Transpose → unsqueeze output back to 5D.
        # Round-tripping through the unary dim preserves the output shape, so ORT parity holds
        # and a subsequent dla-remove-intermediary-squeeze-and-unsqueeze pass can clean up.
        c_5d = cnt_ref[0]
        cnt_ref[0] += 1
        c_5d2 = cnt_ref[0]
        cnt_ref[0] += 1

        # Squeeze input at removed_orig_dim
        sq_axis_init = numpy_helper.from_array(
            np.array([removed_orig_dim], dtype=np.int64),
            name=unique_init.unique_name(f"{pfx}_5d_sq_axes_{c_5d}"),
        )
        add_unique_initializers(graph, unique_init, [sq_axis_init])
        sq_out = unique_init.unique_name(f"{pfx}_5d_sq4d_{c_5d}")
        sq_node = helper.make_node(
            "Squeeze",
            inputs=[in_name, sq_axis_init.name],
            outputs=[sq_out],
            name=unique_init.unique_name(f"{pfx}_5d_sq_{c_5d}"),
        )
        node.input[0] = sq_out

        # Unsqueeze Transpose output at idx_to_remove to restore the 5D output shape
        unsq_axis_init = numpy_helper.from_array(
            np.array([idx_to_remove], dtype=np.int64),
            name=unique_init.unique_name(f"{pfx}_5d_unsq_axes_{c_5d2}"),
        )
        add_unique_initializers(graph, unique_init, [unsq_axis_init])
        orig_out = node.output[0]
        tr_out_4d = unique_init.unique_name(f"{pfx}_5d_tr4d_{c_5d}")
        unsq_node = helper.make_node(
            "Unsqueeze",
            inputs=[tr_out_4d, unsq_axis_init.name],
            outputs=[orig_out],
            name=unique_init.unique_name(f"{pfx}_5d_unsq_{c_5d2}"),
        )
        node.output[0] = tr_out_4d
        return [sq_node, node, unsq_node]

    pre: list = []
    post: list = []

    # ── Unsqueeze input ───────────────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # ── Squeeze output ────────────────────────────────────────────────────────
    c2 = cnt_ref[0]
    cnt_ref[0] += 1
    orig_out = node.output[0]
    temp_out = unique_init.unique_name(f"{pfx}_tr_4d_{c}")
    sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, c2)
    if sq:
        node.output[0] = temp_out
        post.append(sq)

    return [*pre, node, *post]


def _handle_reshape(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Reshape: modify target shape initializer to 4D and Squeeze output back to original rank.

    The Reshape input is left as-is (Reshape accepts any-rank input).
    A Squeeze is added after to restore the original output rank so downstream
    shape_map lookups remain consistent with runtime shapes.
    Leading-singleton 5D targets compute in 4D and then restore their native rank.
    """
    if len(node.input) < 2 or node.input[1] not in init_names:
        return None  # dynamic shape — skip

    shape_init = shape_map._cache.get_init(node.input[1])
    if shape_init is None:
        return None

    shape_arr = numpy_helper.to_array(shape_init)

    # Determine the concrete output shape.
    # Priority 1: use the output shape from shape_map (populated by shape inference).
    # Priority 2: compute from the Reshape's shape initializer, resolving -1/0
    #             using the input shape via resolve_reshape_shape.
    out_name = node.output[0] if node.output else None
    out_shape_known = shape_map.get(out_name) if out_name else None

    if out_shape_known is not None:
        resolved = list(out_shape_known)
    else:
        in_name = node.input[0]
        input_shape = shape_map.get(in_name)
        if input_shape is not None:
            try:
                resolved = resolve_reshape_shape(shape_arr, input_shape)
            except ValueError:
                resolved = [int(d) for d in shape_arr.tolist()]
        else:
            resolved = [int(d) for d in shape_arr.tolist()]

    if 0 in resolved:
        return None
    target_dims = len(resolved)

    if target_dims == 4:
        return None  # already 4D

    if target_dims == 2:
        orig_out_rank = 2
        new_shape = np.array([1, 1, *resolved], dtype=np.int64)
    elif target_dims == 3:
        orig_out_rank = 3
        new_shape = np.array([1, *resolved], dtype=np.int64)
    elif target_dims == 5 and resolved[0] == 1:
        c_r = cnt_ref[0]
        cnt_ref[0] += 1
        new_name = unique_init.unique_name(f"{shape_init.name}_4d_{c_r}")
        inits_to_add.append(numpy_helper.from_array(np.asarray(resolved[1:], np.int64), new_name))
        axes = numpy_helper.from_array(
            np.asarray([0], np.int64), unique_init.unique_name(f"{new_name}_axes")
        )
        add_unique_initializers(graph, unique_init, [axes])
        original = node.output[0]
        temporary = unique_init.unique_name(original + "_reshape_4d")
        node.input[1] = new_name
        node.output[0] = temporary
        restore = helper.make_node(
            "Unsqueeze",
            [temporary, axes.name],
            [original],
            name=unique_init.unique_name(original + "_restore_5d"),
        )
        return [node, restore]
    else:
        return None  # unsupported

    c_r = cnt_ref[0]
    cnt_ref[0] += 1
    new_shape_name = unique_init.unique_name(f"{shape_init.name}_4d_{c_r}")
    # Add new init; leave old for any other consumers.
    inits_to_add.append(numpy_helper.from_array(new_shape, new_shape_name))
    node.input[1] = new_shape_name

    # Squeeze output back to original rank so downstream shape_map stays consistent
    pfx = node.name or f"rs_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1
    orig_out = node.output[0]
    temp_out = unique_init.unique_name(f"{pfx}_rs_4d_{c}")
    sq = _make_squeeze_to_rank(temp_out, orig_out_rank, orig_out, pfx, unique_init, graph, c)
    if sq:
        node.output[0] = temp_out
        return [node, sq]
    return [node]


def _handle_slice(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Adapt fixed slicing axes around leading input promotion."""
    if len(node.input) > 3 and node.input[3] and node.input[3] not in init_names:
        return None
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4:
        return None

    pfx = node.name or f"sl_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    post: list = []
    if len(node.input) < 4 or not node.input[3]:
        starts = shape_map._cache.get_init(node.input[1])
        if starts is None:
            return None
        axes_name = unique_init.unique_name(f"{pfx}_default_axes_{c}")
        inits_to_add.append(
            numpy_helper.from_array(
                np.arange(np.prod(starts.dims), dtype=np.int64) + 4 - rank, axes_name
            )
        )
        while len(node.input) < 4:
            node.input.append("")
        node.input[3] = axes_name

    # ── Update axes (input[3]) ────────────────────────────────────────────────
    if len(node.input) > 3 and node.input[3] and node.input[3] in init_names:
        init = shape_map._cache.get_init(node.input[3])
        if init is not None:
            axes_arr = numpy_helper.to_array(init)
            delta = _axis_delta(rank)
            new_axes = np.array(
                [int(a) + delta if int(a) >= 0 else int(a) for a in axes_arr],
                dtype=np.int64,
            )
            c_a = cnt_ref[0]
            cnt_ref[0] += 1
            new_axes_name = unique_init.unique_name(f"{init.name}_4d_{c_a}")
            # Add new init with unique name; leave old init for other consumers.
            inits_to_add.append(numpy_helper.from_array(new_axes, new_axes_name))
            node.input[3] = new_axes_name

    # ── Unsqueeze data input ──────────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # ── Squeeze outputs ───────────────────────────────────────────────────────
    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_sl_4d_{c}_o{oi}")
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_argmax(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Prepare reduction axes while restoring native ArgMax outputs."""
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)

    pfx = node.name or f"argmax_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    post: list = []

    delta = _axis_delta(rank)
    orig_axis = _get_attr_i(node, "axis", 0)
    adjusted_axis = orig_axis + delta if orig_axis >= 0 else orig_axis
    _set_attr_i(node, "axis", adjusted_axis)

    orig_keepdims = _get_attr_i(node, "keepdims", 1)
    _set_attr_i(node, "keepdims", 1)

    # ── Unsqueeze input ───────────────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # ── Squeeze outputs ───────────────────────────────────────────────────────
    c2 = cnt_ref[0]
    cnt_ref[0] += 1
    orig_out = node.output[0]  # ArgMax always has exactly one output
    temp_out = unique_init.unique_name(f"{pfx}_argmax_4d_{c}")

    if orig_keepdims == 0:
        # squeeze: leading (4-rank) axes + the reduced axis (held at 1 due to keepdims=1)
        leading = list(range(4 - rank))
        sq_axes = sorted({*leading, adjusted_axis})
        axes_init = numpy_helper.from_array(
            np.array(sq_axes, dtype=np.int64),
            name=unique_init.unique_name(f"{pfx}_argmax_sq_axes_{c2}"),
        )
        add_unique_initializers(graph, unique_init, [axes_init])
        sq = helper.make_node(
            "Squeeze",
            inputs=[temp_out, axes_init.name],
            outputs=[orig_out],
            name=unique_init.unique_name(f"{pfx}_argmax_sq_{c2}"),
        )
        node.output[0] = temp_out
        post.append(sq)
    else:
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, c2)
        if sq:
            node.output[0] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_reduce(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Covers ReduceSum / ReduceMean / ReduceMax / ReduceMin."""
    if len(node.input) > 1 and node.input[1] and node.input[1] not in init_names:
        return None
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)

    pfx = node.name or f"reduce_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    post: list = []

    delta = _axis_delta(rank)
    orig_keepdims = _get_attr_i(node, "keepdims", 1)
    _set_attr_i(node, "keepdims", 1)

    # ── Adjust axes ───────────────────────────────────────────────────────────
    adjusted_axes: list[int] = []

    if len(node.input) > 1 and node.input[1] and node.input[1] in init_names:
        # axes as input[1] initializer (opset 13+)
        init = shape_map._cache.get_init(node.input[1])
        if init is not None:
            orig_axes = list(numpy_helper.to_array(init).astype(int))
            adjusted_axes = [a + delta if a >= 0 else a for a in orig_axes]
            c_a = cnt_ref[0]
            cnt_ref[0] += 1
            new_axes_name = unique_init.unique_name(f"{init.name}_4d_{c_a}")
            # Add new init; leave old for any other consumers.
            inits_to_add.append(
                numpy_helper.from_array(np.array(adjusted_axes, dtype=np.int64), new_axes_name)
            )
            node.input[1] = new_axes_name
    else:
        for attr in node.attribute:
            if attr.name == "axes":
                orig_axes = list(attr.ints)
                adjusted_axes = [a + delta if a >= 0 else a for a in orig_axes]
                del attr.ints[:]
                attr.ints.extend(int(x) for x in adjusted_axes)
                break

    # ── Unsqueeze input ───────────────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # ── Squeeze outputs (Reduce always has one output) ────────────────────────
    c2 = cnt_ref[0]
    cnt_ref[0] += 1
    orig_out = node.output[0]
    temp_out = unique_init.unique_name(f"{pfx}_reduce_4d_{c}")

    if orig_keepdims == 0:
        leading = list(range(4 - rank))
        noop = _get_attr_i(node, "noop_with_empty_axes", 0)
        reduced = (
            [a % 4 for a in adjusted_axes] if adjusted_axes else [] if noop else list(range(4))
        )
        sq_axes = sorted(set(leading + reduced))
        axes_init = numpy_helper.from_array(
            np.array(sq_axes, dtype=np.int64),
            name=unique_init.unique_name(f"{pfx}_reduce_sq_axes_{c2}"),
        )
        add_unique_initializers(graph, unique_init, [axes_init])
        sq = helper.make_node(
            "Squeeze",
            inputs=[temp_out, axes_init.name],
            outputs=[orig_out],
            name=unique_init.unique_name(f"{pfx}_reduce_sq_{c2}"),
        )
        node.output[0] = temp_out
        post.append(sq)
    else:
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, c2)
        if sq:
            node.output[0] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_gather(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Promote Gather to 4D output.

    Strategy
    --------
    Gather output rank = data_rank - 1 + indices_rank.
    After unsqueezing data to 4D, we squeeze indices to rank-1 so that:
        output_rank = 4 - 1 + 1 = 4  (no output squeeze needed).

    Indices are also cast to INT32 (DLA requirement).
    Fixed indices use a shared private flattened view; runtime defaults stay dynamic.
    For dynamic indices: Cast node followed by Squeeze node(s).
    """
    in_name = node.input[0]
    idx_name = node.input[1]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    idx_shape = shape_map.get(idx_name)
    if idx_shape is None or sum(dim != 1 for dim in idx_shape) > 1:
        return None
    native_axis = _get_attr_i(node, "axis", 0) % rank
    native_output = [*orig_shape[:native_axis], *idx_shape, *orig_shape[native_axis + 1 :]]
    promoted = pad4d(orig_shape)
    axis_4d = native_axis + _axis_delta(rank)
    converted_output = [*promoted[:axis_4d], math.prod(idx_shape), *promoted[axis_4d + 1 :]]
    if len(native_output) > 4:
        return None
    restore_axes = _find_squeeze_axes(converted_output, native_output)
    if (
        any(converted_output[axis] != 1 for axis in restore_axes)
        or [dim for axis, dim in enumerate(converted_output) if axis not in restore_axes]
        != native_output
    ):
        return None

    pfx = node.name or f"gather_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    is_init = (
        idx_name in init_names
        and idx_name not in shape_map._runtime_inputs
        and not _has_unloaded_external(shape_map._cache, [idx_name])
    )

    # ── 1. Cast indices to INT32 and flatten to 1-D ───────────────────────────
    if is_init:
        if idx_name not in shape_map._index_views:
            flat = (
                numpy_helper.to_array(shape_map._cache.get_init(idx_name))
                .reshape(-1)
                .astype(np.int32, copy=False)
            )
            private = unique_init.unique_name(f"{pfx}_idx_i32_{c}")
            inits_to_add.append(numpy_helper.from_array(flat, private))
            shape_map._index_views[idx_name] = private
        node.input[1] = shape_map._index_views[idx_name]
    else:
        # Check if the producer of idx_name is already a Cast node — reuse it.
        cache = shape_map._cache
        producer = cache.get_producer(idx_name)
        producer_cast = (
            producer
            if producer is not None
            and producer.op_type == "Cast"
            and producer.output
            and producer.output[0] == idx_name
            else None
        )
        exclusive = (
            producer_cast is not None
            and len(cache.get_consumers(idx_name)) == 1
            and idx_name not in cache.graph_output_names | shape_map._captured
            and id(producer_cast) in shape_map._standard_nodes
        )
        if exclusive:
            _set_attr_i(producer_cast, "to", TensorProto.INT32)
            value = cache.vi_map.get(idx_name)
            if value is not None:
                value.type.tensor_type.elem_type = TensorProto.INT32
            cast_name = idx_name
        elif shape_map._cache.get_dtype(idx_name) != TensorProto.INT32:
            # Insert a new Cast to INT32
            c_cast = cnt_ref[0]
            cnt_ref[0] += 1
            cast_out = unique_init.unique_name(f"{pfx}_idx_i32_{c_cast}")
            pre.append(
                helper.make_node(
                    "Cast",
                    inputs=[idx_name],
                    outputs=[cast_out],
                    to=TensorProto.INT32,
                    name=unique_init.unique_name(f"{pfx}_idx_cast_{c_cast}"),
                )
            )
            cast_name = cast_out
        else:
            cast_name = idx_name

        # Unsqueeze indices to 4D first, then Squeeze all unit dims to rank-1.
        # This ensures the Squeeze axes are stable after upstream 4D promotion.
        idx_shape = shape_map.get(idx_name)
        idx_rank = len(idx_shape) if idx_shape is not None else None

        if idx_rank is not None and (idx_rank not in (1, 4)):
            # Unsqueeze to 4D
            c_unsq = cnt_ref[0]
            cnt_ref[0] += 1
            unsq_axes = list(range(4 - idx_rank))
            unsq_axes_init = numpy_helper.from_array(
                np.array(unsq_axes, dtype=np.int64),
                unique_init.unique_name(f"{pfx}_idx_unsq_axes_{c_unsq}"),
            )
            add_unique_initializers(graph, unique_init, [unsq_axes_init])
            unsq_out = unique_init.unique_name(f"{pfx}_idx_4d_{c_unsq}")
            pre.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[cast_name, unsq_axes_init.name],
                    outputs=[unsq_out],
                    name=unique_init.unique_name(f"{pfx}_idx_unsq_{c_unsq}"),
                )
            )
            cast_name = unsq_out
            idx_shape_4d = pad4d(idx_shape)
        else:
            idx_shape_4d = list(idx_shape) if idx_shape is not None else None

        # Squeeze all unit dims from 4D shape to get rank-1
        if idx_shape_4d is not None and len(idx_shape_4d) > 1:
            sq_axes = [i for i in range(len(idx_shape_4d)) if idx_shape_4d[i] == 1]
            if len(sq_axes) == len(idx_shape_4d):
                sq_axes = sq_axes[:-1]  # keep at least one dim
            if sq_axes:
                c_sq = cnt_ref[0]
                cnt_ref[0] += 1
                sq_axes_init = numpy_helper.from_array(
                    np.array(sq_axes, dtype=np.int64),
                    unique_init.unique_name(f"{pfx}_idx_sq_axes_{c_sq}"),
                )
                add_unique_initializers(graph, unique_init, [sq_axes_init])
                sq_out = unique_init.unique_name(f"{pfx}_idx_1d_{c_sq}")
                pre.append(
                    helper.make_node(
                        "Squeeze",
                        inputs=[cast_name, sq_axes_init.name],
                        outputs=[sq_out],
                        name=unique_init.unique_name(f"{pfx}_idx_sq_{c_sq}"),
                    )
                )
                node.input[1] = sq_out
            else:
                node.input[1] = cast_name
        else:
            node.input[1] = cast_name

    # ── 2. Unsqueeze data to 4D + adjust axis ────────────────────────────────
    adjusted_axis = _get_attr_i(node, "axis", 0)
    if rank != 4:
        delta = _axis_delta(rank)
        orig_axis = _get_attr_i(node, "axis", 0)  # ONNX Gather default: 0
        if orig_axis >= 0:
            adjusted_axis = orig_axis + delta
        else:
            adjusted_axis = orig_axis  # negative axes don't shift
        _set_attr_i(node, "axis", adjusted_axis)
        out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
        node.input[0] = out4d
        if unsq:
            pre.insert(0, unsq)

    # ── 3. Squeeze 4D output back to original output shape ────────────────────
    # After promoting data to 4D and flattening indices to 1D, the Gather output
    # is 4D: data_4d[:axis] + [indices_len] + data_4d[axis+1:].
    # The extra unary dims may NOT be at the leading positions, so we must use
    # _find_squeeze_axes with the actual 4D output shape rather than falling back
    # to leading-axis removal.
    orig_out = node.output[0]
    orig_out_shape = native_output
    post: list = []
    if len(orig_out_shape) < 4:
        c_sq = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_gather_4d_{c_sq}")

        sq = _make_squeeze_to_rank(
            temp_out,
            len(orig_out_shape),
            orig_out,
            pfx,
            unique_init,
            graph,
            c_sq,
            shape_4d=converted_output,
            orig_out_shape=list(orig_out_shape),
        )
        if sq:
            node.output[0] = temp_out
            post.append(sq)

    if not pre and not post:
        return None
    return [*pre, node, *post]


def _handle_gatherelements(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Promote GatherElements to 4D output.

    GatherElements requires equal data and index ranks and produces output
    with the same shape as indices.  After unsqueezing data to 4D, indices must
    also be unsqueezed to 4D so the ranks match → output is naturally 4D.

    Indices are cast to INT32 (DLA requirement).
    Index Cast and Unsqueeze views preserve fixed storage and runtime defaults.
    """
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4:
        return None

    pfx = node.name or f"gathel_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    delta = _axis_delta(rank)

    # ── 1. Adjust axis ────────────────────────────────────────────────────────
    orig_axis = _get_attr_i(node, "axis", 0)  # ONNX GatherElements default: 0
    if orig_axis >= 0:
        _set_attr_i(node, "axis", orig_axis + delta)
    # negative axes don't need adjustment

    # ── 2. Unsqueeze data to 4D ───────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # ── 3. Cast indices to INT32 and unsqueeze to 4D (must match data shape) ──
    if len(node.input) > 1 and node.input[1]:
        idx_name = node.input[1]
        # Only cast to INT32 if indices are not already INT32
        ge_need_cast = shape_map._cache.get_dtype(idx_name) != TensorProto.INT32

        if ge_need_cast:
            c_cast = cnt_ref[0]
            cnt_ref[0] += 1
            cast_out = unique_init.unique_name(f"{pfx}_idx_i32_{c_cast}")
            pre.append(
                helper.make_node(
                    "Cast",
                    inputs=[idx_name],
                    outputs=[cast_out],
                    to=TensorProto.INT32,
                    name=unique_init.unique_name(f"{pfx}_idx_cast_{c_cast}"),
                )
            )
            idx_after_cast = cast_out
        else:
            idx_after_cast = idx_name

        # Unsqueeze indices to 4D to match data
        if rank < 4:
            c_usq = cnt_ref[0]
            cnt_ref[0] += 1
            idx_4d, idx_unsq = _make_unsqueeze_to_4d(
                idx_after_cast,
                rank,
                unique_init.unique_name(f"{pfx}_idx"),
                unique_init,
                graph,
                c_usq,
            )
            node.input[1] = idx_4d
            if idx_unsq:
                pre.append(idx_unsq)
        else:
            node.input[1] = idx_after_cast

    # ── 4. Squeeze outputs back to original output shapes ─────────────────────
    post: list = []
    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        if rank < 4:
            c_sq = cnt_ref[0]
            cnt_ref[0] += 1
            temp_out = unique_init.unique_name(f"{pfx}_gathel_4d_{c_sq}_o{oi}")
            sq = _make_squeeze_to_rank(
                temp_out,
                rank,
                orig_out,
                pfx,
                unique_init,
                graph,
                c_sq,
            )
            if sq:
                node.output[oi] = temp_out
                post.append(sq)

    return [*pre, node, *post]


def _handle_expand(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Promote fixed Expand targets and restore native output shapes."""
    if len(node.input) < 2 or node.input[1] not in init_names:
        return None
    in_name = node.input[0]
    orig_in_shape = shape_map.get(in_name)  # may be None if input shape is unknown

    native_output = None
    if orig_in_shape is not None:
        target = shape_map._cache.get_init_array(node.input[1])
        native_output = list(np.broadcast_shapes(orig_in_shape, target.tolist()))
    if native_output is None and any(shape_map.get(name) is None for name in node.output if name):
        return None
    pfx = node.name or f"exp_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []

    # ── Update shape initializer (input[1]) to 4D ────────────────────────────
    if len(node.input) > 1 and node.input[1] and node.input[1] in init_names:
        init = shape_map._cache.get_init(node.input[1])
        if init is not None:
            arr = numpy_helper.to_array(init)
            if len(arr) < 4:
                new_arr = np.array(pad4d(arr.tolist()), dtype=arr.dtype)
                c_t = cnt_ref[0]
                cnt_ref[0] += 1
                new_tile_name = unique_init.unique_name(f"{init.name}_4d_{c_t}")
                inits_to_add.append(numpy_helper.from_array(new_arr, new_tile_name))
                node.input[1] = new_tile_name

    # ── Unsqueeze data input to 4D (skip for scalar / unknown input shape) ───
    in_rank = len(orig_in_shape) if orig_in_shape is not None else None
    if in_rank is not None and in_rank > 0:
        out4d, unsq = _make_unsqueeze_to_4d(in_name, in_rank, pfx, unique_init, graph, c)
        node.input[0] = out4d
        if unsq:
            pre.append(unsq)

    # ── Squeeze outputs based on each output's original shape ────────────────
    # The transformed Expand now produces 4D. Use each output's known original
    # shape (not the input rank) to determine squeeze axes. This handles:
    #   * ops where input rank != output rank
    #   * ops where input shape is unknown but output shape is known
    post: list = []
    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        orig_out_shape = shape_map.get(orig_out)
        if orig_out_shape is None:
            orig_out_shape = native_output
        if orig_out_shape is None or len(orig_out_shape) >= 4:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_exp_4d_{c}_o{oi}")
        sq = _make_squeeze_to_rank(
            temp_out, len(orig_out_shape), orig_out, pfx, unique_init, graph, ci
        )
        if sq:
            node.output[oi] = temp_out
            post.append(sq)
    return [*pre, node, *post]


def _handle_tile(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Promote fixed Tile repeats and restore native output shapes."""
    if len(node.input) < 2 or node.input[1] not in init_names:
        return None
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4:
        return None

    pfx = node.name or f"tile_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []

    # ── Update repeats initializer (input[1]) if 3D ───────────────────────────
    if len(node.input) > 1 and node.input[1] and node.input[1] in init_names:
        init = shape_map._cache.get_init(node.input[1])
        if init is not None:
            arr = numpy_helper.to_array(init)
            if len(arr) < 4:
                new_arr = np.asarray(pad4d(arr.tolist()), dtype=arr.dtype)
                name = unique_init.unique_name(f"{init.name}_tile4d_{c}")
                inits_to_add.append(numpy_helper.from_array(new_arr, name))
                node.input[1] = name

    # ── Unsqueeze data input ──────────────────────────────────────────────────
    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    # Squeeze outputs back to original rank
    post: list = []
    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_tile_4d_{c}_o{oi}")
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)
    return [*pre, node, *post]


def _handle_lpnorm(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Adapt normalization axes around leading input promotion."""
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4:
        return None

    pfx = node.name or f"lpnorm_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    post: list = []

    axis = _get_attr_i(node, "axis", -1)
    _set_attr_i(node, "axis", axis + 4 - rank if axis >= 0 else axis)

    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_lpnorm_4d_{c}_o{oi}")
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_axis_op(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Softmax, LogSoftmax, Concat, Split, TopK, DequantizeLinear."""
    skip_input_indices = (
        {1, 2}
        if node.op_type == "DequantizeLinear"
        else ({1} if node.op_type in {"Split", "TopK"} else set())
    )
    data_inputs = [inp for i, inp in enumerate(node.input) if inp and i not in skip_input_indices]
    if not data_inputs or any(shape_map.get(inp) is None for inp in data_inputs):
        return None
    first_inp_name = data_inputs[0]
    orig_rank = len(shape_map.get(first_inp_name))
    if orig_rank == 4:
        return None

    pfx = node.name or f"axop_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    # Expected 4D shape (shape-preserving assumption) for robust squeeze-axis finding
    first_in_shape = shape_map.get(first_inp_name) if first_inp_name else None
    expected_4d_shape = pad4d(first_in_shape) if first_in_shape else None

    pre: list = []
    post: list = []

    delta = _axis_delta(orig_rank)
    if not any(attr.name == "axis" for attr in node.attribute):
        defaults = {
            "Softmax": -1,
            "LogSoftmax": -1,
            "Split": 0,
            "TopK": -1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 1,
        }
        if node.op_type in defaults:
            node.attribute.append(helper.make_attribute("axis", defaults[node.op_type]))

    # ── Adjust axis attribute ─────────────────────────────────────────────────
    for attr in node.attribute:
        if attr.name == "axis":
            if attr.i >= 0:
                attr.i += delta
            break

    # ── Unsqueeze tensor inputs; expand initializer inputs ────────────────────
    for i, inp in enumerate(node.input):
        if not inp or i in skip_input_indices:
            continue
        if inp in init_names:
            _expand_init_dims(shape_map._cache.get_init(inp), shape_map._cache)
        else:
            s = shape_map.get(inp)
            if s is None or len(s) == 4:
                continue
            ci = cnt_ref[0]
            cnt_ref[0] += 1
            inp_4d, unsq = _make_unsqueeze_to_4d(
                inp, len(s), unique_init.unique_name(f"{pfx}_in{i}"), unique_init, graph, ci
            )
            node.input[i] = inp_4d
            if unsq:
                pre.append(unsq)

    # ── Squeeze outputs ───────────────────────────────────────────────────────
    for oi, out in enumerate(node.output):
        if not out:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp = unique_init.unique_name(f"{pfx}_axop_4d_{c}_o{oi}")
        out_shape = shape_map.get(out)
        sq = _make_squeeze_to_rank(
            temp,
            orig_rank,
            out,
            pfx,
            unique_init,
            graph,
            ci,
            shape_4d=expected_4d_shape,
            orig_out_shape=out_shape,
        )
        if sq:
            node.output[oi] = temp
            post.append(sq)

    return [*pre, node, *post]


def _handle_flatten(node, graph, shape_map, _init_names, unique_init, _inits_to_add, cnt_ref):
    """Flatten → Unsqueeze(to 4D) → Reshape([1,1,outer,inner]) → Squeeze([outer,inner]).

    Same logic as the former ``dla_flatten_to_reshape`` transform, now inlined.
    Input already 4D: Unsqueeze step is skipped.
    """
    in_name = node.input[0]
    out_name = node.output[0]

    in_shape = shape_map.get(in_name)
    if in_shape is None:
        logger.debug("Flatten %r: input shape unknown; skipping.", node.name)
        return None

    rank = len(in_shape)
    raw_axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
    axis = int(raw_axis)
    if axis < 0:
        axis = rank + axis
    axis = max(0, min(axis, rank))

    outer = int(np.prod(in_shape[:axis])) if axis > 0 else 1

    pfx = node.name or f"fl_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1
    chain: list = []

    # ── Step 1: Unsqueeze input to 4D (skip if already 4D) ───────────────────
    if rank < 4:
        n_new = 4 - rank
        unsq_axes_init = numpy_helper.from_array(
            np.arange(n_new, dtype=np.int64), name=unique_init.unique_name(f"{pfx}_unsq_axes_{c}")
        )
        add_unique_initializers(graph, unique_init, [unsq_axes_init])
        in4d = unique_init.unique_name(f"{pfx}_in4d_{c}")
        chain.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[in_name, unsq_axes_init.name],
                outputs=[in4d],
                name=unique_init.unique_name(f"{pfx}_unsq_{c}"),
            )
        )
    else:
        in4d = in_name

    # ── Step 2: Reshape to [1, 1, outer, -1] (-1 infers inner automatically) ─
    shape_init = numpy_helper.from_array(
        np.array([1, 1, outer, -1], dtype=np.int64),
        name=unique_init.unique_name(f"{pfx}_reshape_shape_{c}"),
    )
    add_unique_initializers(graph, unique_init, [shape_init])
    flat4d = unique_init.unique_name(f"{pfx}_flat4d_{c}")
    chain.append(
        helper.make_node(
            "Reshape",
            inputs=[in4d, shape_init.name],
            outputs=[flat4d],
            name=unique_init.unique_name(f"{pfx}_reshape_{c}"),
        )
    )

    # ── Step 3: Squeeze [1,1,outer,-1] → [outer,-1] ──────────────────────────
    sq_axes_init = numpy_helper.from_array(
        np.array([0, 1], dtype=np.int64), name=unique_init.unique_name(f"{pfx}_sq_axes_{c}")
    )
    add_unique_initializers(graph, unique_init, [sq_axes_init])
    chain.append(
        helper.make_node(
            "Squeeze",
            inputs=[flat4d, sq_axes_init.name],
            outputs=[out_name],
            name=unique_init.unique_name(f"{pfx}_sq_{c}"),
        )
    )

    logger.debug(
        "Flatten %r [axis=%d]: %s → Unsqueeze→Reshape[1,1,%d,-1]→Squeeze",
        node.name,
        axis,
        in_shape,
        outer,
    )
    return chain  # replaces the Flatten node entirely


def _handle_instancenorm(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """3D InstanceNorm: add trailing dim [N,C,D]→[N,C,D,1] before; remove after."""
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank != 3:
        return None  # 4D already fine; other ranks not handled

    pfx = node.name or f"in_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1
    c2 = cnt_ref[0]
    cnt_ref[0] += 1

    # Unsqueeze trailing dim: [N,C,D] → [N,C,D,1]
    unsq_axes = numpy_helper.from_array(
        np.array([3], dtype=np.int64), name=unique_init.unique_name(f"{pfx}_in_unsq_axes_{c}")
    )
    add_unique_initializers(graph, unique_init, [unsq_axes])
    in4d = unique_init.unique_name(f"{pfx}_in_unsq4d_{c}")
    unsq = helper.make_node(
        "Unsqueeze",
        inputs=[in_name, unsq_axes.name],
        outputs=[in4d],
        name=unique_init.unique_name(f"{pfx}_in_unsq_{c}"),
    )
    node.input[0] = in4d

    # Squeeze trailing dim from output: [N,C,D,1] → [N,C,D]
    sq_axes = numpy_helper.from_array(
        np.array([3], dtype=np.int64), name=unique_init.unique_name(f"{pfx}_in_sq_axes_{c2}")
    )
    add_unique_initializers(graph, unique_init, [sq_axes])
    out4d = unique_init.unique_name(f"{pfx}_in_4d_out_{c}")
    sq = helper.make_node(
        "Squeeze",
        inputs=[out4d, sq_axes.name],
        outputs=[node.output[0]],
        name=unique_init.unique_name(f"{pfx}_in_sq_{c2}"),
    )
    node.output[0] = out4d

    return [unsq, node, sq]


def _handle_layernorm(node, graph, shape_map, _init_names, unique_init, _inits_to_add, cnt_ref):
    """LayerNormalization: unsqueeze data input to 4D, keep scale/bias unchanged, update axis.

    LayerNormalization inputs:
      input[0]: data   — promoted to 4D via Unsqueeze; output squeezed back to orig rank.
      input[1]: scale  — 1D, shape matches the normalised dims; left untouched.
      input[2]: bias   — 1D, optional; left untouched.

    The ``axis`` attribute is shifted by ``4 - orig_rank`` so it still addresses
    the correct dimension after the data tensor is promoted.
    """
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None or len(orig_shape) == 4:
        return None

    orig_rank = len(orig_shape)
    delta = 4 - orig_rank
    pfx = node.name or f"ln_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    # ── Unsqueeze data input to 4D ────────────────────────────────────────────
    in4d, unsq = _make_unsqueeze_to_4d(in_name, orig_rank, pfx, unique_init, graph, c)
    node.input[0] = in4d

    # ── Update axis attribute ─────────────────────────────────────────────────
    # LayerNorm default axis is -1 (last dim) — negative stays the same.
    orig_axis = _get_attr_i(node, "axis", -1)
    if orig_axis >= 0:
        _set_attr_i(node, "axis", orig_axis + delta)

    pre = [unsq] if unsq else []
    post = []
    native_axis = orig_axis % orig_rank
    for index, original in enumerate(list(node.output)):
        if not original:
            continue
        native_shape = shape_map.get(original)
        if native_shape is None:
            native_shape = (
                orig_shape
                if index == 0
                else orig_shape[:native_axis] + [1] * (orig_rank - native_axis)
            )
        c2 = cnt_ref[0]
        cnt_ref[0] += 1
        temporary = unique_init.unique_name(f"{pfx}_ln_4d_{c}_{index}")
        squeeze = _make_squeeze_to_rank(
            temporary,
            orig_rank,
            original,
            pfx,
            unique_init,
            graph,
            c2,
            shape_4d=pad4d(native_shape),
            orig_out_shape=native_shape,
        )
        if squeeze:
            node.output[index] = temporary
            post.append(squeeze)
    return [*pre, node, *post]


def _handle_generic(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Default handler for unregistered ops (Add, Sub, Mul, Relu, Cast, etc.)."""
    if node.op_type not in {
        "Abs",
        "Neg",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Sqrt",
        "Exp",
        "Log",
        "Erf",
        "Cast",
        "Clip",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Greater",
        "Less",
        "Equal",
        "And",
        "Or",
        "Xor",
        "Where",
        "Max",
        "Min",
        "PRelu",
        "Sum",
        "Conv",
        "ConvTranspose",
    }:
        return None
    # Determine orig_rank from first non-initializer input
    first_inp_name = None
    orig_rank = None
    for inp in node.input:
        if inp and inp not in init_names:
            s = shape_map.get(inp)
            if s is not None and len(s) > 0:
                first_inp_name = inp
                orig_rank = len(s)
                break
    if orig_rank is None:
        return None
    if node.op_type in {"Conv", "ConvTranspose"} and orig_rank != 4:
        return None
    broadcast_output = None
    if node.op_type not in {"Conv", "ConvTranspose"}:
        shapes = [shape_map.get(name) for name in node.input if name]
        if any(shape is None for shape in shapes):
            return None
        broadcast_output = list(np.broadcast_shapes(*shapes))
        orig_rank = len(broadcast_output)

    pfx = node.name or f"gen_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    # Pre-compute expected 4D shape (shape-preserving assumption: 4D_out = 4D_in).
    # This is used by _make_squeeze_to_rank to find the correct squeeze axis even
    # when the unary dim is not at the leading position (e.g. [1,1,3,1] → [1,1,3]).
    first_in_shape = shape_map.get(first_inp_name) if first_inp_name else None
    expected_4d_shape = (
        pad4d(broadcast_output if broadcast_output is not None else first_in_shape)
        if first_in_shape
        else None
    )

    pre: list = []
    post: list = []

    # Conv / ConvTranspose bias (input[2]) must stay 1-D per the ONNX spec.
    conv_bias_indices: set[int] = {2} if node.op_type in ("Conv", "ConvTranspose") else set()

    for i, inp in enumerate(node.input):
        if not inp:
            continue
        if i in conv_bias_indices:
            continue  # Conv/ConvTranspose bias — leave untouched
        if inp in init_names:
            init = shape_map._cache.get_init(inp)
            if len(init.dims) != 0:
                _expand_init_dims(init, shape_map._cache)
        else:
            s = shape_map.get(inp)
            if s is None:
                continue
            if len(s) == 0:
                continue  # scalar dynamic tensor — leave untouched
            # Convert all dynamic inputs to 4D — even already-4D inputs go through
            # _make_unsqueeze_to_4d which is a no-op (returns the same name) for rank==4.
            ci = cnt_ref[0]
            cnt_ref[0] += 1
            inp_4d, unsq = _make_unsqueeze_to_4d(
                inp, len(s), unique_init.unique_name(f"{pfx}_in{i}"), unique_init, graph, ci
            )
            node.input[i] = inp_4d
            if unsq:
                pre.append(unsq)

    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        orig_out_shape = shape_map.get(orig_out, broadcast_output)
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = unique_init.unique_name(f"{pfx}_gen_4d_{c}_o{oi}")
        sq = _make_squeeze_to_rank(
            temp_out,
            orig_rank,
            orig_out,
            pfx,
            unique_init,
            graph,
            ci,
            shape_4d=expected_4d_shape,
            orig_out_shape=orig_out_shape,
        )
        if sq:
            node.output[oi] = temp_out
            post.append(sq)

    return [*pre, node, *post]


# ---------------------------------------------------------------------------
# DequantizeLinear — static initializer data path
# ---------------------------------------------------------------------------


def _handle_dequantize_linear(
    node, graph, shape_map, init_names, unique_init, _inits_to_add, cnt_ref
):
    """Promote fixed nonblocked quantized data and restore its native output rank."""
    if not node.input or node.input[0] not in init_names:
        return None
    data_init = shape_map._cache.get_init(node.input[0])
    scale_shape = shape_map.get(node.input[1])
    if (
        data_init is None
        or len(data_init.dims) >= 4
        or _get_attr_i(node, "block_size", 0)
        or scale_shape is None
        or len(scale_shape) > 1
    ):
        return None
    orig_rank = len(data_init.dims)
    c = cnt_ref[0]
    cnt_ref[0] += 1
    original = node.output[0]
    temporary = shape_map._cache.unique_name(original + "_dq_4d")
    squeeze = _make_squeeze_to_rank(
        temporary, orig_rank, original, node.name or "dq", unique_init, graph, c
    )
    _expand_init_dims(data_init, shape_map._cache)
    if scale_shape:
        axis = _get_attr_i(node, "axis", 1)
        if axis >= 0:
            _set_attr_i(node, "axis", axis + 4 - orig_rank)
    node.output[0] = temporary
    return [node, squeeze]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_OP_HANDLERS = {
    "Reshape": _handle_reshape,
    "Transpose": _handle_transpose,
    "Slice": _handle_slice,
    "ArgMax": _handle_argmax,
    "ReduceSum": _handle_reduce,
    "ReduceMean": _handle_reduce,
    "ReduceMax": _handle_reduce,
    "ReduceMin": _handle_reduce,
    "Gather": _handle_gather,
    "GatherElements": _handle_gatherelements,
    "Expand": _handle_expand,
    "Tile": _handle_tile,
    "LpNormalization": _handle_lpnorm,
    "Softmax": _handle_axis_op,
    "LogSoftmax": _handle_axis_op,
    "Concat": _handle_axis_op,
    "Split": _handle_axis_op,
    "TopK": _handle_axis_op,
    "DequantizeLinear": _handle_dequantize_linear,
    "Flatten": _handle_flatten,
    "InstanceNormalization": _handle_instancenorm,
    "LayerNormalization": _handle_layernorm,
}


# ---------------------------------------------------------------------------
# Post-pass: split squeeze nodes whose output is both a graph output and
# consumed by internal nodes
# ---------------------------------------------------------------------------


def _split_graph_output_squeezes(model: onnx.ModelProto) -> None:
    """Give internal consumers private native views without changing opaque operators or public bindings."""
    graph = model.graph
    cache = GraphCache(graph)
    outputs = {value.name: value for value in graph.output}
    positions = {id(node): index for index, node in enumerate(graph.node)}
    replacements = {}
    for node in _operator_nodes(model, {"Squeeze"}):
        original = node.output[0]
        consumers = cache.get_consumers(original)
        if original not in outputs or not consumers:
            continue
        internal = cache.unique_name(original + "_sq")
        copied = onnx.NodeProto()
        copied.CopyFrom(node)
        copied.output[0] = internal
        copied.name = cache.unique_name((node.name or original) + "_internal")
        for consumer, port in consumers:
            consumer.input[port] = internal
        metadata = onnx.ValueInfoProto()
        metadata.CopyFrom(outputs[original])
        metadata.name = internal
        graph.value_info.append(metadata)
        replacements[positions[id(node)]] = ([node], [node, copied])
    batch_replace_nodes(graph, replacements)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------


def _has_unloaded_external(cache, names):
    """Identify controls whose payload cannot be read without the caller's model directory."""
    return any(
        (tensor := cache.get_init(name)) is not None
        and onnx.external_data_helper.uses_external_data(tensor)
        and not tensor.HasField("raw_data")
        for name in names
        if name
    )


def _apply_convert_ops_to_4d(model: onnx.ModelProto) -> onnx.ModelProto:
    """Wrap eligible non-4D operations while restoring their native output contracts."""
    if next((op.version for op in model.opset_import if op.domain in {"", "ai.onnx"}), 1) < 13:
        return model
    graph = model.graph

    # ── Build graph cache for O(1) lookups ────────────────────────────────────
    cache = GraphCache(graph)
    unique_init = _InitializerNames(cache)
    standard_nodes = _operator_nodes(model, {node.op_type for node in graph.node})
    captured = _subgraph_captures(graph)
    shape_map = _CacheShapeProxy(cache, standard_nodes, captured, _runtime_input_names(model))
    init_names = cache.init_names

    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    inits_to_add: list = []
    replacements_batch: dict[int, tuple[list, list]] = {}
    cnt_ref = [0]

    # ── Process each node ─────────────────────────────────────────────────────
    for node in standard_nodes:
        if node.op_type in _OP_SKIP:
            continue
        if (
            node.op_type in {"Reshape", "Slice", "Expand", "Tile"}
            or node.op_type.startswith("Reduce")
        ) and _has_unloaded_external(cache, node.input[1:]):
            continue
        if (
            any(
                (shape := cache.get_shape(name)) is not None and len(shape) > 4
                for name in node.input
            )
            and node.op_type != "Transpose"
        ):
            continue
        if any(
            name in cache.vi_map and cache.get_shape(name) is None
            for name in [*node.input, *node.output]
            if name
        ):
            continue
        handler = _OP_HANDLERS.get(node.op_type, _handle_generic)
        if any(
            name in init_names
            and len(cache.init_map[name].dims) != 4
            and (
                name in cache.graph_input_names
                or (
                    handler in {_handle_axis_op, _handle_generic, _handle_dequantize_linear}
                    and (
                        len(cache.get_consumers(name)) > 1
                        or name in cache.graph_output_names | captured
                    )
                )
            )
            for name in node.input
            if name
        ):
            continue

        # Quick check: does any input OR output have rank ≠ 4?
        # We need to transform if any tensor is non-4D so the graph is fully 4D.
        any_non4d = False
        for inp in node.input:
            if inp:
                s = shape_map.get(inp)
                if s is not None and len(s) != 4:
                    any_non4d = True
                    break
        if not any_non4d:
            for out in node.output:
                if out:
                    s = shape_map.get(out)
                    if s is not None and len(s) != 4:
                        any_non4d = True
                        break
        if not any_non4d:
            continue

        # Warn and skip 5D tensors without a unary dim (only Transpose handles 5D)
        has_bad_5d = False
        if node.op_type != "Transpose":
            for inp in node.input:
                if inp:
                    s = shape_map.get(inp)
                    if s is not None and len(s) == 5 and 1 not in s:
                        logger.warning(
                            "Node %r (%s): input %r has 5D shape %s with no unary dim; skipping.",
                            node.name,
                            node.op_type,
                            inp,
                            s,
                        )
                        has_bad_5d = True
                        break
        if has_bad_5d:
            continue

        result = handler(
            node,
            graph,
            shape_map,
            init_names,
            unique_init,
            inits_to_add,
            cnt_ref,
        )
        if result is not None:
            replacements_batch[node_idx[id(node)]] = ([node], result)

    # ── Apply all node replacements ───────────────────────────────────────────
    batch_replace_nodes(graph, replacements_batch)

    add_unique_initializers(graph, unique_init, inits_to_add)

    # ── Split squeeze nodes shared between graph outputs and internal nodes ────
    _split_graph_output_squeezes(model)
    _add_legacy_initializer_inputs(model)

    logger.debug("convert_ops_to_4d: transformed %d node(s).", len(replacements_batch))
    return model
