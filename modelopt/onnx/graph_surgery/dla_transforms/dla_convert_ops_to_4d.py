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

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    axes_for_rank,
    batch_replace_nodes,
    get_node_attr_i,
    pad4d,
    resolve_reshape_shape,
    run_onnx_file_transform,
    set_node_attr_i,
)


class _CacheShapeProxy:
    """Dict-like wrapper around ``GraphCache.get_shape`` for handler compatibility."""

    __slots__ = ("_cache",)

    def __init__(self, cache: GraphCache) -> None:
        self._cache = cache

    def get(self, name, default=None):
        r = self._cache.get_shape(name)
        return r if r is not None else default

    def __contains__(self, name):
        return self._cache.get_shape(name) is not None

    def __getitem__(self, name):
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


def _get_dtype_from_graph(graph: onnx.GraphProto, name: str) -> int | None:
    """Return the ONNX element type of tensor ``name`` from value_info / inputs / outputs."""
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    for init in graph.initializer:
        if init.name == name:
            return init.data_type
    return None


_axes_for_rank = axes_for_rank


def _find_squeeze_axes(shape_4d: list, target_shape: list) -> list[int]:
    """Return the axes to remove from ``shape_4d`` to obtain ``target_shape``.

    Handles cases where the extra unary dim is at a non-leading position.
    Example: shape_4d=[1,1,3,1], target=[1,1,3] → need axis=3, not 0.

    Falls back to leading axes when no exact match is found.
    """
    from itertools import combinations

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
    unique_init: set,
    graph,
    c: int,
):
    """Return (out_name, unsq_node).  If rank==4 return (input_name, None)."""
    if rank == 4:
        return input_name, None
    axes = _axes_for_rank(rank)
    axes_init = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=f"{prefix}_unsq_axes_{c}"
    )
    add_unique_initializers(graph, unique_init, [axes_init])
    out_name = f"{prefix}_unsq4d_{c}"
    node = helper.make_node(
        "Unsqueeze",
        inputs=[input_name, axes_init.name],
        outputs=[out_name],
        name=f"{prefix}_unsq4d_node_{c}",
    )
    return out_name, node


def _make_squeeze_to_rank(
    input_name: str,
    orig_rank: int,
    output_name: str,
    prefix: str,
    unique_init: set,
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
        axes = _axes_for_rank(orig_rank)
    axes_init = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=f"{prefix}_sq_axes_{c}"
    )
    add_unique_initializers(graph, unique_init, [axes_init])
    node = helper.make_node(
        "Squeeze",
        inputs=[input_name, axes_init.name],
        outputs=[output_name],
        name=f"{prefix}_sq_node_{c}",
    )
    return node


def _expand_init_dims(init) -> None:
    """Prepend 1-dims to a static initializer to make it 4D. No-op if already ≥ 4D."""
    arr = numpy_helper.to_array(init)
    if arr.ndim >= 4:
        return
    new_shape = pad4d(list(arr.shape))
    init.CopyFrom(numpy_helper.from_array(arr.reshape(new_shape), init.name))


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
    in_name = node.input[0]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)
    if rank == 4:
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
                # General algorithm: find leftmost unary dim in the OUTPUT shape,
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
            name=f"{pfx}_5d_sq_axes_{c_5d}",
        )
        add_unique_initializers(graph, unique_init, [sq_axis_init])
        sq_out = f"{pfx}_5d_sq4d_{c_5d}"
        sq_node = helper.make_node(
            "Squeeze",
            inputs=[in_name, sq_axis_init.name],
            outputs=[sq_out],
            name=f"{pfx}_5d_sq_{c_5d}",
        )
        node.input[0] = sq_out

        # Unsqueeze Transpose output at idx_to_remove to restore the 5D output shape
        unsq_axis_init = numpy_helper.from_array(
            np.array([idx_to_remove], dtype=np.int64),
            name=f"{pfx}_5d_unsq_axes_{c_5d2}",
        )
        add_unique_initializers(graph, unique_init, [unsq_axis_init])
        orig_out = node.output[0]
        tr_out_4d = f"{pfx}_5d_tr4d_{c_5d}"
        unsq_node = helper.make_node(
            "Unsqueeze",
            inputs=[tr_out_4d, unsq_axis_init.name],
            outputs=[orig_out],
            name=f"{pfx}_5d_unsq_{c_5d2}",
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
    temp_out = f"{pfx}_tr_4d_{c}"
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
    5D targets with shape[0]==1 are collapsed to 4D; no Squeeze added for those.
    """
    if len(node.input) < 2 or node.input[1] not in init_names:
        return None  # dynamic shape — skip

    shape_init = None
    for init in graph.initializer:
        if init.name == node.input[1]:
            shape_init = init
            break
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
        # Collapse 5D → 4D; output rank stays 4D (no Squeeze)
        c_r = cnt_ref[0]
        cnt_ref[0] += 1
        new_name = f"{shape_init.name}_4d_{c_r}"
        # Add new init; leave old for any other consumers.
        inits_to_add.append(
            numpy_helper.from_array(np.array(resolved[1:], dtype=np.int64), new_name)
        )
        node.input[1] = new_name
        return [node]
    else:
        return None  # unsupported

    c_r = cnt_ref[0]
    cnt_ref[0] += 1
    new_shape_name = f"{shape_init.name}_4d_{c_r}"
    # Add new init; leave old for any other consumers.
    inits_to_add.append(numpy_helper.from_array(new_shape, new_shape_name))
    node.input[1] = new_shape_name

    # Squeeze output back to original rank so downstream shape_map stays consistent
    pfx = node.name or f"rs_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1
    orig_out = node.output[0]
    temp_out = f"{pfx}_rs_4d_{c}"
    sq = _make_squeeze_to_rank(temp_out, orig_out_rank, orig_out, pfx, unique_init, graph, c)
    if sq:
        node.output[0] = temp_out
        return [node, sq]
    return [node]


def _handle_slice(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
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

    # ── Update axes (input[3]) ────────────────────────────────────────────────
    if len(node.input) > 3 and node.input[3] and node.input[3] in init_names:
        for init in graph.initializer:
            if init.name == node.input[3]:
                axes_arr = numpy_helper.to_array(init)
                delta = _axis_delta(rank)
                new_axes = np.array(
                    [int(a) + delta if int(a) >= 0 else int(a) for a in axes_arr],
                    dtype=np.int64,
                )
                c_a = cnt_ref[0]
                cnt_ref[0] += 1
                new_axes_name = f"{init.name}_4d_{c_a}"
                # Add new init with unique name; leave old init for other consumers.
                inits_to_add.append(numpy_helper.from_array(new_axes, new_axes_name))
                node.input[3] = new_axes_name
                break

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
        temp_out = f"{pfx}_sl_4d_{c}_o{oi}"
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_argmax(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
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
    temp_out = f"{pfx}_argmax_4d_{c}"

    if orig_keepdims == 0:
        # squeeze: leading (4-rank) axes + the reduced axis (held at 1 due to keepdims=1)
        leading = list(range(4 - rank))
        sq_axes = sorted({*leading, adjusted_axis})
        axes_init = numpy_helper.from_array(
            np.array(sq_axes, dtype=np.int64), name=f"{pfx}_argmax_sq_axes_{c2}"
        )
        add_unique_initializers(graph, unique_init, [axes_init])
        sq = helper.make_node(
            "Squeeze",
            inputs=[temp_out, axes_init.name],
            outputs=[orig_out],
            name=f"{pfx}_argmax_sq_{c2}",
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
        for init in graph.initializer:
            if init.name == node.input[1]:
                orig_axes = list(numpy_helper.to_array(init).astype(int))
                adjusted_axes = [a + delta if a >= 0 else a for a in orig_axes]
                c_a = cnt_ref[0]
                cnt_ref[0] += 1
                new_axes_name = f"{init.name}_4d_{c_a}"
                # Add new init; leave old for any other consumers.
                inits_to_add.append(
                    numpy_helper.from_array(np.array(adjusted_axes, dtype=np.int64), new_axes_name)
                )
                node.input[1] = new_axes_name
                break
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
    temp_out = f"{pfx}_reduce_4d_{c}"

    if orig_keepdims == 0 and adjusted_axes:
        leading = list(range(4 - rank))
        sq_axes = sorted(set(leading + [a for a in adjusted_axes if a >= 0]))
        axes_init = numpy_helper.from_array(
            np.array(sq_axes, dtype=np.int64), name=f"{pfx}_reduce_sq_axes_{c2}"
        )
        add_unique_initializers(graph, unique_init, [axes_init])
        sq = helper.make_node(
            "Squeeze",
            inputs=[temp_out, axes_init.name],
            outputs=[orig_out],
            name=f"{pfx}_reduce_sq_{c2}",
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
    For initializer indices: cast + flatten done in-place.
    For dynamic indices: Cast node followed by Squeeze node(s).
    """
    in_name = node.input[0]
    idx_name = node.input[1]
    orig_shape = shape_map.get(in_name)
    if orig_shape is None:
        return None
    rank = len(orig_shape)

    pfx = node.name or f"gather_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []
    is_init = idx_name in init_names

    # ── 1. Cast indices to INT32 and flatten to 1-D ───────────────────────────
    if is_init:
        for init in graph.initializer:
            if init.name == idx_name:
                arr = numpy_helper.to_array(init)
                new_arr = (
                    np.array([arr.item()], dtype=np.int32)
                    if arr.ndim == 0
                    else arr.flatten().astype(np.int32)
                )
                init.CopyFrom(numpy_helper.from_array(new_arr, idx_name))
                break
    else:
        # Check if the producer of idx_name is already a Cast node — reuse it.
        producer_cast = None
        for n in graph.node:
            if n.op_type == "Cast" and n.output and n.output[0] == idx_name:
                producer_cast = n
                break

        if producer_cast is not None:
            # Reuse existing Cast — just update its target dtype to INT32
            _set_attr_i(producer_cast, "to", TensorProto.INT32)
            cast_name = idx_name
        elif _get_dtype_from_graph(graph, idx_name) != TensorProto.INT32:
            # Insert a new Cast to INT32
            c_cast = cnt_ref[0]
            cnt_ref[0] += 1
            cast_out = f"{pfx}_idx_i32_{c_cast}"
            pre.append(
                helper.make_node(
                    "Cast",
                    inputs=[idx_name],
                    outputs=[cast_out],
                    to=TensorProto.INT32,
                    name=f"{pfx}_idx_cast_{c_cast}",
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
                np.array(unsq_axes, dtype=np.int64), f"{pfx}_idx_unsq_axes_{c_unsq}"
            )
            add_unique_initializers(graph, unique_init, [unsq_axes_init])
            unsq_out = f"{pfx}_idx_4d_{c_unsq}"
            pre.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[cast_name, unsq_axes_init.name],
                    outputs=[unsq_out],
                    name=f"{pfx}_idx_unsq_{c_unsq}",
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
                    np.array(sq_axes, dtype=np.int64), f"{pfx}_idx_sq_axes_{c_sq}"
                )
                add_unique_initializers(graph, unique_init, [sq_axes_init])
                sq_out = f"{pfx}_idx_1d_{c_sq}"
                pre.append(
                    helper.make_node(
                        "Squeeze",
                        inputs=[cast_name, sq_axes_init.name],
                        outputs=[sq_out],
                        name=f"{pfx}_idx_sq_{c_sq}",
                    )
                )
                node.input[1] = sq_out
            else:
                node.input[1] = cast_name
        else:
            node.input[1] = cast_name

    # ── 2. Unsqueeze data to 4D + adjust axis ────────────────────────────────
    adjusted_axis = None
    data_4d_shape = None
    if rank != 4:
        delta = _axis_delta(rank)
        orig_axis = _get_attr_i(node, "axis", 0)  # ONNX Gather default: 0
        if orig_axis >= 0:
            adjusted_axis = orig_axis + delta
        else:
            adjusted_axis = orig_axis  # negative axes don't shift
        _set_attr_i(node, "axis", adjusted_axis)
        data_4d_shape = pad4d(orig_shape)
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
    orig_out_shape = shape_map.get(orig_out)
    post: list = []
    if orig_out_shape is not None and len(orig_out_shape) < 4:
        c_sq = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = f"{pfx}_gather_4d_{c_sq}"

        # Compute the 4D Gather output shape so _find_squeeze_axes gets the right axes.
        gather_4d_shape = None
        if data_4d_shape is not None and adjusted_axis is not None:
            # Indices were flattened to 1D; length = product of original indices dims
            idx_shape = shape_map.get(idx_name)
            if idx_shape is not None:
                import math

                indices_len = max(1, math.prod(idx_shape)) if idx_shape else 1
            else:
                indices_len = 1  # flattened scalar → length 1
            ax = adjusted_axis
            gather_4d_shape = [*data_4d_shape[:ax], indices_len, *data_4d_shape[ax + 1 :]]

        sq = _make_squeeze_to_rank(
            temp_out,
            len(orig_out_shape),
            orig_out,
            pfx,
            unique_init,
            graph,
            c_sq,
            shape_4d=gather_4d_shape,
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

    GatherElements requires data.shape == indices.shape and produces output
    with the same shape as indices.  After unsqueezing data to 4D, indices must
    also be unsqueezed to 4D so the shapes match → output is naturally 4D.

    Indices are cast to INT32 (DLA requirement).
    For initializer indices: cast + reshape to 4D done in-place.
    For dynamic indices: Cast node followed by Unsqueeze node(s).
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
        if idx_name in init_names:
            for init in graph.initializer:
                if init.name == idx_name:
                    arr = numpy_helper.to_array(init).astype(np.int32)
                    if arr.ndim < 4:
                        new_shape = pad4d(list(arr.shape))
                        init.CopyFrom(numpy_helper.from_array(arr.reshape(new_shape), idx_name))
                    break
        else:
            # Only cast to INT32 if indices are not already INT32
            ge_need_cast = _get_dtype_from_graph(graph, idx_name) != TensorProto.INT32

            if ge_need_cast:
                c_cast = cnt_ref[0]
                cnt_ref[0] += 1
                cast_out = f"{pfx}_idx_i32_{c_cast}"
                pre.append(
                    helper.make_node(
                        "Cast",
                        inputs=[idx_name],
                        outputs=[cast_out],
                        to=TensorProto.INT32,
                        name=f"{pfx}_idx_cast_{c_cast}",
                    )
                )
                idx_after_cast = cast_out
            else:
                idx_after_cast = idx_name

            # Unsqueeze indices to 4D to match data
            idx_shape = shape_map.get(idx_name)
            if idx_shape is not None and len(idx_shape) < 4:
                c_usq = cnt_ref[0]
                cnt_ref[0] += 1
                idx_4d, idx_unsq = _make_unsqueeze_to_4d(
                    idx_after_cast, len(idx_shape), f"{pfx}_idx", unique_init, graph, c_usq
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
        orig_out_shape = shape_map.get(orig_out)
        if orig_out_shape is not None and len(orig_out_shape) < 4:
            c_sq = cnt_ref[0]
            cnt_ref[0] += 1
            temp_out = f"{pfx}_gathel_4d_{c_sq}_o{oi}"
            sq = _make_squeeze_to_rank(
                temp_out,
                len(orig_out_shape),
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
    in_name = node.input[0]
    orig_in_shape = shape_map.get(in_name)  # may be None if input shape is unknown

    pfx = node.name or f"exp_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    pre: list = []

    # ── Update shape initializer (input[1]) to 4D ────────────────────────────
    if len(node.input) > 1 and node.input[1] and node.input[1] in init_names:
        for init in graph.initializer:
            if init.name == node.input[1]:
                arr = numpy_helper.to_array(init)
                if len(arr) < 4:
                    new_arr = np.array(pad4d(arr.tolist()), dtype=arr.dtype)
                    c_t = cnt_ref[0]
                    cnt_ref[0] += 1
                    new_tile_name = f"{init.name}_4d_{c_t}"
                    inits_to_add.append(numpy_helper.from_array(new_arr, new_tile_name))
                    node.input[1] = new_tile_name
                break

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
        if orig_out_shape is None or len(orig_out_shape) >= 4:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = f"{pfx}_exp_4d_{c}_o{oi}"
        sq = _make_squeeze_to_rank(
            temp_out, len(orig_out_shape), orig_out, pfx, unique_init, graph, ci
        )
        if sq:
            node.output[oi] = temp_out
            post.append(sq)
    return [*pre, node, *post]


def _handle_tile(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
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
        for init in graph.initializer:
            if init.name == node.input[1]:
                arr = numpy_helper.to_array(init)
                if len(arr) == 3:
                    new_arr = np.concatenate([[1], arr]).astype(arr.dtype)
                    init.CopyFrom(numpy_helper.from_array(new_arr, init.name))
                break

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
        temp_out = f"{pfx}_tile_4d_{c}_o{oi}"
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)
    return [*pre, node, *post]


def _handle_lpnorm(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
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

    _set_attr_i(node, "axis", -1)

    out4d, unsq = _make_unsqueeze_to_4d(in_name, rank, pfx, unique_init, graph, c)
    node.input[0] = out4d
    if unsq:
        pre.append(unsq)

    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = f"{pfx}_lpnorm_4d_{c}_o{oi}"
        sq = _make_squeeze_to_rank(temp_out, rank, orig_out, pfx, unique_init, graph, ci)
        if sq:
            node.output[oi] = temp_out
            post.append(sq)

    return [*pre, node, *post]


def _handle_axis_op(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Softmax, LogSoftmax, Concat, Split, TopK, DequantizeLinear."""
    # Determine orig_rank from first non-initializer tensor input
    first_inp_name = None
    orig_rank = None
    for inp in node.input:
        if inp and inp not in init_names:
            s = shape_map.get(inp)
            if s is not None:
                first_inp_name = inp
                orig_rank = len(s)
                break
    if orig_rank is None or orig_rank == 4:
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

    # ── Adjust axis attribute ─────────────────────────────────────────────────
    for attr in node.attribute:
        if attr.name == "axis":
            if attr.i >= 0:
                attr.i += delta
            break

    # ── Unsqueeze tensor inputs; expand initializer inputs ────────────────────
    # Some ops have non-data inputs that must NOT be expanded:
    #   DequantizeLinear:  inputs[1] (scale), inputs[2] (zero_point) — 1D per-channel
    #   Split:             input[1] (split_sizes) — 1D size list
    #   TopK:              input[1] (K) — scalar
    if node.op_type == "DequantizeLinear":
        skip_input_indices: set = {1, 2}
    elif node.op_type in ("Split", "TopK"):
        skip_input_indices = {1}
    else:
        skip_input_indices = set()

    for i, inp in enumerate(node.input):
        if not inp or i in skip_input_indices:
            continue
        if inp in init_names:
            for init in graph.initializer:
                if init.name == inp:
                    _expand_init_dims(init)
                    break
        else:
            s = shape_map.get(inp)
            if s is None or len(s) == 4:
                continue
            ci = cnt_ref[0]
            cnt_ref[0] += 1
            inp_4d, unsq = _make_unsqueeze_to_4d(
                inp, len(s), f"{pfx}_in{i}", unique_init, graph, ci
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
        temp = f"{pfx}_axop_4d_{c}_o{oi}"
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
            np.arange(n_new, dtype=np.int64), name=f"{pfx}_unsq_axes_{c}"
        )
        add_unique_initializers(graph, unique_init, [unsq_axes_init])
        in4d = f"{pfx}_in4d_{c}"
        chain.append(
            helper.make_node(
                "Unsqueeze",
                inputs=[in_name, unsq_axes_init.name],
                outputs=[in4d],
                name=f"{pfx}_unsq_{c}",
            )
        )
    else:
        in4d = in_name

    # ── Step 2: Reshape to [1, 1, outer, -1] (-1 infers inner automatically) ─
    shape_init = numpy_helper.from_array(
        np.array([1, 1, outer, -1], dtype=np.int64),
        name=f"{pfx}_reshape_shape_{c}",
    )
    add_unique_initializers(graph, unique_init, [shape_init])
    flat4d = f"{pfx}_flat4d_{c}"
    chain.append(
        helper.make_node(
            "Reshape",
            inputs=[in4d, shape_init.name],
            outputs=[flat4d],
            name=f"{pfx}_reshape_{c}",
        )
    )

    # ── Step 3: Squeeze [1,1,outer,-1] → [outer,-1] ──────────────────────────
    sq_axes_init = numpy_helper.from_array(
        np.array([0, 1], dtype=np.int64), name=f"{pfx}_sq_axes_{c}"
    )
    add_unique_initializers(graph, unique_init, [sq_axes_init])
    chain.append(
        helper.make_node(
            "Squeeze",
            inputs=[flat4d, sq_axes_init.name],
            outputs=[out_name],
            name=f"{pfx}_sq_{c}",
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
        np.array([3], dtype=np.int64), name=f"{pfx}_in_unsq_axes_{c}"
    )
    add_unique_initializers(graph, unique_init, [unsq_axes])
    in4d = f"{pfx}_in_unsq4d_{c}"
    unsq = helper.make_node(
        "Unsqueeze",
        inputs=[in_name, unsq_axes.name],
        outputs=[in4d],
        name=f"{pfx}_in_unsq_{c}",
    )
    node.input[0] = in4d

    # Squeeze trailing dim from output: [N,C,D,1] → [N,C,D]
    sq_axes = numpy_helper.from_array(np.array([3], dtype=np.int64), name=f"{pfx}_in_sq_axes_{c2}")
    add_unique_initializers(graph, unique_init, [sq_axes])
    out4d = f"{pfx}_in_4d_out_{c}"
    sq = helper.make_node(
        "Squeeze",
        inputs=[out4d, sq_axes.name],
        outputs=[node.output[0]],
        name=f"{pfx}_in_sq_{c2}",
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

    # ── Squeeze output back to original rank ──────────────────────────────────
    orig_out = node.output[0]
    orig_out_shape = shape_map.get(orig_out)
    expected_4d_shape = pad4d(orig_shape)
    c2 = cnt_ref[0]
    cnt_ref[0] += 1
    temp_out = f"{pfx}_ln_4d_{c}"
    sq = _make_squeeze_to_rank(
        temp_out,
        orig_rank,
        orig_out,
        pfx,
        unique_init,
        graph,
        c2,
        shape_4d=expected_4d_shape,
        orig_out_shape=orig_out_shape,
    )
    if sq:
        node.output[0] = temp_out

    pre = [unsq] if unsq else []
    post = [sq] if sq else []
    return [*pre, node, *post]


def _handle_generic(node, graph, shape_map, init_names, unique_init, inits_to_add, cnt_ref):
    """Default handler for unregistered ops (Add, Sub, Mul, Relu, Cast, etc.)."""
    # Determine orig_rank from first non-initializer input
    first_inp_name = None
    orig_rank = None
    for inp in node.input:
        if inp and inp not in init_names:
            s = shape_map.get(inp)
            if s is not None:
                first_inp_name = inp
                orig_rank = len(s)
                break
    if orig_rank is None:
        return None

    pfx = node.name or f"gen_{cnt_ref[0]}"
    c = cnt_ref[0]
    cnt_ref[0] += 1

    # Pre-compute expected 4D shape (shape-preserving assumption: 4D_out = 4D_in).
    # This is used by _make_squeeze_to_rank to find the correct squeeze axis even
    # when the unary dim is not at the leading position (e.g. [1,1,3,1] → [1,1,3]).
    first_in_shape = shape_map.get(first_inp_name) if first_inp_name else None
    expected_4d_shape = pad4d(first_in_shape) if first_in_shape else None

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
            for init in graph.initializer:
                if init.name == inp:
                    if len(init.dims) == 0:
                        break  # scalar initializer — leave untouched
                    _expand_init_dims(init)
                    break
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
                inp, len(s), f"{pfx}_in{i}", unique_init, graph, ci
            )
            node.input[i] = inp_4d
            if unsq:
                pre.append(unsq)

    for oi, orig_out in enumerate(node.output):
        if not orig_out:
            continue
        orig_out_shape = shape_map.get(orig_out)
        ci = cnt_ref[0]
        cnt_ref[0] += 1
        temp_out = f"{pfx}_gen_4d_{c}_o{oi}"
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
    node, graph, _shape_map, init_names, _unique_init, inits_to_add, _cnt_ref
):
    """DequantizeLinear whose data input (input[0]) is a static initializer.

    * Expands the data initializer to 4-D by prepending 1-dims.
    * Updates the ``axis`` attribute by ``4 - orig_rank`` so per-channel
      quantization still refers to the correct dimension.
    * Scale (input[1]) and zero_point (input[2]) are left untouched.
    * Returns ``None`` — no structural node change needed.

    When input[0] is dynamic (not an initializer) the node is left unchanged.
    """
    if not node.input or node.input[0] not in init_names:
        return None  # dynamic data tensor — leave untouched

    # Find the data initializer proto
    data_init = next((init for init in graph.initializer if init.name == node.input[0]), None)
    if data_init is None:
        return None

    orig_rank = len(data_init.dims)
    if orig_rank >= 4:
        return None  # already 4-D, nothing to do

    delta = 4 - orig_rank

    # ── Expand the data initializer to 4-D ───────────────────────────────────
    _expand_init_dims(data_init)

    # ── Update the axis attribute ─────────────────────────────────────────────
    # DequantizeLinear default axis is 1. After prepending delta leading
    # 1-dims the channel axis shifts to 1+delta. Negative axis stays the same.
    orig_axis = _get_attr_i(node, "axis", 1)
    if orig_axis >= 0:
        _set_attr_i(node, "axis", orig_axis + delta)

    logger.debug(
        "DequantizeLinear %r: expanded initializer %r to 4-D (rank %d→4), axis delta=%d.",
        node.name,
        node.input[0],
        orig_rank,
        delta,
    )
    return None  # no structural node replacement required


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


def _split_graph_output_squeezes(graph: onnx.GraphProto) -> None:
    """Ensure Squeeze nodes whose output is a graph output are not shared with internal consumers.

    When a Squeeze node produces a tensor that is **both** a graph output **and**
    an input to other nodes, we need two separate Squeeze nodes:

    * **Graph-output Squeeze** keeps the original output name so the graph output
      binding is preserved.
    * **Internal Squeeze** gets a new name ``<orig>_sq`` so downstream compute
      nodes are not coupled to the graph output tensor.  A ``value_info`` entry
      with the same element-type and shape is added for this new tensor.

    Consumer nodes are updated to reference the new ``<orig>_sq`` name.
    """
    graph_output_names = {o.name for o in graph.output}

    # Build consumer map: tensor_name → [(node, input_index), ...]
    consumer_map: dict[str, list] = {}
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp:
                consumer_map.setdefault(inp, []).append((node, i))

    # Collect dtype + shape for each graph output
    go_meta: dict[str, tuple] = {}  # name → (elem_type, shape_list_or_None)
    for go in graph.output:
        elem_type = go.type.tensor_type.elem_type
        shape = None
        if go.type.tensor_type.HasField("shape"):
            shape = []
            for d in go.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                elif d.HasField("dim_param"):
                    shape.append(d.dim_param)  # keep symbolic
                else:
                    shape.append(None)
        go_meta[go.name] = (elem_type, shape)

    to_split: list[tuple] = []  # (squeeze_node, orig_out, internal_consumers)

    for node in list(graph.node):
        if node.op_type != "Squeeze":
            continue
        if not node.output:
            continue
        orig_out = node.output[0]
        if orig_out not in graph_output_names:
            continue

        # Find node consumers (not graph-output references — those are implicit)
        internal = consumer_map.get(orig_out, [])
        if not internal:
            continue  # only a graph output, no internal consumers → nothing to split

        to_split.append((node, orig_out, internal))

    sq_node_idx = {id(n): i for i, n in enumerate(graph.node)}
    sq_batch: dict[int, tuple[list, list]] = {}

    for sq_node, orig_out, internal_consumers in to_split:
        internal_name = f"{orig_out}_sq"
        elem_type, shape = go_meta[orig_out]

        # New Squeeze: same 4D input and same axes as the original squeeze,
        # but produces the internal tensor name.
        new_sq = helper.make_node(
            "Squeeze",
            inputs=list(sq_node.input),  # [4d_tensor, axes_init]
            outputs=[internal_name],
            name=f"{sq_node.name}_internal",
        )

        # Redirect all internal consumers to the new tensor name
        for c_node, c_idx in internal_consumers:
            c_node.input[c_idx] = internal_name

        # Record replacement: keep original squeeze + add new internal squeeze
        sq_batch[sq_node_idx[id(sq_node)]] = ([sq_node], [sq_node, new_sq])

        # Add value_info for the new tensor (same type/shape as the graph output)
        if shape is not None and all(d is not None for d in shape):
            vi = helper.make_tensor_value_info(internal_name, elem_type, shape)
        else:
            # Partial or symbolic shape — create unranked value_info
            vi = onnx.helper.make_tensor_value_info(internal_name, elem_type, None)
        graph.value_info.append(vi)

        logger.debug(
            "Split squeeze %r: graph output keeps %r, internal consumers use %r.",
            sq_node.name,
            orig_out,
            internal_name,
        )

    batch_replace_nodes(graph, sq_batch)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------


def _apply_convert_ops_to_4d(model: onnx.ModelProto) -> onnx.ModelProto:
    """Wrap each non-4D op with Unsqueeze/Squeeze so all data tensors become 4D."""
    graph = model.graph
    unique_init: set = set()

    # ── Build graph cache for O(1) lookups ────────────────────────────────────
    cache = GraphCache(graph)
    shape_map = _CacheShapeProxy(cache)
    init_names = cache.init_names

    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    inits_to_add: list = []
    replacements_batch: dict[int, tuple[list, list]] = {}
    cnt_ref = [0]

    # ── Process each node ─────────────────────────────────────────────────────
    for node in list(graph.node):
        if node.op_type in _OP_SKIP:
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

        handler = _OP_HANDLERS.get(node.op_type, _handle_generic)
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
    _split_graph_output_squeezes(graph)

    logger.debug("convert_ops_to_4d: transformed %d node(s).", len(replacements_batch))
    return model


def dla_convert_ops_to_4d(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply convert-ops-to-4D transform, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_convert_ops_to_4d,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
