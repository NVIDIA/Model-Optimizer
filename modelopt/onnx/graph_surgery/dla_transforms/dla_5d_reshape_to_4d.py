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

"""5D ``Reshape`` → 4D subgraph rewrites for DLA-style graphs.

:func:`_apply_5d_reshape_to_4d` runs ORT symbolic shape inference first
(unless ``skip_symbolic_shape_inference``), then classifies every 5D ``Reshape`` follower chain and applies
all subgraph replacements in discovery order.

Reduce followers use merge-(n-1,n) → ``Slice`` x N → optional ``Clip`` / Q→DQ **per branch** →
per-branch reduce (axes ``[n-1]``, ``keepdims=1``) → ``Concat``, matching
``scratch_space/verify_reduce_5d_vs_4d_decompose.py``. Reduce axis ``n`` must be ``1..4`` (initializer);
axis ``0`` is rejected. ``keepdims=1`` on the original op adds ``Unsqueeze`` at axis ``n``.

Unsupported followers raise ``ValueError`` with ``op_type`` and node ``name``.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    batch_replace_nodes,
    resolve_reshape_shape,
    run_onnx_file_transform,
)

_REDUCE_OPS = frozenset({"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"})


def _consumers_first_input(graph, tensor_name: str) -> list:
    return [n for n in graph.node if n.input and len(n.input) > 0 and n.input[0] == tensor_name]


def _parse_middleware_strict(
    graph, start_out: str, stop_op_types: frozenset, *, path_context: str
) -> tuple:
    """Walk ``Clip`` / Q→DQ middleware then a terminal op; raise ``ValueError`` on dead ends or bad ops."""
    middle: list = []
    cur = start_out
    while True:
        cands = _consumers_first_input(graph, cur)
        if not cands:
            msg = (
                f"{path_context}: no consumer for tensor {cur!r}; "
                f"expected middleware (Clip / Q→DQ) or terminal in {sorted(stop_op_types)}"
            )
            raise ValueError(msg)
        if len(cands) != 1:
            msg = (
                f"{path_context}: expected a single first-input consumer for tensor {cur!r}, "
                f"got {[n.name for n in cands]}"
            )
            raise ValueError(msg)
        n = cands[0]
        if n.op_type in stop_op_types:
            return middle, n
        if n.op_type == "Clip":
            middle.append(n)
            cur = n.output[0]
            continue
        if n.op_type == "QuantizeLinear":
            dq_nodes = [
                x
                for x in _consumers_first_input(graph, n.output[0])
                if x.op_type == "DequantizeLinear"
            ]
            if not dq_nodes:
                msg = (
                    f"{path_context}: QuantizeLinear node {n.name!r} (op_type={n.op_type!r}) "
                    f"has no DequantizeLinear consumer"
                )
                raise ValueError(msg)
            dq = dq_nodes[0]
            middle.extend([n, dq])
            cur = dq.output[0]
            continue
        msg = (
            f"{path_context}: unsupported follower op_type={n.op_type!r} name={n.name!r}; "
            f"allowed middleware: Clip, QuantizeLinear→DequantizeLinear; "
            f"terminals: {', '.join(sorted(stop_op_types))}"
        )
        raise ValueError(msg)


def _is_5d_reshape_candidate(cache: GraphCache, node) -> bool:
    if node.op_type != "Reshape":
        return False
    ini = cache.get_init(node.input[1])
    assert ini is not None, f"initializer for node {node.name} is None"
    shape_vals = numpy_helper.to_array(ini)
    return len(shape_vals) == 5


def _classify_5d_reshape_follower(model, reshape_node) -> tuple:
    r"""Return (``\"transpose\"``, mid_a, transpose_node) or (``\"reduce\"``, middle, reduce_node)."""
    graph = model.graph
    ctx = f"5D Reshape {reshape_node.name!r}"
    r_out = reshape_node.output[0]
    stop = frozenset({"Transpose", *_REDUCE_OPS})
    middle, term = _parse_middleware_strict(graph, r_out, stop, path_context=ctx)
    if term.op_type == "Transpose":
        return ("transpose", middle, term)
    elif term.op_type in _REDUCE_OPS:
        return ("reduce", middle, term)
    else:
        msg = f"unsupported terminal op {term.op_type!r} in 5D Reshape {reshape_node.name!r}"
        raise ValueError(msg)


def _duplicate_clip_qdq_chain(middleware: list, x_in: str, suffix: str) -> tuple[list, str]:
    """Rebuild Clip / Q→DQ sequence on ``x_in`` with fresh tensor names. Returns ``(new_nodes, output_name)``."""
    if not middleware:
        return [], x_in
    out = x_in
    new_nodes: list = []
    i = 0
    m = 0
    while m < len(middleware):
        n = middleware[m]
        if n.op_type == "Clip":
            new_out = f"_u5d_{suffix}_c{m}_{i}"
            inputs = [out, *list(n.input[1:])]
            new_nodes.append(
                helper.make_node("Clip", inputs, [new_out], name=f"{n.name}_u5d_{suffix}_{m}")
            )
            out = new_out
            m += 1
            i += 1
        elif n.op_type == "QuantizeLinear":
            if m + 1 >= len(middleware) or middleware[m + 1].op_type != "DequantizeLinear":
                msg = "QuantizeLinear without following DequantizeLinear in middleware chain"
                raise ValueError(msg)
            q, dq = middleware[m], middleware[m + 1]
            q_out = f"_u5d_{suffix}_q{m}_{i}"
            dq_out = f"_u5d_{suffix}_dq{m}_{i}"
            new_nodes.append(
                helper.make_node(
                    "QuantizeLinear",
                    [out, q.input[1], q.input[2]],
                    [q_out],
                    name=f"{q.name}_u5d_{suffix}_{m}",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [q_out, dq.input[1], dq.input[2]],
                    [dq_out],
                    name=f"{dq.name}_u5d_{suffix}_{m}",
                )
            )
            out = dq_out
            m += 2
            i += 1
        else:
            msg = f"unsupported middleware op {n.op_type!r} in unified 5D reshape path"
            raise ValueError(msg)
    return new_nodes, out


def _infer_transpose_path_slice_axis(
    cache: GraphCache, input_tensor: str, num_slices: int, *, ctx: str
) -> int:
    """Pick Slice axis from static input shape when exactly one dim equals the 5D leading extent."""
    if num_slices <= 1:
        return 0
    in_shape = cache.get_shape(input_tensor)
    if in_shape is None:
        return 0
    matches = [i for i, d in enumerate(in_shape) if d != 0 and int(d) == int(num_slices)]
    if not matches:
        return 0
    if len(matches) > 1:
        msg = (
            f"{ctx}: cannot infer Slice axis: input shape {in_shape} has multiple dims equal to "
            f"first 5D reshape extent {num_slices}"
        )
        raise ValueError(msg)
    return matches[0]


def _transpose_path_concat_axis_4d(slice_axis: int, transpose_perm: list[int]) -> int:
    """Concat axis for 4D branch outputs (scratch: axis 0 when perm[0]==0 and input is sliced on axis 0)."""
    if slice_axis != 0:
        msg = (
            "5D reshape→transpose path: non-zero Slice axis is not supported "
            f"(slice_axis={slice_axis}); only leading batch-dim slicing is supported"
        )
        raise ValueError(msg)
    if int(transpose_perm[0]) != 0:
        raise ValueError("internal: concat axis derivation requires transpose perm[0] == 0")
    return 0


def _build_transpose_replacement_or_raise(
    model, reshape_node, mid_a, tr, unique_initializer: set, counter: int, cache: GraphCache
) -> tuple[list, list]:
    """Slice-based decomposition aligned with scratch_space ``transform_reshape_transpose_reshape``.

    Shapes and permutation come from graph initializers: slice count ``shape1[0]``, 4D reshape ``shape1[1:]``,
    final reshape ``shape2`` (any rank), 4D transpose perm ``perm[1:]-1``. Optional ``Clip`` / Q→DQ middleware is
    duplicated per branch.
    """
    graph = model.graph
    ctx = f"5D Reshape {reshape_node.name!r} → Transpose {tr.name!r}"
    transpose_perm = None
    for attr in tr.attribute:
        if attr.name == "perm":
            transpose_perm = [int(x) for x in attr.ints]
            break
    if transpose_perm is None:
        msg = f"{ctx}: missing `perm` attribute on Transpose"
        raise ValueError(msg)
    if transpose_perm[0] != 0:
        msg = (
            f"{ctx}: Transpose perm[0] must be 0, got perm={transpose_perm} "
            f"(op_type={tr.op_type!r}, name={tr.name!r})"
        )
        raise ValueError(msg)
    if "_transformed" in tr.name:
        msg = f"{ctx}: Transpose node name contains '_transformed' ({tr.name!r})"
        raise ValueError(msg)

    tr_out = tr.output[0]
    mid_b, r2 = _parse_middleware_strict(graph, tr_out, frozenset({"Reshape"}), path_context=ctx)
    if "_transformed" in r2.name:
        msg = f"{ctx}: final Reshape name contains '_transformed' ({r2.name!r})"
        raise ValueError(msg)
    ini2 = cache.get_init(r2.input[1])
    assert ini2 is not None, f"initializer for node {r2.name} is None"
    ini1 = cache.get_init(reshape_node.input[1])
    assert ini1 is not None, f"initializer for node {reshape_node.name} is None"

    shape1_raw = np.asarray(numpy_helper.to_array(ini1), dtype=np.int64).reshape(-1)
    shape2 = np.asarray(numpy_helper.to_array(ini2), dtype=np.int64).reshape(-1)
    if shape1_raw.size != 5:
        msg = (
            f"{ctx}: first Reshape shape initializer must have length 5, got {shape1_raw.tolist()}"
        )
        raise ValueError(msg)
    # Resolve -1/0 in the shape using the actual input tensor shape
    input_tensor_tr = reshape_node.input[0]
    tr_in_shape = cache.get_shape(input_tensor_tr)
    if tr_in_shape is not None and (-1 in shape1_raw or 0 in shape1_raw):
        shape1 = np.array(resolve_reshape_shape(shape1_raw, list(tr_in_shape)), dtype=np.int64)
    else:
        shape1 = shape1_raw
    num_slices = int(shape1[0])
    reshape_4d_vals = shape1[1:]
    if reshape_4d_vals.size != 4:
        msg = f"{ctx}: expected four tail dims after leading slice count, got {reshape_4d_vals.tolist()}"
        raise ValueError(msg)

    new_transpose_perm = [transpose_perm[i + 1] - 1 for i in range(4)]

    input_tensor = reshape_node.input[0]
    output_name = r2.output[0]
    slice_axis = _infer_transpose_path_slice_axis(cache, input_tensor, num_slices, ctx=ctx)
    concat_axis = _transpose_path_concat_axis_4d(slice_axis, transpose_perm)

    slice_axes = helper.make_tensor(
        name=f"u5d_tr_slice_axes_{counter}",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[slice_axis],
    )
    slice_starts = []
    slice_ends = []
    for i in range(num_slices):
        slice_starts.append(
            helper.make_tensor(
                name=f"u5d_tr_{i}_starts_{counter}", data_type=TensorProto.INT64, dims=[1], vals=[i]
            )
        )
        slice_ends.append(
            helper.make_tensor(
                name=f"u5d_tr_{i}_ends_{counter}",
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[i + 1],
            )
        )
    add_unique_initializers(graph, unique_initializer, [slice_axes, *slice_starts, *slice_ends])

    reshape_4d_shape = helper.make_tensor(
        name=f"u5d_tr_r4_{counter}",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=reshape_4d_vals.tolist(),
    )
    final_rank = int(shape2.size)
    reshape_final_shape = helper.make_tensor(
        name=f"u5d_tr_r1_{counter}",
        data_type=TensorProto.INT64,
        dims=[final_rank],
        vals=shape2.tolist(),
    )
    add_unique_initializers(graph, unique_initializer, [reshape_4d_shape, reshape_final_shape])

    nodes_to_add: list = []
    slice_outputs: list[str] = []
    for i in range(num_slices):
        slice_out = f"u5d_tr_slice_{i}_{counter}"
        nodes_to_add.append(
            helper.make_node(
                "Slice",
                [
                    input_tensor,
                    f"u5d_tr_{i}_starts_{counter}",
                    f"u5d_tr_{i}_ends_{counter}",
                    f"u5d_tr_slice_axes_{counter}",
                ],
                [slice_out],
                name=f"u5d_tr_slice_{i}_{counter}",
            )
        )
        reshape_4d_out = f"u5d_tr_r4o_{i}_{counter}"
        nodes_to_add.append(
            helper.make_node(
                "Reshape",
                [slice_out, f"u5d_tr_r4_{counter}"],
                [reshape_4d_out],
                name=f"u5d_tr_r4n_{i}_{counter}",
            )
        )
        br_a, t_a = _duplicate_clip_qdq_chain(mid_a, reshape_4d_out, f"tr{counter}_{i}a")
        nodes_to_add.extend(br_a)
        transpose_branch_out = f"u5d_tr_t_{i}_{counter}"
        nodes_to_add.append(
            helper.make_node(
                "Transpose",
                [t_a],
                [transpose_branch_out],
                perm=new_transpose_perm,
                name=f"u5d_tr_tn_{i}_{counter}",
            )
        )
        br_b, t_b = _duplicate_clip_qdq_chain(mid_b, transpose_branch_out, f"tr{counter}_{i}b")
        nodes_to_add.extend(br_b)
        slice_outputs.append(t_b)

    concat_out = f"u5d_tr_concat_{counter}"
    nodes_to_add.append(
        helper.make_node(
            "Concat",
            slice_outputs,
            [concat_out],
            axis=concat_axis,
            name=f"u5d_tr_concatn_{counter}",
        )
    )
    nodes_to_add.append(
        helper.make_node(
            "Reshape",
            [concat_out, f"u5d_tr_r1_{counter}"],
            [output_name],
            name=f"u5d_tr_final_{counter}",
        )
    )
    to_remove = [reshape_node, *mid_a, tr, *mid_b, r2]
    return to_remove, nodes_to_add


def _build_reduce_replacement_or_raise(
    model, reshape_node, middle, red, unique_initializer: set, counter: int, cache: GraphCache
) -> tuple[list, list]:
    """Replace 5D Reshape → … → Reduce with merge-(n-1,n) → Slice x N → optional Clip/Q→DQ x N → Reduce x N → Concat.

    ``Clip`` / Q→DQ middleware is duplicated **per branch** (after each Slice, before each Reduce)
    so that quantisation is applied to each chunk individually, preserving the original semantics.

    Matches ``scratch_space/verify_reduce_5d_vs_4d_decompose.py`` for reduce axis ``n`` in ``{1,2,3,4}``.
    Axis ``0`` is not supported (no observed models); raises ``ValueError``.
    """
    graph = model.graph
    ctx = f"5D Reshape {reshape_node.name!r} → {red.op_type} {red.name!r}"
    if len(red.input) < 2:
        msg = f"{ctx}: missing axes input (op_type={red.op_type!r})"
        raise ValueError(msg)

    axes_name = red.input[1]
    ini_axes = cache.get_init(axes_name)
    if ini_axes is None:
        msg = f"{ctx}: reduce axes {axes_name!r} must be a graph initializer for decomposition"
        raise ValueError(msg)
    axes_arr = np.asarray(numpy_helper.to_array(ini_axes), dtype=np.int64).reshape(-1)
    if axes_arr.size != 1:
        msg = f"{ctx}: expected a single reduce axis, got {axes_arr.tolist()}"
        raise ValueError(msg)
    n = int(axes_arr[0])
    if n < 0:
        n += 5
    if n <= 0 or n >= 5:
        msg = (
            f"{ctx}: reduce axis must be 1..4 on the 5D Reshape target (axis 0 not supported), "
            f"got raw initializer {numpy_helper.to_array(ini_axes).tolist()}"
        )
        raise ValueError(msg)

    ini1 = cache.get_init(reshape_node.input[1])
    assert ini1 is not None, f"initializer for node {reshape_node.name} is None"
    shape1_raw = np.asarray(numpy_helper.to_array(ini1), dtype=np.int64).reshape(-1)
    if shape1_raw.size != 5:
        msg = f"{ctx}: first Reshape shape must have length 5, got {shape1_raw.tolist()}"
        raise ValueError(msg)

    input_tensor = reshape_node.input[0]
    reshape_in_shape = cache.get_shape(input_tensor)
    if reshape_in_shape is None:
        msg = f"{ctx}: shape of Reshape input {input_tensor!r} is unknown"
        raise ValueError(msg)

    # Resolve -1/0 in the shape using the actual input tensor shape
    if -1 in shape1_raw or 0 in shape1_raw:
        shape1 = np.array(resolve_reshape_shape(shape1_raw, list(reshape_in_shape)), dtype=np.int64)
    else:
        shape1 = shape1_raw

    r = [int(x) for x in shape1.tolist()]
    split_ax = n - 1
    num_splits = r[n - 1]
    chunk = r[n]
    merged = [*r[:split_ax], r[n - 1] * r[n], *r[n + 1 :]]
    if num_splits <= 0 or chunk <= 0:
        msg = f"{ctx}: invalid 5D shape {r}"
        raise ValueError(msg)

    if 0 not in reshape_in_shape and -1 not in r:
        prod_in = int(np.prod(reshape_in_shape))
        prod_r = int(np.prod(r))
        if prod_in != prod_r:
            msg = (
                f"{ctx}: Reshape input element count {prod_in} != product of 5D shape {prod_r} "
                f"(input shape {list(reshape_in_shape)}, 5D {r})"
            )
            raise ValueError(msg)

    output_name = red.output[0]
    op_type = red.op_type
    reduce_kw: dict = {}
    for a in red.attribute:
        if a.name == "keepdims":
            reduce_kw["keepdims"] = a.i
        elif a.name == "noop_with_empty_axes":
            reduce_kw["noop_with_empty_axes"] = a.i

    original_keepdims = bool(reduce_kw.get("keepdims", 1))  # ONNX default: 1
    reduce_branch_kw = dict(reduce_kw)
    reduce_branch_kw["keepdims"] = 1

    msh_name = f"u5d_rs_msh_{counter}"
    ax4_name = f"u5d_rs_ax4_{counter}"
    merged_shape_ini = helper.make_tensor(
        name=msh_name, data_type=TensorProto.INT64, dims=[4], vals=merged
    )
    # Use negative axis so the reduce op is robust to leading-dim padding.
    # For a 4-D tensor, axis k is equivalent to axis k-4 (e.g. 3 → -1).
    axes_4d_ini = helper.make_tensor(
        name=ax4_name, data_type=TensorProto.INT64, dims=[1], vals=[split_ax - 4]
    )
    add_unique_initializers(
        graph,
        unique_initializer,
        [merged_shape_ini, axes_4d_ini],
    )

    merged_out = f"u5d_rs_m4_{counter}"
    nodes_to_add: list = []
    nodes_to_add.append(
        helper.make_node(
            "Reshape",
            [input_tensor, msh_name],
            [merged_out],
            name=f"u5d_rs_reshape4_{counter}",
        )
    )

    # Replace Split with individual Slice nodes — one per chunk along split_ax.
    # Middleware (Clip / Q→DQ) is applied per-branch after each Slice, before the Reduce.
    sl_axes_name = f"u5d_rs_slax_{counter}"
    sl_axes_ini = helper.make_tensor(
        name=sl_axes_name, data_type=TensorProto.INT64, dims=[1], vals=[split_ax]
    )
    add_unique_initializers(graph, unique_initializer, [sl_axes_ini])

    reduce_outs: list[str] = []
    for i in range(num_splits):
        start = i * chunk
        end = start + chunk
        sl_starts_name = f"u5d_rs_slst_{i}_{counter}"
        sl_ends_name = f"u5d_rs_slen_{i}_{counter}"
        sl_starts_ini = helper.make_tensor(
            name=sl_starts_name, data_type=TensorProto.INT64, dims=[1], vals=[start]
        )
        sl_ends_ini = helper.make_tensor(
            name=sl_ends_name, data_type=TensorProto.INT64, dims=[1], vals=[end]
        )
        add_unique_initializers(graph, unique_initializer, [sl_starts_ini, sl_ends_ini])
        sl_out = f"u5d_rs_sp_{i}_{counter}"
        nodes_to_add.append(
            helper.make_node(
                "Slice",
                [merged_out, sl_starts_name, sl_ends_name, sl_axes_name],
                [sl_out],
                name=f"u5d_rs_slice_{i}_{counter}",
            )
        )
        # Apply per-branch middleware (Clip / Q→DQ) between Slice and Reduce
        br_nodes, reduce_in = _duplicate_clip_qdq_chain(middle, sl_out, f"rs{counter}_{i}")
        nodes_to_add.extend(br_nodes)

        rs_out = f"u5d_rs_o_{i}_{counter}"
        nodes_to_add.append(
            helper.make_node(
                op_type,
                [reduce_in, ax4_name],
                [rs_out],
                name=f"u5d_rs_{op_type}_{i}_{counter}",
                **reduce_branch_kw,
            )
        )
        reduce_outs.append(rs_out)

    if original_keepdims:
        concat_out = f"u5d_rs_cat_{counter}"
        usq_name = f"u5d_rs_usq_{counter}"
        unsq_ini = helper.make_tensor(
            name=usq_name, data_type=TensorProto.INT64, dims=[1], vals=[n]
        )
        add_unique_initializers(graph, unique_initializer, [unsq_ini])
        nodes_to_add.append(
            helper.make_node(
                "Concat",
                reduce_outs,
                [concat_out],
                axis=split_ax,
                name=f"u5d_rs_concat_{counter}",
            )
        )
        nodes_to_add.append(
            helper.make_node(
                "Unsqueeze",
                [concat_out, usq_name],
                [output_name],
                name=f"u5d_rs_unsqueeze_{counter}",
            )
        )
    else:
        nodes_to_add.append(
            helper.make_node(
                "Concat",
                reduce_outs,
                [output_name],
                axis=split_ax,
                name=f"u5d_rs_concat_{counter}",
            )
        )

    to_remove = [reshape_node, *middle, red]
    return to_remove, nodes_to_add


def _dispatch_5d_reshape_candidate(
    model, reshape_node, unique_initializer: set, counter: int, cache: GraphCache
) -> tuple[list, list]:
    """Classify a single 5D Reshape and return ``(nodes_to_remove, nodes_to_add)``."""
    branch, a, b = _classify_5d_reshape_follower(model, reshape_node)
    if branch == "transpose":
        return _build_transpose_replacement_or_raise(
            model, reshape_node, a, b, unique_initializer, counter, cache
        )
    return _build_reduce_replacement_or_raise(
        model, reshape_node, a, b, unique_initializer, counter, cache
    )


def _apply_5d_reshape_to_4d(model, *, skip_symbolic_shape_inference: bool = False):
    """Parse the graph without mutating it, then replace every matching 5D ``Reshape`` subgraph.

    Unless ``skip_symbolic_shape_inference`` is true, runs
    :func:`~modelopt.onnx.graph_surgery.utils.symbolic_shape_inference.run_ort_symbolic_shape_inference`
    first so intermediate tensor shapes are available where needed.

    Phase 1 walks a snapshot of ``graph.node``, dispatches each 5D ``Reshape``, and records
    ``(nodes_to_remove, nodes_to_add)``. Phase 2 runs :func:`replace_nodes_with_topological_order`
    for each plan in that same order.

    Classification: after optional ``Clip`` / Q→DQ middleware, the first terminal must be ``Transpose``
    (reshape→transpose→reshape pipeline) or a reduce op in :data:`_REDUCE_OPS`. Any other op raises
    ``ValueError`` with ``op_type`` and node ``name``. Pattern validation errors include the same.
    """
    graph = model.graph
    cache = GraphCache(graph)
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    unique_initializer: set = set()
    counter = 0
    batch: dict[int, tuple[list, list]] = {}

    for node in list(graph.node):
        if not _is_5d_reshape_candidate(cache, node):
            continue
        nodes_to_remove, nodes_to_add = _dispatch_5d_reshape_candidate(
            model, node, unique_initializer, counter, cache
        )
        idx = node_idx[id(nodes_to_remove[0])] if nodes_to_remove else len(graph.node)
        batch[idx] = (nodes_to_remove, nodes_to_add)
        counter += 1

    batch_replace_nodes(graph, batch)

    logger.debug(
        "5D reshape to 4D: classified and rewrote %d 5D Reshape subgraph(s) (transpose or reduce path)",
        len(batch),
    )

    return model


def dla_5d_reshape_to_4d(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    skip_symbolic_shape_inference: bool = False,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX, optional ORT symbolic shape inference, run 5D reshape→4D passes, save.

    Pass ``skip_symbolic_shape_inference=True`` to skip
    ``onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference`` (e.g. minimal env without ORT).
    """
    skip = kwargs.pop("skip_symbolic_shape_inference", skip_symbolic_shape_inference)
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_5d_reshape_to_4d,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        skip_symbolic_shape_inference=skip,
        **kwargs,
    )
