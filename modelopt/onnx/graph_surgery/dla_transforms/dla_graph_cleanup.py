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

"""Combined graph-cleanup pass for DLA ONNX models.

Applies five cleanup steps in sequence:

1. **Canonicalize non-4D model inputs** — Each 1-D/2-D/3-D graph input gets a single
   canonical ``Unsqueeze`` inserted at the front of the graph, padding to 4-D.
   Existing ``Unsqueeze`` consumers of that input are either deduped (same output shape)
   or replaced by a ``Reshape`` (different output shape).  All other consumers are
   redirected to the canonical Unsqueeze output.

2. **Replace intermediary Squeeze / Unsqueeze with 4-D Reshape** — Any ``Squeeze`` or
   ``Unsqueeze`` whose output is **not** a graph output is replaced by a ``Reshape`` whose
   output shape is the original output shape padded to 4-D by prepending 1-s.
   If the padded input and output shapes are identical the node is removed entirely
   and consumers are wired to the input directly.

3. **Collapse Reshape chains** — Consecutive ``Reshape`` nodes whose output is consumed
   only by the next ``Reshape`` are collapsed.  A chain where first-input-shape equals
   last-output-shape is removed entirely; otherwise it is reduced to a single ``Reshape``.

4. **Fold constants** — Uses ``onnx-graphsurgeon`` to evaluate constant subgraphs at
   compile time, skipping ``DequantizeLinear`` and ``Identity`` nodes.  Silently skipped
   if ``onnx-graphsurgeon`` is unavailable.

5. **Remove unused initializers** — Initializers that are not referenced as a node input
   are dropped.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    add_value_info,
    batch_replace_nodes,
    get_node_attr_i,
    infer_shapes,
    insert_nodes_at_position,
    pad4d,
    run_onnx_file_transform,
)
from ._dla_graph_helpers import update_tensor_shape_by_name

# ── Internal utilities ────────────────────────────────────────────────────────


def _get_shape(graph: onnx.GraphProto, name: str) -> list[int] | None:
    """Return the concrete shape of tensor ``name`` as a list of ints, or ``None``."""
    # Check value_info, graph inputs, graph outputs
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return None
            shape: list[int] = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    return None  # symbolic / unknown dim
            return shape
    # Initializer dims are always concrete
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


# ── Step 1: Canonicalize non-4D model inputs ─────────────────────────────────


def _canonicalize_inputs(model: onnx.ModelProto, cache: GraphCache) -> onnx.ModelProto:
    """Insert one canonical Unsqueeze per non-4D graph input; dedup existing Unsqueeze consumers."""
    graph = model.graph
    unique_init: set[str] = set()
    consumer_map = cache.consumer_map
    nodes_to_insert: list[tuple] = []  # (new_node, reference_node)
    nodes_to_remove: list = []
    cnt = 0

    for graph_input in list(graph.input):
        tt = graph_input.type.tensor_type
        if not tt.HasField("shape"):
            continue
        rank = sum(1 for _ in tt.shape.dim)
        if rank == 4:
            continue
        if rank not in (1, 2, 3):
            logger.debug(
                "_canonicalize_inputs: skipping input %r with rank %d.", graph_input.name, rank
            )
            continue

        # Read concrete input shape (may be None for symbolic dims)
        input_shape: list[int | None] = [
            d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim
        ]

        n_new = 4 - rank
        axes_arr = np.arange(n_new, dtype=np.int64)
        canonical_4d = [1] * n_new + list(input_shape)

        axes_name = f"{graph_input.name}_unsqueeze_axes"
        out_name = f"{graph_input.name}_4d"
        add_unique_initializers(
            graph, unique_init, [numpy_helper.from_array(axes_arr, name=axes_name)]
        )

        unsq = helper.make_node(
            "Unsqueeze",
            inputs=[graph_input.name, axes_name],
            outputs=[out_name],
            name=f"{graph_input.name}_unsqueeze",
        )
        add_value_info(graph, out_name, tt.elem_type, canonical_4d)

        consumers = consumer_map.get(graph_input.name, [])
        consumers = [
            (c_node, c_idx)
            for c_node, c_idx in consumers
            if len(c_node.input) > 0
            and graph_input.name in c_node.input
            and not (
                (c_node.op_type == "Squeeze" and "Gather" in c_node.name)
                or c_node.op_type in ("GroupQueryAttention", "Gather")
            )
        ]
        if not consumers:
            continue

        # Find the first consumer node in graph order for insertion point
        consumer_set = {id(n) for n, _ in consumers}
        ref_node = next((n for n in graph.node if id(n) in consumer_set), None)
        nodes_to_insert.append((unsq, ref_node))

        for c_node, c_idx in consumers:
            if c_node.op_type == "Unsqueeze":
                old_out_shape = cache.get_shape(c_node.output[0])
                if old_out_shape is not None:
                    old_4d = pad4d(old_out_shape)
                    if old_4d == canonical_4d:
                        # Same 4-D shape → redirect downstream consumers, remove old Unsqueeze
                        for dn, di in consumer_map.get(c_node.output[0], []):
                            dn.input[di] = out_name
                    else:
                        # Different 4-D shape → Reshape from canonical output
                        rs_name = f"{c_node.name}_reshape_shape"
                        add_unique_initializers(
                            graph,
                            unique_init,
                            [
                                numpy_helper.from_array(
                                    np.array(old_4d, dtype=np.int64), name=rs_name
                                )
                            ],
                        )
                        rs = helper.make_node(
                            "Reshape",
                            inputs=[out_name, rs_name],
                            outputs=list(c_node.output),
                            name=f"{c_node.name}_reshape",
                        )
                        update_tensor_shape_by_name(model, c_node.output[0], old_4d)
                        nodes_to_insert.append((rs, c_node))
                else:
                    # Unknown shape — just update the input to canonical unsqueeze output
                    c_node.input[0] = out_name
                    continue
                nodes_to_remove.append(c_node)
            else:
                c_node.input[c_idx] = out_name
        cnt += 1
        logger.debug(
            "_canonicalize_inputs: processed graph input %r (rank %d).", graph_input.name, rank
        )

    for new_node, ref in nodes_to_insert:
        insert_nodes_at_position(graph, [new_node], ref)
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    logger.debug("canonicalize_inputs: processed %d non-4D graph input(s).", cnt)
    return model


# ── Step 2: Replace intermediary Squeeze / Unsqueeze with 4-D Reshape ────────


def _compute_sq_unsq_out_shape(node: onnx.NodeProto, graph: onnx.GraphProto) -> list[int] | None:
    """Compute the output shape of a Squeeze/Unsqueeze from its axes and input shape.

    Used as a fallback when shape inference did not populate ``value_info``.
    Returns ``None`` if the shape cannot be determined.
    """
    in_shape = _get_shape(graph, node.input[0])
    if in_shape is None:
        return None

    # Resolve axes from input[1]: initializer or inline Constant node
    axes: list[int] | None = None
    if len(node.input) > 1 and node.input[1]:
        for init in graph.initializer:
            if init.name == node.input[1]:
                axes = numpy_helper.to_array(init).tolist()
                break
        if axes is None:
            for n in graph.node:
                if n.op_type == "Constant" and n.output and n.output[0] == node.input[1]:
                    for attr in n.attribute:
                        if attr.name == "value":
                            axes = numpy_helper.to_array(attr.t).tolist()
                            break
                    break

    if node.op_type == "Squeeze":
        if axes is None:
            return [d for d in in_shape if d != 1]
        out = list(in_shape)
        for ax in sorted([a if a >= 0 else len(out) + a for a in axes], reverse=True):
            if 0 <= ax < len(out):
                out.pop(ax)
        return out
    else:  # Unsqueeze
        if axes is None:
            return None
        out = list(in_shape)
        for ax in sorted([a if a >= 0 else len(out) + a + 1 for a in axes]):
            out.insert(ax, 1)
        return out


_GATHER_OPS = frozenset({"Gather", "GatherElements"})
_GQA_OPS = frozenset({"GroupQueryAttention"})


def _should_skip_squeeze_unsqueeze(
    node: onnx.NodeProto,
    cache: GraphCache,
) -> bool:
    """Return True if this Squeeze/Unsqueeze should be left in place.

    Skip conditions:
    * ``Unsqueeze`` whose data input is a graph input — canonical 4D-promotion node.
    * Output feeds ``Gather`` / ``GatherElements`` ``input[1]`` (indices must stay INT32).
    * Output feeds any ``GroupQueryAttention`` input (GQA manages its own shapes).
    * ``Unsqueeze`` whose data input is a GQA output AND the GQA input was 3D while
      the GQA output is 4D — this Unsqueeze was placed to adapt the 3D→4D boundary
      and must be preserved.

    The check looks one level through an intervening ``Cast`` node since
    ``Squeeze → Cast → Gather/GQA`` also requires the original shape to be preserved.
    """
    if node.op_type == "Unsqueeze" and node.input and node.input[0] in cache.graph_input_names:
        return True

    # Keep Unsqueeze whose input is a GQA output, the GQA output is 3D,
    # and the Unsqueeze output is 4D — this is the canonical 3D→4D bridge
    # after GroupQueryAttention and must be preserved.
    if node.op_type == "Unsqueeze" and node.input and node.output:
        prod = cache.get_producer(node.input[0])
        if prod is not None and prod.op_type in _GQA_OPS:
            gqa_out_shape = cache.get_shape(node.input[0])
            unsq_out_shape = cache.get_shape(node.output[0])
            if (
                gqa_out_shape is not None
                and len(gqa_out_shape) == 3
                and unsq_out_shape is not None
                and len(unsq_out_shape) == 4
            ):
                return True

    for c_node, c_idx in cache.get_consumers(node.output[0]):
        if c_node.op_type in _GATHER_OPS and c_idx == 1 and node.op_type == "Squeeze":
            return True
        if c_node.op_type in _GQA_OPS:
            return True
        # Look one level deeper through Cast
        if c_node.op_type == "Cast" and c_node.output:
            for gc_node, gc_idx in cache.get_consumers(c_node.output[0]):
                if gc_node.op_type in _GATHER_OPS and gc_idx == 1 and node.op_type == "Squeeze":
                    return True
                if gc_node.op_type in _GQA_OPS:
                    return True
    return False


def _replace_intermediary_squeeze_unsqueeze(
    model: onnx.ModelProto, cache: GraphCache
) -> onnx.ModelProto:
    """Replace Squeeze/Unsqueeze nodes with 4-D Reshape nodes.

    Skip conditions:
    * Input is a graph input (Unsqueeze) — canonical 4D-promotion node, must be preserved.
    * Output feeds Gather/GatherElements ``input[1]`` (indices must remain INT32).
    * Output feeds GroupQueryAttention (GQA manages its own dimensionality).
    * Output of GroupQueryAttention feeds to Unsqueeze node with 3D input.
    """
    graph = model.graph
    unique_init: set[str] = set()
    graph_output_names = cache.graph_output_names

    nodes_to_remove: list = []
    nodes_to_insert: list[tuple] = []
    cnt_replace = 0

    for node in list(graph.node):
        if node.op_type not in ("Squeeze", "Unsqueeze"):
            continue
        if not node.input or not node.output:
            continue
        if _should_skip_squeeze_unsqueeze(node, cache):
            logger.debug(
                "%s %r: skipped (output feeds Gather indices, GQA, or follows GQA output).",
                node.op_type,
                node.name,
            )
            continue
        out_shape = cache.get_shape(node.output[0]) or _compute_sq_unsq_out_shape(node, graph)
        if out_shape is None:
            logger.debug("%s %r: output shape unknown, skipping.", node.op_type, node.name)
            continue

        is_graph_output = node.output[0] in graph_output_names

        # Graph output: use declared shape (preserves name binding); else pad to 4D.
        reshape_target = out_shape if is_graph_output else pad4d(out_shape)

        rs_name = f"{node.name}_4d_shape"
        add_unique_initializers(
            graph,
            unique_init,
            [numpy_helper.from_array(np.array(reshape_target, dtype=np.int64), name=rs_name)],
        )
        rs = helper.make_node(
            "Reshape",
            inputs=[node.input[0], rs_name],
            outputs=list(node.output),
            name=f"{node.name}_to_reshape",
        )
        nodes_to_insert.append((rs, node))
        nodes_to_remove.append(node)
        update_tensor_shape_by_name(model, node.output[0], reshape_target)
        cnt_replace += 1
        logger.debug("%s %r: replaced by Reshape → %s.", node.op_type, node.name, reshape_target)

    for new_node, ref in nodes_to_insert:
        insert_nodes_at_position(graph, [new_node], ref)
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    logger.debug("replace_intermediary_sq_unsq: replaced %d node(s) with Reshape.", cnt_replace)
    return model


# ── Step 3: Collapse Reshape chains ──────────────────────────────────────────


def _remove_reshape_chains(model: onnx.ModelProto, cache: GraphCache) -> onnx.ModelProto:
    """Remove noop Reshapes then collapse consecutive Reshape chains.

    Pass 1 — Remove every noop Reshape (``in_shape == out_shape``) completely:
    consumers are wired directly to the Reshape input; at graph-output boundaries
    an ``Identity`` node is inserted to keep the output name alive.

    Pass 2 — Collapse consecutive Reshape chains (each member's output consumed
    only by the next Reshape) into a single Reshape; chains that are noops as
    a whole are removed the same way.
    """
    graph = model.graph
    graph_output_names = cache.graph_output_names
    noop_count = 0
    chain_count = 0
    chain_nodes_count = 0
    # ── Pass 1: Remove standalone noop Reshapes ───────────────────────────────
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    pass1_batch: dict[int, tuple[list, list]] = {}

    for node in cache.nodes_by_op("Reshape"):
        in_shape = cache.get_shape(node.input[0])
        out_shape = cache.get_shape(node.output[0])
        if in_shape is None or out_shape is None:
            continue
        if list(in_shape) != list(out_shape):
            continue
        # Noop reshape: rewire downstream node-consumers to bypass this node
        inp = node.input[0]
        out_tensor = node.output[0]
        for c_node, c_idx in cache.get_consumers(out_tensor):
            if c_node.input[c_idx] == out_tensor:
                c_node.input[c_idx] = inp
        if out_tensor in graph_output_names:
            id_node = helper.make_node(
                "Identity",
                [inp],
                [out_tensor],
                name=f"cleanup_noop_id_{out_tensor}",
            )
            pass1_batch[node_idx[id(node)]] = ([node], [id_node])
        else:
            pass1_batch[node_idx[id(node)]] = ([node], [])
        noop_count += 1

    batch_replace_nodes(graph, pass1_batch)

    # Rebuild index after Pass 1 mutations
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    pass2_batch: dict[int, tuple[list, list]] = {}

    # ── Pass 2: Collapse Reshape chains (Solution 2 — output-to-chain map) ────
    # Build a map: current_tail_output → chain_list.
    # For each Reshape, check if its input tensor is an existing chain tail.
    # If yes (and that Reshape is the sole consumer of that tensor), extend the
    # chain.  If no, start a new chain keyed by this Reshape's output.
    # This correctly handles non-Reshape nodes interspersed between chain
    # members (e.g. r1→r2→r5→r3 where r5 is not a Reshape: r1,r2,r3 still
    # form a chain as long as each Reshape is the sole consumer of the previous
    # Reshape's output).
    all_reshapes = [n for n in graph.node if n.op_type == "Reshape"]

    # output_name → chain (list of Reshape nodes whose last output == output_name)
    tail_to_chain: dict[str, list] = {}

    for node in all_reshapes:
        inp_tensor = node.input[0] if node.input else None
        if inp_tensor and inp_tensor in tail_to_chain:
            # Extend existing chain only if this node is the sole consumer
            consumers = cache.get_consumers(inp_tensor)
            if len(consumers) == 1:
                chain = tail_to_chain.pop(inp_tensor)
                chain.append(node)
                tail_to_chain[node.output[0]] = chain
                continue
        # Start a new singleton chain
        tail_to_chain[node.output[0]] = [node]

    # Flush all chains with length > 1
    for ch in tail_to_chain.values():
        if len(ch) <= 1:
            continue
        in_shape = cache.get_shape(ch[0].input[0])
        out_shape = cache.get_shape(ch[-1].output[0])
        if in_shape is None or out_shape is None:
            continue
        inp = ch[0].input[0]
        out_tensor = ch[-1].output[0]
        if list(in_shape) == list(out_shape):
            # Entire chain is a noop
            for c_node, c_idx in cache.get_consumers(out_tensor):
                if c_node.input[c_idx] == out_tensor:
                    c_node.input[c_idx] = inp
            if out_tensor in graph_output_names:
                id_node = helper.make_node(
                    "Identity",
                    [inp],
                    [out_tensor],
                    name=f"cleanup_chain_id_{out_tensor}",
                )
                pass2_batch[node_idx[id(ch[0])]] = (ch[:], [id_node])
            else:
                pass2_batch[node_idx[id(ch[0])]] = (ch[:], [])
        else:
            rs_shape = numpy_helper.from_array(
                np.array(out_shape, dtype=np.int64),
                name=ch[-1].name + "_chain_shape",
            )
            graph.initializer.append(rs_shape)
            rs = helper.make_node(
                "Reshape",
                inputs=[inp, rs_shape.name],
                outputs=[out_tensor],
                name=ch[-1].name + "_chain",
            )
            pass2_batch[node_idx[id(ch[0])]] = (ch[:], [rs])
        chain_count += 1
        chain_nodes_count += len(ch)

    batch_replace_nodes(graph, pass2_batch)

    logger.debug(
        "remove_reshape_chains: removed %d noop Reshape(s); collapsed %d chain(s) (%d nodes) to single Reshape.",
        noop_count,
        chain_count,
        chain_nodes_count,
    )
    return model


# ── Step 4: Fold constants ────────────────────────────────────────────────────


def _fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold constant-valued subgraphs using onnx-graphsurgeon.

    ``DequantizeLinear`` and ``Identity`` nodes are excluded from folding.
    Returns the (possibly new) model.  If onnx-graphsurgeon is unavailable the
    original model is returned unchanged.
    """
    try:
        import onnx_graphsurgeon as gs
    except ImportError:
        logger.warning("onnx-graphsurgeon not installed; skipping constant folding.")
        return model

    def _exclude(node) -> bool:
        return node.op in ("DequantizeLinear", "Identity")

    prev_ir = onnx.IR_VERSION
    onnx.IR_VERSION = model.ir_version
    try:
        g = gs.import_onnx(model)
        g.fold_constants(should_exclude_node=_exclude).cleanup().toposort()
        model2 = gs.export_onnx(g)
    finally:
        onnx.IR_VERSION = prev_ir

    folded = len(model.graph.node) - len(model2.graph.node)
    if folded > 0:
        logger.debug("fold_constants: folded %d node(s).", folded)
    model2 = infer_shapes(model2)
    return model2


# ── Step 5: Replace graph-output Reshape with Squeeze where possible ─────────


def _replace_graph_output_reshape_with_squeeze(
    model: onnx.ModelProto, cache: GraphCache
) -> onnx.ModelProto:
    """Replace a Reshape that feeds a graph output with an equivalent Squeeze.

    A Reshape ``[*in_shape] → [*out_shape]`` at a graph output can be replaced by
    ``Squeeze(axes)`` if and only if:

    * Both shapes are fully known (no symbolic dims).
    * The output shape is a strict sub-sequence of the input shape obtained by
      removing only size-1 dimensions (i.e. the total element count is unchanged
      and no dimension values change — only unit dims are dropped).

    The squeeze axes are the indices in ``in_shape`` of the 1-dims that are not
    present in ``out_shape``.

    Example::

        [1, 3, 4, 1] → Reshape → [3, 4]   (graph output)
        becomes
        [1, 3, 4, 1] → Squeeze(axes=[0, 3]) → [3, 4]
    """
    graph = model.graph
    unique_init: set[str] = set()
    graph_output_names = cache.graph_output_names

    go_node_idx = {id(n): i for i, n in enumerate(graph.node)}
    go_batch: dict[int, tuple[list, list]] = {}
    cnt = 0

    for node in list(graph.node):
        if node.op_type != "Reshape":
            continue
        if not node.output or node.output[0] not in graph_output_names:
            continue

        in_name = node.input[0]
        out_name = node.output[0]

        in_shape = cache.get_shape(in_name)
        out_shape = cache.get_shape(out_name)

        if in_shape is None or out_shape is None:
            continue  # shapes unknown — can't reason
        if None in in_shape or None in out_shape:
            continue  # symbolic dims — skip

        # Verify element count is unchanged
        in_elems = 1
        for d in in_shape:
            in_elems *= d
        out_elems = 1
        for d in out_shape:
            out_elems *= d
        if in_elems != out_elems:
            continue

        # Check that out_shape is a subsequence of in_shape with only 1-dims removed.
        # Match from RIGHT-TO-LEFT so that the rightmost (non-unit) dims are anchored
        # first.  The leftmost 1-dims that cannot be matched become the squeeze axes,
        # guaranteeing we always squeeze from the lowest axis indices (left to right).
        #
        # Example: [1,1,1,512] → [1,512]
        #   Right-to-left: 512 matches, then 1 matches, leaving leading [1,1] → axes=[0,1]
        #   (The old left-to-right approach would yield axes=[1,2] instead.)
        squeeze_axes: list[int] = []
        j = len(out_shape) - 1  # pointer into out_shape, going backwards
        matched = True
        for i in range(len(in_shape) - 1, -1, -1):
            d = in_shape[i]
            if j >= 0 and d == out_shape[j]:
                j -= 1  # matched this dim from the right
            elif d == 1:
                squeeze_axes.append(i)  # unit dim to drop
            else:
                matched = False
                break
        pfx = node.name or f"go_sq_{cnt}"

        if not matched or j != -1:
            # Cannot express as a pure Squeeze of the existing input.
            # If the output is non-4D, pad the Reshape target to 4D with a
            # leading 1-dim and add a Squeeze(axis=[0]) to strip it back.
            #
            # Example:
            #   [5,6,7,8] → Reshape([30,14,4]) → graph_output[30,14,4]
            # becomes:
            #   [5,6,7,8] → Reshape([1,30,14,4]) → Squeeze([0]) → graph_output[30,14,4]
            if len(out_shape) >= 4:
                continue  # already 4D or higher — leave as-is
            # Build a 4D shape by prepending 1
            shape_4d = [1, *out_shape]
            rs_shape_name = f"{pfx}_4d_shape"
            add_unique_initializers(
                graph,
                unique_init,
                [numpy_helper.from_array(np.array(shape_4d, dtype=np.int64), name=rs_shape_name)],
            )
            # Intermediate 4D tensor name
            rs_4d_out = f"{pfx}_4d_out"
            # Update the Reshape node to target the 4D shape
            node.input[1] = rs_shape_name
            node.output[0] = rs_4d_out
            # Squeeze to remove the leading 1-dim
            sq_axes_name = f"{pfx}_sq0_axes"
            add_unique_initializers(
                graph,
                unique_init,
                [numpy_helper.from_array(np.array([0], dtype=np.int64), name=sq_axes_name)],
            )
            sq = helper.make_node(
                "Squeeze",
                inputs=[rs_4d_out, sq_axes_name],
                outputs=[out_name],
                name=f"{pfx}_squeeze_leading",
            )
            go_batch[go_node_idx[id(node)]] = ([node], [node, sq])
            cnt += 1
            logger.debug(
                "Reshape %r → Reshape(%s) + Squeeze([0]) for non-4D graph output %r.",
                node.name,
                shape_4d,
                out_name,
            )
            continue

        squeeze_axes.sort()  # ascending axis order required by ONNX Squeeze

        # Build Squeeze node
        axes_name = f"{pfx}_squeeze_axes"
        add_unique_initializers(
            graph,
            unique_init,
            [numpy_helper.from_array(np.array(squeeze_axes, dtype=np.int64), name=axes_name)],
        )
        sq = helper.make_node(
            "Squeeze",
            inputs=[in_name, axes_name],
            outputs=[out_name],
            name=f"{pfx}_squeeze",
        )
        go_batch[go_node_idx[id(node)]] = ([node], [sq])
        cnt += 1
        logger.debug(
            "Reshape %r → Squeeze(axes=%s) for graph output %r.",
            node.name,
            squeeze_axes,
            out_name,
        )

    batch_replace_nodes(graph, go_batch)

    logger.debug("replace_graph_output_reshape_with_squeeze: replaced %d Reshape(s).", cnt)
    return model


# ── Step 7: Remove unused initializers ───────────────────────────────────────


def _remove_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop initializers that are not referenced by any node input."""
    graph = model.graph
    used: set[str] = set()
    for node in graph.node:
        used.update(inp for inp in node.input if inp)
    # Graph outputs are kept alive by convention even if no node reads them
    for o in graph.output:
        used.add(o.name)

    kept = [init for init in graph.initializer if init.name in used]
    removed = len(graph.initializer) - len(kept)
    graph.ClearField("initializer")
    graph.initializer.extend(kept)
    logger.debug("remove_unused_initializers: removed %d unused initializer(s).", removed)
    return model


# ── Step 7: Remove consecutive Cast chains ──────────────────────────────────


def _remove_consecutive_casts(model: onnx.ModelProto, cache: GraphCache) -> onnx.ModelProto:
    """Collapse consecutive Cast nodes into a single Cast with the last node's target dtype.

    A chain ``Cast(to=A) → Cast(to=B) → Cast(to=C)`` is replaced by a single
    ``Cast(to=C)`` from the first Cast's input to the last Cast's output.
    Chains of length 1 are left untouched.
    """
    graph = model.graph
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    cnt = 0

    # Build chains: tail_output → [cast_node, ...]
    all_casts = [n for n in graph.node if n.op_type == "Cast"]
    tail_to_chain: dict[str, list] = {}

    for node in all_casts:
        inp_tensor = node.input[0] if node.input else None
        if inp_tensor and inp_tensor in tail_to_chain:
            consumers = cache.get_consumers(inp_tensor)
            if len(consumers) == 1:
                chain = tail_to_chain.pop(inp_tensor)
                chain.append(node)
                tail_to_chain[node.output[0]] = chain
                continue
        tail_to_chain[node.output[0]] = [node]

    # Collapse chains with length > 1
    replacements: dict[int, tuple[list, list]] = {}
    for ch in tail_to_chain.values():
        if len(ch) <= 1:
            continue
        first_input = ch[0].input[0]
        last_output = ch[-1].output[0]
        last_to = get_node_attr_i(ch[-1], "to", 1)  # ONNX Cast default: FLOAT

        new_cast = helper.make_node(
            "Cast",
            inputs=[first_input],
            outputs=[last_output],
            to=last_to,
            name=f"{ch[-1].name}_chain",
        )
        idx = node_idx[id(ch[0])]
        replacements[idx] = (ch[:], [new_cast])
        cnt += len(ch)

    batch_replace_nodes(graph, replacements)
    logger.debug(
        "remove_consecutive_casts: collapsed %d Cast node(s) in %d chain(s).",
        cnt,
        len(replacements),
    )
    return model


# ── Master transform ──────────────────────────────────────────────────────────


def _apply_graph_cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply all five cleanup steps in sequence."""
    model = infer_shapes(model)
    cache = GraphCache(model.graph)

    # Step 1 — intermediary Squeeze/Unsqueeze → 4-D Reshape
    model = _replace_intermediary_squeeze_unsqueeze(model, cache)
    cache.rebuild()

    # Step 2 — non-4D inputs → canonical Unsqueeze
    model = _canonicalize_inputs(model, cache)
    model = infer_shapes(model)
    cache.rebuild()

    # Step 3 — collapse Reshape chains
    model = _remove_reshape_chains(model, cache)
    cache.rebuild()

    # Step 4 — replace graph-output Reshape with Squeeze where possible
    model = _replace_graph_output_reshape_with_squeeze(model, cache)

    # Step 5 — fold constants (graphsurgeon creates new model — cache invalidated)
    model = _fold_constants(model)

    # Step 6 — collapse consecutive Cast chains
    cache = GraphCache(model.graph)
    model = _remove_consecutive_casts(model, cache)

    # Step 7 — remove unused initializers
    model = _remove_unused_initializers(model)

    return model


def dla_graph_cleanup(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply graph cleanup, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_graph_cleanup,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
