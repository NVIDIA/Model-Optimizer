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

"""Graph cleanup for DLA ONNX models.

Promote eligible input and Squeeze/Unsqueeze paths to 4D while retaining original
ranks at rank-sensitive consumers. Collapse reshape chains, adapt graph-output
reshapes, fold constants, collapse lossless cast chains, and remove unused
initializers. Constant folding requires the optional onnx-graphsurgeon package.
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    _captured_names,
    _node_inputs,
    add_unique_initializers,
    add_value_info,
    batch_replace_nodes,
    get_node_attr_i,
    infer_shapes,
    insert_nodes_at_position,
    pad4d,
)
from .dla_constants_to_initializers import _GENERATED_INITIALIZER_INPUT

# ── Internal utilities ────────────────────────────────────────────────────────


_RANK_AGNOSTIC_OPS = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Add",
        "And",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Cast",
        "Ceil",
        "Clip",
        "Cos",
        "Cosh",
        "Div",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Floor",
        "Greater",
        "GreaterOrEqual",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "IsInf",
        "IsNaN",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "Max",
        "Mean",
        "Min",
        "Mod",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "PRelu",
        "Pow",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
        "Where",
        "Xor",
    }
)


def _set_axes(model, cache: GraphCache, node: onnx.NodeProto, axes, preferred_name: str) -> None:
    """Attach axes using the model's default-domain Squeeze/Unsqueeze schema."""
    versions = {o.domain: o.version for o in model.opset_import}
    opset = versions.get("", versions.get("ai.onnx", 1))
    if opset < 13:
        node.attribute.append(
            helper.make_attribute("axes", list(axes), attr_type=onnx.AttributeProto.INTS)
        )
    else:
        axes_name = cache.unique_name(preferred_name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=axes_name)
        )
        node.input.append(axes_name)


def _subgraph_captures(graph: onnx.GraphProto) -> set[str]:
    """Collect outer-scope values referenced inside this graph's nested subgraphs."""
    captured: set[str] = set()
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                captured.update(_captured_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    captured.update(_captured_names(subgraph))
    return captured


def _rank_sensitive_consumers(
    graph: onnx.GraphProto, cache: GraphCache
) -> tuple[set[str], set[tuple[int, int]]]:
    """Find edges where leading unit dimensions can change downstream semantics.

    Propagate restrictions through elementwise paths until an already-4D output
    prevents an input promotion from changing downstream ranks.
    """
    sensitive_values: set[str] = set()
    sensitive_edges: set[tuple[int, int]] = set()
    for node in reversed(graph.node):
        rank_agnostic = not node.domain and not node.overload and node.op_type in _RANK_AGNOSTIC_OPS
        if rank_agnostic:
            rank_agnostic = not any(
                output in sensitive_values and len(cache.get_shape(output) or []) < 4
                for output in node.output
            )
        if not rank_agnostic:
            sensitive_values.update(_node_inputs(node))
        for index, name in enumerate(node.input):
            if name and (not rank_agnostic or (node.op_type in ("Clip", "PRelu") and index > 0)):
                sensitive_values.add(name)
                sensitive_edges.add((id(node), index))
    return sensitive_values, sensitive_edges


# ── Step 1: Canonicalize non-4D model inputs ─────────────────────────────────


def _canonicalize_inputs(model: onnx.ModelProto, cache: GraphCache) -> onnx.ModelProto:
    """Promote eligible input consumers to 4D without changing rank-sensitive paths."""
    graph = model.graph
    unique_init: set[str] = set()
    consumer_map = cache.consumer_map
    nodes_to_insert: list[tuple] = []  # (new_node, reference_node)
    nodes_to_remove: list = []
    promotion_safety = None
    captured: set[str] = set()
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

        if promotion_safety is None:
            promotion_safety = _rank_sensitive_consumers(graph, cache)
            captured = _subgraph_captures(graph)
        sensitive_values, sensitive_edges = promotion_safety

        # Read concrete input shape (may be None for symbolic dims)
        input_shape: list[int | None] = [
            d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim
        ]

        n_new = 4 - rank
        axes_arr = np.arange(n_new, dtype=np.int64)
        canonical_4d = [1] * n_new + list(input_shape)

        consumers = []
        for c_node, c_idx in consumer_map.get(graph_input.name, []):
            if c_node.domain or c_node.overload:
                continue
            if (c_node.op_type == "Squeeze" and "Gather" in c_node.name) or c_node.op_type in (
                "GroupQueryAttention",
                "Gather",
            ):
                continue
            if c_node.op_type == "Unsqueeze" and c_idx == 0:
                if c_node.output[0] in captured:
                    continue
                old_shape = cache.get_shape(c_node.output[0])
                if old_shape is None or (
                    len(old_shape) < 4
                    and c_node.output[0] not in cache.graph_output_names
                    and c_node.output[0] in sensitive_values
                ):
                    continue
            elif (id(c_node), c_idx) in sensitive_edges:
                continue
            consumers.append((c_node, c_idx))
        if not consumers:
            continue

        out_name = cache.unique_name(f"{graph_input.name}_4d")
        unsq = helper.make_node(
            "Unsqueeze",
            inputs=[graph_input.name],
            outputs=[out_name],
            name=cache.unique_name(f"{graph_input.name}_unsqueeze"),
        )
        _set_axes(model, cache, unsq, axes_arr, f"{graph_input.name}_unsqueeze_axes")
        add_value_info(graph, out_name, tt.elem_type, canonical_4d)

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
                        if (
                            c_node.output[0] in cache.graph_output_names
                            and old_out_shape != canonical_4d
                        ):
                            continue
                        if c_node.output[0] in cache.graph_output_names:
                            nodes_to_insert.append(
                                (
                                    helper.make_node(
                                        "Identity",
                                        [out_name],
                                        list(c_node.output),
                                        name=cache.unique_name(c_node.name + "_identity"),
                                    ),
                                    c_node,
                                )
                            )
                        # Same 4-D shape → redirect downstream consumers, remove old Unsqueeze
                        for dn, di in consumer_map.get(c_node.output[0], []):
                            dn.input[di] = out_name
                    else:
                        # Different 4-D shape → Reshape from canonical output
                        is_graph_output = c_node.output[0] in cache.graph_output_names
                        reshape_target = old_out_shape if is_graph_output else old_4d
                        rs_name = cache.unique_name(
                            f"{c_node.name or c_node.op_type}_reshape_shape"
                        )
                        add_unique_initializers(
                            graph,
                            unique_init,
                            [
                                numpy_helper.from_array(
                                    np.array(reshape_target, dtype=np.int64), name=rs_name
                                )
                            ],
                        )
                        rs = helper.make_node(
                            "Reshape",
                            inputs=[out_name, rs_name],
                            outputs=list(c_node.output),
                            name=cache.unique_name(f"{c_node.name or c_node.op_type}_reshape"),
                        )
                        if not is_graph_output:
                            cache.update_shape(c_node.output[0], old_4d)
                        nodes_to_insert.append((rs, c_node))
                else:
                    # Unknown shape — just update the input to canonical unsqueeze output
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


def _compute_sq_unsq_out_shape(
    node: onnx.NodeProto, graph: onnx.GraphProto, cache: GraphCache
) -> list[int] | None:
    """Compute the output shape of a Squeeze/Unsqueeze from its axes and input shape.

    Used as a fallback when shape inference did not populate ``value_info``.
    Returns ``None`` if the shape cannot be determined.
    """
    in_shape = cache.get_shape(node.input[0])
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

    if node.op_type == "Squeeze" and axes is None:
        axes = next((list(attr.ints) for attr in node.attribute if attr.name == "axes"), None)

    if node.op_type == "Squeeze":
        if axes is None:
            if len(node.input) > 1 and node.input[1]:
                return None
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
        out_rank = len(in_shape) + len(axes)
        for ax in sorted(a if a >= 0 else out_rank + a for a in axes):
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
    * ``Unsqueeze`` whose data input is a 3D GQA output and whose own output is 4D
      — this Unsqueeze was placed to adapt the 3D→4D boundary
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
    * Promoting the output would change a downstream rank-sensitive operation.
    """
    graph = model.graph
    unique_init: set[str] = set()
    graph_output_names = cache.graph_output_names
    sensitive_values: set[str] | None = None

    nodes_to_remove: list = []
    nodes_to_insert: list[tuple] = []
    cnt_replace = 0

    for node in list(graph.node):
        if node.domain or node.overload or node.op_type not in ("Squeeze", "Unsqueeze"):
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
        out_shape = cache.get_shape(node.output[0]) or _compute_sq_unsq_out_shape(
            node, graph, cache
        )
        if out_shape is None:
            logger.debug("%s %r: output shape unknown, skipping.", node.op_type, node.name)
            continue

        is_graph_output = node.output[0] in graph_output_names
        if not is_graph_output and len(out_shape) < 4:
            if sensitive_values is None:
                sensitive_values, _ = _rank_sensitive_consumers(graph, cache)
            if node.output[0] in sensitive_values:
                continue

        # Graph output: use declared shape (preserves name binding); else pad to 4D.
        reshape_target = out_shape if is_graph_output else pad4d(out_shape)

        rs_name = cache.unique_name(f"{node.name or node.op_type}_4d_shape")
        add_unique_initializers(
            graph,
            unique_init,
            [numpy_helper.from_array(np.array(reshape_target, dtype=np.int64), name=rs_name)],
        )
        rs = helper.make_node(
            "Reshape",
            inputs=[node.input[0], rs_name],
            outputs=list(node.output),
            name=cache.unique_name(f"{node.name or node.op_type}_to_reshape"),
        )
        nodes_to_insert.append((rs, node))
        nodes_to_remove.append(node)
        cache.update_shape(node.output[0], reshape_target)
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

    Pass 1 — Remove noop Reshapes (``in_shape == out_shape``) without captured outputs:
    consumers are wired directly to the Reshape input; at graph-output boundaries
    an ``Identity`` node is inserted to keep the output name alive.

    Pass 2 — Collapse consecutive Reshape chains (each member's output consumed
    only by the next Reshape, without subgraph captures) into a single Reshape;
    chains that are noops as a whole are removed the same way.
    """
    graph = model.graph
    graph_output_names = cache.graph_output_names
    noop_count = 0
    captured = _subgraph_captures(graph)
    chain_count = 0
    chain_nodes_count = 0
    # ── Pass 1: Remove standalone noop Reshapes ───────────────────────────────
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    pass1_batch: dict[int, tuple[list, list]] = {}

    for node in cache.nodes_by_op("Reshape"):
        if node.domain or node.overload:
            continue
        in_shape = cache.get_shape(node.input[0])
        out_shape = cache.get_shape(node.output[0])
        if in_shape is None or out_shape is None:
            continue
        if list(in_shape) != list(out_shape):
            continue
        # Noop reshape: rewire downstream node-consumers to bypass this node
        inp = node.input[0]
        out_tensor = node.output[0]
        if out_tensor in captured:
            continue
        for c_node, c_idx in cache.get_consumers(out_tensor):
            if c_node.input[c_idx] == out_tensor:
                c_node.input[c_idx] = inp
        if out_tensor in graph_output_names:
            id_node = helper.make_node(
                "Identity",
                [inp],
                [out_tensor],
                name=cache.unique_name(f"cleanup_noop_id_{out_tensor}"),
            )
            pass1_batch[node_idx[id(node)]] = ([node], [id_node])
        else:
            pass1_batch[node_idx[id(node)]] = ([node], [])
        noop_count += 1

    batch_replace_nodes(graph, pass1_batch)

    # Rebuild cache/index after Pass 1 mutations so Pass 2 sees the updated
    # topology, consumers, and shape metadata.
    cache.rebuild()
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
    all_reshapes = [
        n for n in graph.node if n.op_type == "Reshape" and not n.domain and not n.overload
    ]

    # output_name → chain (list of Reshape nodes whose last output == output_name)
    tail_to_chain: dict[str, list] = {}

    for node in all_reshapes:
        inp_tensor = node.input[0] if node.input else None
        if inp_tensor and inp_tensor in tail_to_chain:
            # Extend existing chain only if this node is the sole consumer
            consumers = cache.get_consumers(inp_tensor)
            if (
                len(consumers) == 1
                and inp_tensor not in cache.graph_output_names
                and inp_tensor not in captured
            ):
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
            if out_tensor in captured:
                continue
            # Entire chain is a noop
            for c_node, c_idx in cache.get_consumers(out_tensor):
                if c_node.input[c_idx] == out_tensor:
                    c_node.input[c_idx] = inp
            if out_tensor in graph_output_names:
                id_node = helper.make_node(
                    "Identity",
                    [inp],
                    [out_tensor],
                    name=cache.unique_name(f"cleanup_chain_id_{out_tensor}"),
                )
                pass2_batch[node_idx[id(ch[0])]] = (ch[:], [id_node])
            else:
                pass2_batch[node_idx[id(ch[0])]] = (ch[:], [])
        else:
            rs_shape = numpy_helper.from_array(
                np.array(out_shape, dtype=np.int64),
                name=cache.unique_name(f"{ch[-1].name or 'reshape'}_chain_shape"),
            )
            graph.initializer.append(rs_shape)
            rs = helper.make_node(
                "Reshape",
                inputs=[inp, rs_shape.name],
                outputs=[out_tensor],
                name=cache.unique_name(f"{ch[-1].name or 'reshape'}_chain"),
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
        """Keep quantization boundaries and identity bindings out of constant folding."""
        return node.op in ("DequantizeLinear", "Identity")

    g = gs.import_onnx(model)
    g.fold_constants(should_exclude_node=_exclude).cleanup().toposort()
    model2 = gs.export_onnx(g, ir_version=model.ir_version)

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
        if node.domain or node.overload or node.op_type != "Reshape":
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
            # Pad the Reshape target to 4D and restore the declared output rank.
            #
            # Example:
            #   [5,6,7,8] → Reshape([30,14,4]) → graph_output[30,14,4]
            # becomes:
            #   [5,6,7,8] → Reshape([1,30,14,4]) → Squeeze([0]) → graph_output[30,14,4]
            if len(out_shape) >= 4:
                continue  # already 4D or higher — leave as-is
            shape_4d = pad4d(out_shape)
            squeeze_axes = list(range(4 - len(out_shape)))
            rs_shape_name = cache.unique_name(f"{pfx}_4d_shape")
            add_unique_initializers(
                graph,
                unique_init,
                [numpy_helper.from_array(np.array(shape_4d, dtype=np.int64), name=rs_shape_name)],
            )
            # Intermediate 4D tensor name
            rs_4d_out = cache.unique_name(f"{pfx}_4d_out")
            # Update the Reshape node to target the 4D shape
            node.input[1] = rs_shape_name
            node.output[0] = rs_4d_out
            sq = helper.make_node(
                "Squeeze",
                inputs=[rs_4d_out],
                outputs=[out_name],
                name=cache.unique_name(f"{pfx}_squeeze_leading"),
            )
            _set_axes(model, cache, sq, squeeze_axes, f"{pfx}_sq0_axes")
            go_batch[go_node_idx[id(node)]] = ([node], [node, sq])
            cnt += 1
            logger.debug(
                "Reshape %r → Reshape(%s) + Squeeze(%s) for non-4D graph output %r.",
                node.name,
                shape_4d,
                squeeze_axes,
                out_name,
            )
            continue

        squeeze_axes.sort()  # ascending axis order required by ONNX Squeeze
        if not squeeze_axes:
            continue

        # Build Squeeze node
        sq = helper.make_node(
            "Squeeze",
            inputs=[in_name],
            outputs=[out_name],
            name=cache.unique_name(f"{pfx}_squeeze"),
        )
        _set_axes(model, cache, sq, squeeze_axes, f"{pfx}_squeeze_axes")
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
        used.update(_node_inputs(node))
    # Graph outputs are kept alive by convention even if no node reads them
    for o in graph.output:
        used.add(o.name)

    used.update(
        value.name
        for value in graph.input
        if model.ir_version >= 4 or value.doc_string != _GENERATED_INITIALIZER_INPUT
    )
    kept = [init for init in graph.initializer if init.name in used]
    discarded = {init.name for init in graph.initializer if init.name not in used}
    removed = len(graph.initializer) - len(kept)
    if model.ir_version <= 3:
        removed_names = {init.name for init in graph.initializer if init.name not in used}
        for index in reversed(range(len(graph.input))):
            if graph.input[index].name in removed_names:
                del graph.input[index]
    graph.ClearField("initializer")
    graph.initializer.extend(kept)
    if model.ir_version <= 3:
        inputs = [value for value in graph.input if value.name not in discarded]
        graph.ClearField("input")
        graph.input.extend(inputs)
    logger.debug("remove_unused_initializers: removed %d unused initializer(s).", removed)
    return model


# ── Step 6: Remove consecutive Cast chains ──────────────────────────────────


def _remove_consecutive_casts(model: onnx.ModelProto, cache: GraphCache) -> onnx.ModelProto:
    """Collapse standard-domain Cast chains while preserving the final Cast's attributes.

    A chain ``Cast(to=A) → Cast(to=B) → Cast(to=C)`` is replaced by a single
    ``Cast(to=C)`` from the first Cast's input to the last Cast's output.
    Chains of length 1 are left untouched.
    """
    graph = model.graph
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    cnt = 0

    # Build chains: tail_output → [cast_node, ...]
    all_casts = [n for n in graph.node if n.op_type == "Cast" and not n.domain and not n.overload]
    captured = _subgraph_captures(graph)
    tail_to_chain: dict[str, list] = {}

    for node in all_casts:
        inp_tensor = node.input[0] if node.input else None
        if inp_tensor and inp_tensor in tail_to_chain:
            consumers = cache.get_consumers(inp_tensor)
            if (
                len(consumers) == 1
                and inp_tensor not in cache.graph_output_names
                and inp_tensor not in captured
            ):
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
        source_type = cache.get_dtype(first_input)
        # Removing a lossy intermediate cast changes rounding/truncation.
        lossless = {
            onnx.TensorProto.FLOAT16: {onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE},
            onnx.TensorProto.FLOAT: {onnx.TensorProto.DOUBLE},
            onnx.TensorProto.INT8: {
                onnx.TensorProto.INT16,
                onnx.TensorProto.INT32,
                onnx.TensorProto.INT64,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE,
            },
            onnx.TensorProto.INT16: {
                onnx.TensorProto.INT32,
                onnx.TensorProto.INT64,
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.DOUBLE,
            },
            onnx.TensorProto.INT32: {onnx.TensorProto.INT64, onnx.TensorProto.DOUBLE},
        }
        if any(
            get_node_attr_i(node, "to", 1) not in {source_type, *lossless.get(source_type, set())}
            for node in ch[:-1]
        ):
            continue
        last_output = ch[-1].output[0]
        new_cast = helper.make_node(
            "Cast",
            inputs=[first_input],
            outputs=[last_output],
            name=cache.unique_name(f"{ch[-1].name or 'cast'}_chain"),
        )
        new_cast.attribute.extend(ch[-1].attribute)
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


def _restore_promoted_graph_outputs(
    model, cache: GraphCache, original_outputs: dict, original_ranks: dict
) -> bool:
    """Restore inferred native ranks while retaining leading-unit promotion inside the graph."""
    graph = model.graph
    plans = []
    for output in graph.output:
        original = original_outputs[output.name]
        current_type = output.type.tensor_type
        original_rank = original_ranks[output.name]
        if not current_type.HasField("shape") or original_rank is None:
            continue
        padding = len(current_type.shape.dim) - original_rank
        if padding == 0:
            continue
        producer = cache.get_producer(output.name)
        if (
            padding < 0
            or any(dim.dim_value != 1 for dim in current_type.shape.dim[:padding])
            or producer is None
        ):
            raise ValueError(
                f"Cleanup cannot restore non-unit shape changes for output {output.name!r}"
            )
        plans.append((output, original, producer, padding))
    if not plans:
        return False

    positions = {id(node): index for index, node in enumerate(graph.node)}
    insertions = {}
    for output, original, producer, padding in plans:
        promoted_name = cache.unique_name(f"{output.name}_4d_output")
        promoted_info = copy.deepcopy(output)
        promoted_info.name = promoted_name
        graph.value_info.append(promoted_info)
        for index, name in enumerate(producer.output):
            if name == output.name:
                producer.output[index] = promoted_name
        for consumer, index in cache.get_consumers(output.name):
            consumer.input[index] = promoted_name
        squeeze = helper.make_node(
            "Squeeze",
            [promoted_name],
            [output.name],
            name=cache.unique_name(f"{output.name}_restore_rank"),
        )
        _set_axes(model, cache, squeeze, range(padding), f"{output.name}_restore_axes")
        output.CopyFrom(original)
        insertions.setdefault(positions[id(producer)] + 1, []).append(squeeze)
    for position in sorted(insertions, reverse=True):
        for node in reversed(insertions[position]):
            graph.node.insert(position, node)
    return True


# ── Master transform ──────────────────────────────────────────────────────────


def _apply_graph_cleanup(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply seven DLA cleanup stages while retaining output signatures and rank-sensitive paths."""
    original_outputs = {output.name: copy.deepcopy(output) for output in model.graph.output}
    model = infer_shapes(model)
    original_ranks = {
        output.name: (
            len(output.type.tensor_type.shape.dim)
            if output.type.tensor_type.HasField("shape")
            else None
        )
        for output in model.graph.output
    }
    cache = GraphCache(model.graph)

    # Step 1 — intermediary Squeeze/Unsqueeze → 4-D Reshape
    model = _replace_intermediary_squeeze_unsqueeze(model, cache)
    cache.rebuild()

    # Step 2 — non-4D inputs → canonical Unsqueeze
    model = _canonicalize_inputs(model, cache)
    model = infer_shapes(model)
    cache.rebuild()
    if _restore_promoted_graph_outputs(model, cache, original_outputs, original_ranks):
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
    for output in model.graph.output:
        output.CopyFrom(original_outputs[output.name])

    return model
