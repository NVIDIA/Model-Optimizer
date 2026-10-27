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

"""Replace MatMul/Gemm nodes with an Unsqueeze-Transpose-Conv-Transpose-Squeeze subgraph.

Eligible nodes use Conv with rank adapters at the current stage boundary.
Ineligible Gemm nodes remain unchanged. Ineligible MatMul matrix operands use
four-dimensional promotion when requested; vector promotion is unsupported and
can only be deferred to a later stage. Rank adapters require ONNX opset 13+.

Eligibility (checked via :func:`check_to_apply_transpose`):
- The **second input** (weight) must be a constant initializer or a Const→DequantizeLinear chain.
  Dynamic / graph-input weights are NOT eligible for Conv conversion.
- Weight shape must be 2-D, 3-D with ``shape[0]==1``, or 4-D with ``shape[0]==1, shape[1]==1``.
- The **first input** (activation) shape must be known and be 2-D, 3-D, or 4-D with
  ``shape[0]==1``.

Shape contract after transform (per activation rank):

  +--------------+---------------+--------------------------------------+--------------+-------------+
  | Orig rank    | Unsqueeze     | After input Transpose (→ Conv input) | Conv output  | Squeeze     |
  +==============+===============+======================================+==============+=============+
  | 2D [M,N]     | [0,1]→[1,1,M,N]| perm[0,3,2,1]→[1,N,M,1]           | [1,K,M,1]    | [0,1]→[M,K] |
  | 3D [B,M,N]   | [0]→[1,B,M,N] | perm[0,3,2,1]→[1,N,M,B]            | [1,K,M,B]    | [0]→[B,M,K] |
  | 4D [1,B,M,N] | none          | perm[0,3,2,1]→[1,N,M,B]            | [1,K,M,B]    | none        |
  +--------------+---------------+--------------------------------------+--------------+-------------+

Output Transpose always uses perm ``[0,3,2,1]``: ``[1,K,M,B] → [1,B,M,K]``.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ...logging_config import logger
from ..utils.graph_utils import (
    GraphCache,
    add_unique_initializers,
    batch_replace_nodes,
    check_to_apply_transpose,
    get_node_attr_i,
    is_const_dq_input,
    pad4d,
    set_node_attr_i,
)
from .dla_5d_reshape_to_4d import _known_non_scalar_control
from .dla_constants_to_initializers import _add_legacy_initializer_inputs
from .dla_graph_cleanup import _subgraph_captures


def _apply_matmul_to_transpose_conv_transpose(model, *, defer_rank_conversion=False):
    """Convert eligible matrices and optionally defer fallback rank promotion."""
    graph = model.graph
    cache = GraphCache(graph)
    captured = _subgraph_captures(graph)
    local_functions = {
        (function.domain, function.name, function.overload) for function in model.functions
    }
    opsets = {entry.domain: entry.version for entry in model.opset_import}

    def tensor_rank(name):
        """Read rank metadata without requiring concrete dimension values."""
        info = cache.vi_map.get(name)
        if (
            info is not None
            and info.type.HasField("tensor_type")
            and info.type.tensor_type.HasField("shape")
        ):
            return len(info.type.tensor_type.shape.dim)
        shape = cache.get_shape(name)
        return len(shape) if shape is not None else None

    candidates = [
        node
        for node in graph.node
        if node.op_type in {"MatMul", "Gemm"}
        and node.domain in {"", "ai.onnx"}
        and not node.overload
        and (node.domain, node.op_type, node.overload) not in local_functions
    ]
    unique_initializer: set = set()

    # shape map snapshot required by check_to_apply_transpose
    tensor_name_dim_map: dict = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        s = cache.get_shape(vi.name)
        if s is not None:
            tensor_name_dim_map[vi.name] = s

    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    replacements_batch: dict[int, tuple[list, list]] = {}
    fallback_batch: dict[int, tuple[list, list]] = {}
    for candidate in candidates:
        if (
            candidate.op_type == "MatMul"
            and not defer_rank_conversion
            and any(tensor_rank(name) == 1 for name in candidate.input[:2])
        ):
            raise ValueError(f"{candidate.name!r}: vector MatMul rank promotion is unsupported")

    def update_weight_metadata(name, shape):
        """Keep private weight annotations consistent with an in-place layout change."""
        info = cache.vi_map.get(name)
        if info is not None:
            dims = info.type.tensor_type.shape.dim
            del dims[:]
            for size in shape:
                dims.add().dim_value = size

    def update_initializers(name, current_node, nodes_to_add, trans_b=False, dq_node=None):
        """Allocate the Conv layout while retaining every shared, public or captured weight view."""
        initializer = cache.get_init(name)
        if initializer is None:
            raise ValueError(f"Missing weight initializer {name!r}")
        dims = list(initializer.dims)
        rank = len(dims)
        perm = None if trans_b else list(reversed(range(rank)))
        if trans_b and rank == 4:
            perm = [1, 2, 3, 0]
        shape = [dims[index] for index in perm] if perm is not None else dims
        shape = [*shape, *([1] * (4 - rank))]
        axis = None
        if dq_node is not None:
            axis = next(
                (attribute.i for attribute in dq_node.attribute if attribute.name == "axis"), None
            )
            if axis is None and any(
                _known_non_scalar_control(cache, control)
                or (cache.get_init(control) is None and cache.get_shape(control) is None)
                for control in dq_node.input[1:]
                if control
            ):
                axis = 1
            if axis is not None:
                axis %= rank
                if perm is not None:
                    axis = perm.index(axis)
        shared = (
            len(cache.get_consumers(name)) > 1
            or name in cache.graph_output_names | captured
            or (
                dq_node is not None
                and (
                    len(cache.get_consumers(dq_node.output[0])) > 1
                    or dq_node.output[0] in cache.graph_output_names | captured
                )
            )
        )
        clone_idx = len(replacements_batch)
        target_name = cache.unique_name(f"{name}_conv4d_{clone_idx}") if shared else name
        if perm is None:
            target = graph.initializer.add() if shared else initializer
            if shared:
                target.CopyFrom(initializer)
                target.name = target_name
            del target.dims[:]
            target.dims.extend(shape)
        else:
            array = np.transpose(numpy_helper.to_array(initializer), perm).reshape(shape)
            converted = numpy_helper.from_array(array, target_name)
            if shared:
                add_unique_initializers(graph, unique_initializer, [converted])
            else:
                initializer.CopyFrom(converted)
        if not shared:
            update_weight_metadata(name, shape)
            if dq_node is not None:
                update_weight_metadata(dq_node.output[0], shape)
        if shared:
            if dq_node is not None:
                dq_clone = type(dq_node)()
                dq_clone.CopyFrom(dq_node)
                dq_clone.name = cache.unique_name(f"{dq_node.name or name}_conv4d_{clone_idx}")
                dq_clone.input[0] = target_name
                dq_clone.output[0] = cache.unique_name(f"{dq_node.output[0]}_conv4d_{clone_idx}")
                if axis is not None:
                    set_node_attr_i(dq_clone, "axis", axis)
                nodes_to_add.append(dq_clone)
                current_node.input[1] = dq_clone.output[0]
            else:
                current_node.input[1] = target_name
        elif dq_node is not None and axis is not None:
            set_node_attr_i(dq_node, "axis", axis)
            dq_node.name = cache.unique_name(f"{dq_node.name}_axis_updated")

    eligible_nodes = []
    ineligible_nodes = []
    for node in candidates:
        producer = cache.get_producer(node.input[1])
        dq_node = (
            producer if producer is not None and producer.op_type == "DequantizeLinear" else None
        )
        known_dq = dq_node is None or (
            not dq_node.overload
            and (dq_node.domain, dq_node.op_type, dq_node.overload) not in local_functions
            and (
                dq_node.domain in {"", "ai.onnx"}
                or (dq_node.domain == "com.microsoft" and opsets.get(dq_node.domain) == 1)
            )
            and not get_node_attr_i(dq_node, "block_size", 0)
        )
        weight_name = dq_node.input[0] if dq_node is not None else node.input[1]
        weight = cache.get_init(weight_name)
        dtype = cache.get_dtype(node.input[0])
        activation_shape = cache.get_shape(node.input[0])
        unsupported_gemm = False
        if node.op_type == "Gemm":
            unsupported_gemm = any(
                attribute.name in {"alpha", "beta"} and attribute.f != 1.0
                for attribute in node.attribute
            )
            if len(node.input) > 2 and node.input[2]:
                channels = (
                    weight.dims[-2 if get_node_attr_i(node, "transB", 0) else -1]
                    if weight is not None and len(weight.dims) >= 2
                    else None
                )
                unsupported_gemm |= channels is None or cache.get_shape(node.input[2]) != [channels]
        if (
            unsupported_gemm
            or not known_dq
            or weight is None
            or weight_name in cache.graph_input_names
            or any(tensor_rank(name) == 1 for name in node.input[:2])
            or (
                dtype is not None
                and dtype not in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}
            )
            or any(dim == 0 for dim in weight.dims)
            or (activation_shape is not None and 0 in activation_shape)
        ):
            ineligible_nodes.append(node)
        elif (
            activation_shape is not None
            and node.input[0] in tensor_name_dim_map
            and (
                len(activation_shape) in (2, 3)
                or (len(activation_shape) == 4 and activation_shape[0] == 1)
            )
            and check_to_apply_transpose(node, model, tensor_name_dim_map)
        ):
            output_rank = tensor_rank(node.output[0])
            if output_rank is None:
                output_rank = (
                    max(len(tensor_name_dim_map[node.input[0]]), len(weight.dims))
                    if node.op_type == "MatMul"
                    else 2
                )
            eligible_nodes.append((node, weight_name, dq_node, output_rank))
        else:
            ineligible_nodes.append(node)
    active_fallback = [
        node
        for node in ineligible_nodes
        if node.op_type == "MatMul"
        and all(tensor_rank(name) is not None and tensor_rank(name) >= 2 for name in node.input[:2])
        and tensor_rank(node.output[0]) is not None
        and (not defer_rank_conversion or node.input[1] in cache.graph_input_names)
    ]
    needs_adapters = any(
        len(tensor_name_dim_map[node.input[0]]) < 4 or rank < 4
        for node, _, _, rank in eligible_nodes
    ) or any(
        any(
            tensor_rank(name) is not None and tensor_rank(name) < 4
            for name in [*node.input[:2], node.output[0]]
        )
        for node in active_fallback
    )
    if needs_adapters and opsets.get("", opsets.get("ai.onnx", 1)) < 13:
        raise ValueError("DLA matrix rank adapters require ONNX opset 13 or newer")

    for node in active_fallback:
        cnt_fb = len(fallback_batch)
        pfx = node.name or f"mm_fallback_{cnt_fb}"
        initializer_names = cache.init_names
        pre: list = []
        post: list = []

        for inp_idx in range(min(2, len(node.input))):
            inp = node.input[inp_idx]
            if defer_rank_conversion and (inp_idx != 1 or inp not in cache.graph_input_names):
                continue
            if not inp:
                continue
            rank = tensor_rank(inp)
            if rank is None or rank >= 4:
                continue
            # Check whether this is a static initializer (possibly DQ-wrapped)
            dq_src = is_const_dq_input(initializer_names, inp, graph)
            init_name = dq_src or (
                inp if inp in initializer_names and inp not in cache.graph_input_names else None
            )
            if init_name is not None:
                if dq_src is not None:
                    # Preserve the quantization layout when a static DQ weight needs promotion.
                    clone = helper.make_node(
                        "Unsqueeze",
                        [inp, ""],
                        [cache.unique_name(f"{pfx}_in{inp_idx}_4d_{cnt_fb}")],
                        name=cache.unique_name(pfx + "_weight_unsqueeze"),
                    )
                    axes_name = cache.unique_name(f"{pfx}_in{inp_idx}_unsq_axes_{cnt_fb}")
                    graph.initializer.append(
                        numpy_helper.from_array(np.arange(4 - rank, dtype=np.int64), axes_name)
                    )
                    clone.input[1] = axes_name
                    pre.append(clone)
                    node.input[inp_idx] = clone.output[0]
                    continue
                # Static weight: expand initializer dims in-place with 1-padding
                for init in graph.initializer:
                    if init.name == init_name:
                        if len(init.dims) < 4:
                            new_shape = pad4d(list(init.dims))
                            if (
                                len(cache.get_consumers(init.name)) > 1
                                or init.name in cache.graph_output_names | captured
                            ):
                                clone = graph.initializer.add()
                                clone.CopyFrom(init)
                                clone.name = cache.unique_name(init.name + "_4d")
                                del clone.dims[:]
                                clone.dims.extend(new_shape)
                                node.input[inp_idx] = clone.name
                            else:
                                del init.dims[:]
                                init.dims.extend(new_shape)
                                update_weight_metadata(init.name, new_shape)
                        break
            else:
                # Dynamic input: Unsqueeze to 4D
                axes = list(range(4 - rank))
                axes_init = numpy_helper.from_array(
                    np.array(axes, dtype=np.int64),
                    name=cache.unique_name(f"{pfx}_in{inp_idx}_unsq_axes_{cnt_fb}"),
                )
                add_unique_initializers(graph, unique_initializer, [axes_init])
                inp_4d = cache.unique_name(f"{pfx}_in{inp_idx}_4d_{cnt_fb}")
                pre.append(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[inp, axes_init.name],
                        outputs=[inp_4d],
                        name=cache.unique_name(f"{pfx}_in{inp_idx}_unsq_{cnt_fb}"),
                    )
                )
                node.input[inp_idx] = inp_4d

        # Output: Squeeze back to original rank
        out_name = node.output[0] if node.output else None
        if out_name:
            out_rank = tensor_rank(out_name)
            if out_rank is not None and out_rank < 4:
                temp_out = cache.unique_name(f"{pfx}_out_4d_{cnt_fb}")
                sq_axes = list(range(4 - out_rank))
                sq_axes_init = numpy_helper.from_array(
                    np.array(sq_axes, dtype=np.int64),
                    name=cache.unique_name(f"{pfx}_out_sq_axes_{cnt_fb}"),
                )
                add_unique_initializers(graph, unique_initializer, [sq_axes_init])
                post.append(
                    helper.make_node(
                        "Squeeze",
                        inputs=[temp_out, sq_axes_init.name],
                        outputs=[out_name],
                        name=cache.unique_name(f"{pfx}_out_sq_{cnt_fb}"),
                    )
                )
                node.output[0] = temp_out

        if pre or post:
            fallback_batch[node_idx[id(node)]] = ([node], [*pre, node, *post])

    for node, weight_name, weight_dq, output_rank in eligible_nodes:
        cnt = len(replacements_batch)
        pfx = node.name or f"matmul_{cnt}"
        nodes_to_add = []

        # ── Gemm transA / transB attributes ──────────────────────────────────
        trans_a = False
        trans_b = False
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = bool(attr.i)
                elif attr.name == "transB":
                    trans_b = bool(attr.i)

        # ── Activation rank ───────────────────────────────────────────────────
        input_shape = cache.get_shape(node.input[0])
        if input_shape is None:
            raise ValueError(f"Shape unknown for activation tensor {node.input[0]!r}")
        orig_rank = len(input_shape)

        # ── Step 1: Unsqueeze activation to 4D ───────────────────────────────
        # 2D [M,N]   → axes [0,1] → [1,1,M,N]
        # 3D [B,M,N] → axes [0]   → [1,B,M,N]
        activation_4d = node.input[0]
        if orig_rank < 4:
            axes = [0, 1] if orig_rank == 2 else [0]
            axes_init = numpy_helper.from_array(
                np.array(axes, dtype=np.int64), name=cache.unique_name(f"{pfx}_unsq_axes_{cnt}")
            )
            add_unique_initializers(graph, unique_initializer, [axes_init])
            activation_4d = cache.unique_name(f"{pfx}_unsqueeze4d_{cnt}")
            nodes_to_add.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_init.name],
                    outputs=[activation_4d],
                    name=cache.unique_name(f"{pfx}_unsqueeze4d_node_{cnt}"),
                )
            )

        # ── Step 2: Transpose activation (guaranteed 4D at this point) ───────
        # transA=False: [1,B,M,N] → perm[0,3,2,1] → [1,N,M,B]  (C_in=N, H=M, W=B)
        # transA=True : [1,B,N,M] → perm[0,2,3,1] → [1,N,M,B]  (same result)
        perm_a = [0, 2, 3, 1] if trans_a else [0, 3, 2, 1]
        transposed_a = cache.unique_name(f"{pfx}_tr_before0_{cnt}")
        nodes_to_add.append(
            helper.make_node(
                "Transpose",
                inputs=[activation_4d],
                outputs=[transposed_a],
                name=cache.unique_name(f"{pfx}_transpose_before0_{cnt}"),
                perm=perm_a,
            )
        )

        # ── Step 3: Prepare weight ────────────────────────────────────────────
        update_initializers(weight_name, node, nodes_to_add, trans_b, weight_dq)
        conv_input1 = node.input[1]

        conv_inputs = [transposed_a, conv_input1]
        if len(node.input) >= 3 and node.input[2]:
            conv_inputs.append(node.input[2])

        # ── Step 5: Conv ──────────────────────────────────────────────────────
        conv_out = cache.unique_name(f"{pfx}_conv_out_{cnt}")
        nodes_to_add.append(
            helper.make_node(
                "Conv",
                inputs=conv_inputs,
                outputs=[conv_out],
                name=cache.unique_name(f"{pfx}_conv_{cnt}"),
            )
        )

        # ── Step 6: Output Transpose ──────────────────────────────────────────
        # [1,K,M,B] → perm[0,3,2,1] → [1,B,M,K]
        # Write to a temp name when Squeeze follows; otherwise directly to node.output[0].
        tr_after_out = (
            cache.unique_name(f"{pfx}_tr_after_out_{cnt}") if output_rank < 4 else node.output[0]
        )
        nodes_to_add.append(
            helper.make_node(
                "Transpose",
                inputs=[conv_out],
                outputs=[tr_after_out],
                name=cache.unique_name(f"{pfx}_transpose_after_{cnt}"),
                perm=[0, 3, 2, 1],
            )
        )

        # ── Step 7: Squeeze back to original rank ─────────────────────────────
        # 2D case: [1,1,M,K] → squeeze [0,1] → [M,K]
        # 3D case: [1,B,M,K] → squeeze [0]   → [B,M,K]
        if output_rank < 4:
            sq_axes = list(range(4 - output_rank))
            sq_axes_init = numpy_helper.from_array(
                np.array(sq_axes, dtype=np.int64), name=cache.unique_name(f"{pfx}_sq_axes_{cnt}")
            )
            add_unique_initializers(graph, unique_initializer, [sq_axes_init])
            nodes_to_add.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[tr_after_out, sq_axes_init.name],
                    outputs=list(node.output),
                    name=cache.unique_name(f"{pfx}_squeeze_after_{cnt}"),
                )
            )

        replacements_batch[node_idx[id(node)]] = ([node], nodes_to_add)

    # ── Apply all replacements ────────────────────────────────────────────────
    all_batch = {**replacements_batch, **fallback_batch}
    # Later rank passes consume the original shape snapshot without another inference step.
    for _, replacements in all_batch.values():
        for adapter in replacements:
            if adapter.op_type not in {"Unsqueeze", "Squeeze"}:
                continue
            original, padded = (
                (adapter.input[0], adapter.output[0])
                if adapter.op_type == "Unsqueeze"
                else (adapter.output[0], adapter.input[0])
            )
            info = cache.vi_map.get(original)
            if info is not None and info.type.tensor_type.HasField("shape"):
                promoted = graph.value_info.add()
                promoted.CopyFrom(info)
                promoted.name = padded
                dims = promoted.type.tensor_type.shape.dim
                original_dims = list(dims)
                del dims[:]
                for _ in range(4 - len(original_dims)):
                    dims.add().dim_value = 1
                dims.extend(original_dims)
    batch_replace_nodes(graph, all_batch)
    _add_legacy_initializer_inputs(model)

    logger.debug(
        "Replaced %d MatMul/Gemm nodes with Transpose-Conv-Transpose nodes; "
        "promoted %d ineligible MatMul nodes to 4D.",
        len(replacements_batch),
        len(fallback_batch),
    )
    return model
