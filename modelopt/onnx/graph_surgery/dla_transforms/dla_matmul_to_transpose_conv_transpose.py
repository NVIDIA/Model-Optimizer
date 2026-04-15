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

For nodes that pass eligibility (see below), the full Conv conversion is applied.
For **ineligible** nodes, a lightweight 4-D promotion fallback is used instead:
dynamic inputs are Unsqueezed to 4-D, static initializers are padded with leading
1-dims in-place, and a Squeeze is inserted after the node to restore the original
output rank.  This ensures all MatMul/Gemm tensors are 4-D after this transform.

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
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    batch_replace_nodes,
    pad4d,
    run_onnx_file_transform,
)
from ._dla_graph_helpers import check_to_apply_transpose, is_const_dq_input


def _apply_matmul_to_transpose_conv_transpose(model):
    """Replace MatMul/Gemm with Unsqueeze-Transpose-Conv-Transpose-Squeeze."""
    graph = model.graph
    cache = GraphCache(graph)
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

    def update_initializers(graph, name, trans_b=False, is_const_dq_init=False):
        """Reshape weight initializer to a 4-D Conv kernel in-place.

        Maps:
          2D [C,K] → [K,C,1,1]  (or [C,K,1,1] when transB)
          3D [C,K,M] → [M,K,C,1]  (or [C,K,M,1] when transB)
          4D [N,C,K,M] → [M,K,C,N] perm (or [C,K,M,N] when transB)
        """
        for initializer in graph.initializer:
            if initializer.name != name:
                continue
            axis = None
            dq_node = None
            shape = None
            if is_const_dq_init:
                dq_node = next(
                    (
                        n
                        for n in graph.node
                        if n.op_type == "DequantizeLinear" and n.input[0] == name
                    ),
                    None,
                )
                if dq_node is not None:
                    axis = next((attr.i for attr in dq_node.attribute if attr.name == "axis"), None)

            # 2D [C,K]
            if len(initializer.dims) == 2:
                c, k = initializer.dims[0], initializer.dims[1]
                init_arr = numpy_helper.to_array(initializer)
                if not trans_b:
                    init_arr = init_arr.T
                    shape = (k, c, 1, 1)
                    if axis is not None:
                        axis = 1 - axis
                else:
                    shape = (c, k, 1, 1)

            # 3D [C,K,M]
            elif len(initializer.dims) == 3:
                c, k, m = initializer.dims[0], initializer.dims[1], initializer.dims[2]
                init_arr = numpy_helper.to_array(initializer)
                if not trans_b:
                    init_arr = init_arr.T
                    shape = (m, k, c, 1)
                    if axis is not None:
                        axis = 2 - axis
                else:
                    shape = (c, k, m, 1)

            # 4D [N,C,K,M]
            elif len(initializer.dims) == 4:
                n, c, k, m = (
                    initializer.dims[0],
                    initializer.dims[1],
                    initializer.dims[2],
                    initializer.dims[3],
                )
                init_arr = numpy_helper.to_array(initializer)
                if not trans_b:
                    init_arr = np.transpose(init_arr, (3, 2, 1, 0))
                    shape = (m, k, c, n)
                    if axis is not None:
                        axis = 3 - axis
                else:
                    init_arr = np.transpose(init_arr, (1, 2, 3, 0))
                    shape = (c, k, m, n)

            if dq_node is not None:
                for attr in dq_node.attribute:
                    if attr.name == "axis":
                        attr.i = axis
                        dq_node.name = f"{dq_node.name}_axis_updated"

            reshaped_arr = np.reshape(init_arr, shape)
            initializer.CopyFrom(numpy_helper.from_array(reshaped_arr, initializer.name))

    # ── Separate eligible and ineligible nodes in one pass ───────────────────
    eligible_nodes = []
    ineligible_nodes = []
    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if check_to_apply_transpose(node, model, tensor_name_dim_map):
            eligible_nodes.append(node)
        else:
            ineligible_nodes.append(node)

    # ── Fallback: promote inputs/output of ineligible MatMul/Gemm to 4D ─────
    # For each ineligible node:
    #   * Dynamic inputs → Unsqueeze to 4D.
    #   * Static initializer inputs (including DQ-wrapped) → expand dims in-place.
    #   * Output → Squeeze back to original rank when the original was not 4D.
    for node in ineligible_nodes:
        cnt_fb = len(fallback_batch)
        pfx = node.name or f"mm_fallback_{cnt_fb}"
        initializer_names = cache.init_names
        pre: list = []
        post: list = []

        for inp_idx in range(min(2, len(node.input))):
            inp = node.input[inp_idx]
            if not inp:
                continue
            in_shape = cache.get_shape(inp)
            if in_shape is None or len(in_shape) >= 4:
                continue
            rank = len(in_shape)
            # Check whether this is a static initializer (possibly DQ-wrapped)
            dq_src = is_const_dq_input(initializer_names, inp, graph)
            init_name = dq_src if dq_src else (inp if inp in initializer_names else None)
            if init_name is not None:
                # Static weight: expand initializer dims in-place with 1-padding
                for init in graph.initializer:
                    if init.name == init_name:
                        arr = numpy_helper.to_array(init)
                        if arr.ndim < 4:
                            new_shape = pad4d(list(arr.shape))
                            init.CopyFrom(
                                numpy_helper.from_array(arr.reshape(new_shape), init.name)
                            )
                        break
            else:
                # Dynamic input: Unsqueeze to 4D
                axes = list(range(4 - rank))
                axes_init = numpy_helper.from_array(
                    np.array(axes, dtype=np.int64),
                    name=f"{pfx}_in{inp_idx}_unsq_axes_{cnt_fb}",
                )
                add_unique_initializers(graph, unique_initializer, [axes_init])
                inp_4d = f"{pfx}_in{inp_idx}_4d_{cnt_fb}"
                pre.append(
                    helper.make_node(
                        "Unsqueeze",
                        inputs=[inp, axes_init.name],
                        outputs=[inp_4d],
                        name=f"{pfx}_in{inp_idx}_unsq_{cnt_fb}",
                    )
                )
                node.input[inp_idx] = inp_4d

        # Output: Squeeze back to original rank
        out_name = node.output[0] if node.output else None
        if out_name:
            out_shape = cache.get_shape(out_name)
            if out_shape is not None and len(out_shape) < 4:
                temp_out = f"{pfx}_out_4d_{cnt_fb}"
                sq_axes = list(range(4 - len(out_shape)))
                sq_axes_init = numpy_helper.from_array(
                    np.array(sq_axes, dtype=np.int64),
                    name=f"{pfx}_out_sq_axes_{cnt_fb}",
                )
                add_unique_initializers(graph, unique_initializer, [sq_axes_init])
                post.append(
                    helper.make_node(
                        "Squeeze",
                        inputs=[temp_out, sq_axes_init.name],
                        outputs=[out_name],
                        name=f"{pfx}_out_sq_{cnt_fb}",
                    )
                )
                node.output[0] = temp_out

        if pre or post:
            fallback_batch[node_idx[id(node)]] = ([node], [*pre, node, *post])

    for node in eligible_nodes:
        cnt = len(replacements_batch)
        pfx = node.name or f"matmul_{cnt}"
        nodes_to_add = []
        initializer_names = cache.init_names

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
        assert input_shape is not None, f"Shape unknown for {node.input[0]!r}"
        orig_rank = len(input_shape)

        # ── Step 1: Unsqueeze activation to 4D ───────────────────────────────
        # 2D [M,N]   → axes [0,1] → [1,1,M,N]
        # 3D [B,M,N] → axes [0]   → [1,B,M,N]
        activation_4d = node.input[0]
        if orig_rank < 4:
            axes = [0, 1] if orig_rank == 2 else [0]
            axes_init = numpy_helper.from_array(
                np.array(axes, dtype=np.int64), name=f"{pfx}_unsq_axes_{cnt}"
            )
            add_unique_initializers(graph, unique_initializer, [axes_init])
            activation_4d = f"{pfx}_unsqueeze4d_{cnt}"
            nodes_to_add.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_init.name],
                    outputs=[activation_4d],
                    name=f"{pfx}_unsqueeze4d_node_{cnt}",
                )
            )

        # ── Step 2: Transpose activation (guaranteed 4D at this point) ───────
        # transA=False: [1,B,M,N] → perm[0,3,2,1] → [1,N,M,B]  (C_in=N, H=M, W=B)
        # transA=True : [1,B,N,M] → perm[0,2,3,1] → [1,N,M,B]  (same result)
        perm_a = [0, 2, 3, 1] if trans_a else [0, 3, 2, 1]
        transposed_a = f"{pfx}_tr_before0_{cnt}"
        nodes_to_add.append(
            helper.make_node(
                "Transpose",
                inputs=[activation_4d],
                outputs=[transposed_a],
                name=f"{pfx}_transpose_before0_{cnt}",
                perm=perm_a,
            )
        )

        # ── Step 3: Prepare weight ────────────────────────────────────────────
        dq_init_name = is_const_dq_input(initializer_names, node.input[1], graph)
        is_const_dq = dq_init_name is not None
        init_name = (
            dq_init_name
            if is_const_dq
            else (node.input[1] if node.input[1] in initializer_names else None)
        )

        if init_name is not None:
            # Static weight: reshape initializer to 4-D conv kernel format
            update_initializers(graph, init_name, trans_b, is_const_dq)
            conv_input1 = node.input[1]  # tensor name unchanged; dims updated
        else:
            # Dynamic weight: add a Transpose node before Conv
            # (check_to_apply_transpose normally prevents reaching here)
            perm_b = [2, 3, 0, 1] if trans_b else [3, 2, 0, 1]
            transposed_b = f"{pfx}_tr_before1_{cnt}"
            nodes_to_add.append(
                helper.make_node(
                    "Transpose",
                    inputs=[node.input[1]],
                    outputs=[transposed_b],
                    name=f"{pfx}_transpose_before1_{cnt}",
                    perm=perm_b,
                )
            )
            conv_input1 = transposed_b

        # ── Step 4: Build Conv inputs (include optional Gemm bias) ───────────
        conv_inputs = [transposed_a, conv_input1]
        if len(node.input) >= 3 and node.input[2]:
            bias_shape = cache.get_shape(node.input[2])
            if bias_shape is not None and len(bias_shape) != 1:
                raise ValueError(
                    f"[matmul_to_conv] Bias for {node.name!r} must be 1-D, got {bias_shape}"
                )
            conv_inputs.append(node.input[2])

        # ── Step 5: Conv ──────────────────────────────────────────────────────
        conv_out = f"{pfx}_conv_out_{cnt}"
        nodes_to_add.append(
            helper.make_node(
                "Conv",
                inputs=conv_inputs,
                outputs=[conv_out],
                name=f"{pfx}_conv_{cnt}",
            )
        )

        # ── Step 6: Output Transpose ──────────────────────────────────────────
        # [1,K,M,B] → perm[0,3,2,1] → [1,B,M,K]
        # Write to a temp name when Squeeze follows; otherwise directly to node.output[0].
        tr_after_out = f"{pfx}_tr_after_out_{cnt}" if orig_rank < 4 else node.output[0]
        nodes_to_add.append(
            helper.make_node(
                "Transpose",
                inputs=[conv_out],
                outputs=[tr_after_out],
                name=f"{pfx}_transpose_after_{cnt}",
                perm=[0, 3, 2, 1],
            )
        )

        # ── Step 7: Squeeze back to original rank ─────────────────────────────
        # 2D case: [1,1,M,K] → squeeze [0,1] → [M,K]
        # 3D case: [1,B,M,K] → squeeze [0]   → [B,M,K]
        if orig_rank < 4:
            sq_axes = [0, 1] if orig_rank == 2 else [0]
            sq_axes_init = numpy_helper.from_array(
                np.array(sq_axes, dtype=np.int64), name=f"{pfx}_sq_axes_{cnt}"
            )
            add_unique_initializers(graph, unique_initializer, [sq_axes_init])
            nodes_to_add.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[tr_after_out, sq_axes_init.name],
                    outputs=list(node.output),
                    name=f"{pfx}_squeeze_after_{cnt}",
                )
            )

        replacements_batch[node_idx[id(node)]] = ([node], nodes_to_add)

    # ── Apply all replacements ────────────────────────────────────────────────
    all_batch = {**replacements_batch, **fallback_batch}
    batch_replace_nodes(graph, all_batch)

    logger.debug(
        "Replaced %d MatMul/Gemm nodes with Transpose-Conv-Transpose nodes; "
        "promoted %d ineligible MatMul/Gemm nodes to 4D.",
        len(replacements_batch),
        len(fallback_batch),
    )
    return model


def dla_matmul_to_transpose_conv_transpose(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply graph transform, save to ``output_path``."""
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_matmul_to_transpose_conv_transpose,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
