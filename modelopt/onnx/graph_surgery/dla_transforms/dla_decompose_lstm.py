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

"""Expand ONNX ``LSTM`` nodes into a Conv-based subgraph for DLA-style pipelines.

Input/output ranks follow the ONNX ``LSTM`` operator (``layout`` attribute default ``0``).
Full IO semantics and optional inputs (``B``, ``sequence_lens``, ``initial_h``, ``initial_c``,
``P``) are defined in the ONNX spec:

https://onnx.ai/onnx/operators/onnx__LSTM.html

This pass promotes selected inputs to 4D before the internal transpose/conv path:

- **Index 0 (X):** ``[seq_length, batch_size, input_size]`` (3-D).
- **Index 1 (W):** ``[num_directions, 4*hidden_size, input_size]`` (3-D; often an initializer).
- **Index 2 (R):** ``[num_directions, 4*hidden_size, hidden_size]`` (3-D; often an initializer).
- **Index 3 (B), optional:** ``[num_directions, 8*hidden_size]`` (2-D if present).
- **Index 4 (sequence_lens):** not rewritten here (1-D); omitted from promotion.
- **Index 5-6 (initial_h / initial_c), optional:** ``[num_directions, batch_size, hidden_size]`` (3-D).
- **Index 7 (P), optional:** not promoted by this pass.

``layout=1`` swaps batch/sequence layout in the spec; models using non-default layout may need
preprocessing outside this transform.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...logging_config import logger
from ._common import (
    GraphCache,
    add_unique_initializers,
    batch_replace_nodes,
    get_constant_by_name,
    run_onnx_file_transform,
)
from ._dla_graph_helpers import is_const_dq_input

# Promote ONNX LSTM inputs listed at https://onnx.ai/onnx/operators/onnx__LSTM.html (see module docstring).
# Initializers / DQ weights: numpy expand-to-4D. Graph tensors: Unsqueeze (axis 0 for typical 3-D X / states).
_LSTM_PROMOTE_INPUT_INDICES = (0, 1, 2, 3, 5, 6)


def _expand_array_to_rank4(arr: np.ndarray) -> np.ndarray:
    rank = len(arr.shape)
    if rank >= 4:
        return arr
    return np.expand_dims(arr, axis=list(range(4 - rank)))


def _promote_lstm_input_to_rank4(
    model, graph, lstm_node, input_index, prep_nodes, unique_initializer, cache: GraphCache
) -> None:
    """Promote one LSTM input slot to rank 4 for the decomposition path.

    Shapes follow ONNX ``LSTM`` (layout 0): https://onnx.ai/onnx/operators/onnx__LSTM.html
    Initializers and DQ-backed weights are expanded in numpy; runtime tensors get ``Unsqueeze``.
    """
    if input_index >= len(lstm_node.input):
        return
    tensor_name = lstm_node.input[input_index]
    if not tensor_name:
        return

    init_list = list(cache.init_names)
    dq_init_name = is_const_dq_input(init_list, tensor_name, graph)

    if dq_init_name is not None:
        deq_node = next(
            (
                n
                for n in graph.node
                if n.op_type == "DequantizeLinear"
                and n.output[0] == tensor_name
                and n.input[0] == dq_init_name
            ),
            None,
        )
        if deq_node is None:
            return
        initializer = cache.get_init(dq_init_name)
        if initializer is None:
            return
        init_arr = numpy_helper.to_array(initializer)
        init_arr = _expand_array_to_rank4(init_arr)
        new_init = numpy_helper.from_array(
            init_arr, name=initializer.name + f"_lstm4d_{lstm_node.name}_{input_index}"
        )
        add_unique_initializers(graph, unique_initializer, [new_init])
        if len(cache.get_consumers(initializer.name)) <= 1:
            graph.initializer.remove(initializer)
        deq_node.input[0] = new_init.name
        return

    if cache.is_init(tensor_name):
        initializer = cache.get_init(tensor_name)
        if initializer is None:
            return
        init_arr = numpy_helper.to_array(initializer)
        new_arr = _expand_array_to_rank4(init_arr)
        new_init = numpy_helper.from_array(
            new_arr, name=initializer.name + f"_lstm4d_{lstm_node.name}_{input_index}"
        )
        add_unique_initializers(graph, unique_initializer, [new_init])
        if len(cache.get_consumers(tensor_name)) <= 1:
            graph.initializer.remove(initializer)
        lstm_node.input[input_index] = new_init.name
        return

    # Graph tensor: always prepend singleton dims to reach 4D. LSTM activations are expected 3D — one Unsqueeze(0)
    # is enough when rank is unknown or 3; ranks 1-2 still use multiple leading axes; skip if already 4D+.
    shape = cache.get_shape(tensor_name)
    if shape is not None and len(shape) >= 4:
        return
    axes_list = [0] if shape is None or len(shape) == 3 else list(range(4 - len(shape)))

    axes_name = f"{tensor_name}_lstm4d_axes_{lstm_node.name}_{input_index}"
    out_name = f"{tensor_name}_lstm4d_{lstm_node.name}_{input_index}"
    axes_init = numpy_helper.from_array(np.array(axes_list, dtype=np.int64), name=axes_name)
    add_unique_initializers(graph, unique_initializer, [axes_init])
    unsq = helper.make_node(
        "Unsqueeze",
        [tensor_name, axes_name],
        [out_name],
        name=f"lstm_prep_unsqueeze_{lstm_node.name}_{input_index}",
    )
    prep_nodes.append(unsq)
    lstm_node.input[input_index] = out_name
    elem_t = cache.get_dtype(tensor_name)
    if elem_t is not None and shape is not None:
        new_shape = [int(x) for x in shape]
        for ax in sorted(axes_list):
            new_shape.insert(int(ax), 1)
        graph.value_info.append(helper.make_tensor_value_info(out_name, int(elem_t), new_shape))


def _apply_decompose_lstm(model):
    """Decompose ``LSTM`` nodes into basic ONNX ops (IO per ONNX LSTM layout 0)."""
    graph = model.graph
    cache = GraphCache(graph)
    lstm_nodes = [n for n in list(graph.node) if n.op_type == "LSTM"]
    node_idx = {id(n): i for i, n in enumerate(graph.node)}
    replacements_batch: dict[int, tuple[list, list]] = {}
    lstm_counter = 0
    unique_initializer = set()

    def add_nodes_for_time_step(
        time_step,
        h_t,
        c_t,
        z_t_iofc,
        r_t_iofc,
        nodes_to_add,
        node,
        prefix="forward",
        shape_info_map=None,
    ):
        # add it = sigmoid(Ht * R_i + Z_i)
        matmul_h_r_i_node = helper.make_node(
            "Conv",
            [h_t, r_t_iofc["i"]],
            [f"{prefix}_h_r_i_{node.name}_seq_{time_step}"],
            name=f"{prefix}_conv_h_r_i_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(matmul_h_r_i_node)
        add_h_r_i_z_i_node = helper.make_node(
            "Add",
            [matmul_h_r_i_node.output[0], z_t_iofc["i"]],
            [f"{prefix}_h_r_i_z_i_{node.name}_seq_{time_step}"],
            name=f"{prefix}_add_h_r_i_z_i_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(add_h_r_i_z_i_node)
        i_t_node = helper.make_node(
            "Sigmoid",
            [add_h_r_i_z_i_node.output[0]],
            [f"{prefix}_i_t_{node.name}_seq_{time_step}"],
            name=f"{prefix}_sigmoid_i_t_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(i_t_node)

        # add ft = sigmoid(Ht * R_f + Z_f)
        matmul_h_r_f_node = helper.make_node(
            "Conv",
            [h_t, r_t_iofc["f"]],
            [f"{prefix}_h_r_f_{node.name}_seq_{time_step}"],
            name=f"{prefix}_conv_h_r_f_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(matmul_h_r_f_node)
        add_h_r_f_z_f_node = helper.make_node(
            "Add",
            [matmul_h_r_f_node.output[0], z_t_iofc["f"]],
            [f"{prefix}_h_r_f_z_f_{node.name}_seq_{time_step}"],
            name=f"{prefix}_add_h_r_f_z_f_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(add_h_r_f_z_f_node)
        ft_node = helper.make_node(
            "Sigmoid",
            [add_h_r_f_z_f_node.output[0]],
            [f"{prefix}_ft_{node.name}_seq_{time_step}"],
            name=f"{prefix}_sigmoid_ft_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(ft_node)

        # add ot = sigmoid(Ht * R_o + Z_o)
        matmul_h_r_o_node = helper.make_node(
            "Conv",
            [h_t, r_t_iofc["o"]],
            [f"{prefix}_h_r_o_{node.name}_seq_{time_step}"],
            name=f"{prefix}_conv_h_r_o_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(matmul_h_r_o_node)
        add_h_r_o_z_o_node = helper.make_node(
            "Add",
            [matmul_h_r_o_node.output[0], z_t_iofc["o"]],
            [f"{prefix}_h_r_o_z_o_{node.name}_seq_{time_step}"],
            name=f"{prefix}_add_h_r_o_z_o_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(add_h_r_o_z_o_node)
        ot_node = helper.make_node(
            "Sigmoid",
            [add_h_r_o_z_o_node.output[0]],
            [f"{prefix}_ot_{node.name}_seq_{time_step}"],
            name=f"{prefix}_sigmoid_ot_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(ot_node)

        # add ct = tanh(Ht * R_c + Z_c)
        matmul_h_r_c_node = helper.make_node(
            "Conv",
            [h_t, r_t_iofc["c"]],
            [f"{prefix}_h_r_c_{node.name}_seq_{time_step}"],
            name=f"{prefix}_conv_h_r_c_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(matmul_h_r_c_node)
        add_h_r_c_z_c_node = helper.make_node(
            "Add",
            [matmul_h_r_c_node.output[0], z_t_iofc["c"]],
            [f"{prefix}_h_r_c_z_c_{node.name}_seq_{time_step}"],
            name=f"{prefix}_add_h_r_c_z_c_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(add_h_r_c_z_c_node)
        ct_node = helper.make_node(
            "Tanh",
            [add_h_r_c_z_c_node.output[0]],
            [f"{prefix}_ct_{node.name}_seq_{time_step}"],
            name=f"{prefix}_tanh_ct_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(ct_node)

        # add Ct = ft * Ct-1 + it * ct
        mul_ft_ct_1_node = helper.make_node(
            "Mul",
            [ft_node.output[0], c_t],
            [f"{prefix}_ft_ct_1_{node.name}_seq_{time_step}"],
            name=f"{prefix}_mul_ft_ct_1_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(mul_ft_ct_1_node)
        mul_it_ct_node = helper.make_node(
            "Mul",
            [i_t_node.output[0], ct_node.output[0]],
            [f"{prefix}_it_ct_{node.name}_seq_{time_step}"],
            name=f"{prefix}_mul_it_ct_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(mul_it_ct_node)
        add_ft_ct_1_it_ct_node = helper.make_node(
            "Add",
            [mul_ft_ct_1_node.output[0], mul_it_ct_node.output[0]],
            [f"{prefix}_C_t_{node.name}_seq_{time_step}"],
            name=f"{prefix}_add_ft_ct_1_it_ct_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(add_ft_ct_1_it_ct_node)

        # add Ht = ot * tanh(Ct)
        tanh_ct_node = helper.make_node(
            "Tanh",
            [add_ft_ct_1_it_ct_node.output[0]],
            [f"{prefix}_tanh_Ct_{node.name}_seq_{time_step}"],
            name=f"{prefix}_tanh_Ct_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(tanh_ct_node)
        mul_ot_tanh_ct_node = helper.make_node(
            "Mul",
            [ot_node.output[0], tanh_ct_node.output[0]],
            [f"{prefix}_H_t_{node.name}_seq_{time_step}"],
            name=f"{prefix}_mul_ot_tanh_ct_{node.name}_seq_{time_step}",
        )
        nodes_to_add.append(mul_ot_tanh_ct_node)

        return mul_ot_tanh_ct_node.output[0], add_ft_ct_1_it_ct_node.output[0]

    def transpose_tensor(graph, tensor_name, perm, nodes_to_add, unique_initializer):
        init_list = [init.name for init in graph.initializer]
        if is_const_dq_input(init_list, tensor_name, graph) is not None:
            init_name = is_const_dq_input(init_list, tensor_name, graph)
            deq_node = next(
                (
                    n
                    for n in graph.node
                    if n.op_type == "DequantizeLinear"
                    and n.output[0] == tensor_name
                    and n.input[0] == init_name
                ),
                None,
            )
            if deq_node is None:
                raise ValueError(
                    f"LSTM transpose_tensor: DequantizeLinear for DQ input {tensor_name!r} (init {init_name!r}) "
                    "not found in graph."
                )
            initializer = cache.get_init(init_name)
            if initializer is None:
                raise ValueError(
                    f"LSTM transpose_tensor: initializer {init_name!r} missing for DQ tensor {tensor_name!r}."
                )
            init_arr = numpy_helper.to_array(initializer)
            init_arr = _expand_array_to_rank4(init_arr)

            init_arr = np.transpose(init_arr, axes=perm)
            new_init = numpy_helper.from_array(init_arr, name=initializer.name + "_transposed")
            add_unique_initializers(graph, unique_initializer, [new_init])
            if len(cache.get_consumers(initializer.name)) <= 1:
                graph.initializer.remove(initializer)
            deq_node.input[0] = new_init.name
            return tensor_name
        elif tensor_name in init_list:
            initializer = cache.get_init(tensor_name)
            if initializer is None:
                raise ValueError(
                    f"LSTM transpose_tensor: initializer {tensor_name!r} missing from graph."
                )
            init_arr = numpy_helper.to_array(initializer)
            init_arr = _expand_array_to_rank4(init_arr)
            init_arr = np.transpose(init_arr, axes=perm)
            new_init = numpy_helper.from_array(init_arr, name=initializer.name + "_transposed")
            add_unique_initializers(graph, unique_initializer, [new_init])
            if len(cache.get_consumers(tensor_name)) <= 1:
                graph.initializer.remove(initializer)
            return new_init.name
        else:
            transpose_node = helper.make_node(
                "Transpose",
                inputs=[tensor_name],
                outputs=[tensor_name + "_transposed"],
                name=tensor_name + "_transpose",
                perm=perm,
            )
            nodes_to_add.append(transpose_node)
            return transpose_node.output[0]

    def transpose_inputs(graph, node, unique_initializer, shape_info_map, nodes_to_add):
        # return list of names of [X, W, R, B, ih, ic]

        # When batch_size==1 the batch dim is already a leading unary dim after
        # 4D promotion.  Use [0,3,2,1] to keep it in place instead of swapping
        # it to the back with [2,3,0,1].
        transpose_perm1 = [0, 3, 2, 1] if shape_info_map.get("N", 0) == 1 else [2, 3, 0, 1]
        transpose_perm2 = [2, 3, 1, 0]
        transpose_perm3 = [0, 3, 2, 1]

        # transpose X, ih, ic with perm1
        x_transposed = transpose_tensor(
            graph, node.input[0], transpose_perm1, nodes_to_add, unique_initializer
        )
        # transpose W, R with perm2
        w_transposed = transpose_tensor(
            graph, node.input[1], transpose_perm2, nodes_to_add, unique_initializer
        )
        r_transposed = transpose_tensor(
            graph, node.input[2], transpose_perm2, nodes_to_add, unique_initializer
        )
        # transpose using perm3 and split into bw and br
        bw_transposed = None
        br_transposed = None
        if len(node.input) > 3 and node.input[3]:
            b_transposed = transpose_tensor(
                graph, node.input[3], transpose_perm3, nodes_to_add, unique_initializer
            )

            bw_br_split_init = numpy_helper.from_array(
                np.array([4 * shape_info_map["H"], 4 * shape_info_map["H"]], dtype=np.int64),
                name=f"bw_br_split_init_{node.name}",
            )
            add_unique_initializers(graph, unique_initializer, [bw_br_split_init])
            bw_br_split_node = helper.make_node(
                "Split",
                [b_transposed, bw_br_split_init.name],
                [f"bw_{node.name}", f"br_{node.name}"],
                axis=1,
                name=f"split_B_{node.name}_axis_updated",
            )
            nodes_to_add.append(bw_br_split_node)
            bw_transposed = bw_br_split_node.output[0]
            br_transposed = bw_br_split_node.output[1]

        ih_transposed = None
        if len(node.input) > 5 and node.input[5]:
            ih_transposed = transpose_tensor(
                graph, node.input[5], transpose_perm2, nodes_to_add, unique_initializer
            )
        else:
            ih_init = numpy_helper.from_array(
                np.zeros(
                    [shape_info_map["N"], shape_info_map["H"], shape_info_map["d"], 1],
                    dtype=np.float32,
                ),
                name=node.name + "/ih",
            )
            add_unique_initializers(graph, unique_initializer, [ih_init])
            ih_transposed = ih_init.name

        if len(node.input) > 6 and node.input[6]:
            ic_transposed = transpose_tensor(
                graph, node.input[6], transpose_perm2, nodes_to_add, unique_initializer
            )
        else:
            ic_init = numpy_helper.from_array(
                np.zeros(
                    [shape_info_map["N"], shape_info_map["H"], shape_info_map["d"], 1],
                    dtype=np.float32,
                ),
                name=node.name + "/ic",
            )
            add_unique_initializers(graph, unique_initializer, [ic_init])
            ic_transposed = ic_init.name
        if bw_transposed is None and br_transposed is None:
            # Optional B (ONNX LSTM): zero biases broadcast with Add(bw, br) and Add(..., conv_out).
            zscalar = np.zeros((1, 1, 1, 1), dtype=np.float32)
            bw_init = numpy_helper.from_array(
                zscalar,
                name=(node.name + "/zero_bw").replace("/", "_"),
            )
            br_init = numpy_helper.from_array(
                np.array(zscalar, copy=True),
                name=(node.name + "/zero_br").replace("/", "_"),
            )
            add_unique_initializers(graph, unique_initializer, [bw_init, br_init])
            bw_transposed = bw_init.name
            br_transposed = br_init.name
        elif bw_transposed is None or br_transposed is None:
            raise ValueError(
                f"LSTM {node.name!r}: bias B produced only one of bw/br; unsupported partial bias."
            )
        return {
            "x": x_transposed,
            "w": w_transposed,
            "r": r_transposed,
            "bw": bw_transposed,
            "br": br_transposed,
            "ih": ih_transposed,
            "ic": ic_transposed,
        }

    def resolve_lstm_tensor_shape(tensor_name, _seen=None):
        """Static shape for LSTM X/W when value_info is missing (DQ, Unsqueeze prep, initializers)."""
        if _seen is None:
            _seen = set()
        if not tensor_name or tensor_name in _seen:
            return None
        _seen.add(tensor_name)

        s = cache.get_shape(tensor_name)
        if s is not None:
            return list(s)

        if cache.is_init(tensor_name):
            init = cache.get_init(tensor_name)
            if init is None:
                return None
            return list(numpy_helper.to_array(init).shape)

        init_names = list(cache.init_names)
        dq_src = is_const_dq_input(init_names, tensor_name, graph)
        if dq_src:
            return resolve_lstm_tensor_shape(dq_src, _seen)

        prod = cache.get_producer(tensor_name)
        if prod is None:
            return None
        if prod.op_type == "Unsqueeze" and len(prod.input) >= 2:
            inner = prod.input[0]
            axes_name = prod.input[1]
            inner_shape = resolve_lstm_tensor_shape(inner, _seen)
            if inner_shape is None:
                return None
            ax = cache.get_init_array(axes_name)
            if ax is None:
                ax = get_constant_by_name(model, axes_name)
            if ax is None:
                return None
            axes_flat = np.asarray(ax, dtype=np.int64).flatten().tolist()
            out = list(inner_shape)
            r = len(out)
            norm = [int(a) + r + 1 if int(a) < 0 else int(a) for a in axes_flat]
            for a in sorted(norm):
                if a < 0 or a > len(out):
                    return None
                out.insert(a, 1)
            return out
        if prod.op_type in ("Identity", "Cast") and prod.input and prod.input[0]:
            return resolve_lstm_tensor_shape(prod.input[0], _seen)
        return None

    def create_shape_info_map(node):
        x, w = node.input[:2]
        x_shape = resolve_lstm_tensor_shape(x)
        w_shape = resolve_lstm_tensor_shape(w)
        if x_shape is None or w_shape is None:
            parts = []
            if x_shape is None:
                parts.append(f"X ({x!r})")
            if w_shape is None:
                parts.append(f"W ({w!r})")
            raise ValueError(
                f"LSTM {node.name!r}: missing static shapes for {', '.join(parts)}. "
                "Could not derive ranks from value_info, initializers, get_shape_from_graph, "
                "or Unsqueeze/Identity/Cast chains."
            )
        hidden_size = w_shape[-2] // 4
        num_directions = w_shape[-3]
        input_size = w_shape[-1]
        seq_length = x_shape[-3]
        batch_size = x_shape[-2]
        return {
            "D": input_size,
            "H": hidden_size,
            "d": num_directions,
            "T": seq_length,
            "N": batch_size,
        }

    def split_direction_wise(
        graph, node, unique_initializer, shape_info_map, transposed_inputs, nodes_to_add
    ):
        if shape_info_map["d"] == 1:
            # Unidirectional: reverse and forward paths use the same tensors (only timestep order differs).
            return [transposed_inputs, transposed_inputs]

        direction_split_init = numpy_helper.from_array(
            np.array([1, 1], dtype=np.int64), name=f"direction_split_init_{node.name}"
        )
        add_unique_initializers(graph, unique_initializer, [direction_split_init])

        forward_list = {}
        reverse_list = {}
        forward_list["x"] = transposed_inputs["x"]
        reverse_list["x"] = transposed_inputs["x"]
        to_be_split = ["w", "r", "bw", "br", "ih", "ic"]
        for input_name in to_be_split:
            input_value = transposed_inputs[input_name]
            if not input_value:
                raise ValueError(
                    f"LSTM {node.name!r}: missing tensor for direction split ({input_name!r}); "
                    "required inputs may be absent or empty."
                )
            direction_split_node = helper.make_node(
                "Split",
                [str(input_value), direction_split_init.name],
                [
                    f"forward_direction_split_{input_name}_{node.name}",
                    f"reverse_direction_split_{input_name}_{node.name}",
                ],
                axis=2,
                name=f"split_direction_wise_{input_name}_{node.name}_axis_updated",
            )
            nodes_to_add.append(direction_split_node)
            forward_list[input_name] = direction_split_node.output[0]
            reverse_list[input_name] = direction_split_node.output[1]
        return [forward_list, reverse_list]

    def precompute_and_split_seq_len(
        graph, node, unique_initializer, shape_info_map, input_list, nodes_to_add, direction, prefix
    ):
        x_input = input_list["x"]
        w_input = input_list["w"]
        bw_input = input_list["bw"]
        br_input = input_list["br"]
        if not x_input or not w_input:
            raise ValueError(
                f"LSTM {node.name!r} ({prefix}): missing X or W tensor name after transpose "
                f"(x={x_input!r}, w={w_input!r})."
            )
        if bw_input is None or br_input is None:
            raise ValueError(
                f"LSTM {node.name!r} ({prefix}): missing bias tensors bw/br (bw={bw_input!r}, br={br_input!r})."
            )

        add_bw_br_node = helper.make_node(
            "Add",
            [str(bw_input), str(br_input)],
            [f"{prefix}_add_bw_br_{node.name}"],
            name=f"{prefix}_add_bw_br_{node.name}",
        )
        nodes_to_add.append(add_bw_br_node)

        conv_x_w_node = helper.make_node(
            "Conv",
            [str(x_input), str(w_input)],
            [f"{prefix}_conv_x_w_{node.name}"],
            name=f"{prefix}_conv_x_w_{node.name}",
        )
        nodes_to_add.append(conv_x_w_node)

        z_node = helper.make_node(
            "Add",
            [add_bw_br_node.output[0], conv_x_w_node.output[0]],
            [f"{prefix}_add_bw_br_conv_x_w_{node.name}"],
            name=f"{prefix}_add_bw_br_conv_x_w_{node.name}",
        )
        nodes_to_add.append(z_node)

        split_z_iofc_init = numpy_helper.from_array(
            np.array(
                [
                    shape_info_map["H"],
                    shape_info_map["H"],
                    shape_info_map["H"],
                    shape_info_map["H"],
                ],
                dtype=np.int64,
            ),
            name=f"split_z_iofc_init_{node.name}",
        )
        add_unique_initializers(graph, unique_initializer, [split_z_iofc_init])
        split_z_iofc_node = helper.make_node(
            "Split",
            [z_node.output[0], split_z_iofc_init.name],
            [
                f"{prefix}_split_z_i_{node.name}",
                f"{prefix}_split_z_o_{node.name}",
                f"{prefix}_split_z_f_{node.name}",
                f"{prefix}_split_z_c_{node.name}",
            ],
            axis=1,
            name=f"{prefix}_split_z_iofc_{node.name}_axis_updated",
        )
        nodes_to_add.append(split_z_iofc_node)

        split_seq_init = numpy_helper.from_array(
            np.array([1] * shape_info_map["T"], dtype=np.int64),
            name=f"split_z_iofc_seq_init_{node.name}",
        )
        add_unique_initializers(graph, unique_initializer, [split_seq_init])

        split_z_iofc_seq = {}
        for split_z_iofc_output, split_z_iofc_name in zip(
            split_z_iofc_node.output, ["i", "o", "f", "c"]
        ):
            split_seq_node = helper.make_node(
                "Split",
                [split_z_iofc_output, split_seq_init.name],
                [
                    f"{prefix}_split_z_{split_z_iofc_name}_{time_step}_{node.name}"
                    for time_step in range(shape_info_map["T"])
                ],
                axis=3,
                name=f"{prefix}_split_z_iofc_seq_{split_z_iofc_name}_{node.name}_axis_updated",
            )
            nodes_to_add.append(split_seq_node)
            split_z_iofc_seq[split_z_iofc_name] = (split_seq_node.output)[::direction]

        return split_z_iofc_seq

    def split_r_iofc(
        graph, node, unique_initializer, shape_info_map, input_list, nodes_to_add, prefix
    ):
        r_input = input_list["r"]
        if not r_input:
            raise ValueError(
                f"LSTM {node.name!r} ({prefix}): missing R tensor after transpose (r={r_input!r})."
            )
        split_r_iofc_init = numpy_helper.from_array(
            np.array(
                [
                    shape_info_map["H"],
                    shape_info_map["H"],
                    shape_info_map["H"],
                    shape_info_map["H"],
                ],
                dtype=np.int64,
            ),
            name=f"split_r_iofc_init_{node.name}",
        )
        add_unique_initializers(graph, unique_initializer, [split_r_iofc_init])
        split_r_iofc_node = helper.make_node(
            "Split",
            [str(r_input), split_r_iofc_init.name],
            [
                f"{prefix}_split_r_i_{node.name}",
                f"{prefix}_split_r_o_{node.name}",
                f"{prefix}_split_r_f_{node.name}",
                f"{prefix}_split_r_c_{node.name}",
            ],
            axis=0,
            name=f"{prefix}_split_r_iofc_{node.name}_axis_updated",
        )
        nodes_to_add.append(split_r_iofc_node)
        return {
            "i": split_r_iofc_node.output[0],
            "o": split_r_iofc_node.output[1],
            "f": split_r_iofc_node.output[2],
            "c": split_r_iofc_node.output[3],
        }

    def transpose_outputs(
        graph, node, unique_initializer, input_name, output_name, nodes_to_add, squeeze_axes=None
    ):
        """Apply output Transpose then optional Squeeze for hidden/cell states only.

        Y stays 4D [T, d, N, H] after perm [3,2,0,1]. Y_h / Y_c use squeeze axis 0:
        [1, d, N, H] -> [d, N, H] (ONNX 3D states).
        """
        transpose_perm = [3, 2, 0, 1]
        if squeeze_axes:
            transposed_tmp = f"{output_name}_lstm_tr_{node.name}".replace("/", "_")
            transpose_out = transposed_tmp
        else:
            transpose_out = output_name

        if not input_name:
            raise ValueError(
                f"LSTM {node.name!r}: transpose_outputs got empty input for output {output_name!r}."
            )
        transpose_node = helper.make_node(
            "Transpose",
            [str(input_name)],
            [str(transpose_out)],
            name=f"{output_name}_transpose_{node.name}",
            perm=transpose_perm,
        )
        nodes_to_add.append(transpose_node)

        if squeeze_axes:
            axes_name = f"{output_name}_lstm_squeeze_axes_{node.name}".replace("/", "_")
            axes_init = numpy_helper.from_array(
                np.array(squeeze_axes, dtype=np.int64), name=axes_name
            )
            add_unique_initializers(graph, unique_initializer, [axes_init])
            squeeze_node = helper.make_node(
                "Squeeze",
                [transpose_out, axes_name],
                [output_name],
                name=f"{output_name}_lstm_squeeze_{node.name}",
            )
            nodes_to_add.append(squeeze_node)
            return squeeze_node.output[0]

        new_perm = [0, 1, 2, 3]
        consumer_nodes_y_output = cache.get_consumers(transpose_node.output[0])
        if (
            len(consumer_nodes_y_output) == 1
            and consumer_nodes_y_output[0][0].op_type == "Transpose"
        ):
            transpose_node = consumer_nodes_y_output[0][0]
            for attr in transpose_node.attribute:
                if attr.name == "perm":
                    del attr.ints[:]
                    attr.ints.extend(int(x) for x in new_perm)
                    break

        return transpose_node.output[0]

    for node in lstm_nodes:
        prep_nodes = []
        for idx in _LSTM_PROMOTE_INPUT_INDICES:
            _promote_lstm_input_to_rank4(
                model, graph, node, idx, prep_nodes, unique_initializer, cache
            )
        # Rebuild cache after promotion added new initializers/value_info/nodes
        cache.rebuild()
        nodes_to_add = list(prep_nodes)
        # attributes
        direction = None
        for attr in node.attribute:
            if attr.name == "direction":
                direction = attr.s.decode("utf-8").strip().lower()
                break
        if direction is None:
            direction = "forward"

        shape_info_map = create_shape_info_map(node)
        transposed_inputs = transpose_inputs(
            graph, node, unique_initializer, shape_info_map, nodes_to_add
        )
        forward_list, reverse_list = split_direction_wise(
            graph, node, unique_initializer, shape_info_map, transposed_inputs, nodes_to_add
        )
        y_forward = None
        y_h_forward = None
        y_c_forward = None
        y_reverse = None
        y_h_reverse = None
        y_c_reverse = None
        if direction in ("forward", "bidirectional"):
            prefix = "forward"
            split_z_iofc_seq = precompute_and_split_seq_len(
                graph,
                node,
                unique_initializer,
                shape_info_map,
                forward_list,
                nodes_to_add,
                direction=1,
                prefix=prefix,
            )
            split_r_iofc_map = split_r_iofc(
                graph,
                node,
                unique_initializer,
                shape_info_map,
                forward_list,
                nodes_to_add,
                prefix=prefix,
            )
            h_t, c_t = forward_list["ih"], forward_list["ic"]
            h_t_list = []
            c_t_list = []
            for time_step in range(shape_info_map["T"]):
                split_z_iofc_seq_time_step = {k: v[time_step] for k, v in split_z_iofc_seq.items()}
                h_t, c_t = add_nodes_for_time_step(
                    time_step,
                    h_t,
                    c_t,
                    split_z_iofc_seq_time_step,
                    split_r_iofc_map,
                    nodes_to_add,
                    node,
                    prefix=prefix,
                    shape_info_map=shape_info_map,
                )
                h_t_list.append(h_t)
                c_t_list.append(c_t)

            # concatenate along seq_len dim here axis is 3
            h_t_concat_node = helper.make_node(
                "Concat",
                h_t_list,
                [f"{prefix}_h_t_concat_{node.name}"],
                name=f"{prefix}_concat_h_t_concat_{node.name}_axis_updated",
                axis=3,
            )
            nodes_to_add.append(h_t_concat_node)
            y_forward = h_t_concat_node.output[0]
            y_h_forward = h_t_list[-1]
            y_c_forward = c_t_list[-1]
        if direction in ("reverse", "bidirectional"):
            prefix = "reverse"
            split_z_iofc_seq = precompute_and_split_seq_len(
                graph,
                node,
                unique_initializer,
                shape_info_map,
                reverse_list,
                nodes_to_add,
                direction=-1,
                prefix=prefix,
            )
            split_r_iofc_map = split_r_iofc(
                graph,
                node,
                unique_initializer,
                shape_info_map,
                reverse_list,
                nodes_to_add,
                prefix=prefix,
            )
            h_t, c_t = reverse_list["ih"], reverse_list["ic"]
            h_t_list = []
            c_t_list = []
            for time_step in range(shape_info_map["T"]):
                split_z_iofc_seq_time_step = {k: v[time_step] for k, v in split_z_iofc_seq.items()}
                h_t, c_t = add_nodes_for_time_step(
                    time_step,
                    h_t,
                    c_t,
                    split_z_iofc_seq_time_step,
                    split_r_iofc_map,
                    nodes_to_add,
                    node,
                    prefix=prefix,
                    shape_info_map=shape_info_map,
                )
                h_t_list.append(h_t)
                c_t_list.append(c_t)

            # concatenate along seq_len dim here axis is 3
            h_t_concat_node = helper.make_node(
                "Concat",
                h_t_list[::-1],
                [f"{prefix}_h_t_concat_{node.name}"],
                name=f"{prefix}_concat_h_t_concat_{node.name}_axis_updated",
                axis=3,
            )
            nodes_to_add.append(h_t_concat_node)
            y_reverse = h_t_concat_node.output[0]
            y_h_reverse = h_t_list[-1]
            y_c_reverse = c_t_list[-1]
        if direction not in ("forward", "reverse", "bidirectional"):
            raise ValueError(f"Invalid direction: {direction}")
        if direction == "bidirectional" and int(shape_info_map["d"]) != 2:
            raise ValueError(
                f"LSTM {node.name!r}: direction is bidirectional but num_directions (from W) is "
                f"{shape_info_map['d']!r}; expected 2."
            )

        y_before_transpose = None
        y_h_before_transpose = None
        y_c_before_transpose = None
        if direction == "bidirectional":
            if y_forward is None or y_reverse is None:
                raise ValueError(
                    f"LSTM {node.name!r}: bidirectional concat needs both forward and reverse "
                    f"tensors; got y_forward={y_forward!r}, y_reverse={y_reverse!r}. "
                    "If you still see this after updating, the direction attribute may be "
                    "unusual (expected 'bidirectional')."
                )
            y_concat_node = helper.make_node(
                "Concat",
                [str(y_forward), str(y_reverse)],
                [f"y_concat_{node.name}"],
                name=f"concat_y_concat_{node.name}_axis_updated",
                axis=2,
            )
            nodes_to_add.append(y_concat_node)
            y_before_transpose = y_concat_node.output[0]

            if len(node.output) > 1 and node.output[1]:
                if y_h_forward is None or y_h_reverse is None:
                    raise ValueError(
                        f"LSTM {node.name!r}: bidirectional Y_h concat missing "
                        f"y_h_forward={y_h_forward!r}, y_h_reverse={y_h_reverse!r}"
                    )
                y_h_concat_node = helper.make_node(
                    "Concat",
                    [str(y_h_forward), str(y_h_reverse)],
                    [f"y_h_concat_{node.name}"],
                    name=f"concat_y_h_concat_{node.name}_axis_updated",
                    axis=2,
                )
                nodes_to_add.append(y_h_concat_node)
                y_h_before_transpose = y_h_concat_node.output[0]

            if len(node.output) > 2 and node.output[2]:
                if y_c_forward is None or y_c_reverse is None:
                    raise ValueError(
                        f"LSTM {node.name!r}: bidirectional Y_c concat missing "
                        f"y_c_forward={y_c_forward!r}, y_c_reverse={y_c_reverse!r}"
                    )
                y_c_concat_node = helper.make_node(
                    "Concat",
                    [str(y_c_forward), str(y_c_reverse)],
                    [f"y_c_concat_{node.name}"],
                    name=f"concat_y_c_concat_{node.name}_axis_updated",
                    axis=2,
                )
                nodes_to_add.append(y_c_concat_node)
                y_c_before_transpose = y_c_concat_node.output[0]

        elif direction == "forward":
            y_before_transpose = y_forward
            y_h_before_transpose = y_h_forward
            y_c_before_transpose = y_c_forward
        elif direction == "reverse":
            y_before_transpose = y_reverse
            y_h_before_transpose = y_h_reverse
            y_c_before_transpose = y_c_reverse
        else:
            raise ValueError(f"Invalid direction: {direction}")

        transpose_outputs(
            graph, node, unique_initializer, y_before_transpose, node.output[0], nodes_to_add
        )
        if len(node.output) > 1 and node.output[1]:
            transpose_outputs(
                graph,
                node,
                unique_initializer,
                y_h_before_transpose,
                node.output[1],
                nodes_to_add,
                squeeze_axes=[0],
            )
        if len(node.output) > 2 and node.output[2]:
            transpose_outputs(
                graph,
                node,
                unique_initializer,
                y_c_before_transpose,
                node.output[2],
                nodes_to_add,
                squeeze_axes=[0],
            )

        replacements_batch[node_idx[id(node)]] = ([node], nodes_to_add)
        lstm_counter += 1

    batch_replace_nodes(graph, replacements_batch)
    logger.debug("Updated %d LSTM nodes with manual decomposition", lstm_counter)

    return model


def dla_decompose_lstm(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> onnx.ModelProto:
    """Load ONNX from ``model_path``, apply graph transform, save to ``output_path``.

    Expects standard ONNX ``LSTM`` IO (layout 0); see
    https://onnx.ai/onnx/operators/onnx__LSTM.html
    """
    return run_onnx_file_transform(
        model_path,
        output_path,
        _apply_decompose_lstm,
        use_external_data=use_external_data,
        external_data_name=external_data_name,
        verbose=verbose,
        **kwargs,
    )
