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

"""The three separately ordered five-dimensional reshape pattern passes."""

import numpy as np
from onnx import TensorProto, helper

from ..utils.graph_utils import GraphCache, _attribute, add_initializer
from .dla_5d_reshape_to_4d import _build_reduce_replacement_or_raise

__all__ = []


def _follower(cache, output, op_type):
    consumers = cache.get_consumers(output)
    if output in cache.graph_output_names or len(consumers) != 1:
        return None
    return next(
        (
            node
            for node, port in cache.get_consumers(output)
            if port == 0 and node.op_type == op_type
        ),
        None,
    )


def _replace(graph, replacements):
    nodes = list(graph.node)
    del graph.node[:]
    for node in nodes:
        graph.node.extend(replacements.get(id(node), [node]))


def _apply_reshape_reduce_sum(model):
    return _reshape_reduce_sum(model, quantized=False)


def _apply_reshape_qdq_reduce_sum(model):
    return _reshape_reduce_sum(model, quantized=True)


def _reshape_reduce_sum(model, *, quantized):
    graph = model.graph
    cache = GraphCache(graph)
    replacements = {}
    count = 0
    for reshape in graph.node:
        if reshape.op_type != "Reshape":
            continue
        shape = cache.get_init_array(reshape.input[1])
        if shape is None or shape.size != 5:
            continue
        middle = []
        output = reshape.output[0]
        if quantized:
            quantize = _follower(cache, output, "QuantizeLinear")
            if quantize is None:
                continue
            dequantize = _follower(cache, quantize.output[0], "DequantizeLinear")
            if dequantize is None:
                continue
            middle = [quantize, dequantize]
            output = dequantize.output[0]
        reduce = _follower(cache, output, "ReduceSum")
        if reduce is None:
            continue
        # This reference pattern splits the final width of eight into two fours.
        input_shape = cache.get_shape(reshape.input[0])
        axes_values = cache.get_init_array(reduce.input[1]) if len(reduce.input) > 1 else None
        keepdims = _attribute(reduce, "keepdims")
        fast_path = (
            input_shape is not None
            and len(input_shape) == 4
            and input_shape[-1] == 8
            and shape.tolist() == [*input_shape[:-1], 2, 4]
            and axes_values is not None
            and axes_values.reshape(-1).tolist() == [-1]
            and keepdims is not None
            and keepdims.i == 0
        )
        if not fast_path:
            while any(
                name.startswith("u5d_rs_") and name.endswith(f"_{count}")
                for name in cache.init_names
            ):
                count += 1
            removed, added = _build_reduce_replacement_or_raise(
                model, reshape, middle, reduce, set(cache.init_names), count, cache
            )
            replacements.update({id(node): [] for node in removed})
            replacements[id(reshape)] = added
            count += 1
            continue
        prefix = f"reshape_{'qdq_' if quantized else ''}reduce_{count}"
        axes = add_initializer(
            graph,
            prefix + "_slice_axes",
            np.asarray([3], dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        nodes, outputs = [], []
        for branch in range(2):
            start = add_initializer(
                graph,
                f"{prefix}_{branch}_starts",
                np.asarray([branch * 4], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            end = add_initializer(
                graph,
                f"{prefix}_{branch}_ends",
                np.asarray([(branch + 1) * 4], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            current = f"{prefix}_{branch}_slice"
            nodes.append(
                helper.make_node(
                    "Slice", [reshape.input[0], start, end, axes], [current], name=current
                )
            )
            for index, original in enumerate(middle):
                name = f"{prefix}_{branch}_qdq_{index}"
                # The C++ reference uses the quantizer's scale and zero point for both.
                nodes.append(
                    helper.make_node(
                        original.op_type, [current, *middle[0].input[1:3]], [name], name=name
                    )
                )
                current = name
            output = f"{prefix}_{branch}_reduce"
            nodes.append(
                helper.make_node("ReduceSum", [current, *reduce.input[1:]], [output], name=output)
            )
            outputs.append(output)
        nodes.append(
            helper.make_node(
                "Concat", outputs, list(reduce.output), name=prefix + "_axis_updated", axis=3
            )
        )
        for original in [*middle, reduce]:
            replacements[id(original)] = []
        replacements[id(reshape)] = nodes
        count += 1
    _replace(graph, replacements)
    return model


def _apply_reshape_transpose_reshape(model):
    graph = model.graph
    cache = GraphCache(graph)
    replacements = {}
    count = 0
    for first in graph.node:
        if first.op_type != "Reshape" or "_transformed" in first.name:
            continue
        shape = cache.get_init_array(first.input[1])
        if shape is None or shape.size != 5:
            continue
        transpose = _follower(cache, first.output[0], "Transpose")
        if transpose is None:
            continue
        perm = _attribute(transpose, "perm")
        if perm is None:
            continue
        last = _follower(cache, transpose.output[0], "Reshape")
        if last is None:
            continue
        target = cache.get_init_array(last.input[1])
        if target is None:
            continue
        input_shape = cache.get_shape(first.input[0])
        if not input_shape or input_shape[0] != int(shape[0]):
            raise ValueError(f"{first.name!r}: leading input dimension must equal 5D slice count")
        if shape[0] <= 0 or not perm.ints or perm.ints[0] != 0:
            raise ValueError(f"{first.name!r}: unsupported leading dimension/permutation")
        prefix = f"reshape_transpose_{count}"
        axes = add_initializer(
            graph,
            prefix + "_slice_axes",
            np.asarray([0], dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        middle_shape = add_initializer(
            graph,
            prefix + "_4d_shape",
            np.asarray(shape.tolist()[1:], dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        final_shape = add_initializer(
            graph,
            prefix + "_final_shape",
            np.asarray(target, dtype=np.int64),
            dtype=TensorProto.INT64,
            with_value_info=False,
        )
        nodes, outputs = [], []
        for branch in range(int(shape[0])):
            start = add_initializer(
                graph,
                f"{prefix}_{branch}_start",
                np.asarray([branch], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            end = add_initializer(
                graph,
                f"{prefix}_{branch}_end",
                np.asarray([branch + 1], dtype=np.int64),
                dtype=TensorProto.INT64,
                with_value_info=False,
            )
            sliced, reshaped, transposed = [
                f"{prefix}_{branch}_{s}" for s in ("slice", "reshape", "transpose")
            ]
            nodes.extend(
                [
                    helper.make_node(
                        "Slice", [first.input[0], start, end, axes], [sliced], name=sliced
                    ),
                    helper.make_node("Reshape", [sliced, middle_shape], [reshaped], name=reshaped),
                    helper.make_node(
                        "Transpose",
                        [reshaped],
                        [transposed],
                        name=transposed,
                        perm=[value - 1 for value in perm.ints[1:]],
                    ),
                ]
            )
            outputs.append(transposed)
        concat = prefix + "_concat"
        nodes.extend(
            [
                helper.make_node(
                    "Concat", outputs, [concat], name=prefix + "_axis_updated", axis=0
                ),
                helper.make_node(
                    "Reshape", [concat, final_shape], list(last.output), name=prefix + "_final"
                ),
            ]
        )
        replacements.update({id(first): nodes, id(transpose): [], id(last): []})
        count += 1
    _replace(graph, replacements)
    return model
