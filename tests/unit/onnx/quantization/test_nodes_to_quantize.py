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

"""Tests for the ``nodes_to_quantize`` allow-list filter (symmetric with ``nodes_to_exclude``).

Builds a small two-Conv ONNX graph and asserts that ``nodes_to_quantize=["conv_keep"]`` produces
Q/DQ around ``conv_keep`` only, leaving ``conv_skip`` in its original precision. This is the
primitive the ONNX sensitivity scanner relies on to isolate a single node for a per-target probe.
"""

from __future__ import annotations

import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto, helper, numpy_helper

import modelopt.onnx.quantization as moq


def _build_two_conv_onnx(path: str, opset: int = 17) -> None:
    """Emit a 2-Conv ONNX with the node names the test filters on."""
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
    b1 = np.zeros((4,), dtype=np.float32)
    w2 = rng.standard_normal((4, 4, 3, 3)).astype(np.float32) * 0.1
    b2 = np.zeros((4,), dtype=np.float32)

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "w1", "b1"],
            ["conv_keep_out"],
            name="conv_keep",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "Conv",
            ["conv_keep_out", "w2", "b2"],
            ["output"],
            name="conv_skip",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
    ]
    initializers = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(b2, "b2"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="nodes_to_quantize_test",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=initializers,
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8),
        path,
    )


def _has_dq_predecessor(node: gs.Node, input_idx: int) -> bool:
    """Return True when the input at ``input_idx`` of ``node`` is produced by DequantizeLinear."""
    inp = node.inputs[input_idx]
    if not isinstance(inp, gs.Variable):
        return False
    producer = node.i(input_idx)
    if producer and producer.op == "Cast":
        producer = producer.i(0)
    return bool(producer and producer.op == "DequantizeLinear")


def test_nodes_to_quantize_restricts_qdq_to_single_conv(tmp_path):
    """`nodes_to_quantize=["conv_keep"]` inserts Q/DQ around conv_keep only."""
    onnx_path = str(tmp_path / "two_conv.onnx")
    _build_two_conv_onnx(onnx_path)
    calibration_data = {
        "input": np.random.default_rng(0).standard_normal((2, 3, 8, 8)).astype(np.float32)
    }

    moq.quantize(
        onnx_path,
        quantize_mode="int8",
        calibration_data=calibration_data,
        calibration_eps=["cpu"],
        nodes_to_quantize=["^conv_keep$"],
        high_precision_dtype="fp32",
    )

    quantized_path = onnx_path.replace(".onnx", ".quant.onnx")
    assert os.path.isfile(quantized_path)

    graph = gs.import_onnx(onnx.load(quantized_path))
    keep_nodes = [n for n in graph.nodes if n.name == "conv_keep"]
    skip_nodes = [n for n in graph.nodes if n.name == "conv_skip"]
    assert len(keep_nodes) == 1, (
        f"conv_keep not found in quantized graph: {[n.name for n in graph.nodes]}"
    )
    assert len(skip_nodes) == 1, (
        f"conv_skip not found in quantized graph: {[n.name for n in graph.nodes]}"
    )

    # conv_keep must have DQ on its activation input; conv_skip must not.
    assert _has_dq_predecessor(keep_nodes[0], 0), (
        "conv_keep is not quantized despite nodes_to_quantize=['conv_keep']"
    )
    assert not _has_dq_predecessor(skip_nodes[0], 0), (
        "conv_skip was quantized but nodes_to_quantize only listed conv_keep"
    )
