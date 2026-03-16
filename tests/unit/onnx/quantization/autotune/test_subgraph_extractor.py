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

"""Tests for subgraph_extractor module: subgraph extraction and boundary resolution."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.autotune.subgraph_extractor import (
    _find_reachable_graph_inputs,
    extract_subgraph,
    extract_subgraph_by_nodes,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _build_linear_graph():
    """Input -> Conv -> Relu -> MatMul -> Output."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    w_conv = gs.Constant("w_conv", values=np.random.randn(16, 3, 3, 3).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32)
    relu_out = gs.Variable("relu_out", dtype=np.float32)
    w_mm = gs.Constant("w_mm", values=np.random.randn(16, 16).astype(np.float32))
    mm_out = gs.Variable("mm_out", dtype=np.float32)

    conv = gs.Node(op="Conv", name="conv_0", inputs=[inp, w_conv], outputs=[conv_out])
    relu = gs.Node(op="Relu", name="relu_0", inputs=[conv_out], outputs=[relu_out])
    matmul = gs.Node(op="MatMul", name="matmul_0", inputs=[relu_out, w_mm], outputs=[mm_out])

    graph = gs.Graph(nodes=[conv, relu, matmul], inputs=[inp], outputs=[mm_out])
    graph.cleanup().toposort()
    return graph


def _build_diamond_graph():
    """Input -> Conv -> {Add branch1, Add branch2} -> Add -> Output."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 16, 8, 8])
    w = gs.Constant("w", values=np.random.randn(16, 16, 1, 1).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32)
    b1_out = gs.Variable("b1_out", dtype=np.float32)
    b2_out = gs.Variable("b2_out", dtype=np.float32)
    add_out = gs.Variable("add_out", dtype=np.float32)
    bias1 = gs.Constant("bias1", values=np.ones((1,), dtype=np.float32))
    bias2 = gs.Constant("bias2", values=np.ones((1,), dtype=np.float32) * 2)

    conv = gs.Node(op="Conv", name="conv", inputs=[inp, w], outputs=[conv_out])
    add1 = gs.Node(op="Add", name="add1", inputs=[conv_out, bias1], outputs=[b1_out])
    add2 = gs.Node(op="Add", name="add2", inputs=[conv_out, bias2], outputs=[b2_out])
    add_final = gs.Node(op="Add", name="add_final", inputs=[b1_out, b2_out], outputs=[add_out])

    graph = gs.Graph(nodes=[conv, add1, add2, add_final], inputs=[inp], outputs=[add_out])
    graph.cleanup().toposort()
    return graph


# ── extract_subgraph ────────────────────────────────────────────────────────


class TestExtractSubgraph:
    def test_extract_middle_section(self):
        graph = _build_linear_graph()
        sub_bytes = extract_subgraph(graph, ["conv_out"], ["relu_out"])
        sub_model = onnx.load_from_string(sub_bytes)
        assert len(sub_model.graph.node) == 1
        assert sub_model.graph.node[0].op_type == "Relu"
        assert sub_model.graph.input[0].name == "conv_out"
        assert sub_model.graph.output[0].name == "relu_out"

    def test_extract_full_graph(self):
        graph = _build_linear_graph()
        sub_bytes = extract_subgraph(graph, ["input"], ["mm_out"])
        sub_model = onnx.load_from_string(sub_bytes)
        assert len(sub_model.graph.node) >= 3

    def test_invalid_input_raises(self):
        graph = _build_linear_graph()
        with pytest.raises(ValueError, match="not found"):
            extract_subgraph(graph, ["nonexistent"], ["relu_out"])

    def test_invalid_output_raises(self):
        graph = _build_linear_graph()
        with pytest.raises(ValueError, match="not found"):
            extract_subgraph(graph, ["input"], ["nonexistent"])

    def test_returns_valid_onnx_bytes(self):
        graph = _build_linear_graph()
        sub_bytes = extract_subgraph(graph, ["conv_out"], ["relu_out"])
        assert isinstance(sub_bytes, bytes)
        sub_model = onnx.load_from_string(sub_bytes)
        assert len(sub_model.graph.node) > 0
        assert len(sub_model.graph.input) > 0
        assert len(sub_model.graph.output) > 0


# ── extract_subgraph_by_nodes ───────────────────────────────────────────────


class TestExtractSubgraphByNodes:
    def test_single_node(self):
        graph = _build_linear_graph()
        sub_bytes = extract_subgraph_by_nodes(graph, ["relu_0"])
        sub_model = onnx.load_from_string(sub_bytes)
        assert len(sub_model.graph.node) == 1
        assert sub_model.graph.node[0].op_type == "Relu"

    def test_multiple_nodes(self):
        graph = _build_linear_graph()
        sub_bytes = extract_subgraph_by_nodes(graph, ["conv_0", "relu_0"])
        sub_model = onnx.load_from_string(sub_bytes)
        op_types = {n.op_type for n in sub_model.graph.node}
        assert "Conv" in op_types
        assert "Relu" in op_types

    def test_empty_node_list_raises(self):
        graph = _build_linear_graph()
        with pytest.raises(ValueError, match="None of the specified nodes"):
            extract_subgraph_by_nodes(graph, ["nonexistent"])

    def test_diamond_graph_subset(self):
        graph = _build_diamond_graph()
        sub_bytes = extract_subgraph_by_nodes(graph, ["add1", "add2"])
        sub_model = onnx.load_from_string(sub_bytes)
        assert len(sub_model.graph.node) >= 2


# ── _find_reachable_graph_inputs ────────────────────────────────────────────


class TestFindReachableGraphInputs:
    def test_direct_input(self):
        graph = _build_linear_graph()
        conv_node = [n for n in graph.nodes if n.name == "conv_0"][0]
        result = _find_reachable_graph_inputs(graph, [conv_node])
        assert "input" in result

    def test_transitive_input(self):
        graph = _build_linear_graph()
        relu_node = [n for n in graph.nodes if n.name == "relu_0"][0]
        result = _find_reachable_graph_inputs(graph, [relu_node])
        assert "input" in result

    def test_deep_transitive(self):
        graph = _build_linear_graph()
        mm_node = [n for n in graph.nodes if n.name == "matmul_0"][0]
        result = _find_reachable_graph_inputs(graph, [mm_node])
        assert "input" in result
