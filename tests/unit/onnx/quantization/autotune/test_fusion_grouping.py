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

"""Tests for fusion_grouping module: metadata parsing, graph.json, and FusionGroup creation."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.autotune.fusion_grouping import (
    FusionGroup,
    TRTLayer,
    _find_boundary_tensors,
    _parse_metadata,
    create_fusion_groups,
    generate_graph_json,
    parse_graph_json,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _build_conv_relu_graph():
    """Input -> Conv -> Relu -> Output with a weight constant."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("weight", values=np.random.randn(16, 3, 3, 3).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32)
    relu_out = gs.Variable("relu_out", dtype=np.float32)
    output = gs.Variable("output", dtype=np.float32, shape=[1, 16, 6, 6])

    conv = gs.Node(op="Conv", name="conv_0", inputs=[inp, weight], outputs=[conv_out])
    relu = gs.Node(op="Relu", name="relu_0", inputs=[conv_out], outputs=[relu_out])

    graph = gs.Graph(nodes=[conv, relu], inputs=[inp], outputs=[relu_out])
    graph.cleanup().toposort()
    return graph


def _build_two_branch_graph():
    """Input -> Conv -> {Relu -> out1, Sigmoid -> out2} for boundary testing."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("w", values=np.random.randn(16, 3, 3, 3).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32)
    relu_out = gs.Variable("relu_out", dtype=np.float32)
    sig_out = gs.Variable("sig_out", dtype=np.float32)

    conv = gs.Node(op="Conv", name="conv", inputs=[inp, weight], outputs=[conv_out])
    relu = gs.Node(op="Relu", name="relu", inputs=[conv_out], outputs=[relu_out])
    sig = gs.Node(op="Sigmoid", name="sigmoid", inputs=[conv_out], outputs=[sig_out])

    graph = gs.Graph(nodes=[conv, relu, sig], inputs=[inp], outputs=[relu_out, sig_out])
    graph.cleanup().toposort()
    return graph


# ── _parse_metadata ─────────────────────────────────────────────────────────


class TestParseMetadata:
    def test_empty_string(self):
        assert _parse_metadata("") == []

    def test_single_node(self):
        meta = "something [ONNX Layer: /encoder/conv1/Conv]"
        assert _parse_metadata(meta) == ["/encoder/conv1/Conv"]

    def test_multiple_nodes_us_delimiter(self):
        meta = "x [ONNX Layer: nodeA]\x1fy [ONNX Layer: nodeB]"
        assert _parse_metadata(meta) == ["nodeA", "nodeB"]

    def test_multiple_nodes_rs_delimiter(self):
        meta = "x [ONNX Layer: nodeA]\x1ey [ONNX Layer: nodeB]"
        assert _parse_metadata(meta) == ["nodeA", "nodeB"]

    def test_no_onnx_layer_marker(self):
        assert _parse_metadata("some random text") == []

    def test_mixed_delimiters(self):
        meta = "[ONNX Layer: a]\x1f[ONNX Layer: b]\x1e[ONNX Layer: c]"
        assert _parse_metadata(meta) == ["a", "b", "c"]


# ── parse_graph_json ────────────────────────────────────────────────────────


class TestParseGraphJson:
    def test_parse_valid_json(self, tmp_path):
        data = {
            "Layers": [
                {
                    "Name": "conv_fused",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: conv_0]",
                    "Inputs": [{"Name": "input"}],
                    "Outputs": [{"Name": "conv_out"}],
                },
                {
                    "Name": "relu_fused",
                    "LayerType": "CaskPointWise",
                    "Metadata": "[ONNX Layer: relu_0]",
                    "Inputs": [{"Name": "conv_out"}],
                    "Outputs": [{"Name": "output"}],
                },
            ]
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(data))

        layers = parse_graph_json(str(path))
        assert len(layers) == 2
        assert layers[0].name == "conv_fused"
        assert layers[0].onnx_nodes == ["conv_0"]
        assert layers[0].input_names == ["input"]
        assert layers[0].output_names == ["conv_out"]

    def test_parse_empty_layers(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"Layers": []}))
        assert parse_graph_json(str(path)) == []

    def test_layer_without_metadata(self, tmp_path):
        data = {
            "Layers": [
                {
                    "Name": "reformat",
                    "LayerType": "Reformat",
                    "Inputs": [{"Name": "x"}],
                    "Outputs": [{"Name": "y"}],
                }
            ]
        }
        path = tmp_path / "no_meta.json"
        path.write_text(json.dumps(data))
        layers = parse_graph_json(str(path))
        assert len(layers) == 1
        assert layers[0].onnx_nodes == []


# ── create_fusion_groups ────────────────────────────────────────────────────


class TestCreateFusionGroups:
    def test_single_conv_group(self):
        graph = _build_conv_relu_graph()
        trt_layers = [
            TRTLayer(
                name="conv_relu_fused",
                layer_type="CaskConvolution",
                onnx_nodes=["conv_0", "relu_0"],
                input_names=["input"],
                output_names=["relu_out"],
            )
        ]

        groups = create_fusion_groups(trt_layers, graph)
        assert len(groups) == 1
        g = groups[0]
        assert g.has_quantizable_op is True
        assert "conv_0" in g.onnx_node_names
        assert "relu_0" in g.onnx_node_names
        assert "conv_0" in g.quantizable_node_names
        assert "relu_0" not in g.quantizable_node_names

    def test_no_onnx_nodes_skipped(self):
        graph = _build_conv_relu_graph()
        trt_layers = [
            TRTLayer(name="reformat", layer_type="Reformat", onnx_nodes=[], input_names=[], output_names=[])
        ]
        groups = create_fusion_groups(trt_layers, graph)
        assert len(groups) == 0

    def test_unresolved_node_names_skipped(self):
        graph = _build_conv_relu_graph()
        trt_layers = [
            TRTLayer(
                name="ghost",
                layer_type="X",
                onnx_nodes=["nonexistent_node"],
                input_names=[],
                output_names=[],
            )
        ]
        groups = create_fusion_groups(trt_layers, graph)
        assert len(groups) == 0

    def test_boundary_tensors_resolved(self):
        graph = _build_conv_relu_graph()
        trt_layers = [
            TRTLayer(
                name="fused",
                layer_type="CaskConvolution",
                onnx_nodes=["conv_0"],
                input_names=["input"],
                output_names=["conv_out"],
            )
        ]
        groups = create_fusion_groups(trt_layers, graph)
        g = groups[0]
        assert "input" in g.input_tensors
        assert "conv_out" in g.output_tensors


# ── _find_boundary_tensors ──────────────────────────────────────────────────


class TestFindBoundaryTensors:
    def test_single_node_boundaries(self):
        graph = _build_conv_relu_graph()
        conv_node = [n for n in graph.nodes if n.name == "conv_0"][0]
        inputs, outputs = _find_boundary_tensors([conv_node], {"conv_0"}, graph)
        assert "input" in inputs
        assert "conv_out" in outputs

    def test_full_group_boundaries(self):
        graph = _build_conv_relu_graph()
        all_nodes = list(graph.nodes)
        names = {n.name for n in all_nodes}
        inputs, outputs = _find_boundary_tensors(all_nodes, names, graph)
        assert "input" in inputs
        assert len(outputs) > 0

    def test_two_branch_output(self):
        graph = _build_two_branch_graph()
        conv_node = [n for n in graph.nodes if n.name == "conv"][0]
        inputs, outputs = _find_boundary_tensors([conv_node], {"conv"}, graph)
        assert "input" in inputs
        assert "conv_out" in outputs


# ── generate_graph_json ─────────────────────────────────────────────────────


class TestGenerateGraphJson:
    @patch("modelopt.onnx.quantization.autotune.fusion_grouping.subprocess.run")
    def test_success(self, mock_run, tmp_path):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "build ok"
        mock_run.return_value.stderr = ""

        result = generate_graph_json("model.onnx", str(tmp_path))
        assert result.endswith(".fp16.graph.json")
        assert mock_run.called
        cmd_args = mock_run.call_args[0][0]
        assert "trtexec" in cmd_args[0]
        assert any("--onnx=" in a for a in cmd_args)

    @patch("modelopt.onnx.quantization.autotune.fusion_grouping.subprocess.run")
    def test_failure_raises(self, mock_run, tmp_path):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "error"

        with pytest.raises(RuntimeError, match="trtexec FP16 build failed"):
            generate_graph_json("model.onnx", str(tmp_path))

    @patch("modelopt.onnx.quantization.autotune.fusion_grouping.subprocess.run")
    def test_plugin_libraries_passed(self, mock_run, tmp_path):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        generate_graph_json("model.onnx", str(tmp_path), plugin_libraries=["/lib/plugin.so"])
        cmd_args = mock_run.call_args[0][0]
        assert any("--staticPlugins=" in a for a in cmd_args)

    @patch("modelopt.onnx.quantization.autotune.fusion_grouping.subprocess.run")
    def test_extra_args_passed(self, mock_run, tmp_path):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        generate_graph_json("model.onnx", str(tmp_path), extra_trtexec_args=["--fp16", "--verbose"])
        cmd_args = mock_run.call_args[0][0]
        assert "--fp16" in cmd_args
        assert "--verbose" in cmd_args
