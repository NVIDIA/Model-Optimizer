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

"""Tests for subgraph_workflow module: shape parsing, profile parsing, cache, QDQ insertion, schemes."""

import json

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.autotune.fusion_grouping import FusionGroup
from modelopt.onnx.quantization.autotune.subgraph_workflow import (
    GroupResult,
    LayerTiming,
    QDQScheme,
    SchemeResult,
    _build_subgraph_shape_args,
    _cache_entry_to_group_result,
    _compute_time_from_profile,
    _extract_shape_specs,
    _group_result_to_cache_entry,
    _is_reformat_layer,
    _load_cache,
    _parse_profile_json,
    _parse_shape_spec,
    _save_cache,
    generate_heuristic_schemes,
    insert_qdq_on_graph,
)


# ── Shape parsing ──────────────────────────────────────────────────────────


class TestParseShapeSpec:
    def test_single_input(self):
        assert _parse_shape_spec("input:1x3x224x224") == {"input": [1, 3, 224, 224]}

    def test_multiple_inputs(self):
        result = _parse_shape_spec("a:1x2x3,b:4x5")
        assert result == {"a": [1, 2, 3], "b": [4, 5]}

    def test_empty_string(self):
        assert _parse_shape_spec("") == {}

    def test_scalar_shape(self):
        assert _parse_shape_spec("x:1") == {"x": [1]}


class TestExtractShapeSpecs:
    def test_all_three_specs(self):
        args = [
            "--minShapes=inp:1x3x32x32",
            "--optShapes=inp:1x3x224x224",
            "--maxShapes=inp:1x3x512x512",
        ]
        min_s, opt_s, max_s = _extract_shape_specs(args)
        assert min_s == {"inp": [1, 3, 32, 32]}
        assert opt_s == {"inp": [1, 3, 224, 224]}
        assert max_s == {"inp": [1, 3, 512, 512]}

    def test_none_args(self):
        min_s, opt_s, max_s = _extract_shape_specs(None)
        assert min_s == opt_s == max_s == {}

    def test_partial_specs(self):
        args = ["--optShapes=x:1x3x8x8", "--verbose"]
        min_s, opt_s, max_s = _extract_shape_specs(args)
        assert min_s == {}
        assert opt_s == {"x": [1, 3, 8, 8]}
        assert max_s == {}


class TestBuildSubgraphShapeArgs:
    def test_full_args(self):
        result = _build_subgraph_shape_args(
            ["input"],
            min_shapes={"input": [1, 3, 32, 32]},
            opt_shapes={"input": [1, 3, 64, 64]},
            max_shapes={"input": [1, 3, 128, 128]},
        )
        assert len(result) == 3
        assert any("--minShapes=" in a for a in result)
        assert any("--optShapes=" in a for a in result)
        assert any("--maxShapes=" in a for a in result)

    def test_no_matching_inputs(self):
        result = _build_subgraph_shape_args(
            ["other_tensor"],
            min_shapes={"input": [1, 3, 32, 32]},
            opt_shapes={"input": [1, 3, 64, 64]},
            max_shapes={},
        )
        assert result is None

    def test_partial_shapes(self):
        result = _build_subgraph_shape_args(
            ["input"], min_shapes={}, opt_shapes={"input": [1, 8]}, max_shapes={},
        )
        assert len(result) == 1
        assert "--optShapes=input:1x8" in result[0]


# ── Profile parsing ────────────────────────────────────────────────────────


class TestIsReformatLayer:
    def test_reformat(self):
        assert _is_reformat_layer("Reformatting CopyNode for Input") is True

    def test_not_reformat(self):
        assert _is_reformat_layer("CaskConvolution_conv_0") is False

    def test_case_insensitive(self):
        assert _is_reformat_layer("REFORMAT_layer") is True


class TestParseProfileJson:
    def test_valid_profile(self, tmp_path):
        data = [
            {"name": "conv_layer", "medianMs": 0.5, "percentage": 80},
            {"name": "Reformatting CopyNode", "medianMs": 0.1, "percentage": 20},
        ]
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(data))

        timings = _parse_profile_json(str(path))
        assert len(timings) == 2
        assert timings[0].name == "conv_layer"
        assert timings[0].median_ms == 0.5

    def test_missing_file(self):
        assert _parse_profile_json("/nonexistent/profile.json") == []

    def test_uses_average_ms_fallback(self, tmp_path):
        data = [{"name": "layer", "averageMs": 1.0, "percentage": 100}]
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(data))
        timings = _parse_profile_json(str(path))
        assert timings[0].median_ms == 1.0


class TestComputeTimeFromProfile:
    def test_filters_reformat(self):
        profile = [
            LayerTiming("conv", 0.5, 60),
            LayerTiming("Reformatting CopyNode", 0.3, 30),
            LayerTiming("relu", 0.1, 10),
        ]
        assert _compute_time_from_profile(profile) == pytest.approx(0.6)

    def test_empty_profile(self):
        assert _compute_time_from_profile([]) == 0.0

    def test_all_reformat(self):
        profile = [LayerTiming("Reformatting A", 0.5, 100)]
        assert _compute_time_from_profile(profile) == 0.0


# ── Cache I/O ──────────────────────────────────────────────────────────────


class TestCacheIO:
    def test_save_and_load(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        data = {"version": 1, "model_path": "test.onnx", "phase2": {"results": []}}
        _save_cache(cache_path, data)
        loaded = _load_cache(cache_path)
        assert loaded == data

    def test_load_nonexistent(self, tmp_path):
        assert _load_cache(tmp_path / "missing.json") is None

    def test_load_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        assert _load_cache(path) is None

    def test_atomic_write(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        _save_cache(cache_path, {"a": 1})
        assert not (tmp_path / "cache.tmp").exists()
        assert cache_path.exists()


class TestCacheEntryConversion:
    def _make_group_result(self):
        group = FusionGroup(id=5, trt_layer_name="conv_fused")
        scheme = QDQScheme(name="full_qdq", target_tensors=["t1", "t2"], description="test")
        return GroupResult(
            group=group,
            baseline_latency_ms=10.0,
            best_scheme=scheme,
            best_latency_ms=8.0,
            baseline_compute_ms=5.0,
            best_compute_ms=4.0,
        )

    def test_roundtrip(self):
        gr = self._make_group_result()
        entry = _group_result_to_cache_entry(gr)
        restored = _cache_entry_to_group_result(entry)
        assert restored.group.id == 5
        assert restored.baseline_latency_ms == 10.0
        assert restored.best_scheme.name == "full_qdq"
        assert restored.best_scheme.target_tensors == ["t1", "t2"]
        assert restored.best_latency_ms == 8.0

    def test_baseline_scheme(self):
        group = FusionGroup(id=1, trt_layer_name="x")
        gr = GroupResult(group=group, baseline_latency_ms=5.0, best_scheme=None, best_latency_ms=5.0)
        entry = _group_result_to_cache_entry(gr)
        restored = _cache_entry_to_group_result(entry)
        assert restored.best_scheme is None


# ── Data classes ───────────────────────────────────────────────────────────


class TestGroupResult:
    def test_speedup(self):
        gr = GroupResult(
            group=FusionGroup(id=0, trt_layer_name="x"),
            baseline_latency_ms=10.0,
            best_scheme=None,
            best_latency_ms=5.0,
        )
        assert gr.speedup == pytest.approx(2.0)

    def test_speedup_with_inf(self):
        gr = GroupResult(
            group=FusionGroup(id=0, trt_layer_name="x"),
            baseline_latency_ms=float("inf"),
            best_scheme=None,
            best_latency_ms=5.0,
        )
        assert gr.speedup == 0.0

    def test_compute_speedup_fallback(self):
        gr = GroupResult(
            group=FusionGroup(id=0, trt_layer_name="x"),
            baseline_latency_ms=10.0,
            best_scheme=None,
            best_latency_ms=5.0,
            baseline_compute_ms=0.0,
            best_compute_ms=0.0,
        )
        assert gr.compute_speedup == gr.speedup

    def test_compute_speedup_direct(self):
        gr = GroupResult(
            group=FusionGroup(id=0, trt_layer_name="x"),
            baseline_latency_ms=10.0,
            best_scheme=None,
            best_latency_ms=8.0,
            baseline_compute_ms=6.0,
            best_compute_ms=3.0,
        )
        assert gr.compute_speedup == pytest.approx(2.0)


# ── QDQ insertion ──────────────────────────────────────────────────────────


def _build_simple_graph():
    """Build a simple Conv graph for QDQ insertion testing."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 8, 8])
    weight = gs.Constant("weight", values=np.random.randn(16, 3, 3, 3).astype(np.float32))
    conv_out = gs.Variable("conv_out", dtype=np.float32)
    output = gs.Variable("output", dtype=np.float32)

    conv = gs.Node(op="Conv", name="conv_0", inputs=[inp, weight], outputs=[conv_out])
    relu = gs.Node(op="Relu", name="relu_0", inputs=[conv_out], outputs=[output])

    graph = gs.Graph(nodes=[conv, relu], inputs=[inp], outputs=[output])
    graph.cleanup().toposort()
    return graph


class TestInsertQDQ:
    def test_insert_on_activation(self):
        graph = _build_simple_graph()
        original_nodes = len(graph.nodes)
        insert_qdq_on_graph(graph, ["input"], quant_type="int8")
        assert len(graph.nodes) == original_nodes + 2  # Q + DQ

    def test_insert_on_weight(self):
        graph = _build_simple_graph()
        original_nodes = len(graph.nodes)
        insert_qdq_on_graph(graph, ["weight"], quant_type="int8")
        assert len(graph.nodes) == original_nodes + 2

    def test_skip_nonexistent_tensor(self):
        graph = _build_simple_graph()
        original_nodes = len(graph.nodes)
        insert_qdq_on_graph(graph, ["nonexistent"], quant_type="int8")
        assert len(graph.nodes) == original_nodes

    def test_already_qdqd_skipped(self):
        graph = _build_simple_graph()
        seen = set()
        insert_qdq_on_graph(graph, ["input"], already_qdqd=seen)
        n1 = len(graph.nodes)
        insert_qdq_on_graph(graph, ["input"], already_qdqd=seen)
        assert len(graph.nodes) == n1  # no new nodes

    def test_qdq_produces_valid_onnx(self):
        graph = _build_simple_graph()
        insert_qdq_on_graph(graph, ["input"], quant_type="int8")
        model = gs.export_onnx(graph)
        assert len(model.graph.node) > 0
        op_types = {n.op_type for n in model.graph.node}
        assert "QuantizeLinear" in op_types
        assert "DequantizeLinear" in op_types


# ── Heuristic scheme generation ─────────────────────────────────────────────


class TestGenerateHeuristicSchemes:
    def test_conv_group_produces_schemes(self):
        graph = _build_simple_graph()
        group = FusionGroup(
            id=0,
            trt_layer_name="conv_fused",
            onnx_node_names=["conv_0", "relu_0"],
            has_quantizable_op=True,
            quantizable_node_names=["conv_0"],
        )
        schemes = generate_heuristic_schemes(graph, group)
        names = [s.name for s in schemes]
        assert "baseline" in names
        assert "full_qdq" in names
        assert "weight_only" in names
        assert len(schemes) >= 3

    def test_baseline_always_first(self):
        graph = _build_simple_graph()
        group = FusionGroup(
            id=0,
            trt_layer_name="x",
            onnx_node_names=["conv_0"],
            has_quantizable_op=True,
            quantizable_node_names=["conv_0"],
        )
        schemes = generate_heuristic_schemes(graph, group)
        assert schemes[0].name == "baseline"
        assert schemes[0].target_tensors == []

    def test_no_quantizable_ops(self):
        graph = _build_simple_graph()
        group = FusionGroup(
            id=0,
            trt_layer_name="relu_only",
            onnx_node_names=["relu_0"],
            has_quantizable_op=False,
            quantizable_node_names=[],
        )
        schemes = generate_heuristic_schemes(graph, group)
        assert len(schemes) == 1
        assert schemes[0].name == "baseline"
