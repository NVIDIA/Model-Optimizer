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

"""Tests for ONNX quantization API handling."""

import copy
import importlib
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx
from onnx import TensorProto, helper
from packaging import version

import modelopt.onnx.quantization as moq
import modelopt.onnx.trt_utils as trt_utils
from modelopt.onnx.utils import get_opset_version

# Mapping of quantization mode to minimum required opset
MIN_OPSET = {
    "int8": 19,
    "fp8": 19,
    "int4": 21,
}

# onnxruntime version that supports opset 22+
ORT_VERSION_FOR_OPSET_22 = version.parse("1.23.0")


@pytest.fixture(autouse=True)
def _disable_tensorrt_model_parsing(monkeypatch):
    monkeypatch.setattr(trt_utils, "TRT_PYTHON_AVAILABLE", False)


class _GuardAutotuner:
    def __init__(self):
        self.config = SimpleNamespace(performance_threshold=1.02)
        self.force_no_qdq = False
        self.final_baseline_measurement = None
        self.final_decision = None
        self.saved_state_paths = []

    def set_force_no_qdq(self, force_no_qdq=True):
        self.force_no_qdq = force_no_qdq

    def record_final_baseline_measurement(self, **measurement):
        self.final_baseline_measurement = measurement
        self.final_decision = dict(measurement)

    def record_final_decision(self, **decision):
        self.final_decision.update(decision)

    def save_state(self, output_path):
        self.saved_state_paths.append(output_path)


def _make_precision_matched_guard_models():
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    to_fp16 = helper.make_node("Cast", ["input"], ["input_fp16"], to=TensorProto.FLOAT16)
    to_fp32 = helper.make_node("Cast", ["input_fp16"], ["output"], to=TensorProto.FLOAT)
    opset_imports = [helper.make_opsetid("", 19)]

    baseline = helper.make_model(
        helper.make_graph([to_fp16, to_fp32], "baseline", [graph_input], [graph_output]),
        opset_imports=opset_imports,
    )
    baseline.ir_version = 10

    scale = helper.make_tensor("scale", TensorProto.FLOAT16, [], [0.25])
    zero_point = helper.make_tensor("zero_point", TensorProto.UINT8, [], [0])
    q_node = helper.make_node(
        "QuantizeLinear", ["input_fp16", "scale", "zero_point"], ["input_quantized"]
    )
    dq_node = helper.make_node(
        "DequantizeLinear", ["input_quantized", "scale", "zero_point"], ["input_dequantized"]
    )
    candidate_to_fp32 = helper.make_node(
        "Cast", ["input_dequantized"], ["output"], to=TensorProto.FLOAT
    )
    candidate = helper.make_model(
        helper.make_graph(
            [to_fp16, q_node, dq_node, candidate_to_fp32],
            "candidate",
            [graph_input],
            [graph_output],
            [scale, zero_point],
        ),
        opset_imports=opset_imports,
    )
    candidate.ir_version = 10
    return baseline, candidate


def _make_guard_context(quantize_module, tmp_path, baseline, autotuner):
    output_dir = tmp_path / "autotune"
    output_dir.mkdir()
    return quantize_module._AutotuneContext(
        ort_config=([], [], [], []),
        autotuner=autotuner,
        baseline_model=baseline,
        output_dir=output_dir,
        state_path=output_dir / "state.yaml",
    )


# Test scenarios: (scenario_name, export_opset_offset, request_opset_offset, expected_opset_offset)
# Offsets are relative to MIN_OPSET[quant_mode].
OPSET_SCENARIOS = [
    # Requesting opset below minimum should upgrade to minimum
    ("below_min_upgrades", -1, -1, 0),
    # Requesting opset below original model's opset (but above minimum) should preserve original
    ("below_original_preserves", 1, 0, 1),
    # Requesting opset above minimum should be respected
    ("above_min_respected", 0, 1, 1),
]


def test_realign_input_shapes_profile_after_calibration_eps_update():
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")

    profiles = quantize_module._realign_input_shapes_profile(
        [{"cpu_profile": "cpu"}, {"trt_profile": "trt"}],
        ["cpu", "trt"],
        ["trt", "cpu"],
    )

    assert profiles == [{"trt_profile": "trt"}, {"cpu_profile": "cpu"}]


def test_realign_input_shapes_profile_rejects_duplicate_calibration_eps():
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")

    with pytest.raises(AssertionError, match="Calibration EPs must be unique"):
        quantize_module._realign_input_shapes_profile(
            [{"cpu_profile": "first"}, {"cpu_profile": "second"}],
            ["cpu", "cpu"],
            ["cpu"],
        )


def test_calibration_source_identity_is_stable_content_sensitive_and_redacted(tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    cache_path = tmp_path / "private_calibration.cache"
    cache_path.write_bytes(b"private-cache-content-a")
    calibration_data = {
        "private_input_name": np.array([[1.25, 2.5]], dtype=np.float32),
    }

    class Reader(quantize_module.CalibrationDataProvider):
        def __init__(self, data):
            self.calibration_data_list = [data]
            self.calibration_data_reader = iter(self.calibration_data_list)

    reader = Reader(calibration_data)
    identity = quantize_module._get_calibration_source_identity(
        calibration_cache_path=str(cache_path),
        calibration_eps=["private_ep:0", "cpu"],
        calibration_data=calibration_data,
        calibration_data_reader=reader,
    )
    assert identity == quantize_module._get_calibration_source_identity(
        calibration_cache_path=str(cache_path),
        calibration_eps=["private_ep:0", "cpu"],
        calibration_data=copy.deepcopy(calibration_data),
        calibration_data_reader=Reader(copy.deepcopy(calibration_data)),
    )

    cache_path.write_bytes(b"private-cache-content-b")
    changed_cache_identity = quantize_module._get_calibration_source_identity(
        calibration_cache_path=str(cache_path),
        calibration_eps=["private_ep:0", "cpu"],
        calibration_data=calibration_data,
        calibration_data_reader=reader,
    )
    assert changed_cache_identity != identity

    changed_data = copy.deepcopy(calibration_data)
    changed_data["private_input_name"][0, 0] = 9.0
    assert (
        quantize_module._get_calibration_source_identity(
            calibration_cache_path=str(cache_path),
            calibration_eps=["private_ep:0", "cpu"],
            calibration_data=changed_data,
            calibration_data_reader=Reader(changed_data),
        )
        != changed_cache_identity
    )
    assert (
        quantize_module._get_calibration_source_identity(
            calibration_cache_path=str(cache_path),
            calibration_eps=["cpu", "private_ep:0"],
            calibration_data=calibration_data,
            calibration_data_reader=reader,
        )
        != changed_cache_identity
    )

    serialized_identity = repr(identity)
    assert str(cache_path) not in serialized_identity
    assert "private-cache-content" not in serialized_identity
    assert "private_input_name" not in serialized_identity
    assert "private_ep" not in serialized_identity

    class OpaqueReader:
        def __init__(self):
            self.source_id = "sensitive-source"
            self.consumed = False

        @property
        def calibration_data_list(self):
            raise AssertionError("custom reader properties must not be evaluated")

        def get_next(self):
            self.consumed = True
            raise AssertionError("custom readers must not be consumed")

    opaque_reader = OpaqueReader()
    opaque_identity = quantize_module._get_calibration_reader_identity(opaque_reader)
    assert opaque_identity == quantize_module._get_calibration_reader_identity(opaque_reader)
    assert opaque_reader.consumed is False
    assert "sensitive-source" not in repr(opaque_identity)


def test_changed_calibration_cache_invalidates_persisted_autotune_decisions(tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    autotuner_module = importlib.import_module("modelopt.onnx.quantization.autotune.autotuner_base")
    cache_path = tmp_path / "calibration.cache"
    state_path = tmp_path / "autotuner_state.yaml"
    source_model, _ = _make_precision_matched_guard_models()

    cache_path.write_bytes(b"cache-a")
    first_fingerprint = quantize_module._get_calibration_source_identity(
        calibration_cache_path=str(cache_path),
        calibration_eps=["cpu"],
        calibration_data=None,
        calibration_data_reader=None,
    )
    autotuner = autotuner_module.QDQAutotunerBase(source_model)
    autotuner.initialize()
    autotuner.set_resume_fingerprint(calibration_source=first_fingerprint)
    autotuner.submit(100.0)
    autotuner.set_force_no_qdq()
    autotuner.record_proxy_decision(
        proxy_selection="no_qdq",
        baseline_latency_ms=100.0,
        candidate_latency_ms=99.0,
        candidate_quantization_site_count=1,
        candidate_model_sha256="candidate-proxy",
        selected_model_sha256="baseline-proxy",
    )
    autotuner.record_final_baseline_measurement(
        decision_stage="calibrated_baseline",
        baseline_final_latency_ms=100.0,
        baseline_model_sha256="baseline-final",
    )
    autotuner.record_final_decision(
        decision_stage="calibrated_final",
        final_selection="no_qdq",
        candidate_final_latency_ms=99.0,
        final_latency_ms=100.0,
        candidate_quantization_site_count=1,
        candidate_model_sha256="candidate-final",
        selected_model_sha256="baseline-final",
    )
    autotuner.save_state(str(state_path))

    cache_path.write_bytes(b"cache-b")
    second_fingerprint = quantize_module._get_calibration_source_identity(
        calibration_cache_path=str(cache_path),
        calibration_eps=["cpu"],
        calibration_data=None,
        calibration_data_reader=None,
    )
    restored = autotuner_module.QDQAutotunerBase(source_model)
    restored.initialize()
    restored.set_resume_fingerprint(calibration_source=second_fingerprint)
    restored.load_state(str(state_path))

    assert restored.baseline_latency_ms is None
    assert restored.force_no_qdq is False
    assert restored.proxy_decision is None
    assert restored.final_decision is None


@pytest.mark.parametrize(
    ("argument", "value"),
    [("nodes_to_quantize", ["MatMul_0"]), ("op_types_to_quantize", ["MatMul"])],
)
def test_autotune_rejects_explicit_quantization_includes(argument, value):
    with pytest.raises(ValueError, match="Autotune cannot be combined"):
        moq.quantize("model.onnx", autotune=True, **{argument: value})


def test_autotune_rejects_prequantized_source(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "output.onnx"
    onnx_path.write_bytes(b"")
    _, qdq_model = _make_precision_matched_guard_models()
    monkeypatch.setattr(
        quantize_module,
        "_preprocess_onnx",
        lambda *args, **kwargs: (
            str(onnx_path),
            qdq_model,
            [],
            False,
            False,
            False,
            {},
            {},
        ),
    )

    with pytest.raises(ValueError, match="unquantized source model"):
        quantize_module.quantize(
            str(onnx_path),
            output_path=str(output_path),
            quantize_mode="fp8",
            calibration_data_reader=object(),
            calibration_eps=["cpu"],
            autotune=True,
        )


@pytest.mark.parametrize(
    ("candidate_latency", "expected_selection", "expected_selected_site_count"),
    [(99.0, "no_qdq", 0), (95.0, "qdq", 1)],
)
def test_autotune_final_guard_applies_threshold_and_persists_state(
    monkeypatch,
    tmp_path,
    candidate_latency,
    expected_selection,
    expected_selected_site_count,
):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    latencies = iter([100.0, candidate_latency])
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: next(latencies))
    baseline, candidate = _make_precision_matched_guard_models()
    autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, autotuner)

    selected = quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    assert quantize_module._count_quantization_sites(selected) == expected_selected_site_count
    assert autotuner.force_no_qdq is (expected_selection == "no_qdq")
    assert autotuner.final_baseline_measurement["decision_stage"] == "calibrated_baseline"
    assert autotuner.final_baseline_measurement["baseline_final_latency_ms"] == 100.0
    assert autotuner.final_baseline_measurement["baseline_model_sha256"]
    assert autotuner.final_decision["decision_stage"] == "calibrated_final"
    assert autotuner.final_decision["baseline_final_latency_ms"] == 100.0
    assert autotuner.final_decision["baseline_model_sha256"]
    assert autotuner.final_decision["final_selection"] == expected_selection
    assert autotuner.final_decision["candidate_final_latency_ms"] == candidate_latency
    assert autotuner.final_decision["candidate_quantization_site_count"] == 1
    assert autotuner.saved_state_paths == [str(context.state_path)] * 2
    assert selected.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert selected.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


@pytest.mark.parametrize(
    ("candidate_latency", "expected_selection"),
    [(99.0, "no_qdq"), (95.0, "qdq")],
)
def test_autotune_final_guard_reuses_matching_persisted_decision(
    monkeypatch, tmp_path, candidate_latency, expected_selection
):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    baseline, candidate = _make_precision_matched_guard_models()
    initial_autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, initial_autotuner)
    latencies = iter([100.0, candidate_latency])
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: next(latencies))
    quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    restored_autotuner = _GuardAutotuner()
    restored_autotuner.final_baseline_measurement = copy.deepcopy(
        initial_autotuner.final_baseline_measurement
    )
    restored_autotuner.final_decision = copy.deepcopy(initial_autotuner.final_decision)
    restored_autotuner.force_no_qdq = expected_selection != "no_qdq"
    context.autotuner = restored_autotuner
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: pytest.fail("a validated final decision must not be remeasured"),
    )

    selected = quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    assert restored_autotuner.force_no_qdq is (expected_selection == "no_qdq")
    assert restored_autotuner.final_decision["final_selection"] == expected_selection
    assert quantize_module._count_quantization_sites(selected) == (
        1 if expected_selection == "qdq" else 0
    )
    assert restored_autotuner.saved_state_paths == [str(context.state_path)]


def test_autotune_final_guard_remeasures_stale_candidate_hash(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    baseline, candidate = _make_precision_matched_guard_models()
    autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, autotuner)
    initial_latencies = iter([100.0, 99.0])
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda *args, **kwargs: next(initial_latencies),
    )
    quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )
    autotuner.final_decision["candidate_model_sha256"] = "stale"

    remeasurements = iter([100.0, 95.0])
    benchmark_count = 0

    def benchmark(*args, **kwargs):
        nonlocal benchmark_count
        benchmark_count += 1
        return next(remeasurements)

    monkeypatch.setattr(workflows, "benchmark_onnx_model", benchmark)
    selected = quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    assert benchmark_count == 2
    assert autotuner.force_no_qdq is False
    assert autotuner.final_decision["final_selection"] == "qdq"
    assert autotuner.final_decision["candidate_model_sha256"] != "stale"
    assert quantize_module._count_quantization_sites(selected) == 1


def test_autotune_final_guard_accepts_q_only_candidate(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    monkeypatch.setattr(
        workflows,
        "benchmark_onnx_model",
        lambda model_path, *args, **kwargs: 100.0 if "baseline" in model_path else 95.0,
    )
    baseline, candidate = _make_precision_matched_guard_models()
    dq_index = next(
        index
        for index, node in enumerate(candidate.graph.node)
        if node.op_type == "DequantizeLinear"
    )
    del candidate.graph.node[dq_index]
    output_cast = next(
        node
        for node in candidate.graph.node
        if node.op_type == "Cast" and node.output[0] == "output"
    )
    output_cast.input[0] = "input_quantized"
    autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, autotuner)

    selected = quantize_module._apply_autotune_final_guard(
        candidate,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    assert [node.op_type for node in selected.graph.node].count("QuantizeLinear") == 1
    assert not any(node.op_type == "DequantizeLinear" for node in selected.graph.node)
    assert autotuner.force_no_qdq is False
    assert autotuner.final_decision["candidate_quantization_site_count"] == 1


def test_autotune_final_guard_reports_zero_site_fallback(monkeypatch, tmp_path, caplog):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    latencies = iter([100.0, 90.0])
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: next(latencies))
    baseline, _ = _make_precision_matched_guard_models()
    autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, autotuner)

    selected = quantize_module._apply_autotune_final_guard(
        baseline,
        context,
        quantize_mode="fp8",
        use_external_data_format=False,
    )

    assert quantize_module._count_quantization_sites(selected) == 0
    assert "the calibrated candidate contained no quantization sites" in caplog.text
    assert "best valid quantized output" not in caplog.text


@pytest.mark.parametrize("node_to_remove", ["QuantizeLinear", "DequantizeLinear"])
def test_quantization_site_count_accepts_single_sided_qdq(node_to_remove):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    _, candidate = _make_precision_matched_guard_models()
    node_index = next(
        index for index, node in enumerate(candidate.graph.node) if node.op_type == node_to_remove
    )
    del candidate.graph.node[node_index]

    assert quantize_module._count_quantization_sites(candidate) == 1


@pytest.mark.parametrize("baseline_latency", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_autotune_final_guard_rejects_invalid_baseline(monkeypatch, tmp_path, baseline_latency):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    benchmarked_paths = []

    def benchmark(model_path, *args, **kwargs):
        benchmarked_paths.append(model_path)
        return baseline_latency

    monkeypatch.setattr(workflows, "benchmark_onnx_model", benchmark)
    baseline, candidate = _make_precision_matched_guard_models()
    autotuner = _GuardAutotuner()
    context = _make_guard_context(quantize_module, tmp_path, baseline, autotuner)

    with pytest.raises(RuntimeError, match="finite positive latency"):
        quantize_module._apply_autotune_final_guard(
            candidate,
            context,
            quantize_mode="fp8",
            use_external_data_format=False,
        )

    assert len(benchmarked_paths) == 1
    assert autotuner.final_baseline_measurement is None
    assert autotuner.final_decision is None
    assert autotuner.saved_state_paths == []


def test_autotune_tempdir_is_cleaned_when_calibration_fails(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "output.onnx"
    onnx_path.write_bytes(b"")
    temporary_output_dir = tempfile.TemporaryDirectory(dir=tmp_path)
    temporary_path = temporary_output_dir.name
    source_model, _ = _make_precision_matched_guard_models()
    context = quantize_module._AutotuneContext(
        ort_config=([], [], [], []),
        autotuner=_GuardAutotuner(),
        baseline_model=object(),
        output_dir=tmp_path,
        state_path=tmp_path / "state.yaml",
        temporary_output_dir=temporary_output_dir,
    )

    monkeypatch.setattr(
        quantize_module,
        "_preprocess_onnx",
        lambda *args, **kwargs: (
            str(onnx_path),
            source_model,
            [],
            False,
            False,
            False,
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        quantize_module,
        "update_trt_ep_support",
        lambda calibration_eps, has_dds_op, has_custom_op, trt_plugins: trt_plugins,
    )
    monkeypatch.setattr(quantize_module, "validate_op_types_spelling", lambda *args: None)
    monkeypatch.setattr(quantize_module, "find_nodes_from_mha_to_exclude", lambda *args: [])
    monkeypatch.setattr(
        quantize_module, "_find_nodes_to_quantize_autotune", lambda *args, **kwargs: context
    )

    def fail_calibration(**kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(quantize_module, "quantize_fp8", fail_calibration)

    with pytest.raises(RuntimeError, match="calibration failed"):
        quantize_module.quantize(
            str(onnx_path),
            output_path=str(output_path),
            quantize_mode="fp8",
            calibration_data_reader=object(),
            calibration_eps=["cpu"],
            autotune=True,
        )

    assert context.temporary_output_dir is None
    assert not os.path.exists(temporary_path)


def test_fp8_autotune_subthreshold_result_uses_precision_matched_fallback(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    input_tensor = torch.randn(2, 16, 16)
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "autotuned.onnx"
    autotune_dir = tmp_path / "autotune"
    autotune_dir.mkdir()
    export_as_onnx(SimpleMLP(), input_tensor, onnx_filename=str(onnx_path), opset=19)
    autotuner = _GuardAutotuner()

    def fake_find_nodes(
        model,
        quantize_mode,
        trt_plugins,
        high_precision_dtype,
        direct_io_types,
        op_types_to_exclude_fp16,
        custom_ops_to_cast_fp32,
        opset,
        mha_accumulation_dtype,
        **kwargs,
    ):
        selected_nodes = [
            node.name for node in model.graph.node if node.op_type in {"Gemm", "MatMul"}
        ]
        assert selected_nodes
        baseline = quantize_module._convert_to_runtime_precision(
            copy.deepcopy(model),
            quantize_mode=quantize_mode,
            high_precision_dtype=high_precision_dtype,
            direct_io_types=direct_io_types,
            op_types_to_exclude_fp16=op_types_to_exclude_fp16,
            custom_ops_to_cast_fp32=custom_ops_to_cast_fp32,
            trt_extra_plugin_lib_paths=trt_plugins,
            opset=opset,
            mha_accumulation_dtype=mha_accumulation_dtype,
        )
        return quantize_module._AutotuneContext(
            ort_config=(selected_nodes, ["Gemm", "MatMul"], [], []),
            autotuner=autotuner,
            baseline_model=baseline,
            output_dir=autotune_dir,
            state_path=autotune_dir / "state.yaml",
        )

    latencies = iter([100.0, 99.0])
    monkeypatch.setattr(quantize_module, "_find_nodes_to_quantize_autotune", fake_find_nodes)
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args, **kwargs: next(latencies))

    moq.quantize(
        str(onnx_path),
        output_path=str(output_path),
        quantize_mode="fp8",
        calibration_eps=["cpu"],
        autotune=True,
        autotune_output_dir=str(autotune_dir),
    )

    selected_model = onnx.load(output_path)
    assert not any(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in selected_model.graph.node
    )
    assert selected_model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert selected_model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(
        node.op_type == "Cast"
        and helper.get_attribute_value(node.attribute[0]) == TensorProto.FLOAT16
        for node in selected_model.graph.node
        if node.attribute
    )
    assert autotuner.force_no_qdq is True
    assert autotuner.final_decision["final_selection"] == "no_qdq"


def test_quantize_infers_input_profiles_after_ep_support_update(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")
    captured = {}

    def fake_preprocess(
        onnx_path,
        use_external_data_format,
        output_path,
        enable_shared_constants_duplication,
        trt_plugins,
        trt_plugins_precision,
        override_shapes,
        simplify,
        quantize_mode,
        opset,
    ):
        return onnx_path, object(), [], True, False, False, {}, {}

    def fake_update_trt_ep_support(calibration_eps, has_dds_op, has_custom_op, trt_plugins):
        assert has_custom_op is True
        calibration_eps.remove("trt")
        calibration_eps.insert(0, "trt")
        return trt_plugins

    def fake_create_input_shapes_profile(model_id, calibration_eps, trust_remote_code=False):
        captured["profile_eps"] = list(calibration_eps)
        captured["trust_remote_code"] = trust_remote_code
        return [{"trt_profile_min_shapes": "trt_profile"}, {}]

    def fake_find_nodes_from_mha_to_exclude(*args):
        captured["find_eps"] = list(args[-2])
        captured["find_profile"] = args[-1]
        return []

    def fake_quantize_int8(**kwargs):
        captured["quantize_eps"] = list(kwargs["calibration_eps"])
        captured["quantize_profile"] = kwargs["input_shapes_profile"]

    monkeypatch.setattr(quantize_module, "_preprocess_onnx", fake_preprocess)
    monkeypatch.setattr(quantize_module, "update_trt_ep_support", fake_update_trt_ep_support)
    monkeypatch.setattr(
        quantize_module, "create_input_shapes_profile", fake_create_input_shapes_profile
    )
    monkeypatch.setattr(
        quantize_module, "find_nodes_from_mha_to_exclude", fake_find_nodes_from_mha_to_exclude
    )
    monkeypatch.setattr(quantize_module, "validate_op_types_spelling", lambda *args: None)
    monkeypatch.setattr(quantize_module, "quantize_int8", fake_quantize_int8)
    monkeypatch.setattr(quantize_module.onnx.checker, "check_model", lambda *args: None)

    quantize_module.quantize(
        str(onnx_path),
        calibration_eps=["cpu", "trt"],
        calibration_data_reader=object(),
        model_id="local-config",
        trust_remote_code=True,
    )

    assert captured["profile_eps"] == ["trt", "cpu"]
    assert captured["trust_remote_code"] is True
    assert captured["find_eps"] == ["trt", "cpu"]
    assert captured["quantize_eps"] == ["trt", "cpu"]
    assert captured["find_profile"] == [{"trt_profile_min_shapes": "trt_profile"}, {}]
    assert captured["quantize_profile"] == [{"trt_profile_min_shapes": "trt_profile"}, {}]


@pytest.mark.parametrize("quant_mode", ["int8", "fp8", "int4"])
@pytest.mark.parametrize(
    ("scenario_name", "export_opset_offset", "request_opset_offset", "expected_opset_offset"),
    OPSET_SCENARIOS,
    ids=[s[0] for s in OPSET_SCENARIOS],
)
def test_quantize_opset_handling(
    tmp_path,
    quant_mode,
    scenario_name,
    export_opset_offset,
    request_opset_offset,
    expected_opset_offset,
):
    """Test opset handling in quantization API.

    Scenarios:
    - below_min_upgrades: Requesting opset below minimum upgrades to minimum.
    - below_original_preserves: Requesting opset below original model's opset preserves original.
    - above_min_respected: Requesting opset at or above minimum is respected.
    """
    min_opset = MIN_OPSET[quant_mode]

    # Calculate actual opset values from offsets
    export_opset = min_opset + export_opset_offset
    request_opset = min_opset + request_opset_offset
    expected_opset = min_opset + expected_opset_offset

    # Skip if required opset exceeds onnxruntime support
    max_opset = max(export_opset, request_opset, expected_opset)
    if max_opset >= 22:
        ort_version = version.parse(onnxruntime.__version__)
        if ort_version < ORT_VERSION_FOR_OPSET_22:
            pytest.skip(
                f"Opset {max_opset} requires onnxruntime >= {ORT_VERSION_FOR_OPSET_22}, have {ort_version}"
            )

    # Setup: create and export model
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)
    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path, opset=export_opset)

    # Run quantization
    moq.quantize(
        onnx_path,
        quantize_mode=quant_mode,
        opset=request_opset,
        calibration_eps=["cpu"],
    )

    # Verify output opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == expected_opset, (
        f"[{scenario_name}] Expected opset {expected_opset} for {quant_mode}, got {output_opset}"
    )
