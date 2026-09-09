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

import onnx
import onnxruntime
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx
from onnx import TensorProto, helper
from packaging import version

import modelopt.onnx.quantization as moq
import modelopt.onnx.trt_utils as trt_utils
from modelopt.onnx.quantization.autotune import Config, QDQAutotuner
from modelopt.onnx.quantization.autotune.insertion_points import get_autotuner_quantizable_ops
from modelopt.onnx.utils import get_opset_version

# Mapping of quantization mode to minimum required opset
MIN_OPSET = {
    "int8": 19,
    "fp8": 19,
    "int4": 21,
}

# onnxruntime version that supports opset 22+
ORT_VERSION_FOR_OPSET_22 = version.parse("1.23.0")


def _make_guard_models(site_ops=("QuantizeLinear", "DequantizeLinear")):
    graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    baseline = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "baseline",
            [graph_input],
            [graph_output],
        ),
        opset_imports=[helper.make_opsetid("", 19)],
    )
    candidate = copy.deepcopy(baseline)
    candidate.graph.node.extend(
        helper.make_node(op_type, ["input"], [f"site_{index}"])
        for index, op_type in enumerate(site_ops)
    )
    return baseline, candidate


def _make_guard_context(quantize_module, tmp_path, baseline):
    tmp_path.mkdir(exist_ok=True)
    return quantize_module._AutotuneContext(
        ort_config=([], [], [], []),
        baseline_model=baseline,
        performance_threshold=1.02,
        output_dir=tmp_path,
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


def test_quantize_rejects_trt_plugins_with_trt_rtx_abi():
    with pytest.raises(
        ValueError,
        match="TensorRT plugin paths are not supported with the TensorRT-RTX backend",
    ):
        moq.quantize(
            "model.onnx",
            calibration_eps=["NvTensorRtRtx"],
            trt_rtx_backend="abi",
            trt_plugins=["custom_plugin.dll"],
        )


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


@pytest.mark.parametrize("quant_type", ["int8", "fp8"])
def test_autotune_ort_op_types_match_quantization_mode(quant_type):
    model, _ = _make_guard_models(())
    autotuner = QDQAutotuner(model)
    autotuner.initialize(Config(default_quant_type=quant_type))
    expected = get_autotuner_quantizable_ops()
    if quant_type == "fp8":
        expected &= {"Conv", "Gemm", "MatMul", "Add"}
    op_types = autotuner.get_ort_quantization_config()[1]
    assert isinstance(op_types, list)
    assert set(op_types) == expected


@pytest.mark.parametrize(
    ("candidate_latency", "site_ops", "expect_qdq"),
    [
        (101.0, ("QuantizeLinear", "DequantizeLinear"), False),
        (100.0, ("QuantizeLinear", "DequantizeLinear"), True),
        (99.0, ("QuantizeLinear", "DequantizeLinear"), True),
        (90.0, ("QuantizeLinear",), True),
        (90.0, ("DequantizeLinear",), True),
        (90.0, (), False),
    ],
)
def test_autotune_final_guard_selection(
    monkeypatch, tmp_path, candidate_latency, site_ops, expect_qdq
):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    latencies = iter([102.0, candidate_latency])
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args: next(latencies))
    baseline, candidate = _make_guard_models(site_ops)

    selected = quantize_module._apply_autotune_final_guard(
        candidate,
        _make_guard_context(quantize_module, tmp_path, baseline),
        use_external_data_format=False,
    )

    assert quantize_module._has_qdq_site(selected) is expect_qdq


@pytest.mark.parametrize("invalid_latency", [float("nan"), float("inf"), float("-inf"), 0, -1])
@pytest.mark.parametrize("invalid_model", ["baseline", "candidate"])
def test_autotune_final_guard_invalid_measurements(
    monkeypatch, tmp_path, invalid_latency, invalid_model
):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    latencies = [invalid_latency] if invalid_model == "baseline" else [102.0, invalid_latency]
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args: latencies.pop(0))
    baseline, candidate = _make_guard_models()
    context = _make_guard_context(quantize_module, tmp_path, baseline)

    if invalid_model == "baseline":
        with pytest.raises(RuntimeError, match="finite positive latency"):
            quantize_module._apply_autotune_final_guard(
                candidate, context, use_external_data_format=False
            )
    else:
        selected = quantize_module._apply_autotune_final_guard(
            candidate, context, use_external_data_format=False
        )
        assert not quantize_module._has_qdq_site(selected)


def test_autotune_rejects_prequantized_source(monkeypatch, tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")
    _, qdq_model = _make_guard_models()
    preprocessed = (str(onnx_path), qdq_model, [], False, False, False, {}, {})
    monkeypatch.setattr(quantize_module, "_preprocess_onnx", lambda *args, **kwargs: preprocessed)

    with pytest.raises(ValueError, match="unquantized source model"):
        quantize_module.quantize(
            str(onnx_path),
            output_path=str(tmp_path / "output.onnx"),
            quantize_mode="fp8",
            calibration_data_reader=object(),
            calibration_eps=["cpu"],
            autotune=True,
        )


def test_autotune_tempdir_is_cleaned_after_failure(tmp_path):
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    temporary_output_dir = tempfile.TemporaryDirectory(dir=tmp_path)
    temporary_path = temporary_output_dir.name
    baseline, _ = _make_guard_models()
    context = _make_guard_context(quantize_module, tmp_path, baseline)
    context.temporary_output_dir = temporary_output_dir

    with pytest.raises(RuntimeError, match="failed"):
        quantize_module._run_with_autotune_cleanup(
            context, lambda: (_ for _ in ()).throw(RuntimeError("failed"))
        )

    assert not os.path.exists(temporary_path)


def test_fp8_autotune_subthreshold_result_uses_precision_matched_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(trt_utils, "TRT_PYTHON_AVAILABLE", False)
    quantize_module = importlib.import_module("modelopt.onnx.quantization.quantize")
    workflows = importlib.import_module("modelopt.onnx.quantization.autotune.workflows")
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "autotuned.onnx"
    autotune_dir = tmp_path / "autotune"
    autotune_dir.mkdir()
    export_as_onnx(SimpleMLP(), torch.randn(2, 16, 16), onnx_filename=str(onnx_path), opset=19)

    def fake_find_nodes(model, quantize_mode, trt_plugins, high_precision_dtype, **kwargs):
        nodes = [node.name for node in model.graph.node if node.op_type in {"Gemm", "MatMul"}]
        baseline = quantize_module._convert_to_runtime_precision(
            copy.deepcopy(model),
            quantize_mode=quantize_mode,
            high_precision_dtype=high_precision_dtype,
            direct_io_types=kwargs["direct_io_types"],
            op_types_to_exclude_fp16=kwargs["op_types_to_exclude_fp16"],
            custom_ops_to_cast_fp32=kwargs["custom_ops_to_cast_fp32"],
            trt_extra_plugin_lib_paths=trt_plugins,
            opset=kwargs["opset"],
            mha_accumulation_dtype=kwargs["mha_accumulation_dtype"],
        )
        return quantize_module._AutotuneContext(
            ort_config=(nodes, ["Gemm", "MatMul"], [], []),
            baseline_model=baseline,
            performance_threshold=1.02,
            output_dir=autotune_dir,
        )

    latencies = iter([100.0, 99.0])
    monkeypatch.setattr(quantize_module, "_find_nodes_to_quantize_autotune", fake_find_nodes)
    monkeypatch.setattr(workflows, "benchmark_onnx_model", lambda *args: next(latencies))

    moq.quantize(
        str(onnx_path),
        output_path=str(output_path),
        quantize_mode="fp8",
        calibration_eps=["cpu"],
        autotune=True,
        autotune_output_dir=str(autotune_dir),
    )

    candidate = onnx.load(autotune_dir / "calibrated_candidate.onnx")
    assert quantize_module._has_qdq_site(candidate)
    selected = onnx.load(output_path)
    assert not quantize_module._has_qdq_site(selected)
    assert selected.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert selected.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


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
        captured["find_eps"] = list(args[-3])
        captured["find_profile"] = args[-2]
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
    moq.quantize(onnx_path, quantize_mode=quant_mode, opset=request_opset)

    # Verify output opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == expected_opset, (
        f"[{scenario_name}] Expected opset {expected_opset} for {quant_mode}, got {output_opset}"
    )
