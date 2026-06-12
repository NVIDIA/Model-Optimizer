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

import importlib
import os

import onnx
import onnxruntime
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx
from packaging import version

import modelopt.onnx.quantization as moq
from modelopt.onnx.utils import get_opset_version

# Mapping of quantization mode to minimum required opset
MIN_OPSET = {
    "int8": 19,
    "fp8": 19,
    "int4": 21,
}

# onnxruntime version that supports opset 22+
ORT_VERSION_FOR_OPSET_22 = version.parse("1.23.0")


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
    moq.quantize(onnx_path, quantize_mode=quant_mode, opset=request_opset)

    # Verify output opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == expected_opset, (
        f"[{scenario_name}] Expected opset {expected_opset} for {quant_mode}, got {output_opset}"
    )
