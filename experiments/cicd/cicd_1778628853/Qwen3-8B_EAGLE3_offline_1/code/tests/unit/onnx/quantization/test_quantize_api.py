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

"""Tests for ONNX quantization opset handling."""

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
