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
import pytest
import torch
from _test_utils.onnx.lib_test_models import SimpleMLP, export_as_onnx

import modelopt.onnx.quantization as moq
from modelopt.onnx.utils import get_opset_version

# Mapping of quantization mode to minimum required opset
MIN_OPSET = {
    "int8": 19,
    "fp8": 19,
    "int4": 21,
}


@pytest.mark.parametrize("quant_mode", ["int8", "fp8", "int4"])
def test_opset_below_minimum_upgrades_to_minimum(tmp_path, quant_mode):
    """Test that specifying opset below minimum upgrades to minimum."""
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    min_opset = MIN_OPSET[quant_mode]

    # Request opset below minimum
    moq.quantize(onnx_path, quantize_mode=quant_mode, opset=min_opset - 1)

    # Verify output model was upgraded to minimum opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == min_opset, (
        f"Expected opset {min_opset} for {quant_mode}, got {output_opset}"
    )


@pytest.mark.parametrize("quant_mode", ["int8", "fp8", "int4"])
def test_opset_below_original_uses_original(tmp_path, quant_mode):
    """Test that specifying opset below original model's opset uses original."""
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    min_opset = MIN_OPSET[quant_mode]
    higher_opset = min_opset + 1

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path, opset=higher_opset)

    # Verify the exported model has the higher opset
    original_model = onnx.load(onnx_path)
    assert get_opset_version(original_model) == higher_opset

    # Request opset below original (but above minimum)
    moq.quantize(onnx_path, quantize_mode=quant_mode, opset=min_opset)

    # Verify output model preserves the higher original opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == higher_opset, (
        f"Expected original opset {higher_opset} to be preserved, got {output_opset}"
    )


@pytest.mark.parametrize("quant_mode", ["int8", "fp8", "int4"])
def test_opset_above_minimum(tmp_path, quant_mode):
    """Test that specifying opset at or above minimum is respected."""
    model_torch = SimpleMLP()
    input_tensor = torch.randn(2, 16, 16)

    min_opset = MIN_OPSET[quant_mode]
    target_opset = min_opset + 1

    onnx_path = os.path.join(tmp_path, "model.onnx")
    export_as_onnx(model_torch, input_tensor, onnx_filename=onnx_path)

    moq.quantize(onnx_path, quantize_mode=quant_mode, opset=target_opset)

    # Verify output model has the requested opset
    output_onnx_path = onnx_path.replace(".onnx", ".quant.onnx")
    output_model = onnx.load(output_onnx_path)
    output_opset = get_opset_version(output_model)

    assert output_opset == target_opset, (
        f"Expected opset {target_opset} for {quant_mode}, got {output_opset}"
    )
