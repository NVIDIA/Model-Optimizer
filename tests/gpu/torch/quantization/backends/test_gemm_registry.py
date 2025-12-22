# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.gemm_registry import (
    GEMMRegistry,
    disable_real_quant_gemm,
    enable_real_quant_gemm,
)


class MockModule(torch.nn.Module):
    """Mock module for testing."""

    def __init__(self, module_type: str = "linear", has_quantizers: bool = False):
        super().__init__()
        self.module_type = module_type

        self.input_quantizer = type("InputQuantizer", (), {"num_bits": (4, 3)})()
        self.weight_quantizer = type("WeightQuantizer", (), {"num_bits": (4, 3)})()

    def forward(self, input, args, kwargs):
        return "default_gemm_called"


def prepare_quantized_model(model_cls, config):
    """Helper to prepare a quantized model for testing."""
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    return model, calib_data


@pytest.mark.parametrize("model_cls", [SimpleLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
    ],
)
def test_enable_real_quant_gemm(model_cls, config):
    """Test enabling real quant GEMM on quantized models."""
    model = prepare_quantized_model(model_cls, config)[0]

    # Get the linear module (first layer in SimpleLinear)
    linear_module = model.net[0]

    # Verify initial state (no GEMM flag)
    assert not hasattr(linear_module, "_use_real_quant_gemm")

    # Enable real quant GEMM
    enable_real_quant_gemm(model)

    # Check that the module was correctly marked
    assert hasattr(linear_module, "_use_real_quant_gemm")
    assert linear_module._use_real_quant_gemm is True


@pytest.mark.parametrize("model_cls", [SimpleLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
    ],
)
def test_disable_real_quant_gemm(model_cls, config):
    """Test disabling real quant GEMM on quantized models."""
    model = prepare_quantized_model(model_cls, config)[0]
    linear_module = model.net[0]

    # First enable real quant GEMM
    enable_real_quant_gemm(model)
    assert hasattr(linear_module, "_use_real_quant_gemm")
    assert linear_module._use_real_quant_gemm is True

    # Now disable it
    disable_real_quant_gemm(model)

    # Check that the module was correctly marked as disabled
    assert hasattr(linear_module, "_use_real_quant_gemm")
    assert linear_module._use_real_quant_gemm is False


@pytest.mark.parametrize("model_cls", [SimpleLinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.FP8_DEFAULT_CFG,
    ],
)
def test_gemm_enabled_after_compress(model_cls, config):
    """Test compressing a quantized model."""
    model = prepare_quantized_model(model_cls, config)[0]
    model_uncompressed = prepare_quantized_model(model_cls, config)[0]
    mtq.compress(model)

    # Get the first module for both compressed and uncompressed models
    linear_module = model.net[0]
    linear_module_uncompressed = model_uncompressed.net[0]

    # Check the flag is enabled after compression
    assert hasattr(linear_module, "_use_real_quant_gemm")
    assert linear_module._use_real_quant_gemm is True

    # Check the flag does not exist in the uncompressed model
    assert not hasattr(linear_module_uncompressed, "_use_real_quant_gemm")


def test_gemm_registry_find_match():
    """Test finding matches in the registry."""
    registry = GEMMRegistry()

    # Define multiple GEMM implementations
    def gemm_fp8(module, input, args, kwargs):
        return "fp8_gemm_called"

    # Register with different availability checks
    registry.register(
        gemm_func=gemm_fp8,
        availability_check=lambda m, i, a, k: hasattr(m, "input_quantizer")
        and hasattr(m.input_quantizer, "num_bits")
        and m.input_quantizer.num_bits == (4, 3),
    )

    # Create test modules
    module_fp8 = MockModule(has_quantizers=True)
    dummy_input = torch.ones(1)

    # Should match the fp8 implementation
    match_fp8 = registry.find_match(module_fp8, dummy_input, [], {})
    assert match_fp8(module_fp8, dummy_input, [], {}) == "fp8_gemm_called"
