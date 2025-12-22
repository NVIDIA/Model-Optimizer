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

import copy

import pytest
import torch
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import OneLayerLinear, SimpleLinear
from _test_utils.torch.quantization.quantize_common import compute_backward_grad

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.fp8_per_tensor_gemm import Fp8PerTensorLinear
from modelopt.torch.quantization.backends.nvfp4_gemm import Nvfp4Linear
from modelopt.torch.quantization.backends.utils import fp4_compatible, fp8_compatible

set_seed()


@pytest.fixture(autouse=True)
def setup_seed():
    """Set seed before each test function."""
    set_seed()


@pytest.mark.parametrize(
    ("config", "gemm_forward", "atol", "rtol"),
    [
        pytest.param(
            mtq.NVFP4_DEFAULT_CFG,
            Nvfp4Linear.apply,
            0.5,
            0.1,
            marks=[
                pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU"),
            ],
        ),
        pytest.param(
            mtq.FP8_DEFAULT_CFG,
            Fp8PerTensorLinear.apply,
            0.05,
            0.1,
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("input_dim", [2, 3])
def test_gemm(config, gemm_forward, atol, rtol, input_dim):
    model = OneLayerLinear(in_features=64, out_features=32)
    model = model.to(torch.float16).cuda()
    calib_data = [model.get_input().to(torch.float16).cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    module = model.net[0]
    if input_dim <= 2:
        input_tensor = calib_data[0].clone()
    else:
        # Use expand_dims to match input_dim directly
        input_tensor = calib_data[0]
        if input_tensor.dim() < input_dim:
            shape = (1,) * (input_dim - input_tensor.dim()) + input_tensor.shape
            input_tensor = input_tensor.reshape(shape)
    expected = torch.nn.functional.linear(input_tensor, module.weight, bias=None)

    # Test without bias
    result_no_bias = gemm_forward(module, input_tensor, module.weight)
    diff = (result_no_bias - expected).abs()
    assert torch.allclose(result_no_bias, expected, atol=atol, rtol=rtol), (
        f"Test without bias failed: {diff.amax()}\n"
        f"{result_no_bias[diff > atol]} != {expected[diff > atol]}"
    )

    # Generate a random bias for testing
    bias = torch.randn(module.weight.shape[0], device="cuda", dtype=torch.float16)
    expected_with_bias = torch.nn.functional.linear(input_tensor, module.weight, bias=bias)

    # Test 1: Bias as keyword argument (kwargs)
    result_with_bias_kwargs = gemm_forward(module, input_tensor, module.weight, bias=bias)
    diff = (result_with_bias_kwargs - expected_with_bias).abs()
    assert torch.allclose(result_with_bias_kwargs, expected_with_bias, atol=atol, rtol=rtol), (
        f"Bias as kwargs failed: {diff.amax()}\n"
        f"{result_with_bias_kwargs[diff > atol]} != {expected_with_bias[diff > atol]}"
    )

    # Test 2: Bias as positional argument (args)
    result_with_bias_args = gemm_forward(module, input_tensor, module.weight, bias)
    diff = (result_with_bias_args - expected_with_bias).abs()
    assert torch.allclose(result_with_bias_args, expected_with_bias, atol=atol, rtol=rtol), (
        f"Bias as args failed: {diff.amax()}\n"
        f"{result_with_bias_args[diff > atol]} != {expected_with_bias[diff > atol]}"
    )

    # Verify both methods produce the same result
    assert torch.equal(result_with_bias_kwargs, result_with_bias_args), (
        "Args and kwargs methods produced different results"
    )


@pytest.mark.parametrize(
    ("model", "config", "atol_bias", "atol_input"),
    [
        pytest.param(
            SimpleLinear(),
            mtq.NVFP4_DEFAULT_CFG,
            0.5,
            0.2,
            marks=[
                pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU"),
            ],
        ),
        pytest.param(
            SimpleLinear(),
            mtq.FP8_DEFAULT_CFG,
            0.02,
            0.02,
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU"),
            ],
        ),
    ],
)
def test_compressed_backward_to_input(model, config, atol_bias, atol_input):
    model = model.to(torch.float16).cuda()
    input_tensor = model.get_input().to(torch.float16).cuda()
    input_tensor.requires_grad = True

    _, bias_grads = compute_backward_grad(model, input_tensor, config=config, quantize=True)
    input_grad = input_tensor.grad
    input_tensor.grad = None

    weight_grads_quantized, bias_grads_quantized = compute_backward_grad(
        model, input_tensor, config=config, compress=True
    )
    input_grad_quantized = input_tensor.grad
    for weight_grad in weight_grads_quantized:
        assert weight_grad is None

    for bias_grad, bias_grad_quantized in zip(bias_grads, bias_grads_quantized):
        diff = (bias_grad - bias_grad_quantized).abs()
        assert torch.allclose(bias_grad, bias_grad_quantized, atol=atol_bias), (
            f"bias grad mismatch: {bias_grad[diff > atol_bias]} != {bias_grad_quantized[diff > atol_bias]}"
        )

    diff = (input_grad - input_grad_quantized).abs()
    assert torch.allclose(input_grad, input_grad_quantized, atol=atol_input), (
        f"input grad mismatch: {diff.amax()}\n"
        f"{input_grad[diff > atol_input]} != {input_grad_quantized[diff > atol_input]}"
    )


@pytest.mark.parametrize(
    ("model", "config", "gemm_forward", "atol", "rtol"),
    [
        pytest.param(
            OneLayerLinear(in_features=64, out_features=32),
            mtq.NVFP4_DEFAULT_CFG,
            Nvfp4Linear.apply,
            0.3,
            0.1,
            marks=[
                pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU"),
            ],
        ),
        pytest.param(
            OneLayerLinear(in_features=64, out_features=32),
            mtq.FP8_DEFAULT_CFG,
            Fp8PerTensorLinear.apply,
            0.1,
            0.1,
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="FP8 is not supported on this GPU"),
            ],
        ),
    ],
)
def test_dynamic_gemm(model, config, gemm_forward, atol, rtol):
    model_fp16 = model.to(torch.float16).cuda()
    calib_data = [model.get_input().to(torch.float16).cuda() for _ in range(8)]

    model_dynamic_quant = copy.deepcopy(model_fp16)
    mtq.quantize(model_dynamic_quant, config)

    model_dynamic_quant_compressed = copy.deepcopy(model_dynamic_quant)
    mtq.compress(model_dynamic_quant_compressed)

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    model_calib_quant = copy.deepcopy(model_fp16)
    mtq.quantize(model_calib_quant, config, forward_loop)

    model_calib_quant_compressed = copy.deepcopy(model_calib_quant)
    mtq.compress(model_calib_quant_compressed)

    result_fp16 = [model_fp16(input_tensor) for input_tensor in calib_data]

    result_dynamic_quant = [model_dynamic_quant(input_tensor) for input_tensor in calib_data]

    module = model_dynamic_quant.net[0]
    result_dynamic_quant_gemm = [
        gemm_forward(module, input_tensor, module.weight, bias=module.bias)
        for input_tensor in calib_data
    ]
    result_dynamic_quant_compressed = [
        model_dynamic_quant_compressed(input_tensor) for input_tensor in calib_data
    ]

    result_calib_quant = [model_calib_quant(input_tensor) for input_tensor in calib_data]

    module = model_calib_quant.net[0]
    result_calib_quant_gemm = [
        gemm_forward(module, input_tensor, module.weight, bias=module.bias)
        for input_tensor in calib_data
    ]

    result_calib_quant_compressed = [
        model_calib_quant_compressed(input_tensor) for input_tensor in calib_data
    ]

    for (
        output_fp16,
        output_dynamic_quant,
        output_dynamic_quant_gemm,
        output_dynamic_quant_compressed,
        output_calib_quant,
        output_calib_quant_gemm,
        output_calib_quant_compressed,
    ) in zip(
        result_fp16,
        result_dynamic_quant,
        result_dynamic_quant_gemm,
        result_dynamic_quant_compressed,
        result_calib_quant,
        result_calib_quant_gemm,
        result_calib_quant_compressed,
    ):
        assert torch.allclose(output_fp16, output_dynamic_quant_gemm, atol=atol, rtol=rtol)
        assert torch.allclose(output_fp16, output_calib_quant_gemm, atol=atol, rtol=rtol)

        # The way the compression of the weights and inputs might be different.
        # E.g. we may use torch.compile in the gemms.
        assert torch.allclose(output_dynamic_quant_gemm, output_dynamic_quant, atol=atol / 2)
        assert torch.allclose(output_calib_quant_gemm, output_calib_quant, atol=atol / 2)
        assert torch.allclose(
            output_dynamic_quant_gemm, output_dynamic_quant_compressed, atol=atol / 2
        )
        assert torch.allclose(output_calib_quant_gemm, output_calib_quant_compressed, atol=atol / 2)
