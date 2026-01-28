# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for quantized tensors."""

import math

import pytest
import torch
from _test_utils.torch.misc import set_seed

from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import MXFP8QTensor, NVFP4QTensor

set_seed()


class TestQTensor:
    @pytest.mark.parametrize(
        ("num_bits", "block_sizes"),
        [(4, {-1: 64, "scale_bits": 8, "scale_block_sizes": {-1: 256}}), (4, {-1: 64})],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        ("input_shape", "check_memory"), [((256, 64), True), ((256, 32), False)]
    )  # test
    def test_qtensor(self, num_bits, block_sizes, device, input_dtype, input_shape, check_memory):
        nf4_attr_cfg = QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes=block_sizes,
            fake_quant=False,
        )
        nf4_quantizer = TensorQuantizer(nf4_attr_cfg).to(device)

        # Original tensor
        base_mem = torch.cuda.memory_allocated("cuda")
        x = torch.rand(input_shape).to(device).to(dtype=input_dtype)
        x_allocated = torch.cuda.memory_allocated("cuda")
        bf16_mem_usage = x_allocated - base_mem

        # Perform real quantize
        base_mem = torch.cuda.memory_allocated("cuda")
        nf4_x = nf4_quantizer(x)
        nf4_x_allocated = torch.cuda.memory_allocated("cuda")
        nf4_mem_usage = nf4_x_allocated - base_mem

        # Check the memory saving
        if bf16_mem_usage > 0 and check_memory:
            assert (nf4_mem_usage) / bf16_mem_usage < 0.3

        # De-quantize to origin dtype
        deq_x = nf4_quantizer(nf4_x)

        # Verify the dequantized tensor is close to the original tensor
        assert torch.allclose(deq_x, x, rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize(
        ("num_bits", "block_sizes", "scale_lambda", "scale_to_check"),
        [
            (
                (2, 1),
                {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                lambda x: x / 448.0 / 6.0,
                "_double_scale",
            ),  # NVFP4
            ((4, 3), None, lambda x: x / 448.0, "_scale"),  # FP8
        ],
    )
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cuda"])
    def test_amax_from_tensor_quantizer(
        self, num_bits, block_sizes, scale_lambda, scale_to_check, device, input_dtype
    ):
        # Test FP8 and NVFP4 can get amax from tensor quantizer
        quant_cfg = QuantizerAttributeConfig(
            num_bits=num_bits,
            block_sizes=block_sizes,
            fake_quant=False,
        )
        quantizer = TensorQuantizer(quant_cfg).to(device)

        # Mock amax
        mock_amax = torch.tensor(1.1, device=device)
        quantizer.amax = mock_amax

        x = torch.rand(32, 32).to(device).to(dtype=input_dtype)
        _ = quantizer(x)

        assert hasattr(quantizer, scale_to_check)
        assert torch.allclose(
            getattr(quantizer, scale_to_check), scale_lambda(mock_amax).to(device)
        )

    # Validate the result is consistent with reference implementation
    @pytest.mark.parametrize(
        ("num_bits", "block_sizes", "axis", "test_input", "test_output"),
        [
            # NF4
            (
                4,
                {-1: 2, "scale_bits": 8, "scale_block_sizes": {-1: 4}},
                None,
                torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 1.0156, 2.1719, 3.0156, 3.6094, 5.0000, 5.0625, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # NF4 w/ input padding
            (
                4,
                {-1: 2, "scale_bits": 8, "scale_block_sizes": {-1: 4}},
                None,
                torch.tensor([[0, 1, 2, 3, 4, 5, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 1.0156, 2.1719, 3.0156, 3.6094, 5.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # INT4
            # Note: range of quantize scale is -127 to 127 instead of -128 to 127
            (
                4,
                {-1: 4},
                None,
                torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 0.8516, 2.1406, 2.9844, 4.0000, 5.0000, 6.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # INT4 w/ input padding
            (
                4,
                {-1: 4},
                None,
                torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 3, 3]], dtype=torch.bfloat16),
                torch.tensor(
                    [
                        [
                            0.0000,
                            0.8516,
                            2.1406,
                            2.9844,
                            4.0000,
                            5.0000,
                            6.0000,
                            7.0000,
                            2.9844,
                            2.9844,
                        ]
                    ],
                    dtype=torch.bfloat16,
                ),
            ),
            # INT8 per channel quantization
            (
                8,
                None,
                0,
                torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 0.9922, 1.9844, 2.9844, 3.9688, 4.9688, 5.9688, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # INT8 2D block quantization
            (
                8,
                {-1: 2, -2: 2},
                None,
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 1.0234, 1.9844, 2.9844], [4.0000, 5.0000, 5.9688, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # FP8, 2D block scales
            (
                (4, 3),
                {-1: 2, -2: 2},
                None,
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 0.9844, 2.0000, 3.0000], [3.9375, 5.0000, 6.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # FP8, 2D block scales w/ input padding
            (
                (4, 3),
                {-1: 2, -2: 2},
                None,
                torch.tensor([[0, 1, 3], [4, 5, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 0.9844, 3.0000], [3.9375, 5.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # FP8, 1D block
            (
                (4, 3),
                {-1: 2},
                None,
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 1.0000, 1.9219, 3.0000], [3.9375, 5.0000, 6.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # FP8, per-channel quantization
            (
                (4, 3),
                None,
                0,
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor(
                    [[0.0000, 0.9609, 1.9219, 3.0000], [4.0000, 5.0000, 6.0000, 7.0000]],
                    dtype=torch.bfloat16,
                ),
            ),
            # FP8, per-tensor quantization
            (
                (4, 3),
                None,
                None,
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
                torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.bfloat16),
            ),
            # MXFP4
            (
                (2, 1),
                {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
                None,
                torch.randn([512, 512], dtype=torch.float32),
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_qtensor_accuracy(self, num_bits, axis, block_sizes, test_input, test_output, device):
        quant_attr_cfg = QuantizerAttributeConfig(
            num_bits=num_bits, block_sizes=block_sizes, fake_quant=False, axis=axis
        )
        quantizer = TensorQuantizer(quant_attr_cfg).to(device)

        x = test_input.to(device)

        # Quantize
        q_x = quantizer(x)

        # De-quantize to origin dtype
        deq_x = quantizer(q_x)

        if test_output is not None:
            assert torch.allclose(deq_x, test_output.to(device))

        # compare with fake quant as well
        if device == "cuda":
            # skip for nf4
            if block_sizes and "scale_block_sizes" in block_sizes:
                return
            fake_quant_attr_cfg = QuantizerAttributeConfig(
                num_bits=num_bits, block_sizes=block_sizes, fake_quant=True, axis=axis
            )
            fake_quantizer = TensorQuantizer(fake_quant_attr_cfg).to(device)
            fake_quant_x = fake_quantizer(x)
            assert torch.allclose(fake_quant_x, deq_x.to(device), rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("block_size", [8])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "test_input",
        [
            torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]).unsqueeze(
                0
            ),
            torch.tensor(
                [
                    -0.2500,
                    0.2500,
                    0.7500,
                    1.2500,
                    1.7500,
                    2.7500,
                    3.7500,
                    5.7500,
                    -0.2500,
                    -0.7500,
                    -1.2500,
                    -1.7500,
                    -2.2500,
                    -3.2500,
                    -4.2500,
                    -6.2500,
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    0.2500,
                    0.7500,
                    1.2500,
                    1.7500,
                    2.2500,
                    3.2500,
                    4.2500,
                    6.2500,
                    0.2500,
                    -0.2500,
                    -0.7500,
                    -1.2500,
                    -1.7500,
                    -2.7500,
                    -3.7500,
                    -5.7500,
                ]
            ).unsqueeze(0),
            torch.tensor(
                [
                    0.5000,
                    1.0000,
                    1.5000,
                    2.0000,
                    2.5000,
                    3.5000,
                    4.5000,
                    6.5000,
                    0.5000,
                    0.0000,
                    -0.5000,
                    -1.0000,
                    -1.5000,
                    -2.5000,
                    -3.5000,
                    -5.5000,
                ]
            ).unsqueeze(0),
        ],
    )
    def test_nvfp4_quantize(self, test_input, device, block_size, input_dtype):
        # Define unpack function
        def _unpack_tensor(x):
            # Mapping
            e2m1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
            # Initialize storage for unpacked tensor
            shape = list(x.shape)
            shape[-1] = shape[-1] * 2
            unpacked = torch.zeros(shape, dtype=torch.float16)

            # Get even and odd weights
            odd_weights = x & 0x0F
            even_weights = x >> 4

            # Get unpacked tensor, Verify this with code
            unpacked[..., 1::2] = even_weights
            unpacked[..., 0::2] = odd_weights

            unpacked.apply_(lambda i: e2m1_values[int(i)])

            return unpacked

        # Define input
        x = test_input.to(input_dtype)
        # Quantize inputs
        nvfp4_x = (NVFP4QTensor.quantize(x, block_size))[0]._quantized_data

        # TODO: Move dequantize logic to NVFP4QTensor
        # Compute unscale
        unscale, _ = NVFP4QTensor.get_weights_scaling_factor(x, block_size)
        unscale = unscale.to(torch.float32).unsqueeze(
            -1
        ) * NVFP4QTensor.get_weights_scaling_factor_2(x)

        # Dequantize tensor
        deq_x = _unpack_tensor(nvfp4_x)
        deq_x = deq_x.view(x.shape[0], x.shape[1] // block_size, -1) * unscale
        # Reshape to original dimensions
        deq_x = deq_x.view(x.shape[0], -1)
        deq_x = deq_x.to(input_dtype)

        # Compare with input tensor
        assert torch.allclose(deq_x, x, rtol=2e-1, atol=2e-1)

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "test_input",
        [
            torch.randn((32, 16), dtype=torch.float32),
            torch.tensor([[0.25, 0.75, 1.25], [1.75, 2.5, 3.5]], dtype=torch.float32),
            torch.tensor([[0.1, 2.5, 1.0, 4.8], [1.5, 1.25, 3.25, 5.0]], dtype=torch.float32),
            torch.tensor([[0, 0.75, 1.25], [1.75, 2.5, 5.5]], dtype=torch.float32),
            torch.tensor([[-0.25, -0.75, -1.25], [-1.75, -2.5, -3.5]], dtype=torch.float32),
            torch.tensor(
                [[-0.1, -2.5, -1.0, -4.8], [-1.5, -1.25, -3.25, -5.0]], dtype=torch.float32
            ),
            torch.tensor([[0, -0.75, -1.25], [-1.75, -2.5, -5.5]], dtype=torch.float32),
        ],
    )
    def test_cast_fp4_equivalence(self, test_input, device):
        e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

        def _cast_fp4(weight: torch.Tensor):
            """Converts tensor to uint4."""
            # Get device
            device = weight.device

            # Define mask to perform rounding
            mask = torch.tensor([0, 1, 0, 1, 0, 1, 0]).to(device)
            mask_shape = list(weight.shape)
            mask = mask.expand([*mask_shape, 7])

            sign_bit = (weight < 0).to(torch.uint8)

            # Calculate the ordinal value based on the bounds
            ord = torch.sum(weight.abs().unsqueeze(-1) > e2m1_bounds.to(device), dim=-1).to(
                torch.uint8
            )
            # All values equal to e2m1_bounds at odd indices are rounded up and even indices are rounded down
            round = torch.sum(
                (weight.abs().unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1
            ).to(torch.uint8)
            fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
            return fp4_val

        ref = _cast_fp4(test_input.to(device))
        output = NVFP4QTensor._cast_fp4(test_input.to(device))

        assert torch.all(torch.eq(ref, output))

    @pytest.mark.parametrize(
        "input_shape",
        [(1600, 1600)],
    )
    def test_cast_fp4_impl_gpu_mem(self, input_shape):
        def _get_gpu_mem_used():
            device = torch.device("cuda:0")
            free, total = torch.cuda.mem_get_info(device)
            mem_used = total - free
            return mem_used

        # Do a warmup
        test_input = torch.rand((8, 8), dtype=torch.float32).to("cuda")
        NVFP4QTensor._cast_fp4(test_input)

        test_input = torch.rand((input_shape), dtype=torch.float32).to("cuda")
        torch.cuda.empty_cache()
        # Define input and thresholds
        input_size = test_input.element_size() * test_input.numel()
        before_quantize = _get_gpu_mem_used()
        NVFP4QTensor._cast_fp4(test_input)
        after_quantize = _get_gpu_mem_used()

        assert (after_quantize - before_quantize) < input_size * 2.1

    @pytest.mark.parametrize(
        ("num_bits", "block_sizes", "axis", "input_shape", "expected_output_shape"),
        [
            # FP8, 2D block
            (
                (4, 3),
                {-1: 128, -2: 128},
                None,
                (128, 576),
                (128, 576),
            ),
            # FP8, 2D block
            (
                (4, 3),
                {-1: 128, -2: 128},
                None,
                (576, 128),
                (576, 128),
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_quantized_data_shape(
        self, num_bits, axis, block_sizes, input_shape, expected_output_shape, device
    ):
        quant_attr_cfg = QuantizerAttributeConfig(
            num_bits=num_bits, block_sizes=block_sizes, fake_quant=False, axis=axis
        )
        quantizer = TensorQuantizer(quant_attr_cfg).to(device)
        test_input = torch.rand(input_shape, device=device)

        x = test_input.to(device)
        q_x = quantizer(x)

        assert q_x._quantized_data.shape == expected_output_shape

    @pytest.mark.parametrize("shape", [(128, 64), (64, 128, 32)])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_nvfp4_qdq_correctness(self, shape, input_dtype):
        """Test NVFP4 quantization and dequantization with fast option."""
        block_sizes = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}

        # Create test tensor
        test_tensor = torch.randn(shape, dtype=input_dtype, device="cuda")

        # Quantize tensor
        qtensor, scale, double_scale = NVFP4QTensor.quantize(
            test_tensor, block_sizes[-1], try_tensorrt=False
        )

        # Dequantize using standard approach
        dequant_standard = qtensor.dequantize(
            dtype=input_dtype,
            fast=False,
            scale=scale,
            double_scale=double_scale,
            block_sizes=block_sizes,
        )

        # Check that standard dequantization is close to original
        assert torch.allclose(dequant_standard, test_tensor, atol=0.5, rtol=0.1), (
            f"Standard dequantization differs from original: "
            f"max diff = {(dequant_standard - test_tensor).abs().max()}"
        )

    @pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
    @pytest.mark.parametrize("shape", [(128, 64), (64, 128, 32)])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_nvfp4_dequantize_fast(self, shape, input_dtype):
        """Test NVFP4 quantization and dequantization with fast option."""
        block_sizes = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}

        # Create test tensor
        test_tensor = torch.randn(shape, dtype=input_dtype, device="cuda")

        # Quantize tensor
        qtensor, scale, double_scale = NVFP4QTensor.quantize(
            test_tensor, block_sizes[-1], try_tensorrt=False
        )

        # Dequantize using standard approach
        dequant_standard = qtensor.dequantize(
            dtype=input_dtype,
            fast=False,
            scale=scale,
            double_scale=double_scale,
            block_sizes=block_sizes,
        )

        # Dequantize using fast approach
        dequant_fast = qtensor.dequantize(
            dtype=input_dtype,
            fast=True,
            scale=scale,
            double_scale=double_scale,
            block_sizes=block_sizes,
        )

        # Check that fast and standard dequantization produce the same results
        assert torch.allclose(dequant_fast, dequant_standard, atol=1e-6, rtol=1e-5), (
            f"Fast and standard dequantization differ: "
            f"max diff = {(dequant_fast - dequant_standard).abs().max()}"
        )

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        ("input_shape", "block_sizes"),
        [
            ((128, 1152), {-1: 128}),
            ((256, 256), {-1: 64, -2: 64}),  # 2D block sizes
        ],
    )
    def test_fp8_with_amax_and_block_sizes(self, device, input_dtype, input_shape, block_sizes):
        """Test FP8 quantization with both amax and block_sizes specified."""
        quant_cfg = QuantizerAttributeConfig(
            num_bits=(4, 3),
            block_sizes=block_sizes,
            fake_quant=False,
        )
        quantizer = TensorQuantizer(quant_cfg).to(device)

        # Set a mock amax (scalar) - this was causing the bug
        mock_amax = torch.tensor(1.5, device=device)
        quantizer.amax = mock_amax

        # Create input tensor
        x = torch.randn(input_shape, dtype=input_dtype, device=device)

        # QDQ
        q_x = quantizer(x)
        deq_x = quantizer(q_x)

        assert torch.allclose(deq_x, x, rtol=1e-1, atol=1e-1)
        assert hasattr(quantizer, "_scale")
        assert quantizer._scale.numel() > 1

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (128, 128),
            (256, 64),
            (512, 512),
            # 3D shapes (MoE): (num_experts, out_dim, in_dim)
            (4, 64, 128),
            (1, 64, 128),  # single expert edge case
            (32, 256, 512),  # large-scale MoE
            # Shapes requiring padding (last dim not divisible by block size 32)
            (8, 128, 65),  # odd in_dim
            (128, 65),
            (256, 100),
            (64, 33),
        ],
    )
    def test_mxfp8_quantize_dequantize(self, device, input_dtype, input_shape):
        """Test MXFP8 quantization and dequantization produces correct E8M0 scales."""
        # Create test tensor
        test_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

        # Quantize using MXFP8QTensor
        qtensor, e8m0_scale = MXFP8QTensor.quantize(test_tensor)

        # Verify scale is uint8 (E8M0 format)
        assert e8m0_scale.dtype == torch.uint8, f"Expected uint8 scale, got {e8m0_scale.dtype}"

        # Verify scale shape: last dim is ceil(in_dim / 32), other dims preserved
        expected_scale_shape = (
            *input_shape[:-1],
            math.ceil(input_shape[-1] / MXFP8QTensor.BLOCK_SIZE),
        )
        assert e8m0_scale.shape == expected_scale_shape, (
            f"Expected scale shape {expected_scale_shape}, got {e8m0_scale.shape}"
        )

        # Verify quantized data is FP8 E4M3 and preserves original shape
        assert qtensor._quantized_data.dtype == torch.float8_e4m3fn, (
            f"Expected float8_e4m3fn, got {qtensor._quantized_data.dtype}"
        )
        assert qtensor._quantized_data.shape == input_shape, (
            f"Expected quantized data shape {input_shape}, got {qtensor._quantized_data.shape}"
        )

        # Dequantize
        dequant_tensor = qtensor.dequantize(
            dtype=input_dtype,
            scale=e8m0_scale,
        )

        # Verify dequantized tensor shape and values match original
        assert dequant_tensor.shape == input_shape, (
            f"Expected dequantized shape {input_shape}, got {dequant_tensor.shape}"
        )
        assert torch.allclose(dequant_tensor, test_tensor, rtol=5e-2, atol=5e-2), (
            f"Dequantized tensor differs from original: "
            f"max diff = {(dequant_tensor - test_tensor).abs().max()}"
        )

    @pytest.mark.parametrize("device", ["cuda"])
    def test_mxfp8_e8m0_scale_values(self, device):
        """Test that MXFP8 produces correct E8M0 scale values (power-of-2 only)."""
        # Create a tensor with known amax values per block
        # MXFP8 block size is 32, so create a 2x64 tensor (2 rows, 2 blocks per row)
        test_tensor = torch.zeros((2, 64), dtype=torch.float32, device=device)

        # First block (row 0, elements 0-31): max abs = 1.0, should give exponent ~127-8 = 119
        # (since E4M3 max is 448, log2(1/448) ≈ -8.8, ceil = -8, biased = 127 + (-8) = 119)
        test_tensor[0, :32] = 1.0

        # Second block (row 0, elements 32-63): max abs = 448.0, should give exponent = 127
        # (since 448/448 = 1, log2(1) = 0, biased = 127)
        test_tensor[0, 32:64] = 448.0

        # Third block (row 1, elements 0-31): max abs = 2.0
        test_tensor[1, :32] = 2.0

        # Fourth block (row 1, elements 32-63): max abs = 0.5
        test_tensor[1, 32:64] = 0.5

        # Quantize
        qtensor, e8m0_scale = MXFP8QTensor.quantize(test_tensor)

        # Verify all scales are valid uint8 values
        assert e8m0_scale.dtype == torch.uint8
        assert e8m0_scale.shape == (2, 2)

        # Verify dequantization works
        dequant = qtensor.dequantize(
            dtype=torch.float32,
            scale=e8m0_scale,
        )

        # Check that the dequantized max values per block are close to original
        assert torch.allclose(dequant[0, :32].max(), torch.tensor(1.0, device=device), rtol=0.1)
        assert torch.allclose(dequant[0, 32:64].max(), torch.tensor(448.0, device=device), rtol=0.1)
        assert torch.allclose(dequant[1, :32].max(), torch.tensor(2.0, device=device), rtol=0.1)
        assert torch.allclose(dequant[1, 32:64].max(), torch.tensor(0.5, device=device), rtol=0.1)

    # fmt: off
    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "test_input",
        [
            # FP8 E4M3 boundary test values (max is 448, various powers of 2)
            torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0, 0.5, 0.25,
                           0.125, 0.0625, 0.03125, 0.015625, -1.0, -2.0, -4.0, -8.0, -16.0, -32.0,
                           -64.0, -128.0, -256.0, -448.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625]]),
            # Mix of positive and negative values near E4M3 boundaries
            torch.tensor([[448.0, 416.0, 384.0, 352.0, 320.0, 288.0, 256.0, 224.0, 192.0, 160.0,
                           128.0, 96.0, 64.0, 48.0, 32.0, 24.0, -448.0, -416.0, -384.0, -352.0, -320.0,
                           -288.0, -256.0, -224.0, -192.0, -160.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0]]),
        ],
    )
    def test_mxfp8_quantize_boundary_values(self, test_input, device, input_dtype):
        # fmt: on
        """Test MXFP8 quantization with E4M3 boundary values."""
        x = test_input.to(input_dtype).to(device)
        qtensor, e8m0_scale = MXFP8QTensor.quantize(x)

        # Verify scale is uint8 (E8M0 format)
        assert e8m0_scale.dtype == torch.uint8, f"Expected uint8 scale, got {e8m0_scale.dtype}"

        dequant = qtensor.dequantize(
            dtype=input_dtype,
            scale=e8m0_scale,
        )

        # FP8 E4M3 has limited precision, allow reasonable tolerance
        assert torch.allclose(dequant, x, rtol=5e-2, atol=5e-2), (
            f"Dequantized tensor differs from original: max diff = {(dequant - x).abs().max()}"
        )

    @pytest.mark.parametrize(
        "input_shape",
        [(1600, 1600)],
    )
    def test_mxfp8_quantize_gpu_mem(self, input_shape):
        """Test MXFP8 GPU memory usage during quantization."""

        def _get_gpu_mem_used():
            device = torch.device("cuda:0")
            free, total = torch.cuda.mem_get_info(device)
            return total - free

        # Warmup
        test_input = torch.rand((32, 32), dtype=torch.float32, device="cuda")
        MXFP8QTensor.quantize(test_input)

        test_input = torch.rand(input_shape, dtype=torch.float32, device="cuda")
        torch.cuda.empty_cache()

        input_size = test_input.element_size() * test_input.numel()
        before_quantize = _get_gpu_mem_used()
        MXFP8QTensor.quantize(test_input)
        after_quantize = _get_gpu_mem_used()

        # Memory increase should be reasonable (less than 3x input size)
        # MXFP8 stores FP8 data (1 byte) + uint8 scales, so should be efficient
        assert (after_quantize - before_quantize) < input_size * 3, (
            f"Memory increase too large: {after_quantize - before_quantize} bytes "
            f"for input size {input_size} bytes"
        )

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "input_shape",
        [(128, 64), (256, 128), (512, 256)],
    )
    def test_mxfp8_get_weights_scaling_factor(self, device, input_shape):
        """Test MXFP8 get_weights_scaling_factor returns correct E8M0 scales."""
        weight = torch.randn(input_shape, dtype=torch.float32, device=device)

        # Get scaling factor
        e8m0_scale = MXFP8QTensor.get_weights_scaling_factor(weight)

        # Verify dtype and shape
        assert e8m0_scale.dtype == torch.uint8, f"Expected uint8 scale, got {e8m0_scale.dtype}"
        expected_shape = (input_shape[0], input_shape[1] // MXFP8QTensor.BLOCK_SIZE)
        assert e8m0_scale.shape == expected_shape, (
            f"Expected scale shape {expected_shape}, got {e8m0_scale.shape}"
        )

        # Verify E8M0 values are in valid range [0, 254] (biased exponent = unbiased + 127)
        # The code clamps unbiased exponent to [-127, 127], giving biased range [0, 254]
        # Note: 255 (0xFF) represents NaN in E8M0 and should never appear from valid weights
        assert torch.all(e8m0_scale <= 254), "E8M0 scale contains NaN value (255)"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (64, 64),
            (128, 128),
            (4, 64, 128),  # 3D MoE shape
            # Note: All shapes must have last dim divisible by 32 since
            # get_weights_scaling_factor() requires this (unlike quantize() which pads)
        ],
    )
    def test_mxfp8_quantize_with_precomputed_scale(self, device, input_dtype, input_shape):
        """Test MXFP8 quantize() with pre-computed weights_scaling_factor."""
        test_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

        # Quantize without pre-computed scale (baseline)
        qtensor_auto, scale_auto = MXFP8QTensor.quantize(test_tensor)

        # Pre-compute scale and pass to quantize
        precomputed_scale = MXFP8QTensor.get_weights_scaling_factor(test_tensor)
        qtensor_precomputed, scale_precomputed = MXFP8QTensor.quantize(
            test_tensor, weights_scaling_factor=precomputed_scale
        )

        # Verify scales match
        assert torch.equal(scale_auto, scale_precomputed), (
            "Pre-computed scale should match auto-computed scale"
        )

        # Verify quantized data matches
        assert torch.equal(qtensor_auto._quantized_data, qtensor_precomputed._quantized_data), (
            "Quantized data should match when using pre-computed scale"
        )

        # Verify dequantized results match
        dequant_auto = qtensor_auto.dequantize(dtype=input_dtype, scale=scale_auto)
        dequant_precomputed = qtensor_precomputed.dequantize(
            dtype=input_dtype, scale=scale_precomputed
        )
        assert torch.equal(dequant_auto, dequant_precomputed), (
            "Dequantized results should match"
        )

    @pytest.mark.parametrize(
        ("amax_value", "expected_exponent"),
        [
            (0.0, -127.0),  # Zero amax: minimum exponent
            (448.0, 0.0),  # E4M3_MAX: exponent 0
            (1.0, -8.0),  # log2(1/448) ~ -8.8, ceil = -8
            (1e40, 127.0),  # Very large amax: clamps to max
            (1e-50, -127.0),  # Very small amax: clamps to min
        ],
    )
    def test_mxfp8_compute_e8m0_exponent_edge_cases(self, amax_value, expected_exponent):
        """Test _compute_e8m0_exponent handles edge cases correctly."""
        amax = torch.tensor([amax_value], device="cuda")
        exponent = MXFP8QTensor._compute_e8m0_exponent(amax)
        assert exponent.item() == expected_exponent, (
            f"amax={amax_value} should give exponent {expected_exponent}, got {exponent.item()}"
        )

    def test_mxfp8_get_weights_scaling_factor_asserts_1d_weight(self):
        """Test get_weights_scaling_factor raises assertion for 1D tensor."""
        weight_1d = torch.randn(64, device="cuda")
        with pytest.raises(AssertionError, match="Weight must be at least 2D"):
            MXFP8QTensor.get_weights_scaling_factor(weight_1d)

    def test_mxfp8_get_weights_scaling_factor_asserts_non_divisible(self):
        """Test get_weights_scaling_factor raises assertion when dim not divisible by 32."""
        # 33 is not divisible by 32
        weight = torch.randn(64, 33, device="cuda")
        with pytest.raises(AssertionError, match="must be divisible by MXFP8 block size"):
            MXFP8QTensor.get_weights_scaling_factor(weight)

    @pytest.mark.parametrize("device", ["cuda"])
    def test_mxfp8_quantize_with_scale_asserts(self, device):
        """Test quantize_with_scale raises assertions for invalid inputs."""
        # Test wrong scale dtype assertion
        weight = torch.randn(64, 64, dtype=torch.float32, device=device)
        wrong_dtype_scale = torch.randn(64, 2, dtype=torch.float32, device=device)
        with pytest.raises(AssertionError, match="weights_scaling_factor must be"):
            MXFP8QTensor.quantize_with_scale(weight, wrong_dtype_scale)

        # Test non-divisible dimension assertion
        weight_bad_dim = torch.randn(64, 33, dtype=torch.float32, device=device)
        scale = torch.randint(0, 255, (64, 1), dtype=torch.uint8, device=device)
        with pytest.raises(AssertionError, match="must be divisible by MXFP8 block size"):
            MXFP8QTensor.quantize_with_scale(weight_bad_dim, scale)

    @pytest.mark.parametrize("device", ["cuda"])
    def test_mxfp8_get_weights_scaling_factor_from_quantizer_3d_moe(self, device):
        """Test get_weights_scaling_factor_from_quantizer handles 3D MoE tensors."""
        input_shape = (4, 64, 128)  # (num_experts, out_dim, in_dim)
        weight = torch.randn(input_shape, dtype=torch.float32, device=device)

        class MockQuantizer:
            block_sizes = {-1: MXFP8QTensor.BLOCK_SIZE}
            _scale = None

        quantizer = MockQuantizer()

        # Test when _scale is None (should compute from weight)
        scale = MXFP8QTensor.get_weights_scaling_factor_from_quantizer(weight, quantizer)

        expected_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2] // MXFP8QTensor.BLOCK_SIZE,
        )
        assert scale.shape == expected_shape

        # Test when _scale is provided with correct 3D shape
        quantizer._scale = torch.randint(0, 255, expected_shape, dtype=torch.uint8, device=device)
        scale_from_quantizer = MXFP8QTensor.get_weights_scaling_factor_from_quantizer(
            weight, quantizer
        )
        assert torch.equal(scale_from_quantizer, quantizer._scale)

    @pytest.mark.parametrize("device", ["cuda"])
    def test_mxfp8_get_weights_scaling_factor_from_quantizer_scale_shape_mismatch(self, device):
        """Test get_weights_scaling_factor_from_quantizer raises assertion on shape mismatch."""
        input_shape = (4, 64, 128)  # (num_experts, out_dim, in_dim)
        weight = torch.randn(input_shape, dtype=torch.float32, device=device)

        class MockQuantizer:
            block_sizes = {-1: MXFP8QTensor.BLOCK_SIZE}
            # Wrong shape: 2D instead of 3D (missing num_experts dimension)
            _scale = torch.randint(
                0, 255, (64, 4), dtype=torch.uint8, device=device
            )

        quantizer = MockQuantizer()

        with pytest.raises(AssertionError, match="Scale shape .* does not match expected shape"):
            MXFP8QTensor.get_weights_scaling_factor_from_quantizer(weight, quantizer)

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_mxfp8_dequantize_default_dtype(self, device, input_dtype):
        """Test dequantize uses original dtype when dtype=None."""
        input_tensor = torch.randn(64, 64, dtype=input_dtype, device=device)
        qtensor, e8m0_scale = MXFP8QTensor.quantize(input_tensor)

        # Dequantize without specifying dtype
        dequant = qtensor.dequantize(scale=e8m0_scale)

        assert dequant.dtype == input_dtype

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "input_shape",
        [
            (64, 64),
            (128, 128),
            (4, 64, 128),  # 3D MoE shape
        ],
    )
    def test_mxfp8_fake_quant(self, device, input_dtype, input_shape):
        """Test MXFP8 fake quantization via TensorQuantizer matches real quant+dequant."""
        block_sizes = {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}

        # Create fake quant quantizer
        fake_quant_cfg = QuantizerAttributeConfig(
            num_bits=(4, 3), block_sizes=block_sizes, fake_quant=True, axis=None
        )
        fake_quantizer = TensorQuantizer(fake_quant_cfg).to(device)

        # Create real quant quantizer
        real_quant_cfg = QuantizerAttributeConfig(
            num_bits=(4, 3), block_sizes=block_sizes, fake_quant=False, axis=None
        )
        real_quantizer = TensorQuantizer(real_quant_cfg).to(device)

        # Test tensor
        test_tensor = torch.randn(input_shape, dtype=input_dtype, device=device)

        # Fake quant output
        fake_quant_output = fake_quantizer(test_tensor)

        # Real quant + dequant
        q_tensor = real_quantizer(test_tensor)
        real_dequant_output = real_quantizer(q_tensor)

        # Verify fake quant matches real quant+dequant
        assert fake_quant_output.shape == test_tensor.shape
        assert fake_quant_output.dtype == test_tensor.dtype
        assert torch.allclose(fake_quant_output, real_dequant_output, rtol=5e-2, atol=5e-2), (
            f"Fake quant differs from real quant+dequant: "
            f"max diff = {(fake_quant_output - real_dequant_output).abs().max()}"
        )
