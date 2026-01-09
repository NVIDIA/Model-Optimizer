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

import pytest
import torch
from _test_utils.torch.misc import set_seed

from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import NVFP4QTensor

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
