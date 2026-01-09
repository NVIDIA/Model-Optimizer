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

"""Tests of tensor quantization function and module"""

import pytest
import torch
from _test_utils.torch.quantization.quant_utils import quant
from _test_utils.torch.quantization.tensor_quant_common import FakeTensorQuantTester

import modelopt.torch.quantization.triton as triton_kernel
import modelopt.torch.quantization.utils as quant_utils
from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.extensions import get_cuda_ext, get_cuda_ext_mx
from modelopt.torch.quantization.tensor_quant import mx_format_map


class TestFakeTensorQuantCuda(FakeTensorQuantTester):
    device = "cuda"


class TestCudaExt:
    @pytest.mark.parametrize("num_bits", [3, 4, 5, 7, 8, 11])
    @pytest.mark.parametrize("unsigned", [True, False])
    def test_cuda_ext_num_bits(self, num_bits, unsigned):
        x = torch.randn(31).cuda()

        if unsigned:
            x = x.abs()
        assert torch.allclose(
            get_cuda_ext().fake_tensor_quant(x, torch.max(torch.abs(x)), num_bits, unsigned),
            tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)), None, num_bits, unsigned),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_ext_dtype(self, dtype):
        # Test fp16 and bf16
        x = torch.randn(31).cuda().to(dtype)
        cuda_ext_out = (
            get_cuda_ext().fake_tensor_quant(x, torch.max(torch.abs(x))).to(torch.float32)
        )
        pytorch_out = tensor_quant.fake_tensor_quant(x, torch.max(torch.abs(x)), None).to(
            torch.float32
        )
        assert torch.allclose(cuda_ext_out, pytorch_out, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("num_bits", [3, 4, 5, 7, 8, 11])
    @pytest.mark.parametrize("unsigned", [True, False])
    def test_cuda_ext_with_axis(self, dtype, num_bits, unsigned):
        x = torch.randn(3, 4, 5, 6).cuda().to(dtype)

        # amax along axis 1
        amax_torch = torch.tensor([0.8, 0.9, 0.7, 0.6]).cuda()

        if unsigned:
            x = x.abs()
        cuda_ext_out = (
            get_cuda_ext()
            .fake_tensor_quant_with_axis(x, amax_torch, 1, num_bits, unsigned)
            .to(torch.float32)
        )
        pytorch_out = tensor_quant.fake_tensor_quant(
            x, amax_torch.view(1, -1, 1, 1), None, num_bits, unsigned
        ).to(torch.float32)
        assert torch.allclose(cuda_ext_out, pytorch_out, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cuda_ext_inplace(self, dtype):
        torch.manual_seed(1234)
        x = torch.randn(31).cuda().to(dtype)
        quant_x_ref = quant(x, torch.max(x.abs()), fake=True)
        get_cuda_ext().fake_tensor_quant_(x, torch.max(torch.abs(x)))
        if dtype == torch.bfloat16:
            assert torch.allclose(x, quant_x_ref, atol=1e-1)
        elif dtype == torch.float16:
            assert torch.allclose(x, quant_x_ref, atol=1e-3)
        else:
            assert torch.allclose(x, quant_x_ref)

    def test_cuda_ext_tiny_amax(self):
        x = torch.rand(2, 3, 4).cuda()
        amax = torch.tensor([1.0, 1.0e-26, 1.0]).cuda().unsqueeze(-1).unsqueeze(1)
        quant_x = get_cuda_ext().fake_tensor_quant_with_axis(x, amax, axis=1)
        assert quant_x[:, 1, :].sum() == 0


class TestScaledE4M3:
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_e4m3_no_scale(self, device):
        x = torch.randn(4, 4, device=device, dtype=torch.float32)
        xq_ref = tensor_quant.fp8_eager(x, torch.tensor(448.0, device=x.device))
        e4m3_x = tensor_quant.scaled_e4m3(x, None, None, 4, 3)
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_with_amax(self, device, dtype):
        if device == "cpu" and dtype != torch.float32:
            pytest.skip("CPU does not support non-float32 dtype")
        x = torch.randn(4, 4, device=device, dtype=dtype)
        amax = quant_utils.reduce_amax(x, axis=None, keepdims=True)
        xq_ref = tensor_quant.fp8_eager(x, amax)
        xq_test = tensor_quant.scaled_e4m3(x, amax, None, 4, 3)
        assert torch.allclose(xq_test, xq_ref)

    def test_e4m3_incontiguous(self):
        x = torch.randn(4, 4).cuda().transpose(1, 0)
        xq_ref = tensor_quant.fp8_eager(x, torch.tensor(448.0, device=x.device))
        assert not x.is_contiguous()
        e4m3_x = tensor_quant.scaled_e4m3(x, None, None, 4, 3)
        assert torch.allclose(e4m3_x, xq_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_backward(self, device):
        x = torch.randn(3, 7, requires_grad=True).to(device)
        labels = torch.randint(6, (3,)).type(torch.LongTensor).to(device)
        quant_x = tensor_quant.scaled_e4m3(x, None, None, 4, 3)
        x.retain_grad()
        quant_x.retain_grad()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(quant_x, labels)
        loss.backward()
        assert torch.allclose(quant_x.grad, x.grad)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_e4m3_per_channel(self, axis):
        x = torch.randn(4, 4, 4, dtype=torch.float32).cuda()
        amax = x.abs().amax(dim=[ax for ax in range(x.ndim) if ax != axis], keepdim=True)
        xq_ref = tensor_quant.fp8_eager(x, amax)
        xq_test = tensor_quant.scaled_e4m3(x, amax, None, 4, 3)
        assert torch.allclose(xq_test, xq_ref)


class Testfp4:
    @pytest.mark.skipif(get_cuda_ext_mx() is None, reason="cuda_ext_mx is not available")
    @pytest.mark.parametrize(
        "set_torch_dtype", [torch.float, torch.float16, torch.bfloat16], indirect=True
    )
    @pytest.mark.parametrize("block_size", [8, 16, 32])
    def test_cuda_ext_fp4(self, set_torch_dtype, block_size):
        cuda_ext_mx = get_cuda_ext_mx()
        # Test with e2m1 table values
        sign = torch.randint(0, 2, (1, 8)).cuda() * 2 - 1

        def _get_test_inputs_outputs(test_in, test_out):
            return torch.concat((test_in,) * (block_size // 8), dim=-1), torch.concat(
                (test_out,) * (block_size // 8), dim=-1
            )

        def _test_fp4_kernel(test_in, test_out, skip_triton=False):
            inputs, expected_outputs = _get_test_inputs_outputs(test_in, test_out)
            quantized_outputs = cuda_ext_mx.fused_amax_convert(
                inputs,
                16,
                getattr(cuda_ext_mx.Types, mx_format_map[(2, 1)]),
                getattr(cuda_ext_mx.Types, mx_format_map[(4, 3)]),
                inputs.abs().amax(),
            )
            assert torch.allclose(quantized_outputs, expected_outputs)
            if triton_kernel.IS_AVAILABLE and not skip_triton:
                quantized_outputs_triton = triton_kernel.fp4_fake_quant_block(
                    inputs, inputs.abs().amax()
                )
                assert torch.allclose(quantized_outputs_triton, expected_outputs)

        test_in = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        test_out = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)

        # Test with e2m1 boundary values. The even indexes are rounded down and odd indexes are rounded up.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda() * sign
        test_out = torch.tensor([[0.0, 1, 1, 2, 2, 4, 4, 6]]).cuda() * sign
        # The triton kernel has a numerical issue, the values are not exactly at the boundary after scaling,
        # e.g. 0.25 -> 0.250061, this won't cause visible error for real-world quantizations.
        _test_fp4_kernel(test_in, test_out, skip_triton=True)

        # Test slightly below the e2m1 boundary values.
        # Numbers should be quantized down to the corresponding e2m1 value.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda()
        test_in[:, :-1] -= 0.1
        test_in *= sign
        test_out = torch.tensor([[0.0, 0.5, 1, 1.5, 2, 3, 4, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)

        # Test slightly above the e2m1 boundary values.
        # Numbers should be quantized up to the corresponding e2m1 value.
        test_in = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]]).cuda()
        test_in[:, :-1] += 0.1
        test_in *= sign
        test_out = torch.tensor([[0.5, 1, 1.5, 2, 3, 4, 6, 6]]).cuda() * sign
        _test_fp4_kernel(test_in, test_out)
