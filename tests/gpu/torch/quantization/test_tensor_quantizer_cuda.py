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

"""Tests of tensor quantizer"""

import contextlib

import pytest
import torch
from _test_utils.torch.quantization.tensor_quantizer_common import (
    BlockQuantTester,
    TensorQuantizerTester,
)
from pydantic import ValidationError

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.nn.modules import tensor_quantizer


class TestTensorQuantizerCuda(TensorQuantizerTester):
    device = "cuda"


class TestBlockQuantCuda(BlockQuantTester):
    device = "cuda"


class TestTensorQuantizerE4M3:
    @pytest.mark.parametrize(
        ("E", "M", "axis"), [(5, 2, None), (4, 3, None), (4, 3, 1), (7, 3, None)]
    )
    def test_e4m3(self, E, M, axis):  # noqa: N803
        is_error_expected = E != 4 or M != 3
        with pytest.raises(ValidationError) if is_error_expected else contextlib.nullcontext():
            e4m3_desc = QuantizerAttributeConfig(num_bits=(E, M), axis=axis)
            e4m3_quantizer = tensor_quantizer.TensorQuantizer(e4m3_desc).cuda()

            x = torch.rand(3, 6, 7, 7).cuda()

            e4m3_x = e4m3_quantizer(x)
            ref = tensor_quant.scaled_e4m3(x, e4m3_quantizer._get_amax(x), None, E, M)
            assert torch.allclose(e4m3_x, ref)

    def test_non_current_gpu(self, need_2_gpus):
        x = torch.randn(3, 4)
        e4m3_desc = QuantizerAttributeConfig(num_bits=(4, 3), axis=None)
        quantizer = tensor_quantizer.TensorQuantizer(e4m3_desc).cuda()
        xq_ref = quantizer(x.to("cuda:0"))
        xq_test = quantizer(x.to("cuda:1"))
        assert torch.allclose(xq_ref, xq_test.to("cuda:0"))


@pytest.mark.skipif(get_cuda_ext_mx() is None, reason="cuda_ext_mx is not available")
class TestTensorQuantizerfp4:
    def test_fp4(self):
        fp4_quantizer = tensor_quantizer.TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
            )
        ).cuda()

        x = torch.rand(2, 1, 16).cuda()

        fp4_x = fp4_quantizer(x)
        ref = tensor_quant.dynamic_block_quant(x, 16, x.abs().amax(), None, (2, 1), (4, 3))
        assert torch.allclose(fp4_x, ref)

        assert fp4_quantizer._get_amax(x) == x.abs().amax()

    def test_fp4_backward(self):
        fp4_quantizer = tensor_quantizer.TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
            )
        ).cuda()

        x = torch.rand(2, 1, 16).cuda()
        with torch.no_grad():
            fp4_quantizer.amax = torch.median(x.abs())

        x.requires_grad = True
        loss = fp4_quantizer(x).sum()
        loss.backward()

        assert torch.allclose(x.grad, torch.ones_like(x.grad) * (x.abs() <= fp4_quantizer.amax))

    def test_fp4_non_contiguous_input(self):
        contiguous_tensor = torch.ones(2, 16).cuda()
        large_tensor = torch.ones(2, 32).cuda()
        large_tensor[:, :16] = torch.randn(2, 16).cuda()
        non_contiguous_tensor = large_tensor[:, 16:].cuda()

        assert torch.equal(contiguous_tensor, non_contiguous_tensor)
        assert contiguous_tensor.is_contiguous()
        assert not non_contiguous_tensor.is_contiguous()

        quantizer = tensor_quantizer.TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=(2, 1),
                block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                axis=None,
                enable=True,
            )
        ).cuda()
        output_contiguous = quantizer(contiguous_tensor)
        output_non_contiguous = quantizer(non_contiguous_tensor)
        assert torch.equal(output_contiguous, output_non_contiguous)
