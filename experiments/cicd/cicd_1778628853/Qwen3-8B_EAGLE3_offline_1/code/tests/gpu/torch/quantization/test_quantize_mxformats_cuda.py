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

import pytest
import torch

from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.tensor_quant import mx_format_map

if get_cuda_ext_mx() is None:
    pytest.skip("cuda_ext_mx is not available", allow_module_level=True)

cuda_ext_mx = get_cuda_ext_mx()

shapes = [[8, 32], [128, 256]]
block_sizes = [8, 16, 32]
dtypes = [torch.float, torch.half, torch.bfloat16]
num_bits_all = [8, (2, 1), (3, 2), (2, 3), (4, 3), (5, 2)]
scale_bits_all = [(4, 3), (8, 0)]


def _get_test_inputs_outputs(test_in, test_out, block_size, in_size):
    return test_in.repeat((1, max(block_size // in_size, 1))), test_out.repeat(
        (1, max(block_size // in_size, 1))
    )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("block_size", block_sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("num_bits", num_bits_all)
@pytest.mark.parametrize("scale_bits", scale_bits_all)
def test_encode(shape, block_size, dtype, num_bits, scale_bits):
    x = torch.randn(*shape).cuda().to(dtype)

    y = cuda_ext_mx.fused_amax_convert(
        x,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[num_bits]),
        getattr(cuda_ext_mx.Types, mx_format_map[scale_bits]),
        None,
    )

    assert y.dtype == x.dtype


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("block_size", block_sizes)
def test_mxfp4(block_size, dtype):
    test_in = torch.tensor(
        [
            [0.1, 0.2, 0.35, 0.16, 0.85, 0.9, 0.67, 0.76],  # small values
            [0, 0.5, 1, 1.5, 2, 3, 4, 6],  # e2m1 table values
            [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5],  # e2m2 table bounds
            [0.1, 2.0, 10, 16, 39, 49.5, 30.5, 98.8],  # large values
            [2.3, 9.0, 0.5, 100.5, 900.3, 10000, 2000, 360000],  # large values
        ],
        dtype=dtype,
        device="cuda",
    )
    test_out = torch.tensor(
        [
            [0.125, 0.25, 0.375, 0.125, 0.75, 1.0, 0.75, 0.75],
            [-0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            [-0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
            [0.0, 0.0, 16.0, 16.0, 32.0, 48.0, 32.0, 96.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 393216.0],
        ],
        dtype=dtype,
        device="cuda",
    )
    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 8
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[(2, 1)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    assert torch.allclose(expected_outputs, outputs)


def test_mxfp6():
    dtype = torch.float32
    block_size = 16
    # fmt: off
    test_in = torch.tensor(
        [
            [0.02, 0.4, 0.7, 0.96, 0.04, 0.52, 0.76, 0.89, 0.24, 0.93, 0.41, 0.05, 0.03, 0.1, 0.76, 0.73, 0.38, 0.1, 0.11, 0.46, 0.35, 0.0, 0.37, 0.99, 0.36, 0.98, 0.31, 0.22, 0.55, 0.18, 0.06, 0.61],  # small values  # noqa: E501
            [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28],  # e3m2 table values  # noqa: E501
            [0, 0.03125, 0.09375, 0.15625, 0.21875, 0.28125, 0.34375, 0.40625, 0.46875, 0.5625, 0.6875, 0.8125, 0.9375, 1.125, 1.375, 1.625, 1.875, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 6.5, 7.5, 9, 11, 13, 15, 18, 22, 26],  # e3m2 table bounds  # noqa: E501
            [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],  # e2m3 table values  # noqa: E501
            [0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125, 1.4375, 1.5625, 1.6875, 1.8125, 1.9375, 2.125, 2.375, 2.625, 2.875, 3.125, 3.375, 3.625, 3.875, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25],  # e2m3 table bounds  # noqa: E501
            [0.1, 2.0, 10, 16, 39, 49.5, 30.5, 98.8, 66.0, 77.0, 55.0, 88.0, 22.0, 11.0, 18.9, 92.9, 73.0, 9.0, 63.0, 23.0, 87.5, 53.2, 65.5, 62.5, 89.4, 76.9, 65.2, 33.0, 32.5, 22.5, 29.0, 83.0],  # large values  # noqa: E501
            [1056, 5504, 8083, 4205, 6471, 9509, 1168, 9106, 8210, 2961, 6371, 8404, 7319, 3836, 8661, 4801, 4942, 9202, 2605, 505, 3571, 4010, 3853, 4719, 50, 9243, 5110, 167, 7073, 6110, 889, 8416],  # large values  # noqa: E501
        ],
        dtype=dtype,
        device="cuda",
    )
    # e3m2
    test_out = torch.tensor(
        [
            [0.0195, 0.375, 0.75, 1.0, 0.0391, 0.5, 0.75, 0.875, 0.25, 0.875, 0.4375, 0.0469, 0.0312, 0.0938, 0.75, 0.75, 0.375, 0.0938, 0.1094, 0.4375, 0.375, -0.0, 0.375, 1.0, 0.375, 1.0, 0.3125, 0.2188, 0.5, 0.1875, 0.0625, 0.625],  # noqa: E501
            [-0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0],  # noqa: E501
            [-0.0, 0.0312, 0.0938, 0.1562, 0.2188, 0.25, 0.375, 0.375, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 12.0, 12.0, 16.0, 16.0, 24.0, 24.0],  # noqa: E501
            [-0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.25, 1.5, 1.5, 1.5, 1.75, 2.0, 2.0, 2.0, 2.5, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0],  # noqa: E501
            [-0.0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],  # noqa: E501
            [0.0, 2.0, 10.0, 16.0, 40.0, 48.0, 32.0, 96.0, 64.0, 80.0, 56.0, 96.0, 24.0, 12.0, 20.0, 96.0, 80.0, 8.0, 64.0, 24.0, 80.0, 56.0, 64.0, 64.0, 96.0, 80.0, 64.0, 32.0, 32.0, 24.0, 28.0, 80.0],  # noqa: E501
            [1024.0, 5120.0, 8192.0, 4096.0, 6144.0, 10240.0, 1280.0, 8192.0, 8192.0, 3072.0, 6144.0, 8192.0, 7168.0, 3584.0, 8192.0, 5120.0, 5120.0, 8192.0, 2560.0, 512.0, 3584.0, 4096.0, 4096.0, 5120.0, 64.0, 10240.0, 5120.0, 160.0, 7168.0, 6144.0, 896.0, 8192.0],  # noqa: E501
        ],
        dtype=dtype,
        device="cuda",
    )
    # fmt: on
    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 32
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[(3, 2)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    # this suffers from the precision issue, mx formats have very low precision
    assert torch.allclose(expected_outputs, outputs, atol=1e-3)

    # fmt: off
    # e2m3
    test_out = torch.tensor(
        [
            [0.0312, 0.4062, 0.6875, 0.9375, 0.0312, 0.5, 0.75, 0.875, 0.25, 0.9375, 0.4062, 0.0625, 0.0312, 0.0938, 0.75, 0.75, 0.375, 0.0938, 0.125, 0.4688, 0.3438, -0.0, 0.375, 1.0, 0.375, 1.0, 0.3125, 0.2188, 0.5625, 0.1875, 0.0625, 0.625],  # noqa: E501
            [-0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0],  # noqa: E501
            [-0.0, 0.0312, 0.0938, 0.1562, 0.2188, 0.2812, 0.3438, 0.4062, 0.4688, 0.5625, 0.6875, 0.8125, 0.9375, 1.125, 1.375, 1.625, 2.0, 2.0, 3.0, 3.0, 4.0, 4.5, 5.5, 6.5, 7.5, 9.0, 11.0, 13.0, 15.0, 18.0, 22.0, 26.0],  # noqa: E501
            [-0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],  # noqa: E501
            [-0.0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 3.5, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],  # noqa: E501
            [0.0, 2.0, 10.0, 16.0, 40.0, 48.0, 30.0, 96.0, 64.0, 80.0, 56.0, 88.0, 22.0, 12.0, 18.0, 96.0, 72.0, 8.0, 64.0, 24.0, 88.0, 52.0, 64.0, 64.0, 88.0, 80.0, 64.0, 32.0, 32.0, 22.0, 28.0, 80.0],  # noqa: E501
            [1024.0, 5632.0, 8192.0, 4096.0, 6656.0, 9216.0, 1280.0, 9216.0, 8192.0, 3072.0, 6144.0, 8192.0, 7168.0, 3840.0, 8192.0, 4608.0, 5120.0, 9216.0, 2560.0, 512.0, 3584.0, 4096.0, 3840.0, 4608.0, 0.0, 9216.0, 5120.0, 256.0, 7168.0, 6144.0, 768.0, 8192.0],  # noqa: E501
        ],
        dtype=dtype,
        device="cuda",
    )
    # fmt: on

    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 32
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[(2, 3)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    # this suffers from the precision issue, mx formats have very low precision
    assert torch.allclose(expected_outputs, outputs, atol=1e-3)


def test_mxfp8():
    dtype = torch.float32
    block_size = 8
    # fmt: off
    test_in = torch.tensor(
        [
            [0.084, 0.273, 0.049, 0.165, 0.229, 0.827, 0.571, 0.361],
            [2.098, 0.135, 3.819, 2.659, 9.841, 9.227, 8.088, 8.806],
            [86.013, 8.952, 89.435, 76.524, 72.668, 59.243, 79.387, 57.657],
            [284.205, 189.103, 127.758, 305.991, 345.731, 98.383, 676.081, 540.471],
            [4913.512, 946.531, 4334.426, 3519.271, 8020.535, 5235.864, 7549.414, 3807.634],
            [7.874e04, 8.358e04, 8.027e04, 4.939e02, 1.534e03, 1.159e04, 1.118e04, 1.481e04],
            [6.169e05, 7.527e05, 6.475e05, 8.206e05, 6.720e04, 3.948e05, 7.105e05, 7.402e05],
            [8.121e06, 2.110e06, 8.273e06, 2.180e06, 3.217e06, 7.207e06, 3.305e06, 1.853e06],
        ],
        dtype=dtype,
        device="cuda",
    )
    # e4m3
    test_out = torch.tensor(
        [
            [0.0859, 0.2812, 0.0508, 0.1719, 0.2344, 0.8125, 0.5625, 0.3750],
            [2, 0.1406, 3.7500, 2.7500, 10, 9, 8, 9],
            [88, 9, 88, 80, 72, 60, 80, 56],
            [288, 192, 128, 320, 352, 96, 704, 512],
            [5120, 960, 4096, 3584, 8192, 5120, 7680, 3840],
            [81920, 81920, 81920, 480, 1536, 11264, 11264, 14336],
            [589824, 720896, 655360, 851968, 65536, 393216, 720896, 720896],
            [7864320, 2097152, 8388608, 2097152, 3145728, 7340032, 3407872, 1835008],
        ],
        dtype=dtype,
        device="cuda",
    )
    # fmt: on

    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 8
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[(4, 3)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    # this suffers from the precision issue, mx formats have very low precision
    assert torch.allclose(expected_outputs, outputs, atol=1e-3)

    # fmt: off
    # e5m2
    test_out = torch.tensor(
        [
            [0.0781, 0.2500, 0.0469, 0.1562, 0.2188, 0.8750, 0.6250, 0.3750],
            [2, 0.1250, 4, 2.5000, 10, 10, 8, 8],
            [80, 8, 96, 80, 80, 56, 80, 56],
            [256, 192, 128, 320, 320, 96, 640, 512],
            [5120, 896, 4096, 3584, 8192, 5120, 7168, 3584],
            [81920, 81920, 81920, 512, 1536, 12288, 10240, 14336],
            [655360, 786432, 655360, 786432, 65536, 393216, 655360, 786432],
            [8388608, 2097152, 8388608, 2097152, 3145728, 7340032, 3145728, 1835008],
        ],
        dtype=dtype,
        device="cuda",
    )
    # fmt: on
    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 8
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[(5, 2)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    # this suffers from the precision issue, mx formats have very low precision
    assert torch.allclose(expected_outputs, outputs, atol=1e-3)


def test_mxint8():
    dtype = torch.float32
    block_size = 8
    # fmt: off
    test_in = torch.tensor(
        [
            [0.999, 0.790, 0.298, 0.795, 0.928, 0.190, 0.187, 0.400],
            [0.084, 0.273, 0.049, 0.165, 0.229, 0.827, 0.571, 0.361],
            [2.098, 0.135, 3.819, 2.659, 9.841, 9.227, 8.088, 8.806],
            [86.013, 8.952, 89.435, 76.524, 72.668, 59.243, 79.387, 57.657],
            [284.205, 189.103, 127.758, 305.991, 345.731, 98.383, 676.081, 540.471],
            [4913.512, 946.531, 4334.426, 3519.271, 8020.535, 5235.864, 7549.414, 3807.634],
            [7.874e04, 8.358e04, 8.027e04, 4.939e02, 1.534e03, 1.159e04, 1.118e04, 1.481e04],
            [6.169e05, 7.527e05, 6.475e05, 8.206e05, 6.720e04, 3.948e05, 7.105e05, 7.402e05],
            [8.121e06, 2.110e06, 8.273e06, 2.180e06, 3.217e06, 7.207e06, 3.305e06, 1.853e06],
        ],
        dtype=dtype,
        device="cuda",
    )
    test_out = torch.tensor(
        [
            [1.0, 0.7969, 0.2969, 0.7969, 0.9219, 0.1875, 0.1875, 0.4062],
            [0.0859, 0.2734, 0.0469, 0.1641, 0.2266, 0.8281, 0.5703, 0.3594],
            [2.1250, 0.1250, 3.8750, 2.6250, 9.8750, 9.2500, 8.1250, 8.7500],
            [86, 9, 89, 77, 73, 59, 79, 58],
            [288, 192, 128, 304, 344, 96, 680, 544],
            [4928, 960, 4352, 3520, 8000, 5248, 7552, 3776],
            [78848, 83968, 79872, 0, 1024, 11264, 11264, 14336],
            [614400, 753664, 647168, 819200, 65536, 393216, 712704, 737280],
            [8126464, 2097152, 8257536, 2162688, 3211264, 7208960, 3276800, 1835008],
        ],
        dtype=dtype,
        device="cuda",
    )
    # fmt: on

    sign = torch.randint(0, 2, test_in.shape).cuda() * 2 - 1

    inputs, expected_outputs = _get_test_inputs_outputs(
        test_in * sign, test_out * sign, block_size, 8
    )

    outputs = cuda_ext_mx.fused_amax_convert(
        inputs,
        block_size,
        getattr(cuda_ext_mx.Types, mx_format_map[8]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )
    # this suffers from the precision issue, mx formats have very low precision
    assert torch.allclose(expected_outputs, outputs, atol=1e-3)
