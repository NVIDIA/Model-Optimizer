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

import modelopt.torch.quantization.extensions as ext

# Override default timeout as these tests JIT-compile the CUDA extensions, which is slow
pytestmark = pytest.mark.timeout(240)


# Compile extensions first so it does not count towards time used to run a test that needs it
def test_cuda_ext():
    assert ext.get_cuda_ext() is not None


def test_cuda_ext_fp8():
    assert ext.get_cuda_ext_fp8() is not None


def test_cuda_ext_mx():
    assert ext.get_cuda_ext_mx() is not None


def test_cuda_ext_iq1_s():
    assert ext.get_cuda_ext_iq1_s() is not None


def test_cuda_ext_iq2_xs():
    assert ext.get_cuda_ext_iq2_xs() is not None


_IQ_EXTENSIONS = (
    pytest.param(ext.get_cuda_ext_iq1_s, (2048, 8), 50, id="iq1_s"),
    pytest.param(ext.get_cuda_ext_iq2_xs, (512, 8), 74, id="iq2_xs"),
)


@pytest.mark.parametrize(("get_extension", "grid_shape", "payload_bytes"), _IQ_EXTENSIONS)
def test_cuda_ext_iq_zero_block_layout(get_extension, grid_shape, payload_bytes):
    extension = get_extension(raise_if_failed=True)
    weight = torch.zeros((2, 256), device="cuda", dtype=torch.bfloat16)
    grid = torch.zeros(grid_shape, device="cuda", dtype=torch.float32)

    packed = extension.pack(weight, grid)

    assert packed.shape == (2, payload_bytes)
    assert not packed.any()


@pytest.mark.parametrize(("get_extension", "grid_shape", "_payload_bytes"), _IQ_EXTENSIONS)
def test_cuda_ext_iq_rejects_unsupported_dtype(get_extension, grid_shape, _payload_bytes):
    extension = get_extension(raise_if_failed=True)
    weight = torch.ones((1, 256), device="cuda").to(torch.float8_e4m3fn)
    grid = torch.zeros(grid_shape, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="supports float32, float64, float16, and bfloat16"):
        extension.pack(weight, grid)


@pytest.mark.parametrize(("get_extension", "grid_shape", "_payload_bytes"), _IQ_EXTENSIONS)
def test_cuda_ext_iq_rejects_row_straddling_input(get_extension, grid_shape, _payload_bytes):
    extension = get_extension(raise_if_failed=True)
    weight = torch.ones((512, 384), device="cuda", dtype=torch.bfloat16)
    grid = torch.zeros(grid_shape, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="innermost dimension must be a multiple of 256"):
        extension.pack(weight, grid)
