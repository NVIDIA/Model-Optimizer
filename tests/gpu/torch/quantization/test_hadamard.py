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
import torch.nn as nn

fast_hadamard_transform = pytest.importorskip("fast_hadamard_transform")

try:
    fast_hadamard_transform.hadamard_transform(torch.randn(1, 2, device="cuda"))
except Exception:
    pytest.skip(
        "fast_hadamard_transform CUDA kernels not available for this GPU", allow_module_level=True
    )

from _test_utils.torch.quantization.models import SDPAAttention

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.conversion import (
    set_quantizer_by_cfg,
    set_quantizer_by_cfg_context,
)
from modelopt.torch.quantization.nn.functional import normalized_hadamard_transform


@pytest.mark.parametrize(
    "dim",
    [8, 16],
)
def test_hadamard_transform(dim):
    x = torch.rand(4, dim).cuda()
    xxt = x @ x.T
    x_h = normalized_hadamard_transform(x)
    xxt_h = x_h @ x_h.T
    # The numerical error can be large, especially for 16-bit floats.
    assert torch.allclose(xxt_h, xxt, atol=0.05)
    x_h_fp32 = normalized_hadamard_transform(x, rotate_fp32=True)
    xxt_h_fp32 = x_h_fp32 @ x_h_fp32.T
    assert torch.allclose(xxt_h_fp32, xxt, atol=0.05)


@pytest.mark.parametrize(
    "rotate_fp32",
    [True, False],
)
def test_kv_rotate(rotate_fp32):
    mtq.plugins.register_attention_for_kv_quant(SDPAAttention)
    model = nn.Sequential(SDPAAttention())
    mtq.replace_quant_module(model)

    set_quantizer_by_cfg(model, {"*": {"enable": False}})
    dummy_input = SDPAAttention.get_input(device="cuda")
    output_ref = model(dummy_input)
    if rotate_fp32:
        rotate = {"enable": True, "rotate_fp32": True}
    else:
        rotate = True
    with set_quantizer_by_cfg_context(
        model,
        {
            "*[qk]_bmm_quantizer": {
                "rotate": rotate,
            },
        },
    ):
        output_test = model(dummy_input)
    assert torch.allclose(output_ref, output_test, atol=0.05)

    # Test the rotation is actually applied by turning on only one of the query, key quantizers
    with set_quantizer_by_cfg_context(
        model,
        {
            "*k_bmm_quantizer": {
                "rotate": rotate,
            },
        },
    ):
        output_test1 = model(dummy_input)
    assert not torch.allclose(output_ref, output_test1, atol=0.05)

    mtq.unregister(SDPAAttention)
