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

import pytest
import torch
from _test_utils.torch.linear_attention import FP8, inputs, reference_matmul, values_and_gradients

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    LinearAttentionMatmulSites,
    LinearAttentionSolveConfig,
    chunk_gdn_reference,
    chunk_kda_reference,
    matmul_gdn,
    matmul_kda,
    neumann_inverse_reference,
    triangular_inverse,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.fixture(autouse=True)
def full_precision_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 7, 15, 31, 63])
def test_triton_polynomial_and_actual_gradient(degree):
    torch.manual_seed(82)
    lower = (torch.randn(2, 3, 64, 128, device="cuda") * 0.05)[..., :64].requires_grad_()
    policy = LinearAttentionSolveConfig(method="neumann", degree=degree, implementation="triton")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = triangular_inverse(lower, policy)
    expected = neumann_inverse_reference(lower, degree)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    probe = torch.randn_like(actual)
    gradients = [torch.autograd.grad((a * probe).sum(), lower)[0] for a in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=3e-4, atol=4e-6)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("degree", [3, 7])
@pytest.mark.parametrize("qdq", [False, True])
def test_prefill_polynomial_output_state_and_gradient(kda, degree, qdq):
    args, state = inputs("cuda", packed=True)
    if kda:
        args[3] = (
            -torch.rand(*args[3].shape, args[0].shape[-1], device="cuda") * 0.05
        ).requires_grad_()
    function = matmul_kda if kda else matmul_gdn
    reference = chunk_kda_reference if kda else chunk_gdn_reference
    policy = LinearAttentionConfig(
        backend="matmul", solve={"method": "neumann", "degree": degree, "implementation": "triton"}
    )
    kwargs = {"initial_state": state, "cu_seqlens": torch.tensor([0, 5, 73], device="cuda")}
    sites = LinearAttentionMatmulSites()
    handles = ["wy_value.lhs_quantizer"] if qdq else []
    for handle in handles:
        sites.get_submodule(handle).set_from_attribute_config(FP8)
        sites.get_submodule(handle).enable()
    actual = function(
        *args,
        sites=sites,
        policy=policy,
        w_quantizer=TensorQuantizer(QuantizerAttributeConfig(enable=False)),
        output_final_state=True,
        **kwargs,
    )
    expected = reference(
        *args,
        inverse_fn=lambda lower: neumann_inverse_reference(lower, degree),
        matmul=reference_matmul(handles, policy),
        **kwargs,
    )
    for a, e in zip(
        values_and_gradients(actual, args, state), values_and_gradients(expected, args, state)
    ):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=5e-6)
