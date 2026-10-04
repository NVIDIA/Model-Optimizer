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

"""Minimal forward/backward checks for GDN training fake quantization."""

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.quantization.linear_attention_reference import chunk_gdn_reference

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

pytest.importorskip("fla.ops.gated_delta_rule")

from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
    chunk_gated_delta_rule,
)


def make_inputs():
    torch.manual_seed(123)
    # Two chunks exercise recurrence with one shared BF16 shape for every mode.
    shape = (1, 128, 1, 32)
    q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1) for _ in range(2)]
    v = torch.randn(shape, device="cuda")
    g = -torch.rand(shape[:3], device="cuda") * 0.1
    beta = torch.rand_like(g)
    args = [x.to(torch.bfloat16).requires_grad_() for x in (q, k, v, g, beta)]
    state = (torch.randn(1, 1, 32, 32, device="cuda") * 0.1).requires_grad_()
    return args, state


def values_and_grads(fn, args, state, **kwargs):
    result = fn(*args, initial_state=state, **kwargs)
    torch.manual_seed(15)
    probes = [torch.randn(x.shape, device=x.device, dtype=torch.float32) for x in result]
    grads = torch.autograd.grad(sum((x * p).sum() for x, p in zip(result, probes)), (*args, state))
    return result, grads


def compare(actual, expected, tolerance):
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        error = (a.float() - e.float()).norm()
        bound = tolerance * e.float().norm().clamp_min(1e-6)
        assert error <= bound, (
            f"relative L2 error {(error / e.float().norm()).item():.5g} > {tolerance}"
        )


@pytest.fixture(
    scope="module", params=["disabled", "w", "state-w", "state-int8", "state-int8-block"]
)
def compiled_gdn_case(request):
    """Compile only the selected BF16 forward/backward path, outside the test-call timer."""
    state_qdq = {"state-w": 1, "state-int8": 2}.get(request.param, 0)
    if state_qdq == 1 and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("State QDQ needs native E4M3 conversion (SM89+)")
    quantizer = (
        TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=(0, 1, 2), type="dynamic"))
        if request.param in ("w", "state-w")
        else None
    )
    args, state = make_inputs()
    kwargs = {"state_qdq": state_qdq, "w_quantizer": quantizer}
    if request.param == "state-int8-block":
        kwargs["state_quantizer"] = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=8, type="dynamic", block_sizes={-1: 16}, narrow_range=True
            )
        )
        # Compile the reference quantizer's CUDA extension outside the test-call timer too.
        kwargs["state_quantizer"](state[0])
    values_and_grads(chunk_gated_delta_rule, args, state, output_final_state=True, **kwargs)
    torch.cuda.synchronize()
    return args, state, kwargs


def test_gdn_forward_and_backward(compiled_gdn_case):
    args, state, kwargs = compiled_gdn_case
    reference_args = [x.detach().float().requires_grad_() for x in args]
    reference_state = state.detach().clone().requires_grad_()
    expected = values_and_grads(
        chunk_gdn_reference,
        reference_args,
        reference_state,
        state_format="int8" if kwargs["state_qdq"] == 2 else "fp8_e4m3",
        **kwargs,
    )
    actual = values_and_grads(
        chunk_gated_delta_rule, args, state, output_final_state=True, **kwargs
    )
    compare(actual[0], expected[0], 0.03)
    compare(actual[1], expected[1], 0.05)
