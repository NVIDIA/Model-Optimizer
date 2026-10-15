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
from _test_utils.torch.quantization.linear_attention_reference import recurrent_delta_rule_reference

from modelopt.torch.quantization.linear_attention._vllm_autograd import prefix, step
from modelopt.torch.quantization.linear_attention.utils import forward_value

vllm = pytest.importorskip("vllm")
if vllm.__version__ != "0.30.0":
    pytest.skip("State QAT profiles target vLLM 0.30.0", allow_module_level=True)


@pytest.fixture(scope="module", params=[(False, True), (True, True), (False, False)])
def compiled_decode(request):
    """Compile the existing one-head decode shape outside the test timer."""
    channel, normalize = request.param
    torch.manual_seed(97)
    q, k, v = [torch.randn(1, 32, device="cuda").bfloat16().float() for _ in range(3)]
    raw_gate = torch.randn((1, 32) if channel else (1,), device="cuda").bfloat16().float()
    raw_beta = torch.randn(1, device="cuda").bfloat16().float()
    rate, bias = torch.full((1,), -3.0, device="cuda"), torch.zeros_like(raw_gate)
    initial = torch.randn(1, 32, 32, device="cuda")
    inputs = [x.requires_grad_() for x in (q, k, v, raw_gate, raw_beta, rate, bias, initial)]
    gate = -rate.exp() * torch.nn.functional.softplus(raw_gate + bias)
    operands = q, k, v, gate, raw_beta.sigmoid(), initial
    kwargs = {"scale": 0.3, "normalize": normalize, "gate_inputs": inputs[3:7]}
    with torch.no_grad():
        native = step(*operands, **kwargs)
    torch.cuda.synchronize()
    return inputs, operands, kwargs, native


@pytest.mark.parametrize("loss_on", ["both", "output", "state"])
def test_decode_backward_matches_recurrence(compiled_decode, loss_on):
    inputs, operands, kwargs, native = compiled_decode
    output, final = step(*operands, **kwargs)
    for actual, expected in zip((output, final), native):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    q, k, v, gate, beta, initial = operands
    if kwargs["normalize"]:
        q, k = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k)]
    _, reference_state = recurrent_delta_rule_reference(
        *[x[None, None] for x in (q, k, v, gate, beta)],
        initial_state=initial[None],
        scale=kwargs["scale"],
    )
    # The former Torch adjoint evaluated the readout at the native working state.
    working = forward_value(reference_state[0], native[1])
    reference_output = torch.einsum("hk,hkv->hv", q * kwargs["scale"], working)
    probes = [torch.randn_like(x) for x in (output, final)]
    active = (loss_on != "state", loss_on != "output")
    losses = [
        sum((value * probe).sum() for value, probe, use in zip(values, probes, active) if use)
        for values in ((output, final), (reference_output, working))
    ]
    gradients = [
        torch.autograd.grad(loss, inputs, allow_unused=True, retain_graph=True) for loss in losses
    ]
    for actual, expected in zip(*gradients):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.fixture(scope="module", params=[False, True])
def compiled_prefix(request):
    """Reuse the integration tests' 65-token prefix shape for both recurrence types."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda

    channel = request.param
    torch.manual_seed(98)
    inputs = [torch.randn(65, 1, 32, device="cuda").bfloat16().float() for _ in range(3)]
    inputs += [
        torch.randn((65, 1, 32) if channel else (65, 1), device="cuda").bfloat16().float(),
        torch.randn(65, 1, device="cuda").bfloat16().float(),
        torch.full((1,), -3.0, device="cuda"),
        torch.zeros(32 if channel else 1, device="cuda"),
        torch.randn(1, 32, 32, device="cuda"),
    ]
    inputs = [x.requires_grad_() for x in inputs]
    q, k, v, raw_gate, raw_beta, rate, bias, state = inputs
    gate = -rate.exp() * torch.nn.functional.softplus(raw_gate + bias)
    beta = raw_beta.sigmoid()
    kwargs = {
        "scale": 32**-0.5,
        "beta_dtype": torch.float32,
        "normalize": True,
        "gate_inputs": (raw_gate, raw_beta, rate, bias),
    }
    operands = q, k, v, gate, beta, state
    actual = prefix(*operands, **kwargs)
    torch.autograd.grad(sum(x.sum() for x in actual), inputs, retain_graph=True)
    chunk = chunk_kda if channel else chunk_gated_delta_rule
    expected = chunk(
        *[x[None].bfloat16() for x in (q, k, v)],
        gate[None],
        beta[None],
        initial_state=state[None],
        output_final_state=True,
        scale=kwargs["scale"],
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return inputs, operands, kwargs, tuple(x[0].float() for x in expected)


@pytest.mark.parametrize("loss_on", ["both", "output", "state"])
def test_prefix_backward_matches_fla(compiled_prefix, loss_on):
    inputs, operands, kwargs, expected = compiled_prefix
    actual = prefix(*operands, **kwargs)
    with torch.no_grad():
        native = prefix(*operands, **kwargs)
    for value, reference in zip(actual, native):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    probes = [torch.randn_like(x) for x in actual]
    active = (loss_on != "state", loss_on != "output")
    gradients = [
        torch.autograd.grad(
            sum((x * p).sum() for x, p, use in zip(values, probes, active) if use),
            inputs,
            retain_graph=True,
        )
        for values in (actual, expected)
    ]
    for value, reference in zip(*gradients):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=2e-6)
