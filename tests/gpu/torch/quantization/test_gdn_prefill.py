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

"""GDN prefill operand-QDQ parity against an independent token recurrence."""

import pytest
import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention.matmul import (
    GDNMatmulSites,
    _apply_operand_quantizer,
)
from modelopt.torch.quantization.linear_attention.prefill import gdn_prefill_qat
from modelopt.torch.quantization.nn import TensorQuantizer


def _recurrence(q, k, v, g, beta, initial, scale=None):
    """Dense token oracle; it does not use the production chunk/WY equations."""
    dtype = q.dtype
    q, k, v, g, beta, state = (value.float() for value in (q, k, v, g, beta, initial))
    heads = v.shape[1]
    q, k = (value.repeat_interleave(heads // value.shape[1], dim=1) for value in (q, k))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs = []
    for token in range(len(q)):
        state = state * g[token].exp()[:, None, None]
        residual = v[token] - torch.einsum("hk,hkv->hv", k[token], state)
        state = state + torch.einsum("hk,hv->hkv", k[token], beta[token, :, None] * residual)
        outputs.append(torch.einsum("hk,hkv->hv", q[token] * scale, state))
    return torch.stack(outputs).to(dtype), state


def _inputs(length=65, *, requires_grad=False):
    torch.manual_seed(7201 + length)
    q = torch.randn(length, 1, 32, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(length, 2, 48, device="cuda", dtype=torch.bfloat16) * 0.2
    g = -(torch.rand(length, 2, device="cuda") * 0.04 + 0.01)
    beta = torch.rand(length, 2, device="cuda") * 0.5
    state = torch.randn(2, 32, 48, device="cuda") * 0.02
    if requires_grad:
        for value in (q, k, v, g, beta, state):
            value.requires_grad_()
    return q, k, v, g, beta, state


def test_disabled_sites_match_independent_recurrence_and_backward():
    inputs = _inputs(requires_grad=True)
    q, k, v, g, beta, state = inputs
    output, final = gdn_prefill_qat(
        q[None],
        k[None],
        v[None],
        g[None],
        beta[None],
        sites=GDNMatmulSites().cuda(),
        initial_state=state[None],
        output_final_state=True,
    )
    expected_output, expected_final = _recurrence(q, k, v, g, beta, state)
    torch.testing.assert_close(output[0], expected_output, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(final[0], expected_final, rtol=1e-5, atol=1e-5)

    probe_output = torch.randn_like(output)
    probe_final = torch.randn_like(final)
    actual_grad = torch.autograd.grad(
        (output * probe_output).sum() + (final * probe_final).sum(), inputs
    )
    expected_grad = torch.autograd.grad(
        (expected_output * probe_output[0]).sum() + (expected_final * probe_final[0]).sum(),
        inputs,
    )
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)


def test_packed_sequences_match_independent_calls():
    first = _inputs(33)
    second = _inputs(47)
    q, k, v, g, beta = (torch.cat((first[index], second[index]))[None] for index in range(5))
    initial = torch.stack((first[-1], second[-1]))
    sites = GDNMatmulSites().cuda()
    packed_output, packed_final = gdn_prefill_qat(
        q,
        k,
        v,
        g,
        beta,
        sites=sites,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens_cpu=(0, 33, 80),
    )
    pieces, finals = [], []
    for values in (first, second):
        output, final = gdn_prefill_qat(
            *(value[None] for value in values[:5]),
            sites=sites,
            initial_state=values[-1][None],
            output_final_state=True,
        )
        pieces.append(output[0])
        finals.append(final[0])
    torch.testing.assert_close(packed_output[0], torch.cat(pieces), rtol=0, atol=0)
    torch.testing.assert_close(packed_final, torch.stack(finals), rtol=0, atol=0)


def test_nvfp4_operand_scale_is_isolated_per_sequence_and_head():
    torch.manual_seed(993)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            type="dynamic",
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            pass_through_bwd=True,
        )
    ).cuda()
    small = torch.randn(1, 1, 8, 32, device="cuda")
    expected = _apply_operand_quantizer(quantizer, small)
    for dimension in (0, 1):
        mixed = torch.cat((small, small * 2**20), dim=dimension)
        selection = (slice(0, 1), slice(0, 1))
        torch.testing.assert_close(
            _apply_operand_quantizer(quantizer, mixed)[selection], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("format_name", ["fp8", "nvfp4"])
def test_enabled_operand_qdq_runs_forward_and_backward(format_name):
    config = {
        "num_bits": (4, 3),
        "type": "dynamic",
        "axis": (0, 1, 2),
        "pass_through_bwd": True,
    }
    if format_name == "nvfp4":
        config = {
            "num_bits": (2, 1),
            "type": "dynamic",
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "pass_through_bwd": True,
        }
    sites = GDNMatmulSites().cuda()
    for site in sites.values():
        for quantizer in site.children():
            quantizer.set_from_attribute_config(config)
    inputs = _inputs(17, requires_grad=True)
    output, final = gdn_prefill_qat(
        *(value[None] for value in inputs[:5]),
        sites=sites,
        initial_state=inputs[-1][None],
        output_final_state=True,
    )
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    gradients = torch.autograd.grad(output.float().square().mean() + final.square().mean(), inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_strong_forgetting_does_not_overflow_masked_decay():
    q, k, v, g, beta, state = _inputs(64, requires_grad=True)
    g = torch.full_like(g, -100, requires_grad=True)
    inputs = (q, k, v, g, beta, state)
    output, final = gdn_prefill_qat(
        *(value[None] for value in inputs[:5]),
        sites=GDNMatmulSites().cuda(),
        initial_state=state[None],
        output_final_state=True,
    )
    gradients = torch.autograd.grad(output.float().square().mean() + final.square().mean(), inputs)
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
