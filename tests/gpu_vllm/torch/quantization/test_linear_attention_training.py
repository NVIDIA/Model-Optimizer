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

from functools import partial

import pytest
import torch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    gdn_state_qat,
    kda_state_qat,
)
from modelopt.torch.quantization.nn import TensorQuantizer

vllm = pytest.importorskip("vllm")

from modelopt.torch.kernels.quantization.linear_attention.serving._compat import fla_module


@torch.no_grad()
def _native_forward(args, quantizer, kda):
    """Execute native prefill/decode directly, with QDQ in ModelOpt's [K,V] basis."""
    prefill = fla_module("kda").chunk_kda if kda else fla_module("chunk").chunk_gated_delta_rule
    step = (
        fla_module("kda").fused_recurrent_kda
        if kda
        else fla_module("fused_recurrent").fused_recurrent_gated_delta_rule
    )
    value_first = vllm.__version_tuple__[:2] >= (0, 16)

    def layout(state):
        return state.transpose(-1, -2).contiguous() if value_first else state

    state = layout(args[0].new_zeros(1, 1, args[0].shape[-1], args[2].shape[-1]).float())
    prefix, state = prefill(
        *[x[:, :65].detach().clone() for x in args],
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, 65], device="cuda", dtype=torch.int32),
    )
    outputs = [prefix]
    state = layout(quantizer(layout(state)[0])[None])
    for token in range(65, 73):
        output, state = step(
            *[x[:, token : token + 1].detach().clone() for x in args],
            initial_state=state,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        )
        outputs.append(output)
        state = layout(quantizer(layout(state)[0])[None])
    return torch.cat(outputs, dim=1), layout(state)


@pytest.fixture(scope="module", params=[False, True])
def compiled_serving_case(request):
    """Compile one shared BF16 shape per model, outside the test-call timer."""
    kda = request.param
    torch.manual_seed(73)
    # A rectangular GDN state detects accidental key/value-axis swaps.
    args = [
        torch.randn(1, 73, 1, dim, device="cuda", dtype=torch.bfloat16)
        for dim in (32, 32, 32 if kda else 64)
    ]
    args += [
        -torch.rand((1, 73, 1, 32) if kda else (1, 73, 1), device="cuda") * 0.03,
        torch.rand(1, 73, 1, device="cuda") * 0.4,
    ]
    args = [x.requires_grad_() for x in args]
    policy = LinearAttentionConfig(backend="serving", precision="vllm")
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8,
            type="dynamic",
            block_sizes={-1: 32},
            narrow_range=True,
            pass_through_bwd=True,
        )
    ).cuda()
    forward = partial(
        kda_state_qat if kda else gdn_state_qat,
        *args,
        policy=policy,
        state_quantizer=quantizer,
        prefill_lengths=[65],
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )
    output, state = forward()
    torch.autograd.grad(output.float().sum() + state.sum(), args)
    expected = _native_forward(args, quantizer, kda)
    torch.cuda.synchronize()
    return args, quantizer, forward, expected


def test_serving_state_qdq_and_handoff_gradient(compiled_serving_case):
    args, quantizer, forward, (expected, expected_state) = compiled_serving_case
    calls = []
    handle = quantizer.register_forward_hook(lambda *_: calls.append(True))
    output, state = forward()
    handle.remove()
    # One handoff QDQ, then one QDQ per suffix update; none inside the fresh prefill.
    assert len(calls) == 9
    with torch.no_grad():
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        quantizer.disable()
        plain, _ = forward()
        assert not torch.equal(output[:, 65:], plain[:, 65:])
    gradients = torch.autograd.grad(
        output[:, 65:].float().square().sum() + state.square().sum(), args
    )
    assert all(torch.isfinite(x).all() for x in gradients)
    assert gradients[1][:, :65].abs().sum() > 0
