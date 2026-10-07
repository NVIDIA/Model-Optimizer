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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal native-cache and backward checks for the optional ReplaySSM serving fork."""

import pytest
import torch

from modelopt.torch.quantization.linear_attention import LinearAttentionConfig, recurrent_decode


@pytest.fixture(scope="module", params=[False, True], ids=["gdn-token", "kda-replay"])
def compiled_replay(request):
    native = pytest.importorskip(
        "vllm.model_executor.layers.fla.ops.fused_recurrent_replayssm",
        exc_type=ModuleNotFoundError,
    )
    channel = request.param
    window = 4 if channel else 1
    torch.manual_seed(71)
    q, k = [
        torch.nn.functional.normalize(torch.randn(5, 1, 64, device="cuda"), dim=-1).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn_like(q)
    raw = torch.randn(5, 1, device="cuda", dtype=torch.bfloat16)
    raw_beta = torch.randn_like(raw)
    rate = torch.full((1,), -3.0, device="cuda")
    bias = torch.zeros_like(rate)
    g = (
        -torch.rand_like(q, dtype=torch.float32) * 0.03
        if channel
        else -rate.exp() * torch.nn.functional.softplus(raw.float())
    )
    beta = raw_beta.float().sigmoid()
    args = tuple(x.detach().requires_grad_() for x in (q, k, v, g, beta))
    initial = (torch.randn(1, 64, 64, device="cuda") * 0.1).requires_grad_()
    config = LinearAttentionConfig(
        backend="serving",
        precision="replayssm",
        replay_window=window,
    )
    kwargs = {
        "config": config,
        "state_qdq": True,
        "state_format": "int8",
        "replay_gate_inputs": None if channel else (raw, raw_beta, rate, bias),
    }

    def forward():
        return recurrent_decode(*args, initial_state=initial, **kwargs)

    out, carry = forward()
    torch.autograd.grad(out.sum() + carry.reconstruct().sum(), (*args, initial))
    torch.cuda.synchronize()
    return native, args, initial, kwargs, forward


def test_replay_matches_persistent_serving_cache(compiled_replay):
    native, args, initial, kwargs, forward = compiled_replay
    output, carry = forward()
    channel = args[3].ndim == 3
    window = kwargs["config"].replay_window
    state = torch.zeros(2, 1, 64, 64, device="cuda", dtype=torch.int8)
    scales = torch.zeros(2, 1, 2, 64, device="cuda", dtype=torch.float16)
    updates = torch.zeros(2, 1, window, 64, device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros_like(updates)
    gates = torch.zeros((2, 1, window, 64) if channel else (2, 1, window), device="cuda")
    indices = torch.ones(1, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        native.prefill_write_checkpoint(
            initial.transpose(-1, -2)[None].contiguous(),
            state,
            None,
            None,
            scales,
            None,
            None,
            indices,
            8,
            hadamard=True,
        )
        for t in range(len(args[0])):
            q, k, v, gate, beta = (x[t] for x in args)
            if channel:
                a, b, rate, bias = gate.flatten()[None], beta[None], None, None
            else:
                raw, raw_beta, rate, bias = kwargs["replay_gate_inputs"]
                a, b = raw[t : t + 1], raw_beta[t : t + 1]
            expected = torch.empty(1, 1, 64, device="cuda", dtype=torch.bfloat16)
            native.fused_recurrent_gated_delta_rule_replayssm(
                torch.cat([x.flatten() for x in (q, k, v)])[None],
                a,
                b,
                rate,
                bias,
                64**-0.5,
                state,
                updates,
                keys,
                gates,
                expected,
                indices,
                torch.tensor([t % window], device="cuda", dtype=torch.int32),
                quant_state_bits=8,
                state_scale=scales,
                hadamard_value_basis=True,
                block_v=64,
                _is_kda=channel,
            )
            torch.testing.assert_close(output[t].bfloat16(), expected[0], rtol=0, atol=0)
        decoded = state[1].transpose(-1, -2).float().unflatten(-1, (-1, 32))
        metadata = scales[1].transpose(-1, -2)
        torch.testing.assert_close(
            carry.anchor.values, (decoded * metadata[..., None]).flatten(-2), rtol=0, atol=0
        )
        torch.testing.assert_close(carry.anchor.scales, metadata, rtol=0, atol=0)
        for i, entry in enumerate(carry.entries):
            torch.testing.assert_close(
                entry.key.values.float(), keys[1, :, i].float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                entry.update.values.float(), updates[1, :, i].float(), rtol=0, atol=0
            )
    grads = torch.autograd.grad(
        output.square().mean() + carry.reconstruct().square().mean(), (*args, initial)
    )
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
