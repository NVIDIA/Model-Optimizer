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

"""Differentiable adapter for the installed serving runtime's arithmetic.

Native kernels supply forward values. Prefill uses FLA's recurrence backward;
decode uses a Torch adjoint. QDQ uses identity STE.
The optional vLLM dependency supplies kernels; training owns its state and needs no server.
"""

import torch

from .utils import forward_value


def normalized(value):
    return value / (value.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def prefix(
    q, k, v, g, beta, state, scale, beta_dtype, normalize=False, gate_inputs=None, precision="vllm"
):
    # vLLM supplies the forward; the optional FLA dependency supplies the recurrence adjoint.
    from ...kernels.quantization.linear_attention.serving.forward import prefill

    with torch.no_grad():
        out, final = prefill(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            g.float(),
            beta.to(beta_dtype),
            state.float(),
            scale,
            normalize,
            gate_inputs,
            kimi_k3=precision == "vllm_kimi_k3",
        )
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in (q, k, v, g, beta, state)):
        return out.float(), final
    # FLA differentiates the same recurrence. Its intermediate rounding is a backward
    # surrogate, not a claim that vLLM inference kernels have an exact backward.
    if g.ndim == 3:
        from fla.ops.kda import chunk_kda as chunk
    else:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk
    adjoint, adjoint_state = chunk(
        *[x.to(torch.bfloat16)[None] for x in (q, k, v)],
        g[None],
        beta[None],
        initial_state=state[None],
        output_final_state=True,
        scale=scale,
        use_qk_l2norm_in_kernel=normalize,
    )
    return forward_value(adjoint[0].float(), out), forward_value(adjoint_state[0], final)


def step(q, k, v, gate, beta, state, scale, normalize=False, gate_inputs=None, precision="vllm"):
    # Keep the optional vLLM dependency isolated to the native precision profile.
    from ...kernels.quantization.linear_attention.serving.forward import step as native_step

    with torch.no_grad():
        out, final = native_step(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            gate.float(),
            beta.float(),
            state.float(),
            scale,
            normalize,
            gate_inputs,
            kimi_k3=precision == "vllm_kimi_k3",
        )
    if not torch.is_grad_enabled() or not any(
        x.requires_grad for x in (q, k, v, gate, beta, state)
    ):
        return out.float(), final
    if normalize:
        q, k = normalized(q), normalized(k)
    decay = gate.exp().unsqueeze(-1)
    if gate.ndim == 1:
        decay = decay.unsqueeze(-1)
    decayed = state * decay
    residual = v - (decayed * k.unsqueeze(-1)).sum(-2)
    update = residual * beta.unsqueeze(-1)
    working = forward_value(decayed + k.unsqueeze(-1) * update.unsqueeze(-2), final)
    output = ((q * scale).unsqueeze(-1) * working).sum(-2)
    return forward_value(output, out), working
