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

"""Differentiable adapter for pinned serving arithmetic.

The kernels supply forward values. Autograd differentiates the corresponding
operations evaluated at those values; operand casts and QDQ use identity STE.
The optional vLLM dependency supplies kernels; training owns its state and needs no server.
"""

import torch

from .utils import forward_value


def rounded(value):
    return forward_value(value, value.to(torch.bfloat16))


def normalized(value):
    return value / (value.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def prefix(q, k, v, g, beta, state, scale, beta_dtype, normalize=False):
    # Keep the optional vLLM dependency isolated to the native precision profile.
    from ...kernels.quantization.linear_attention.serving.forward import prefill

    with torch.no_grad():
        out, final, saved = prefill(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            g.float(),
            beta.to(beta_dtype),
            state.float(),
            scale,
            normalize,
        )
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in (q, k, v, g, beta, state)):
        return out.float(), final
    q, k = [
        forward_value(normalized(x) if normalize else x, saved[n]) for n, x in (("q", q), ("k", k))
    ]
    outputs = []
    channel = g.ndim == 3
    for chunk, lo in enumerate(range(0, len(q), 64)):
        hi = min(lo + 64, len(q))
        qc, kc, vc = [x[lo:hi].transpose(0, 1) for x in (q, k, v)]
        bc = beta[lo:hi].transpose(0, 1).unsqueeze(-1)
        gc = forward_value(g[lo:hi].transpose(0, 1).cumsum(1), saved["g"][lo:hi].transpose(0, 1))
        state = forward_value(state, saved["h"][chunk])
        hs = rounded(state)
        count = hi - lo

        def matrix(name, value):
            return forward_value(value, saved[name][lo:hi].transpose(0, 1)[..., :count])

        if channel:
            lower_rows, score_rows = [], []
            for row in range(count):
                right = (
                    kc[:, : row + 1] * (gc[:, row : row + 1] - gc[:, : row + 1]).exp()
                ).transpose(-1, -2)
                lower_rows.append(
                    torch.nn.functional.pad(
                        (bc[:, row : row + 1] * kc[:, row : row + 1]) @ right, (0, count - row - 1)
                    )
                )
                score_rows.append(
                    torch.nn.functional.pad(
                        (qc[:, row : row + 1] * scale) @ right, (0, count - row - 1)
                    )
                )
            lower = matrix("lower", torch.cat(lower_rows, dim=1).tril(-1))
            scores = matrix("scores", torch.cat(score_rows, dim=1))
            gate = gc.exp()
        else:
            causal = torch.ones(count, count, device=q.device, dtype=torch.bool).tril()
            decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, 0).exp()
            lower = matrix("lower", (bc * (kc @ kc.transpose(-1, -2)) * decay).tril(-1))
            scores = ((qc @ kc.transpose(-1, -2)) * decay).tril()
            gate = gc.exp().unsqueeze(-1)
        eye = torch.eye(count, device=q.device, dtype=q.dtype).expand_as(lower)
        inverse = matrix(
            "inverse",
            torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True),
        )
        u = forward_value(inverse @ rounded(bc * vc), saved["u"][lo:hi].transpose(0, 1))
        kb = rounded(bc * kc) if beta_dtype == torch.bfloat16 else bc * kc
        w = forward_value(inverse @ rounded(kb * gate), saved["w"][lo:hi].transpose(0, 1))
        updated = forward_value(u - w @ hs, saved["updated"][lo:hi].transpose(0, 1))
        if channel:
            output = rounded(rounded(qc * scale) * gate) @ hs + rounded(scores) @ rounded(updated)
            kg = forward_value(kc * (gc[:, -1:] - gc).exp(), saved["kg"][lo:hi].transpose(0, 1))
            state = state * gate[:, -1, :, None] + kg.transpose(-1, -2) @ rounded(updated)
        else:
            output = ((qc @ hs) * gate + rounded(scores) @ rounded(updated)) * scale
            weighted = rounded(updated * (gc[:, -1:] - gc).exp().unsqueeze(-1))
            state = state * gate[:, -1, :, None] + kc.transpose(-1, -2) @ weighted
        outputs.append(forward_value(output.transpose(0, 1), out[lo:hi]))
    return torch.cat(outputs), forward_value(state, final)


def step(q, k, v, gate, beta, state, scale, normalize=False):
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
