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

"""GDN replay with an FP32 recurrence anchor and an independently rounded FP8 read copy."""

import torch
from torch.autograd.function import once_differentiable

from .utils import forward_value, validate_gdn_quantizer

__all__ = []


def validate_two_precision(state_quantizer, read_quantizer):
    """Keep checkpoint precision separate from read precision and its enable switch."""
    if state_quantizer is not None and state_quantizer.is_enabled:
        raise ValueError("Two-precision GDN requires an FP32 anchor; disable gdn_state_quantizer")
    if read_quantizer is not None and read_quantizer.is_enabled:
        validate_gdn_quantizer(read_quantizer, name="gdn_read_quantizer", block_sizes=(128,))
        if read_quantizer.block_sizes != {-1: 128}:
            raise ValueError("gdn_read_quantizer requires FP8 block128 along the key dimension")
        return read_quantizer._if_quant
    return False


def _native_replay(state, enabled):
    # Optional standalone backend; installing public vLLM does not install this package.
    from emulation import GDNReplay

    return GDNReplay(
        state.transpose(-1, -2).unsqueeze(0).contiguous(),
        state.shape[0],
        mode="tp8_hi32" if enabled else "s32_rs",
    )


class _TwoPrecision(torch.autograd.Function):
    """Execute one native suffix; rematerialize at most one 16-token window in backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, state, gate_inputs, enabled):
        runner = _native_replay(state, enabled)
        raw_g, raw_beta, rate, bias = gate_inputs
        operands = torch.cat([x.flatten(1) for x in (q, k, v)], -1).to(torch.bfloat16)
        raw_g, raw_beta = (x.bfloat16().float().contiguous() for x in (raw_g, raw_beta))
        rate, bias = (x.reshape(-1).float().contiguous() for x in (rate, bias))
        outputs, anchors, reads = [], [], []
        for t in range(len(q)):
            if t % 16 == 0:
                anchors.append(runner.state_hi[0].clone())
            outputs.append(
                runner.step(operands[t : t + 1], raw_g[t : t + 1], raw_beta[t : t + 1], rate, bias)[
                    0
                ].clone()
            )
            if t % 16 == 0:
                reads.append((runner.state_lo if enabled else runner.state_hi)[0].clone())
        ctx.save_for_backward(q, k, v, g, beta, torch.stack(anchors), torch.stack(reads))
        return torch.stack(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, g, beta, anchors, reads = ctx.saved_tensors
        inputs = (q, k, v, g, beta)
        gradients = [torch.empty_like(x) for x in inputs]
        grad_anchor = torch.zeros_like(anchors[0])
        # Checkpoint values are immutable snapshots; no live native cache survives forward.
        with torch.enable_grad(), torch.autocast(q.device.type, enabled=False):
            for window in reversed(range(len(anchors))):
                start, end = window * 16, min((window + 1) * 16, len(q))
                xs = [x[start:end].detach().requires_grad_() for x in inputs]
                anchor = anchors[window].detach().requires_grad_()
                read = forward_value(anchor, reads[window])
                output, final = _window_forward(
                    xs[0], xs[1], xs[2], xs[3], xs[4], anchor, read, seed=window == 0
                )
                grads = torch.autograd.grad(
                    (output, final), (*xs, anchor), (grad_output[start:end], grad_anchor)
                )
                for dest, value in zip(gradients, grads[:-1]):
                    dest[start:end] = value
                grad_anchor = grads[-1]
        return (*gradients, grad_anchor.transpose(-1, -2), None, None)


def two_precision_decode(q, k, v, g, beta, state, gate_inputs, enabled):
    """Return native GDN output with an identity-STE window adjoint, without a server."""
    return _TwoPrecision.apply(q, k, v, g, beta, state, gate_inputs, enabled)


def _rounded(x, dtype=torch.bfloat16):
    return forward_value(x, x.to(dtype))


def _window_forward(q, k, v, g, beta, anchor, read, *, seed):
    """Differentiate checkpoint reads and the raw-input triangular solve at a flush.

    States are value-first [H,V,K]. Cached updates affect reads only; the flush
    reconstructs residuals from raw inputs and the recurrence anchor. This graph
    is built during backward, never alongside the native forward.
    """
    queries = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * q.shape[-1] ** -0.5
    keys = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    cumulative = g.cumsum(0)
    beta = _rounded(beta)
    updates, outputs = [], []
    for t in range(len(q)):
        checkpoint = anchor if (seed and t == 0) or t == 15 else read
        checkpoint = _rounded(checkpoint)
        # The native tensor-core contraction rounds the checkpoint but normalizes
        # each raw BF16 query/key after its contraction.
        cq = (checkpoint * q[t].unsqueeze(-2)).sum(-1)
        cq = cq / (q[t].square().sum(-1) + 1e-6).sqrt().unsqueeze(-1) * q.shape[-1] ** -0.5
        weights = (cumulative[t] - cumulative[: t + 1]).exp()
        kq = (keys[: t + 1] * queries[t]).sum(-1)
        if t == 15:
            # Rebuild the 16 residuals independently of the read-side update ring.
            key_matrix = _rounded(keys).transpose(0, 1)
            gram = key_matrix @ key_matrix.transpose(-1, -2)
            projected = (_rounded(anchor) @ key_matrix.transpose(-1, -2)).permute(2, 0, 1)
            residual = _rounded(v - cumulative.exp().unsqueeze(-1) * projected)
            solved = []
            for j in range(16):
                previous = 0
                if j:
                    factors = (cumulative[j] - cumulative[:j]).exp() * gram[:, j, :j].T
                    previous = (torch.stack(solved) * factors.unsqueeze(-1)).sum(0)
                solved.append(beta[j].unsqueeze(-1) * (residual[j] - previous))
            solved = torch.stack(solved)
            output = cumulative[t].exp().unsqueeze(-1) * cq
            output = output + (solved * (weights * kq).unsqueeze(-1)).sum(0)
            weighted = _rounded(solved * weights.unsqueeze(-1)).permute(1, 2, 0)
            final = cumulative[t].exp()[:, None, None] * anchor + weighted @ key_matrix
        else:
            ck = (checkpoint * k[t].unsqueeze(-2)).sum(-1)
            ck = ck / (k[t].square().sum(-1) + 1e-6).sqrt().unsqueeze(-1)
            sk = cumulative[t].exp().unsqueeze(-1) * ck
            output = cumulative[t].exp().unsqueeze(-1) * cq
            if t:
                cached = torch.stack(updates)
                kk = (keys[:t] * keys[t]).sum(-1)
                sk = sk + (cached * (weights[:t] * kk).unsqueeze(-1)).sum(0)
                output = output + (cached * (weights[:t] * kq[:t]).unsqueeze(-1)).sum(0)
            update = beta[t].unsqueeze(-1) * (v[t] - sk)
            output = output + update * kq[t].unsqueeze(-1)
            updates.append(_rounded(update, torch.float16))
            final = anchor
        outputs.append(output)
    return torch.stack(outputs), final
