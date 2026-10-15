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

"""Differentiable checkpoint and update replay for the optional INT8/Hadamard fork."""

from dataclasses import dataclass

import torch

from .utils import forward_value

__all__ = []


@dataclass
class _ReplayEntry:
    """One saved rank-one update since the last state checkpoint.

    ``key`` is [H,K] and ``update`` is the beta-scaled residual [H,V], both stored
    as BF16 by this fork. ``log_retention`` is [H] for GDN or
    [H,K] for KDA and records this token's decay for subsequent reconstruction.
    """

    key: torch.Tensor
    update: torch.Tensor
    log_retention: torch.Tensor


@dataclass
class _ReplayState:
    """One shared read/update checkpoint and pending updates within a training suffix.

    Refreshing the replay window replaces the checkpoint and clears the entries.
    Tensors retain their autograd history; this state never crosses forward calls.
    """

    anchor: torch.Tensor  # Decoded checkpoint [H,K,V] in the Hadamard value basis.
    scales: torch.Tensor | None  # Native FP16 scales needed to rebuild an INT8 cache.
    entries: tuple[_ReplayEntry, ...] = ()

    def reconstruct(self):
        """Return the dense state in the original value basis without another QDQ."""
        # The optional serving fork owns the checkpoint basis transform.
        from ...kernels.quantization.linear_attention.serving.replay import original_basis

        return original_basis(_replay_state(self))


def _hadamard32(value):
    """Apply the orthonormal Sylvester transform to contiguous 32-value groups."""
    shape = value.shape
    for width in (1, 2, 4, 8, 16):
        pairs = value.reshape(*shape[:-1], -1, 2, width)
        left, right = pairs.unbind(-2)
        value = torch.stack((left + right, left - right), dim=-2).reshape(shape)
    return value * (32**-0.5)


def _replay_state(carry: _ReplayState, current_gate=None):
    """Decay the anchor and add saved updates weighted by their subsequent gates.

    KDA weights keys per channel; GDN weights updates per head. ``current_gate``
    includes the next KDA token's decay before BF16 weighting, without adding its update.
    """
    state = carry.anchor
    if not carry.entries:
        if current_gate is not None:
            state = state * current_gate.exp().unsqueeze(-1)
        return state
    gates = torch.stack([entry.log_retention for entry in carry.entries], dim=1)
    keys = torch.stack([entry.key for entry in carry.entries], dim=1)
    updates = torch.stack([entry.update for entry in carry.entries], dim=1)
    total = gates.sum(1)
    if current_gate is not None:
        total = total + current_gate
    weights = (total.unsqueeze(1) - gates.cumsum(1)).exp()
    if gates.ndim == 3:
        weighted = keys * weights
        weighted = forward_value(weighted, weighted.to(torch.bfloat16))
        return state * total.exp().unsqueeze(-1) + weighted.transpose(-1, -2) @ updates
    weighted = updates * weights.unsqueeze(-1)
    weighted = forward_value(weighted, weighted.to(torch.bfloat16))
    return state * total.exp()[:, None, None] + keys.transpose(-1, -2) @ weighted


def _sum_keys(value):
    keys = value.shape[-2]
    padded = 1 << (keys - 1).bit_length()
    if keys != padded:
        value = torch.nn.functional.pad(value, (0, 0, 0, padded - keys))
    while value.shape[-2] > 1:
        half = value.shape[-2] // 2
        value = value[..., :half, :] + value[..., half:, :]
    return value[..., 0, :]


def _normalized(value):
    return value / (value.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def _replay_decode(q, k, v, g, beta, initial_state, window, enabled, scale, normalize, gate_inputs):
    """Attach native ReplaySSM values to the differentiable checkpoint/update recurrence."""
    # This optional fork owns checkpoint rounding and the Hadamard value basis.
    from ...kernels.quantization.linear_attention.serving.replay import checkpoint, step

    initial = _hadamard32(initial_state)
    with torch.no_grad():
        decoded, scales = checkpoint(initial_state, enabled)
    carry = _ReplayState(forward_value(initial, decoded), scales)
    original_v, v = v, _hadamard32(v)
    with torch.autocast(device_type=q.device.type, enabled=False):
        outputs, native_outputs = [], []
        for t in range(len(q)):
            gate = g[t]
            with torch.no_grad():
                native = step(
                    q[t],
                    k[t],
                    original_v[t],
                    gate,
                    beta[t],
                    carry,
                    window,
                    enabled,
                    scale,
                    normalize,
                    (gate_inputs[0][t], gate_inputs[1][t], *gate_inputs[2:])
                    if gate_inputs is not None
                    else None,
                )
            native_outputs.append(native[0])
            decay = gate.exp().unsqueeze(-1)
            if gate.ndim == 1:
                decay = decay.unsqueeze(-1)
            key = k[t]
            if normalize:
                key = _normalized(key)
            decayed = _replay_state(carry, gate) if gate.ndim == 2 else _replay_state(carry) * decay
            residual = v[t] - _sum_keys(key.unsqueeze(-1) * decayed)
            update = beta[t].unsqueeze(-1) * residual
            working = decayed + key.unsqueeze(-1) * update.unsqueeze(-2)
            if len(carry.entries) + 1 == window:
                # Native step already quantized the checkpoint; preserve its scales and adjoint.
                anchor, scales = forward_value(working, native[1]), native[2]
                entries = ()
            else:
                anchor, scales = carry.anchor, carry.scales
                entries = (
                    *carry.entries,
                    _ReplayEntry(
                        forward_value(key, native[3]),
                        forward_value(update, native[4]),
                        forward_value(gate, native[5]),
                    ),
                )
            carry = _ReplayState(anchor, scales, entries)
            query = q[t]
            if normalize:
                query = _normalized(query)
            # Read the working state before checkpoint rounding, including when the window fills.
            outputs.append(_sum_keys(query.unsqueeze(-1) * working) * scale)
        return forward_value(_hadamard32(torch.stack(outputs)), torch.stack(native_outputs)), carry
