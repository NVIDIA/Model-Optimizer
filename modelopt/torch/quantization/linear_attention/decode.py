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

"""Explicit token-state and encoded-update replay references for QAT."""

from dataclasses import dataclass

import torch

from .config import LinearAttentionDecodeConfig

__all__ = [
    "EncodedLinearAttentionTensor",
    "LinearAttentionCarry",
    "ReplayEntry",
    "recurrent_decode",
    "recurrent_decode_reference",
]


@dataclass
class EncodedLinearAttentionTensor:
    """Fake decoded values plus detached scale metadata; no compressed storage claim."""

    values: torch.Tensor
    scales: torch.Tensor | None
    format: str
    block_v: int | None


@dataclass
class ReplayEntry:
    """An already computed, encoded rank-one update and its log retention."""

    key: EncodedLinearAttentionTensor
    update: EncodedLinearAttentionTensor
    log_retention: torch.Tensor


@dataclass
class LinearAttentionCarry:
    """Explicit anchor/update state, position, and codec contract across calls."""

    anchor: EncodedLinearAttentionTensor
    entries: tuple[ReplayEntry, ...]
    position: int
    started: bool
    signature: str

    @property
    def cursor(self):
        """Number of encoded updates since the last anchor refresh."""
        return len(self.entries)

    def reconstruct(self):
        """Replay stored entries in order, preserving all anchor/update gradients."""
        state = self.anchor.values
        for entry in self.entries:
            gate = entry.log_retention
            decay = gate.exp().unsqueeze(-1)
            if gate.ndim == 1:
                decay = decay.unsqueeze(-1)
            state = state * decay + entry.key.values.unsqueeze(-1) * entry.update.values.unsqueeze(
                -2
            )
        return state


class _IdentityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, rounded):
        return rounded

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _encode(value, enabled, block_v, *, state=False, state_format="fp8_e4m3"):
    if not enabled:
        return EncodedLinearAttentionTensor(value, None, "identity", None)
    rounded, scales = [], []
    with torch.no_grad():
        for part in value.float().split(block_v, dim=-1):
            axes = (-2, -1) if state else (-1,)
            amax = part.abs().amax(dim=axes, keepdim=True)
            if state_format == "int8":
                scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
                rounded.append((part / scale).round().clamp(-127, 127) * scale)
            else:
                scale = torch.where(amax > 0, amax * (1.0 / 448.0), torch.ones_like(amax))
                rounded.append(
                    (part / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
                )
            scales.append(scale[..., 0, 0] if state else scale[..., 0])
        decoded = torch.cat(rounded, dim=-1).to(value.dtype)
        metadata = torch.stack(scales, dim=-1)
    return EncodedLinearAttentionTensor(
        _IdentityGradient.apply(value, decoded), metadata, state_format, block_v
    )


def _signature(config, state_qdq, block_v, state_format):
    return (
        config.model_dump_json(exclude={"implementation", "prefill_state_qdq"})
        + f"/{state_qdq}/{block_v}/{state_format}"
    )


def _sum_keys(value):
    keys = value.shape[-2]
    padded = 1 << (keys - 1).bit_length()
    if keys != padded:
        value = torch.nn.functional.pad(value, (0, 0, 0, padded - keys))
    while value.shape[-2] > 1:
        half = value.shape[-2] // 2
        value = value[..., :half, :] + value[..., half:, :]
    return value[..., 0, :]


def _round_log_gate(gate, step):
    if step is None:
        return gate
    rounded = (gate / step).round() * step
    return _IdentityGradient.apply(gate, rounded.detach())


def _prepare_carry(
    q, k, v, g, beta, config, state_qdq, block_v, initial_state, carry, position, state_format
):
    if q.ndim != 3 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k/v must have aligned [T,H,D] shapes")
    if beta.shape != q.shape[:2] or g.shape not in (beta.shape, q.shape):
        raise ValueError("beta must be [T,H]; g must be [T,H] or [T,H,Dk]")
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    signature = _signature(config, state_qdq, block_v, state_format)
    if carry is not None and initial_state is not None:
        raise ValueError("Supply either carry or initial_state")
    shape = (q.shape[1], q.shape[2], v.shape[2])
    if carry is None:
        initial_state = q.new_zeros(shape) if initial_state is None else initial_state
        carry = LinearAttentionCarry(
            _encode(initial_state, False, block_v, state=True, state_format=state_format),
            (),
            position,
            False,
            signature,
        )
    if carry.signature != signature or carry.anchor.values.shape != shape:
        raise ValueError("Carry policy or state shape does not match this recurrence")
    if config.mode == "token" and carry.entries:
        raise ValueError("Token carry cannot contain replay entries")
    if config.replay is not None and carry.cursor >= config.replay.window:
        raise ValueError("Replay cursor must be below the refresh window")
    if len(q) and not carry.started:
        anchor = _encode(
            carry.anchor.values,
            state_qdq and config.quantize_initial,
            block_v,
            state=True,
            state_format=state_format,
        )
        carry = LinearAttentionCarry(anchor, (), carry.position, True, signature)
    return carry, signature


def recurrent_decode_reference(
    q,
    k,
    v,
    g,
    beta,
    *,
    config: LinearAttentionDecodeConfig,
    state_qdq=False,
    state_format="fp8_e4m3",
    block_v=64,
    initial_state=None,
    carry=None,
    position=0,
    scale=None,
):
    """Run one preactivated sequence [T,H,D] with explicit token/replay write events.

    Keys and value heads must already be aligned. Scalar GDN or per-key-channel
    KDA log gates are accepted. Outputs and all returned carry values retain their
    graphs. An empty call performs no write or initial-state quantization.
    """
    carry, signature = _prepare_carry(
        q, k, v, g, beta, config, state_qdq, block_v, initial_state, carry, position, state_format
    )
    if len(q) == 0:
        # Keep empty input gradients defined without introducing a state write.
        zero = (q.sum() + k.sum() + v.sum() + g.sum() + beta.sum()) * 0
        output = v + zero
        return output, carry
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for t in range(len(q)):
            entries = carry.entries
            if config.replay is not None and config.replay.encoding == "reencode":
                entries = tuple(
                    ReplayEntry(
                        _encode(e.key.values, config.replay.factor_qdq, k.shape[-1]),
                        _encode(e.update.values, config.replay.factor_qdq, block_v),
                        e.log_retention,
                    )
                    for e in entries
                )
                carry = LinearAttentionCarry(carry.anchor, entries, carry.position, True, signature)
            state = carry.reconstruct()
            gate = _round_log_gate(g[t], config.decay_log_step)
            decay = gate.exp().unsqueeze(-1)
            if gate.ndim == 1:
                decay = decay.unsqueeze(-1)
            key = _encode(k[t], config.replay is not None and config.replay.factor_qdq, k.shape[-1])
            decayed = state * decay
            residual = v[t] - _sum_keys(key.values.unsqueeze(-1) * decayed)
            update = _encode(
                beta[t].unsqueeze(-1) * residual,
                config.replay is not None and config.replay.factor_qdq,
                block_v,
            )
            working = decayed + key.values.unsqueeze(-1) * update.values.unsqueeze(-2)
            if config.mode == "token":
                anchor = _encode(working, state_qdq, block_v, state=True, state_format=state_format)
                next_carry = LinearAttentionCarry(anchor, (), carry.position + 1, True, signature)
                stored = anchor.values
            else:
                assert config.replay is not None
                entries = (*entries, ReplayEntry(key, update, gate))
                if len(entries) == config.replay.window:
                    anchor = _encode(
                        working, state_qdq, block_v, state=True, state_format=state_format
                    )
                    next_carry = LinearAttentionCarry(
                        anchor, (), carry.position + 1, True, signature
                    )
                    stored = anchor.values
                else:
                    next_carry = LinearAttentionCarry(
                        carry.anchor, entries, carry.position + 1, True, signature
                    )
                    stored = working
            read = working if config.readout == "working" else stored
            outputs.append(_sum_keys(q[t].unsqueeze(-1) * read) * scale)
            carry = next_carry
    return torch.stack(outputs), carry


def recurrent_decode(
    q,
    k,
    v,
    g,
    beta,
    *,
    config: LinearAttentionDecodeConfig,
    state_qdq=False,
    state_format="fp8_e4m3",
    block_v=64,
    initial_state=None,
    carry=None,
    position=0,
    scale=None,
    checkpoint_interval=8,
):
    """Run the selected training implementation with explicit differentiable carry.

    The fused encode-once path carries the same reconstructed state incrementally
    and exposes the anchor and encoded updates needed for continuation gradients.
    """
    if config.implementation == "torch":
        return recurrent_decode_reference(
            q,
            k,
            v,
            g,
            beta,
            config=config,
            state_qdq=state_qdq,
            state_format=state_format,
            block_v=block_v,
            initial_state=initial_state,
            carry=carry,
            position=position,
            scale=scale,
        )
    carry, signature = _prepare_carry(
        q, k, v, g, beta, config, state_qdq, block_v, initial_state, carry, position, state_format
    )
    if len(q) == 0:
        return v + (q.sum() + k.sum() + v.sum() + g.sum() + beta.sum()) * 0, carry
    if config.replay is not None and config.replay.encoding != "once":
        raise ValueError("The fused implementation supports encode-once replay only")
    # Keep the CPU reference importable without the optional CUDA/Triton backend.
    from modelopt.torch.kernels.quantization.linear_attention.decode import fused_recurrence

    replay = config.replay is not None
    factor_qdq = config.replay is not None and config.replay.factor_qdq
    key = _encode(k, factor_qdq, k.shape[-1])
    gate = _round_log_gate(g, config.decay_log_step)
    window = config.replay.window if config.replay is not None else 1
    result = fused_recurrence(
        q,
        key.values,
        v,
        gate,
        beta,
        carry.reconstruct(),
        state_qdq=state_qdq,
        state_format=state_format,
        block_v=block_v,
        replay=replay,
        factor_qdq=factor_qdq,
        window=window,
        cursor=carry.cursor,
        read_stored=config.readout == "stored",
        scale=scale,
        checkpoint_interval=checkpoint_interval,
    )
    output, final, updates, last_anchor, anchor_scales, update_scales = result
    state_format = state_format if state_qdq else "identity"
    if not replay:
        anchor = EncodedLinearAttentionTensor(
            final,
            anchor_scales if state_qdq else None,
            state_format,
            block_v if state_qdq else None,
        )
        return output, LinearAttentionCarry(anchor, (), carry.position + len(q), True, signature)
    if carry.cursor + len(q) >= window:
        anchor = EncodedLinearAttentionTensor(
            last_anchor,
            anchor_scales if state_qdq else None,
            state_format,
            block_v if state_qdq else None,
        )
        start = ((carry.cursor + len(q)) // window) * window - carry.cursor
        entries = ()
    else:
        anchor, entries, start = carry.anchor, carry.entries, 0
    additions = []
    for t in range(start, len(q)):
        encoded_key = EncodedLinearAttentionTensor(
            key.values[t],
            key.scales[t] if key.scales is not None else None,
            key.format,
            key.block_v,
        )
        encoded_update = EncodedLinearAttentionTensor(
            updates[t],
            update_scales[t] if factor_qdq else None,
            "fp8_e4m3" if factor_qdq else "identity",
            block_v if factor_qdq else None,
        )
        additions.append(ReplayEntry(encoded_key, encoded_update, gate[t]))
    return output, LinearAttentionCarry(
        anchor, (*entries, *additions), carry.position + len(q), True, signature
    )
