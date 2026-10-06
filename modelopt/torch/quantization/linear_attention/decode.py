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

"""Differentiable token-state and encoded-update replay for QAT."""

from dataclasses import dataclass

import torch

from .config import LinearAttentionDecodeConfig
from .utils import _resolve_state_quantizer, _tile_qdq, forward_value

__all__ = [
    "EncodedLinearAttentionTensor",
    "LinearAttentionCarry",
    "ReplayEntry",
    "recurrent_decode",
]


@dataclass
class EncodedLinearAttentionTensor:
    """Fake decoded values plus optional scale metadata; no compressed storage claim."""

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
    value_basis: str = "identity"

    @property
    def cursor(self):
        """Number of encoded updates since the last anchor refresh."""
        return len(self.entries)

    def reconstruct(self, *, original_basis=True):
        """Replay entries, returning the original value basis unless explicitly disabled."""
        state = self.anchor.values
        for entry in self.entries:
            gate = entry.log_retention
            decay = gate.exp().unsqueeze(-1)
            if gate.ndim == 1:
                decay = decay.unsqueeze(-1)
            state = state * decay + entry.key.values.unsqueeze(-1) * entry.update.values.unsqueeze(
                -2
            )
        if original_basis and self.value_basis == "hadamard32":
            state = _hadamard32(state)
        return state


def _hadamard32(value):
    """Apply the orthonormal Sylvester transform to contiguous 32-value groups."""
    shape = value.shape
    for width in (1, 2, 4, 8, 16):
        pairs = value.reshape(*shape[:-1], -1, 2, width)
        left, right = pairs.unbind(-2)
        value = torch.stack((left + right, left - right), dim=-2).reshape(shape)
    return value * (32**-0.5)


def _encode(
    value,
    enabled,
    block_v,
    *,
    state=False,
    state_format="fp8_e4m3",
    state_codec="tile",
    state_quantizer=None,
):
    if not enabled:
        return EncodedLinearAttentionTensor(value, None, "identity", None)
    if state and state_quantizer is not None and state_quantizer.block_sizes is not None:
        # TensorQuantizer owns dynamic scales; the floating carry needs only its QDQ output.
        return EncodedLinearAttentionTensor(
            state_quantizer(value), None, state_format, state_quantizer.block_sizes[-1]
        )
    if state and state_codec == "int8_hadamard32":
        with torch.no_grad():
            groups = value.float().reshape(*value.shape[:-1], -1, 32)
            scales = (groups.abs().amax(-1) / 127).clamp_min(6e-8)
            normalized = groups / scales.unsqueeze(-1)
            codes = normalized.sign() * (normalized.abs() + 0.5).floor()
            metadata = scales.to(torch.float16)
            decoded = (codes.clamp(-127, 127) * metadata.float().unsqueeze(-1)).reshape_as(value)
        return EncodedLinearAttentionTensor(forward_value(value, decoded), metadata, "int8", 32)
    decoded, scales = _tile_qdq(value, block_v, state_format, state_quantizer=state_quantizer)
    return EncodedLinearAttentionTensor(decoded, scales, state_format, block_v)


def _encode_factor(value, quantizer):
    if quantizer is None:
        return EncodedLinearAttentionTensor(value, None, "identity", None)
    decoded = quantizer(value)
    enabled = quantizer.is_enabled and quantizer._if_quant
    format = str(quantizer.num_bits) if enabled else "identity"
    return EncodedLinearAttentionTensor(decoded, None, format, None)


def _signature(config, state_qdq, block_v, state_format, state_quantizer, replay_quantizers):
    signature = (
        config.model_dump_json(exclude={"prefill_state_qdq"})
        + f"/{state_qdq}/{block_v}/{state_format}"
    )
    if state_quantizer is not None and state_quantizer.block_sizes is not None:
        signature += f"/group={state_quantizer.block_sizes[-1]}"
    for quantizer in replay_quantizers:
        if quantizer is None:
            signature += "/factor=None"
        else:
            grouping = (
                quantizer.block_sizes if quantizer.block_sizes is not None else quantizer.axis
            )
            signature += (
                f"/factor={quantizer.is_enabled}/{quantizer._if_quant}/{quantizer.num_bits}"
                f"/{grouping}/{quantizer.backend}/{quantizer.backend_extra_args}"
            )
    return signature


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
    return forward_value(gate, rounded)


def _prepare_carry(
    q,
    k,
    v,
    g,
    beta,
    config,
    state_qdq,
    block_v,
    initial_state,
    carry,
    position,
    state_format,
    state_quantizer,
    replay_quantizers,
):
    if q.ndim != 3 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k/v must have aligned [T,H,D] shapes")
    if beta.shape != q.shape[:2] or g.shape not in (beta.shape, q.shape):
        raise ValueError("beta must be [T,H]; g must be [T,H] or [T,H,Dk]")
    if block_v not in (16, 32, 64, 128):
        raise ValueError("block_v must be 16, 32, 64, or 128")
    if state_format not in ("fp8_e4m3", "int8"):
        raise ValueError("State format must be fp8_e4m3 or int8")
    hadamard = config.state_codec == "int8_hadamard32"
    if hadamard:
        if state_quantizer is not None and state_quantizer.block_sizes is not None:
            raise ValueError("TensorQuantizer block_sizes requires state_codec='tile'")
        if state_qdq and state_format != "int8":
            raise ValueError("int8_hadamard32 requires INT8 state quantization")
        if v.shape[-1] % 32 or block_v < 32:
            raise ValueError("int8_hadamard32 requires Dv divisible by 32 and block_v >= 32")
    signature = _signature(
        config, state_qdq, block_v, state_format, state_quantizer, replay_quantizers
    )
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
        initial = _hadamard32(carry.anchor.values) if hadamard else carry.anchor.values
        anchor = _encode(
            initial,
            state_qdq and config.quantize_initial,
            block_v,
            state=True,
            state_format=state_format,
            state_codec=config.state_codec,
            state_quantizer=state_quantizer,
        )
        carry = LinearAttentionCarry(
            anchor, (), carry.position, True, signature, "hadamard32" if hadamard else "identity"
        )
    return carry, signature


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
    state_quantizer=None,
    replay_key_quantizer=None,
    replay_update_quantizer=None,
    block_v=64,
    initial_state=None,
    carry=None,
    position=0,
    scale=None,
    use_qk_l2norm_in_kernel=False,
):
    """Run one preactivated sequence [T,H,D] with explicit token/replay write events.

    Keys and value heads must already be aligned. Scalar GDN or per-key-channel
    KDA log gates are accepted. Outputs and all returned carry values retain their
    graphs. An empty call performs no write or initial-state quantization.
    Replay factors use the supplied TensorQuantizers; omitted quantizers leave factors unchanged.
    """
    state_quantizer, state_qdq, state_format = _resolve_state_quantizer(
        state_quantizer, state_qdq, state_format
    )
    serving = config.precision == "vllm_0_15"
    if serving and q.device.type != "cuda":
        raise ValueError("vllm_0_15 requires CUDA")
    replay_quantizers = (replay_key_quantizer, replay_update_quantizer)
    if config.replay is not None and config.replay._legacy_factor_qdq:
        if any(quantizer is None for quantizer in replay_quantizers):
            raise ValueError("Replace factor_qdq with explicit replay key/update TensorQuantizers")
    for quantizer in replay_quantizers:
        if quantizer is not None and quantizer.is_enabled:
            if config.replay is None or not quantizer.fake_quant:
                raise ValueError(
                    "Replay factor quantizers require replay mode and fake quantization"
                )
    carry, signature = _prepare_carry(
        q,
        k,
        v,
        g,
        beta,
        config,
        state_qdq,
        block_v,
        initial_state,
        carry,
        position,
        state_format,
        state_quantizer,
        replay_quantizers,
    )
    if len(q) == 0:
        # Keep empty input gradients defined without introducing a state write.
        zero = (q.sum() + k.sum() + v.sum() + g.sum() + beta.sum()) * 0
        output = v + zero
        return output, carry
    if config.state_codec == "int8_hadamard32":
        v = _hadamard32(v)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for t in range(len(q)):
            entries = carry.entries
            if config.replay is not None and config.replay.encoding == "reencode":
                entries = tuple(
                    ReplayEntry(
                        _encode_factor(e.key.values, replay_key_quantizer),
                        _encode_factor(e.update.values, replay_update_quantizer),
                        e.log_retention,
                    )
                    for e in entries
                )
                carry = LinearAttentionCarry(
                    carry.anchor, entries, carry.position, True, signature, carry.value_basis
                )
            state = carry.reconstruct(original_basis=False)
            gate = _round_log_gate(g[t], config.decay_log_step)
            if serving:
                from ._vllm_autograd import step as serving_step

                native_output, working = serving_step(
                    q[t], k[t], v[t], gate, beta[t], state, scale, use_qk_l2norm_in_kernel
                )
            else:
                decay = gate.exp().unsqueeze(-1)
                if gate.ndim == 1:
                    decay = decay.unsqueeze(-1)
                key = _encode_factor(k[t], replay_key_quantizer)
                decayed = state * decay
                residual = v[t] - _sum_keys(key.values.unsqueeze(-1) * decayed)
                update = _encode_factor(beta[t].unsqueeze(-1) * residual, replay_update_quantizer)
                working = decayed + key.values.unsqueeze(-1) * update.values.unsqueeze(-2)
            if config.replay is not None:
                entries = (*entries, ReplayEntry(key, update, gate))
            refresh = config.replay is None or len(entries) == config.replay.window
            if refresh:
                anchor = _encode(
                    working,
                    state_qdq,
                    block_v,
                    state=True,
                    state_format=state_format,
                    state_codec=config.state_codec,
                    state_quantizer=state_quantizer,
                )
                entries = ()
                stored = anchor.values
            else:
                anchor = carry.anchor
                stored = working
            next_carry = LinearAttentionCarry(
                anchor, entries, carry.position + 1, True, signature, carry.value_basis
            )
            read = working if config.readout == "working" else stored
            outputs.append(
                native_output if serving else _sum_keys(q[t].unsqueeze(-1) * read) * scale
            )
            carry = next_carry
    output = torch.stack(outputs)
    if config.state_codec == "int8_hadamard32":
        output = _hadamard32(output)
    return output, carry
