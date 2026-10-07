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

from .config import LinearAttentionConfig
from .utils import _resolve_state_quantizer, _tile_qdq, forward_value

__all__ = [
    "EncodedLinearAttentionTensor",
    "LinearAttentionCarry",
    "LinearAttentionState",
    "ReplayEntry",
    "recurrent_decode",
]


@dataclass
class EncodedLinearAttentionTensor:
    """Floating QDQ values with autograd history and optional quantization scales.

    ``format`` and ``block_v`` describe the emulated encoding; ``values`` stores
    decoded values, not packed INT8/FP8 codes. Scale metadata is non-differentiable.
    """

    values: torch.Tensor
    scales: torch.Tensor | None
    format: str
    block_v: int | None


@dataclass
class ReplayEntry:
    """One saved rank-one update since the last state checkpoint.

    ``key`` is [H,K] and ``update`` is the beta-scaled residual [H,V]. ReplaySSM
    stores their BF16 forward values once. ``log_retention`` is [H] for GDN or
    [H,K] for KDA and records this token's decay for subsequent reconstruction.
    """

    key: EncodedLinearAttentionTensor
    update: EncodedLinearAttentionTensor
    log_retention: torch.Tensor


@dataclass
class LinearAttentionState:
    """Runtime state passed between recurrent tokens and successive decode calls.

    With ``replay_window=1``, every token refreshes ``anchor`` and ``entries``
    stays empty. With a larger window, the anchor stays fixed while updates
    accumulate; reaching the window refreshes the anchor and clears the entries.
    Tensor values retain their autograd history for QAT/QAD across call boundaries.
    """

    anchor: EncodedLinearAttentionTensor  # Checkpoint [H,K,V]; QDQ is applied when enabled.
    entries: tuple[ReplayEntry, ...]  # Updates since the checkpoint, in token order.
    position: int  # Next token position, including the supplied prefix offset.
    started: bool  # Initial-state encoding has run; resuming must not repeat it.
    signature: str  # Execution/quantizer settings used to check compatible reuse.
    value_basis: str = "identity"  # Anchor and updates may use the Hadamard value basis.
    precision: str = "vllm_0_15"  # Selects the native profile's reconstruction arithmetic.

    @property
    def cursor(self):
        """Number of encoded updates since the last anchor refresh."""
        return len(self.entries)

    def reconstruct(self, *, original_basis=True):
        """Return the current dense [H,K,V] state without refreshing or quantizing it.

        Apply pending updates and decay to the anchor, preserving gradients.
        Decode uses ``original_basis=False`` to continue in the stored basis;
        the default converts Hadamard state back for the caller's final-state output.
        Keep this object, rather than only this tensor, to resume a replay window.
        """
        state = _replay_state(self) if self.precision == "replayssm" else self.anchor.values
        if original_basis and self.value_basis == "hadamard32":
            # The optional serving fork owns the checkpoint basis transform.
            from ...kernels.quantization.linear_attention.serving.replay import (
                original_basis as restore_basis,
            )

            state = restore_basis(state)
        return state


# Compatibility name for existing imports and pickled runtime state objects.
LinearAttentionCarry = LinearAttentionState


def _hadamard32(value):
    """Apply the orthonormal Sylvester transform to contiguous 32-value groups."""
    shape = value.shape
    for width in (1, 2, 4, 8, 16):
        pairs = value.reshape(*shape[:-1], -1, 2, width)
        left, right = pairs.unbind(-2)
        value = torch.stack((left + right, left - right), dim=-2).reshape(shape)
    return value * (32**-0.5)


def _replay_state(carry: LinearAttentionState, current_gate=None):
    """Decay the anchor and add saved updates weighted by their subsequent gates.

    KDA weights keys per channel; GDN weights updates per head. ``current_gate``
    includes the next KDA token's decay before BF16 weighting, without adding its update.
    """
    state = carry.anchor.values
    if not carry.entries:
        if current_gate is not None:
            state = state * current_gate.exp().unsqueeze(-1)
        return state
    gates = torch.stack([entry.log_retention for entry in carry.entries], dim=1)
    keys = torch.stack([entry.key.values for entry in carry.entries], dim=1)
    updates = torch.stack([entry.update.values for entry in carry.entries], dim=1)
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


def _encode(
    value,
    enabled,
    block_v,
    *,
    state_format="fp8_e4m3",
    state_quantizer=None,
):
    """Apply TensorQuantizer state QDQ with straight-through gradients."""
    if not enabled:
        return EncodedLinearAttentionTensor(value, None, "identity", None)
    if state_quantizer is not None and state_quantizer.block_sizes is not None:
        # TensorQuantizer owns dynamic scales; the floating carry needs only its QDQ output.
        return EncodedLinearAttentionTensor(
            state_quantizer(value), None, state_format, state_quantizer.block_sizes[-1]
        )
    decoded, scales = _tile_qdq(value, block_v, state_format, state_quantizer=state_quantizer)
    return EncodedLinearAttentionTensor(decoded, scales, state_format, block_v)


def _signature(config, state_qdq, block_v, state_format, state_quantizer):
    """Record the execution and quantizer settings required when resuming a state."""
    signature = config.model_dump_json() + f"/{state_qdq}/{block_v}/{state_format}"
    if state_quantizer is not None and state_quantizer.block_sizes is not None:
        signature += f"/group={state_quantizer.block_sizes[-1]}"
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
):
    """Initialize from a dense state or validate and resume an existing runtime state.

    The first nonempty call applies the handoff transform/QDQ once. A resumed
    state preserves its anchor, replay cursor, and token position; an empty call
    leaves initial encoding deferred until a token is actually processed.
    """
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
    signature = _signature(config, state_qdq, block_v, state_format, state_quantizer)
    if carry is not None and initial_state is not None:
        raise ValueError("Supply either carry or initial_state")
    shape = (q.shape[1], q.shape[2], v.shape[2])
    if carry is None:
        initial_state = q.new_zeros(shape) if initial_state is None else initial_state
        carry = LinearAttentionState(
            _encode(initial_state, False, block_v, state_format=state_format),
            (),
            position,
            False,
            signature,
        )
    if carry.signature != signature or carry.anchor.values.shape != shape:
        raise ValueError("Carry policy or state shape does not match this recurrence")
    if carry.cursor >= config.replay_window:
        raise ValueError("Replay cursor must be below the refresh window")
    if len(q) and not carry.started:
        initial = _hadamard32(carry.anchor.values) if hadamard else carry.anchor.values
        if config.precision == "replayssm":
            # The optional serving fork owns checkpoint rounding and Hadamard arithmetic.
            from ...kernels.quantization.linear_attention.serving.replay import checkpoint

            with torch.no_grad():
                decoded, metadata = checkpoint(carry.anchor.values, state_qdq)
            anchor = EncodedLinearAttentionTensor(
                forward_value(initial, decoded),
                metadata,
                "int8" if state_qdq else "identity",
                32 if state_qdq else None,
            )
        else:
            anchor = _encode(
                initial,
                state_qdq,
                block_v,
                state_format=state_format,
                state_quantizer=state_quantizer,
            )
        carry = LinearAttentionState(
            anchor,
            (),
            carry.position,
            True,
            signature,
            "hadamard32" if hadamard else "identity",
            config.precision,
        )
    return carry, signature


def recurrent_decode(
    q,
    k,
    v,
    g,
    beta,
    *,
    config: LinearAttentionConfig,
    state_qdq=False,
    state_format="fp8_e4m3",
    state_quantizer=None,
    initial_state=None,
    carry: LinearAttentionState | None = None,
    position=0,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    replay_gate_inputs=None,
) -> tuple[torch.Tensor, LinearAttentionState]:
    """Run a recurrent suffix and return its outputs and resumable runtime state.

    Q/K are [T,H,K], V is [T,H,V], and beta is [T,H]; heads must be aligned.
    Log gates are [T,H] for GDN or [T,H,K] for KDA. The first token consumes the
    handoff state after enabled QDQ. Each token reads its output from the working
    state before checkpoint rounding; replay_window determines when that state
    is saved. Inputs and the initial state are promoted to FP32 working values;
    native forward values use a differentiable adjoint for training.

    Args:
        config: The same LinearAttentionConfig used by the chunked prefix.
        initial_state: Dense [H,K,V] state in the original value basis, normally
            from prefill. Defaults to zeros; mutually exclusive with ``carry``.
        carry: LinearAttentionState returned by an earlier call. Pass it directly
            to preserve the replay window and autograd history across calls.
        position: Starting token offset when creating a state, usually the prefix
            length. An existing carry retains its own position.

    Returns:
        Outputs [T,H,V] and the updated LinearAttentionState, both retaining their
        graphs. An empty call performs no state write or initial-state QDQ.
    """
    state_quantizer, state_qdq, state_format = _resolve_state_quantizer(
        state_quantizer, state_qdq, state_format
    )
    if config.backend != "serving":
        raise ValueError("State QAT requires backend='serving'")
    block_v = config.state_block_v
    serving = config.precision == "vllm_0_15"
    native_replay = config.precision == "replayssm"
    if q.device.type != "cuda":
        raise ValueError("Serving arithmetic requires CUDA")
    # BF16 inputs need FP32 working values for both replay reconstruction and its adjoint.
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.float()
    if (
        native_replay
        and len(q)
        and (
            q.shape[-1] < 32
            or q.shape[-1] & (q.shape[-1] - 1)
            or (g.ndim == 2 and replay_gate_inputs is None)
        )
    ):
        raise ValueError("ReplaySSM requires power-of-two K >= 32 and raw GDN gate inputs")
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
    )
    if len(q) == 0:
        # Keep empty input gradients defined without introducing a state write.
        zero = (q.sum() + k.sum() + v.sum() + g.sum() + beta.sum()) * 0
        output = v + zero
        return output, carry
    original_v = v
    if config.state_codec == "int8_hadamard32":
        v = _hadamard32(v)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    outputs = []
    native_outputs = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for t in range(len(q)):
            entries = carry.entries
            state = carry.reconstruct(original_basis=False)
            gate = g[t]
            if native_replay:
                from ...kernels.quantization.linear_attention.serving.replay import step

                with torch.no_grad():
                    native = step(
                        q[t],
                        k[t],
                        original_v[t],
                        gate,
                        beta[t],
                        carry,
                        config.replay_window,
                        state_qdq,
                        scale,
                        use_qk_l2norm_in_kernel,
                        (
                            replay_gate_inputs[0][t],
                            replay_gate_inputs[1][t],
                            *replay_gate_inputs[2:],
                        )
                        if replay_gate_inputs is not None
                        else None,
                    )
                native_outputs.append(native[0])
            if serving:
                from ._vllm_autograd import step as serving_step

                native_output, working = serving_step(
                    q[t], k[t], v[t], gate, beta[t], state, scale, use_qk_l2norm_in_kernel
                )
            else:
                decay = gate.exp().unsqueeze(-1)
                if gate.ndim == 1:
                    decay = decay.unsqueeze(-1)
                current_key = k[t]
                if native_replay and use_qk_l2norm_in_kernel:
                    current_key = (
                        current_key / (current_key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                    )
                key = EncodedLinearAttentionTensor(current_key, None, "identity", None)
                decayed = state * decay
                if native_replay and gate.ndim == 2:
                    decayed = _replay_state(carry, gate)
                residual = v[t] - _sum_keys(key.values.unsqueeze(-1) * decayed)
                update = EncodedLinearAttentionTensor(
                    beta[t].unsqueeze(-1) * residual, None, "identity", None
                )
                working = decayed + key.values.unsqueeze(-1) * update.values.unsqueeze(-2)
            if config.replay_window > 1:
                if native_replay and carry.cursor + 1 < config.replay_window:
                    key.values = forward_value(key.values, native[3])
                    update.values = forward_value(update.values, native[4])
                    gate = forward_value(gate, native[5])
                entries = (*entries, ReplayEntry(key, update, gate))
            # Window 1 checkpoints every token; replay checkpoints only at a full window.
            # Between checkpoints, keep the anchor and save the native BF16 update values.
            refresh = config.replay_window == 1 or len(entries) == config.replay_window
            if refresh:
                if native_replay:
                    # Native step already quantized the checkpoint; attach its values to the adjoint.
                    anchor = EncodedLinearAttentionTensor(
                        forward_value(working, native[1]),
                        native[2],
                        "int8" if state_qdq else "identity",
                        32 if state_qdq else None,
                    )
                else:
                    anchor = _encode(
                        working,
                        state_qdq,
                        block_v,
                        state_format=state_format,
                        state_quantizer=state_quantizer,
                    )
                entries = ()
            else:
                anchor = carry.anchor
            # A new container keeps tensors needed by earlier tokens' backward passes intact.
            next_carry = LinearAttentionState(
                anchor,
                entries,
                carry.position + 1,
                True,
                signature,
                carry.value_basis,
                config.precision,
            )
            query = q[t]
            if native_replay and use_qk_l2norm_in_kernel:
                query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            # Read this token from the working state, before the next-state checkpoint QDQ.
            outputs.append(
                native_output if serving else _sum_keys(query.unsqueeze(-1) * working) * scale
            )
            carry = next_carry
    output = torch.stack(outputs)
    if config.state_codec == "int8_hadamard32":
        output = _hadamard32(output)
    if native_replay:
        output = forward_value(output, torch.stack(native_outputs))
    return output, carry
