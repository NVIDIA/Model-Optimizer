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

"""Training forwards with a chunked prefill prefix and recurrent decode suffix."""

from contextlib import contextmanager
from itertools import pairwise
from numbers import Integral

import torch

from .decode import _recurrent_decode
from .utils import _resolve_state_quantizer, _state_qdq, forward_value

__all__ = ["get_linear_attention_layers", "linear_attention_training_phase"]


def _lengths(values, name):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    values = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values
    ):
        raise ValueError(f"{name} must be nonnegative integers")
    return tuple(int(value) for value in values)


class _PackedBoundaries:
    """Own immutable host boundaries and verify each layer's current packed tensor."""

    def __init__(self, values):
        self.values = None if values is None else _lengths(values, "Packed boundaries")

    def resolve(self, actual):
        if actual is None:
            if self.values is not None:
                raise ValueError("Phase cu_seqlens requires packed boundaries from the layer")
            return None
        # Tensor versions do not track writes through .data, NumPy or DLPack aliases.
        values = _lengths(actual, "Packed boundaries")
        if self.values is not None and values != self.values:
            raise ValueError("Phase cu_seqlens does not match the layer's packed boundaries")
        self.values = values
        return values


def get_linear_attention_layers(model, *, enabled_only=False, unphased_only=False):
    """Return serving-policy layers from a module or iterable of model chunks.

    Disabled state quantizers are included by default so a baseline can use the
    same serving arithmetic and phase as QAT. Use ``enabled_only`` for setup checks
    and ``unphased_only`` to select layers needing adapter-provided defaults.
    """
    # The plugin imports this numerical package during quantization initialization.
    from ..plugins.linear_attention import _LinearAttentionQuantMixin

    roots = (model,) if isinstance(model, torch.nn.Module) else model
    return tuple(
        dict.fromkeys(
            module
            for root in roots
            for module in root.modules()
            if isinstance(module, _LinearAttentionQuantMixin)
            and module.linear_attention_config.backend == "serving"
            and (not enabled_only or module._linear_attn_state.is_enabled)
            and (not unphased_only or module._linear_attention_prefill_lengths is None)
        )
    )


def _prefix_lengths(values, valid_lengths):
    prefixes = _lengths(values, "Prefill lengths")
    if valid_lengths is not None and (
        len(prefixes) != len(valid_lengths)
        or any(prefix > end for prefix, end in zip(prefixes, valid_lengths))
    ):
        raise ValueError("Supply one prefill length <= sequence length per sequence")
    return prefixes


@contextmanager
def linear_attention_training_phase(
    model, prefill_lengths, *, sequence_lengths=None, cu_seqlens=None, preserve_existing=False
):
    """Supply per-sequence phases to a module or iterable of model chunks/layers.

    Enabled state quantization requires a phase, including calibration/eval. Disabled
    quantizers use the original module outside a phase and serving arithmetic inside.
    Megatron selective recompute captures the phase for backward after context exit.

    ``sequence_lengths`` excludes right padding in dense rows or padded THD segments.
    ``cu_seqlens`` describes physical packed storage, including alignment padding.
    Megatron derives omitted valid lengths from unpadded ``cu_seqlens_q`` metadata;
    padded-only metadata requires explicit ``sequence_lengths``.
    Padding produces zero attention outputs and never updates the recurrent state.
    ``preserve_existing=True`` installs defaults only on layers without a phase.
    Empty pipeline stages participate as no-ops.
    """
    layers = get_linear_attention_layers(model)
    if preserve_existing:
        layers = tuple(m for m in layers if m._linear_attention_prefill_lengths is None)
    valid_lengths = (
        None if sequence_lengths is None else _lengths(sequence_lengths, "Sequence lengths")
    )
    lengths = _prefix_lengths(prefill_lengths, valid_lengths)
    boundaries = _PackedBoundaries(cu_seqlens)
    previous = [
        (
            m._linear_attention_prefill_lengths,
            m._linear_attention_sequence_lengths,
            m._linear_attention_cu_seqlens,
        )
        for m in layers
    ]
    try:
        for module in layers:
            module._linear_attention_prefill_lengths = lengths
            module._linear_attention_sequence_lengths = valid_lengths
            module._linear_attention_cu_seqlens = boundaries
        yield model
    finally:
        for module, (original, original_lengths, original_boundaries) in zip(layers, previous):
            module._linear_attention_prefill_lengths = original
            module._linear_attention_sequence_lengths = original_lengths
            module._linear_attention_cu_seqlens = original_boundaries


def _prepare_prefill_inputs(q, k, v, g, beta, *, policy, chunk_size):
    """Validate the native policy and prepare FP32 values for the training adjoint."""
    if policy.backend != "serving" or chunk_size != 64:
        raise ValueError("State training requires backend='serving' and chunk_size=64")
    if q.device.type != "cuda" or any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Serving arithmetic requires CUDA BF16 Q/K/V inputs")
    return tuple(x.float() for x in (q, k, v, g, beta))


def _prefill_decode_forward(
    q,
    k,
    v,
    g,
    beta,
    *,
    policy,
    state_qdq,
    state_format,
    state_quantizer,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    cu_seqlens_cpu,
    state_v_first,
    output_dtype,
    beta_dtype,
    prefill_lengths,
    sequence_lengths=None,
    use_qk_l2norm_in_kernel=False,
    gate_inputs=None,
):
    """Run both prefill and decode phases in one differentiable training forward.

    Each sequence's chunked prefix produces the state for its token or ReplaySSM
    suffix. ``recurrent_decode`` wraps that dense state in LinearAttentionState
    and preserves its gradient connection to prefill. Outputs are joined in token
    order; the final runtime state is reconstructed to the caller's dense layout.

    Args:
        prefill_lengths: Prefix token count per sequence. For 128 tokens, a value
            of 64 selects 64 chunked prefill tokens followed by 64 recurrent tokens.
        sequence_lengths: Valid token count per sequence, excluding right padding.
            Defaults to the dense row length or each packed storage segment length.
    """
    state_quantizer, state_qdq, state_format = _resolve_state_quantizer(
        state_quantizer, state_qdq, state_format
    )
    if state_qdq and prefill_lengths is None:
        raise ValueError(
            "State quantization requires explicit prefill lengths through "
            "linear_attention_training_phase"
        )
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k and v must have compatible [B,T,H,D] shapes")
    batch, length, key_heads, keys = q.shape
    heads, values = v.shape[2:]
    if batch < 1 or key_heads < 1 or heads % key_heads or beta.shape != (batch, length, heads):
        raise ValueError("Invalid batch/head dimensions or beta shape")
    if g.shape not in (beta.shape, (*beta.shape, keys)):
        raise ValueError("Invalid GDN/KDA log-retention shape")
    q, k = (x.repeat_interleave(heads // key_heads, dim=2) for x in (q, k))
    if isinstance(cu_seqlens_cpu, _PackedBoundaries):
        boundaries = cu_seqlens_cpu.resolve(cu_seqlens)
    elif cu_seqlens_cpu is not None:
        boundaries = _PackedBoundaries(cu_seqlens_cpu).resolve(cu_seqlens)
    else:
        boundaries = cu_seqlens
    if boundaries is None:
        sequences = [(b, 0, length) for b in range(batch)]
    else:
        bounds = _lengths(boundaries, "Packed boundaries")
        if (
            batch != 1
            or len(bounds) < 2
            or bounds[0] != 0
            or bounds[-1] != length
            or any(a > b for a, b in pairwise(bounds))
        ):
            raise ValueError(
                "Packed boundaries must partition a batch of one, allowing empty entries"
            )
        sequences = [(0, a, b) for a, b in pairwise(bounds)]
    valid_lengths = (
        tuple(end - start for _, start, end in sequences)
        if sequence_lengths is None
        else _lengths(sequence_lengths, "Sequence lengths")
    )
    if len(valid_lengths) != len(sequences) or any(
        valid > end - start for valid, (_, start, end) in zip(valid_lengths, sequences)
    ):
        raise ValueError("Supply one sequence length within its storage segment per sequence")
    prefixes = _prefix_lengths(
        valid_lengths if prefill_lengths is None else prefill_lengths, valid_lengths
    )
    if (
        keys > 256
        or (g.ndim == 4 and values != keys)
        or (use_qk_l2norm_in_kernel and keys & (keys - 1))
    ):
        raise ValueError(
            "Serving arithmetic requires K <= 256, KDA V=K, and power-of-two K for normalization"
        )
    if policy.precision != "replayssm":
        # Validate the optional native backend before launching any prefix kernels.
        from ...kernels.quantization.linear_attention.serving.forward import validate_profile

        validate_profile(
            gate_inputs,
            use_qk_l2norm_in_kernel,
            kimi_k3=policy.precision == "vllm_kimi_k3",
        )
    if initial_state is None:
        states = q.new_zeros(len(sequences), heads, keys, values)
    else:
        states = initial_state.transpose(-1, -2) if state_v_first else initial_state
        states = states.to(q.dtype)
        if states.shape != (len(sequences), heads, keys, values):
            raise ValueError("Initial state shape does not match sequence/head dimensions")
    outputs, finals = [], []
    for n, (b, start, storage_end) in enumerate(sequences):
        end = start + valid_lengths[n]
        split = start + prefixes[n]
        prefix, state = q.new_empty(0, heads, values), states[n]
        if prefixes[n]:
            with torch.autocast(device_type=q.device.type, enabled=False):
                from ._vllm_autograd import prefix as serving_prefix

                # A continuation prefill consumes a stored cache just as native serving
                # does. A fresh zero-state prefix has no incoming cache to quantize.
                if initial_state is not None and state_qdq:
                    if policy.precision == "replayssm":
                        from ...kernels.quantization.linear_attention.serving.replay import (
                            checkpoint,
                            original_basis,
                        )

                        with torch.no_grad():
                            decoded, _ = checkpoint(state, True)
                            decoded = original_basis(decoded)
                        state = forward_value(state, decoded)
                    else:
                        state = _state_qdq(
                            state, policy.state_block_v, state_format, state_quantizer
                        )
                prefix, state = serving_prefix(
                    q=q[b, start:split],
                    k=k[b, start:split],
                    v=v[b, start:split],
                    g=g[b, start:split],
                    beta=beta[b, start:split],
                    state=state,
                    scale=keys**-0.5 if scale is None else scale,
                    beta_dtype=beta_dtype,
                    precision=policy.precision,
                    normalize=use_qk_l2norm_in_kernel,
                    gate_inputs=(
                        (
                            gate_inputs[0][b, start:split],
                            gate_inputs[1][b, start:split],
                            *gate_inputs[2:],
                        )
                        if gate_inputs is not None
                        else None
                    ),
                )
        # Keep the prefix state attached so suffix losses backpropagate through prefill.
        # recurrent_decode applies configured state QDQ at the handoff and suffix writes.
        suffix, carry = _recurrent_decode(
            *(x[b, split:end] for x in (q, k, v, g, beta)),
            config=policy,
            state_qdq=state_qdq,
            state_format=state_format,
            state_quantizer=state_quantizer,
            initial_state=state,
            position=prefixes[n],
            gate_inputs=(
                (
                    gate_inputs[0][b, split:end],
                    gate_inputs[1][b, split:end],
                    *gate_inputs[2:],
                )
                if gate_inputs is not None
                else None
            ),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
        # Keep storage offsets for downstream layers while excluding padding from the graph.
        pieces = (prefix, suffix)
        if storage_end > end:
            pieces += (v.new_zeros(storage_end - end, heads, values),)
        outputs.append(torch.cat(pieces))
        if output_final_state:
            finals.append(carry.reconstruct())
    output = torch.stack(outputs) if boundaries is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals) if output_final_state else None
    if final is not None and state_v_first:
        final = final.transpose(-1, -2)
    return output.to(output_dtype), final
