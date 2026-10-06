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

import torch

from ._chunk_prefill import chunk_gdn, chunk_kda
from .decode import recurrent_decode
from .utils import _resolve_state_quantizer, _state_qdq, forward_value

__all__ = ["linear_attention_training_phase"]


def _lengths(values):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    values = tuple(values)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Prefill lengths must be nonnegative integers")
    return values


@contextmanager
def linear_attention_training_phase(model, prefill_lengths):
    """Supply explicit sequence phases through forward and checkpointed backward.

    Runtime phase metadata is local to the converted layers and restored on exit.
    Keep this context active through backward when activation checkpointing is used.
    """
    # The plugin imports this numerical package during quantization initialization.
    from ..plugins.linear_attention import _LinearAttentionQuantMixin

    lengths = _lengths(prefill_lengths)
    layers = [
        m
        for m in model.modules()
        if isinstance(m, _LinearAttentionQuantMixin)
        and m.linear_attention_config.decode is not None
    ]
    if not layers:
        raise ValueError("The model has no converted decode-aware linear-attention layers")
    previous = [getattr(m, "_linear_attention_prefill_lengths", None) for m in layers]
    try:
        for module in layers:
            module._linear_attention_prefill_lengths = lengths
        yield model
    finally:
        for module, original in zip(layers, previous):
            module._linear_attention_prefill_lengths = original


def _prepare_prefill_inputs(q, k, v, g, beta, *, policy, chunk_size, normalize):
    """Validate the shared policy and prepare GDN/KDA working dtypes and Q/K normalization."""
    if policy.backend not in ("serving", "reference") or chunk_size != policy.chunk_size:
        raise ValueError(
            "State training requires backend='serving' or 'reference' and its configured chunk size"
        )
    if policy.decode is None:
        raise ValueError("An explicit decode policy is required")
    serving = policy.decode.precision != "full"
    if serving and (q.device.type != "cuda" or any(x.dtype != torch.bfloat16 for x in (q, k, v))):
        raise ValueError("Serving arithmetic requires CUDA BF16 Q/K/V inputs")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    if normalize and not serving:
        q, k = (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt() for x in (q, k))
    return q, k, v, g, beta, serving


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
    replay_key_quantizer,
    replay_update_quantizer,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    cu_seqlens_cpu,
    state_v_first,
    output_dtype,
    beta_dtype,
    prefill_lengths,
    use_qk_l2norm_in_kernel=False,
    replay_gate_inputs=None,
):
    """Run both prefill and decode phases in one differentiable training forward.

    Each sequence's chunked prefix produces the state for its token or ReplaySSM
    suffix. Their outputs are joined in token order for the training loss.

    Args:
        prefill_lengths: Prefix token count per sequence. For 128 tokens, a value
            of 64 selects 64 chunked prefill tokens followed by 64 recurrent tokens.
    """
    state_quantizer, state_qdq, state_format = _resolve_state_quantizer(
        state_quantizer, state_qdq, state_format
    )
    if prefill_lengths is None:
        raise ValueError("Decode-aware training requires explicit per-sequence prefill lengths")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("q/k and v must have compatible [B,T,H,D] shapes")
    batch, length, key_heads, keys = q.shape
    heads, values = v.shape[2:]
    if batch < 1 or key_heads < 1 or heads % key_heads or beta.shape != (batch, length, heads):
        raise ValueError("Invalid batch/head dimensions or beta shape")
    if g.shape not in (beta.shape, (*beta.shape, keys)):
        raise ValueError("Invalid GDN/KDA log-retention shape")
    q, k = (x.repeat_interleave(heads // key_heads, dim=2) for x in (q, k))
    boundaries = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if boundaries is None:
        sequences = [(b, 0, length) for b in range(batch)]
    else:
        bounds = _lengths(boundaries)
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
    prefixes = _lengths(prefill_lengths)
    if len(prefixes) != len(sequences) or any(
        p > end - start for p, (_, start, end) in zip(prefixes, sequences)
    ):
        raise ValueError("Supply one valid prefill length per sequence")
    serving_precision = policy.decode.precision != "full"
    if serving_precision and (
        keys > 256
        or (g.ndim == 4 and values != keys)
        or (use_qk_l2norm_in_kernel and keys & (keys - 1))
    ):
        raise ValueError(
            "Serving arithmetic requires K <= 256, KDA V=K, and power-of-two K for normalization"
        )
    if initial_state is None:
        states = q.new_zeros(len(sequences), heads, keys, values)
    else:
        states = initial_state.transpose(-1, -2) if state_v_first else initial_state
        states = states.to(q.dtype)
        if states.shape != (len(sequences), heads, keys, values):
            raise ValueError("Initial state shape does not match sequence/head dimensions")
    prefix_fn = chunk_kda if g.ndim == 4 else chunk_gdn
    outputs, finals = [], []
    for n, (b, start, end) in enumerate(sequences):
        split = start + prefixes[n]
        prefix, state = q.new_empty(0, heads, values), states[n]
        if prefixes[n]:
            with torch.autocast(device_type=q.device.type, enabled=False):
                if serving_precision:
                    from ._vllm_autograd import prefix as serving_prefix

                    # A continuation prefill consumes a stored cache just as native serving
                    # does. A fresh zero-state prefix has no incoming cache to quantize.
                    if initial_state is not None and state_qdq:
                        if policy.decode.precision == "replayssm":
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
                                state, policy.state.block_v, state_format, state_quantizer
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
                        normalize=use_qk_l2norm_in_kernel,
                    )
                else:
                    prefix, state = prefix_fn(
                        *(x[b, start:split] for x in (q, k, v, g, beta)),
                        state_qdq=state_qdq and policy.decode.prefill_state_qdq,
                        state_format=state_format,
                        state_quantizer=state_quantizer,
                        scale=scale,
                        initial_state=state,
                        chunk_size=policy.chunk_size,
                        state_qdq_block_v=policy.state.block_v,
                    )
        # Keep the prefix state attached so suffix losses backpropagate through prefill.
        # recurrent_decode applies configured state QDQ at the handoff and suffix writes.
        suffix, carry = recurrent_decode(
            *(x[b, split:end] for x in (q, k, v, g, beta)),
            config=policy.decode,
            replay_key_quantizer=replay_key_quantizer,
            replay_update_quantizer=replay_update_quantizer,
            state_qdq=state_qdq,
            state_format=state_format,
            state_quantizer=state_quantizer,
            block_v=policy.state.block_v,
            initial_state=state,
            position=prefixes[n],
            replay_gate_inputs=(
                (
                    replay_gate_inputs[0][b, split:end],
                    replay_gate_inputs[1][b, split:end],
                    *replay_gate_inputs[2:],
                )
                if replay_gate_inputs is not None
                else None
            ),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
        outputs.append(torch.cat((prefix, suffix)))
        finals.append(carry.reconstruct())
    output = torch.stack(outputs) if boundaries is None else torch.cat(outputs).unsqueeze(0)
    final = torch.stack(finals)
    if state_v_first:
        final = final.transpose(-1, -2)
    return output.to(output_dtype), final if output_final_state else None
