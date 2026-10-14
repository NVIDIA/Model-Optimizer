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

"""Native vLLM 0.30 Triton forwards; training owns private, differentiable state."""

from functools import cache, lru_cache

import torch


@cache
def _kernels(precision):
    # Defer optional serving/model dependencies until their explicit profile is selected.
    if precision == "replayssm":
        from .replay import _prefill_kernels

        return _prefill_kernels()
    import vllm

    if vllm.__version__ != "0.30.0":
        raise ImportError("Linear-attention vLLM profiles require vllm==0.30.0")
    from vllm.third_party.flash_linear_attention import ops as gdn_ops

    if precision == "vllm_kimi_k3":
        from vllm.models.kimi_k3.nvidia.ops.third_party import kda as kda_ops
    else:
        from vllm.third_party.flash_linear_attention.ops import kda as kda_ops
    return gdn_ops, kda_ops


def _state_layout(state):
    # ModelOpt uses [..., K, V]; both supported native backends store [..., V, K].
    return state.transpose(-1, -2).contiguous()


def prefill(q, k, v, g, beta, state, scale, normalize=False, gate_inputs=None, *, precision="vllm"):
    """Run native chunk prefill, including the selected profile's raw-gate preparation."""
    # Match BF16 projection outputs; A_log and bias retain their parameter precision.
    gdn, kda = _kernels(precision)
    kimi_k3 = precision == "vllm_kimi_k3"
    channel = g.ndim == 3
    if not channel and gate_inputs is not None:
        raw_g, raw_beta, rate, bias = gate_inputs
        q, k, v, g, beta = gdn.fused_post_conv_prep(
            torch.cat([x.flatten(1) for x in (q, k, v)], dim=-1),
            raw_g.flatten(0, -2).bfloat16().contiguous(),
            raw_beta.flatten(0, -2).bfloat16().contiguous(),
            rate.reshape(-1).contiguous(),
            bias.contiguous(),
            q.shape[-2],
            q.shape[-1],
            v.shape[-1],
            apply_l2norm=normalize,
        )
        normalize = False
    q, k, v, g, beta = [x.unsqueeze(0).contiguous() for x in (q, k, v, g, beta)]
    # Native chunk implementations may reuse V as their output buffer.
    v = v.clone()
    kwargs = {
        "scale": scale,
        "initial_state": _state_layout(state.unsqueeze(0)),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": normalize,
        "cu_seqlens": _prefill_metadata(q.device, q.shape[1]),
    }
    if channel and gate_inputs is not None:
        raw_g, raw_beta, rate, bias = gate_inputs
        # Kimi fuses beta activation; standalone KDA takes already activated beta.
        beta_arg = (
            {"raw_beta": raw_beta[None].bfloat16().contiguous()} if kimi_k3 else {"beta": beta}
        )
        out, final = kda.chunk_kda_with_fused_gate(
            q,
            k,
            v,
            raw_g=raw_g[None].bfloat16().contiguous(),
            A_log=rate.reshape(-1).contiguous(),
            g_bias=bias.contiguous() if bias is not None else None,
            **beta_arg,
            **kwargs,
        )
    elif channel:
        out, final = kda.chunk_kda(q, k, v, g, beta, **kwargs)
    else:
        out, final = gdn.chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    return out[0], _state_layout(final[0])


def step(q, k, v, g, beta, state, scale, normalize=False, gate_inputs=None, *, precision="vllm"):
    """Run native fused decode on a private cache, preserving the incoming autograd state."""
    gdn, kda = _kernels(precision)
    kimi_k3 = precision == "vllm_kimi_k3"
    channel = g.ndim == 2
    cu, indices = _decode_metadata(q.device)
    if channel or gate_inputs is not None:
        # Only slot one is read/written; slot zero is reserved and needs no initialization.
        native_state = state.new_empty(2, state.shape[0], state.shape[2], state.shape[1])
        native_state[1].copy_(state.transpose(-1, -2))
    else:
        native_state = _state_layout(state.unsqueeze(0))
    if kimi_k3:
        # The Kimi kernel always normalizes Q/K; validate_profile enforces normalize=True.
        raw_g, raw_beta, rate, bias = gate_inputs
        out, final = kda.fused_recurrent_kda_packed_decode(
            mixed_qkv=torch.cat([x.flatten() for x in (q, k, v)])[None],
            raw_g=raw_g[None, None].bfloat16().contiguous(),
            raw_beta=raw_beta[None, None].bfloat16().contiguous(),
            A_log=rate.reshape(-1).contiguous(),
            dt_bias=(torch.zeros_like(raw_g).flatten() if bias is None else bias.contiguous()),
            lower_bound=None,
            initial_state=native_state,
            state_indices=indices,
            scale=scale,
        )
    elif channel:
        out, final = kda.fused_recurrent_kda_fwd(
            *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
            initial_state=native_state,
            inplace_final_state=True,
            scale=scale,
            use_qk_l2norm_in_kernel=normalize,
            cu_seqlens=cu,
            ssm_state_indices=indices,
        )
    elif gate_inputs is not None:
        raw_g, raw_beta, rate, bias = gate_inputs
        out, final = gdn.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=torch.cat([x.flatten() for x in (q, k, v)])[None],
            a=raw_g[None].bfloat16().contiguous(),
            b=raw_beta[None].bfloat16().contiguous(),
            A_log=rate.reshape(-1).contiguous(),
            dt_bias=bias.contiguous(),
            scale=scale,
            initial_state=native_state,
            out=torch.empty_like(v, memory_format=torch.contiguous_format)[None, None],
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=normalize,
        )
    else:
        out, final = gdn.fused_recurrent_gated_delta_rule(
            *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
            initial_state=native_state,
            inplace_final_state=False,
            scale=scale,
            use_qk_l2norm_in_kernel=normalize,
            cu_seqlens=cu,
        )
    # Indexed kernels update slot 1; the non-in-place fallback returns one state.
    return out[0, 0], _state_layout(final[-1])


@lru_cache(maxsize=128)
def _prefill_metadata(device, length):
    # Native kernels only read these boundaries; reuse them across equal-length layers.
    return torch.arange(2, device=device, dtype=torch.int32) * length


@cache
def _decode_metadata(device):
    # Immutable single-token boundaries and cache slot; construct without a CPU-to-GPU copy.
    return torch.arange(2, device=device, dtype=torch.int32), torch.ones(
        1, device=device, dtype=torch.int32
    )


def validate_profile(gate_inputs, normalize, *, precision="vllm"):
    """Validate the selected native serving contract before launching prefill."""
    _kernels(precision)
    if precision == "vllm_kimi_k3":
        if gate_inputs is None:
            raise ValueError("vllm_kimi_k3 requires raw gate_inputs")
        if not normalize:
            raise ValueError("vLLM packed KDA decode requires Q/K normalization")
