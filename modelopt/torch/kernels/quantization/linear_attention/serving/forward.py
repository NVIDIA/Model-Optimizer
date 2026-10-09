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

"""Native vLLM Triton forwards; training owns private, differentiable state."""

from functools import cache, lru_cache

import torch

from ._compat import fla_module, gdn_module, kda_chunk, kda_module, optional_module, state_layout


def _gdn_inputs(q, k, v, gate_inputs, normalize):
    raw_g, raw_beta, a_log, bias = gate_inputs
    prep = gdn_module("fused_gdn_prefill_post_conv")
    if prep is not None:
        return prep.fused_post_conv_prep(
            torch.cat([x.flatten(1) for x in (q, k, v)], dim=-1),
            raw_g.flatten(0, -2),
            raw_beta.flatten(0, -2),
            a_log,
            bias,
            q.shape[-2],
            q.shape[-1],
            v.shape[-1],
            apply_l2norm=normalize,
        ), False
    g, beta = _gdn_gate()(
        a_log, raw_g.reshape(-1, raw_g.shape[-1]), raw_beta.reshape(-1, raw_beta.shape[-1]), bias
    )
    return (q, k, v, g[0], beta[0]), normalize


def prefill(q, k, v, g, beta, state, scale, normalize=False, gate_inputs=None, *, kimi_k3=False):
    """Run the complete native chunk entry point, including raw-gate preparation."""
    if kimi_k3 and gate_inputs is None:
        raise ValueError("vllm_kimi_k3 requires raw gate_inputs")
    channel = g.ndim == 3
    if not channel and gate_inputs is not None:
        (q, k, v, g, beta), normalize = _gdn_inputs(q, k, v, gate_inputs, normalize)
    q, k, v, g, beta = [x.unsqueeze(0).contiguous() for x in (q, k, v, g, beta)]
    # Some native chunk implementations reuse V as their output buffer.
    v = v.clone()
    kwargs = {
        "scale": scale,
        "initial_state": state_layout(
            state.unsqueeze(0), kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
        ),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": normalize,
        "cu_seqlens": _prefill_metadata(q.device, q.shape[1]),
    }
    if channel:
        gates = None
        if gate_inputs is not None:
            raw_g, raw_beta, rate, bias = gate_inputs
            gates = (raw_g[None].contiguous(), raw_beta[None].contiguous(), rate, bias)
        out, final = kda_chunk(q, k, v, g, beta, gate_inputs=gates, kimi_k3=kimi_k3, **kwargs)
    else:
        out, final = fla_module("chunk").chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    return out[0], state_layout(
        final[0], kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
    )


def step(q, k, v, g, beta, state, scale, normalize=False, gate_inputs=None, *, kimi_k3=False):
    """Run native fused decode on a private cache; never mutate the incoming state."""
    if kimi_k3 and gate_inputs is None:
        raise ValueError("vllm_kimi_k3 requires raw gate_inputs")
    channel = g.ndim == 2
    cu, indices = _decode_metadata(q.device)
    native_state = state_layout(
        state.unsqueeze(0), kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
    )
    if gate_inputs is not None:
        raw_g, raw_beta, a_log, bias = gate_inputs
        if channel and kimi_k3:
            if not normalize:
                raise ValueError("vLLM packed KDA decode requires Q/K normalization")
            # Slot zero is reserved by vLLM's paged-cache kernels.
            native_state = torch.cat((torch.zeros_like(native_state), native_state), dim=0)
            out, final = kda_module(kimi_k3).fused_recurrent_kda_packed_decode(
                mixed_qkv=torch.cat([x.flatten() for x in (q, k, v)])[None],
                raw_g=raw_g[None, None].contiguous(),
                raw_beta=raw_beta[None, None].contiguous(),
                A_log=a_log.reshape(-1).contiguous(),
                dt_bias=(torch.zeros_like(raw_g).flatten() if bias is None else bias.contiguous()),
                lower_bound=None,
                initial_state=native_state,
                state_indices=indices,
                scale=scale,
            )
            return out[0, 0], state_layout(
                final[1], kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
            )
        packed = (
            None
            if channel
            else getattr(
                fla_module("fused_recurrent"),
                "fused_recurrent_gated_delta_rule_packed_decode",
                None,
            )
        )
        if packed is not None:
            # Match vLLM's pure, non-speculative decode path on private cache slot one.
            native_state = torch.cat((torch.zeros_like(native_state), native_state), dim=0)
            out, final = packed(
                mixed_qkv=torch.cat([x.flatten() for x in (q, k, v)])[None],
                a=raw_g[None].contiguous(),
                b=raw_beta[None].contiguous(),
                A_log=a_log.reshape(-1).contiguous(),
                dt_bias=bias.contiguous(),
                scale=scale,
                initial_state=native_state,
                out=torch.empty_like(v)[None, None],
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=normalize,
            )
            return out[0, 0], state_layout(
                final[1], kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
            )
        fused = None if channel else gdn_module("fused_sigmoid_gating")
        if fused is not None:
            out, final = fused.fused_sigmoid_gating_delta_rule_update(
                A_log=a_log,
                a=raw_g[None].contiguous(),
                b=raw_beta[None].contiguous(),
                dt_bias=bias,
                q=q[None, None],
                k=k[None, None],
                v=v[None, None],
                initial_state=native_state,
                inplace_final_state=False,
                scale=scale,
                use_qk_l2norm_in_kernel=normalize,
                cu_seqlens=cu,
            )
            return out[0, 0], state_layout(
                final[0], kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
            )
        if not channel:
            g, beta = _gdn_gate()(
                a_log, raw_g[None].contiguous(), raw_beta[None].contiguous(), bias
            )
            g, beta = g[0, 0], beta[0, 0]
    # Older vLLM models and standalone calls consume already activated gates.
    if channel:
        native_state = torch.cat((torch.zeros_like(native_state), native_state), dim=0)
        out, final = kda_module(kimi_k3).fused_recurrent_kda_fwd(
            *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
            initial_state=native_state,
            inplace_final_state=True,
            scale=scale,
            use_qk_l2norm_in_kernel=normalize,
            cu_seqlens=cu,
            ssm_state_indices=indices,
        )
        return out[0, 0], state_layout(
            final[1], kda=True, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
        )
    recurrent = fla_module("fused_recurrent").fused_recurrent_gated_delta_rule
    out, final = recurrent(
        *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
        initial_state=native_state,
        inplace_final_state=False,
        scale=scale,
        use_qk_l2norm_in_kernel=normalize,
        cu_seqlens=cu,
        ssm_state_indices=None,
    )
    return out[0, 0], state_layout(
        final[0], kda=channel, kimi_k3=kimi_k3, raw_gates=gate_inputs is not None
    )


# Old runtimes keep gate preparation beside the model rather than in the FLA package.
@cache
def _gdn_gate():
    for path in (
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "vllm.model_executor.models.qwen3_next",
    ):
        module = optional_module(path)
        if module is not None and hasattr(module, "fused_gdn_gating"):
            return module.fused_gdn_gating
    raise NotImplementedError("The installed vLLM does not expose supported GDN gate preparation")


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


def validate_profile(gate_inputs, normalize, *, kimi_k3=False):
    """Validate the selected native serving contract before launching prefill."""
    if kimi_k3:
        if gate_inputs is None:
            raise ValueError("vllm_kimi_k3 requires raw gate_inputs")
        if not normalize:
            raise ValueError("vLLM packed KDA decode requires Q/K normalization")
