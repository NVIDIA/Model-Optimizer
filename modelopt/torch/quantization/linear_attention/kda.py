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

"""Differentiable KDA prefill with stable per-channel decay interactions."""

import torch
import torch.nn.functional as F

from .training import _prefill_decode_forward, _prepare_prefill_inputs
from .utils import forward_value

__all__ = ["matmul_kda"]


def matmul_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    policy,
    state_qdq=False,
    state_format="fp8_e4m3",
    state_quantizer=None,
    replay_key_quantizer=None,
    replay_update_quantizer=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    A_log=None,  # noqa: N803 - match the FLA kernel signature
    dt_bias=None,
    safe_gate=False,
    lower_bound=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    state_v_first=False,
    chunk_size=64,
    cp_context=None,
    disable_recompute=False,
    return_intermediate_states=False,
    prefill_lengths=None,
):
    """Normalize KDA inputs and run a chunked prefix plus configured suffix recurrence.

    Gates follow FLA's kernel activation formula and per-key log retention.
    The serving precision profile supplies matching native forward values
    and a Torch adjoint through the rounded state trajectory.
    """
    if cp_context is not None or disable_recompute or return_intermediate_states:
        raise NotImplementedError(
            "KDA matmul does not support CP or FLA recompute/intermediate flags"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if lower_bound is not None or safe_gate:
        raise ValueError("Serving arithmetic uses the native softplus KDA gate")
    output_dtype = q.dtype
    beta_dtype = beta.dtype
    q, k, v, g, beta = _prepare_prefill_inputs(
        q, k, v, g, beta, policy=policy, chunk_size=chunk_size
    )
    dtype = q.dtype
    if g.ndim != 4:
        raise ValueError("matmul_kda requires per-key-channel KDA log gates")
    if use_gate_in_kernel:
        raw_gate = g
        if A_log is None:
            raise ValueError("Fused KDA gate requires A_log")
        if dt_bias is not None:
            g = g + dt_bias.to(dtype).reshape(g.shape[-2:])
        rate = A_log.to(dtype).exp().reshape(g.shape[-2], 1)
        g = -rate * F.softplus(g)
        # Import the optional vLLM backend only for the native precision profile.
        from ...kernels.quantization.linear_attention.serving.forward import fused_kda_gate

        with torch.no_grad():
            native_gate = fused_kda_gate(
                raw_gate.flatten(-2).contiguous(), A_log, raw_gate.shape[-1], g_bias=dt_bias
            )
        g = forward_value(g, native_gate)
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    return _prefill_decode_forward(
        q,
        k,
        v,
        g,
        beta,
        policy=policy,
        state_qdq=state_qdq,
        state_format=state_format,
        state_quantizer=state_quantizer,
        replay_key_quantizer=replay_key_quantizer,
        replay_update_quantizer=replay_update_quantizer,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        state_v_first=state_v_first,
        output_dtype=output_dtype,
        beta_dtype=beta_dtype,
        prefill_lengths=prefill_lengths,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
