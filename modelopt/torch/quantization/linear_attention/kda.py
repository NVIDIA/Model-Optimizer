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

from .decode_prefill import _decode_prefill

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
    """Normalize KDA inputs and run exact prefix plus configured suffix recurrence.

    Gates use the loaded FLA model's activation formula and per-key log retention.
    """
    if policy.backend != "matmul" or chunk_size != policy.chunk_size:
        raise ValueError("matmul_kda requires backend='matmul' and its configured chunk size")
    if cp_context is not None or disable_recompute or return_intermediate_states:
        raise NotImplementedError(
            "KDA matmul does not support CP or FLA recompute/intermediate flags"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if safe_gate and use_gate_in_kernel and (lower_bound is None or not -5 <= lower_bound < 0):
        raise ValueError("safe_gate requires a lower_bound in [-5, 0)")
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError("lower_bound must be negative")
    output_dtype = q.dtype
    dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    if g.ndim != 4:
        raise ValueError("matmul_kda requires per-key-channel KDA log gates")
    if use_qk_l2norm_in_kernel:
        q, k = (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt() for x in (q, k))
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("Fused KDA gate requires A_log")
        if dt_bias is not None:
            g = g + dt_bias.to(dtype).reshape(g.shape[-2:])
        rate = A_log.to(dtype).exp().reshape(g.shape[-2], 1)
        g = -rate * F.softplus(g) if lower_bound is None else lower_bound * (rate * g).sigmoid()
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    if policy.decode is not None:
        return _decode_prefill(
            q,
            k,
            v,
            g,
            beta,
            policy=policy,
            state_qdq=state_qdq,
            state_format=state_format,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            state_v_first=state_v_first,
            output_dtype=output_dtype,
            prefill_lengths=prefill_lengths,
        )
    raise ValueError("An explicit decode policy is required")
