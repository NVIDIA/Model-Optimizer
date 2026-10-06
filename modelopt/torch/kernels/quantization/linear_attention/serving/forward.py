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

"""vLLM forward primitives with FP32 saved values; autograd lives in quantization."""

import torch
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.fla.ops.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_scaled_dot_kkt_fwd,
    fused_recurrent_kda,
)
from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate as fused_kda_gate
from vllm.model_executor.layers.fla.ops.kda import recompute_w_u_fwd as kda_wu
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd as gdn_wu

from .chunk_delta_h import chunk_state


def prefill(q, k, v, g, beta, state, scale, normalize=False):
    """Return one sequence's output, final state, and rounded forward intermediates."""
    q, k, v, g, beta = [x.unsqueeze(0).contiguous() for x in (q, k, v, g, beta)]
    cu = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
    if normalize:
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    channel = g.ndim == 4
    gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    if channel:
        lower, scores = chunk_kda_scaled_dot_kkt_fwd(q, k, gc, beta, scale=scale, cu_seqlens=cu)
    else:
        lower = chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g=gc, cu_seqlens=cu, output_dtype=torch.float32
        )
        scores = None
    inverse = solve_tril(A=lower, cu_seqlens=cu, output_dtype=k.dtype)
    if channel:
        w, u, _, kg = kda_wu(k=k, v=v, beta=beta, A=inverse, gk=gc, cu_seqlens=cu)
    else:
        w, u = gdn_wu(k=k, v=v, beta=beta, A=inverse, g_cumsum=gc, cu_seqlens=cu)
        kg = k
    assert kg is not None
    h, updated, final = chunk_state(
        k=kg,
        w=w,
        u=u,
        g=None if channel else gc,
        gk=gc if channel else None,
        initial_state=state.unsqueeze(0).contiguous(),
        cu_seqlens=cu,
    )
    assert updated is not None and final is not None
    if channel:
        out = chunk_gla_fwd_o_gk(
            q=q,
            v=updated.to(v.dtype),
            g=gc,
            A=scores,
            h=h.to(k.dtype),
            scale=scale,
            o=torch.empty_like(v),
            cu_seqlens=cu,
            chunk_size=64,
        )
    else:
        out = chunk_fwd_o(
            q=q, k=k, v=updated.to(v.dtype), h=h.to(k.dtype), g=gc, scale=scale, cu_seqlens=cu
        )
    intermediates = {
        "q": q[0],
        "k": k[0],
        "g": gc[0],
        "lower": lower[0],
        "inverse": inverse[0],
        "w": w[0],
        "u": u[0],
        "h": h[0],
        "updated": updated[0],
        "kg": kg[0],
        "scores": None if scores is None else scores[0],
    }
    return out[0], final[0], intermediates


def step(q, k, v, g, beta, state, scale, normalize=False):
    """Run one native token update without modifying the incoming state."""
    recurrent = fused_recurrent_kda if g.ndim == 2 else fused_recurrent_gated_delta_rule
    out, final = recurrent(
        *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
        initial_state=state.unsqueeze(0).contiguous(),
        inplace_final_state=False,
        scale=scale,
        use_qk_l2norm_in_kernel=normalize,
        cu_seqlens=torch.tensor([0, 1], device=q.device, dtype=torch.int32),
        ssm_state_indices=torch.tensor([0], device=q.device, dtype=torch.int32),
    )
    return out[0, 0], final[0]


def fused_gdn_gating(*args, **kwargs):
    """Load the optional vLLM model only when Megatron needs GDN gate preparation."""
    from vllm.model_executor.models.qwen3_next import fused_gdn_gating as native_gate

    return native_gate(*args, **kwargs)
