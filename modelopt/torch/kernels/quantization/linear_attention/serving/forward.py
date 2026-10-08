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

from functools import cache
from importlib import import_module

import torch

from ._compat import fla_module, state_layout
from .chunk_delta_h import chunk_state

chunk_fwd_o = fla_module("chunk_o").chunk_fwd_o
chunk_scaled_dot_kkt_fwd = fla_module("chunk_scaled_dot_kkt").chunk_scaled_dot_kkt_fwd
chunk_local_cumsum = fla_module("cumsum").chunk_local_cumsum
fused_recurrent_gated_delta_rule = fla_module("fused_recurrent").fused_recurrent_gated_delta_rule
_kda = fla_module("kda")
chunk_gla_fwd_o_gk = _kda.chunk_gla_fwd_o_gk
chunk_kda_scaled_dot_kkt_fwd = _kda.chunk_kda_scaled_dot_kkt_fwd
fused_recurrent_kda = _kda.fused_recurrent_kda
fused_kda_gate = _kda.fused_kda_gate
kda_wu = _kda.recompute_w_u_fwd
l2norm_fwd = fla_module("l2norm").l2norm_fwd
solve_tril = fla_module("solve_tril").solve_tril
gdn_wu = fla_module("wy_fast").recompute_w_u_fwd
_KDA_GATE_SCALE = getattr(_kda, "RCP_LN2", 1.0)


def prefill(q, k, v, g, beta, state, scale, normalize=False):
    """Return one sequence's output, final state, and rounded forward intermediates."""
    q, k, v, g, beta = [x.unsqueeze(0).contiguous() for x in (q, k, v, g, beta)]
    cu = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
    if normalize:
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    channel = g.ndim == 4
    gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    # Newer native KDA kernels evaluate exp2 of base-2 cumulative log gates.
    natural_gc = gc
    if channel:
        gc = gc * _KDA_GATE_SCALE
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
        initial_state=state_layout(state.unsqueeze(0)),
        cu_seqlens=cu,
        use_exp2=channel and _KDA_GATE_SCALE != 1.0,
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
        "g": natural_gc[0],
        "lower": lower[0],
        "inverse": inverse[0],
        "w": w[0],
        "u": u[0],
        "h": state_layout(h[0]),
        "updated": updated[0],
        "kg": kg[0],
        "scores": None if scores is None else scores[0],
    }
    return out[0], state_layout(final[0]), intermediates


def step(q, k, v, g, beta, state, scale, normalize=False):
    """Run one native token update without modifying the incoming state."""
    recurrent = fused_recurrent_kda if g.ndim == 2 else fused_recurrent_gated_delta_rule
    out, final = recurrent(
        *[x[None, None].contiguous() for x in (q, k, v, g, beta)],
        initial_state=state_layout(state.unsqueeze(0)),
        inplace_final_state=False,
        scale=scale,
        use_qk_l2norm_in_kernel=normalize,
        cu_seqlens=torch.tensor([0, 1], device=q.device, dtype=torch.int32),
        # Private dense training state needs no paged-cache index (0 is now reserved).
        ssm_state_indices=None,
    )
    return out[0, 0], state_layout(final[0])


def fused_gdn_gating(*args, **kwargs):
    """Load the optional vLLM model only when Megatron needs GDN gate preparation."""
    return _gdn_gate()(*args, **kwargs)


# Megatron compiles gate preparation; module discovery must execute outside that graph.
@torch.compiler.disable
@cache
def _gdn_gate():
    for path in (
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "vllm.model_executor.models.qwen3_next",
    ):
        try:
            return import_module(path).fused_gdn_gating
        except ModuleNotFoundError as error:  # noqa: PERF203 - cached, one-time import discovery
            if error.name is None or not (path == error.name or path.startswith(error.name + ".")):
                raise
    raise ImportError("The installed vLLM does not provide fused_gdn_gating")
