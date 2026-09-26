# Adapted from: https://github.com/fla-org/flash-linear-attention/blob/516143e31fce/fla/ops/gated_delta_rule/chunk.py
# Adapted with modifications (marked [ModelOpt]): threads state_qdq / state_qdq_block_v through
# the autograd function, applies an optional w_quantizer to the WY tensor w, and imports the
# state kernels from the vendored sibling module.
#
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

import warnings

import fla
import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
    expand_h0,
)
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_chunk_cumsum
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import (
    IS_NVIDIA_HOPPER,
    TRITON_ABOVE_3_4_0,
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)

from modelopt.torch.quantization.linear_attention.utils import validate_gdn_quantizer
from modelopt.torch.quantization.nn import TensorQuantizer

from .fla_chunk_delta_h import (
    STATE_QDQ_FP8_DYNAMIC,
    STATE_QDQ_OFF,
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_qdq: int = STATE_QDQ_OFF,
    state_qdq_block_v: int | None = None,
    w_quantizer: TensorQuantizer | None = None,
):
    g_input = g if use_gate_in_kernel else None
    if use_gate_in_kernel:
        g = gdn_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    # [ModelOpt] w is an activation (the WY form of the chunk's keys) that lands in memory here,
    # so it is fake-quantized once per forward instead of tile by tile inside the kernel.
    if w_quantizer is not None:
        w = w_quantizer(w)

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        state_qdq=state_qdq,
        state_qdq_block_v=state_qdq_block_v,
    )

    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state, initial_state, g_input, w if w_quantizer is not None else None


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    g_input: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_qdq: int = STATE_QDQ_OFF,
    state_qdq_block_v: int | None = None,
    quantized_w: torch.Tensor | None = None,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    # [ModelOpt] Reuse forward QDQ exactly; the validated policy uses identity STE.
    # This also avoids calling observers or dynamic scale computation in backward.
    if quantized_w is not None:
        w = quantized_w

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

    # [ModelOpt] The backward recomputes the forward's chunk states, so it sees the same
    # fake-quantized states; the state gradient itself passes straight through the QDQ.
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
        state_qdq=state_qdq,
        state_qdq_block_v=state_qdq_block_v,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    if cp_context is not None:
        # initial_state is None in the CP mode
        # We only need to compute dht of current rank and pass it to the backward kernel
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        chunk_size=chunk_size,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    dA_log, ddt_bias = None, None
    if use_gate_in_kernel:
        dg, dA_log, ddt_bias = gdn_gate_bwd(g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg)
    return dq, dk, dv, db, dg, dh0, dA_log, ddt_bias


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        cp_context: FLACPContext | None = None,
        chunk_size: int = 64,
        state_qdq: int = STATE_QDQ_OFF,
        state_qdq_block_v: int | None = None,
        w_quantizer: TensorQuantizer | None = None,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu
            )
        g, o, A, final_state, initial_state, g_input, quantized_w = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            state_qdq=state_qdq,
            state_qdq_block_v=state_qdq_block_v,
            w_quantizer=w_quantizer,
        )
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta_raw,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            g_input,
            A_log,
            dt_bias,
            quantized_w,
        )
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.cp_context = cp_context
        ctx.state_v_first = state_v_first
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.state_qdq = state_qdq
        ctx.state_qdq_block_v = state_qdq_block_v
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta_raw,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            g_input,
            A_log,
            dt_bias,
            quantized_w,
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            cp_context=ctx.cp_context,
            chunk_indices=chunk_indices,
            state_v_first=ctx.state_v_first,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            g_input=g_input,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=ctx.chunk_size,
            state_qdq=ctx.state_qdq,
            state_qdq_block_v=ctx.state_qdq_block_v,
            quantized_w=quantized_w,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta_raw),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dA_log,
            ddt_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# [ModelOpt] Not registered with fla's backend dispatch: another backend must not take over a
# call that asks for state quantization.
@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA (Grouped Value Attention) is applied if `HV > H`, where `HV` must be divisible by `H`.
        g (torch.Tensor):
            (forget) gating tensor of shape `[B, T, HV]`.
            When `use_gate_in_kernel=False` (default), `g` should be in log space (pre-computed decay).
            When `use_gate_in_kernel=True`, `g` is the raw input before gate activation;
            the kernel fuses `-exp(A_log) * softplus(g + dt_bias)` + chunk cumsum internally.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space GDN decay internally.
            When `True`, the passed `g` is the raw input, and `A_log` must be provided.
            The kernel fuses gate activation + chunk cumsum in a single pass.
            Default: `False`.
        A_log (Optional[torch.Tensor]):
            Decay parameter of shape `[HV]`. Required when `use_gate_in_kernel=True`.
        dt_bias (Optional[torch.Tensor]):
            Bias added to `g` before activation, of shape `[HV]`.
            Only used when `use_gate_in_kernel=True`.
        use_beta_sigmoid_in_kernel (bool):
            Whether to apply `torch.sigmoid(beta)` before launching the chunk kernel.
            - If `True`, the passed `beta` acts as the raw beta logits.
            - If `False`, `beta` is expected to already be in post-sigmoid space.
            Default: `False`.
        allow_neg_eigval (bool):
            Whether to allow negative eigenvalues by scaling `beta` to `[0, 2)`.
            Only takes effect together with `use_beta_sigmoid_in_kernel=True`, in which case
            the kernel computes `2 * sigmoid(beta)` instead of `sigmoid(beta)`. Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first ``[V, K]`` layout instead of the default ``[K, V]``. Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        chunk_indices (Optional[torch.LongTensor]):
            Pre-computed chunk indices for variable-length inputs.
            If provided, they are used directly instead of being computed from `cu_seqlens`. Default: `None`.
        cp_context (Optional[FLACPContext]):
            Context parallel context for distributed training across multiple devices.
            When provided, `initial_state` and `output_final_state` are not supported,
            and `cu_seqlens` will be overridden by the context. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, HV, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if fla.__version__ != "0.5.1":
        raise RuntimeError(f"ModelOpt GDN requires fla-core==0.5.1, got {fla.__version__}.")
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError(
                "Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`."
            )
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")

    # Validate head dimensions
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    H, HV = q.shape[2], v.shape[2]
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by "
            f"num_heads (H={H}), but got HV % H = {HV % H}"
        )

    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size != 64:
        raise ValueError("ModelOpt GDN supports only chunk_size=64; FLA WY backward assumes 64.")

    # [ModelOpt] state_qdq: 0 keeps fla's numerics; 1 fake-quantizes the state carried between
    # chunks to FP8 E4M3 with a dynamic scale per [K, state_qdq_block_v] tile of each head.
    state_qdq = kwargs.pop("state_qdq", STATE_QDQ_OFF)
    state_qdq_block_v = kwargs.pop("state_qdq_block_v", None)
    # w_quantizer: dynamic FP8 TensorQuantizer applied to the WY tensor
    # ``w`` of shape [B, T, HV, K] before it multiplies the state, emulating an FP8 x FP8 matmul.
    w_quantizer = kwargs.pop("w_quantizer", None)
    if state_qdq not in (STATE_QDQ_OFF, STATE_QDQ_FP8_DYNAMIC):
        raise ValueError(f"`state_qdq` must be 0 or 1, got {state_qdq}.")
    if w_quantizer is not None:
        validate_gdn_quantizer(w_quantizer, name="gdn_w_quantizer")
    if state_qdq and (not q.is_cuda or torch.cuda.get_device_capability(q.device) < (8, 9)):
        raise RuntimeError("GDN state QDQ requires native E4M3 conversion on CUDA SM89 or newer.")
    if (state_qdq != STATE_QDQ_OFF or w_quantizer is not None) and cp_context is not None:
        raise ValueError("State or w quantization is not supported together with `cp_context`.")

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    use_gate_in_kernel = kwargs.get("use_gate_in_kernel", False)
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    if scale is None:
        scale = k.shape[-1] ** -0.5
    # [ModelOpt] Hopper's TileLang backward needs BF16 and equal head counts. Expand outside
    # custom autograd so repeat_interleave reduces q/k gradients back to the original heads.
    if IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0:
        if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
            raise ValueError("Hopper with Triton >= 3.4 requires BF16 q/k/v for GDN training.")
        if H != HV:
            q = q.repeat_interleave(HV // H, dim=2)
            k = k.repeat_interleave(HV // H, dim=2)
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        cp_context,
        chunk_size,
        state_qdq,
        state_qdq_block_v,
        w_quantizer,
    )
    return o, final_state


chunk_gdn = chunk_gated_delta_rule
