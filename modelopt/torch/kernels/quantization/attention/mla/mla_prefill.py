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

"""Varlen MLA prefill attention with fused Q/K/P/V fake quantization.

MLA prefill runs standard multi-head attention over the up-projected per-head
K/V, but with asymmetric head dims: Q and K carry ``qk_nope_head_dim +
qk_rope_head_dim`` features (192 for DeepSeek-family models) while V and the
output carry ``v_head_dim`` (128). The kernel keeps two independent feature
axes so no V padding is required.

Quantization follows the ModelOpt conventions: NVFP4 uses 1x16 blocks along
the BMM contraction axis (Q/K along the feature axis, P/V along the key/token
axis) with E4M3 block scales and a per-tensor global scale ``amax / (6*448)``;
FP8 is per-tensor E4M3 with scale ``amax / 448``. The softmax denominator
accumulates the unquantized P; only the quantized P feeds ``P @ V``. NVFP4
operands run their dots on FP32 carriers with IEEE precision. RoPE feature
dims of Q/K participate in the same single quantization pass (no PE
special-casing).

Causal masking treats Q as the suffix of the KV span (``k_len >= q_len``).
Cached-context chunks run with ``causal=False`` and ``return_lse=True``; the
caller (e.g. vLLM's MLA layer) merges chunk states via ``merge_attn_states``
using the returned natural-log LSE. Inference-only: no autograd support.
"""

import math

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.common.attention.triton_fa import LOG2E, _apply_mask
from modelopt.torch.kernels.quantization.attention.bmm2_qdq import (
    _a_qdq_nvfp4,
    _p_qdq_nvfp4,
    _v_qdq_nvfp4,
)
from modelopt.torch.kernels.quantization.common.fp8_quant import fp8_scalar_qdq
from modelopt.torch.kernels.sparsity.attention.skip_softmax_helpers import (
    _apply_sparse_nm_to_qk_tile,
)

__all__ = ["mla_prefill_attention"]

# Referenced inside @triton.jit code, so it must be a tl.constexpr global.
LN2 = tl.constexpr(0.6931471805599453)

# Maps public operand quant options to kernel constexpr values.
_OPERAND_MODES = {None: 0, "fp8": 1, "nvfp4": 2}


def _operand_scale(mode: int, amax: float | None, name: str) -> float:
    """Convert a per-tensor amax to the kernel scale for one operand."""
    if mode == 0 or amax is None:
        return 1.0
    if not (math.isfinite(amax) and amax > 0):
        raise ValueError(f"{name} must be a finite positive value, got {amax}")
    return amax / 448.0 if mode == 1 else amax / (6.0 * 448.0)


def _resolve_mode(mode: str | None, name: str) -> int:
    if mode not in _OPERAND_MODES:
        raise ValueError(
            f"{name} must be one of {sorted(m for m in _OPERAND_MODES if m)} or None, got {mode!r}"
        )
    return _OPERAND_MODES[mode]


@triton.jit
def _qdq_a(x, MODE: tl.constexpr, scale, M: tl.constexpr, K: tl.constexpr):
    """A-side (Q) operand QDQ: FP8 per-tensor or NVFP4 1x16 along K (axis 1)."""
    if MODE == 1:
        x = fp8_scalar_qdq(x, scale).to(x.dtype)
    elif MODE == 2:
        x = _a_qdq_nvfp4(x, scale, M, K)
    return x


@triton.jit
def _qdq_b(x, MODE: tl.constexpr, scale, K: tl.constexpr, N: tl.constexpr):
    """B-side (K^T / V) operand QDQ: FP8 per-tensor or NVFP4 1x16 along K (axis 0)."""
    if MODE == 1:
        x = fp8_scalar_qdq(x, scale).to(x.dtype)
    elif MODE == 2:
        x = _v_qdq_nvfp4(x, scale, K, N)
    return x


@triton.jit
def _apply_sparse_nm_dense_tokens(
    scores,
    kv_start,
    q_pos,
    kv_pos,
    seq_len_q,
    seq_len_kv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSITY_N: tl.constexpr,
    SPARSITY_M: tl.constexpr,
    DENSE_SINK_TOKENS: tl.constexpr,
    DENSE_RECENT_TOKENS: tl.constexpr,
):
    """Apply N:M sparsity outside token-exact sink and recent regions.

    Same semantics as ``triton_fa._apply_sparse_nm_with_dense_tokens`` (Q is
    the suffix of the KV span), duplicated here because that helper resolves
    the sparsity primitive through lazily-populated module globals.
    """
    sparse_scores = _apply_sparse_nm_to_qk_tile(scores, BLOCK_M, BLOCK_N, SPARSITY_N, SPARSITY_M)
    q_abs_pos = q_pos[:, None] + seq_len_kv - seq_len_q
    kv_abs_pos = kv_start + kv_pos[None, :]
    token_distance = q_abs_pos - kv_abs_pos
    dense_tokens = (
        (seq_len_q <= 1)
        | (kv_abs_pos < DENSE_SINK_TOKENS)
        | ((token_distance >= 0) & (token_distance < DENSE_RECENT_TOKENS))
    )
    return tl.where(dense_tokens, scores, sparse_scores)


@triton.jit
def _mla_prefill_kernel(
    Q,  # [total_q, num_heads, LQK]
    K,  # [total_k, num_kv_heads, LQK]
    V,  # [total_k, num_kv_heads, LV]
    Out,  # [total_q, num_heads, LV]
    Lse,  # [num_heads, total_q] natural-log LSE (dummy when not RETURN_LSE)
    qk_scale,  # softmax_scale * log2(e)
    Cu_seqlens_q,  # [batch + 1]
    Cu_seqlens_k,  # [batch + 1]
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_h,
    stride_lse_s,
    q_scale,  # runtime per-tensor scales (host-converted from amax)
    k_scale,
    p_scale,
    v_scale,
    kv_group_num: tl.constexpr,  # num_heads // num_kv_heads
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # next_power_of_2(LQK)
    BLOCK_DV: tl.constexpr,  # next_power_of_2(LV)
    LQK: tl.constexpr,
    LV: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    Q_QUANT: tl.constexpr,  # 0=off, 1=FP8 E4M3, 2=NVFP4
    K_QUANT: tl.constexpr,
    P_QUANT: tl.constexpr,
    V_QUANT: tl.constexpr,
    IEEE_QK: tl.constexpr,  # FP32 carriers + IEEE dot for BMM1
    IEEE_PV: tl.constexpr,  # FP32 carriers + IEEE dot for BMM2
    SPARSITY_N: tl.constexpr = 0,  # N:M sparsity - keep top-N of every M (0 = off)
    SPARSITY_M: tl.constexpr = 4,
    DENSE_SINK_TOKENS: tl.constexpr = 0,
    DENSE_RECENT_TOKENS: tl.constexpr = 64,
):
    # --- Grid: (batch, num_heads, num_q_tiles) ---
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    tile_q = tl.program_id(2)
    kv_head_idx = head_idx // kv_group_num

    q_start = tl.load(Cu_seqlens_q + batch_idx)
    q_end = tl.load(Cu_seqlens_q + batch_idx + 1)
    k_start = tl.load(Cu_seqlens_k + batch_idx)
    k_end = tl.load(Cu_seqlens_k + batch_idx + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    if tile_q * BLOCK_M >= q_len:
        return

    q_pos = tile_q * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_pos = tl.arange(0, BLOCK_N)
    offs_dqk = tl.arange(0, BLOCK_DQK)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dqk = offs_dqk < LQK
    mask_dv = offs_dv < LV
    q_mask = q_pos < q_len

    # --- Load Q tile [BLOCK_M, BLOCK_DQK]: stays in registers for the KV loop ---
    q_ptrs = (q_start + q_pos)[:, None] * stride_qbs + head_idx * stride_qh + offs_dqk[None, :]
    q = tl.load(Q + q_ptrs, mask=q_mask[:, None] & mask_dqk[None, :], other=0.0)
    if IEEE_QK:
        q = q.to(tl.float32)
    q = _qdq_a(q, Q_QUANT, q_scale, BLOCK_M, BLOCK_DQK)

    # --- Online softmax state ---
    row_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    # Causal: Q is the suffix of the KV span (k_len >= q_len).
    causal_offset = k_len - q_len
    kv_bound = k_len if not IS_CAUSAL else tl.minimum(causal_offset + (tile_q + 1) * BLOCK_M, k_len)

    for kv_start in range(0, kv_bound, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)
        kv_valid = (kv_start + kv_pos) < k_len

        # K^T tile [BLOCK_DQK, BLOCK_N]
        k_ptrs = (
            (k_start + kv_start + kv_pos)[None, :] * stride_kbs
            + kv_head_idx * stride_kh
            + offs_dqk[:, None]
        )
        k = tl.load(K + k_ptrs, mask=kv_valid[None, :] & mask_dqk[:, None], other=0.0)
        if IEEE_QK:
            k = k.to(tl.float32)
        k = _qdq_b(k, K_QUANT, k_scale, BLOCK_DQK, BLOCK_N)

        if IEEE_QK:
            scores = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            scores = tl.dot(q, k) * qk_scale
        scores = _apply_mask(scores, q_pos, kv_pos, q_len, k_len, kv_start, IS_CAUSAL)

        if SPARSITY_N > 0:
            scores = _apply_sparse_nm_dense_tokens(
                scores,
                kv_start,
                q_pos,
                kv_pos,
                q_len,
                k_len,
                BLOCK_M,
                BLOCK_N,
                SPARSITY_N,
                SPARSITY_M,
                DENSE_SINK_TOKENS,
                DENSE_RECENT_TOKENS,
            )

        # --- Online softmax update: the denominator uses unquantized p ---
        m_new = tl.maximum(row_max, tl.max(scores, 1))
        p = tl.math.exp2(scores - m_new[:, None])
        l_new = tl.sum(p, 1)
        correction = tl.math.exp2(row_max - m_new)
        row_sum = row_sum * correction + l_new
        acc = acc * correction[:, None]

        if P_QUANT == 1:
            p = fp8_scalar_qdq(p, p_scale)
        elif P_QUANT == 2:
            # Native packing consumes the model dtype; the QDQ value stays FP32.
            p = p.to(V.dtype.element_ty).to(tl.float32)
            p = _p_qdq_nvfp4(p, p_scale, BLOCK_M, BLOCK_N)

        # V tile [BLOCK_N, BLOCK_DV]
        v_ptrs = (
            (k_start + kv_start + kv_pos)[:, None] * stride_vbs
            + kv_head_idx * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(V + v_ptrs, mask=kv_valid[:, None] & mask_dv[None, :], other=0.0)
        if IEEE_PV:
            v = v.to(tl.float32)
        v = _qdq_b(v, V_QUANT, v_scale, BLOCK_N, BLOCK_DV)

        if IEEE_PV:
            acc = tl.dot(p.to(tl.float32), v, acc, input_precision="ieee")
        else:
            acc = tl.dot(p.to(v.dtype), v, acc)
        row_max = m_new

    # Clamp the denominator: empty context chunks (k_len == 0) leave acc at 0.
    acc = acc / tl.maximum(row_sum[:, None], 1e-6)

    if RETURN_LSE:
        lse = LN2 * (row_max + tl.math.log2(row_sum))
        lse = tl.where(row_sum == 0.0, float("-inf"), lse)
        lse_ptrs = head_idx * stride_lse_h + (q_start + q_pos) * stride_lse_s
        tl.store(Lse + lse_ptrs, lse, mask=q_mask)

    o_ptrs = (q_start + q_pos)[:, None] * stride_obs + head_idx * stride_oh + offs_dv[None, :]
    tl.store(Out + o_ptrs, acc, mask=q_mask[:, None] & mask_dv[None, :])


def mla_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float | None = None,
    causal: bool = True,
    return_lse: bool = False,
    *,
    q_quant: str | None = None,
    k_quant: str | None = None,
    p_quant: str | None = None,
    v_quant: str | None = None,
    q_amax: float | None = None,
    k_amax: float | None = None,
    p_amax: float = 1.0,
    v_amax: float | None = None,
    sparsity_n: int = 0,
    sparsity_m: int = 4,
    dense_sink_tokens: int = 0,
    dense_recent_tokens: int = 64,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Varlen MLA prefill attention with fused Q/K/P/V fake quantization.

    Args:
        q: ``[total_q, num_heads, qk_head_dim]`` packed queries.
        k: ``[total_k, num_kv_heads, qk_head_dim]`` packed keys (NOPE ++ RoPE).
        v: ``[total_k, num_kv_heads, v_head_dim]`` packed values.
        cu_seqlens_q: ``[batch + 1]`` cumulative Q sequence lengths.
        cu_seqlens_k: ``[batch + 1]`` cumulative K/V sequence lengths.
        max_seqlen_q: Maximum Q sequence length (grid sizing).
        softmax_scale: Scale factor (default ``qk_head_dim ** -0.5``).
        causal: Causal masking; Q is treated as the suffix of the KV span.
            Cached-context chunks use ``causal=False``.
        return_lse: Also return the natural-log LSE ``[num_heads, total_q]``
            for chunk-state merging (``merge_attn_states``).
        q_quant: Q fake quant-dequant: ``None``, ``"fp8"`` (per-tensor E4M3),
            or ``"nvfp4"`` (1x16 blocks along the feature/contraction axis).
        k_quant: K fake quant-dequant, same modes; blocks along features.
        p_quant: Softmax-P fake quant-dequant, blocks along the key axis.
            The softmax denominator stays unquantized.
        v_quant: V fake quant-dequant, blocks along the key/token axis.
        q_amax: Per-tensor amax for Q (``None`` = scale 1.0).
        k_amax: Per-tensor amax for K (``None`` = scale 1.0).
        p_amax: Per-tensor amax for P; defaults to 1.0, the theoretical upper
            bound of the unnormalized P's amax.
        v_amax: Per-tensor amax for V (``None`` = scale 1.0).
        sparsity_n: N:M score sparsity along the key axis (0 = off).
        sparsity_m: N:M group size (4 or 8).
        dense_sink_tokens: Leading KV tokens kept dense (token-exact).
        dense_recent_tokens: Recent KV tokens kept dense (token-exact).
        block_m: Q tile size (multiple of 16).
        block_n: KV tile size (multiple of 16).

    Returns:
        Output ``[total_q, num_heads, v_head_dim]`` in ``v.dtype``; with
        ``return_lse`` a tuple ``(out, lse)``.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be [tokens, heads, head_dim] tensors")
    total_q, num_heads, lqk = q.shape
    total_k, num_kv_heads, lv = v.shape[0], v.shape[1], v.shape[2]
    if k.shape != (total_k, num_kv_heads, lqk):
        raise ValueError(f"k shape {tuple(k.shape)} != ({total_k}, {num_kv_heads}, {lqk})")
    if num_heads % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    if block_m % 16 or block_n % 16:
        raise ValueError("block_m and block_n must be multiples of 16")

    q_mode = _resolve_mode(q_quant, "q_quant")
    k_mode = _resolve_mode(k_quant, "k_quant")
    p_mode = _resolve_mode(p_quant, "p_quant")
    v_mode = _resolve_mode(v_quant, "v_quant")
    if (q_mode == 2 or k_mode == 2) and lqk % 16:
        raise ValueError(f"NVFP4 Q/K requires qk_head_dim % 16 == 0, got {lqk}")
    q_scale = _operand_scale(q_mode, q_amax, "q_amax")
    k_scale = _operand_scale(k_mode, k_amax, "k_amax")
    p_scale = _operand_scale(p_mode, p_amax, "p_amax")
    v_scale = _operand_scale(v_mode, v_amax, "v_amax")

    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()

    sm_scale = lqk**-0.5 if softmax_scale is None else softmax_scale
    block_dqk = triton.next_power_of_2(lqk)
    block_dv = triton.next_power_of_2(lv)
    if block_dqk >= 512:
        # Very wide QK tiles (e.g. absorbed 576-d shapes) blow the shared
        # memory budget at 64x64 tiles (q + k^T staging ~ 3 * BLOCK_DQK KB);
        # shrink to ~96 KB so the kernel fits 100 KB-class SMs.
        block_m = min(block_m, 16)
        block_n = min(block_n, 32)
    batch = cu_seqlens_q.numel() - 1

    out = torch.empty(total_q, num_heads, lv, dtype=v.dtype, device=q.device)
    if return_lse:
        lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty(1, dtype=torch.float32, device=q.device)

    grid = (batch, num_heads, triton.cdiv(max(1, max_seqlen_q), block_m))
    with torch.cuda.device(q.device):
        _mla_prefill_kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            sm_scale * LOG2E,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            out.stride(0),
            out.stride(1),
            lse.stride(0) if return_lse else 0,
            lse.stride(1) if return_lse else 0,
            q_scale,
            k_scale,
            p_scale,
            v_scale,
            kv_group_num=num_heads // num_kv_heads,
            IS_CAUSAL=causal,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_DQK=block_dqk,
            BLOCK_DV=block_dv,
            LQK=lqk,
            LV=lv,
            RETURN_LSE=return_lse,
            Q_QUANT=q_mode,
            K_QUANT=k_mode,
            P_QUANT=p_mode,
            V_QUANT=v_mode,
            IEEE_QK=(q_mode == 2 or k_mode == 2 or q.dtype == torch.float32),
            IEEE_PV=(p_mode == 2 or v_mode == 2),
            SPARSITY_N=sparsity_n,
            SPARSITY_M=sparsity_m,
            DENSE_SINK_TOKENS=dense_sink_tokens,
            DENSE_RECENT_TOKENS=dense_recent_tokens,
            # 256-wide FP32 tiles are register-heavy; revisit if tuning.  # tune
            num_warps=8 if block_dqk >= 256 else 4,
            num_stages=1,
        )
    if return_lse:
        return out, lse
    return out
