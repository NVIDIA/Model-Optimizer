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

"""Split-K absorbed-MLA decode over a paged latent cache, with fused P QDQ.

Absorbed MLA decode is MQA: every query head attends to the single latent KV
"head". The absorbed query carries ``kv_lora_rank + qk_rope_head_dim``
features (576 for DeepSeek-family models); K is the full latent cache row and
V is its first ``kv_lora_rank`` features — the same memory read once per tile
and reused for both BMMs.

Quantization contract: the latent cache is expected to hold write-once
fake-quantized values (module-level ``kv_c``/``k_pe`` quantizers applied
before the cache write). Both BMM1-K and BMM2-V consume that single
representation as-is — there is deliberately no on-read re-quantization. The
absorbed Q is expected to be fake-quantized by the caller (dynamic NVFP4 uses
an FP32 QDQ carrier, like ``triton_fa``'s ``Q_IS_FP32``). Only the softmax P
is quantized inside the kernel, after the row-sum, so the softmax denominator
stays unquantized.

P QDQ operates on split-local, unnormalized online-softmax probabilities;
its numerics therefore include the fixed split count and tile size as part of
the kernel schedule. Split bounds are tile-aligned so quant-relevant tile
boundaries sit at absolute token positions and results are stable as the
sequence grows. Inference-only.
"""

import torch
import triton
import triton.language as tl

from modelopt.torch.kernels.common.attention.decode_attention import _qdq_scale
from modelopt.torch.kernels.common.attention.triton_fa import LOG2E
from modelopt.torch.kernels.quantization.attention.bmm2_qdq import _p_qdq_nvfp4
from modelopt.torch.kernels.quantization.common.fp8_quant import fp8_scalar_qdq

__all__ = ["mla_attention_decode"]

# Referenced inside @triton.jit code, so it must be a tl.constexpr global.
LN2 = tl.constexpr(0.6931471805599453)

_BLOCK_N = 32
_DEFAULT_KV_SPLITS = 32
_MAX_KV_SPLITS = 32
_P_QDQ_MODES = {None: 0, "fp8": 1, "nvfp4": 2}


@triton.jit
def _mla_decode_split_kernel(
    Q,  # [batch, num_heads, KV_LORA_RANK + QK_ROPE_DIM] absorbed query
    Latent_cache,  # [num_blocks, page_size, KV_LORA_RANK + QK_ROPE_DIM]
    Block_table,  # [batch, max_blocks_per_seq]
    B_seq_len,  # [batch]
    M_partial,  # [batch, num_heads, NUM_KV_SPLITS]
    L_partial,  # [batch, num_heads, NUM_KV_SPLITS]
    Acc_partial,  # [batch, num_heads, NUM_KV_SPLITS, BLOCK_DL]
    qk_scale,  # softmax_scale * log2(e)
    stride_qb,
    stride_qh,
    stride_lc_block,
    stride_lc_pos,
    stride_mb,
    stride_mh,
    stride_ab,
    stride_ah,
    stride_as,
    p_qdq_scale,
    max_blocks_per_seq,
    H: tl.constexpr,  # number of query heads
    BLOCK_H: tl.constexpr,  # query heads per program (grouped MQA)
    BLOCK_N: tl.constexpr,
    BLOCK_DL: tl.constexpr,  # next_power_of_2(KV_LORA_RANK)
    BLOCK_DPE: tl.constexpr,  # next_power_of_2(QK_ROPE_DIM)
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    P_QDQ: tl.constexpr,  # 0=off, 1=FP8 E4M3, 2=NVFP4
    Q_IS_FP32: tl.constexpr,  # dynamic NVFP4 QDQ carrier uses FP32
):
    """Compute one partial softmax for one head group, request, and KV split."""
    # Head groups on axis 0 (fastest varying) so programs sharing the same
    # latent KV tiles are co-scheduled for L2 reuse (MQA: kv_group_num == H).
    head_group = tl.program_id(0)
    batch_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    head_ids = head_group * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = head_ids < H
    seq_len = tl.load(B_seq_len + batch_idx)

    dl_pos = tl.arange(0, BLOCK_DL)
    dpe_pos = tl.arange(0, BLOCK_DPE)
    dl_mask = dl_pos < KV_LORA_RANK
    dpe_mask = dpe_pos < QK_ROPE_DIM
    kv_pos = tl.arange(0, BLOCK_N)

    q_base = Q + batch_idx * stride_qb + head_ids[:, None] * stride_qh
    q_nope = tl.load(q_base + dl_pos[None, :], mask=mask_h[:, None] & dl_mask[None, :], other=0.0)
    q_pe = tl.load(
        q_base + KV_LORA_RANK + dpe_pos[None, :],
        mask=mask_h[:, None] & dpe_mask[None, :],
        other=0.0,
    )
    if Q_IS_FP32:
        q_nope = q_nope.to(tl.float32)
        q_pe = q_pe.to(tl.float32)

    # Tile-aligned split bounds: quant-relevant tile boundaries sit at
    # absolute token positions, so numerics are stable as the sequence grows.
    num_tiles = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_tiles, NUM_KV_SPLITS)
    kv_lo = split_idx * tiles_per_split * BLOCK_N
    kv_hi = tl.minimum(kv_lo + tiles_per_split * BLOCK_N, seq_len)

    running_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DL], dtype=tl.float32)

    for kv_start in range(kv_lo, kv_hi, BLOCK_N):
        kv_start = tl.multiple_of(kv_start, BLOCK_N)
        kv_abs = kv_start + kv_pos
        kv_valid = kv_abs < seq_len

        page = tl.load(
            Block_table + batch_idx * max_blocks_per_seq + kv_abs // PAGE_SIZE,
            mask=kv_valid,
            other=0,
        ).to(tl.int64)
        pos_ptrs = page * stride_lc_block + (kv_abs % PAGE_SIZE) * stride_lc_pos

        # K^T tiles from the latent cache: NOPE [BLOCK_DL, BLOCK_N], PE [BLOCK_DPE, BLOCK_N]
        k_nope = tl.load(
            Latent_cache + pos_ptrs[None, :] + dl_pos[:, None],
            mask=kv_valid[None, :] & dl_mask[:, None],
            other=0.0,
        )
        k_pe = tl.load(
            Latent_cache + pos_ptrs[None, :] + KV_LORA_RANK + dpe_pos[:, None],
            mask=kv_valid[None, :] & dpe_mask[:, None],
            other=0.0,
        )

        if Q_IS_FP32:
            scores = tl.dot(q_nope, k_nope.to(tl.float32), input_precision="ieee")
            scores += tl.dot(q_pe, k_pe.to(tl.float32), input_precision="ieee")
        else:
            scores = tl.dot(q_nope, k_nope) + tl.dot(q_pe, k_pe)
        scores = scores * qk_scale
        scores = tl.where(kv_valid[None, :], scores, float("-inf"))

        # --- Online softmax update: the denominator uses unquantized p ---
        m_new = tl.maximum(running_max, tl.max(scores, 1))
        p = tl.math.exp2(scores - m_new[:, None])
        p = tl.where(kv_valid[None, :], p, 0.0)
        correction = tl.math.exp2(running_max - m_new)
        running_sum = running_sum * correction + tl.sum(p, 1)
        acc = acc * correction[:, None]

        if P_QDQ == 1:
            p = fp8_scalar_qdq(p, p_qdq_scale)
        elif P_QDQ == 2:
            # Native packing consumes the cache dtype; the QDQ value stays FP32.
            p = p.to(Latent_cache.dtype.element_ty).to(tl.float32)
            p = _p_qdq_nvfp4(p, p_qdq_scale, BLOCK_H, BLOCK_N)

        # V is the first KV_LORA_RANK features of the same latent tile,
        # consumed as-is (single stored representation; no on-read re-quant).
        v = tl.trans(k_nope)
        if P_QDQ == 2:
            acc = tl.dot(p, v.to(tl.float32), acc, input_precision="ieee")
        else:
            acc = tl.dot(p.to(v.dtype), v, acc)
        running_max = m_new

    partial_offset = batch_idx * stride_mb + head_ids * stride_mh + split_idx
    tl.store(M_partial + partial_offset, running_max, mask=mask_h)
    tl.store(L_partial + partial_offset, running_sum, mask=mask_h)
    acc_ptrs = (
        batch_idx * stride_ab
        + head_ids[:, None] * stride_ah
        + split_idx * stride_as
        + dl_pos[None, :]
    )
    tl.store(Acc_partial + acc_ptrs, acc, mask=mask_h[:, None] & dl_mask[None, :])


@triton.jit
def _mla_decode_combine_kernel(
    M_partial,
    L_partial,
    Acc_partial,
    Out,  # [batch, num_heads, KV_LORA_RANK]
    Lse,  # [batch, num_heads] natural-log LSE (dummy when not STORE_LSE)
    stride_mb,
    stride_mh,
    stride_ab,
    stride_ah,
    stride_as,
    stride_ob,
    stride_oh,
    stride_lse_b,
    BLOCK_DL: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    """Merge split-local online-softmax states."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dl_pos = tl.arange(0, BLOCK_DL)
    dl_mask = dl_pos < KV_LORA_RANK
    base_ml = batch_idx * stride_mb + head_idx * stride_mh
    base_acc = batch_idx * stride_ab + head_idx * stride_ah

    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([BLOCK_DL], dtype=tl.float32)
    for split_idx in range(NUM_KV_SPLITS):
        split_sum = tl.load(L_partial + base_ml + split_idx)
        if split_sum > 0.0:
            split_max = tl.load(M_partial + base_ml + split_idx)
            split_acc = tl.load(
                Acc_partial + base_acc + split_idx * stride_as + dl_pos,
                mask=dl_mask,
                other=0.0,
            )
            new_max = tl.maximum(running_max, split_max)
            correction = tl.math.exp2(running_max - new_max)
            split_correction = tl.math.exp2(split_max - new_max)
            acc = acc * correction + split_acc * split_correction
            running_sum = running_sum * correction + split_sum * split_correction
            running_max = new_max

    output = acc / tl.maximum(running_sum, 1e-6)
    tl.store(
        Out + batch_idx * stride_ob + head_idx * stride_oh + dl_pos,
        output,
        mask=dl_mask,
    )
    if STORE_LSE:
        lse = LN2 * (running_max + tl.math.log2(running_sum))
        lse = tl.where(running_sum == 0.0, float("-inf"), lse)
        tl.store(Lse + batch_idx * stride_lse_b + head_idx, lse)


def mla_attention_decode(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    block_table: torch.Tensor,
    b_seq_len: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    page_size: int | None = None,
    num_kv_splits: int = _DEFAULT_KV_SPLITS,
    p_qdq: str | None = None,
    p_qdq_amax: float = 1.0,
    return_lse: bool = True,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode one absorbed query token per request over a paged latent cache.

    Args:
        q: ``[batch, num_heads, kv_lora_rank + qk_rope_head_dim]`` absorbed
            query. Pass FP32 for the dynamic-NVFP4 QDQ carrier (Q is expected
            to be fake-quantized by the caller); BF16/FP16 otherwise.
        latent_cache: ``[num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]``
            paged latent cache. Expected to hold write-once fake-quantized
            values; consumed as-is for both BMM1-K and BMM2-V.
        block_table: ``[batch, max_blocks_per_seq]`` page table.
        b_seq_len: ``[batch]`` KV sequence lengths.
        softmax_scale: Softmax scale (required; MLA layers fold in mscale).
        kv_lora_rank: Latent width (V/output width).
        qk_rope_head_dim: RoPE feature width appended to the latent.
        page_size: Tokens per page; defaults to ``latent_cache.shape[1]``.
        num_kv_splits: Fixed split count. P QDQ numerics follow the
            split-local schedule, so this stays fixed by default for
            reproducibility across batch shapes and devices.
        p_qdq: Softmax-P fake quant-dequant: ``None``, ``"fp8"``, ``"nvfp4"``.
        p_qdq_amax: Per-tensor P amax (default 1.0, the theoretical bound).
        return_lse: Also return the natural-log LSE ``[batch, num_heads]``.
        out_dtype: Output dtype (default ``latent_cache.dtype``, the model
            compute dtype expected by the V up-projection).

    Returns:
        ``(out [batch, num_heads, kv_lora_rank], lse [batch, num_heads] | None)``.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [batch, heads, head_dim], got {tuple(q.shape)}")
    if latent_cache.ndim != 3:
        raise ValueError(
            f"latent_cache must be [num_blocks, page_size, head_dim], "
            f"got {tuple(latent_cache.shape)}"
        )
    head_dim = kv_lora_rank + qk_rope_head_dim
    if q.shape[2] != head_dim or latent_cache.shape[2] != head_dim:
        raise ValueError(
            f"q and latent_cache feature dims must equal kv_lora_rank + qk_rope_head_dim "
            f"({head_dim}), got {q.shape[2]} and {latent_cache.shape[2]}"
        )
    if page_size is None:
        page_size = latent_cache.shape[1]
    if page_size != latent_cache.shape[1]:
        raise ValueError(f"page_size {page_size} must match latent_cache.shape[1]")
    if not 1 <= num_kv_splits <= _MAX_KV_SPLITS:
        raise ValueError(f"num_kv_splits must be in [1, {_MAX_KV_SPLITS}], got {num_kv_splits}")
    if p_qdq == "nvfp4" and (kv_lora_rank % 16 or qk_rope_head_dim % 16):
        raise ValueError("NVFP4 decode requires dimensions divisible by 16")
    batch, num_heads = q.shape[0], q.shape[1]
    if b_seq_len.shape != (batch,) or block_table.shape[0] != batch:
        raise ValueError("decode metadata batch dimension must match q")

    p_qdq_scale = _qdq_scale(p_qdq, p_qdq_amax, "p")
    q = q.contiguous()
    if latent_cache.stride(-1) != 1:
        raise ValueError("latent_cache last dim must be contiguous")

    block_dl = triton.next_power_of_2(kv_lora_rank)
    block_dpe = triton.next_power_of_2(qk_rope_head_dim)
    block_h = 16 if num_heads <= 16 else 32
    qk_scale = softmax_scale * LOG2E
    if out_dtype is None:
        out_dtype = latent_cache.dtype

    m_partial = torch.empty(batch, num_heads, num_kv_splits, dtype=torch.float32, device=q.device)
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        batch, num_heads, num_kv_splits, block_dl, dtype=torch.float32, device=q.device
    )
    out = torch.empty(batch, num_heads, kv_lora_rank, dtype=out_dtype, device=q.device)
    if return_lse:
        lse = torch.empty(batch, num_heads, dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty(1, dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device):
        _mla_decode_split_kernel[(triton.cdiv(num_heads, block_h), batch, num_kv_splits)](
            q,
            latent_cache,
            block_table,
            b_seq_len,
            m_partial,
            l_partial,
            acc_partial,
            qk_scale,
            q.stride(0),
            q.stride(1),
            latent_cache.stride(0),
            latent_cache.stride(1),
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            p_qdq_scale,
            block_table.shape[1],
            H=num_heads,
            BLOCK_H=block_h,
            BLOCK_N=_BLOCK_N,
            BLOCK_DL=block_dl,
            BLOCK_DPE=block_dpe,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_DIM=qk_rope_head_dim,
            PAGE_SIZE=page_size,
            NUM_KV_SPLITS=num_kv_splits,
            P_QDQ=_P_QDQ_MODES[p_qdq],
            Q_IS_FP32=q.dtype == torch.float32,
            num_warps=4,
            num_stages=2,
        )
        _mla_decode_combine_kernel[(batch, num_heads)](
            m_partial,
            l_partial,
            acc_partial,
            out,
            lse,
            m_partial.stride(0),
            m_partial.stride(1),
            acc_partial.stride(0),
            acc_partial.stride(1),
            acc_partial.stride(2),
            out.stride(0),
            out.stride(1),
            lse.stride(0) if return_lse else 0,
            BLOCK_DL=block_dl,
            KV_LORA_RANK=kv_lora_rank,
            NUM_KV_SPLITS=num_kv_splits,
            STORE_LSE=return_lse,
            num_warps=4,
        )
    return out, (lse if return_lse else None)
