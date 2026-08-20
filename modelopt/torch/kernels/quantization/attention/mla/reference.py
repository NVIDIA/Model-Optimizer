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

"""Pure-torch oracles for the MLA attention kernels (no Triton imports).

Independent eager re-implementations of the fake-quant math and the kernels'
tile schedules, used by the GPU golden tests. The FP4 rounding ladder mirrors
``fp4_round_magnitude`` (RNE ties-to-even on the E2M1 grid) and the two-level
NVFP4 scale mirrors ``fp8_quantize_scale``, but no production QDQ helper is
called here.
"""

import math

import torch

LOG2E: float = 1.4426950408889634
LN2: float = 0.6931471805599453

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to even (matches the kernels)."""
    a = x.abs()
    q = torch.where(
        a <= 0.25,
        torch.zeros_like(a),
        torch.where(
            a < 0.75,
            torch.full_like(a, 0.5),
            torch.where(
                a <= 1.25,
                torch.full_like(a, 1.0),
                torch.where(
                    a < 1.75,
                    torch.full_like(a, 1.5),
                    torch.where(
                        a <= 2.5,
                        torch.full_like(a, 2.0),
                        torch.where(
                            a < 3.5,
                            torch.full_like(a, 3.0),
                            torch.where(a <= 5.0, torch.full_like(a, 4.0), torch.full_like(a, 6.0)),
                        ),
                    ),
                ),
            ),
        ),
    )
    return torch.where(x >= 0, q, -q)


def _quant_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through FP8 E4M3 with saturation at +-448."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def nvfp4_fake_quant(
    x: torch.Tensor,
    amax: float | None = None,
    block_axis: int = -1,
    block: int = 16,
) -> torch.Tensor:
    """Two-level NVFP4 fake quant: E2M1 elements, E4M3 block scales along one axis."""
    x = x.float()
    x = x.movedim(block_axis, -1)
    shape = x.shape
    assert shape[-1] % block == 0, f"axis size {shape[-1]} not divisible by {block}"
    g = x.reshape(*shape[:-1], shape[-1] // block, block)
    block_amax = g.abs().amax(dim=-1, keepdim=True)
    global_scale = 1.0 if amax is None else amax / (6.0 * 448.0)
    scale = _quant_e4m3(block_amax / (6.0 * global_scale)) * global_scale
    scale_safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = round_to_e2m1(g / scale_safe) * scale_safe
    q = torch.where(scale == 0, torch.zeros_like(q), q)
    return q.reshape(shape).movedim(-1, block_axis)


def fp8_tensor_fake_quant(x: torch.Tensor, amax: float | None = None) -> torch.Tensor:
    """Per-tensor FP8 E4M3 fake quant with scale ``amax / 448``."""
    scale = 1.0 if amax is None else amax / 448.0
    return _quant_e4m3(x.float() / scale) * scale


def apply_operand_quant(
    x: torch.Tensor,
    mode: str | None,
    amax: float | None,
    block_axis: int = -1,
) -> torch.Tensor:
    """Dispatch one operand's fake quant by mode name."""
    if mode is None:
        return x.float()
    if mode == "fp8":
        return fp8_tensor_fake_quant(x, amax)
    if mode == "nvfp4":
        return nvfp4_fake_quant(x, amax, block_axis=block_axis)
    raise ValueError(f"unknown quant mode {mode!r}")


def apply_sparse_nm(
    scores: torch.Tensor,
    sparsity_n: int,
    sparsity_m: int,
    q_abs_pos: torch.Tensor,
    kv_abs_pos: torch.Tensor,
    seq_len_q: int,
    dense_sink_tokens: int,
    dense_recent_tokens: int,
) -> torch.Tensor:
    """Token-exact N:M score sparsity with dense sink/recent regions.

    ``scores`` is ``[..., Lq, Lk]`` with the key axis last; ``q_abs_pos`` and
    ``kv_abs_pos`` are the absolute positions matching those two axes.
    """
    shape = scores.shape
    grouped = scores.reshape(*shape[:-1], shape[-1] // sparsity_m, sparsity_m)
    topk = grouped.topk(sparsity_n, dim=-1).indices
    keep = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, topk, True)
    sparse = torch.where(keep, grouped, torch.full_like(grouped, float("-inf")))
    sparse = sparse.reshape(shape)
    distance = q_abs_pos[:, None] - kv_abs_pos[None, :]
    dense = (
        (seq_len_q <= 1)
        | (kv_abs_pos[None, :] < dense_sink_tokens)
        | ((distance >= 0) & (distance < dense_recent_tokens))
    )
    return torch.where(dense, scores, sparse)


def _quantize_p_tile(
    p: torch.Tensor,
    mode: str | None,
    amax: float,
    carrier_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize an unnormalized softmax tile ``[..., block_n]`` like the kernels do."""
    if mode is None:
        return p
    if mode == "fp8":
        return fp8_tensor_fake_quant(p, amax)
    if mode == "nvfp4":
        p = p.to(carrier_dtype).float()
        return nvfp4_fake_quant(p, amax, block_axis=-1)
    raise ValueError(f"unknown quant mode {mode!r}")


def mla_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
    *,
    q_mode: str | None = None,
    k_mode: str | None = None,
    p_mode: str | None = None,
    v_mode: str | None = None,
    q_amax: float | None = None,
    k_amax: float | None = None,
    p_amax: float = 1.0,
    v_amax: float | None = None,
    sparsity_n: int = 0,
    sparsity_m: int = 4,
    dense_sink_tokens: int = 0,
    dense_recent_tokens: int = 64,
    block_n: int = 64,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Eager oracle for :func:`mla_prefill_attention`.

    Replicates the kernel schedule: Q/K quantized along the feature axis and
    V along the token axis up front; P quantized per ``block_n`` KV tile on
    the unnormalized online-softmax probabilities, with the denominator kept
    unquantized. Returns FP32 output ``[total_q, num_heads, LV]`` (and
    natural-log LSE ``[num_heads, total_q]`` with ``return_lse``).
    """
    total_q, num_heads, lqk = q.shape
    num_kv_heads, lv = k.shape[1], v.shape[2]
    scale = lqk**-0.5 if softmax_scale is None else softmax_scale
    carrier_dtype = v.dtype
    batch = cu_seqlens_q.numel() - 1

    out = torch.zeros(total_q, num_heads, lv, dtype=torch.float32, device=q.device)
    lse_out = torch.full((num_heads, total_q), float("-inf"), dtype=torch.float32, device=q.device)

    for b in range(batch):
        q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        k0, k1 = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        lq, lk = q1 - q0, k1 - k0
        if lq == 0:
            continue
        qb = apply_operand_quant(q[q0:q1], q_mode, q_amax, block_axis=-1)  # [Lq, H, LQK]
        kb = apply_operand_quant(k[k0:k1], k_mode, k_amax, block_axis=-1)  # [Lk, Hkv, LQK]
        # V groups along tokens: the kernel's masked (zero) tail rows are
        # equivalent to zero-padding the sequence to a multiple of 16.
        vb = v[k0:k1].float()
        if v_mode is not None:
            pad = (-lk) % 16
            vb = torch.nn.functional.pad(vb, (0, 0, 0, 0, 0, pad))
            vb = apply_operand_quant(vb, v_mode, v_amax, block_axis=0)[:lk]
        if num_kv_heads != num_heads:
            kb = kb.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vb = vb.repeat_interleave(num_heads // num_kv_heads, dim=1)

        qh = qb.permute(1, 0, 2)  # [H, Lq, LQK]
        kh = kb.permute(1, 2, 0)  # [H, LQK, Lk]
        vh = vb.permute(1, 0, 2)  # [H, Lk, LV]

        q_pos = torch.arange(lq, device=q.device)
        row_max = torch.full((num_heads, lq), float("-inf"), device=q.device)
        row_sum = torch.zeros(num_heads, lq, device=q.device)
        acc = torch.zeros(num_heads, lq, lv, device=q.device)

        for kv_start in range(0, lk, block_n):
            kv_end = min(kv_start + block_n, lk)
            kv_abs = torch.arange(kv_start, kv_start + block_n, device=q.device)
            scores = torch.matmul(qh, kh[:, :, kv_start:kv_end]) * scale * LOG2E
            if kv_end - kv_start < block_n:  # pad tile to block_n like the kernel
                pad = block_n - (kv_end - kv_start)
                scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
            valid = kv_abs < lk
            if causal:
                allowed = (q_pos[:, None] + (lk - lq)) >= kv_abs[None, :]
                mask = valid[None, :] & allowed
            else:
                mask = valid[None, :].expand(lq, block_n)
            scores = scores.masked_fill(~mask[None], float("-inf"))
            if sparsity_n > 0:
                scores = apply_sparse_nm(
                    scores,
                    sparsity_n,
                    sparsity_m,
                    q_pos + (lk - lq),
                    kv_abs,
                    lq,
                    dense_sink_tokens,
                    dense_recent_tokens,
                )
                scores = scores.masked_fill(~mask[None], float("-inf"))

            m_new = torch.maximum(row_max, scores.amax(dim=-1))
            # Rows with no valid keys yet keep -inf; guard exp2 of (-inf) - (-inf).
            shifted = scores - m_new.unsqueeze(-1)
            p = torch.where(torch.isneginf(scores), torch.zeros_like(scores), torch.exp2(shifted))
            correction = torch.where(
                torch.isneginf(row_max), torch.zeros_like(row_max), torch.exp2(row_max - m_new)
            )
            row_sum = row_sum * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1)
            p_q = _quantize_p_tile(p, p_mode, p_amax, carrier_dtype)
            v_tile = vh[:, kv_start:kv_end]
            if kv_end - kv_start < block_n:
                v_tile = torch.nn.functional.pad(v_tile, (0, 0, 0, block_n - (kv_end - kv_start)))
            if p_mode != "nvfp4" and v_mode != "nvfp4":
                # Non-IEEE kernel path dots p (and unquantized v) in the
                # compute dtype; the IEEE path keeps p in FP32.
                p_q = p_q.to(carrier_dtype).float()
                v_tile = v_tile.to(carrier_dtype).float() if v_mode is None else v_tile
            acc = acc + torch.matmul(p_q, v_tile)
            row_max = m_new

        out[q0:q1] = (acc / row_sum.clamp_min(1e-6).unsqueeze(-1)).permute(1, 0, 2)
        lse = LN2 * (row_max + torch.log2(row_sum.clamp_min(1e-30)))
        lse = torch.where(row_sum == 0, torch.full_like(lse, float("-inf")), lse)
        lse_out[:, q0:q1] = lse

    if return_lse:
        return out, lse_out
    return out


def mla_decode_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    kv_lora_rank: int,
    *,
    p_mode: str | None = None,
    p_amax: float = 1.0,
    num_kv_splits: int = 32,
    block_n: int = 32,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Eager split-local oracle for :func:`mla_attention_decode`.

    ``latent`` is the dense (unpaged) cache ``[batch, max_seq, head_dim]``;
    ``q`` is the absorbed query ``[batch, num_heads, head_dim]``. Replicates
    the fixed-split, tile-aligned stage-1 schedule and the stage-2 merge.
    """
    batch, num_heads, head_dim = q.shape
    carrier_dtype = latent.dtype
    out = torch.zeros(batch, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device)
    lse_out = torch.zeros(batch, num_heads, dtype=torch.float32, device=q.device)

    qf = q.float()
    q_nope, q_pe = qf[..., :kv_lora_rank], qf[..., kv_lora_rank:]

    for b in range(batch):
        s = int(seq_lens[b])
        k_all = latent[b, :s].float()  # [s, head_dim]
        k_nope, k_pe = k_all[:, :kv_lora_rank], k_all[:, kv_lora_rank:]
        num_tiles = math.ceil(s / block_n)
        tiles_per_split = math.ceil(num_tiles / num_kv_splits)

        split_m, split_l, split_acc = [], [], []
        for split in range(num_kv_splits):
            kv_lo = split * tiles_per_split * block_n
            kv_hi = min(kv_lo + tiles_per_split * block_n, s)
            m = torch.full((num_heads,), float("-inf"), device=q.device)
            lsum = torch.zeros(num_heads, device=q.device)
            acc = torch.zeros(num_heads, kv_lora_rank, device=q.device)
            for kv_start in range(kv_lo, kv_hi, block_n):
                kv_end = min(kv_start + block_n, s)
                kn = k_nope[kv_start:kv_end]
                kp = k_pe[kv_start:kv_end]
                scores = (q_nope[b] @ kn.T + q_pe[b] @ kp.T) * softmax_scale * LOG2E  # [H, tile]
                if kv_end - kv_start < block_n:
                    # Pad to block_n like the kernel: -inf scores -> p == 0,
                    # zero V rows -> no BMM2 contribution.
                    pad = block_n - (kv_end - kv_start)
                    scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
                    kn = torch.nn.functional.pad(kn, (0, 0, 0, pad))
                m_new = torch.maximum(m, scores.amax(dim=-1))
                shifted = scores - m_new.unsqueeze(-1)
                p = torch.where(
                    torch.isneginf(scores), torch.zeros_like(scores), torch.exp2(shifted)
                )
                correction = torch.where(
                    torch.isneginf(m), torch.zeros_like(m), torch.exp2(m - m_new)
                )
                lsum = lsum * correction + p.sum(dim=-1)
                acc = acc * correction.unsqueeze(-1)
                p_q = _quantize_p_tile(p, p_mode, p_amax, carrier_dtype)
                if p_mode != "nvfp4":
                    p_q = p_q.to(carrier_dtype).float()
                acc = acc + p_q @ kn
                m = m_new
            split_m.append(m)
            split_l.append(lsum)
            split_acc.append(acc)

        m = torch.full((num_heads,), float("-inf"), device=q.device)
        lsum = torch.zeros(num_heads, device=q.device)
        acc = torch.zeros(num_heads, kv_lora_rank, device=q.device)
        for sm, sl, sa in zip(split_m, split_l, split_acc):
            has = sl > 0
            new_max = torch.where(has, torch.maximum(m, sm), m)
            corr = torch.where(torch.isneginf(m), torch.zeros_like(m), torch.exp2(m - new_max))
            scorr = torch.where(has, torch.exp2(sm - new_max), torch.zeros_like(sm))
            acc = acc * corr.unsqueeze(-1) + sa * scorr.unsqueeze(-1)
            lsum = lsum * corr + sl * scorr
            m = new_max
        out[b] = acc / lsum.clamp_min(1e-6).unsqueeze(-1)
        lse = LN2 * (m + torch.log2(lsum.clamp_min(1e-30)))
        lse_out[b] = torch.where(lsum == 0, torch.full_like(lse, float("-inf")), lse)

    if return_lse:
        return out, lse_out
    return out
