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


"""Research-only K128 E2M1/FP32 codec ported from Igor's study.

This is NOT NVIDIA NVFP4. Midpoint ties round away from zero. Only the
nearest training path is included; no stochastic rounding is released here.
"""

import torch
import triton
import triton.language as tl

SIXTH: tl.constexpr = tl.constexpr(0.16666667163372040)


def _layout(state):
    if state.ndim not in (3, 4) or state.shape[-2] != 128 or state.dtype != torch.float32:
        raise ValueError("Reference codec requires FP32 [H,128,V] or [B,H,128,V]")
    return state.unsqueeze(0) if state.ndim == 3 else state


@torch.library.custom_op("modelopt::reference_k128_qdq", mutates_args=())
def reference_k128_qdq(state: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    """Quantize one current state per request/head using capped K128 FP32 scales."""
    work = _layout(state)
    count, heads, keys, values = work.shape
    if (
        cap.shape != (heads,)
        or cap.dtype != torch.float32
        or cap.device != state.device
        or not cap.is_contiguous()
    ):
        raise ValueError(
            "Reference codec requires contiguous FP32 caps, one per head on the state device"
        )
    out = torch.empty(work.shape, device=state.device, dtype=state.dtype)
    _fp32block_cache_qdq_kernel[(count, heads, triton.cdiv(values, 16))](
        work,
        out,
        work,
        work,
        cap,
        cap,
        NUM_SLOTS=count,
        V=values,
        K=keys,
        HEADS=heads,
        STRIDE_SLOT=work.stride(0),
        STRIDE_HEAD=work.stride(1),
        STRIDE_V=work.stride(3),
        STRIDE_K=work.stride(2),
        OUT_STRIDE_SLOT=out.stride(0),
        OUT_STRIDE_HEAD=out.stride(1),
        OUT_STRIDE_V=out.stride(3),
        OUT_STRIDE_K=out.stride(2),
        HAS_INDICES=False,
        HAS_VALID=False,
        HAS_CAP=True,
        CAP_P=32,
        AXIS_K=True,
        BLOCK=128,
        ROUND=0,
        ROUNDS=5,
        TILE_V=16,
    )
    return out[0] if state.ndim == 3 else out


@reference_k128_qdq.register_fake
def _reference_k128_fake(state, cap):
    return torch.empty_like(state, memory_format=torch.contiguous_format)


@torch.library.custom_op("modelopt::reference_k128_cap_grad", mutates_args=())
def _cap_gradient(state: torch.Tensor, grad: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    work, grad_work = _layout(state), _layout(grad.contiguous())
    count, heads, keys, values = work.shape
    tiles = triton.cdiv(values, 16)
    partials = torch.zeros(count, tiles, heads, device=state.device, dtype=torch.float32)
    out_strides = torch.empty(work.shape, device="meta").stride()
    _cap_grad_kernel[(count, heads, tiles)](
        work,
        grad_work,
        cap,
        partials,
        cap,
        V=values,
        K=keys,
        STRIDE_SLOT=work.stride(0),
        STRIDE_HEAD=work.stride(1),
        STRIDE_V=work.stride(3),
        STRIDE_K=work.stride(2),
        G_STRIDE_SLOT=grad_work.stride(0),
        G_STRIDE_HEAD=grad_work.stride(1),
        G_STRIDE_V=grad_work.stride(3),
        G_STRIDE_K=grad_work.stride(2),
        OUT_STRIDE_SLOT=out_strides[0],
        OUT_STRIDE_HEAD=out_strides[1],
        OUT_STRIDE_V=out_strides[3],
        OUT_STRIDE_K=out_strides[2],
        CAP_P=32,
        AXIS_K=True,
        BLOCK=128,
        ROUND=0,
        ROUNDS=5,
        TILE_V=16,
    )
    return partials.sum((0, 1))


@_cap_gradient.register_fake
def _cap_gradient_fake(state, grad, cap):
    return torch.empty_like(cap)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    state, cap = ctx.saved_tensors
    # Match the peer's identity STE even when a value was clipped.
    return grad, _cap_gradient(state, grad, cap)


reference_k128_qdq.register_autograd(_backward, setup_context=_setup_context)


@triton.jit
def _round_to_grid(a, index, Seed, offsets, ROUND: tl.constexpr, ROUNDS: tl.constexpr):
    tl.static_assert(ROUND == 0)
    return _grid_value(index)


@triton.jit
def _grid_value(index):
    # E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} by index.
    return tl.where(
        index <= 4,
        index.to(tl.float32) * 0.5,
        tl.where(index == 5, 3.0, tl.where(index == 6, 4.0, 6.0)),
    )


@triton.jit
def _capped_amax(amax, cap, P: tl.constexpr):
    """Block amax limited by the per-head cap: smooth minimum ``(amax^-P + cap^-P)^(-1/P)`` (``P == 0``: hard minimum).

    The smooth form keeps a nonzero derivative with respect to ``cap`` for blocks near the cap, so a cap that starts
    at the head's maximum amax can still learn. For ``P = 32`` it is within 2.2% of the hard minimum everywhere.
    """
    small = tl.minimum(amax, cap)
    if P == 0:
        return small
    big = tl.maximum(amax, cap)
    ratio = tl.where(big > 0, small / tl.where(big > 0, big, 1.0), 0.0)
    ratio_p = tl.exp(P * tl.log(tl.maximum(ratio, 1e-30)))
    return small * tl.exp(-tl.log(1.0 + ratio_p) / P)


@triton.jit
def _block_scale(amax):
    """``amax / 6``, or 1 for an all-zero block and for a block so small that its scale underflows to zero.

    A zero scale would give ``0 / 0 = NaN`` for the zeros of such a block (the sparse, decayed states of a long decode
    reach this); the elements of such a block round to zero with scale 1 as well.
    """
    scale = amax * SIXTH
    return tl.where(scale > 0, scale, 1.0)


@triton.jit
def _fp32block_cache_qdq_kernel(
    State,
    Out,
    Indices,
    Valid,
    Seed,
    Cap,
    NUM_SLOTS,  # slot count and strides are runtime arguments: they must not trigger recompiles
    V: tl.constexpr,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    STRIDE_SLOT,
    STRIDE_HEAD,
    STRIDE_V,
    STRIDE_K,
    OUT_STRIDE_SLOT,
    OUT_STRIDE_HEAD,
    OUT_STRIDE_V,
    OUT_STRIDE_K,
    HAS_INDICES: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_CAP: tl.constexpr,
    CAP_P: tl.constexpr,
    AXIS_K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    ROUNDS: tl.constexpr,
    TILE_V: tl.constexpr,
):
    n = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    if HAS_INDICES:
        slot = tl.load(Indices + n).to(tl.int64)
        ok = (slot > 0) & (slot < NUM_SLOTS)
    else:
        slot = n.to(tl.int64)
        ok = slot < NUM_SLOTS
    if HAS_VALID:
        ok = ok & (tl.load(Valid + n) != 0)
    rows = tile * TILE_V + tl.arange(0, TILE_V)
    cols = tl.arange(0, K)
    offsets = (
        slot * STRIDE_SLOT
        + head * STRIDE_HEAD
        + rows[:, None] * STRIDE_V
        + cols[None, :] * STRIDE_K
    )
    out_offsets = (
        slot * OUT_STRIDE_SLOT
        + head * OUT_STRIDE_HEAD
        + rows[:, None] * OUT_STRIDE_V
        + cols[None, :] * OUT_STRIDE_K
    )
    mask = ok & (rows[:, None] < V)
    x = tl.load(State + offsets, mask=mask, other=0.0).to(tl.float32)
    if AXIS_K:
        # Blocks of BLOCK consecutive key columns inside each value row.
        blocks = tl.reshape(tl.abs(x), (TILE_V, K // BLOCK, BLOCK))
        amax = tl.max(blocks, axis=2)
        if HAS_CAP:
            amax = _capped_amax(amax, tl.load(Cap + head), CAP_P)
        scale = _block_scale(amax)
        a = tl.minimum(tl.div_rn(blocks, tl.broadcast_to(scale[:, :, None], blocks.shape)), 6.0)
        a = tl.reshape(a, (TILE_V, K))
        scale_full = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (TILE_V, K // BLOCK, BLOCK)), (TILE_V, K)
        )
    else:
        # Blocks of BLOCK value rows (TILE_V == BLOCK) in each key column.
        absx = tl.abs(x)
        amax = tl.max(absx, axis=0)
        if HAS_CAP:
            amax = _capped_amax(amax, tl.load(Cap + head), CAP_P)
        scale_col = _block_scale(amax)
        scale_full = tl.broadcast_to(scale_col[None, :], (TILE_V, K))
        a = tl.minimum(tl.div_rn(absx, scale_full), 6.0)
    index = (
        (a >= 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a >= 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a >= 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a >= 5.0).to(tl.int32)
    )
    q = _round_to_grid(a, index, Seed, out_offsets, ROUND, ROUNDS)
    out = tl.where(x < 0, -(q * scale_full), q * scale_full)
    tl.store(Out + out_offsets, out.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _cap_grad_kernel(
    State,
    Grad,
    Cap,
    GradCap,  # [N, TILES, H] per-program partial sums; the host adds them in a fixed order
    Seed,
    V: tl.constexpr,
    K: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    STRIDE_HEAD: tl.constexpr,
    STRIDE_V: tl.constexpr,
    STRIDE_K: tl.constexpr,
    G_STRIDE_SLOT: tl.constexpr,
    G_STRIDE_HEAD: tl.constexpr,
    G_STRIDE_V: tl.constexpr,
    G_STRIDE_K: tl.constexpr,
    OUT_STRIDE_SLOT: tl.constexpr,
    OUT_STRIDE_HEAD: tl.constexpr,
    OUT_STRIDE_V: tl.constexpr,
    OUT_STRIDE_K: tl.constexpr,
    CAP_P: tl.constexpr,
    AXIS_K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    ROUNDS: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Per-program partial of the gradient of ``sum(Grad * y)`` with respect to the head's cap.

    LSQ-style (learned step size quantization): the rounding is treated as the identity, so with
    ``y = scale * Q(clip(x / scale, -6, 6))`` and ``z = x / scale`` an unclipped element has ``dy/dscale = Q(z) - z``
    and a clipped one ``+-6``, and the cap reaches ``y`` only through ``scale = capped_amax / 6``. With the smooth
    minimum ``dscale/dcap = (1/6) * (capped_amax / cap) ** (P + 1)``, which is nonzero for every block, also those
    with amax above the cap, but is small unless the block's amax is within a few percent of the cap (``0.5 ** 33``
    at amax = cap / 2); the hard minimum (``P == 0``) is the indicator ``amax > cap``. ``Q`` is the rounding the
    forward used (with stochastic rounding the same noise, from the same seed).
    """
    n = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    rows = tile * TILE_V + tl.arange(0, TILE_V)
    cols = tl.arange(0, K)
    offsets = (
        n.to(tl.int64) * STRIDE_SLOT
        + head * STRIDE_HEAD
        + rows[:, None] * STRIDE_V
        + cols[None, :] * STRIDE_K
    )
    g_offsets = (
        n.to(tl.int64) * G_STRIDE_SLOT
        + head * G_STRIDE_HEAD
        + rows[:, None] * G_STRIDE_V
        + cols[None, :] * G_STRIDE_K
    )
    mask = rows[:, None] < V
    x = tl.load(State + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Grad + g_offsets, mask=mask, other=0.0).to(tl.float32)
    cap = tl.load(Cap + head)
    if AXIS_K:
        blocks = tl.reshape(tl.abs(x), (TILE_V, K // BLOCK, BLOCK))
        amax = tl.max(blocks, axis=2)
    else:
        absx = tl.abs(x)
        amax = tl.max(absx, axis=0)
    capped = _capped_amax(amax, cap, CAP_P)
    if CAP_P == 0:
        factor = (amax > cap).to(tl.float32)
    else:
        factor = tl.exp((CAP_P + 1) * tl.log(tl.maximum(capped / cap, 1e-30)))
    scale = _block_scale(capped)
    factor = tl.where(capped * SIXTH > 0, factor, 0.0)
    if AXIS_K:
        scale_full = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (TILE_V, K // BLOCK, BLOCK)), (TILE_V, K)
        )
        factor_full = tl.reshape(
            tl.broadcast_to(factor[:, :, None], (TILE_V, K // BLOCK, BLOCK)), (TILE_V, K)
        )
    else:
        scale_full = tl.broadcast_to(scale[None, :], (TILE_V, K))
        factor_full = tl.broadcast_to(factor[None, :], (TILE_V, K))
    a = tl.div_rn(tl.abs(x), scale_full)
    clipped = a > 6.0
    a_c = tl.minimum(a, 6.0)
    index = (
        (a_c >= 0.25).to(tl.int32)
        + (a_c >= 0.75).to(tl.int32)
        + (a_c >= 1.25).to(tl.int32)
        + (a_c >= 1.75).to(tl.int32)
        + (a_c >= 2.5).to(tl.int32)
        + (a_c >= 3.5).to(tl.int32)
        + (a_c >= 5.0).to(tl.int32)
    )
    sign = tl.where(x < 0, -1.0, 1.0)
    out_offsets = (
        n.to(tl.int64) * OUT_STRIDE_SLOT
        + head * OUT_STRIDE_HEAD
        + rows[:, None] * OUT_STRIDE_V
        + cols[None, :] * OUT_STRIDE_K
    )
    q = _round_to_grid(a_c, index, Seed, out_offsets, ROUND, ROUNDS)
    d_scale = sign * tl.where(clipped, 6.0, q - a)
    # Subnormal scales can make the unused division NaN on hardware that flushes
    # denormals. A zero cap derivative contributes exactly zero, never NaN * 0.
    contribution = tl.where(mask & (factor_full > 0), g * d_scale * SIXTH * factor_full, 0.0)
    tl.store(
        GradCap + (n * tl.num_programs(2) + tile) * tl.num_programs(1) + head,
        tl.sum(tl.sum(contribution, axis=1), axis=0),
    )
