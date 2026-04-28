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

# Research scratch script — relax some style rules that don't add value here.
# ruff: noqa: D103, RUF003

"""MXFP4 -> NVFP4 conversion MSE experiment.

Compares two algorithms for converting an MXFP4 tensor (block_size=32, E2M1 + E8M0
power-of-2 scales) to NVFP4 (block_size=16, E2M1 + E4M3 scales + global FP32 scale):

  Algo 1 (dequant-requant): dequantize MXFP4 to BF16, then quantize to NVFP4 the
    standard way. This re-buckets nibbles and computes new scales from scratch.

  Algo 2 (verbatim nibbles): keep the E2M1 nibbles unchanged. Each MXFP4 block of 32
    splits into two NVFP4 blocks of 16, both inheriting the same exponent k_j.
    Pick a global scale S = 2^m (integer m) and store the per-block E4M3 scale as
    2^(k_j - m). E4M3 exactly represents 2^k for k in [-9, 8], so as long as
    max(k) - min(k) <= 17 there is a valid m and the conversion is exact (zero
    MSE). For blocks outside that window, snap the per-block exponent to the
    [-9, 8] boundary; nibbles stay verbatim, and that snap is provably MSE-optimal
    given the constraint.

Reference for both algos: the MXFP4-dequantized tensor (i.e. what the source
representation faithfully encodes). MSE is computed against that reference in fp32.
"""

import math

import torch

from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MX_BLOCK = 32
NV_BLOCK = 16
E4M3_KMIN, E4M3_KMAX = -9, 8  # E4M3 represents 2^k exactly for k in [-9, 8]

# E2M1 magnitude squared, indexed by nibble bits (sign bit ignored — squared anyway).
# Sign bit is the high bit (0b1000); low 3 bits are the magnitude index into
# [0, 0.5, 1, 1.5, 2, 3, 4, 6]. Squared magnitude lookup for all 16 nibble values:
_E2M1_SQ = torch.tensor(
    [0.0, 0.25, 1.0, 2.25, 4.0, 9.0, 16.0, 36.0, 0.0, 0.25, 1.0, 2.25, 4.0, 9.0, 16.0, 36.0],
    dtype=torch.float32,
)


# ---------- Algorithm 1: dequant -> requant ----------------------------------


def algo1_dequant_requant(mxfp4_qt: MXFP4QTensor, e8m0_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 then quantize to NVFP4 the normal way; return float32 reconstruction."""
    deq_bf16 = mxfp4_qt.dequantize(
        dtype=torch.bfloat16, scale=e8m0_scale, block_sizes={-1: MX_BLOCK}
    )
    nv_qt, per_block_e4m3, double_scale = NVFP4QTensor.quantize(deq_bf16, block_size=NV_BLOCK)
    out = nv_qt.dequantize(
        dtype=torch.float32,
        scale=per_block_e4m3,
        double_scale=double_scale,
        block_sizes={-1: NV_BLOCK},
    )
    return out.float()


# ---------- Algorithm 2: keep nibbles, just rescale --------------------------


def _block_sum_sq_nibbles(
    mxfp4_qt: MXFP4QTensor,
) -> torch.Tensor:
    """For each MXFP4 block, sum of squared E2M1 magnitudes (used for closed-form MSE).

    Returns a 1D tensor of length num_blocks, in float32.
    """
    original_shape = mxfp4_qt.metadata["shape"]
    packed = mxfp4_qt._quantized_data.view(*original_shape[:-1], -1, MX_BLOCK // 2)
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    sq = _E2M1_SQ.to(packed.device)
    per_block = (sq[low] + sq[high]).sum(dim=-1)  # one entry per MXFP4 block
    return per_block.reshape(-1)


def _find_best_m(
    k_flat: torch.Tensor,
    sum_sq_flat: torch.Tensor,
    k_min: int,
    k_max: int,
) -> tuple[int, float]:
    """Sweep integer m and return (best_m, best_total_squared_error).

    Per-block squared error when verbatim nibbles are kept and scale snaps to E4M3:
        delta_j     = k_j - m
        snap_j      = clamp(delta_j, [-9, 8])
        scale_diff  = 2^k_j - 2^(m + snap_j)
        err_j       = sum_sq_j * scale_diff^2

    In-range blocks (delta_j in [-9, 8]) contribute zero. Search range is symmetric
    around the k window — outside it, every block snaps and error grows monotonically.
    """
    candidates = list(range(k_min - E4M3_KMAX - 1, k_max - E4M3_KMIN + 2))
    k_f = k_flat.float()
    pow2_k = torch.exp2(k_f)
    best_m: int = candidates[0]
    best_err: float = float("inf")
    for m_cand in candidates:
        delta = k_flat - m_cand
        snap = torch.clamp(delta, E4M3_KMIN, E4M3_KMAX)
        # snapped scale exponent: m + snap, but only differs from k when |delta| > 8/9
        snapped_scale = torch.exp2((m_cand + snap).float())
        diff = pow2_k - snapped_scale
        err = (sum_sq_flat * diff * diff).sum().item()
        if err < best_err:
            best_err = err
            best_m = m_cand
    return best_m, best_err


def algo2_keep_nibbles(
    mxfp4_qt: MXFP4QTensor,
    e8m0_scale: torch.Tensor,
    m_strategy: str = "midpoint",
) -> tuple[torch.Tensor, int, int, int]:
    """Keep MXFP4 E2M1 nibbles verbatim and rescale.

    Choose a per-tensor m (S=2^m) and per-block E4M3 scales = 2^(k_j - m),
    snapping out-of-range blocks to E4M3's boundary.

    m_strategy:
        "midpoint" — when spread <=17, any valid m gives MSE=0 (we pick midpoint).
                     When spread >17, fall back to a heuristic: median(k) - center.
        "search"   — when spread <=17, behaves like "midpoint" (already optimal).
                     When spread >17, sweep integer m and pick the value that
                     minimizes total snap error in closed form.
    """
    # Recover signed integer exponents k_j from E8M0 (stored as uint8 with bias 127).
    k = e8m0_scale.to(torch.int32) - 127

    # Identify blocks whose scale is irrelevant: all E2M1 nibbles have magnitude 0
    # (sign bit may be 0 or 1; MXFP4's cast_fp4 emits sign_bit=1 for value 0, giving
    # "negative zero" nibbles 0x08 / 0x80, so packed bytes are 0x88). Mask 0x77.
    original_shape = mxfp4_qt.metadata["shape"]
    packed = mxfp4_qt._quantized_data.view(*original_shape[:-1], -1, MX_BLOCK // 2)
    block_is_zero = ((packed & 0x77) == 0).all(dim=-1).reshape(-1)
    k_flat = k.reshape(-1)
    nonzero_mask = ~block_is_zero
    nonzero_k = k_flat[nonzero_mask] if nonzero_mask.any() else k_flat
    k_min = int(nonzero_k.min().item())
    k_max = int(nonzero_k.max().item())

    spread_fits = (k_max - k_min) <= (E4M3_KMAX - E4M3_KMIN)
    if spread_fits:
        m = (k_max - E4M3_KMAX + k_min - E4M3_KMIN + 1) // 2
        m = max(k_max - E4M3_KMAX, min(m, k_min - E4M3_KMIN))
    elif m_strategy == "search":
        sum_sq = _block_sum_sq_nibbles(mxfp4_qt)
        # zero-blocks contribute 0 to S_j so they don't affect search either way;
        # leave them in to keep shapes aligned.
        m, _ = _find_best_m(k_flat, sum_sq, k_min, k_max)
    else:
        m = int(nonzero_k.median().item()) - (E4M3_KMAX + E4M3_KMIN) // 2

    # Per-block exponent stored in the NVFP4 E4M3 scale: 2^(k_j - m), clamped to [-9, 8].
    e4m3_exp = torch.clamp(k - m, E4M3_KMIN, E4M3_KMAX)
    e4m3_scale_fp32 = torch.exp2(e4m3_exp.float())  # exact powers of 2

    # NVFP4 per-block scale lives on 16-element blocks; each MXFP4 block (32) splits
    # into two NVFP4 blocks that share the same exponent. Round-trip through fp32
    # before casting to float8_e4m3fn to avoid repeat_interleave dtype quirks.
    num_mx_blocks_per_row = original_shape[-1] // MX_BLOCK
    e4m3_scale_nv = (
        e4m3_scale_fp32.view(*original_shape[:-1], num_mx_blocks_per_row)
        .repeat_interleave(2, dim=-1)
        .contiguous()
        .to(torch.float8_e4m3fn)
    )

    # MXFP4 and NVFP4 use identical nibble packing (even idx low, odd idx high), so
    # the bytes carry over verbatim.
    nv_qt = NVFP4QTensor(original_shape, mxfp4_qt.metadata["dtype"], mxfp4_qt._quantized_data)
    double_scale = torch.tensor(float(2.0**m), device=DEVICE, dtype=torch.float32)

    out = nv_qt.dequantize(
        dtype=torch.float32,
        scale=e4m3_scale_nv,
        double_scale=double_scale,
        block_sizes={-1: NV_BLOCK},
    )
    return out.float(), m, k_min, k_max


# ---------- Algorithm 3: hybrid (verbatim where exact, NVFP4-requant elsewhere) ---


def _algo3_recon_for_m(
    deq_ref: torch.Tensor,
    e8m0_scale: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Build Algo 3's fp32 reconstruction for a given m.

    For MXFP4 blocks where (k_j - m) ∈ [-9, 8]: use the exact MXFP4 dequant value
    (zero error vs reference). For OOR blocks: dequant the block to fp32 (already
    done — that's deq_ref), then NVFP4-quantize each 16-element half with the
    fixed global scale 2^m and dequantize. The per-NVFP4-block amax can be
    smaller than the full-MXFP4-block amax, so OOR blocks at the MXFP4 level
    may still fit cleanly into E4M3 per-NVFP4-block scales.
    """
    scale_2 = torch.tensor(float(2.0**m), device=deq_ref.device, dtype=torch.float32)
    nv_qt, pb_scale, _ = NVFP4QTensor.quantize(
        deq_ref.to(torch.bfloat16),
        block_size=NV_BLOCK,
        weights_scaling_factor_2=scale_2,
    )
    nv_recon = nv_qt.dequantize(
        dtype=torch.float32,
        scale=pb_scale,
        double_scale=scale_2,
        block_sizes={-1: NV_BLOCK},
    ).view_as(deq_ref)

    k_flat = e8m0_scale.to(torch.int32).reshape(-1) - 127
    delta = k_flat - m
    in_range = (delta >= E4M3_KMIN) & (delta <= E4M3_KMAX)  # per MXFP4 block

    deq_blocks = deq_ref.reshape(-1, MX_BLOCK)
    nv_blocks = nv_recon.reshape(-1, MX_BLOCK)
    recon_blocks = torch.where(in_range.unsqueeze(-1), deq_blocks, nv_blocks)
    return recon_blocks.view_as(deq_ref).float()


def algo3_hybrid_requant(
    mxfp4_qt: MXFP4QTensor,
    e8m0_scale: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    """Hybrid: verbatim for in-range blocks, NVFP4-requant for out-of-range blocks.

    For OOR MXFP4 blocks, dequantize and re-quantize each 16-element half with
    the fixed global scale 2^m. m is chosen to minimize the actual post-hybrid
    MSE: brute-force over the same integer range Algo 2 considers, but evaluating
    the real reconstruction error rather than a closed form (because NVFP4-requant's
    E4M3 mantissa quantization isn't a clean function of m alone).
    """
    k = e8m0_scale.to(torch.int32) - 127
    original_shape = mxfp4_qt.metadata["shape"]
    packed = mxfp4_qt._quantized_data.view(*original_shape[:-1], -1, MX_BLOCK // 2)
    block_is_zero = ((packed & 0x77) == 0).all(dim=-1).reshape(-1)
    k_flat = k.reshape(-1)
    nonzero_mask = ~block_is_zero
    nonzero_k = k_flat[nonzero_mask] if nonzero_mask.any() else k_flat
    k_min = int(nonzero_k.min().item())
    k_max = int(nonzero_k.max().item())

    deq_ref = reference_from_mxfp4(mxfp4_qt, e8m0_scale)

    # If everything fits in E4M3, midpoint m gives exact zero-error reconstruction.
    if (k_max - k_min) <= (E4M3_KMAX - E4M3_KMIN):
        m = (k_max - E4M3_KMAX + k_min - E4M3_KMIN + 1) // 2
        m = max(k_max - E4M3_KMAX, min(m, k_min - E4M3_KMIN))
        return deq_ref.float(), m, k_min, k_max

    # Otherwise, brute-force search over candidate m and pick best MSE.
    candidates = list(range(k_min - E4M3_KMAX - 1, k_max - E4M3_KMIN + 2))
    best_m: int = candidates[0]
    best_mse: float = float("inf")
    best_recon: torch.Tensor = _algo3_recon_for_m(deq_ref, e8m0_scale, best_m)
    for m_cand in candidates:
        recon = _algo3_recon_for_m(deq_ref, e8m0_scale, m_cand)
        mse_val = mse(deq_ref, recon)
        if mse_val < best_mse:
            best_mse = mse_val
            best_m = m_cand
            best_recon = recon
    return best_recon, best_m, k_min, k_max


# ---------- Reference and metrics --------------------------------------------


def reference_from_mxfp4(mxfp4_qt: MXFP4QTensor, e8m0_scale: torch.Tensor) -> torch.Tensor:
    """The true value the MXFP4 representation encodes (in fp32)."""
    return mxfp4_qt.dequantize(
        dtype=torch.float32, scale=e8m0_scale, block_sizes={-1: MX_BLOCK}
    ).float()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def snr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB. +inf when MSE=0."""
    sig = (ref.float() ** 2).mean().item()
    err = ((ref.float() - approx.float()) ** 2).mean().item()
    if err <= 0:
        return float("inf")
    if sig <= 0:
        return float("-inf")
    return 10.0 * math.log10(sig / err)


# ---------- Test scenarios ---------------------------------------------------
# Each scenario returns a tensor with last dim divisible by 32 (MXFP4 block size).
# Most are 256×1024 (8192 MXFP4 blocks) for a quick run; some test other shapes.

R, C = 256, 1024  # default rows × cols


def gen_uniform() -> torch.Tensor:
    return torch.empty(R, C, device=DEVICE, dtype=torch.bfloat16).uniform_(-1, 1)


def gen_gaussian() -> torch.Tensor:
    return (torch.randn(R, C, device=DEVICE) * 1.0).bfloat16()


def gen_heavy_tail() -> torch.Tensor:
    # x = N(0,1) * |N(0,1)|  → fatter tails than gaussian
    return (torch.randn(R, C, device=DEVICE) * torch.randn(R, C, device=DEVICE).abs()).bfloat16()


def gen_rare_outliers() -> torch.Tensor:
    x = torch.randn(R, C, device=DEVICE) * 0.05
    mask = torch.rand_like(x) < 1e-3
    x[mask] = 100.0 * torch.sign(torch.randn_like(x[mask]) + 1e-6)
    return x.bfloat16()


def gen_mixed_block_scales_25() -> torch.Tensor:
    """Each row chunk gets a different magnitude — forces wide block-exponent spread."""
    x = torch.randn(R, C, device=DEVICE) * 0.3
    n_chunks = 16
    chunk = max(R // n_chunks, 1)
    for i in range(n_chunks):
        start, end = i * chunk, R if i == n_chunks - 1 else (i + 1) * chunk
        s = 2.0 ** (-12 + (i * 25 // (n_chunks - 1)))
        x[start:end] *= s
    return x.bfloat16()


def gen_narrow_range() -> torch.Tensor:
    return (torch.randn(R, C, device=DEVICE) * 0.5 + 1.0).bfloat16()


def gen_llm_weight() -> torch.Tensor:
    # Dense linear-layer init: small std, rare outliers
    x = torch.randn(R, C, device=DEVICE) * (1.0 / math.sqrt(C))
    mask = torch.rand_like(x) < 1e-4
    x[mask] *= 50.0
    return x.bfloat16()


def gen_zero_block() -> torch.Tensor:
    x = torch.zeros(R, C, device=DEVICE)
    mask = torch.rand_like(x) < 0.01
    x[mask] = torch.randn_like(x[mask]) * 0.5
    return x.bfloat16()


# --- Wider/tighter spread tests around the 17-exponent boundary -------------


def _per_row_geom_scale(rows: int, cols: int, log2_range: int) -> torch.Tensor:
    """Each row chunk gets a power-of-2 magnitude spanning [-r/2, r/2]."""
    x = torch.randn(rows, cols, device=DEVICE) * 0.5
    n_chunks = 16
    chunk = max(rows // n_chunks, 1)
    half = log2_range // 2
    for i in range(n_chunks):
        start, end = i * chunk, rows if i == n_chunks - 1 else (i + 1) * chunk
        s = 2.0 ** (-half + (i * log2_range // (n_chunks - 1)))
        x[start:end] *= s
    return x.bfloat16()


def gen_spread_15() -> torch.Tensor:
    """Block exponent spread ≈ 15 — fits in E4M3 window, midpoint should be exact."""
    return _per_row_geom_scale(R, C, log2_range=15)


def gen_spread_17() -> torch.Tensor:
    """Block exponent spread = 17 — at the in-range boundary."""
    return _per_row_geom_scale(R, C, log2_range=17)


def gen_spread_18() -> torch.Tensor:
    """Block exponent spread = 18 — just past the boundary; midpoint loses, search wins."""
    return _per_row_geom_scale(R, C, log2_range=18)


def gen_spread_50() -> torch.Tensor:
    return _per_row_geom_scale(R, C, log2_range=50)


# --- Distribution variations ------------------------------------------------


def gen_bimodal() -> torch.Tensor:
    """Two gaussian clusters at very different magnitudes."""
    x = torch.randn(R, C, device=DEVICE) * 0.01
    mask = torch.rand_like(x) < 0.5
    x[mask] = torch.randn_like(x[mask]) * 8.0
    return x.bfloat16()


def gen_power_law() -> torch.Tensor:
    """Pareto(1.5)-like distribution — long-tailed."""
    u = torch.rand(R, C, device=DEVICE).clamp(min=1e-6)
    x = (u ** -(1.0 / 1.5) - 1.0) * torch.sign(torch.randn_like(u))
    return (x * 0.05).bfloat16()


def gen_per_row_outlier() -> torch.Tensor:
    """LLM-activation-style: a few rows are dominated by outlier columns."""
    x = torch.randn(R, C, device=DEVICE) * 0.01
    n_outlier_rows = 4
    outlier_rows = torch.randperm(R)[:n_outlier_rows]
    n_outlier_cols = max(C // 64, 1)
    outlier_cols = torch.randperm(C)[:n_outlier_cols]
    for r in outlier_rows:
        x[r, outlier_cols] = 30.0 * torch.sign(torch.randn(n_outlier_cols, device=DEVICE) + 1e-6)
    return x.bfloat16()


def gen_per_col_outlier() -> torch.Tensor:
    """Whole columns are systematically larger — like a single outlier feature."""
    x = torch.randn(R, C, device=DEVICE) * 0.01
    outlier_cols = torch.randperm(C)[: max(C // 128, 1)]
    x[:, outlier_cols] *= 200.0
    return x.bfloat16()


def gen_single_extreme() -> torch.Tensor:
    """One absurdly large value in an otherwise small tensor."""
    x = torch.randn(R, C, device=DEVICE) * 0.005
    x[R // 2, C // 2] = 1e4
    return x.bfloat16()


def gen_subnormal_heavy() -> torch.Tensor:
    """Many values smaller than E2M1's smallest representable nonzero (0.5*2^k_min)."""
    return (torch.randn(R, C, device=DEVICE) * 1e-8).bfloat16()


def gen_saturating() -> torch.Tensor:
    """Values pushed to E2M1's max boundary — stresses cast_fp4 rounding."""
    x = torch.randn(R, C, device=DEVICE)
    x = torch.sign(x) * torch.min(x.abs(), torch.tensor(6.0, device=DEVICE))
    return x.bfloat16()


def gen_mixed_signs_zero_mean() -> torch.Tensor:
    """Strongly bimodal sign distribution, near-zero mean."""
    x = torch.where(
        torch.rand(R, C, device=DEVICE) < 0.5,
        torch.full((R, C), 3.0, device=DEVICE),
        torch.full((R, C), -3.0, device=DEVICE),
    )
    x += torch.randn(R, C, device=DEVICE) * 0.1
    return x.bfloat16()


def gen_constant() -> torch.Tensor:
    """All identical values — degenerate; one block exponent, two distinct nibble values."""
    return torch.full((R, C), 1.5, device=DEVICE, dtype=torch.bfloat16)


# --- Layer-shaped LLM-like patterns ----------------------------------------


def gen_qkv_weight() -> torch.Tensor:
    """Attention QKV weight: tall, gaussian init w/ mild outliers."""
    rows, cols = 4096, 4096
    x = torch.randn(rows, cols, device=DEVICE) * (1.0 / math.sqrt(cols))
    mask = torch.rand_like(x) < 5e-5
    x[mask] *= 30.0
    return x.bfloat16()


def gen_mlp_gate_up() -> torch.Tensor:
    """MLP gate/up projection: wide & has activation-driven scale variation."""
    rows, cols = 1024, 4096
    x = torch.randn(rows, cols, device=DEVICE) * (1.0 / math.sqrt(cols))
    # A few channels have larger weights (often seen post-fine-tuning)
    hot = torch.randperm(rows)[: rows // 32]
    x[hot] *= 5.0
    return x.bfloat16()


def gen_embedding() -> torch.Tensor:
    """Embedding-style: vocab × hidden, ~N(0, 1) range with row-sparse outliers."""
    rows, cols = 2048, 1024
    x = torch.randn(rows, cols, device=DEVICE) * 0.5
    rare = torch.randperm(rows)[: rows // 256]
    x[rare] *= 20.0
    return x.bfloat16()


def gen_layernorm_gain() -> torch.Tensor:
    """LayerNorm gain vector (1D-ish, padded to 2D with cols=64 for blockability)."""
    rows = 32
    x = torch.ones(rows, 1024, device=DEVICE) + torch.randn(rows, 1024, device=DEVICE) * 0.05
    return x.bfloat16()


# --- Other shapes ----------------------------------------------------------


def gen_4d_conv() -> torch.Tensor:
    """4D conv-like weight: (oc, ic, kh, kw). Last 3 dims flattened block-wise."""
    return (torch.randn(64, 64, 4, 4, device=DEVICE) * 0.1).bfloat16().reshape(64, -1)


def gen_large_flat() -> torch.Tensor:
    """Bigger tensor to confirm scaling: 1k × 4k."""
    return (torch.randn(1024, 4096, device=DEVICE) * 0.02).bfloat16()


SCENARIOS = [
    # Original 8 (kept for continuity with earlier results)
    ("uniform [-1,1]", gen_uniform),
    ("gaussian std=1", gen_gaussian),
    ("heavy-tail", gen_heavy_tail),
    ("rare outliers (1e-3, mag=100)", gen_rare_outliers),
    ("mixed block scales (spread 25)", gen_mixed_block_scales_25),
    ("narrow range (~1.0)", gen_narrow_range),
    ("typical LLM weight", gen_llm_weight),
    ("mostly zeros, 1% nonzero", gen_zero_block),
    # Boundary tests around the 17-exponent E4M3 window
    ("spread 15 (in-range)", gen_spread_15),
    ("spread 17 (boundary)", gen_spread_17),
    ("spread 18 (just over)", gen_spread_18),
    ("spread 50 (extreme)", gen_spread_50),
    # Distribution variations
    ("bimodal magnitudes", gen_bimodal),
    ("Pareto(1.5) power-law", gen_power_law),
    ("per-row outliers", gen_per_row_outlier),
    ("per-col outliers", gen_per_col_outlier),
    ("single extreme outlier", gen_single_extreme),
    ("subnormal-heavy (1e-8)", gen_subnormal_heavy),
    ("saturating at E2M1_max", gen_saturating),
    ("strong bimodal signs", gen_mixed_signs_zero_mean),
    ("constant (degenerate)", gen_constant),
    # Layer-shaped LLM patterns
    ("QKV weight (4096x4096)", gen_qkv_weight),
    ("MLP gate/up (1024x4096)", gen_mlp_gate_up),
    ("embedding (2048x1024)", gen_embedding),
    ("LayerNorm gain", gen_layernorm_gain),
    # Other shapes
    ("conv weight 4D (64x64x4x4)", gen_4d_conv),
    ("large flat (1024x4096)", gen_large_flat),
]


# ---------- Driver -----------------------------------------------------------


def run_one(name: str, x: torch.Tensor) -> dict:
    """Quantize x to MXFP4, run all algos, return metrics."""
    # Pad/skip if last dim isn't divisible by MX_BLOCK
    if x.shape[-1] % MX_BLOCK != 0:
        raise ValueError(f"{name}: last dim {x.shape[-1]} not divisible by {MX_BLOCK}")

    # MXFP4/NVFP4 quantizers expect a 2D-ish view (block on last dim). Keep original shape;
    # the implementation views (-1, block_size) internally.
    mx_qt, e8m0 = MXFP4QTensor.quantize(x.clone(), block_size=MX_BLOCK)
    ref = reference_from_mxfp4(mx_qt, e8m0)

    out1 = algo1_dequant_requant(mx_qt, e8m0)
    mse1 = mse(ref, out1)

    out2_mid, m_mid, k_min, k_max = algo2_keep_nibbles(mx_qt, e8m0, m_strategy="midpoint")
    mse2_mid = mse(ref, out2_mid)

    out2_best, m_best, _, _ = algo2_keep_nibbles(mx_qt, e8m0, m_strategy="search")
    mse2_best = mse(ref, out2_best)

    out3, m3, _, _ = algo3_hybrid_requant(mx_qt, e8m0)
    mse3 = mse(ref, out3)

    k_int = (e8m0.to(torch.int32) - 127 - m_best).flatten()
    n_oor_algo2 = int(((k_int < E4M3_KMIN) | (k_int > E4M3_KMAX)).sum().item())
    k_int3 = (e8m0.to(torch.int32) - 127 - m3).flatten()
    n_oor_algo3 = int(((k_int3 < E4M3_KMIN) | (k_int3 > E4M3_KMAX)).sum().item())

    return {
        "name": name,
        "shape": tuple(x.shape),
        "k_range": (k_min, k_max),
        "spread": k_max - k_min,
        "m_mid": m_mid,
        "m_best": m_best,
        "m_algo3": m3,
        "n_blocks": int(e8m0.numel()),
        "n_oor": n_oor_algo2,
        "n_oor_algo3": n_oor_algo3,
        "mse_algo1": mse1,
        "mse_algo2_mid": mse2_mid,
        "mse_algo2_best": mse2_best,
        "mse_algo3": mse3,
        "snr1": snr_db(ref, out1),
        "snr2_best": snr_db(ref, out2_best),
        "snr3": snr_db(ref, out3),
        "max_err1": max_abs_err(ref, out1),
        "max_err2_best": max_abs_err(ref, out2_best),
        "max_err3": max_abs_err(ref, out3),
    }


def _fmt_snr(v: float) -> str:
    if v == float("inf"):
        return "  inf"
    if v == float("-inf"):
        return " -inf"
    return f"{v:6.1f}"


def main():
    print(f"device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)

    rows_hdr = (
        f"{'scenario':<34}"
        f"{'spread':>7}"
        f"{'m2/m3':>8}"
        f"{'oor':>9}"
        f"{'algo1 MSE':>11}"
        f"{'algo2_best':>11}"
        f"{'algo3':>11}"
        f"{'SNR1':>7}"
        f"{'SNR2':>7}"
        f"{'SNR3':>7}"
    )
    print(rows_hdr)
    print("-" * len(rows_hdr))

    win = {"algo1": 0, "algo2": 0, "algo3": 0, "tie": 0}
    n_algo3_exact = 0
    n_total = 0
    all_results = []
    for name, gen in SCENARIOS:
        x = gen()
        r = run_one(name, x)
        all_results.append(r)
        n_total += 1

        mses = {"algo1": r["mse_algo1"], "algo2": r["mse_algo2_best"], "algo3": r["mse_algo3"]}
        best_v = min(mses.values())
        winners = [k for k, v in mses.items() if v == best_v]
        if len(winners) > 1:
            win["tie"] += 1
        else:
            win[winners[0]] += 1
        if r["mse_algo3"] == 0.0:
            n_algo3_exact += 1

        oor_str = f"{r['n_oor']:>4}/{r['n_blocks']:<4}"
        m_str = f"{r['m_best']:>3}/{r['m_algo3']:<3}"
        print(
            f"{r['name']:<34}"
            f"{r['spread']:>7}"
            f"{m_str:>8}"
            f"{oor_str:>9}"
            f"{r['mse_algo1']:>11.2e}"
            f"{r['mse_algo2_best']:>11.2e}"
            f"{r['mse_algo3']:>11.2e}"
            f"{_fmt_snr(r['snr1']):>7}"
            f"{_fmt_snr(r['snr2_best']):>7}"
            f"{_fmt_snr(r['snr3']):>7}"
        )

    print()
    print(f"Summary across {n_total} scenarios:")
    print(f"  Algo 3 wins outright: {win['algo3']}")
    print(f"  Algo 2 wins outright: {win['algo2']}")
    print(f"  Algo 1 wins outright: {win['algo1']}")
    print(f"  Tied (≥ 2 algos at same MSE): {win['tie']}")
    print(f"  Algo 3 is exact (MSE=0): {n_algo3_exact}/{n_total}")
    print()

    # Losses: scenarios where algo3 is strictly worse than algo1 or algo2 (full precision)
    losses_vs_1 = [r for r in all_results if r["mse_algo3"] > r["mse_algo1"]]
    losses_vs_2 = [r for r in all_results if r["mse_algo3"] > r["mse_algo2_best"]]

    print("Cases where Algo 3 loses to Algo 1 (mse_algo3 > mse_algo1):")
    if not losses_vs_1:
        print("  (none — Algo 3 ≤ Algo 1 in every scenario)")
    else:
        print(f"  {'scenario':<34}{'algo1 MSE':>16}{'algo3 MSE':>16}{'ratio (3/1)':>14}")
        for r in losses_vs_1:
            ratio = r["mse_algo3"] / max(r["mse_algo1"], 1e-300)
            print(f"  {r['name']:<34}{r['mse_algo1']:>16.6e}{r['mse_algo3']:>16.6e}{ratio:>14.4f}")
    print()
    print("Cases where Algo 3 loses to Algo 2 (mse_algo3 > mse_algo2_best):")
    if not losses_vs_2:
        print("  (none — Algo 3 ≤ Algo 2 in every scenario)")
    else:
        print(f"  {'scenario':<34}{'algo2 MSE':>16}{'algo3 MSE':>16}{'ratio (3/2)':>14}")
        for r in losses_vs_2:
            ratio = r["mse_algo3"] / max(r["mse_algo2_best"], 1e-300)
            print(
                f"  {r['name']:<34}"
                f"{r['mse_algo2_best']:>16.6e}"
                f"{r['mse_algo3']:>16.6e}"
                f"{ratio:>14.4f}"
            )
    print()
    print("Notes:")
    print(" - Reference: MXFP4 dequantized tensor.")
    print(" - m2/m3:   global-scale exponent picked by algo2 / algo3 (may differ).")
    print(" - oor:     MXFP4 blocks whose (k - m_best) is outside [-9, 8] for algo2.")
    print(" - algo3:   verbatim where in-range, NVFP4-requant (with fixed scale_2=2^m3)")
    print("            where out-of-range, with m3 chosen by direct-MSE 1D sweep.")


if __name__ == "__main__":
    main()
