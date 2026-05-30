# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Study NVFP4 quantize->dequantize error vs. the choice of global amax (g_amax).

Reproduces the figures in this directory's README:
  Part 1: prove the hand-derived math matches the REAL NVFP4QTensor code path.
  Part 2: lock (e, b_amax), sweep g_amax, plot signed error dequant(quant(e)) - e.
  Part 3: relative FP8 block-scale error vs. b_amax/g_amax (shows all regimes).
  Part 4: activation calibration robustness — B_max- vs B_min-anchored g_amax.

Run from anywhere:  python nvfp4_global_scale_study.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 16
torch.set_printoptions(precision=8)


def build_block(e: float, b_amax: float) -> torch.Tensor:
    """One block of 16 elements whose abs-max is exactly b_amax and that contains e.

    Element 0 = e, element 1 = +b_amax (forces the block amax), rest = 0.
    Requires |e| <= b_amax.
    """
    assert abs(e) <= b_amax + 1e-12, "need |e| <= b_amax"
    blk = torch.zeros(1, BLOCK, dtype=torch.float32)
    blk[0, 0] = e
    blk[0, 1] = b_amax
    return blk


def real_code_qdq(e: float, b_amax: float, g_amax: float):
    """Run the actual NVFP4QTensor quantize + dequantize. Returns (deq_e, block_scale_fp8, prod)."""
    blk = build_block(e, b_amax)
    # Inject g_amax independently of the tensor's own amax via weights_scaling_factor_2.
    wsf2 = torch.tensor(g_amax / (6.0 * 448.0), dtype=torch.float32)

    qtensor, wsf, wsf2_out = NVFP4QTensor.quantize(
        blk, block_size=BLOCK, weights_scaling_factor_2=wsf2
    )
    deq = qtensor.dequantize(
        dtype=torch.float32,
        scale=wsf,  # per-block scale (fp8 e4m3)
        double_scale=wsf2_out,  # global / per-tensor scale
        block_sizes={-1: BLOCK},
    )
    deq_e = deq[0, 0].item()
    block_scale_fp8 = wsf.float().flatten()[0].item()
    prod = block_scale_fp8 * wsf2_out.item()  # effective divisor used in (de)quant
    return deq_e, block_scale_fp8, prod


def manual_qdq(e: float, b_amax: float, g_amax: float):
    """Hand-derived math, reusing only the e2m1 snapping primitive (_cast_fp4)."""
    global_scale = g_amax / (6.0 * 448.0)
    # block_scale (high precision) = b_amax / (6 * global_scale) = b_amax*448/g_amax
    block_scale_hp = b_amax / (6.0 * global_scale)
    # stored as fp8 e4m3, clamped to [2**-9, 448]
    block_scale_fp8 = (
        torch.tensor(block_scale_hp, dtype=torch.float32)
        .clamp(min=2**-9, max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .item()
    )
    prod = block_scale_fp8 * global_scale  # effective divisor
    scaled = torch.tensor([[e / prod]], dtype=torch.float32)

    # snap to e2m1 grid using the SAME primitive the library uses
    codes = NVFP4QTensor._cast_fp4(scaled.clone())
    snapped = NVFP4QTensor.get_e2m1_values("cpu")[codes.long()].flatten()[0].item()
    deq_e = snapped * prod
    return deq_e, block_scale_fp8, prod


# ----------------------------------------------------------------------------
# PART 1 — numeric proof on concrete scenarios
# ----------------------------------------------------------------------------
print("=" * 100)
print("PART 1: real NVFP4 code vs. hand-derived math")
print("=" * 100)

scenarios = [
    # (e,     b_amax, g_amax)
    (0.37, 0.50, 1.0),
    (0.37, 0.50, 0.5),  # g_amax == b_amax (block is the global max block)
    (0.37, 0.50, 4.0),  # large g_amax -> tiny block scale, fp8 precision loss
    (0.37, 0.50, 0.05),  # g_amax < b_amax -> block scale wants >448, gets clamped
    (-2.9, 3.0, 6.0),
    (1.234, 5.0, 5.0),
    (0.018, 0.02, 12.0),  # extreme: block scale underflow toward 2**-9 clamp
    # ---- small e + large b_amax, sweeping the ratio r = e/b_amax (g_amax = b_amax) ----
    (3.0, 6.0, 6.0),  # r = 1/2   -> scaled 3.0  (on-grid)
    (1.5, 6.0, 6.0),  # r = 1/4   -> scaled 1.5  (on-grid)
    (0.75, 6.0, 6.0),  # r = 1/8   -> scaled 0.75 (tie bound, round-to-even)
    (0.30, 6.0, 6.0),  # r = 1/20  -> scaled 0.30 -> snaps to 0.5, big rel err
    (0.10, 6.0, 6.0),  # r = 1/60  -> scaled 0.10 -> snaps to 0,   total loss
    (0.02, 6.0, 6.0),  # r = 1/300 -> scaled 0.02 -> snaps to 0,   total loss
    (0.30, 30.0, 30.0),  # tiny ratio, large b_amax (same r=1/100 behaviour)
    (0.30, 30.0, 60.0),  # same but g_amax 2x b_amax -> fp8 block-scale error adds
]

hdr = (
    f"{'e':>8} {'b_amax':>7} {'g_amax':>7} | {'deq(real)':>12} {'deq(manual)':>12} | "
    f"{'bscale_fp8':>11} {'prod':>12} | {'err(real)':>11} {'match?':>7}"
)
print(hdr)
print("-" * len(hdr))
all_match = True
for e, b, g in scenarios:
    dr, bs_r, pr = real_code_qdq(e, b, g)
    dm, bs_m, pm = manual_qdq(e, b, g)
    err = dr - e  # signed error (deq - e)
    match = abs(dr - dm) < 1e-6 and abs(pr - pm) <= 1e-6 * max(1.0, abs(pr))
    all_match &= match
    print(
        f"{e:>8.4f} {b:>7.3f} {g:>7.3f} | {dr:>12.6f} {dm:>12.6f} | "
        f"{bs_r:>11.5f} {pr:>12.8f} | {err:>11.6f} {match!s:>7}"
    )

print("-" * len(hdr))
print(f"ALL real==manual: {all_match}")
print()
print(
    "Note prod = block_scale_fp8 * global_scale; analytically prod -> b_amax/6 "
    "when no fp8 clamp/rounding."
)
for e, b, g in scenarios:
    _, _, pr = real_code_qdq(e, b, g)
    print(
        f"  b_amax={b:>5.3f}: prod={pr:.8f}  vs  b_amax/6={b / 6:.8f}  (ratio {pr / (b / 6):.5f})"
    )

# ----------------------------------------------------------------------------
# PART 2 — lock (e, b_amax), sweep g_amax, plot error
# ----------------------------------------------------------------------------
print()
print("=" * 100)
print("PART 2: error vs g_amax (e and b_amax locked)")
print("=" * 100)

cases = [
    (0.37, 0.50),
    (0.123, 2.00),
    (1.70, 2.00),
    # small e + large b_amax (various ratios)
    (0.30, 6.00),  # r = 1/20 -> in the e2m1 dead zone for moderate g_amax
    (0.02, 6.00),  # r = 1/300 -> deep dead zone
    (0.30, 30.00),  # tiny ratio, large b_amax
    # very large b_amax = 1000, e from 100 / 500 / 900
    (100.0, 1000.0),  # r = 0.1
    (500.0, 1000.0),  # r = 0.5
    (900.0, 1000.0),  # r = 0.9
]

g_grid = torch.logspace(-2, 6, 800).tolist()  # g_amax from 0.01 to 1e6

# One subplot per (e, b_amax) case.
ncols = 3
nrows = (len(cases) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
for idx, (e, b) in enumerate(cases):
    ax = axes[idx // ncols][idx % ncols]
    errs = []
    scale_rel_errs = []  # RELATIVE fp8 block-scale error: (fp8(x) - x) / x, x = b*448/g
    for g in g_grid:
        dr, bs, _ = real_code_qdq(e, b, g)
        errs.append(dr - e)  # signed error (deq - e)
        ideal_block_scale = b * 448.0 / g
        scale_rel_errs.append((bs - ideal_block_scale) / ideal_block_scale)
    ax.plot(g_grid, errs, ".", color=f"C{idx}", markersize=3, label="deq - e")
    # RELATIVE FP8 block-scale quant error on a secondary axis (this is what perturbs prod)
    ax_r = ax.twinx()
    ax_r.plot(
        g_grid,
        scale_rel_errs,
        ".",
        color="gray",
        markersize=2,
        alpha=0.7,
        label="(fp8(bscale)-bscale)/bscale",
    )
    ax_r.set_ylabel("rel. fp8 bscale err", color="gray", fontsize=8)
    ax_r.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax_r.axhline(0, color="gray", ls="-", lw=0.4, alpha=0.5)
    ax.axvline(b, color="gray", ls=":", lw=0.8)  # natural choice g_amax = b_amax
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("g_amax  [log]")
    ax.set_ylabel("deq - e  (signed err)")
    ax.set_title(f"e={e}, b_amax={b}")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_zorder(ax_r.get_zorder() + 1)  # keep deq-e curve on top
    ax.patch.set_visible(False)
    lines = ax.get_lines()[:1] + ax_r.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="best")
# Hide any unused panels.
for idx in range(len(cases), nrows * ncols):
    axes[idx // ncols][idx % ncols].axis("off")
fig.suptitle("NVFP4 quantize→dequantize signed error vs. choice of global amax")
fig.tight_layout()
out = os.path.join(HERE, "error_vs_gamax.png")
fig.savefig(out, dpi=130)
print(f"saved plot -> {out}")

# Also print a small table for one case so the trend is visible in text.
e, b = cases[0]
print(f"\nSample (e={e}, b_amax={b}):")
print(f"{'g_amax':>10} {'block_scale_fp8':>16} {'prod':>12} {'deq_e':>12} {'signed_err':>12}")
for g in [0.01, 0.05, 0.1, b, 0.7, 1.0, 2.0, 10.0, 50.0, 100.0]:
    dr, bs, pr = real_code_qdq(e, b, g)
    print(f"{g:>10.3f} {bs:>16.6f} {pr:>12.8f} {dr:>12.6f} {dr - e:>12.6f}")

# ----------------------------------------------------------------------------
# PART 3 — relative FP8 block-scale error vs the ratio b_amax / g_amax.
#          block_scale (ideal) = b_amax*448/g_amax = 448 * (b_amax/g_amax), so the
#          relative quant error depends ONLY on t = b_amax/g_amax -> one curve that
#          cleanly shows every regime (saturation / normal / subnormal / underflow).
# ----------------------------------------------------------------------------
print()
print("=" * 100)
print("PART 3: relative FP8 block-scale error  (fp8(bscale)-bscale)/bscale  vs  b_amax/g_amax")
print("=" * 100)

# FP8-E4M3FN landmarks and the resulting regime boundaries in t = b_amax/g_amax.
FP8_MAX = 448.0  # max normal
FP8_MIN_NORMAL = 2.0**-6  # min normal
FP8_MIN_SUBNORMAL = 2.0**-9  # min subnormal == lower clamp
T_SAT = FP8_MAX / FP8_MAX  # = 1.0          : bscale hits 448 (upper clamp) for t > 1
T_SUBNORMAL = FP8_MIN_NORMAL / FP8_MAX  # = 1/28672     : normal -> subnormal
T_LOWER_CLAMP = FP8_MIN_SUBNORMAL / FP8_MAX  # = 1/229376    : subnormal -> lower clamp


def block_scale_rel_err(t: float) -> float:
    """(fp8(bscale) - bscale)/bscale for ideal bscale = 448*t, using the library clamp+cast."""
    bscale = FP8_MAX * t
    fp8 = (
        torch.tensor(bscale, dtype=torch.float32)
        .clamp(min=FP8_MIN_SUBNORMAL, max=FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        .item()
    )
    return (fp8 - bscale) / bscale


t_grid = torch.logspace(-7, 2, 1500).tolist()  # b_amax/g_amax from 1e-7 to 1e2
rel_errs = [block_scale_rel_err(t) for t in t_grid]

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
ax2.plot(t_grid, rel_errs, ".", color="C0", markersize=2.5)
ax2.axhline(0, color="black", lw=0.8)

# Regime boundaries + shaded normal zone.
for tb, lbl in [
    (T_SAT, "t=1  (upper clamp)"),
    (T_SUBNORMAL, "t=1/28672"),
    (T_LOWER_CLAMP, "t=1/229376"),
]:
    ax2.axvline(tb, color="gray", ls="--", lw=0.9)
    ax2.text(
        tb,
        0.92,
        lbl,
        rotation=90,
        va="top",
        ha="right",
        fontsize=7,
        transform=ax2.get_xaxis_transform(),
        color="gray",
    )
ax2.axvspan(T_SUBNORMAL, T_SAT, color="green", alpha=0.07)
# Regime labels.
for xpos, txt in [
    (3.0, "saturation\n(values clipped)"),
    (3e-3, "normal FP8\n|rel err| <= 6.25%"),
    (3e-6, "subnormal"),
    (2e-7, "lower\nclamp"),
]:
    ax2.text(
        xpos, 0.5, txt, ha="center", va="center", fontsize=7.5, transform=ax2.get_xaxis_transform()
    )

ax2.set_xscale("log")
ax2.set_yscale("symlog", linthresh=0.1)
ax2.set_xlabel("b_amax / g_amax   [log scale]   (= 1 / rho)")
ax2.set_ylabel("(fp8(bscale) - bscale) / bscale   [symlog]")
ax2.set_title("NVFP4 relative FP8 block-scale quantization error across regimes")
ax2.grid(True, which="both", ls="--", alpha=0.3)
fig2.tight_layout()
out2 = os.path.join(HERE, "error_vs_ratio.png")
fig2.savefig(out2, dpi=130)
print(f"saved plot -> {out2}")

print(
    f"\n{'b_amax/g_amax':>13} {'bscale=448t':>12} {'fp8(bscale)':>12} {'rel_err':>10} {'regime':>12}"
)
for t in [1e2, 1e1, 1.0, 0.1, 1e-3, T_SUBNORMAL, 1e-5, T_LOWER_CLAMP, 1e-6, 1e-7]:
    bscale = FP8_MAX * t
    fp8 = (
        torch.tensor(bscale)
        .clamp(min=FP8_MIN_SUBNORMAL, max=FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        .item()
    )
    rel = (fp8 - bscale) / bscale
    if t > T_SAT:
        regime = "saturation"
    elif t >= T_SUBNORMAL:
        regime = "normal"
    elif t >= T_LOWER_CLAMP:
        regime = "subnormal"
    else:
        regime = "lower-clamp"
    print(f"{t:>13.3e} {bscale:>12.5g} {fp8:>12.5g} {rel:>10.4f} {regime:>12}")

# ----------------------------------------------------------------------------
# PART 4 — activation calibration: B_max-anchored vs B_min-anchored g_amax as
#          unseen inference outliers grow. Demonstrates that anchoring g_amax to
#          the stable B_min (g = rho * B_min) is robust to outliers that
#          calibration never saw, whereas B_max-anchoring saturates immediately.
# ----------------------------------------------------------------------------
print()
print("=" * 100)
print("PART 4: calibration strategy robustness  (B_max-anchored vs B_min-anchored)")
print("=" * 100)

NORMAL_WINDOW = FP8_MAX / FP8_MIN_NORMAL  # 28672: width of the normal-FP8 g_amax window


def build_tensor_from_block_amaxes(block_amaxes: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Build a (num_blocks, 16) tensor whose per-block abs-max equals block_amaxes."""
    g = torch.Generator().manual_seed(seed)
    n = block_amaxes.numel()
    x = torch.randn(n, BLOCK, generator=g)
    x = x / x.abs().amax(dim=-1, keepdim=True)  # normalize each block to amax 1
    return x * block_amaxes.view(-1, 1).float()


def quant_mse(tensor: torch.Tensor, g_amax: float) -> float:
    """NVFP4 quant->dequant MSE over all elements for a given per-tensor g_amax."""
    wsf2 = torch.tensor(g_amax / (6.0 * 448.0), dtype=torch.float32)
    qt, wsf, wsf2o = NVFP4QTensor.quantize(tensor, block_size=BLOCK, weights_scaling_factor_2=wsf2)
    deq = qt.dequantize(dtype=torch.float32, scale=wsf, double_scale=wsf2o, block_sizes={-1: BLOCK})
    return float(((deq - tensor) ** 2).mean())


# Calibration block-amax distribution: stable lognormal bulk + a few moderate outliers.
gen = torch.Generator().manual_seed(0)
n_blocks = 2000
bulk = torch.exp(torch.randn(n_blocks, generator=gen) * 0.7)  # lognormal, median ~1
bulk[:: max(1, n_blocks // 20)] *= 8.0  # ~5% moderate outlier blocks
calib_amaxes = bulk

B_min = float(torch.quantile(calib_amaxes, 0.01))  # robust floor (1st percentile)
B_max_calib = float(calib_amaxes.max())
rho = 16384.0  # leaning high, just under the 28672 cliff

g_bmin = rho * B_min  # B_min-anchored (does not depend on inference outliers)
g_bmax = 1.5 * B_max_calib  # B_max-anchored with a 1.5x margin

print(f"calib: B_min(1%)={B_min:.4g}  B_max={B_max_calib:.4g}  range={B_max_calib / B_min:.1f}x")
print(f"  B_min-anchored g_amax = {rho:.0f} * B_min = {g_bmin:.4g}")
print(f"  B_max-anchored g_amax = 1.5 * B_max   = {g_bmax:.4g}")
print(f"  normal-window width = {NORMAL_WINDOW:.0f}x ; feasible iff range < that\n")

# At inference, the outlier blocks grow by factor k (unseen during calibration).
k_grid = torch.logspace(0, 1.5, 40).tolist()  # 1x .. ~31x
mse_bmin, mse_bmax, mse_oracle = [], [], []
outlier_mask = torch.zeros(n_blocks, dtype=torch.bool)
outlier_mask[:: max(1, n_blocks // 20)] = True
for k in k_grid:
    infer_amaxes = calib_amaxes.clone()
    infer_amaxes[outlier_mask] *= k  # outliers grow at inference
    tensor = build_tensor_from_block_amaxes(infer_amaxes, seed=1)
    mse_bmin.append(quant_mse(tensor, g_bmin))
    mse_bmax.append(quant_mse(tensor, g_bmax))
    mse_oracle.append(quant_mse(tensor, float(infer_amaxes.max())))  # knows inference max

fig4, ax4 = plt.subplots(figsize=(9, 5.5))
ax4.plot(
    k_grid, mse_bmax, "o-", ms=3, color="C3", label=f"B_max-anchored (g=1.5·B_max={g_bmax:.2g})"
)
ax4.plot(
    k_grid, mse_bmin, "s-", ms=3, color="C2", label=f"B_min-anchored (g=16384·B_min={g_bmin:.2g})"
)
ax4.plot(k_grid, mse_oracle, "--", color="gray", label="oracle (g=inference B_max)")
ax4.axvline(g_bmin / B_max_calib, color="C2", ls=":", lw=0.8)
ax4.text(
    g_bmin / B_max_calib,
    0.97,
    "B_min-anchor saturates here",
    rotation=90,
    va="top",
    ha="right",
    fontsize=7,
    color="C2",
    transform=ax4.get_xaxis_transform(),
)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("inference outlier growth factor k  (B_max_infer / B_max_calib)")
ax4.set_ylabel("quantization MSE on inference data")
ax4.set_title("Robustness to unseen activation outliers: B_min- vs B_max-anchored g_amax")
ax4.legend(fontsize=8)
ax4.grid(True, which="both", ls="--", alpha=0.3)
fig4.tight_layout()
out4 = os.path.join(HERE, "calib_strategy.png")
fig4.savefig(out4, dpi=130)
print(f"saved plot -> {out4}")

print(f"\n{'k':>6} {'MSE B_max-anch':>15} {'MSE B_min-anch':>15} {'MSE oracle':>12}")
for k, mx, mn, mo in zip(k_grid, mse_bmax, mse_bmin, mse_oracle):
    if any(abs(k - kk) < 1e-9 for kk in [k_grid[0], k_grid[13], k_grid[26], k_grid[-1]]):
        print(f"{k:>6.2f} {mx:>15.6g} {mn:>15.6g} {mo:>12.6g}")
