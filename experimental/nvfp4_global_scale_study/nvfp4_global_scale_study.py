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
  Part 3: lock g_amax = b_amax, sweep the ratio r = e / b_amax (isolates the e2m1 grid).

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
# PART 3 — small e + large b_amax: error vs the ratio r = e/b_amax
#          g_amax = b_amax (the "sweet spot"), so this isolates the e2m1 grid.
# ----------------------------------------------------------------------------
print()
print("=" * 100)
print("PART 3: relative error vs ratio r = e/b_amax  (g_amax = b_amax, isolates e2m1 grid)")
print("=" * 100)

# scaled = e/(b_amax/6) = 6r ; an element at ratio r sits at 6r on the e2m1 grid.
b_amax_fixed = 8.0
ratios = torch.logspace(-3, 0, 600).tolist()  # r from 0.001 to 1.0

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
signed_errs = []
for r in ratios:
    e = r * b_amax_fixed
    dr, _, _ = real_code_qdq(e, b_amax_fixed, b_amax_fixed)
    signed_errs.append(dr - e)

ax2.plot(
    ratios,
    signed_errs,
    ".",
    color="C3",
    markersize=3,
    label=f"b_amax={b_amax_fixed}, g_amax=b_amax",
)
# annotate the e2m1 grid points in scaled space: scaled=6r => r = grid/6
for gridval in [0.5, 1, 1.5, 2, 3, 4, 6]:
    ax2.axvline(gridval / 6.0, color="gray", ls=":", lw=0.7)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xscale("log")
ax2.set_xlabel("ratio  r = e / b_amax   [log scale]")
ax2.set_ylabel("deq - e  (signed error)")
ax2.set_title("NVFP4 signed error for small e in a block with large b_amax")
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.3)
fig2.tight_layout()
out2 = os.path.join(HERE, "error_vs_ratio.png")
fig2.savefig(out2, dpi=130)
print(f"saved plot -> {out2}")

print(f"\nSample (b_amax={b_amax_fixed}, g_amax=b_amax):")
print(
    f"{'r=e/b_amax':>11} {'e':>10} {'scaled=6r':>10} {'deq_e':>12} {'signed_err':>11} {'rel_err':>10}"
)
for r in [0.5, 0.25, 0.125, 1 / 12, 1 / 20, 1 / 24, 1 / 60, 1 / 100, 1 / 300]:
    e = r * b_amax_fixed
    dr, _, pr = real_code_qdq(e, b_amax_fixed, b_amax_fixed)
    print(f"{r:>11.5f} {e:>10.5f} {6 * r:>10.4f} {dr:>12.6f} {dr - e:>11.6f} {(dr - e) / e:>10.4f}")
