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
"""Generate result visualizations for Nemotron-Nano-9B-v2 → Pruned 7B.

Produces three figures in ./figures/:
  learning_curves.png   — per-benchmark score vs. training tokens (2×4 small multiples)
  delta_comparison.png  — score gap: 7B@80B vs 9B, compared to 9B vs 12B
  radar_chart.png       — capability profile across four model snapshots

Usage:
    python plot_results.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette (Paul Tol colorblind-safe) ────────────────────────────────────────
C_7B = "#4477AA"  # blue  — 7B main line
C_9B = "#EE6677"  # rose  — official 9B reference
C_12B = "#AAAAAA"  # grey  — official 12B reference

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "#F9F9F9",
        "figure.facecolor": "white",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#444444",
        "text.color": "#333333",
    }
)

# ── Data ──────────────────────────────────────────────────────────────────────
CHECKPOINTS = ["2.5B", "20B", "40B", "60B", "80B"]

SCORES = {
    "MMLU": [70.7, 71.3, 71.1, 72.1, 72.2],
    "MMLU Pro": [68.4, 71.7, 71.6, 72.1, 73.0],
    "GPQA": [52.7, 54.8, 53.7, 54.9, 56.9],
    "LCB v6": [57.0, 62.0, 60.9, 61.6, 62.6],
    "AIME 2025": [63.0, 69.1, 70.4, 70.3, 72.0],
    "Math 500": [93.7, 95.2, 95.6, 95.4, 95.8],
    "IFEval": [63.2, 63.8, 68.0, 64.7, 66.2],
    "SciCode": [11.6, 20.9, 21.1, 24.1, 22.2],
}

OFFICIAL_9B = {
    "MMLU": 74.7,
    "MMLU Pro": 74.9,
    "GPQA": 56.1,
    "LCB v6": 64.4,
    "AIME 2025": 73.2,
    "Math 500": 95.9,
    "IFEval": 65.8,
    "SciCode": 21.9,
}
OFFICIAL_12B = {
    "MMLU": 78.5,
    "MMLU Pro": 77.9,
    "GPQA": 58.2,
    "LCB v6": 66.6,
    "AIME 2025": 76.1,
    "Math 500": 96.9,
    "IFEval": 67.9,
    "SciCode": 28.4,
}

BENCHMARKS = list(SCORES.keys())
X = np.arange(len(CHECKPOINTS))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 -- Learning Curves (2 x 4 small multiples)
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    axes = axes.flatten()

    for i, (bm, ax) in enumerate(zip(BENCHMARKS, axes)):
        scores = SCORES[bm]
        ref9 = OFFICIAL_9B[bm]
        ref12 = OFFICIAL_12B[bm]

        lo = min([*scores, ref9, ref12])
        hi = max([*scores, ref9, ref12])
        pad = (hi - lo) * 0.16

        # Reference lines (drawn first so they sit beneath the main line)
        ax.axhline(ref12, color=C_12B, lw=1.3, ls=":", zorder=1, alpha=0.9)
        ax.axhline(ref9, color=C_9B, lw=1.5, ls="--", zorder=2, alpha=0.9)

        # Light area fill under curve
        ax.fill_between(X, scores, lo - pad, alpha=0.10, color=C_7B, zorder=2)

        # Main trajectory
        ax.plot(
            X,
            scores,
            color=C_7B,
            lw=2.2,
            marker="o",
            ms=5.5,
            zorder=4,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=C_7B,
        )

        ax.set_title(bm, pad=5)
        ax.set_xticks(X)
        ax.set_xticklabels(CHECKPOINTS, fontsize=8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_color("#E0E0E0")

        if i % 4 == 0:
            ax.set_ylabel("Score", labelpad=4)

    # Shared legend
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=C_7B,
            lw=2.2,
            marker="o",
            ms=6,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=C_7B,
            label="Pruned 7B (ours)",
        ),
        plt.Line2D([0], [0], color=C_9B, lw=1.5, ls="--", label="Official Nano-9B-v2"),
        plt.Line2D([0], [0], color=C_12B, lw=1.3, ls=":", label="Official Nano-12B-v2"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=9.5,
        columnspacing=1.5,
    )
    fig.suptitle(
        "Benchmark Recovery During Knowledge Distillation  ·  Nemotron-Nano-9B-v2 → Pruned 7B",
        fontsize=13,
        fontweight="bold",
        y=1.10,
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "learning_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_learning_curves()
    print(f"\nFigure saved to: {FIGURES_DIR}")
