#!/usr/bin/env python3
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

"""Analyze the per-quantizer stats dumped by NVFP4ActMaxCalibrator (NVFP4_ACT_MAX_STATS_PATH).

Confirms the up_proj/down_proj asymmetry: does down_proj have a heavy tail
(literal_max >> p99.99) that makes act_max land below the literal max (ref)?

Usage: python analyze_act_max_stats.py STATS_JSON [OUT_HTML]
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np

KEY = re.compile(
    r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.input_quantizer$"
)


def load(path):
    """Load the stats JSON into ``{proj: {layer: [per-expert stat dicts]}}``."""
    raw = json.load(open(path))
    # proj -> layer -> list of per-expert stat dicts
    out = {"up": defaultdict(list), "down": defaultdict(list)}
    for name, st in raw.items():
        m = KEY.match(name)
        if m and st:
            out[m.group(3)][int(m.group(1))].append(st)
    return out


def col(stats, key):
    """Extract one stat key across a list of stat dicts as a float array (drops None)."""
    return np.array([s[key] for s in stats if s.get(key) is not None], dtype=float)


def summarize(proj_data, proj):
    """Print aggregate tail / dynamic-range / term stats for one projection."""
    allst = [s for layer in proj_data[proj].values() for s in layer]
    tail = col(allst, "tail_literal_over_p99_99")
    dyn = col(allst, "dyn_range_p99_99_over_p1")
    terms = defaultdict(int)
    for s in allst:
        terms[s.get("term", "?")] += 1
    g = col(allst, "g_amax")
    lit = col(allst, "literal_max")
    below = int((g < lit).sum())
    print(f"\n=== {proj}_proj  ({len(allst)} expert-quantizers) ===")
    print(
        f"  tail literal_max/p99.99 : median {np.median(tail):.2f}  "
        f"p90 {np.percentile(tail, 90):.2f}  max {tail.max():.2f}"
    )
    print(f"  dyn range p99.99/p1     : median {np.median(dyn):.2f}  max {dyn.max():.2f}")
    print(f"  g_amax < literal_max    : {below}/{len(allst)} experts")
    print("  term that set g_amax    : " + ", ".join(f"{k}={v}" for k, v in sorted(terms.items())))


FP8_RANGE = 28672.0


def g_amax_for(bmin, bmax, rho=16384.0, margin=1.0):
    """Recompute the recipe g_amax for a given B_min anchor (b_max held at p99.99)."""
    if not bmin or not bmax or bmin <= 0 or bmax <= 0:
        return None, "na"
    if bmax / bmin > FP8_RANGE:
        return bmax, "guardrail"  # range exceeds format -> no-saturation fallback
    cand_rho, cand_floor = rho * bmin, margin * bmax
    return (max(cand_rho, cand_floor), "rho*B_min" if cand_rho >= cand_floor else "margin*B_max")


def counterfactual(data, proj):
    """How g_amax / term / act-vs-ref change if B_min = p1 vs p3 vs p5."""
    allst = [s for layer in data[proj].values() for s in layer]
    print(
        f"\n=== {proj}_proj — counterfactual B_min anchor (rho=16384, b_max=p99.99, ref=literal_max) ==="
    )
    print(
        f"  {'B_min':>6} {'guardrail':>11} {'rho*B_min':>10} {'margin*Bmax':>12} "
        f"{'med g_amax':>11} {'med g/lit':>10} {'act>=ref':>9}"
    )
    for pk in ("p1", "p3", "p5"):
        gs, ratios, terms, ge = [], [], defaultdict(int), 0
        n = 0
        for s in allst:
            g, t = g_amax_for(s.get(pk), s.get("p99_99"), rho=(s.get("rho") or 16384.0))
            lit = s.get("literal_max")
            if g is None:
                continue
            n += 1
            gs.append(g)
            terms[t] += 1
            if lit:
                ratios.append(g / lit)
                if g >= lit:
                    ge += 1
        gs, ratios = np.array(gs), np.array(ratios)
        print(
            f"  {pk:>6} {terms['guardrail']:>11} {terms['rho*B_min']:>10} {terms['margin*B_max']:>12} "
            f"{np.median(gs):>11.1f} {np.median(ratios):>10.2f} {ge:>6}/{n}"
        )


def per_layer_table(proj_data, proj):
    """Return per-layer medians: (layer, literal_max, p99.99, p1, tail, g_amax)."""
    rows = []
    for layer in sorted(proj_data[proj]):
        st = proj_data[proj][layer]
        rows.append(
            (
                layer,
                np.median(col(st, "literal_max")),
                np.median(col(st, "p99_99")),
                np.median(col(st, "p1")),
                np.median(col(st, "tail_literal_over_p99_99")),
                np.median(col(st, "g_amax")),
            )
        )
    return rows


def main():
    """CLI entry point: print summaries, counterfactuals, and per-layer tables."""
    path = sys.argv[1]
    data = load(path)
    for proj in ("up", "down"):
        summarize(data, proj)
    for proj in ("up", "down"):
        counterfactual(data, proj)

    print("\n=== down_proj per-layer medians (across experts) ===")
    print(
        f"{'layer':>5} {'literal_max':>12} {'p99.99':>10} {'p1':>10} {'tail(lit/p99.99)':>16} {'g_amax':>10}"
    )
    for layer, lit, p99, p1, tail, g in per_layer_table(data, "down"):
        print(f"{layer:>5} {lit:>12.1f} {p99:>10.1f} {p1:>10.3f} {tail:>16.2f} {g:>10.1f}")

    print("\n=== up_proj per-layer medians (across experts) ===")
    print(
        f"{'layer':>5} {'literal_max':>12} {'p99.99':>10} {'p1':>10} {'tail(lit/p99.99)':>16} {'g_amax':>10}"
    )
    for layer, lit, p99, p1, tail, g in per_layer_table(data, "up"):
        print(f"{layer:>5} {lit:>12.1f} {p99:>10.1f} {p1:>10.3f} {tail:>16.2f} {g:>10.1f}")

    if len(sys.argv) > 2:
        out = sys.argv[2]
        html = [
            "<!doctype html><meta charset=utf-8><title>act_max calib stats</title>",
            "<style>body{font-family:sans-serif;margin:20px}table{border-collapse:collapse;font-size:13px}",
            "td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}</style>",
            "<h2>act_max calibration stats (per-expert, medians per layer)</h2>",
        ]
        for proj in ("down", "up"):
            html.append(
                f"<h3>{proj}_proj</h3><table><tr><th>layer</th><th>literal_max</th><th>p99.99</th>"
                "<th>p1</th><th>tail lit/p99.99</th><th>g_amax</th></tr>"
            )
            for layer, lit, p99, p1, tail, g in per_layer_table(data, proj):
                html.append(
                    f"<tr><td>{layer}</td><td>{lit:.1f}</td><td>{p99:.1f}</td>"
                    f"<td>{p1:.3f}</td><td>{tail:.2f}</td><td>{g:.1f}</td></tr>"
                )
            html.append("</table>")
        open(out, "w").write("\n".join(html))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
