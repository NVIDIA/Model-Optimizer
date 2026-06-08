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

"""Compare per-layer NVFP4 activation g_amax across checkpoints and models (config-driven).

Reads a JSON config grouping checkpoints by model and emits one self-contained HTML report:
a per-model chart (MLP-input and down_proj g_amax over all decoder layers, one line per
calibration variant), per-layer tables, a run-configuration table (recipe / dataset /
#samples / seq len), and the raw recipe YAMLs used.

g_amax = stored input_scale * 6 * 448. Keys from several architectures are handled:
  - Nemotron-H per-expert:  backbone.layers.N.mixer.experts.M.{up,down}_proj.input_scale
  - Qwen dense:             model.language_model.layers.N.mlp.{gate,up,down}_proj.input_scale
  - Qwen fused MoE:         model.language_model.layers.N.mlp.experts.{gate_up,down}_proj.input_scale
Values are aggregated (mean) per (layer, group) across experts/duplicates.

Usage:
    python compare_input_scales.py CONFIG.json OUT.html [RECIPE_BASE_DIR]
"""

import base64
import html as _html
import io
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from safetensors import safe_open

SCALE_TO_AMAX = 6.0 * 448.0  # input_scale -> g_amax
LAYER_RE = re.compile(r"layers\.(\d+)\.")
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
RECIPE_BASE_DEFAULT = "/workspace/Workspace/ammo4/modelopt_recipes"


def parse_key(k: str):
    """Map an *.input_scale key to (layer, group) where group is 'in' or 'down'; else None."""
    m = LAYER_RE.search(k)
    if not m:
        return None
    last = k[: -len(".input_scale")].split(".")[-1]
    if "down" in last:
        grp = "down"
    elif "up" in last:  # up_proj or gate_up_proj — the projection that takes the block input
        grp = "in"
    else:
        return None  # gate_proj (redundant input with up_proj), routers, etc.
    return int(m.group(1)), grp


def load_amax(ckpt: str):
    """Return {'in': {layer: mean_g_amax}, 'down': {...}} for a checkpoint."""
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    by_shard = defaultdict(list)
    for k, sh in idx.items():
        if k.endswith("input_scale"):
            by_shard[sh].append(k)
    acc = {"in": defaultdict(list), "down": defaultdict(list)}
    for sh, keys in by_shard.items():
        with safe_open(os.path.join(ckpt, sh), framework="numpy") as f:
            for k in keys:
                pk = parse_key(k)
                if not pk:
                    continue
                layer, grp = pk
                v = float(np.asarray(f.get_tensor(k)).reshape(-1)[0]) * SCALE_TO_AMAX
                acc[grp][layer].append(v)
    return {g: {ly: float(np.mean(vs)) for ly, vs in d.items()} for g, d in acc.items()}


def chart(model_name, series):
    """Render MLP-input + down_proj per-layer g_amax (one line per checkpoint); base64 PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({ly for _, d, _ in series for g in ("in", "down") for ly in d.get(g, {})})
    lo, hi = (min(layers), max(layers)) if layers else (0, 0)
    ticks = list(range(lo, hi + 1))
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for ax, grp, title in zip(
        axes, ("in", "down"), ("MLP input (up / gate_up_proj)", "down_proj (intermediate)")
    ):
        for label, d, color in series:
            dd = d.get(grp, {})
            ly = sorted(dd)
            if ly:
                ax.plot(ly, [dd[x] for x in ly], "-o", color=color, ms=4, lw=1.4, label=label)
        ax.set_yscale("log")
        ax.set_title(f"{model_name} — {title}")
        ax.set_xlabel("decoder layer index (gaps = non-quantized layers)")
        ax.set_ylabel("g_amax  (input_scale x 6 x 448)")
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks, fontsize=7)
            ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def table(series, grp):
    """Per-layer table of g_amax, one column per checkpoint, for group 'in' or 'down'."""
    layers = sorted({ly for _, d, _ in series for ly in d.get(grp, {})})
    head = "".join(f"<th>{_html.escape(lab)}</th>" for lab, _, _ in series)
    rows = []
    for ly in layers:
        cells = []
        for _, d, _ in series:
            dd = d.get(grp, {})
            cells.append(f"<td>{dd[ly]:,.2f}</td>" if ly in dd else "<td>-</td>")
        rows.append(f"<tr><td>{ly}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>layer</th>{head}</tr>{''.join(rows)}</table>"


def _series_entry(label, path, color):
    """Load one checkpoint into a (label, data, color) tuple, or None if it can't be read."""
    try:
        return (label, load_amax(path), color)
    except Exception as e:
        print(f"WARN: failed to load {path}: {e}")
        return None


def main():
    """Build the unified HTML report from a JSON study config."""
    cfg = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    recipe_base = sys.argv[3] if len(sys.argv) > 3 else RECIPE_BASE_DEFAULT

    h = [
        "<!doctype html><html><head><meta charset='utf-8'><title>NVFP4 g_amax study</title>",
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;"
        "max-width:1320px} h1{font-size:21px} h2{font-size:17px;margin-top:34px} h3{font-size:14px;margin-top:18px}"
        " table{border-collapse:collapse;font-size:12px;margin-top:6px} th,td{border:1px solid #ddd;padding:3px 7px;"
        "text-align:right} th{background:#f4f4f4} td:first-child,th:first-child{text-align:left}"
        " img{max-width:100%;border:1px solid #eee} details{margin:6px 0}"
        " pre{background:#f7f7f9;padding:8px;font-size:11px;overflow-x:auto;border:1px solid #eee}"
        " .note{background:#f7f9fc;border-left:4px solid #1f77b4;padding:10px 14px;font-size:14px}</style>",
        "</head><body>",
        "<h1>NVFP4 activation global amax (g_amax) study — per-layer comparison</h1>",
        "<div class='note'>g_amax = stored <code>input_scale &times; 6 &times; 448</code>, aggregated (mean) "
        "across MoE experts. Variants: <b>ref-max</b> (default dataset, max calib), <b>act-max-p5</b> "
        "(nvfp4_act_max, b_min_percentile=5), <b>ref-code</b> (coding calib, max), <b>ref-reasoning</b> "
        "(reasoning calib, max).</div>",
        "<h2>Run configuration</h2>",
        "<table><tr><th>model</th><th>checkpoint</th><th>recipe</th><th>calib dataset</th>"
        "<th>#samples</th><th>seq len</th></tr>",
    ]
    recipes = []
    for m in cfg:
        for c in m["checkpoints"]:
            if c["recipe"] not in recipes:
                recipes.append(c["recipe"])
            h.append(
                f"<tr><td>{_html.escape(m['model'])}</td><td>{_html.escape(c['label'])}</td>"
                f"<td>{_html.escape(c['recipe'])}</td><td>{_html.escape(str(c['dataset']))}</td>"
                f"<td>{c['samples']}</td><td>{c['seq']}</td></tr>"
            )
    h.append("</table>")

    for m in cfg:
        entries = [
            _series_entry(c["label"], c["path"], PALETTE[i % len(PALETTE)])
            for i, c in enumerate(m["checkpoints"])
        ]
        series = [e for e in entries if e is not None]
        if not series:
            continue
        h.append(f"<h2>{_html.escape(m['model'])}</h2>")
        h.append(f"<img src='data:image/png;base64,{chart(m['model'], series)}'/>")
        h.append("<h3>MLP input g_amax per layer</h3>" + table(series, "in"))
        h.append("<h3>down_proj g_amax per layer</h3>" + table(series, "down"))

    h.append("<h2>Recipe YAML files used</h2>")
    for r in recipes:
        path = os.path.join(recipe_base, r + ".yaml")
        text = open(path).read() if os.path.exists(path) else f"(not found: {path})"
        h.append(
            f"<details open><summary><b>{_html.escape(r)}</b></summary><pre>{_html.escape(text)}</pre></details>"
        )

    h.append("</body></html>")
    with open(out, "w") as f:
        f.write("\n".join(h))
    print(f"wrote {out} ({len(cfg)} models)")


if __name__ == "__main__":
    main()
