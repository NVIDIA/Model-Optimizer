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

"""Compare per-layer NVFP4 activation global amax across N exported checkpoints.

Computes g_amax = input_scale * 6 * 448 for each MLP input quantizer in each checkpoint and
emits a self-contained HTML report (chart over all decoder layers + per-layer tables). The
up_proj input is the "MLP input"; down_proj is the intermediate. Per MoE layer the value is
aggregated across experts (they are usually synced, so min == max == mean).

Usage:
    python compare_input_scales.py OUT_HTML LABEL1=CKPT1 LABEL2=CKPT2 [...]
"""

import base64
import io
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from safetensors import safe_open

MAXBOUND = 6.0  # NVFP4 e2m1 max
FP8_MAX = 448.0
SCALE_TO_AMAX = MAXBOUND * FP8_MAX  # input_scale * (6*448) -> g_amax

KEY_RE = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.input_scale$")

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf"]


def load_amax(ckpt: str):
    """Return {proj: {layer: np.array([amax per expert])}} from a checkpoint's input_scales."""
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    by_shard = defaultdict(list)
    for k, shard in index.items():
        if k.endswith("input_scale") and KEY_RE.match(k):
            by_shard[shard].append(k)

    data = {"up": defaultdict(dict), "down": defaultdict(dict)}  # proj -> layer -> {expert: amax}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(ckpt, shard), framework="numpy") as f:
            for k in keys:
                m = KEY_RE.match(k)
                if m is None:
                    continue
                layer, expert, proj = m.groups()
                v = np.asarray(f.get_tensor(k)).astype(np.float64).reshape(-1)
                data[proj][int(layer)][int(expert)] = float(v[0]) * SCALE_TO_AMAX
    out = {}
    for proj, layers in data.items():
        out[proj] = {ly: np.array([layers[ly][e] for e in sorted(layers[ly])]) for ly in layers}
    return out


def make_chart(series):
    """Render up/down-proj per-layer g_amax over all decoder layers; return a base64 PNG.

    series: list of (label, data, color).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_layers = set()
    for _, dat, _ in series:
        for p in ("up", "down"):
            all_layers |= set(dat.get(p, {}))
    lo_l, hi_l = (min(all_layers), max(all_layers)) if all_layers else (0, 0)
    ticks = list(range(lo_l, hi_l + 1))

    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for ax, proj, title in zip(
        axes, ("up", "down"), ("up_proj (MLP input)", "down_proj (intermediate)")
    ):
        for label, dat, color in series:
            d = dat.get(proj, {})
            ly = sorted(d)
            if not ly:
                continue
            mean = np.array([d[lyr].mean() for lyr in ly])
            ax.plot(ly, mean, "-o", color=color, ms=4, lw=1.4, label=label)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("decoder layer index (gaps = non-MLP mamba/attention layers)")
        ax.set_ylabel("g_amax  (input_scale x 6 x 448)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_xlim(lo_l - 0.5, hi_l + 0.5)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
    fig.suptitle("Per-layer NVFP4 activation g_amax across checkpoints (all decoder layers)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(series, proj):
    """HTML table of per-layer mean g_amax, one column per checkpoint, for one projection."""
    layers = sorted({ly for _, dat, _ in series for ly in dat.get(proj, {})})
    head = "".join(f"<th>{label}</th>" for label, _, _ in series)
    rows = []
    for ly in layers:
        cells = []
        for _, dat, _ in series:
            d = dat.get(proj, {})
            cells.append(f"<td>{d[ly].mean():,.2f}</td>" if ly in d else "<td>-</td>")
        rows.append(f"<tr><td>{ly}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>layer</th>{head}</tr>{''.join(rows)}</table>"


def build_html(series, out_html):
    """Write the self-contained HTML report (chart + up/down per-layer tables)."""
    png_b64 = make_chart(series)
    legend = ", ".join(f"<b style='color:{c}'>{label}</b>" for label, _, c in series)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NVFP4 activation g_amax across checkpoints</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;max-width:1300px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}} th{{background:#f4f4f4}}
td:first-child,th:first-child{{text-align:center}}
.note{{background:#f7f9fc;border-left:4px solid #1f77b4;padding:10px 14px;margin:12px 0;font-size:14px}}
img{{max-width:100%;border:1px solid #eee}}
</style></head><body>
<h1>NVFP4 activation global amax (g_amax) per layer</h1>
<div class="note">g_amax = stored <code>input_scale &times; 6 &times; 448</code>, aggregated across MoE
experts (synced &rarr; one value per layer). Checkpoints: {legend}.</div>
<img src="data:image/png;base64,{png_b64}"/>
<h2>up_proj (MLP input) g_amax per layer</h2>
{_table(series, "up")}
<h2>down_proj (intermediate) g_amax per layer</h2>
{_table(series, "down")}
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html} with {len(series)} checkpoints")


if __name__ == "__main__":
    out_html = sys.argv[1]
    series = []
    for i, arg in enumerate(sys.argv[2:]):
        label, _, path = arg.partition("=")
        series.append((label, load_amax(path), PALETTE[i % len(PALETTE)]))
    build_html(series, out_html)
