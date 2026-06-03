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

"""Compare per-layer NVFP4 activation global amax between two exported checkpoints.

Computes g_amax = input_scale * 6 * 448 for each MLP input quantizer in two checkpoints
(e.g. nvfp4 ref-max vs nvfp4 act-max) and emits a self-contained HTML report (chart +
per-layer table).

Usage:
    python compare_input_scales.py REF_CKPT ACT_CKPT OUT_HTML [--ref-name N] [--act-name N]

"input of the MLP" == the expert up_proj input_scale. down_proj (the intermediate) is
also reported. For MoE layers each value is aggregated across experts (mean + min/max band).
"""

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict

import numpy as np
from safetensors import safe_open

MAXBOUND = 6.0  # NVFP4 e2m1 max
FP8_MAX = 448.0
SCALE_TO_AMAX = MAXBOUND * FP8_MAX  # input_scale * (6*448) -> g_amax

KEY_RE = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up|down)_proj\.input_scale$")


def load_amax(ckpt: str):
    """Return {proj: {layer: np.array([amax per expert])}} from a checkpoint's input_scales."""
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    # group keys by shard to open each shard once
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
                amax = float(v[0]) * SCALE_TO_AMAX  # per-tensor scalar
                data[proj][int(layer)][int(expert)] = amax
    # collapse expert dict -> sorted np array
    out = {}
    for proj, layers in data.items():
        out[proj] = {ly: np.array([layers[ly][e] for e in sorted(layers[ly])]) for ly in layers}
    return out


def agg(per_expert: dict):
    """Layer -> array  =>  (layers, mean, lo, hi, median)."""
    layers = sorted(per_expert)
    mean = np.array([per_expert[lyr].mean() for lyr in layers])
    lo = np.array([per_expert[lyr].min() for lyr in layers])
    hi = np.array([per_expert[lyr].max() for lyr in layers])
    med = np.array([np.median(per_expert[lyr]) for lyr in layers])
    return np.array(layers), mean, lo, hi, med


def make_chart(ref, act, ref_name, act_name):
    """Render the per-layer g_amax chart (up/down proj) and return it as a base64 PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot on the TRUE decoder-layer index so every layer 0..N appears on the axis; layers
    # without an MLP (mamba/attention) simply have no marker (a gap), not a skipped tick.
    all_layers = set()
    for dat in (ref, act):
        for p in ("up", "down"):
            all_layers |= set(dat.get(p, {}))
    lo_l, hi_l = (min(all_layers), max(all_layers)) if all_layers else (0, 0)
    ticks = list(range(lo_l, hi_l + 1))

    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for ax, proj, title in zip(
        axes, ("up", "down"), ("up_proj (MLP input)", "down_proj (intermediate)")
    ):
        for name, dat, color in ((ref_name, ref, "#1f77b4"), (act_name, act, "#d62728")):
            d = dat.get(proj, {})
            ly = sorted(d)
            if not ly:
                continue
            mean = np.array([d[lyr].mean() for lyr in ly])
            lo = np.array([d[lyr].min() for lyr in ly])
            hi = np.array([d[lyr].max() for lyr in ly])
            ax.fill_between(ly, lo, hi, color=color, alpha=0.15)
            ax.plot(ly, mean, "-o", color=color, ms=4, lw=1.4, label=name)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("decoder layer index (gaps = non-MLP mamba/attention layers)")
        ax.set_ylabel("g_amax  (input_scale x 6 x 448)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_xlim(lo_l - 0.5, hi_l + 0.5)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Per-layer NVFP4 activation g_amax — all decoder layers (MLP exists only on MoE layers)"
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_html(ref, act, ref_name, act_name, out_html):
    """Build the self-contained HTML report (chart + up_proj per-layer table) and write it."""
    png_b64 = make_chart(ref, act, ref_name, act_name)

    # per-layer up_proj table (the MLP input)
    rly, rmean, rlo, rhi, rmed = agg(ref["up"])
    aly, amean, alo, ahi, amed = agg(act["up"])
    ref_by = dict(zip(rly, zip(rmean, rlo, rhi)))
    act_by = dict(zip(aly, zip(amean, alo, ahi)))
    layers = sorted(set(rly) | set(aly))

    rows = []
    ratios = []
    for lyr in layers:
        rm, rl, rh = ref_by.get(lyr, (float("nan"),) * 3)
        am, al, ah = act_by.get(lyr, (float("nan"),) * 3)
        ratio = am / rm if rm else float("nan")
        ratios.append(ratio)
        rows.append(
            f"<tr><td>{lyr}</td>"
            f"<td>{rm:,.1f}</td><td>{rl:,.1f}</td><td>{rh:,.1f}</td>"
            f"<td>{am:,.1f}</td><td>{al:,.1f}</td><td>{ah:,.1f}</td>"
            f"<td>{ratio:,.2f}x</td></tr>"
        )
    ratios = np.array([r for r in ratios if np.isfinite(r)])
    summary = (
        f"up_proj (MLP input): act/ref g_amax ratio — mean {ratios.mean():.2f}x, "
        f"median {np.median(ratios):.2f}x, min {ratios.min():.2f}x, max {ratios.max():.2f}x "
        f"across {len(ratios)} MoE layers. ratio &gt; 1 means act_max raised the global scale "
        f"(more saturation headroom)."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NVFP4 activation g_amax: {ref_name} vs {act_name}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;max-width:1200px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}} th{{background:#f4f4f4}}
td:first-child,th:first-child{{text-align:center}}
.note{{background:#f7f9fc;border-left:4px solid #1f77b4;padding:10px 14px;margin:12px 0;font-size:14px}}
img{{max-width:100%;border:1px solid #eee}}
caption{{font-weight:600;margin-bottom:6px;text-align:left}}
</style></head><body>
<h1>NVFP4 activation global amax (g_amax) per layer — {ref_name} vs {act_name}</h1>
<div class="note">g_amax = stored <code>input_scale &times; 6 &times; 448</code>. Each point aggregates
the per-expert up_proj/down_proj input quantizers within a MoE layer; the shaded band spans
min..max across experts. <b>{summary}</b></div>
<img src="data:image/png;base64,{png_b64}"/>
<h2>Per-layer up_proj (MLP input) g_amax</h2>
<table><caption></caption>
<tr><th rowspan=2>layer</th><th colspan=3>{ref_name} g_amax</th>
<th colspan=3>{act_name} g_amax</th><th rowspan=2>act/ref<br>(mean)</th></tr>
<tr><th>mean</th><th>min</th><th>max</th><th>mean</th><th>min</th><th>max</th></tr>
{"".join(rows)}
</table>
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html}")
    print(summary.replace("&gt;", ">"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_ckpt")
    ap.add_argument("act_ckpt")
    ap.add_argument("out_html")
    ap.add_argument("--ref-name", default="ref-max")
    ap.add_argument("--act-name", default="act-max")
    a = ap.parse_args()
    ref = load_amax(a.ref_ckpt)
    act = load_amax(a.act_ckpt)
    build_html(ref, act, a.ref_name, a.act_name, a.out_html)
