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

"""Propose a pipeline layout for QAD from two profile_distill.py runs.

Both runs use the same layer split, e.g. the even default:
  --full     full recompute of every layer (``--recompute_granularity full --recompute_method uniform
             --recompute_num_layers 1``)
  --partial  the same, but block-recompute fewer layers on at least one stage

From them it fits, per micro-batch: the time of a recomputed and of a stored layer, each stage's
fixed extra time (embedding on the first, output layers + KD loss on the last), and the memory of a
stored layer per micro-batch in flight (stage k holds min(PP - k, micro-batches) of them under 1F1B).
It then searches first/last-stage layer counts, recomputes on each stage only as many layers as its
memory requires, and picks the layout with the fastest slowest stage. Predictions are estimates:
confirm the chosen layout with a measured A/B before training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_profile(directory):
    """Per-stage medians of a profile directory, skipping each rank's warm-up iteration."""
    by_stage = {}
    for path in sorted(Path(directory).glob("rank*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for r in records[1:]:
            by_stage.setdefault(r["pp_rank"], []).append(r)
    if not by_stage:
        raise ValueError(f"{directory}: no profiled iterations after warm-up")
    stages = []
    for k in sorted(by_stage):
        rs = by_stage[k]
        stages.append(
            {
                "layers": rs[0]["layers"],
                "recomputed": rs[0]["recomputed_layers"],
                "ms_per_mb": statistics.median(
                    (r["fwd_ms"] + r["bwd_ms"]) / r["microbatches"] for r in rs
                ),
                "static_gib": max(r["static_gib"] for r in rs),
                "peak_gib": max(r["peak_alloc_gib"] for r in rs),
            }
        )
    first = by_stage[0][0]
    return {
        "stages": stages,
        "microbatches": first["microbatches"],
        "device_gib": first["device_gib"],
    }


def fit_costs(full, partial):
    """Unit costs from a full-recompute profile and a partial-recompute one with the same split."""
    fs, ps = full["stages"], partial["stages"]
    if len(fs) != len(ps) or any(f["layers"] != p["layers"] for f, p in zip(fs, ps)):
        raise ValueError("--full and --partial must use the same pipeline split")
    if any(f["recomputed"] != f["layers"] for f in fs):
        raise ValueError("--full must recompute every layer on every stage")
    pp, mbs = len(fs), full["microbatches"]
    # Stages before the last run only decoder layers (plus a small embedding on the first).
    body = fs[:-1] if pp > 1 else fs
    recomputed_ms = statistics.mean(s["ms_per_mb"] / s["layers"] for s in body)
    saved_ms, layer_gib = [], []
    for k, (f, p) in enumerate(zip(fs, ps)):
        fewer = f["recomputed"] - p["recomputed"]
        if fewer > 0:
            saved_ms.append((f["ms_per_mb"] - p["ms_per_mb"]) / fewer)
            layer_gib.append((p["peak_gib"] - f["peak_gib"]) / (min(pp - k, mbs) * fewer))
    if not saved_ms:
        raise ValueError("--partial must recompute fewer layers than --full on at least one stage")
    middle = fs[1:-1] or fs[:1]
    return {
        "recomputed_ms": recomputed_ms,
        "stored_ms": recomputed_ms - statistics.mean(saved_ms),
        # The largest observed value: underestimating memory is what causes an OOM.
        "stored_layer_gib": max(layer_gib),
        "static_layer_gib": statistics.mean(s["static_gib"] / s["layers"] for s in middle),
        "extra_ms": [s["ms_per_mb"] - recomputed_ms * s["layers"] for s in fs],
    }


def candidate_splits(total_layers, pp):
    """(first, last) layer counts whose middle stages split evenly, as distill.py requires."""
    if pp == 1:
        yield total_layers, total_layers
        return
    for first in range(1, total_layers):
        for last in range(1, total_layers - first + 1):
            rest = total_layers - first - last
            if (pp == 2 and rest == 0) or (pp > 2 and rest >= pp - 2 and rest % (pp - 2) == 0):
                yield first, last


def plan(full, costs, cap):
    """Fastest layout whose predicted peak stays under ``cap`` GiB on every stage, or None."""
    fs, mbs = full["stages"], full["microbatches"]
    pp, total = len(fs), sum(s["layers"] for s in fs)
    best = None
    for first, last in candidate_splits(total, pp):
        middle = (total - first - last) // (pp - 2) if pp > 2 else 0
        layers = [first] + [middle] * (pp - 2) + [last] if pp > 1 else [total]
        stages = []
        for k, n in enumerate(layers):
            measured = fs[k]
            in_flight = min(pp - k, mbs)
            fixed = (
                measured["static_gib"]
                + costs["static_layer_gib"] * (n - measured["layers"])
                + (
                    measured["peak_gib"] - measured["static_gib"]
                )  # dynamic memory at full recompute
            )
            if fixed > cap:
                break
            stored = min(n, math.floor((cap - fixed) / (in_flight * costs["stored_layer_gib"])))
            stages.append(
                {
                    "layers": n,
                    "recompute": n - stored,
                    "ms_per_mb": costs["recomputed_ms"] * (n - stored)
                    + costs["stored_ms"] * stored
                    + costs["extra_ms"][k],
                    "peak_gib": fixed + in_flight * costs["stored_layer_gib"] * stored,
                }
            )
        else:
            slowest = max(s["ms_per_mb"] for s in stages)
            key = (slowest, max(s["peak_gib"] for s in stages))
            if best is None or key < best["key"]:
                best = {"key": key, "stages": stages, "iteration_s": (mbs + pp - 1) * slowest / 1e3}
    if best is None:
        return None
    best.pop("key")
    return best


def main(argv=None):
    """CLI entry point: print the measured baseline, the fitted costs and the proposed layout."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--full", required=True, help="profile dir with full recompute everywhere")
    p.add_argument("--partial", required=True, help="profile dir with fewer recomputed layers")
    p.add_argument(
        "--max_peak_gib",
        type=float,
        help="peak allocated memory allowed per GPU; defaults to 80%% of the device, which leaves room "
        "for allocator fragmentation, validation and checkpoint saves",
    )
    args = p.parse_args(argv)
    try:
        full, partial = load_profile(args.full), load_profile(args.partial)
        costs = fit_costs(full, partial)
    except (OSError, KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return 2
    pp, mbs = len(full["stages"]), full["microbatches"]
    measured_s = (mbs + pp - 1) * max(s["ms_per_mb"] for s in full["stages"]) / 1e3
    layout = plan(full, costs, args.max_peak_gib or 0.8 * full["device_gib"])
    result = {"full_recompute_iteration_s": measured_s, "costs": costs, "layout": layout}
    if layout:
        layers = [s["layers"] for s in layout["stages"]]
        result["distill_flags"] = (
            f"--num_layers_in_first_pipeline_stage {layers[0]} "
            f"--num_layers_in_last_pipeline_stage {layers[-1]} "
            "--recompute_granularity full --recompute_method block "
            "--recompute_num_layers_per_stage "
            + " ".join(str(s["recompute"]) for s in layout["stages"])
        )
    print(json.dumps(result, indent=2))
    return 0 if layout else 1


if __name__ == "__main__":
    sys.exit(main())
