# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compare acceptance measured by specdec_bench's in-process path and by AIPerf.

Run both harnesses against the same model, draft, workload and hardware, then
point this at their output directories. It reports both AL conventions side by
side so a disagreement is visible as a number rather than inferred from two
reports in different formats.

Two means are reported because they answer different questions and are easy to
confuse:

  per-request AL  mean over requests of (that request's emitted tokens per
                  step). Every request counts equally regardless of length.
  pooled AL       total emitted tokens / total verification steps across the
                  whole run. Long requests dominate.

They differ whenever acceptance correlates with length, which it usually does,
so quoting one against the other is a common way to manufacture a discrepancy
that isn't there.

The two harnesses also bin differently: specdec_bench histograms the number of
tokens *emitted* per step (1..K+1, the bonus token included), while AIPerf
histograms the number of *drafts accepted* per step (0..K). This normalises
AIPerf's to the emitted-length convention so the rows line up.

Usage:
    python3 compare_harnesses.py --specdec-dir <dir> --aiperf-dir <dir>
"""

import argparse
import json
import sys
from pathlib import Path


def _load_specdec(directory: Path) -> dict:
    """Read specdec_bench's acceptance_rate.json.

    The metric writer wraps its output in a single-element list, so accept both
    shapes rather than depending on which writer version produced the file.
    """
    path = directory / "acceptance_rate.json"
    if not path.is_file():
        raise SystemExit(f"no acceptance_rate.json in {directory}")
    raw = json.loads(path.read_text())
    data = raw[0] if isinstance(raw, list) else raw

    hist = {int(k): int(v) for k, v in (data.get("Acceptance_Length_Histogram") or {}).items()}
    steps = sum(hist.values())
    if not steps:
        raise SystemExit(f"empty acceptance histogram in {path}")
    return {
        "per_request_al": data.get("Average_AL"),
        "pooled_al": sum(k * v for k, v in hist.items()) / steps,
        "steps": steps,
        # Already in emitted-length terms.
        "emitted_hist": hist,
    }


def _load_aiperf(directory: Path) -> dict:
    """Read AIPerf's profile_export_aiperf.json."""
    path = directory / "profile_export_aiperf.json"
    if not path.is_file():
        raise SystemExit(f"no profile_export_aiperf.json in {directory}")
    data = json.loads(path.read_text())

    def avg(key):
        entry = data.get(key)
        return entry.get("avg") if isinstance(entry, dict) else entry

    accepted_hist = {
        int(k): int(v)
        for k, v in (data.get("pooled_spec_decode_acceptance_histogram") or {}).items()
    }
    if not accepted_hist:
        raise SystemExit(
            f"{path} carries no pooled_spec_decode_acceptance_histogram. "
            "The server needs --per-request-spec-decode-metrics detailed, and "
            "it is only honoured for n == 1."
        )
    # accepted drafts -> tokens emitted that step (accepted + the bonus token).
    emitted_hist = {k + 1: v for k, v in accepted_hist.items()}
    steps = sum(emitted_hist.values())
    return {
        "per_request_al": avg("spec_decode_acceptance_length"),
        # AIPerf's token-weighted AL is the pooled mean; recompute from the
        # histogram too and let the caller see both agree.
        "pooled_al": avg("spec_decode_token_weighted_acceptance_length"),
        "pooled_al_from_hist": sum(k * v for k, v in emitted_hist.items()) / steps,
        "steps": steps,
        "emitted_hist": emitted_hist,
        "draft_acceptance_rate": avg("spec_decode_overall_draft_acceptance_rate"),
    }


def _fmt(value, width=10, places=4):
    return f"{value:>{width}.{places}f}" if isinstance(value, (int, float)) else f"{'n/a':>{width}}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specdec-dir", required=True, type=Path, help="run.py --save_dir output")
    parser.add_argument(
        "--aiperf-dir", required=True, type=Path, help="aiperf --artifact-dir output"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="fractional disagreement in pooled AL to treat as agreement (default 0.05)",
    )
    args = parser.parse_args()

    spec = _load_specdec(args.specdec_dir)
    ap = _load_aiperf(args.aiperf_dir)

    print(f"{'harness':<24}{'per-request AL':>16}{'pooled AL':>12}{'steps':>12}")
    print("-" * 64)
    print(
        f"{'specdec_bench':<24}{_fmt(spec['per_request_al'], 16)}{_fmt(spec['pooled_al'], 12)}{spec['steps']:>12}"
    )
    print(
        f"{'aiperf':<24}{_fmt(ap['per_request_al'], 16)}{_fmt(ap['pooled_al'], 12)}{ap['steps']:>12}"
    )

    print("\nemitted tokens per step (share of steps):")
    lengths = sorted(set(spec["emitted_hist"]) | set(ap["emitted_hist"]))
    header = "".join(f"{n:>9}" for n in lengths)
    print(f"{'  length':<24}{header}")
    for name, payload in (("  specdec_bench", spec), ("  aiperf", ap)):
        total = sum(payload["emitted_hist"].values())
        row = "".join(f"{payload['emitted_hist'].get(n, 0) / total:>8.1%} " for n in lengths)
        print(f"{name:<24}{row}")

    # AIPerf reports the pooled mean directly and we also recompute it from the
    # histogram; if those two disagree the export is internally inconsistent and
    # no cross-harness conclusion should be drawn from it.
    if ap["pooled_al"] is not None:
        drift = abs(ap["pooled_al"] - ap["pooled_al_from_hist"])
        if drift > 1e-3:
            print(
                f"\nWARNING: AIPerf's reported pooled AL ({ap['pooled_al']:.4f}) and the value "
                f"implied by its own histogram ({ap['pooled_al_from_hist']:.4f}) differ by "
                f"{drift:.4f}. Treat this export as unreliable."
            )

    if spec["pooled_al"] and ap["pooled_al"]:
        delta = ap["pooled_al"] - spec["pooled_al"]
        rel = abs(delta) / spec["pooled_al"]
        print(f"\npooled AL delta: {delta:+.4f} ({rel:.1%} of specdec_bench)")
        if rel <= args.tolerance:
            print(
                f"AGREE within {args.tolerance:.0%}: the two harnesses measure the same quantity."
            )
            return 0
        print(
            f"DISAGREE beyond {args.tolerance:.0%}. Before blaming either harness, confirm both "
            "ran the same prompts, draft length, sampling and output cap -- those explain most "
            "gaps. A real disagreement at matched settings is a bug worth filing."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
