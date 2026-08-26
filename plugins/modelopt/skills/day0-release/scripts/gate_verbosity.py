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

"""Day-0 verbosity gate.

Decides whether a candidate's average output length is within threshold of its
baseline. Pure decision logic in ``evaluate_verbosity`` (unit-tested); ``main``
harvests token counts from NEL eval artifacts.

Read ``response_stats.avg_completion_tokens``. The ``reasoning.*_tokens`` fields
are always 0 (only the reasoning/content split is missing, not the total) and the
``*_words`` siblings are a proxy that can disagree with the gate.

Two filters run before averaging: skip run dirs matching ``--exclude`` (mismatched
reasoning effort) and keep only runs at a task's max ``successful_count``. Tasks
with no common sample count are reported ``not_comparable``.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import statistics
import sys

# Below this, a few tokens of difference is a large percentage. Flagged, not failed.
_SHORT_OUTPUT_TOKENS = 1000


def evaluate_verbosity(baseline, candidate, threshold=0.05):
    """Decide the verbosity gate from harvested per-run token counts.

    Args:
        baseline: ``{task: [(avg_completion_tokens, successful_count), ...]}``
        candidate: same shape as ``baseline``
        threshold: max allowed |delta| as a fraction of baseline (default 0.05)

    Returns:
        dict ``{pass, failure_class, detail, per_task, max_abs_delta}``.
    """
    if not baseline or not candidate:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": f"no metrics harvested (baseline={len(baseline)}, candidate={len(candidate)})",
            "per_task": {},
            "max_abs_delta": None,
        }

    per_task, exceeded, worst = {}, [], 0.0
    for task in sorted(set(baseline) | set(candidate)):
        b_runs, c_runs = baseline.get(task, []), candidate.get(task, [])
        if not b_runs or not c_runs:
            per_task[task] = {"status": "not_comparable", "reason": "present on one side only"}
            continue

        full = max(n for _, n in b_runs + c_runs)
        b = [t for t, n in b_runs if n == full]
        c = [t for t, n in c_runs if n == full]
        dropped = (len(b_runs) - len(b)) + (len(c_runs) - len(c))
        if not b or not c:
            per_task[task] = {
                "status": "not_comparable",
                "reason": (
                    f"no runs at a common sample count (baseline n="
                    f"{sorted({n for _, n in b_runs})}, candidate n={sorted({n for _, n in c_runs})})"
                ),
            }
            continue

        b_mean, c_mean = statistics.mean(b), statistics.mean(c)
        delta = (c_mean - b_mean) / b_mean
        within = abs(delta) <= threshold
        entry = {
            "status": "compared",
            "baseline_tokens": round(b_mean, 1),
            "candidate_tokens": round(c_mean, 1),
            "delta": round(delta, 4),
            "within_threshold": within,
            "runs": [len(b), len(c)],
            "dropped_partial_runs": dropped,
        }
        if min(b_mean, c_mean) < _SHORT_OUTPUT_TOKENS:
            entry["short_output_warning"] = (
                f"mean under {_SHORT_OUTPUT_TOKENS} tokens; ratio unstable, read absolute counts"
            )
        per_task[task] = entry
        worst = max(worst, abs(delta))
        if not within:
            exceeded.append(task)

    if exceeded:
        return {
            "pass": False,
            "failure_class": "VERBOSITY_EXCEEDED",
            "detail": f"tasks exceeding threshold ({threshold}): {exceeded}",
            "per_task": per_task,
            "max_abs_delta": round(worst, 4),
        }
    if not any(v.get("status") == "compared" for v in per_task.values()):
        return {
            "pass": False,
            "failure_class": "SAMPLE_ACCOUNTING_FAILED",
            "detail": "no task had comparable runs on both sides",
            "per_task": per_task,
            "max_abs_delta": None,
        }
    n_cmp = sum(1 for v in per_task.values() if v.get("status") == "compared")
    return {
        "pass": True,
        "failure_class": None,
        "detail": f"all {n_cmp} comparable task(s) within threshold {threshold}",
        "per_task": per_task,
        "max_abs_delta": round(worst, 4),
    }


def harvest(side, glob="eval_*", exclude="_high"):
    """Collect ``{task: [(avg_completion_tokens, successful_count), ...]}`` from NEL artifacts."""
    out = {}
    pattern = os.path.join(side, glob, "*", "*", "artifacts", "eval_factory_metrics.json")
    for path in globmod.glob(pattern):
        parts = path.split(os.sep)
        if exclude and exclude in parts[-5]:
            continue
        task = parts[-3].rsplit(".", 1)[0].split(".", 1)[-1]
        try:
            with open(path) as f:
                stats = json.load(f).get("response_stats", {})
        except (OSError, json.JSONDecodeError):
            continue
        tokens, count = stats.get("avg_completion_tokens"), stats.get("successful_count")
        if tokens and count:
            out.setdefault(task, []).append((tokens, count))
    return out


def main(argv=None):
    """CLI entry point: harvest both sides from eval artifacts and print the verdict."""
    p = argparse.ArgumentParser(description="Day-0 verbosity gate")
    p.add_argument("--baseline", required=True, help="baseline dir containing eval run dirs")
    p.add_argument("--candidate", required=True, help="candidate dir containing eval run dirs")
    p.add_argument("--glob", default="eval_*", help="run-dir glob within each side")
    p.add_argument(
        "--exclude",
        default="_high",
        help="skip run dirs whose name contains this (reasoning-effort mismatch); '' disables",
    )
    p.add_argument(
        "--threshold", type=float, default=0.05, help="max |delta| fraction (default 0.05)"
    )
    args = p.parse_args(argv)

    baseline = harvest(args.baseline, args.glob, args.exclude)
    candidate = harvest(args.candidate, args.glob, args.exclude)
    result = evaluate_verbosity(baseline, candidate, args.threshold)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
