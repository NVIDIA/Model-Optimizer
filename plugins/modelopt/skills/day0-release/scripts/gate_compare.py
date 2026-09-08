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

"""Day-0 compare gate.

Decides whether a quantized candidate is within the accuracy threshold of its
baseline, per task. Pure decision logic in ``evaluate_comparison`` (unit-tested
without GPU/cluster); ``main`` reads score JSON files and prints the verdict.

Score files are ``{task_name: score}`` dicts. Most AA task references report
``*_avg_of_N`` on a 0-100 scale, but some tasks (e.g. ``tau2_bench_telecom``
``Result``) report on a 0-1 scale. The gate is therefore scale-aware: each
task's scale is inferred per task (0-1 if both scores are within [0, 1], else
0-100) or supplied explicitly via ``--scales``, and the drop is normalized to a
fraction of that scale so the threshold applies uniformly. The drop is an
absolute (scale-normalized) delta unless ``--relative`` is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _is_valid_score(val):
    """True only for a finite real number in [_SCORE_MIN, _SCORE_MAX] (not bool)."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
        and _SCORE_MIN <= val <= _SCORE_MAX
    )


# Decisions
ACCEPT = "ACCEPT"
REGRESSION = "REGRESSION"
ANOMALOUS = "ANOMALOUS"

# Plausibility bounds. Scores may be on a 0-1 or 0-100 scale (see _infer_scale);
# the upper bound is the larger of the two so both are accepted.
_SCORE_MIN = 0.0
_SCORE_MAX = 100.0
# A candidate scoring this fraction of its scale ABOVE baseline is implausible
# for quantization (quantization should not meaningfully improve accuracy); flag
# it rather than silently passing. 0.05 = 5 pts on a 0-100 task, 0.05 on a 0-1 task.
_IMPLAUSIBLE_GAIN_FRAC = 0.05


def _infer_scale(*vals):
    """Infer a task's score scale: 1.0 if every score is within [0, 1], else 100.0.

    Most AA tasks report 0-100; a few (e.g. ``tau2_bench_telecom``) report 0-1.
    Without scale metadata in the score files, we treat a task as 0-1 only when
    every score for it fits in [0, 1] — a 0-100 task with sub-1.0 accuracy is
    degenerate and caught elsewhere. Pass an explicit scale to override.
    """
    return 1.0 if all(0.0 <= v <= 1.0 for v in vals) else 100.0


def evaluate_comparison(baseline, candidate, threshold=0.01, relative=False, scales=None):
    """Compare candidate vs baseline scores per task.

    Args:
        baseline: dict ``{task: score}``.
        candidate: dict ``{task: score}``.
        threshold: max allowed drop, as a fraction of the task's scale
            (0.01 = 1 percentage point on a 0-100 task / 0.01 on a 0-1 task,
            or 1% relative if ``relative``).
        relative: if True, drop is measured relative to the baseline score
            (scale-invariant).
        scales: optional dict ``{task: max_scale}`` to override per-task scale
            inference (e.g. ``{"tau2_bench_telecom": 1.0}``).

    Returns:
        dict ``{pass, decision, failure_class, detail, per_task}``.
    """
    scales = scales or {}
    if not isinstance(scales, dict) or any(
        not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0)
        for v in scales.values()
    ):
        return {
            "pass": False,
            "decision": ANOMALOUS,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": f"invalid scales: expected {{task: positive finite number}}, got {scales!r}",
            "per_task": {},
        }
    missing = sorted((set(baseline) | set(candidate)) - (set(baseline) & set(candidate)))
    if missing:
        return {
            "pass": False,
            "decision": ANOMALOUS,
            "failure_class": "SAMPLE_ACCOUNTING_FAILED",
            "detail": f"task sets differ; missing on one side: {missing}",
            "per_task": {},
        }
    if not baseline:
        return {
            "pass": False,
            "decision": ANOMALOUS,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "no tasks to compare",
            "per_task": {},
        }

    per_task = {}
    regressed = []
    anomalies = []
    for task in sorted(baseline):
        b, c = baseline[task], candidate[task]
        invalid = False
        for label, val in (("baseline", b), ("candidate", c)):
            if not _is_valid_score(val):
                anomalies.append(f"{task}: {label} score {val!r} not a finite number in [0, 100]")
                invalid = True
        if invalid:
            # Don't compute deltas on non-numeric/out-of-range scores (would raise
            # TypeError); record the anomaly and move on — the run is ANOMALOUS.
            per_task[task] = {
                "baseline": b,
                "candidate": c,
                "drop": None,
                "within_threshold": False,
            }
            continue
        scale = scales.get(task) or _infer_scale(b, c)
        delta = b - c  # native units, for reporting
        if relative:
            drop = delta / b if b else 0.0  # fraction of baseline (scale-invariant)
        else:
            drop = delta / scale  # fraction of the task's scale
        within = drop <= threshold
        gain = (c - b) / scale
        if gain > _IMPLAUSIBLE_GAIN_FRAC:
            anomalies.append(
                f"{task}: candidate exceeds baseline by {c - b:.4g} ({gain:.1%} of scale, implausible)"
            )
        per_task[task] = {
            "baseline": b,
            "candidate": c,
            "drop": round(delta, 4),
            "drop_fraction": round(drop, 4),
            "scale": scale,
            "within_threshold": within,
        }
        if not within:
            regressed.append(task)

    if anomalies:
        return {
            "pass": False,
            "decision": ANOMALOUS,
            "failure_class": "UNKNOWN",
            "detail": "; ".join(anomalies),
            "per_task": per_task,
        }
    if regressed:
        return {
            "pass": False,
            "decision": REGRESSION,
            "failure_class": None,
            "detail": f"tasks exceeding threshold ({threshold}): {regressed}",
            "per_task": per_task,
        }
    return {
        "pass": True,
        "decision": ACCEPT,
        "failure_class": None,
        "detail": f"all {len(per_task)} task(s) within threshold {threshold}",
        "per_task": per_task,
    }


def main(argv=None):
    """CLI entry point: read baseline/candidate score JSON and print the verdict."""
    p = argparse.ArgumentParser(description="Day-0 compare gate")
    p.add_argument("--baseline", required=True, help="baseline score JSON {task: score}")
    p.add_argument("--candidate", required=True, help="candidate score JSON {task: score}")
    p.add_argument("--threshold", type=float, default=0.01, help="max drop fraction (default 0.01)")
    p.add_argument("--relative", action="store_true", help="measure drop relative to baseline")
    p.add_argument(
        "--scales",
        help="optional JSON {task: max_scale} to override per-task scale inference",
    )
    args = p.parse_args(argv)

    try:
        with open(args.baseline) as f:
            baseline = json.load(f)
        with open(args.candidate) as f:
            candidate = json.load(f)
        scales = json.loads(args.scales) if args.scales else None
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}))
        return 2

    result = evaluate_comparison(baseline, candidate, args.threshold, args.relative, scales)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
