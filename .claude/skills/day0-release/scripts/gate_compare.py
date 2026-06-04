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

Score files are ``{task_name: score}`` dicts, scores on a 0-100 scale (the AA
task references report ``*_avg_of_N`` on 0-100). The drop is computed as an
absolute percentage-point delta unless ``--relative`` is passed.
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

# Plausibility bounds for a 0-100 score.
_SCORE_MIN = 0.0
_SCORE_MAX = 100.0
# A candidate scoring this many points ABOVE baseline is implausible for
# quantization (quantization should not meaningfully improve accuracy); flag it
# rather than silently passing.
_IMPLAUSIBLE_GAIN = 5.0


def evaluate_comparison(baseline, candidate, threshold=0.01, relative=False):
    """Compare candidate vs baseline scores per task.

    Args:
        baseline: dict ``{task: score}`` (0-100).
        candidate: dict ``{task: score}`` (0-100).
        threshold: max allowed drop, as a fraction (0.01 = 1 percentage point
            absolute, or 1% relative if ``relative``).
        relative: if True, drop is measured relative to the baseline score.

    Returns:
        dict ``{pass, decision, failure_class, detail, per_task}``.
    """
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
        if relative:
            drop = (b - c) / b if b else 0.0
            limit = threshold
        else:
            drop = b - c  # percentage points
            limit = threshold * 100.0  # threshold is a fraction of the 0-100 scale
        within = drop <= limit
        if c - b > _IMPLAUSIBLE_GAIN:
            anomalies.append(f"{task}: candidate exceeds baseline by {c - b:.2f} pts (implausible)")
        per_task[task] = {
            "baseline": b,
            "candidate": c,
            "drop": round(drop, 4),
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
    args = p.parse_args(argv)

    try:
        with open(args.baseline) as f:
            baseline = json.load(f)
        with open(args.candidate) as f:
            candidate = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}))
        return 2

    result = evaluate_comparison(baseline, candidate, args.threshold, args.relative)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
