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

"""QAD checkpoint gate: continue, rerun, or accept a QAD run from its evaluated checkpoints.

Input JSON (scores in percentage points, from validated evaluations with matched repeats):
    {
      "baseline": {"<task>": score, ...},
      "checkpoints": {"<iteration>": {"<task>": score, ...}, ...},  # a subset of tasks is allowed
      "se": {"<iteration>": {"<task>": se, ...}, ...},  # SE of baseline - score for every score:
                                                      # sqrt(sd_b**2 / n_b + sd_q**2 / n_q) over runs
      "horizon": 150,           # planned stop: --train_iters, where the cosine LR anneal ends
      "runs": {"<iteration>": {"<task>": n, ...}},  # optional: pooled runs behind each score
      "max_runs": {"<task>": n, ...},               # optional: per-task repeat cap (about 2x the recipe)
      "threshold_pp": 1.0,      # optional
      "band_se": 1.5,           # optional: repeats only while |gap - threshold| < band_se * SE
      "patience": 2             # optional: worse checkpoints in a row that count as a regression
    }

Each task's gap (``baseline - score``) falls in one zone: PASS (gap < threshold), CLEAR_MISS
(gap - 2 SE >= threshold), or NEAR_MISS. A checkpoint takes its worst zone, and the best checkpoint
is the one with the smallest worst gap, since the mean can improve while one benchmark regresses.

Decisions:
  PASS            the best checkpoint passes every baseline task: run the confirmation run.
  SCREEN_PASS     it passes every task it was screened on: run the remaining tasks on it.
  NEEDS_REPEATS   it is a near miss within ``band_se`` SEs of the threshold and its blocking task
                  is under its run cap: submit that task again, pool the runs, then gate again.
  CONTINUE        before the horizon: keep training. Near and clear misses never stop a run early.
  MISS            the run reached its horizon without a pass. ``zone`` says near or clear miss;
                  ``regressed`` says whether later checkpoints fell beyond noise (2 SE of the
                  difference, ``patience`` in a row, or 3 rising in a row) below the best one;
                  ``improving`` says whether the latest checkpoint is the best and gained beyond
                  noise on the first, i.e. the run was still closing the gap when it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

PASS = "PASS"  # nosec B105 - a gate decision, not a password
SCREEN_PASS = "SCREEN_PASS"  # nosec B105
NEEDS_REPEATS = "NEEDS_REPEATS"
CONTINUE = "CONTINUE"
MISS = "MISS"
NEAR_MISS, CLEAR_MISS = "NEAR_MISS", "CLEAR_MISS"
_ZONE_RANK = {PASS: 0, NEAR_MISS: 1, CLEAR_MISS: 2}


def _error(detail):
    return {"decision": None, "failure_class": "USER_CONFIG_ERROR", "detail": detail}


def _is_number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def evaluate_qad(
    baseline,
    checkpoints,
    se,
    horizon,
    runs=None,
    max_runs=None,
    threshold_pp=1.0,
    band_se=1.5,
    patience=2,
):
    """Decide from the evaluated checkpoints so far. Returns ``{decision, best_iteration, ...}``."""
    if not baseline or not checkpoints or not _is_number(horizon):
        return _error("need baseline scores, at least one evaluated checkpoint, and the horizon")
    if not all(_is_number(v) for v in baseline.values()):
        return _error("baseline has a non-numeric score")
    se = {int(k): v for k, v in (se or {}).items()}
    runs = {int(k): v for k, v in (runs or {}).items()}
    max_runs = max_runs or {}
    per_checkpoint = {}
    for iteration in sorted(checkpoints, key=int):
        it, scores = int(iteration), checkpoints[iteration]
        if not scores or not set(scores) <= set(baseline):
            return _error(f"checkpoint {it} tasks {sorted(scores)} not a subset of the baseline's")
        ses = se.get(it, {})
        if not all(_is_number(scores[t]) and _is_number(ses.get(t)) for t in scores):
            return _error(f"checkpoint {it} needs a numeric score and SE for every task")
        tasks = {}
        for t in sorted(scores):
            gap = baseline[t] - scores[t]
            zone = (
                PASS
                if gap < threshold_pp
                else CLEAR_MISS
                if gap - 2 * ses[t] >= threshold_pp
                else NEAR_MISS
            )
            tasks[t] = {"gap": gap, "se": ses[t], "zone": zone, "runs": runs.get(it, {}).get(t)}
        blocking = max(tasks, key=lambda t: tasks[t]["gap"])
        per_checkpoint[it] = {
            "tasks": tasks,
            "max_gap": tasks[blocking]["gap"],
            "blocking_task": blocking,
            "zone": max((v["zone"] for v in tasks.values()), key=lambda z: _ZONE_RANK[z]),
        }

    iterations = sorted(per_checkpoint)
    best = min(iterations, key=lambda it: (per_checkpoint[it]["max_gap"], it))
    b = per_checkpoint[best]
    result = {"failure_class": None, "best_iteration": best, "per_checkpoint": per_checkpoint}

    if b["zone"] == PASS:
        missing = sorted(set(baseline) - set(b["tasks"]))
        if not missing:
            return {**result, "decision": PASS, "detail": f"iteration {best}: every task passes"}
        return {
            **result,
            "decision": SCREEN_PASS,
            "detail": f"iteration {best} passes its screen; run {missing} on it",
        }
    if b["zone"] == NEAR_MISS:
        t = b["tasks"][b["blocking_task"]]
        cap = max_runs.get(b["blocking_task"])
        if abs(t["gap"] - threshold_pp) < band_se * t["se"] and cap and (t["runs"] or 0) < cap:
            return {
                **result,
                "decision": NEEDS_REPEATS,
                "detail": f"iteration {best}: {b['blocking_task']} gap {t['gap']:.2f} pp "
                f"(SE {t['se']:.2f}) on {t['runs']} of {cap} runs; submit it again and pool",
            }
    if iterations[-1] < horizon:
        return {
            **result,
            "decision": CONTINUE,
            "detail": f"best iteration {best} is a {b['zone']}; train on to the horizon {horizon}",
        }

    def beyond_noise(hi, lo):
        """Whether checkpoint ``hi``'s worst gap exceeds ``lo``'s by 2 SE of the difference."""
        h, low = per_checkpoint[hi], per_checkpoint[lo]
        se_diff = math.hypot(
            h["tasks"][h["blocking_task"]]["se"], low["tasks"][low["blocking_task"]]["se"]
        )
        return h["max_gap"] - low["max_gap"] >= 2 * se_diff

    after = iterations[iterations.index(best) + 1 :]
    streak = 0
    for it in after:
        streak = streak + 1 if beyond_noise(it, best) else 0
    gaps = [per_checkpoint[it]["max_gap"] for it in after[-3:]]
    rising = len(gaps) == 3 and gaps[0] < gaps[1] < gaps[2]
    regressed = streak >= patience or rising
    improving = best == iterations[-1] and len(iterations) > 1 and beyond_noise(iterations[0], best)
    return {
        **result,
        "decision": MISS,
        "zone": b["zone"],
        "regressed": regressed,
        "improving": improving,
        "detail": f"horizon {horizon} reached; best iteration {best} is a {b['zone']} "
        f"({b['blocking_task']} gap {b['max_gap']:.2f} pp)"
        + ("; later checkpoints regressed beyond noise" if regressed else "")
        + ("; still improving, train this recipe longer" if improving else ""),
    }


def main(argv=None):
    """CLI entry point: read the checkpoint-scores JSON and print the decision."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scores", required=True, help="JSON with baseline and checkpoint scores")
    args = p.parse_args(argv)
    try:
        with open(args.scores) as f:
            data = json.load(f)
        options = ("runs", "max_runs", "threshold_pp", "band_se", "patience")
        result = evaluate_qad(
            data.get("baseline"),
            data.get("checkpoints"),
            data.get("se"),
            data.get("horizon"),
            **{k: data[k] for k in options if k in data},
        )
    except (OSError, ValueError, AttributeError, TypeError) as e:
        result = _error(str(e))
    print(json.dumps(result, indent=2))
    if result["decision"] is None:
        return 2
    return 1 if result["decision"] == MISS else 0


if __name__ == "__main__":
    sys.exit(main())
