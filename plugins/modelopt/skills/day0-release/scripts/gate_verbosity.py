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

``main`` also attaches ``harvest_diagnostics`` recording anything dropped before the
comparison (excluded run dirs, unreadable artifacts, metrics missing token counts).

Two filters run before averaging: skip run dirs matching ``--exclude`` (mismatched
reasoning effort) and keep only runs at the largest ``successful_count`` present on
*both* sides. Tasks with no common sample count are reported ``not_comparable``;
when that shared count is below a run one side has, ``truncated_comparison`` says so.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import statistics
import sys

# Below this, a few tokens of difference is a large percentage. Flagged, not failed.
_SHORT_OUTPUT_TOKENS = 1000

# Relative tolerance when matching sample counts across sides. Partial sample loss is a
# normal event (a judge 5xx, one dropped rollout); demanding exact equality would make
# such a task unmeasurable, and an unmeasured task still passes the gate.
_SAMPLE_COUNT_TOL = 0.01


def evaluate_verbosity(baseline, candidate, threshold=0.05, dropped_tasks=()):
    """Decide the verbosity gate from harvested per-run token counts.

    Args:
        baseline: ``{task: [(avg_completion_tokens, successful_count), ...]}``
        candidate: same shape as ``baseline``
        threshold: max allowed |delta| as a fraction of baseline (default 0.05)
        dropped_tasks: task names harvest discarded before comparison (excluded run dirs,
            metrics without token counts). They never reach ``per_task``, so without this
            the verdict reads as full coverage over a silently smaller task set.

    Returns:
        dict ``{pass, failure_class, detail, per_task, not_comparable, max_abs_delta}``.
    """
    if not baseline or not candidate:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": f"no metrics harvested (baseline={len(baseline)}, candidate={len(candidate)})",
            "per_task": {},
            "not_comparable": [],
            "max_abs_delta": None,
        }

    per_task, exceeded, worst = {}, [], 0.0
    for task in sorted(set(baseline) | set(candidate)):
        b_runs, c_runs = baseline.get(task, []), candidate.get(task, [])
        if not b_runs or not c_runs:
            per_task[task] = {"status": "not_comparable", "reason": "present on one side only"}
            continue

        # Sample counts must be comparable, not identical. Exact equality is knife-edge:
        # one 5xx'd judge call (294 vs 293) would make a task unmeasurable, and since an
        # unmeasured task still passes, the steady state would be a green gate covering a
        # shrinking subset of the eval set. Anchor on the smaller side's best run and keep
        # runs within _SAMPLE_COUNT_TOL of it -- close enough not to bias the mean, far
        # enough to still reject a genuinely truncated run.
        target = min(max(n for _, n in b_runs), max(n for _, n in c_runs))
        near = lambda n: abs(n - target) <= _SAMPLE_COUNT_TOL * target  # noqa: E731
        b = [t for t, n in b_runs if near(n)]
        c = [t for t, n in c_runs if near(n)]
        dropped = (len(b_runs) - len(b)) + (len(c_runs) - len(c))
        if not b or not c:
            per_task[task] = {
                "status": "not_comparable",
                "reason": (
                    f"no runs within {_SAMPLE_COUNT_TOL:.0%} of a common sample count "
                    f"(baseline n={sorted({n for _, n in b_runs})}, "
                    f"candidate n={sorted({n for _, n in c_runs})})"
                ),
            }
            continue

        b_mean, c_mean = statistics.mean(b), statistics.mean(c)
        delta = (c_mean - b_mean) / b_mean
        within = abs(delta) <= threshold
        best = max(n for _, n in b_runs + c_runs)
        compared_ns = sorted({n for _, n in b_runs + c_runs if near(n)})
        entry = {
            "status": "compared",
            "sample_counts": compared_ns,  # always a list, so consumers need no type-switch
            "baseline_tokens": round(b_mean, 1),
            "candidate_tokens": round(c_mean, 1),
            "delta": round(delta, 4),
            "within_threshold": within,
            "runs": [len(b), len(c)],
            "dropped_mismatched_runs": dropped,
        }
        if max(compared_ns) < best:
            # Both sides truncated to the same n. Same bias applied twice, but it is not
            # the matched-complete answer -- say so rather than implying a full comparison.
            entry["truncated_comparison"] = (
                f"compared at n={compared_ns[0] if len(compared_ns) == 1 else compared_ns}; "
                f"a run at n={best} exists but not on both sides"
            )
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
            "not_comparable": [
                t for t, v in per_task.items() if v.get("status") == "not_comparable"
            ],
            "max_abs_delta": round(worst, 4),
        }
    if not any(v.get("status") == "compared" for v in per_task.values()):
        return {
            "pass": False,
            "failure_class": "SAMPLE_ACCOUNTING_FAILED",
            "detail": "no task had comparable runs on both sides",
            "per_task": per_task,
            "not_comparable": sorted(per_task) if per_task else [],
            "max_abs_delta": None,
        }
    n_cmp = sum(1 for v in per_task.values() if v.get("status") == "compared")
    skipped = [t for t, v in per_task.items() if v.get("status") == "not_comparable"]
    detail = f"all {n_cmp} comparable task(s) within threshold {threshold}"
    if dropped_tasks:
        detail += (
            f"; {len(dropped_tasks)} task(s) DROPPED BEFORE COMPARISON: {sorted(dropped_tasks)}"
        )
    if skipped:
        # Not a failure -- a task can be legitimately incomparable (different sample sets).
        # But it must not be silently absorbed by a passing sibling: the gate is unmeasured
        # for these, and the caller has to decide whether that is acceptable.
        detail += f"; {len(skipped)} task(s) NOT MEASURED: {skipped}"
    return {
        "pass": True,
        "failure_class": None,
        "detail": detail,
        "per_task": per_task,
        "not_comparable": skipped,
        "max_abs_delta": round(worst, 4),
    }


def harvest(side, glob="eval_*", exclude="_high", diagnostics=None):
    """Collect ``{task: [(avg_completion_tokens, successful_count), ...]}`` from NEL artifacts."""
    out, unreadable, excluded, no_metric = {}, [], [], []
    # Depth-agnostic: repo-documented trees put <harness>.<task>/artifacts/ directly under the
    # run dir, while NEL invocations add an invocation level. Hard-coding either one silently
    # harvests nothing (or, worse, only the subset at the matching depth).
    pattern = os.path.join(side, glob, "**", "artifacts", "eval_factory_metrics.json")
    matches = globmod.glob(pattern, recursive=True)
    for path in matches:
        rel = os.path.relpath(path, side).split(os.sep)
        run_dir = rel[0]
        if exclude and exclude in run_dir:
            excluded.append(run_dir)
            continue
        parts = path.split(os.sep)
        # Dir is "<harness>.<task>[.<run_index>]". Strip the run index only when the
        # trailing segment is numeric, and KEEP the harness: two harnesses can expose the
        # same task name, and pooling them would average different generation conditions
        # into one number. Both sides run the same harness (Step 4 config parity), so the
        # fuller key still matches across baseline and candidate.
        name = parts[-3]
        head, _, tail = name.rpartition(".")
        task = head if head and re.fullmatch(r"\d+", tail) else name
        try:
            with open(path) as f:
                stats = json.load(f).get("response_stats", {})
        except (OSError, json.JSONDecodeError) as e:
            unreadable.append(f"{path}: {e}")
            continue
        tokens, count = stats.get("avg_completion_tokens"), stats.get("successful_count")
        if tokens and count:
            out.setdefault(task, []).append((tokens, count))
        else:
            # No usable token count: usually a run that produced no successful samples,
            # otherwise a schema rename. Record it either way -- a silently vanished run
            # makes the remaining pass look more complete than it is.
            no_metric.append(f"{task}: avg_completion_tokens={tokens!r} successful_count={count!r}")
    if diagnostics is not None:
        if not matches:
            diagnostics["no_files_matched"] = pattern
        diagnostics["excluded_run_dirs"] = sorted(set(excluded))
        diagnostics["unreadable_metrics"] = unreadable
        diagnostics["metrics_without_tokens"] = no_metric
    if excluded:
        print(
            f"note: excluded {len(excluded)} run dir(s) matching {exclude!r}: "
            f"{sorted(set(excluded))}",
            file=sys.stderr,
        )
    if unreadable:
        print(f"warning: skipped {len(unreadable)} unreadable metrics file(s)", file=sys.stderr)
        for u in unreadable:
            print(f"  {u}", file=sys.stderr)
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

    diag = {"baseline": {}, "candidate": {}}
    baseline = harvest(args.baseline, args.glob, args.exclude, diag["baseline"])
    candidate = harvest(args.candidate, args.glob, args.exclude, diag["candidate"])
    # Tasks present on neither side because harvest dropped every run for them.
    dropped = set()
    for side_diag in diag.values():
        for entry in side_diag.get("metrics_without_tokens", []):
            dropped.add(entry.split(":", 1)[0])
    dropped -= set(baseline) | set(candidate)
    result = evaluate_verbosity(baseline, candidate, args.threshold, sorted(dropped))
    # Anything harvest dropped belongs in the JSON, not only on stderr: a task whose runs
    # were all excluded never reaches per_task, so the verdict would otherwise look
    # complete for a task set smaller than the eval set.
    result["harvest_diagnostics"] = diag
    print(json.dumps(result, indent=2))
    if result["pass"]:
        return 0
    # Match gate_compare / gate_run / gate_ptq: 2 = the gate could not read its input,
    # 1 = the gate ran and failed. USER_CONFIG_ERROR is this gate's only bad-invocation
    # signal (wrong --baseline root, --glob matched nothing, --exclude swallowed every
    # run dir), so collapsing it into 1 would route an operator error into the
    # VERBOSITY_EXCEEDED triage row -- "a real behavioural change; do not publish".
    return 2 if result["failure_class"] == "USER_CONFIG_ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
