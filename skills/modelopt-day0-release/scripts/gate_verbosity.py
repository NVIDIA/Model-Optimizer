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
harvests token counts either from a tool's result directories on disk (``nel_adapter``) or,
for any tool that exports to MLflow, from MLflow runs (``--baseline-mlflow`` /
``--candidate-mlflow``; metric names are in each tool's adapter module).

Read ``response_stats.avg_completion_tokens``. The ``reasoning.*_tokens`` fields
are always 0 (only the reasoning/content split is missing, not the total) and the
``*_words`` siblings are a proxy that can disagree with the gate.

``main`` also attaches ``harvest_diagnostics`` recording anything dropped before the
comparison (excluded run dirs, unreadable artifacts, metrics missing token counts).

Two filters run before averaging: skip run dirs matching ``--exclude`` (opt-in, for a
mismatched reasoning-effort variant) and keep only runs at the largest ``successful_count`` present on
*both* sides. Tasks with no common sample count are reported ``not_comparable``;
when that shared count is below a run one side has, ``truncated_comparison`` says so.
"""

import argparse
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request

import gym_adapter
import nel_adapter
from nel_adapter import _matches_exclude, _task_from_metadata, harvest  # noqa: F401

# Below this, a few tokens of difference is a large percentage. Flagged, not failed.
_SHORT_OUTPUT_TOKENS = 1000

# Relative tolerance when matching sample counts across sides. Partial sample loss is a
# normal event (a judge 5xx, one dropped rollout); demanding exact equality would make
# such a task unmeasurable, and an unmeasured task still passes the gate.
_SAMPLE_COUNT_TOL = 0.01


def _coverage_caveat(dropped_tasks):
    """Suffix naming tasks harvest discarded, appended to every verdict's ``detail``.

    The hygiene evidence matters most on the failure paths: a caller told the gate
    failed also needs to know the run did not cover the whole task set.
    """
    if not dropped_tasks:
        return ""
    return f"; {len(dropped_tasks)} task(s) DROPPED BEFORE COMPARISON: {sorted(dropped_tasks)}"


def evaluate_verbosity(
    baseline,
    candidate,
    threshold=0.05,
    dropped_tasks=(),
    collapsed_keys=None,
    found_artifacts=False,
):
    """Decide the verbosity gate from harvested per-run token counts.

    Args:
        baseline: ``{task: [(avg_completion_tokens, successful_count), ...]}``
        candidate: same shape as ``baseline``
        threshold: max allowed |delta| as a fraction of baseline (default 0.05)
        dropped_tasks: task names harvest discarded before comparison -- every run
            excluded by ``--exclude``, or no run with a usable token count. They never
            reach ``per_task``, so without this the verdict reads as full coverage over a
            silently smaller task set. Artifacts that failed to parse are reported by path
            in ``harvest_diagnostics.unreadable_metrics``, since no task name exists yet.
        found_artifacts: True if harvest located metrics files at all. Distinguishes "the
            eval produced no usable samples" (a failed run) from "you pointed me at the
            wrong place" (a bad invocation), which route to different exit codes.
        collapsed_keys: ``{key: [dirs]}`` where distinct artifact dirs landed on one task
            key. Blocking: the means would pool unrelated tasks, and a pooled delta can
            fall inside the threshold and report a pass nobody asked for.

    Returns:
        dict ``{pass, failure_class, detail, per_task, not_comparable, max_abs_delta}``.
    """
    if not baseline or not candidate:
        # Artifacts present but nothing usable => the eval failed, not the invocation.
        cls = "SAMPLE_ACCOUNTING_FAILED" if found_artifacts else "USER_CONFIG_ERROR"
        return {
            "pass": False,
            "failure_class": cls,
            "detail": (
                f"no usable metrics (baseline={len(baseline)}, candidate={len(candidate)}); "
                + (
                    "artifacts were found but no run reported successful samples -- check "
                    "the eval, not the invocation"
                    if found_artifacts
                    else "no metrics files matched -- check --baseline/--candidate/--glob"
                )
            ),
            "per_task": {},
            "not_comparable": [],
            "max_abs_delta": None,
        }

    if collapsed_keys:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": (
                f"artifact layout not understood: {len(collapsed_keys)} task key(s) came "
                f"from multiple directories, so their means would pool unrelated tasks: "
                f"{collapsed_keys}"
            ),
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
        # shrinking subset of the eval set.
        #
        # Anchor on the LARGEST count both sides can match within tolerance -- not on
        # min(max(b), max(c)), which can name a count neither side pairs at: with
        # b=[294,200] and c=[280,200] that anchors at 280 and discards the task, even
        # though both sides have a run at 200.
        b_ns = {n for _, n in b_runs}
        c_ns = {n for _, n in c_runs}
        near = lambda ns, a: any(abs(n - a) <= _SAMPLE_COUNT_TOL * a for n in ns)  # noqa: E731
        target = next(
            (a for a in sorted(b_ns | c_ns, reverse=True) if near(b_ns, a) and near(c_ns, a)),
            None,
        )
        if target is None:
            per_task[task] = {
                "status": "not_comparable",
                "reason": (
                    f"no sample count matchable within {_SAMPLE_COUNT_TOL:.0%} on both sides "
                    f"(baseline n={sorted(b_ns)}, candidate n={sorted(c_ns)})"
                ),
            }
            continue
        keep = lambda n: abs(n - target) <= _SAMPLE_COUNT_TOL * target  # noqa: E731
        b = [t for t, n in b_runs if keep(n)]
        c = [t for t, n in c_runs if keep(n)]
        dropped = (len(b_runs) - len(b)) + (len(c_runs) - len(c))

        b_mean, c_mean = statistics.mean(b), statistics.mean(c)
        if not b_mean:
            # Unreachable via harvest (it gates on a truthy token count), but this is the
            # documented pure entry point and is imported directly by callers and tests.
            per_task[task] = {
                "status": "not_comparable",
                "reason": "baseline mean is zero, so a relative delta is undefined",
            }
            continue
        delta = (c_mean - b_mean) / b_mean
        within = abs(delta) <= threshold
        best = max(n for _, n in b_runs + c_runs)
        compared_ns = sorted({n for _, n in b_runs + c_runs if keep(n)})
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
            "detail": f"tasks exceeding threshold ({threshold}): {exceeded}"
            + _coverage_caveat(dropped_tasks),
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
            "detail": "no task had comparable runs on both sides" + _coverage_caveat(dropped_tasks),
            "per_task": per_task,
            "not_comparable": sorted(per_task) if per_task else [],
            "max_abs_delta": None,
        }
    n_cmp = sum(1 for v in per_task.values() if v.get("status") == "compared")
    skipped = [t for t, v in per_task.items() if v.get("status") == "not_comparable"]
    detail = f"all {n_cmp} comparable task(s) within threshold {threshold}"
    if dropped_tasks:
        detail += _coverage_caveat(dropped_tasks)
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


# ── MLflow source ─────────────────────────────────────────────────────────────
# Every supported tool exports to MLflow, so token counts can be read from a run id instead of
# a synced result tree. Each tool's metric names are in its adapter module; the first adapter
# that recognizes the run answers.
_ADAPTERS = (gym_adapter, nel_adapter)


def tokens_from_mlflow_run(run, load_artifact=None):
    """Return ``((avg_completion_tokens, successful_count), None)`` or ``(None, reason)``.

    ``run`` is ``{"status": str, "tags": {..}, "metrics": {..}}``. Only a ``FINISHED`` run
    is used: failed and killed runs are exported too, with few metrics, and their partial
    numbers must not enter a comparison. ``load_artifact(path)`` reads a JSON artifact of
    this run (used by an adapter whose runs carry no token metric). Ambiguous or non-finite
    values are refused with a reason, never guessed.
    """
    status = run.get("status")
    if status != "FINISHED":
        return None, f"run status {status!r} is not FINISHED"
    metrics, tags = run.get("metrics") or {}, run.get("tags") or {}
    for adapter in _ADAPTERS:
        answer = adapter.tokens_from_run(metrics, tags, load_artifact)
        if answer is not None:
            return answer
    return None, "no token metric found"


def harvest_mlflow(run_ids_by_task, get_run, get_artifact, diagnostics=None):
    """Collect ``{task: [(avg_completion_tokens, successful_count), ...]}`` from MLflow runs.

    ``run_ids_by_task`` is ``{task: [run_id, ...]}`` (one run is one benchmark of one
    submission, so a task can have several). ``get_run(run_id)`` returns the run dict and
    ``get_artifact(run_id, path)`` a parsed JSON artifact. Runs that cannot be used are
    listed in ``diagnostics["runs_not_used"]`` with the reason, never dropped silently, and a
    task that kept some runs but lost others is named in ``partially_excluded_tasks`` so the
    verdict says its comparison is incomplete.
    """
    out, not_used, fetched = {}, [], False
    for task, run_ids in sorted(run_ids_by_task.items()):
        for rid in run_ids:
            try:
                run = get_run(rid)
            except Exception as e:  # network or HTTP error: report, do not guess
                not_used.append({"task": task, "run_id": rid, "reason": f"could not fetch: {e}"})
                continue
            fetched = True
            try:
                result, reason = tokens_from_mlflow_run(
                    run, lambda path, rid=rid: get_artifact(rid, path)
                )
            except Exception as e:  # a malformed run is reported, it must not crash the gate
                result, reason = None, f"could not read the run: {e!r}"
            if result is None:
                not_used.append({"task": task, "run_id": rid, "reason": reason})
            else:
                out.setdefault(task, []).append(result)
    if diagnostics is not None:
        diagnostics["runs_not_used"] = not_used
        diagnostics["found_metrics_files"] = fetched
        diagnostics["excluded_tasks"] = []
        diagnostics["partially_excluded_tasks"] = sorted({n["task"] for n in not_used} & set(out))
        diagnostics["metrics_without_tokens"] = [f"{n['task']}: {n['reason']}" for n in not_used]
    return out


def _mlflow_http(uri):
    """Return ``(get_run, get_artifact)`` bound to an MLflow tracking server (read-only)."""
    base = uri.rstrip("/")
    headers = {}
    token = os.environ.get("MLFLOW_TRACKING_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def _get(path):
        req = urllib.request.Request(base + path, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def get_run(run_id):
        r = _get("/api/2.0/mlflow/runs/get?run_id=" + urllib.parse.quote(run_id))["run"]
        return {
            "status": r["info"]["status"],
            "tags": {t["key"]: t["value"] for t in r["data"].get("tags", [])},
            "metrics": {m["key"]: m["value"] for m in r["data"].get("metrics", [])},
        }

    def get_artifact(run_id, path):
        return _get(
            "/get-artifact?path="
            + urllib.parse.quote(path)
            + "&run_uuid="
            + urllib.parse.quote(run_id)
        )

    return get_run, get_artifact


def main(argv=None):
    """CLI entry point: harvest both sides from eval artifacts and print the verdict."""
    p = argparse.ArgumentParser(description="Day-0 verbosity gate")
    p.add_argument("--baseline", help="baseline dir containing eval run dirs (directory mode)")
    p.add_argument("--candidate", help="candidate dir containing eval run dirs (directory mode)")
    p.add_argument(
        "--baseline-mlflow",
        help="JSON file {task: [mlflow_run_id, ...]} for the baseline (any tool that exports to MLflow)",
    )
    p.add_argument("--candidate-mlflow", help="same for the candidate")
    p.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking server (default: $MLFLOW_TRACKING_URI); read-only access",
    )
    p.add_argument("--glob", default="eval_*", help="run-dir glob within each side")
    p.add_argument(
        "--exclude",
        default="",
        help=(
            "skip run dirs carrying this as whole _-delimited token(s), e.g. '_high' for a "
            "reasoning-effort variant. Empty by default: which tier is canonical is "
            "run-specific, so excluding one is an operator decision"
        ),
    )
    p.add_argument(
        "--threshold", type=float, default=0.05, help="max |delta| fraction (default 0.05)"
    )
    args = p.parse_args(argv)

    use_mlflow = bool(args.baseline_mlflow or args.candidate_mlflow)
    use_dirs = bool(args.baseline or args.candidate)
    sides_ok = (
        (args.baseline_mlflow and args.candidate_mlflow)
        if use_mlflow
        else (args.baseline and args.candidate)
    )
    if use_mlflow == use_dirs or not sides_ok:
        print(
            json.dumps(
                {
                    "pass": False,
                    "failure_class": "USER_CONFIG_ERROR",
                    "detail": "give both --baseline and --candidate, or both "
                    "--baseline-mlflow and --candidate-mlflow (not a mix)",
                }
            )
        )
        return 2

    diag = {"baseline": {}, "candidate": {}}
    if use_mlflow:
        try:
            with open(args.baseline_mlflow) as f:
                base_runs = json.load(f)
            with open(args.candidate_mlflow) as f:
                cand_runs = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(
                json.dumps({"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)})
            )
            return 2
        for name, runs in (("--baseline-mlflow", base_runs), ("--candidate-mlflow", cand_runs)):
            if not isinstance(runs, dict) or not all(
                isinstance(v, list) and v and all(isinstance(r, str) and r for r in v)
                for v in runs.values()
            ):
                print(
                    json.dumps(
                        {
                            "pass": False,
                            "failure_class": "USER_CONFIG_ERROR",
                            "detail": f"{name} must be a JSON object {{task: [run_id, ...]}}",
                        }
                    )
                )
                return 2
        if not args.mlflow_uri:
            print(
                json.dumps(
                    {
                        "pass": False,
                        "failure_class": "USER_CONFIG_ERROR",
                        "detail": "no MLflow server: pass --mlflow-uri or set MLFLOW_TRACKING_URI",
                    }
                )
            )
            return 2
        get_run, get_artifact = _mlflow_http(args.mlflow_uri)
        baseline = harvest_mlflow(base_runs, get_run, get_artifact, diag["baseline"])
        candidate = harvest_mlflow(cand_runs, get_run, get_artifact, diag["candidate"])
    else:
        baseline = harvest(args.baseline, args.glob, args.exclude, diag["baseline"])
        candidate = harvest(args.candidate, args.glob, args.exclude, diag["candidate"])
    # Tasks present on neither side because harvest dropped every run for them --
    # via --exclude, or because no run had a usable token count. (unreadable_metrics
    # is keyed by path, so a task name is not recoverable from it.)
    dropped = set()
    for side_diag in diag.values():
        dropped |= set(side_diag.get("excluded_tasks", []))
        for entry in side_diag.get("metrics_without_tokens", []):
            dropped.add(entry.split(":", 1)[0])
    # Tasks excluded ENTIRELY drop out below; a task that merely lost some runs stays a
    # key on both sides, so it would vanish from the caveat -- which is the case where
    # the exclusion actually changed the comparison.
    partial = set()
    for side_diag in diag.values():
        partial |= set(side_diag.get("partially_excluded_tasks", []))
    dropped -= set(baseline) | set(candidate)
    collapsed = {}
    for side, side_diag in diag.items():
        # Namespace by side: a key colliding on both would otherwise report only the
        # last side's directories.
        for k, dirs in side_diag.get("collapsed_keys", {}).items():
            collapsed[f"{side}:{k}"] = dirs
    found = any(sd.get("found_metrics_files") for sd in diag.values())
    result = evaluate_verbosity(
        baseline, candidate, args.threshold, sorted(dropped), collapsed, found
    )
    if partial:
        result["partially_excluded_tasks"] = sorted(partial)
        result["detail"] += f"; {len(partial)} task(s) had SOME runs excluded: {sorted(partial)}"
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
