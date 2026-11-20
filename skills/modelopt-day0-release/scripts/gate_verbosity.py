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
harvests token counts either from NEL eval artifacts on disk or, for any tool that
exports to MLflow, from MLflow runs (``--baseline-mlflow`` / ``--candidate-mlflow``).

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
import glob as globmod
import json
import math
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request

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


def _matches_exclude(run_dir, exclude):
    """True if ``run_dir`` contains ``exclude`` as a run of whole ``_``-delimited tokens.

    Substring matching would drop unrelated dirs by default (``eval_highctx`` contains
    ``_high``), but a single-token check would make any multi-token value such as
    ``high_effort`` match nothing at all -- and "matches nothing" fails *open* here.
    """
    want = [t for t in exclude.strip("_").split("_") if t]
    if not want:
        return False
    toks = run_dir.split("_")
    return any(toks[i : i + len(want)] == want for i in range(len(toks) - len(want) + 1))


def _task_from_metadata(artifacts_dir):
    """Task name declared next to the artifacts, or None if it is not unambiguous.

    Reads ``metadata.yaml`` then ``config.yml``. Both matter: the documented rsync
    (``modelopt-launching-evals``/``analyze-results``) copies ``config.yml`` but not
    ``metadata.yaml``, so keying off metadata alone leaves the documented layout with
    no declared name at all.

    Returns None when the ``tasks:`` block lists more than one entry. Such a file is
    invocation-scoped, so ``tasks[0].name`` is not this directory's task -- taking it
    would pool every job in the invocation, and a metadata-derived key is exempt from
    the collapse guard, so the pooling would be silent. Falling back to the directory
    name re-arms that guard.

    Read with a regex rather than a YAML parser to keep these gates stdlib-only.
    """
    for fname in ("metadata.yaml", "config.yml"):
        try:
            with open(os.path.join(artifacts_dir, fname)) as f:
                text = f.read()
        except OSError:
            continue
        anchor = re.search(r"^(?P<indent>\s*)tasks:\s*$", text, re.MULTILINE)
        if not anchor:
            continue
        # Bound the block by indentation so a later top-level "name:" cannot leak in.
        depth = len(anchor.group("indent"))
        block = []
        for line in text[anchor.end() :].splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # A YAML block sequence puts its "- " items at the SAME indent as the key,
            # so only a non-item at that indent (or anything shallower) ends the block.
            if stripped and (indent < depth or (indent == depth and not stripped.startswith("-"))):
                break
            block.append(line)
        names = re.findall(r"^\s*-?\s*name:\s*(\S+)\s*$", "\n".join(block), re.MULTILINE)
        if len(names) == 1:
            return names[0].strip("\"'")
        if len(names) > 1:
            # Invocation-scoped file. Keep the verdict local to it: a later per-task
            # file (config.yml lives under <harness>.<task>/artifacts/) can still
            # answer unambiguously.
            continue
    return None


def harvest(side, glob="eval_*", exclude="", diagnostics=None):
    """Collect ``{task: [(avg_completion_tokens, successful_count), ...]}`` from NEL artifacts."""
    out, unreadable, excluded, no_metric = {}, [], [], []
    parsed_any = [False]
    source_dirs, excluded_tasks, metadata_keys = {}, set(), set()
    # Depth-agnostic: repo-documented trees put <harness>.<task>/artifacts/ directly under the
    # run dir, while NEL invocations add an invocation level. Hard-coding either one silently
    # harvests nothing (or, worse, only the subset at the matching depth).
    pattern = os.path.join(side, glob, "**", "artifacts", "eval_factory_metrics.json")
    matches = globmod.glob(pattern, recursive=True)
    for path in matches:
        rel = os.path.relpath(path, side).split(os.sep)
        run_dir = rel[0]
        if exclude and _matches_exclude(run_dir, exclude):
            excluded.append(run_dir)
            # Name the task too: a task whose runs are ALL excluded never reaches
            # per_task, so without this it is invisible to the verdict's detail line.
            t = _task_from_metadata(os.path.dirname(path))
            if t is None:
                n = path.split(os.sep)[-3]
                h, _, tl = n.rpartition(".")
                t = h if h and re.fullmatch(r"\d+", tl) else n
            excluded_tasks.add(t)
            continue
        parts = path.split(os.sep)
        # Prefer the task name NEL records next to the artifacts. The directory name
        # alone is not sufficient: some layouts are "<invocation_id>.<job_index>" where
        # the trailing number enumerates TASKS, not repeats, so stripping it collapses
        # every task in an invocation onto one key and pools unrelated means.
        task = _task_from_metadata(os.path.dirname(path))
        from_metadata = task is not None
        if task is None:
            # Fall back to "<harness>.<task>[.<run_index>]": strip the run index only
            # when the trailing segment is numeric, and KEEP the harness, since two
            # harnesses can expose the same task name.
            name = parts[-3]
            head, _, tail = name.rpartition(".")
            task = head if head and re.fullmatch(r"\d+", tail) else name
        try:
            with open(path) as f:
                stats = json.load(f).get("response_stats", {})
        except (OSError, json.JSONDecodeError) as e:
            unreadable.append(f"{path}: {e}")
            continue
        # Registered only after a successful parse: a truncated artifact contributes no
        # data, so counting it toward collapse would block the gate on a partial write.
        if from_metadata:
            metadata_keys.add(task)
        else:
            source_dirs.setdefault(task, set()).add(parts[-3])
        tokens, count = stats.get("avg_completion_tokens"), stats.get("successful_count")
        if tokens and count:
            parsed_any[0] = True
            out.setdefault(task, []).append((tokens, count))
        else:
            # No usable token count: usually a run that produced no successful samples,
            # otherwise a schema rename. Record it either way -- a silently vanished run
            # makes the remaining pass look more complete than it is.
            no_metric.append(f"{task}: avg_completion_tokens={tokens!r} successful_count={count!r}")
    # Only the directory fallback can be ambiguous, and only when the key does not look
    # like "<harness>.<task>": "h.task.1" + "h.task.2" are repeats of one task, whereas
    # "inv.0" + "inv.1" are distinct tasks of one invocation collapsing onto the
    # invocation id. Flagging repeats would make this mandatory gate unpassable.
    # A key is a collapse only if the dirs behind it cannot be explained as repeats:
    #   - a declared name is authoritative                       -> metadata_keys
    #   - "<harness>.<task>" keys come from "<...>.<run_index>"   -> "." in k
    #   - a dir equal to the key sits next to its own indexed
    #     siblings ("ifbench" beside "ifbench.1"), which an
    #     invocation id never does                                -> k in v
    # What remains -- a dotless key with only indexed dirs -- is genuinely ambiguous
    # from the names alone, so it blocks rather than risking a pooled mean.
    collapsed = {
        k: sorted(v)
        for k, v in source_dirs.items()
        if len(v) > 1 and k not in metadata_keys and "." not in k and k not in v
    }
    if diagnostics is not None:
        if collapsed:
            diagnostics["collapsed_keys"] = collapsed
        if not matches:
            diagnostics["no_files_matched"] = pattern
        diagnostics["excluded_run_dirs"] = sorted(set(excluded))
        diagnostics["excluded_tasks"] = sorted(excluded_tasks)
        diagnostics["partially_excluded_tasks"] = sorted(excluded_tasks & set(out))
        diagnostics["unreadable_metrics"] = unreadable
        diagnostics["metrics_without_tokens"] = no_metric
        diagnostics["found_metrics_files"] = bool(matches)
    if collapsed:
        print(
            f"warning: {len(collapsed)} task key(s) came from multiple artifact dirs, so "
            f"unrelated tasks may be pooled: {collapsed}",
            file=sys.stderr,
        )
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


# ── MLflow source ─────────────────────────────────────────────────────────────
# Both NEL and NeMo Gym export to MLflow, so token counts can be read from a run id
# instead of a synced result tree. The metric names differ by tool:
#   NEL:       <task>_pass_at_1_avg-of-<N>_answer_tokens_statistics_avg  (N = repeat count)
#              sample count = <task>_pass_at_1_avg-of-<N>_num_prompts * N
#              NEL ``simple_evals`` runs (GPQA) carry no token metrics; their stored
#              artifact <harness>.<task>/artifacts/eval_factory_metrics.json has the same
#              ``response_stats`` the on-disk harvest reads.
#   NeMo Gym:  <agent>/mean/output_tokens ; sample count = coverage/scored
_NEL_TOKENS = re.compile(
    r"^(?P<prefix>.+)_pass_at_1_avg-of-(?P<n>\d+)_answer_tokens_statistics_avg$"
)
_GYM_TOKENS = re.compile(r"^(?P<agent>[^/]+)/mean/output_tokens$")


def _positive(x):
    """True for a finite number above zero (NaN, infinity, None, strings and booleans are not)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def tokens_from_mlflow_run(run, load_artifact=None):
    """Return ``((avg_completion_tokens, successful_count), None)`` or ``(None, reason)``.

    ``run`` is ``{"status": str, "tags": {..}, "metrics": {..}}``. Only a ``FINISHED`` run
    is used: failed and killed runs are exported too, with few metrics, and their partial
    numbers must not enter a comparison. ``load_artifact(path)`` reads a JSON artifact of
    this run (used only for the NEL ``simple_evals`` fallback). Ambiguous or non-finite
    values are refused with a reason, never guessed.
    """
    status = run.get("status")
    if status != "FINISHED":
        return None, f"run status {status!r} is not FINISHED"
    metrics, tags = run.get("metrics") or {}, run.get("tags") or {}

    gym = {m.group("agent"): k for k in metrics if (m := _GYM_TOKENS.match(k))}
    if len(gym) > 1:
        return None, f"several agents report output tokens: {sorted(gym)}"
    if gym:
        key = next(iter(gym.values()))
        count = metrics.get("coverage/scored")
        if not _positive(metrics[key]) or not _positive(count):
            return None, f"{key}={metrics[key]!r} coverage/scored={count!r}"
        return (metrics[key], count), None

    nel = [
        (int(m.group("n")), m.group("prefix"), k) for k in metrics if (m := _NEL_TOKENS.match(k))
    ]
    if nel:
        prefixes = sorted({prefix for _, prefix, _ in nel})
        if len(prefixes) > 1:
            return None, f"several task prefixes report answer tokens: {prefixes}"
        n, prefix, key = max(nel)  # the largest N is the full repeat count
        prompts = metrics.get(f"{prefix}_pass_at_1_avg-of-{n}_num_prompts")
        if not _positive(metrics[key]) or not _positive(prompts):
            return None, f"{key}={metrics[key]!r} num_prompts={prompts!r}"
        return (metrics[key], round(prompts * n)), None

    harness, task = tags.get("harness"), tags.get("task_name")
    if load_artifact and harness and task:
        path = f"{harness}.{task}/artifacts/eval_factory_metrics.json"
        try:
            stats = load_artifact(path)["response_stats"]
            tokens, count = stats.get("avg_completion_tokens"), stats.get("successful_count")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            return None, f"no token metric and artifact {path} unreadable: {e!r}"
        if _positive(tokens) and _positive(count):
            return (tokens, count), None
        return None, f"artifact {path}: avg_completion_tokens={tokens!r} successful_count={count!r}"
    return None, "no token metric found for either tool"


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
    p.add_argument("--baseline", help="baseline dir containing NEL eval run dirs")
    p.add_argument("--candidate", help="candidate dir containing NEL eval run dirs")
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
