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

"""NEL adapter of the verbosity gate: its result-directory harvest and its MLflow metric names.

NEL metrics in MLflow: ``<task>_pass_at_1_avg-of-<N>_answer_tokens_statistics_avg`` (N = repeat
count) with sample count ``<task>_pass_at_1_avg-of-<N>_num_prompts * N``. ``simple_evals`` runs
(GPQA) carry no token metrics; their stored artifact
``<harness>.<task>/artifacts/eval_factory_metrics.json`` has the same ``response_stats`` the
on-disk harvest reads. Retiring NEL means deleting this module and its references.
"""

import glob as globmod
import json
import os
import re
import sys

from _metric_util import positive


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


_NEL_TOKENS = re.compile(
    r"^(?P<prefix>.+)_pass_at_1_avg-of-(?P<n>\d+)_answer_tokens_statistics_avg$"
)


def tokens_from_run(metrics, tags, load_artifact=None):
    """NEL's ``((avg_tokens, count), None)`` or ``(None, reason)``; ``None`` if the run is not NEL's."""
    nel = [
        (int(m.group("n")), m.group("prefix"), k) for k in metrics if (m := _NEL_TOKENS.match(k))
    ]
    if nel:
        prefixes = sorted({prefix for _, prefix, _ in nel})
        if len(prefixes) > 1:
            return None, f"several task prefixes report answer tokens: {prefixes}"
        n, prefix, key = max(nel)  # the largest N is the full repeat count
        prompts = metrics.get(f"{prefix}_pass_at_1_avg-of-{n}_num_prompts")
        if not positive(metrics[key]) or not positive(prompts):
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
        if positive(tokens) and positive(count):
            return (tokens, count), None
        return None, f"artifact {path}: avg_completion_tokens={tokens!r} successful_count={count!r}"
    return None
