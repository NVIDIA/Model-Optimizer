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

"""Day-0 evaluation-run gate.

Validates that a completed evaluation run is trustworthy before its scores are
compared. Mirrors the checks in evaluation/references/run-validation.md. Pure
decision logic in ``evaluate_run`` (unit-tested without a cluster); ``main``
reads a run-summary JSON and prints the verdict.

The run summary is a dict with, per task:
    {
      "tasks": {
        "<task>": {
          "status": "SUCCESS" | "FAILED" | "RUNNING" | "TIMEOUT",
          "expected_samples": int,
          "scored_samples": int,
          "score": float | null,          # canonical score, if extracted
          "errors": [str, ...]            # judge/parse/sample errors, if any
        }
      }
    }
A walltime TIMEOUT with a healthy resume is not a failure (NEL resumes); only a
terminal FAILED/incomplete run fails the gate.
"""

from __future__ import annotations

import argparse
import json
import sys

_TERMINAL_OK = "SUCCESS"
_RESUMABLE = {"TIMEOUT", "RESUMING"}  # NEL dependency-chain resume — not a failure


def evaluate_run(summary):
    """Validate a completed run summary.

    Returns dict ``{pass, failure_class, detail, per_task}``.
    """
    tasks = (summary or {}).get("tasks")
    if not tasks:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "run summary has no tasks",
            "per_task": {},
        }

    per_task = {}
    problems = []
    for name, t in sorted(tasks.items()):
        status = t.get("status")
        expected = t.get("expected_samples")
        scored = t.get("scored_samples")
        score = t.get("score")
        errors = t.get("errors") or []

        ok = True
        reasons = []

        if status in _RESUMABLE:
            ok = False
            reasons.append(f"status {status}: still resuming, not terminal")
        elif status != _TERMINAL_OK:
            ok = False
            reasons.append(f"status {status!r} is not SUCCESS")

        if errors:
            ok = False
            # Classify the first error to a failure_class hint.
            joined = " ".join(errors).lower()
            if any(k in joined for k in ("judge", "rate limit", "unauthorized", "auth")):
                reasons.append(f"judge/auth error: {errors[0]}")
            else:
                reasons.append(f"error: {errors[0]}")

        if expected is not None and scored is not None and scored != expected:
            ok = False
            reasons.append(f"sample accounting: scored {scored} of {expected}")

        if score is None:
            ok = False
            reasons.append("no score extracted")

        per_task[name] = {"ok": ok, "reasons": reasons}
        if not ok:
            problems.append((name, reasons))

    if not problems:
        return {
            "pass": True,
            "failure_class": None,
            "detail": f"all {len(per_task)} task(s) valid",
            "per_task": per_task,
        }

    # Pick the dominant failure_class for the run.
    flat = " ".join(r for _, rs in problems for r in rs).lower()
    if any(k in flat for k in ("judge", "rate limit", "unauthorized", "auth")):
        fc = "EVAL_JUDGE_FAILED"
    elif "sample accounting" in flat or "no score" in flat:
        fc = "SAMPLE_ACCOUNTING_FAILED"
    elif "resuming" in flat:
        fc = "INFRA_TRANSIENT"
    else:
        fc = "UNKNOWN"

    return {
        "pass": False,
        "failure_class": fc,
        "detail": "; ".join(f"{n}: {', '.join(rs)}" for n, rs in problems),
        "per_task": per_task,
    }


def main(argv=None):
    """CLI entry point: read a run-summary JSON and print the verdict."""
    p = argparse.ArgumentParser(description="Day-0 evaluation-run gate")
    p.add_argument("--run", required=True, help="run-summary JSON (see module docstring)")
    args = p.parse_args(argv)

    try:
        with open(args.run) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}))
        return 2

    result = evaluate_run(summary)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
