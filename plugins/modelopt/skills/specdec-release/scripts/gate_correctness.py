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

"""Spec-dec correctness report: do greedy outputs with and without the drafter match?

REPORT-ONLY: always returns pass true (exit 0) on readable input. How often greedy outputs
drift apart depends on the target's numerics, the engine's kernels and the output length,
so any blocking floor would be a per-model number. The rates go into the report as evidence
for the release review.

Reads ab_summary.json from examples/specdec_bench/compare_ab.py and uses only rows measured
at concurrency 1 with temperature 0 recorded for both arms (batching alone breaks bitwise
equality above concurrency 1), with no comparator problems and speculation on in arm B.
It does not repeat gate_bench's other checks (matched arms, drafter, plausibility), so read
its rates only alongside a passing gate_bench. Pure logic in ``evaluate_correctness``.
"""

from __future__ import annotations

import argparse
import json
import sys


def evaluate_correctness(summary):
    """Report A-vs-B exact match over the greedy, concurrency-1 rows of an A|B summary.

    Returns:
        dict ``{pass, failure_class, blocking, measurable, detail, notes,
        min_exact_match_rate, min_first_turn_exact_match_rate, per_repeat}``.
    """
    greedy = [
        r
        for r in (summary or {}).get("rows") or []
        if r.get("concurrency") == 1
        and r.get("temperature_a") == r.get("temperature_b") == 0
        and isinstance(r.get("match"), dict)
    ]
    rows = [
        r
        for r in greedy
        if not r.get("problems")
        and isinstance(r.get("config_b"), dict)
        and (r["config_b"].get("speculative_algorithm") or "NONE").upper() != "NONE"
    ]
    notes: list[str] = []
    if not (summary or {}).get("ok"):
        notes.append("comparator reported problems; see gate_bench")
    if len(rows) < len(greedy):
        notes.append(
            f"{len(greedy) - len(rows)} greedy row(s) skipped: comparator problems or "
            "speculation off in arm B"
        )
    result = {"pass": True, "failure_class": None, "blocking": False, "notes": notes}
    if not rows:
        return {
            **result,
            "measurable": False,
            "detail": "no concurrency-1 row with temperature 0 recorded for both arms",
        }

    per_repeat = [{"repeat": r.get("repeat"), **r["match"]} for r in rows]
    exact = [m["exact_match_rate"] for m in per_repeat if m.get("exact_match_rate") is not None]
    first = [
        m["first_turn_exact_match_rate"]
        for m in per_repeat
        if m.get("first_turn_exact_match_rate") is not None
    ]
    if any(m.get("unpaired_turns") for m in per_repeat):
        notes.append("some turns had no counterpart in the other arm")
    min_exact = min(exact) if exact else None
    min_first = min(first) if first else None
    detail = (
        f"exact match {min_exact} (all turns), {min_first} (first turns); not gated"
        if exact
        else "no turn pairs to compare"
    )
    return {
        **result,
        "measurable": bool(exact),
        "detail": detail,
        "min_exact_match_rate": min_exact,
        "min_first_turn_exact_match_rate": min_first,
        "per_repeat": per_repeat,
    }


def main(argv=None):
    """CLI entry point: read ab_summary.json and print the report."""
    p = argparse.ArgumentParser(description="Spec-dec correctness report (non-blocking)")
    p.add_argument("--summary", required=True, help="ab_summary.json from compare_ab.py")
    args = p.parse_args(argv)

    try:
        with open(args.summary) as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            raise ValueError(f"{args.summary} does not hold a JSON object")
        rows = summary.get("rows")
        if rows is not None and not (
            isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
        ):
            raise ValueError(f"{args.summary}: 'rows' is not a list of objects")
    except (OSError, ValueError) as e:
        error = {"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}
        print(json.dumps({**error, "notes": [], "inputs": {"summary": args.summary}}, indent=2))
        return 2

    result = evaluate_correctness(summary)
    result["inputs"] = {"summary": args.summary}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
