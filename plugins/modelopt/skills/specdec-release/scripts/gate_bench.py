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

"""Spec-dec benchmark gate: is the matched A|B measurement valid?

Reads the ab_summary.json written by examples/specdec_bench/compare_ab.py (arm A = target
alone, arm B = target + drafter). Like day0-release's gate_run, it decides whether the
numbers can be trusted, not whether they are good: it sets no performance bar. AL,
speedup and TPS are returned as evidence, with the setup they were measured under.

The expected concurrencies, repeats, algorithm and token count are what the run was
submitted with, not thresholds. Pure decision logic in ``evaluate_bench``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# With speculation off every step emits exactly one token; allow rounding only.
SPEC_OFF_MAX_AL = 1.02
# Mean acceptance length counts the bonus token, so 1.0 means no draft token was accepted.
SPEC_ON_MIN_AL = 1.05
# Settings both arms, and every row, must share for the comparison to be matched.
MATCHED_KEYS = (
    "model_dir",
    "engine",
    "engine_version",
    "gpu",
    "tp_size",
    "ep_size",
    "dataset",
    "dataset_path",
    "output_length",
    "num_requests",
)
# run.py passes the speculative-token count as --block_size for these, --draft_length otherwise.
_BLOCK_SIZE_ALGOS = {"DFLASH", "DSPARK"}

_RANKING = ("BENCH_INCOMPLETE", "ARMS_UNMATCHED", "SPEC_CONFIG_MISMATCH", "ANOMALOUS")


def _num(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _spec_tokens(config):
    algo = (config.get("speculative_algorithm") or "").upper()
    return config.get("block_size" if algo in _BLOCK_SIZE_ALGOS else "draft_length")


def evaluate_bench(
    summary,
    concurrencies=None,
    repeats=1,
    algorithm=None,
    num_spec_tokens=None,
    draft=None,
    target=None,
):
    """Validate an A|B summary.

    Args:
        summary: dict from ab_summary.json.
        concurrencies: concurrencies the run was submitted with; each must be present.
        repeats: how many repeats were submitted; each concurrency needs this many rows.
        algorithm: run.py --speculative_algorithm of arm B (e.g. DFLASH); optional.
        num_spec_tokens: speculative tokens arm B was submitted with; optional.
        draft: expected drafter path or HF id; optional.
        target: expected target model path or HF id; optional.

    Returns:
        dict ``{pass, failure_class, detail, notes, evidence, measurement_setup}``.
    """
    rows = (summary or {}).get("rows") or []
    if not rows:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "summary has no rows",
            "notes": [],
        }

    problems, notes = [], []
    if not summary.get("ok"):
        problems.append(("BENCH_INCOMPLETE", "comparator reported problems"))
    found = {}
    for r in rows:
        found.setdefault(r.get("concurrency"), set()).add(r.get("repeat"))
    problems.extend(
        ("BENCH_INCOMPLETE", f"concurrency {c}: {len(found.get(c, ()))} of {repeats} repeats")
        for c in concurrencies or []
        if len(found.get(c, ())) < repeats
    )
    extra = sorted(set(found) - set(concurrencies or found), key=str)
    if extra:
        notes.append(f"concurrencies not requested: {extra}")
    more = sorted((c for c, reps in found.items() if len(reps) > repeats), key=str)
    if more:
        notes.append(f"more than {repeats} repeat(s) at concurrencies {more}")

    reference = rows[0].get("config_a") or {}
    for r in rows:
        where = f"repeat {r.get('repeat')} C={r.get('concurrency')}"
        problems.extend(("BENCH_INCOMPLETE", f"{where}: {p}") for p in r.get("problems") or [])
        ca, cb = r.get("config_a"), r.get("config_b")
        if not ca or not cb:
            problems.append(("BENCH_INCOMPLETE", f"{where}: arm configuration missing"))
            continue

        n_req = ca.get("num_requests")
        responses = (r.get("responses_a"), r.get("responses_b"))
        if not responses[0] or responses[0] != responses[1]:
            problems.append(("BENCH_INCOMPLETE", f"{where}: responses A/B {responses}"))
        elif isinstance(n_req, int) and responses[0] != n_req:
            # The dataset caps --num_requests (MT-Bench has 80 questions), and a run cut short
            # already shows as a missing result entry, so fewer is not a failure by itself.
            notes.append(f"{where}: {responses[0]} responses per arm, {n_req} requested")
        for key in MATCHED_KEYS:
            values = {ca.get(key), cb.get(key), reference.get(key)}
            if len(values) > 1:
                problems.append(
                    ("ARMS_UNMATCHED", f"{where}: {key} differs ({sorted(map(str, values))})")
                )
        temperatures = {
            r.get("temperature_a"),
            r.get("temperature_b"),
            rows[0].get("temperature_a"),
        }
        if len(temperatures) > 1:
            problems.append(
                (
                    "ARMS_UNMATCHED",
                    f"{where}: temperature differs ({sorted(map(str, temperatures))})",
                )
            )
        # The directory name gives the row its concurrency; the arms record what actually ran.
        for arm, config in (("A", ca), ("B", cb)):
            if config.get("concurrency") != r.get("concurrency"):
                problems.append(
                    ("ARMS_UNMATCHED", f"{where}: arm {arm} ran at C={config.get('concurrency')}")
                )

        if (ca.get("speculative_algorithm") or "").upper() != "NONE":
            problems.append(("SPEC_CONFIG_MISMATCH", f"{where}: arm A is not NONE"))
        al_a, al_b = _num(r.get("al_a")), _num(r.get("al_b"))
        if al_a is not None and al_a > SPEC_OFF_MAX_AL:
            problems.append(
                ("SPEC_CONFIG_MISMATCH", f"{where}: arm A AL {al_a} > {SPEC_OFF_MAX_AL}")
            )
        algo_b = (cb.get("speculative_algorithm") or "").upper()
        if algorithm and algo_b != algorithm.upper():
            problems.append(
                ("SPEC_CONFIG_MISMATCH", f"{where}: arm B ran {algo_b}, not {algorithm}")
            )
        tokens = _spec_tokens(cb)
        if tokens is None:
            # run.py leaves --block_size unset and the engine picks its own default, so the
            # token count is unknown and the AL and speedup ceilings below cannot apply.
            problems.append(
                (
                    "SPEC_CONFIG_MISMATCH",
                    f"{where}: arm B token count not recorded; "
                    "pass --block_size/--draft_length explicitly",
                )
            )
        elif num_spec_tokens is not None and tokens != num_spec_tokens:
            problems.append(
                (
                    "SPEC_CONFIG_MISMATCH",
                    f"{where}: arm B used {tokens} speculative tokens, not {num_spec_tokens}",
                )
            )
        if draft is not None and cb.get("draft_model_dir") != draft:
            problems.append(
                ("SPEC_CONFIG_MISMATCH", f"{where}: drafter {cb.get('draft_model_dir')!r}")
            )
        if target is not None and cb.get("model_dir") != target:
            problems.append(("SPEC_CONFIG_MISMATCH", f"{where}: target {cb.get('model_dir')!r}"))
        if al_b is None or al_b <= SPEC_ON_MIN_AL:
            problems.append(
                (
                    "SPEC_CONFIG_MISMATCH",
                    f"{where}: arm B AL {r.get('al_b')!r} shows no speculation",
                )
            )

        problems.extend(
            ("ANOMALOUS", f"{where}: {key} {r.get(key)!r}")
            for key in ("tps_a", "tps_b", "speedup")
            if _num(r.get(key)) is None or r[key] <= 0
        )
        if isinstance(tokens, int) and al_b is not None and al_b > tokens + 1:
            problems.append(("ANOMALOUS", f"{where}: AL {al_b} > {tokens} tokens + 1"))
        if isinstance(tokens, int) and _num(r.get("speedup")) and r["speedup"] > tokens + 1:
            problems.append(("ANOMALOUS", f"{where}: speedup {r['speedup']} > {tokens} tokens + 1"))
        if (r.get("match") or {}).get("unpaired_turns"):
            problems.append(
                ("ANOMALOUS", f"{where}: {r['match']['unpaired_turns']} unpaired turns")
            )

    cb0 = rows[0].get("config_b") or {}
    result = {
        "notes": notes,
        "evidence": [
            {k: r.get(k) for k in ("repeat", "concurrency", "al_b", "speedup", "tps_a", "tps_b")}
            for r in rows
        ],
        "measurement_setup": {
            **{k: cb0.get(k) for k in (*MATCHED_KEYS, "speculative_algorithm", "draft_model_dir")},
            "num_speculative_tokens": _spec_tokens(cb0),
            "temperature": rows[0].get("temperature_b"),
            "concurrencies": sorted(c for c in found if isinstance(c, int)),
        },
    }
    if not problems:
        return {
            "pass": True,
            "failure_class": None,
            "detail": f"{len(rows)} row(s) valid",
            **result,
        }
    fc = min((c for c, _ in problems), key=_RANKING.index)
    return {
        "pass": False,
        "failure_class": fc,
        "detail": "; ".join(f"{c}: {m}" for c, m in problems),
        **result,
    }


def main(argv=None):
    """CLI entry point: read ab_summary.json and print the verdict."""
    p = argparse.ArgumentParser(description="Spec-dec A|B benchmark gate")
    p.add_argument("--summary", required=True, help="ab_summary.json from compare_ab.py")
    p.add_argument(
        "--concurrencies", help="comma-separated concurrencies the run was submitted with"
    )
    p.add_argument("--repeats", type=int, default=1, help="repeats the run was submitted with")
    p.add_argument("--algorithm", help="arm B --speculative_algorithm, e.g. DFLASH")
    p.add_argument(
        "--num-spec-tokens", type=int, help="speculative tokens arm B was submitted with"
    )
    p.add_argument("--draft", help="expected drafter path or HF id")
    p.add_argument("--target", help="expected target model path or HF id")
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
        concurrencies = (
            [int(c) for c in args.concurrencies.split(",")] if args.concurrencies else None
        )
    except (OSError, ValueError) as e:
        error = {"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}
        print(json.dumps({**error, "notes": [], "inputs": {"summary": args.summary}}, indent=2))
        return 2

    result = evaluate_bench(
        summary,
        concurrencies,
        args.repeats,
        args.algorithm,
        args.num_spec_tokens,
        args.draft,
        args.target,
    )
    result["inputs"] = {"summary": args.summary}
    print(json.dumps(result, indent=2))
    if result["failure_class"] == "USER_CONFIG_ERROR":
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
