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

"""Compare matched specdec_bench runs: target alone (A) vs target + draft (B).

Reads run.py --save_dir outputs laid out as <root>/r<repeat>/{A,B}_c<concurrency>/, or
<root>/{A,B}_c<concurrency>/ for a single repeat:
  timing.json              "Output TPS", "TTFT Time", "Request Generation Step Time"
  acceptance_rate.json     "Average_AL", "Category_AL"
  *responses.jsonl         generated text per question and turn (mtbench_, specbench_ or
                           plain responses.jsonl, depending on the dataset)
  configuration.json       "temperature", else runtime_params.sampling_kwargs.temperature
                           ("engine default" when sampling_kwargs set none); CONFIG_KEYS

Writes one summary with a row per (repeat, concurrency): speedup = Output TPS B / Output
TPS A, AL of both arms, step time, TTFT, result-entry and turn counts, and each arm's
CONFIG_KEYS so a gate can confirm what actually ran.

At concurrency 1 and temperature 0 the row also gets an exact-match comparison of A and B
outputs. Above concurrency 1, batching alone breaks bitwise equality; with sampling, a
mismatch is expected. Neither reports a match. The same row's "a_self_match" compares A with
A at the next concurrency of the repeat, also at temperature 0: how far the target alone
drifts with batch shape, the noise floor for the A-vs-B rate (some hybrid targets agree
with themselves on few turns).

A row lists "problems" when an arm is missing, has no configuration.json, a result file has
other than exactly one entry (run.py appends to an existing save_dir), an arm produced no
responses, or the arms produced different numbers of turns. Any problem makes the CLI exit 1.

Usage:
  python3 compare_ab.py <root> [--json ab_summary.json]
"""

import argparse
import json
import os
import re
import sys
from glob import glob

ARM_DIR = re.compile(r"([AB])_c(\d+)")
REPEAT_DIR = re.compile(r"r(\d+)")
# Copied per arm from configuration.json: which algorithm, drafter and speculative-token
# count ran (--block_size for DFLASH/DSPARK, --draft_length otherwise), and the settings
# both arms must share for a matched comparison (target, engine, GPU, parallelism, data).
CONFIG_KEYS = (
    "speculative_algorithm",
    "draft_model_dir",
    "block_size",
    "draft_length",
    "num_requests",
    "model_dir",
    "engine",
    "engine_version",
    "gpu",
    "tp_size",
    "ep_size",
    "dataset",
    "dataset_path",
    "output_length",
    "concurrency",
)


def _load_entries(path):
    """All entries of a run.py result file (Metric.write appends one per run)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _mean(stat):
    """specdec_bench stores statistics as {"mean": "0.1234", ...} with string values."""
    if isinstance(stat, dict):
        return float(stat["mean"])
    return float(stat) if stat is not None else None


def _responses(arm_dir):
    out = {}
    for path in glob(os.path.join(arm_dir, "*responses.jsonl")):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                out[row["question_id"]] = row["choices"][0]["turns"]
    return out


def _configuration(arm_dir):
    path = os.path.join(arm_dir, "configuration.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _temperature(config):
    """Sampling temperature with run.py's precedence: --temperature, then runtime_params."""
    if config is None:
        return None
    if config.get("temperature") is not None:
        return config["temperature"]
    runtime_params = config.get("runtime_params") or {}
    if "sampling_kwargs" not in runtime_params:
        return 0
    # sampling_kwargs reach the engine as-is; without a temperature the engine default
    # applies (1.0 for vLLM, TRT-LLM and AutoDeploy), which this file cannot see.
    return (runtime_params["sampling_kwargs"] or {}).get("temperature", "engine default")


def load_arm(arm_dir):
    """Summarize one run.py save_dir; metrics come from the last entry of each file."""
    timing = _load_entries(os.path.join(arm_dir, "timing.json"))
    acceptance = _load_entries(os.path.join(arm_dir, "acceptance_rate.json"))
    responses = _responses(arm_dir)
    config = _configuration(arm_dir)
    arm = {
        "timing_entries": len(timing),
        "acceptance_entries": len(acceptance),
        "responses": len(responses),
        "turns": sum(len(turns) for turns in responses.values()),
        "temperature": _temperature(config),
        "config": {key: config.get(key) for key in CONFIG_KEYS} if config else None,
        "_responses": responses,
    }
    if timing:
        arm["tps"] = float(timing[-1]["Output TPS"])
        arm["ttft_s"] = _mean(timing[-1].get("TTFT Time"))
        arm["step_s"] = _mean(timing[-1].get("Request Generation Step Time"))
    if acceptance:
        arm["al"] = acceptance[-1].get("Average_AL")
        arm["category_al"] = acceptance[-1].get("Category_AL")
    return arm


def compare_outputs(a, b):
    """Exact-match rate over (question, turn) pairs present in both arms."""
    pairs = exact = first_pairs = first_exact = 0
    for qid in sorted(set(a) & set(b)):
        for turn, (ta, tb) in enumerate(zip(a[qid], b[qid])):
            pairs += 1
            exact += ta == tb
            if turn == 0:
                first_pairs += 1
                first_exact += ta == tb
    total_turns = sum(map(len, a.values())) + sum(map(len, b.values()))
    return {
        "pairs": pairs,
        "unpaired_turns": total_turns - 2 * pairs,
        "exact_match_rate": exact / pairs if pairs else None,
        "first_turn_pairs": first_pairs,
        "first_turn_exact_match_rate": first_exact / first_pairs if first_pairs else None,
    }


def _arm_dirs(root):
    """Yield (repeat, arm, concurrency, path) for every {A,B}_c<N> dir under root."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if m := ARM_DIR.fullmatch(name):
            yield 0, m.group(1), int(m.group(2)), path
        elif r := REPEAT_DIR.fullmatch(name):
            for arm_name in sorted(os.listdir(path)):
                arm_path = os.path.join(path, arm_name)
                if (m := ARM_DIR.fullmatch(arm_name)) and os.path.isdir(arm_path):
                    yield int(r.group(1)), m.group(1), int(m.group(2)), arm_path


def _problems(a, b):
    problems = []
    for name, arm in (("A", a), ("B", b)):
        if arm is None:
            problems.append(f"arm {name} missing")
            continue
        if arm["config"] is None:
            problems.append(f"arm {name} has no configuration.json")
        if arm["timing_entries"] != 1 or arm["acceptance_entries"] != 1:
            problems.append(
                f"arm {name} has {arm['timing_entries']} timing / "
                f"{arm['acceptance_entries']} acceptance entries, expected 1"
            )
        if not arm["responses"]:
            problems.append(f"arm {name} produced no responses")
    if a and b and a["turns"] != b["turns"]:
        problems.append(f"turn count differs: A {a['turns']}, B {b['turns']}")
    return problems


def summarize(root):
    """Build the A|B summary for every (repeat, concurrency) found under root."""
    arms = {}
    for repeat, arm, conc, path in _arm_dirs(root):
        arms[(repeat, conc, arm)] = load_arm(path)

    rows = []
    for repeat, conc in sorted({(r, c) for r, c, _ in arms}):
        a, b = arms.get((repeat, conc, "A")), arms.get((repeat, conc, "B"))
        row = {"repeat": repeat, "concurrency": conc, "problems": _problems(a, b)}
        for name, arm in (("a", a), ("b", b)):
            if arm is None:
                continue
            for key in ("tps", "ttft_s", "step_s", "al", "timing_entries", "responses", "turns"):
                row[f"{key}_{name}"] = arm.get(key)
            row[f"temperature_{name}"] = arm["temperature"]
            row[f"config_{name}"] = arm["config"]
        if b and b.get("category_al"):
            row["category_al_b"] = b["category_al"]
        if row.get("tps_a") and row.get("tps_b"):
            row["speedup"] = row["tps_b"] / row["tps_a"]
        # Exact match is only meaningful for greedy decoding without batching effects, so
        # both arms must record temperature 0; an unknown temperature gets no match.
        if (
            conc == 1
            and a is not None
            and b is not None
            and a["temperature"] == b["temperature"] == 0
        ):
            row["match"] = compare_outputs(a["_responses"], b["_responses"])
            higher = sorted(c for r, c, n in arms if r == repeat and n == "A" and c > conc)
            control = arms[(repeat, higher[0], "A")] if higher else None
            if control is not None and control["temperature"] == 0:
                row["a_self_match"] = {
                    "against_concurrency": higher[0],
                    **compare_outputs(a["_responses"], control["_responses"]),
                }
        rows.append(row)
    return {
        "root": os.path.abspath(root),
        "ok": bool(rows) and not any(r["problems"] for r in rows),
        "rows": rows,
    }


def _fmt(value, fmt):
    return format(value, fmt) if isinstance(value, (int, float)) else "-"


def main(argv=None):
    """CLI entry point: print a table, optionally write JSON; exit 1 on any problem."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Directory holding the A/B save_dirs")
    parser.add_argument("--json", dest="json_path", help="Write the summary to this file")
    args = parser.parse_args(argv)

    summary = summarize(args.root)
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(summary, f, indent=2)
    if not summary["rows"]:
        print(f"no {{A,B}}_c<N> directories under {args.root}", file=sys.stderr)
        return 2

    print("| Repeat | C | AL (B) | TPS A | TPS B | Speedup | Turns A/B | Exact match (C=1, T=0) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in summary["rows"]:
        match = r.get("match") or {}
        print(
            f"| {r['repeat']} | {r['concurrency']} | {_fmt(r.get('al_b'), '.3f')} | "
            f"{_fmt(r.get('tps_a'), '.1f')} | {_fmt(r.get('tps_b'), '.1f')} | "
            f"{_fmt(r.get('speedup'), '.2f')} | {r.get('turns_a', '-')}/{r.get('turns_b', '-')} | "
            f"{_fmt(match.get('exact_match_rate'), '.1%')} |"
        )
    for r in summary["rows"]:
        for problem in r["problems"]:
            print(f"PROBLEM repeat {r['repeat']} concurrency {r['concurrency']}: {problem}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
