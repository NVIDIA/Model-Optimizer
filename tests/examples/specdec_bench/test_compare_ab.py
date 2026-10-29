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

"""compare_ab.py: speedup, AL and exact match from matched A|B run.py outputs."""

import json

import pytest
from compare_ab import main, summarize


def _write_arm(
    arm_dir,
    tps_entries=(100.0,),
    al=1.0,
    turns_by_question=None,
    responses_file="mtbench_responses.jsonl",
    config=None,
):
    turns_by_question = turns_by_question or {1: ["x", "y"]}
    arm_dir.mkdir(parents=True)
    timing = [
        {
            "Output TPS": tps,
            "TTFT Time": {"mean": "0.0300"},
            "Request Generation Step Time": {"mean": "0.0040"},
        }
        for tps in tps_entries
    ]
    (arm_dir / "timing.json").write_text(json.dumps(timing))
    (arm_dir / "acceptance_rate.json").write_text(
        json.dumps([{"Average_AL": al, "Category_AL": {"math": al}}])
    )
    if config is not False:
        (arm_dir / "configuration.json").write_text(
            json.dumps({"temperature": 0.0, **(config or {})})
        )
    lines = [
        json.dumps({"question_id": q, "choices": [{"turns": turns}]})
        for q, turns in turns_by_question.items()
    ]
    (arm_dir / responses_file).write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    ("layout", "responses_file"),
    [("repeats", "mtbench_responses.jsonl"), ("flat", "responses.jsonl")],
)
def test_summary_reports_speedup_and_matches_only_at_concurrency_1(
    tmp_path, layout, responses_file
):
    """Both directory layouts; plain responses.jsonl is what non-MT-Bench datasets write."""
    base = tmp_path / "r1" if layout == "repeats" else tmp_path
    a_turns = {1: ["x", "y"], 2: ["z", "w"]}
    b_turns = {1: ["x", "y"], 2: ["z", "DIFFERENT"]}
    config_a = {"speculative_algorithm": "NONE"}
    config_b = {
        "speculative_algorithm": "DFLASH",
        "block_size": 7,
        "gpu": "NVIDIA GB300",
        "serving_config": {},
    }
    a_turns_c8 = {1: ["x", "y"], 2: ["DRIFT", "w"]}  # the target alone drifts with batch shape
    for conc in (1, 8):
        turns = a_turns if conc == 1 else a_turns_c8
        _write_arm(base / f"A_c{conc}", [100.0 * conc], 1.0, turns, responses_file, config_a)
        _write_arm(base / f"B_c{conc}", [140.0 * conc], 1.77, b_turns, responses_file, config_b)

    summary = summarize(str(tmp_path))
    rows = {r["concurrency"]: r for r in summary["rows"]}
    assert summary["ok"]
    assert rows[1]["repeat"] == (1 if layout == "repeats" else 0)
    for conc in (1, 8):
        assert rows[conc]["speedup"] == pytest.approx(1.4)
        assert rows[conc]["al_b"] == pytest.approx(1.77)
        assert rows[conc]["turns_a"] == rows[conc]["turns_b"] == 4
        # What each arm ran, for the gate; only CONFIG_KEYS are copied.
        assert rows[conc]["config_a"]["speculative_algorithm"] == "NONE"
        assert rows[conc]["config_b"]["speculative_algorithm"] == "DFLASH"
        assert rows[conc]["config_b"]["block_size"] == 7
        assert rows[conc]["config_b"]["gpu"] == "NVIDIA GB300"
        assert "serving_config" not in rows[conc]["config_b"]
    assert rows[1]["match"] == {
        "pairs": 4,
        "unpaired_turns": 0,
        "exact_match_rate": 0.75,
        "first_turn_pairs": 2,
        "first_turn_exact_match_rate": 1.0,
    }
    assert rows[1]["a_self_match"]["against_concurrency"] == 8
    assert rows[1]["a_self_match"]["exact_match_rate"] == 0.75
    assert rows[1]["a_self_match"]["first_turn_exact_match_rate"] == 0.5
    # Batching alone breaks bitwise equality above concurrency 1, so no match there.
    assert "match" not in rows[8]
    assert "a_self_match" not in rows[8]


@pytest.mark.parametrize(
    ("config", "temperature"),
    [
        ({"temperature": 0.6}, 0.6),
        ({"temperature": None, "runtime_params": {"sampling_kwargs": {"temperature": 0.6}}}, 0.6),
        # The engine then samples at its own default (1.0 for vLLM), not greedily.
        ({"temperature": None, "runtime_params": {"sampling_kwargs": {"top_p": 0.9}}}, None),
        # No configuration.json: the temperature is unknown, so greedy is not assumed.
        (False, None),
    ],
)
def test_no_match_reported_when_sampling(tmp_path, config, temperature):
    """With temperature > 0 a mismatch is expected and the rate would mean nothing."""
    for name in ("A", "B"):
        _write_arm(tmp_path / f"{name}_c1", config=config if config is False else None)
        if config is not False:
            (tmp_path / f"{name}_c1" / "configuration.json").write_text(json.dumps(config))
    (row,) = summarize(str(tmp_path))["rows"]
    assert "match" not in row
    if temperature is not None:
        assert row["temperature_b"] == pytest.approx(temperature)


@pytest.mark.parametrize(
    ("arms", "problem"),
    [
        ({"A": {}, "B": {}}, None),
        ({"A": {}}, "arm B missing"),
        ({"A": {"tps_entries": (50.0, 100.0)}, "B": {}}, "arm A has 2 timing"),
        ({"A": {}, "B": {"turns_by_question": {1: ["x"]}}}, "turn count differs"),
        ({"A": {"config": False}, "B": {}}, "arm A has no configuration.json"),
    ],
)
def test_cli_exit_code_reflects_problems(tmp_path, capsys, arms, problem):
    """A missing arm, appended results or a turn mismatch fails the CLI, not just the JSON."""
    for name, kwargs in arms.items():
        _write_arm(tmp_path / f"{name}_c1", **kwargs)
    out = tmp_path / "ab_summary.json"
    assert main([str(tmp_path), "--json", str(out)]) == (0 if problem is None else 1)
    (row,) = json.loads(out.read_text())["rows"]
    if problem is None:
        assert row["problems"] == []
    else:
        assert any(problem in p for p in row["problems"])
        assert problem in capsys.readouterr().out
