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

"""Regression-gate behaviour, especially the unevaluated-threshold case."""

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "common" / "check_regression.py"
_spec = importlib.util.spec_from_file_location("check_regression", _MODULE)
assert _spec is not None and _spec.loader is not None, f"cannot load {_MODULE}"
check_regression_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_regression_mod)


@pytest.fixture(autouse=True)
def _clear_thresholds(monkeypatch):
    for var in ("MAX_FINAL_LOSS", "MIN_FINAL_ACC", "MAX_FINAL_PERPLEXITY"):
        monkeypatch.delenv(var, raising=False)


def test_threshold_with_no_metric_fails():
    """A configured threshold whose metric never reached trainer_state.json must fail.

    The DFlash/EAGLE trainers print accuracy every log step but did not persist it, so
    this path used to warn and pass -- stamping PASS on a run nothing had checked.
    """
    passed, messages = check_regression_mod.check_regression({"loss": 0.4}, {"MIN_FINAL_ACC": 0.15})
    assert not passed
    assert any("never evaluated" in m for m in messages)


def test_unset_threshold_is_not_a_failure():
    """Only a threshold the operator actually set is enforced."""
    passed, messages = check_regression_mod.check_regression({"loss": 0.4}, {})
    assert passed
    assert not messages


def test_empty_threshold_var_is_unset(tmp_path, monkeypatch):
    """An empty MIN_FINAL_ACC disables that check instead of crashing on float('')."""
    monkeypatch.setenv("MAX_FINAL_LOSS", "5.0")
    monkeypatch.setenv("MIN_FINAL_ACC", "")
    (tmp_path / "trainer_state.json").write_text('{"log_history": [{"loss": 1.0}]}')
    verdict = check_regression_mod.evaluate(str(tmp_path))
    assert verdict["pass"]
    assert verdict["thresholds"] == {"MAX_FINAL_LOSS": 5.0}


@pytest.mark.parametrize(("acc", "expected"), [(0.52, True), (0.10, False)])
def test_accuracy_threshold_is_enforced_when_present(acc, expected):
    """A present metric is compared against the threshold in both directions."""
    passed, _ = check_regression_mod.check_regression(
        {"loss": 0.4, "accuracy": acc}, {"MIN_FINAL_ACC": 0.15}
    )
    assert passed is expected


def test_metrics_pick_up_any_accuracy_key(tmp_path):
    """The trainer logs train_acc/parallel_0_step_0, not a key literally named accuracy."""
    state = tmp_path / "trainer_state.json"
    state.write_text(
        '{"log_history": [{"loss": 0.39, "train_acc/parallel_0_step_0": 0.5153, "step": 6000}]}'
    )
    metrics = check_regression_mod.get_final_metrics(state)
    assert metrics["loss"] == pytest.approx(0.39)
    assert metrics["accuracy"] == pytest.approx(0.5153)


def test_latest_checkpoint_is_chosen_by_step_number(tmp_path):
    """checkpoint-12000 is newer than checkpoint-8000, though it sorts first as a string."""
    for step in (8000, 12000):
        ckpt = tmp_path / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text("{}")
    latest = check_regression_mod.find_trainer_state(str(tmp_path))
    assert Path(latest).parent.name == "checkpoint-12000"


@pytest.mark.parametrize(
    ("state_text", "exit_code", "failure_class"),
    [
        (None, 1, "METRICS_MISSING"),  # no trainer_state.json at all
        ('{"log_hist', 1, "METRICS_MISSING"),  # truncated by a job killed mid-save
        ("[]", 1, "METRICS_MISSING"),  # valid JSON, wrong shape
        ('{"log_history": [{"loss": null}]}', 1, "METRICS_MISSING"),
        ('{"log_history": [{"loss": NaN}]}', 1, "TRAIN_REGRESSION"),  # a diverged run
        # An unchecked non-finite metric still serializes, as null.
        ('{"log_history": [{"loss": 4.5, "acc": Infinity}]}', 0, None),
        (json.dumps({"log_history": [{"loss": 6.5}]}), 1, "TRAIN_REGRESSION"),
        (json.dumps({"log_history": [{"loss": 4.5}]}), 0, None),
    ],
)
def test_cli_exit_code_and_json_verdict(
    tmp_path, monkeypatch, state_text, exit_code, failure_class
):
    """With a threshold set, the CLI fails both a regression and a run it cannot read."""
    monkeypatch.setenv("MAX_FINAL_LOSS", "5.0")
    if state_text is not None:
        ckpt = tmp_path / "checkpoint-2000"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(state_text)
    verdict_path = tmp_path / "verdict.json"
    assert check_regression_mod.main([str(tmp_path), "--json", str(verdict_path)]) == exit_code
    # Strict JSON: NaN/Infinity would break non-Python readers of the verdict.
    verdict = json.loads(verdict_path.read_text(), parse_constant=pytest.fail)
    assert verdict["pass"] is (exit_code == 0)
    assert verdict["failure_class"] == failure_class
    assert verdict["thresholds"] == {"MAX_FINAL_LOSS": 5.0}


@pytest.mark.parametrize("raw", ["7,0", "nan", "inf"])
def test_invalid_threshold_writes_failed_verdict(tmp_path, monkeypatch, raw):
    """A malformed or non-finite threshold fails as a readable verdict, not a traceback."""
    monkeypatch.setenv("MAX_FINAL_LOSS", raw)
    verdict_path = tmp_path / "verdict.json"
    assert check_regression_mod.main([str(tmp_path), "--json", str(verdict_path)]) == 1
    verdict = json.loads(verdict_path.read_text(), parse_constant=pytest.fail)
    assert verdict["failure_class"] == "METRICS_MISSING"
    assert verdict["thresholds"] == {}
