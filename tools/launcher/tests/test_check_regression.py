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


def test_threshold_with_no_metric_fails(monkeypatch):
    """A configured threshold whose metric never reached trainer_state.json must fail.

    The DFlash/EAGLE trainers print accuracy every log step but did not persist it, so
    this path used to warn and pass -- stamping PASS on a run nothing had checked.
    """
    monkeypatch.setenv("MIN_FINAL_ACC", "0.15")
    passed, messages = check_regression_mod.check_regression({"loss": 0.4})
    assert not passed
    assert any("never evaluated" in m for m in messages)


def test_unset_threshold_is_not_a_failure():
    """Only a threshold the operator actually set is enforced."""
    passed, messages = check_regression_mod.check_regression({"loss": 0.4})
    assert passed
    assert not messages


@pytest.mark.parametrize(("acc", "expected"), [(0.52, True), (0.10, False)])
def test_accuracy_threshold_is_enforced_when_present(monkeypatch, acc, expected):
    """A present metric is compared against the threshold in both directions."""
    monkeypatch.setenv("MIN_FINAL_ACC", "0.15")
    passed, _ = check_regression_mod.check_regression({"loss": 0.4, "accuracy": acc})
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
