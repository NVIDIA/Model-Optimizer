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

"""Unit tests for the QAD checkpoint gate (stdlib only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_qad import (
    CLEAR_MISS,
    CONTINUE,
    MISS,
    NEAR_MISS,
    NEEDS_REPEATS,
    PASS,
    SCREEN_PASS,
    evaluate_qad,
)

BASELINE = {"gpqa": 88.0, "ifbench": 80.0, "mmmu": 70.0}


def _gate(gaps_by_it, se=0.4, horizon=150, **kw):
    """gaps_by_it: {iteration: {task: gap}} -> gate result with one SE for every score."""
    ckpts = {it: {t: BASELINE[t] - g for t, g in gaps.items()} for it, gaps in gaps_by_it.items()}
    ses = {it: dict.fromkeys(gaps, se) for it, gaps in gaps_by_it.items()}
    return evaluate_qad(BASELINE, ckpts, ses, horizon, **kw)


def _all(gpqa, ifbench, mmmu):
    return {"gpqa": gpqa, "ifbench": ifbench, "mmmu": mmmu}


def test_pass_needs_every_task_and_picks_the_smallest_worst_gap():
    r = _gate({"50": _all(0.3, 0.2, 0.4), "100": _all(0.1, 0.2, 0.1), "150": _all(2.0, 0.5, 1.0)})
    assert r["decision"] == PASS and r["best_iteration"] == 100
    r = _gate({"50": {"mmmu": 0.6}})  # staged screen: only the failing task so far
    assert r["decision"] == SCREEN_PASS and "gpqa" in r["detail"]


def test_misses_never_stop_a_run_before_the_horizon():
    near = _gate({"50": _all(0.2, 0.3, 1.2)})
    assert near["decision"] == CONTINUE and near["per_checkpoint"][50]["zone"] == NEAR_MISS
    clear = _gate({"50": _all(0.2, 0.3, 2.5), "100": _all(0.2, 0.3, 3.0)})
    assert clear["decision"] == CONTINUE and clear["per_checkpoint"][100]["zone"] == CLEAR_MISS


def test_repeats_only_inside_the_se_band_and_under_the_task_cap():
    gaps = {"100": _all(0.2, 0.3, 1.1)}
    kw = {"runs": {"100": {"mmmu": 4}}, "max_runs": {"mmmu": 8}}
    assert _gate(gaps, **kw)["decision"] == NEEDS_REPEATS
    assert _gate(gaps, runs={"100": {"mmmu": 8}}, max_runs={"mmmu": 8})["decision"] == CONTINUE
    assert _gate({"100": _all(0.2, 0.3, 1.7)}, **kw)["decision"] == CONTINUE  # outside 1.5 SE


def test_miss_at_the_horizon_flags_regression_and_improvement_only_beyond_noise():
    noisy = _gate({"100": _all(0, 0, 1.6), "125": _all(0, 0, 1.9), "150": _all(0, 0, 1.8)})
    assert noisy["decision"] == MISS and noisy["zone"] == NEAR_MISS
    assert not noisy["regressed"] and not noisy["improving"]
    real = _gate({"50": _all(0, 0, 1.2), "100": _all(0, 0, 2.6), "150": _all(0, 0, 2.8)})
    assert real["decision"] == MISS and real["best_iteration"] == 50 and real["regressed"]
    # A clear miss still closing the gap at the horizon: train the same recipe longer.
    closing = _gate({"50": _all(0, 0, 3.2), "100": _all(0, 0, 2.5), "150": _all(0, 0, 1.9)})
    assert closing["zone"] == CLEAR_MISS and closing["improving"] and not closing["regressed"]


def test_bad_inputs_are_config_errors():
    unknown = evaluate_qad(BASELINE, {"50": {"other": 1.0}}, {"50": {"other": 0.4}}, 150)
    assert unknown["failure_class"] == "USER_CONFIG_ERROR"
    assert _gate({"50": _all(0, 0, float("nan"))})["failure_class"] == "USER_CONFIG_ERROR"
    no_se = evaluate_qad(BASELINE, {"50": {"gpqa": 88.0}}, {}, 150)
    assert no_se["failure_class"] == "USER_CONFIG_ERROR"
    assert _gate({"50": _all(0, 0, 0)}, horizon=None)["failure_class"] == "USER_CONFIG_ERROR"
