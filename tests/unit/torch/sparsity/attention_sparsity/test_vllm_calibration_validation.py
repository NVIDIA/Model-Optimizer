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

"""Serving calibration must not export unmeasured or failed sparsity targets."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[5] / "examples/vllm_serve/calibrate_sparse_attn.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_sparse_attn", _SCRIPT)
calib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calib)


def _config():
    return {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "target_sparsity": {"prefill": 0.5, "decode": 0.5},
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "prefill": {"a": 4.0, "b": 0.0},
                    "decode": {"a": 4.0, "b": 0.0},
                },
            },
        },
    }


def _counts(skipped, total=1000):
    return {
        "total": total,
        "skipped": skipped,
        "launches": 1,
        "unmeasured_launches": 0,
        "min_seq_len": 1024,
    }


def test_validation_uses_tile_weighted_counts_across_workers():
    stats = calib._summarize_validation(
        [
            {"prefill": _counts(0, 10), "decode": _counts(0, 10)},
            {"prefill": _counts(90, 90), "decode": _counts(90, 90)},
        ],
        0.5,
        0.02,
    )
    assert stats["prefill"]["achieved"] == 0.9
    assert stats["decode"]["error_pp"] == 40
    assert not stats["decode"]["passed"]


@pytest.mark.parametrize(
    ("skipped", "passed"), [(480, True), (520, True), (479, False), (521, False)]
)
def test_tolerance_includes_exact_endpoints(skipped, passed):
    stats = calib._summarize_validation(
        [{phase: _counts(skipped) for phase in ("prefill", "decode")}], 0.5, 0.02
    )
    assert all(row["passed"] is passed for row in stats.values())


@pytest.mark.parametrize(
    "bad", [{}, _counts(0, 0), _counts(11, 10), {**_counts(500), "unmeasured_launches": 1}]
)
def test_missing_or_invalid_counters_cannot_pass(bad):
    result = calib._summarize_validation(
        [
            {"prefill": _counts(500), "decode": _counts(500)},
            {"prefill": bad},
        ],
        0.5,
        0.02,
    )
    for phase in ("prefill", "decode"):
        assert result[phase]["achieved"] is None
        assert not result[phase]["passed"]


def test_refinement_measures_prefill_before_decode_and_preserves_input():
    initial = _config()
    calls = []

    def measure(config):
        calls.append(deepcopy(config))
        scales = config["config_groups"]["group_0"]["threshold_scale_factor"]
        return calib._summarize_validation(
            [
                {
                    "prefill": _counts(min(1000, int(scales["prefill"]["a"] / 128 * 1000))),
                    "decode": _counts(min(1000, int(scales["decode"]["a"] / 32 * 1000))),
                }
            ],
            0.5,
            0.02,
        )

    result, measured = calib._refine_scales(initial, measure, 10)
    assert initial == _config()
    assert all(row["passed"] for row in measured.values())
    assert result["config_groups"]["group_0"]["threshold_scale_factor"]["prefill"]["a"] == 64
    assert result["config_groups"]["group_0"]["threshold_scale_factor"]["decode"]["a"] == 16
    for call in calls:
        if call["config_groups"]["group_0"]["threshold_scale_factor"]["decode"]["a"] != 4:
            assert call["config_groups"]["group_0"]["threshold_scale_factor"]["prefill"]["a"] == 64


def test_unreachable_target_is_bounded_and_remains_failed():
    calls = []

    def measure(config):
        calls.append(deepcopy(config))
        return calib._summarize_validation(
            [{phase: _counts(100) for phase in ("prefill", "decode")}], 0.5, 0.02
        )

    _, measured = calib._refine_scales(_config(), measure, 3)
    assert len(calls) <= 1 + 2 * (3 + 1)
    assert not any(row["passed"] for row in measured.values())
    assert all(
        call["config_groups"]["group_0"]["threshold_scale_factor"][phase]["a"] < 1024
        for call in calls
        for phase in ("prefill", "decode")
    )


def test_out_of_domain_fit_is_remeasured_below_lambda_one():
    initial = _config()
    initial["config_groups"]["group_0"]["threshold_scale_factor"]["prefill"]["a"] = 2048

    def measure(config):
        scale = config["config_groups"]["group_0"]["threshold_scale_factor"]["prefill"]["a"]
        prefill = _counts(500)
        if scale >= 1024:
            prefill["unmeasured_launches"] = 1
        return calib._summarize_validation(
            [
                {"prefill": prefill, "decode": _counts(500)},
            ],
            0.5,
            0.02,
        )

    result, stats = calib._refine_scales(initial, measure, 2)
    assert result["config_groups"]["group_0"]["threshold_scale_factor"]["prefill"]["a"] < 1024
    assert stats["prefill"]["passed"]


@pytest.mark.parametrize("split", ["calibration", "held_out"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_failed_gate_writes_receipt_but_never_modifies_checkpoint(tmp_path, split, phase):
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text('{"sentinel": true}')
    report = {
        s: {p: {"passed": True} for p in ("prefill", "decode")} for s in ("calibration", "held_out")
    }
    report[split][phase]["passed"] = False
    with pytest.raises(RuntimeError, match="no config exported"):
        calib._export_validated_config(str(tmp_path), _config(), report, tmp_path, True)
    assert checkpoint.read_text() == '{"sentinel": true}'
    assert not (tmp_path / "sparse_attention_config.json").exists()
    assert (
        json.loads((tmp_path / "sparse_attention_validation.json").read_text())["passed"] is False
    )


def test_passing_gate_exports_and_preserves_existing_checkpoint_fields(tmp_path):
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text('{"sentinel": true}')
    report = {
        s: {p: {"passed": True} for p in ("prefill", "decode")} for s in ("calibration", "held_out")
    }
    calib._export_validated_config(str(tmp_path), _config(), report, tmp_path, True)
    assert json.loads(checkpoint.read_text()) == {
        "sentinel": True,
        "sparse_attention_config": _config(),
    }
    assert json.loads((tmp_path / "sparse_attention_config.json").read_text()) == _config()


def test_prefill_only_refinement_and_export(tmp_path):
    config = _config()
    group = config["config_groups"]["group_0"]
    del group["target_sparsity"]["decode"]
    del group["threshold_scale_factor"]["decode"]

    def measure(candidate):
        return calib._summarize_validation([{"prefill": _counts(500)}], 0.5, 0.02, ("prefill",))

    candidate, stats = calib._refine_scales(config, measure, 10)
    report = {"phases": ["prefill"], "calibration": stats, "held_out": stats}
    calib._export_validated_config("unused", candidate, report, tmp_path, False)
    assert report["passed"]
    assert candidate == config
