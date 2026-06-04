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

"""Unit tests for the day-0 gate scripts.

These are deterministic — no GPU, cluster, or network. They test the pure
decision functions that the gates rest on. Run with:

    python -m pytest .claude/skills/day0-release/tests/test_gates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_compare import evaluate_comparison
from gate_ptq import evaluate_checkpoint
from gate_run import evaluate_run

# ── gate_compare ──────────────────────────────────────────────────────


def test_compare_accept_within_threshold():
    r = evaluate_comparison(
        {"gpqa": 50.0, "scicode": 30.0}, {"gpqa": 49.5, "scicode": 29.8}, threshold=0.01
    )
    assert r["pass"] and r["decision"] == "ACCEPT"


def test_compare_regression_exceeds_threshold():
    r = evaluate_comparison({"gpqa": 50.0}, {"gpqa": 47.5}, threshold=0.01)  # 2.5 pt drop
    assert not r["pass"] and r["decision"] == "REGRESSION"
    assert "gpqa" in r["detail"]


def test_compare_anomalous_implausible_gain():
    r = evaluate_comparison({"gpqa": 50.0}, {"gpqa": 60.0}, threshold=0.01)  # +10 pts
    assert not r["pass"] and r["decision"] == "ANOMALOUS"


def test_compare_anomalous_out_of_range():
    r = evaluate_comparison({"gpqa": 50.0}, {"gpqa": 150.0}, threshold=0.01)
    assert r["decision"] == "ANOMALOUS"


def test_compare_mismatched_task_sets():
    r = evaluate_comparison({"gpqa": 50.0}, {"scicode": 30.0}, threshold=0.01)
    assert not r["pass"] and r["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"


def test_compare_relative_threshold():
    # 1% relative of 50 = 0.5 pts; a 0.4 pt drop passes, 0.6 fails.
    assert evaluate_comparison({"t": 50.0}, {"t": 49.6}, threshold=0.01, relative=True)["pass"]
    assert not evaluate_comparison({"t": 50.0}, {"t": 49.4}, threshold=0.01, relative=True)["pass"]


# ── gate_run ──────────────────────────────────────────────────────────


def _task(**kw):
    base = {
        "status": "SUCCESS",
        "expected_samples": 100,
        "scored_samples": 100,
        "score": 42.0,
        "errors": [],
    }
    base.update(kw)
    return base


def test_run_all_valid():
    r = evaluate_run({"tasks": {"gpqa": _task(), "scicode": _task()}})
    assert r["pass"]


def test_run_dropped_samples():
    r = evaluate_run({"tasks": {"gpqa": _task(scored_samples=90)}})
    assert not r["pass"] and r["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"


def test_run_judge_error():
    r = evaluate_run({"tasks": {"gpqa": _task(errors=["judge rate limit exceeded"])}})
    assert not r["pass"] and r["failure_class"] == "EVAL_JUDGE_FAILED"


def test_run_missing_score():
    r = evaluate_run({"tasks": {"gpqa": _task(score=None)}})
    assert not r["pass"] and r["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"


def test_run_timeout_is_not_terminal():
    r = evaluate_run({"tasks": {"gpqa": _task(status="TIMEOUT")}})
    assert not r["pass"] and r["failure_class"] == "INFRA_TRANSIENT"


def test_run_no_tasks():
    r = evaluate_run({"tasks": {}})
    assert not r["pass"] and r["failure_class"] == "USER_CONFIG_ERROR"


# ── gate_ptq ──────────────────────────────────────────────────────────


def _ckpt(**kw):
    base = {
        "source_bytes": 16_000_000_000,
        "output_bytes": 8_000_000_000,
        "recipe": "nvfp4",
        "layer_precision_counts": {
            "NVFP4": 224,
            "BF16_or_excluded": 3,
            "unexpected_unquantized": 0,
            "declaration_mismatch": 0,
        },
        "metadata_diffs": [],
    }
    base.update(kw)
    return base


def test_ptq_pass():
    assert evaluate_checkpoint(_ckpt())["pass"]


def test_ptq_not_smaller():
    r = evaluate_checkpoint(_ckpt(output_bytes=16_000_000_000))
    assert not r["pass"] and r["failure_class"] == "QUANT_COVERAGE_FAILURE"


def test_ptq_zero_coverage_is_model_unsupported():
    r = evaluate_checkpoint(
        _ckpt(
            layer_precision_counts={
                "NVFP4": 0,
                "unexpected_unquantized": 0,
                "declaration_mismatch": 0,
            }
        )
    )
    assert not r["pass"] and r["failure_class"] == "MODEL_UNSUPPORTED"


def test_ptq_unexpected_unquantized():
    r = evaluate_checkpoint(
        _ckpt(
            layer_precision_counts={
                "NVFP4": 200,
                "unexpected_unquantized": 24,
                "declaration_mismatch": 0,
            }
        )
    )
    assert not r["pass"] and r["failure_class"] == "QUANT_COVERAGE_FAILURE"


def test_ptq_metadata_diff():
    r = evaluate_checkpoint(_ckpt(metadata_diffs=["chat_template changed"]))
    assert not r["pass"] and r["failure_class"] == "QUANT_COVERAGE_FAILURE"


def test_ptq_unknown_recipe():
    r = evaluate_checkpoint(_ckpt(recipe="mystery"))
    assert not r["pass"] and r["failure_class"] == "USER_CONFIG_ERROR"


# ── regression tests for malformed inputs ────────────────────────────


def test_compare_non_numeric_score_is_anomalous_not_crash():
    # A string/None score must not raise TypeError; it's ANOMALOUS.
    for bad in ("42", None, float("nan"), True):
        r = evaluate_comparison({"gpqa": 50.0}, {"gpqa": bad}, threshold=0.01)
        assert not r["pass"] and r["decision"] == "ANOMALOUS", bad


def test_run_non_numeric_score_fails():
    r = evaluate_run({"tasks": {"gpqa": _task(score="42")}})
    assert not r["pass"] and r["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"


def test_run_running_is_infra_transient():
    r = evaluate_run({"tasks": {"gpqa": _task(status="RUNNING", score=None)}})
    assert not r["pass"] and r["failure_class"] == "INFRA_TRANSIENT"


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
