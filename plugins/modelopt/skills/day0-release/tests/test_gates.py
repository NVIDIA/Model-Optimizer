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

    python -m pytest "$SKILL_DIR/tests/test_gates.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_compare import evaluate_comparison
from gate_ptq import evaluate_checkpoint
from gate_run import evaluate_run
from gate_verbosity import evaluate_verbosity, harvest
from gate_verbosity import main as verbosity_main

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


def test_compare_0_to_1_scale_full_collapse_is_regression():
    # tau2_bench_telecom reports Result on a 0-1 scale. A full collapse
    # (1.0 -> 0.0) must REGRESS, not pass via the old 0-100 limit assumption.
    r = evaluate_comparison(
        {"tau2_bench_telecom": 1.0}, {"tau2_bench_telecom": 0.0}, threshold=0.01
    )
    assert not r["pass"] and r["decision"] == "REGRESSION"
    assert "tau2_bench_telecom" in r["detail"]


def test_compare_0_to_1_scale_within_threshold_accepts():
    # A 0.005 drop on a 0-1 task is within the 0.01 threshold.
    r = evaluate_comparison({"t": 0.900}, {"t": 0.895}, threshold=0.01)
    assert r["pass"] and r["decision"] == "ACCEPT"


def test_compare_explicit_scale_override():
    # Force a 0-100 scale even though both scores fit in [0, 1]: a 0.5 -> 0.4
    # drop is 0.1 pts on a 0-100 scale, well within threshold.
    r = evaluate_comparison({"t": 0.5}, {"t": 0.4}, threshold=0.01, scales={"t": 100.0})
    assert r["pass"] and r["decision"] == "ACCEPT"


def test_compare_mixed_scales_in_one_suite():
    # 0-100 task within threshold + 0-1 task collapsing -> overall REGRESSION.
    r = evaluate_comparison(
        {"gpqa": 50.0, "tau2_bench_telecom": 1.0},
        {"gpqa": 49.8, "tau2_bench_telecom": 0.0},
        threshold=0.01,
    )
    assert not r["pass"] and r["decision"] == "REGRESSION"
    assert "tau2_bench_telecom" in r["detail"] and "gpqa" not in r["detail"]


def test_compare_invalid_scales_rejected():
    # Non-dict, or non-positive / non-numeric scale values must be rejected
    # (USER_CONFIG_ERROR) rather than crashing the arithmetic.
    for bad in ([1, 2], 5, {"t": "100"}, {"t": 0}, {"t": -5}, {"t": float("nan")}):
        r = evaluate_comparison({"t": 50.0}, {"t": 49.5}, threshold=0.01, scales=bad)
        assert not r["pass"] and r["failure_class"] == "USER_CONFIG_ERROR", bad


def test_compare_empty_or_none_scales_ok():
    # Empty/None scales are valid (fall back to per-task inference).
    for ok in (None, {}, []):
        r = evaluate_comparison({"t": 50.0}, {"t": 49.5}, threshold=0.01, scales=ok)
        assert r["pass"], ok


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


def test_ptq_growth_blocks_when_source_precision_is_undeclared():
    """Default is blocking: for a BF16 source, 'not smaller' is the failure this catches."""
    r = evaluate_checkpoint(_ckpt(output_bytes=16_000_000_000))
    assert not r["pass"] and r["failure_class"] == "SIZE_NOT_REDUCED"


def test_ptq_growth_waived_only_for_a_declared_sub8_source():
    """An already-4-bit source cannot shrink; waive it, but only on that declaration."""
    r = evaluate_checkpoint(_ckpt(output_bytes=16_000_000_000, source_precision="mxfp4"))
    assert r["pass"]
    assert any("SIZE_NOT_REDUCED waived" in n for n in r["notes"])


def test_ptq_coverage_failure_outranks_size():
    """A real coverage problem must not be masked by the size complaint."""
    r = evaluate_checkpoint(
        _ckpt(
            output_bytes=16_000_000_000,
            layer_precision_counts={
                "NVFP4": 200,
                "unexpected_unquantized": 24,
                "declaration_mismatch": 0,
            },
        )
    )
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


# ── gate_verbosity ────────────────────────────────────────────────────


def test_verbosity_within_threshold():
    r = evaluate_verbosity({"gpqa": [(13358.9, 3168)]}, {"gpqa": [(13524.3, 3168)]})
    assert r["pass"] and r["per_task"]["gpqa"]["within_threshold"]
    assert r["max_abs_delta"] == 0.0124


def test_verbosity_exceeds_threshold():
    r = evaluate_verbosity({"t": [(1000.0, 100)]}, {"t": [(1100.0, 100)]})
    assert not r["pass"] and r["failure_class"] == "VERBOSITY_EXCEEDED"


def test_verbosity_shorter_output_also_fails():
    """Two-sided: a large drop in output length is a change too."""
    r = evaluate_verbosity({"t": [(1000.0, 100)]}, {"t": [(800.0, 100)]})
    assert not r["pass"] and r["failure_class"] == "VERBOSITY_EXCEEDED"


def test_verbosity_partial_runs_dropped():
    """A truncated run must not drag a side's mean; only max-n runs count."""
    r = evaluate_verbosity(
        {"ifbench": [(3575.3, 200), (5503.3, 294), (5757.1, 294)]},
        {"ifbench": [(5674.0, 294)]},
    )
    task = r["per_task"]["ifbench"]
    assert task["baseline_tokens"] == 5630.2 and task["dropped_mismatched_runs"] == 1
    assert r["pass"]


def test_verbosity_unequal_sample_counts_not_comparable():
    """Different successful_count on each side means different samples."""
    r = evaluate_verbosity({"tbh": [(2389.7, 4559)]}, {"tbh": [(2257.9, 3394)]})
    assert r["per_task"]["tbh"]["status"] == "not_comparable"
    assert not r["pass"] and r["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"


def test_verbosity_short_output_warns_but_still_gates():
    """Short means make the ratio unstable; warn, but still gate."""
    r = evaluate_verbosity({"tau2": [(355.8, 2872)]}, {"tau2": [(410.1, 2872)]})
    assert "short_output_warning" in r["per_task"]["tau2"]
    assert not r["pass"] and r["failure_class"] == "VERBOSITY_EXCEEDED"


def test_verbosity_task_on_one_side_only():
    r = evaluate_verbosity({"a": [(100.0, 10)]}, {"b": [(100.0, 10)]})
    assert r["per_task"]["a"]["status"] == "not_comparable"
    assert not r["pass"]


def test_verbosity_no_metrics_is_user_config_error():
    r = evaluate_verbosity({}, {})
    assert not r["pass"] and r["failure_class"] == "USER_CONFIG_ERROR"


def test_verbosity_intersection_not_union_of_sample_counts():
    """A count common to both sides must be used even if one side also has a larger run."""
    r = evaluate_verbosity(
        {"t": [(1000.0, 294), (1000.0, 200)]},
        {"t": [(1010.0, 200)]},
    )
    assert r["per_task"]["t"]["status"] == "compared"
    assert r["per_task"]["t"]["runs"] == [1, 1] and r["pass"]


def test_verbosity_not_comparable_is_surfaced_not_masked():
    """A passing task must not hide a sibling the gate could not measure."""
    r = evaluate_verbosity(
        {"ok": [(1000.0, 10)], "skip": [(100.0, 10)]},
        {"ok": [(1005.0, 10)], "skip": [(100.0, 11)]},
    )
    assert r["pass"] and r["not_comparable"] == ["skip"]
    assert "NOT MEASURED" in r["detail"]


def test_verbosity_reports_the_sample_count_compared():
    """A reader must be able to tell n=294 from n=200."""
    r = evaluate_verbosity({"t": [(1000.0, 294)]}, {"t": [(1010.0, 294)]})
    assert r["per_task"]["t"]["sample_count"] == 294
    assert "truncated_comparison" not in r["per_task"]["t"]


def test_verbosity_flags_a_truncated_comparison():
    """Comparing at a shared-but-smaller n is not the matched-complete answer."""
    r = evaluate_verbosity({"t": [(1000.0, 294), (1000.0, 200)]}, {"t": [(1010.0, 200)]})
    t = r["per_task"]["t"]
    assert t["sample_count"] == 200 and "n=294" in t["truncated_comparison"]


def test_verbosity_schema_is_uniform_across_paths():
    """Callers read not_comparable on every verdict, including failures."""
    for r in (
        evaluate_verbosity({"t": [(1000.0, 10)]}, {"t": [(1500.0, 10)]}),  # fail
        evaluate_verbosity({"t": [(1000.0, 10)]}, {"t": [(1000.0, 10)]}),  # pass
        evaluate_verbosity({}, {}),  # config error
    ):
        assert "not_comparable" in r and "max_abs_delta" in r


def test_ptq_schema_has_notes_on_every_path():
    assert "notes" in evaluate_checkpoint({})
    assert "notes" in evaluate_checkpoint(_ckpt())


def test_ptq_large_growth_fails_even_for_a_declared_sub8_source():
    """The waiver covers inherent growth, not an arbitrarily oversized output."""
    r = evaluate_checkpoint(
        _ckpt(
            source_bytes=10_000_000_000,
            output_bytes=20_000_000_000,
            source_precision="mxfp4",
        )
    )
    assert not r["pass"] and r["failure_class"] == "SIZE_NOT_REDUCED"


def test_harvest_keys_by_task_not_harness(tmp_path):
    """Regression: the old parser collapsed every task under a harness into one key."""
    for name in ("simple_evals.gpqa_diamond", "simple_evals.aime.1", "ifbench"):
        d = tmp_path / "eval_run" / "inv123" / name / "artifacts"
        d.mkdir(parents=True)
        (d / "eval_factory_metrics.json").write_text(
            json.dumps({"response_stats": {"avg_completion_tokens": 100.0, "successful_count": 10}})
        )
    # Harness is kept: two harnesses can expose the same task name, and pooling them
    # would average different generation conditions together.
    assert set(harvest(str(tmp_path))) == {
        "simple_evals.gpqa_diamond",
        "simple_evals.aime",
        "ifbench",
    }


def test_harvest_reports_what_it_dropped(tmp_path):
    """A run silently vanishing makes the remaining pass look more complete than it is."""
    good = tmp_path / "eval_run" / "inv" / "h.good" / "artifacts"
    good.mkdir(parents=True)
    good.joinpath("eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
    )
    bad = tmp_path / "eval_run" / "inv" / "h.notok" / "artifacts"
    bad.mkdir(parents=True)
    bad.joinpath("eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"successful_count": 2}})  # no token count
    )
    skipped = tmp_path / "eval_high" / "inv" / "h.excl" / "artifacts"
    skipped.mkdir(parents=True)
    skipped.joinpath("eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
    )
    diag = {}
    out = harvest(str(tmp_path), diagnostics=diag)
    assert set(out) == {"h.good"}
    assert diag["excluded_run_dirs"] == ["eval_high"]
    assert any("h.notok" in m for m in diag["metrics_without_tokens"])


def test_ptq_waiver_requires_canonical_values():
    """A JSON string "false" is truthy, and "not_mxfp4" contains "mxfp4"."""
    grew = {"output_bytes": 16_800_000_000}
    assert not evaluate_checkpoint(_ckpt(**grew, accept_size_growth="false"))["pass"]
    assert not evaluate_checkpoint(_ckpt(**grew, source_precision="not_mxfp4"))["pass"]
    assert evaluate_checkpoint(_ckpt(**grew, accept_size_growth=True))["pass"]
    assert evaluate_checkpoint(_ckpt(**grew, source_precision=" MXFP4 "))["pass"]


def _write_metrics(d, tokens=100.0, count=10):
    d.mkdir(parents=True)
    d.joinpath("eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"avg_completion_tokens": tokens, "successful_count": count}})
    )


def test_harvest_handles_both_documented_depths(tmp_path):
    """Repo docs put <harness>.<task>/artifacts one level under the run dir; NEL adds one."""
    _write_metrics(tmp_path / "eval_shallow" / "h.taskA" / "artifacts")
    _write_metrics(tmp_path / "eval_deep" / "invocation123" / "h.taskB" / "artifacts")
    assert set(harvest(str(tmp_path))) == {"h.taskA", "h.taskB"}


def test_harvest_reports_a_glob_miss(tmp_path):
    """A pattern that matches nothing was the one drop with no diagnostic."""
    diag = {}
    assert harvest(str(tmp_path), diagnostics=diag) == {}
    assert "no_files_matched" in diag


def test_ptq_explicit_acceptance_is_not_capped():
    """accept_size_growth is a human override; the docs state it without a bound."""
    r = evaluate_checkpoint(_ckpt(output_bytes=32_000_000_000, accept_size_growth=True))
    assert r["pass"]
    assert "accepted explicitly" in r["notes"][0]
    assert "''" not in r["notes"][0]  # must not claim an undeclared precision


def test_ptq_declared_source_is_capped():
    """A checkable claim stays bounded by the growth it explains."""
    r = evaluate_checkpoint(_ckpt(output_bytes=32_000_000_000, source_precision="mxfp4"))
    assert not r["pass"] and "explains at most" in r["detail"]


def test_ptq_rejected_source_precision_is_self_diagnosing():
    """An operator who wrote a rejected value must be able to tell that from writing none."""
    grew = {"output_bytes": 16_800_000_000}
    bad = evaluate_checkpoint(_ckpt(**grew, source_precision="mxfp4 experts / bf16 attention"))
    absent = evaluate_checkpoint(_ckpt(**grew))
    assert "mxfp4 experts / bf16 attention" in bad["detail"]  # echoes what it read
    assert "None" in absent["detail"]
    for r in (bad, absent):
        assert "mxfp4" in r["detail"] and "int4" in r["detail"]  # lists the vocabulary


def test_verbosity_exit_codes_match_sibling_gates(tmp_path, capsys):
    """2 = bad invocation, 1 = the gate failed, 0 = pass -- as gate_compare/run/ptq do."""

    def side(name, tokens):
        d = tmp_path / name / "eval_x" / "inv" / "h.t" / "artifacts"
        _write_metrics(d, tokens=tokens)
        return str(tmp_path / name)

    assert (
        verbosity_main(
            ["--baseline", str(tmp_path / "nope"), "--candidate", str(tmp_path / "nope2")]
        )
        == 2
    )
    b, c = side("b", 100.0), side("c", 300.0)
    assert verbosity_main(["--baseline", b, "--candidate", c]) == 1
    c2 = side("c2", 101.0)
    assert verbosity_main(["--baseline", b, "--candidate", c2]) == 0
    capsys.readouterr()


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
