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
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_compare import evaluate_comparison
from gate_ptq import _recipe_bits, evaluate_checkpoint
from gate_run import evaluate_run
from gate_verbosity import _matches_exclude, _task_from_metadata, evaluate_verbosity, harvest
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
    assert r["per_task"]["t"]["sample_counts"] == [294]
    assert "truncated_comparison" not in r["per_task"]["t"]


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
    out = harvest(str(tmp_path), exclude="_high", diagnostics=diag)
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


# Emitted by a gate script but deliberately absent from the triage table. Empty today:
# every emitted class has a row, including USER_CONFIG_ERROR. Exempting a class that
# HAS a row would silently un-pin that row.
_NOT_TRIAGED: set[str] = set()


def test_verbosity_tolerates_small_sample_count_drift():
    """One dropped sample must not make a task unmeasurable -- an unmeasured task passes."""
    r = evaluate_verbosity({"t": [(1000.0, 294)]}, {"t": [(1010.0, 293)]})
    assert r["per_task"]["t"]["status"] == "compared"
    r2 = evaluate_verbosity({"t": [(1000.0, 294)]}, {"t": [(1010.0, 200)]})
    assert r2["per_task"]["t"]["status"] == "not_comparable"  # genuinely truncated


def test_every_emitted_failure_class_has_a_triage_row():
    """The triage table is the dispatch contract; a class with no row is undefined behaviour."""
    scripts = Path(__file__).parent.parent / "scripts"
    # Match only strings used AS a failure_class, not every uppercase literal -- decisions
    # (ACCEPT/REGRESSION) and SLURM states (PENDING/RUNNING) are not failure classes.
    emitted = set()
    for f in scripts.glob("gate_*.py"):
        src = f.read_text()
        emitted |= set(re.findall(r'"failure_class":\s*"([A-Z_]+)"', src))
        emitted |= set(re.findall(r'failures\.append\(\s*\(\s*\n?\s*"([A-Z_]+)"', src))
    rows = set(
        re.findall(r"^\| `([A-Z_]+)` \|", (scripts.parent / "SKILL.md").read_text(), re.MULTILINE)
    )
    # Subtract only declared exemptions: intersecting with an allowlist would filter out
    # exactly the newly-emitted class this test exists to catch.
    missing = emitted - _NOT_TRIAGED - rows
    assert not missing, f"failure classes emitted but absent from the triage table: {missing}"


def test_ptq_note_names_which_waiver_won_when_both_are_set():
    """The note is the only record of why the gate was waived; it must not assert a falsehood."""
    r = evaluate_checkpoint(
        _ckpt(output_bytes=16_800_000_000, source_precision="mxfp4", accept_size_growth=True)
    )
    note = r["notes"][0]
    assert r["pass"] and "no source precision declared" not in note
    assert "takes precedence" in note and "mxfp4" in note


def test_verbosity_detail_names_tasks_dropped_before_comparison():
    """Tasks harvest drops on both sides never reach per_task, so detail must say so."""
    r = evaluate_verbosity({"t": [(1000.0, 10)]}, {"t": [(1005.0, 10)]}, dropped_tasks=["gone"])
    assert r["pass"] and "DROPPED BEFORE COMPARISON" in r["detail"] and "gone" in r["detail"]


def test_verbosity_anchors_on_the_largest_matchable_count_not_min_of_maxes():
    """b=[294,200] vs c=[280,200] share 200; anchoring on min(max) would discard the task."""
    r = evaluate_verbosity(
        {"t": [(1000.0, 294), (1000.0, 200)]},
        {"t": [(1010.0, 280), (1010.0, 200)]},
    )
    t = r["per_task"]["t"]
    assert t["status"] == "compared" and t["sample_counts"] == [200]
    assert "n=294" in t["truncated_comparison"]


def test_harvest_prefers_the_task_name_from_metadata(tmp_path):
    """<invocation>.<job_index> layouts collapse every task onto one key without this."""
    for job, name in ((0, "simple_evals.gpqa"), (1, "tau2.telecom")):
        d = tmp_path / "eval_run" / f"inv123.{job}" / "artifacts"
        d.mkdir(parents=True)
        (d / "metadata.yaml").write_text(f"evaluation:\n  tasks:\n    - name: {name}\n")
        (d / "eval_factory_metrics.json").write_text(
            json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
        )
    assert set(harvest(str(tmp_path))) == {"simple_evals.gpqa", "tau2.telecom"}


def test_harvest_flags_collapsed_task_keys(tmp_path):
    """Without metadata, distinct dirs mapping to one key must be reported, not pooled silently."""
    for job in (0, 1):
        d = tmp_path / "eval_run" / f"inv123.{job}" / "artifacts"
        d.mkdir(parents=True)
        (d / "eval_factory_metrics.json").write_text(
            json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
        )
    diag = {}
    harvest(str(tmp_path), diagnostics=diag)
    assert "collapsed_keys" in diag and diag["collapsed_keys"]["inv123"] == [
        "inv123.0",
        "inv123.1",
    ]


def test_dropped_tasks_covers_the_excluded_channel(tmp_path):
    """A task whose runs are all excluded must still be nameable in the verdict."""
    d = tmp_path / "eval_high" / "inv" / "h.only_high" / "artifacts"
    d.mkdir(parents=True)
    (d / "eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
    )
    diag = {}
    assert harvest(str(tmp_path), exclude="_high", diagnostics=diag) == {}
    assert diag["excluded_tasks"] == ["h.only_high"]


def _ckpt_fp8(**kw):
    return _ckpt(
        recipe="fp8",
        layer_precision_counts={
            "FP8": 224,
            "unexpected_unquantized": 0,
            "declaration_mismatch": 0,
        },
        **kw,
    )


def test_verbosity_collapsed_keys_block_the_gate():
    """A pooled mean can land inside threshold; detecting collapse must change the verdict."""
    r = evaluate_verbosity(
        {"k": [(100.0, 10)]},
        {"k": [(101.0, 10)]},
        collapsed_keys={"inv": ["inv.0", "inv.1"]},
    )
    assert not r["pass"] and r["failure_class"] == "USER_CONFIG_ERROR"
    assert "layout not understood" in r["detail"]


def test_coverage_caveat_appears_on_failure_paths_too():
    """A caller told the gate failed also needs to know coverage was incomplete."""
    d = ["gone"]
    fail = evaluate_verbosity({"t": [(100.0, 10)]}, {"t": [(200.0, 10)]}, dropped_tasks=d)
    nocmp = evaluate_verbosity({"t": [(100.0, 10)]}, {"t": [(100.0, 99)]}, dropped_tasks=d)
    assert not fail["pass"] and "DROPPED BEFORE COMPARISON" in fail["detail"]
    assert not nocmp["pass"] and "DROPPED BEFORE COMPARISON" in nocmp["detail"]


_CFG_ONE = "evaluation:\n  tasks:\n  - name: {n}\n    container: x\nexport:\n  name: notthis\n"
_CFG_MANY = "evaluation:\n  tasks:\n  - name: task_a\n  - name: task_b\n"


def _mk_run(root, leaf, cfg=None, meta=None):
    d = root / "eval_run" / leaf / "artifacts"
    d.mkdir(parents=True)
    if cfg:
        (d / "config.yml").write_text(cfg)
    if meta:
        (d / "metadata.yaml").write_text(meta)
    (d / "eval_factory_metrics.json").write_text(
        json.dumps({"response_stats": {"avg_completion_tokens": 10.0, "successful_count": 2}})
    )
    return d


def test_task_name_read_from_config_yml_when_metadata_absent(tmp_path):
    """The documented rsync copies config.yml but not metadata.yaml."""
    _mk_run(tmp_path, "inv123.0", cfg=_CFG_ONE.format(n="simple_evals.gpqa"))
    _mk_run(tmp_path, "inv123.1", cfg=_CFG_ONE.format(n="tau2.telecom"))
    diag = {}
    out = harvest(str(tmp_path), diagnostics=diag)
    assert set(out) == {"simple_evals.gpqa", "tau2.telecom"}
    assert "collapsed_keys" not in diag


def test_multi_task_declaration_is_ambiguous_and_rearms_the_guard(tmp_path):
    """An invocation-wide task list must not key every job onto tasks[0].name."""
    _mk_run(tmp_path, "inv9.0", meta=_CFG_MANY)
    _mk_run(tmp_path, "inv9.1", meta=_CFG_MANY)
    diag = {}
    harvest(str(tmp_path), diagnostics=diag)
    assert diag["collapsed_keys"] == {"inv9": ["inv9.0", "inv9.1"]}


def test_task_name_block_is_bounded_by_indentation(tmp_path):
    """A later top-level `name:` must not be mistaken for the task name."""
    d = _mk_run(tmp_path, "y.0", cfg=_CFG_ONE.format(n="only_this"))
    assert _task_from_metadata(str(d)) == "only_this"


def test_ambiguous_file_does_not_veto_an_unambiguous_later_one(tmp_path):
    """An invocation-wide metadata.yaml must not discard a per-task config.yml."""
    d = _mk_run(tmp_path, "z.0", cfg=_CFG_ONE.format(n="real.task"), meta=_CFG_MANY)
    assert _task_from_metadata(str(d)) == "real.task"


def test_verbosity_zero_baseline_mean_is_not_a_crash():
    """The pure entry point is imported directly; it must not raise ZeroDivisionError."""
    r = evaluate_verbosity({"t": [(0.0, 10)]}, {"t": [(5.0, 10)]})
    assert r["per_task"]["t"]["status"] == "not_comparable"
    assert "zero" in r["per_task"]["t"]["reason"]


def test_verbosity_zero_samples_is_an_eval_failure_not_a_config_error():
    """Artifacts present but no successful samples means the eval died, not the command."""
    found = evaluate_verbosity({}, {}, found_artifacts=True)
    absent = evaluate_verbosity({}, {}, found_artifacts=False)
    assert found["failure_class"] == "SAMPLE_ACCOUNTING_FAILED"
    assert absent["failure_class"] == "USER_CONFIG_ERROR"


def test_unreadable_artifact_does_not_count_toward_collapse(tmp_path):
    """A truncated write contributes no data, so it must not block the gate."""
    good = _mk_run(tmp_path, "inv1.0")
    bad = tmp_path / "eval_run" / "inv1.1" / "artifacts"
    bad.mkdir(parents=True)
    (bad / "eval_factory_metrics.json").write_text("{truncated")
    assert good.exists()
    diag = {}
    out = harvest(str(tmp_path), diagnostics=diag)
    assert set(out) == {"inv1"}
    assert "collapsed_keys" not in diag
    assert diag["unreadable_metrics"]


def test_ptq_unknown_recipe_surfaces_the_recipe_error_not_a_size_verdict():
    """Growth cannot be assessed without a target precision; the recipe name is the fix."""
    r = evaluate_checkpoint(
        _ckpt(recipe="bogus_xyz", output_bytes=16_800_000_000, source_precision="bf16")
    )
    assert r["failure_class"] == "USER_CONFIG_ERROR" and "unknown recipe" in r["detail"]


def test_recipe_bits_is_exact_not_substring():
    """A name mentioning a KV/excluded precision must not resolve to that precision.

    Substring matching made "fp8_bf16_kv" resolve to 16, which would waive the size
    gate for a bf16 source -- a real failed compression reported as pass.
    """
    assert _recipe_bits("fp8") == 8
    assert _recipe_bits("nvfp4_mlp_only") == 4
    assert _recipe_bits("int4_awq") == 4
    assert _recipe_bits("fp8_bf16_kv") is None
    assert _recipe_bits("unknown_recipe") is None


def test_recipe_table_entries_are_self_consistent():
    """One table carries bucket and bits, so a new recipe cannot omit either."""
    from gate_ptq import _PRECISION_BITS, _RECIPE_EXPECTED_PRECISION

    for recipe, entry in _RECIPE_EXPECTED_PRECISION.items():
        bucket, bits = entry
        assert isinstance(bucket, str) and isinstance(bits, int), recipe
        assert bits in set(_PRECISION_BITS.values()), recipe


# ── consolidated: one test per rule ────────


def test_ptq_size_waiver_rules():
    """The waiver fires only when the recipe cannot shrink the declared source.

    Table-driven so the rule reads as a rule: waive iff source bits <= recipe target
    bits, bounded by _INHERENT_GROWTH_MAX, with accept_size_growth as an unbounded
    override. Replaces five single-case tests that each pinned one row.
    """
    small, huge = 16_800_000_000, 32_000_000_000  # 1.05x and 2.0x of _ckpt's source
    cases: list[tuple[Any, dict[str, Any], int, bool, str]] = [
        # (checkpoint helper, kwargs, out_bytes, expect_pass, needle)
        (_ckpt, {"source_precision": "mxfp4"}, small, True, "waived"),
        (_ckpt, {"source_precision": "mxfp4"}, huge, False, "explains at most"),
        (_ckpt, {}, small, False, "not a recognised"),
        (_ckpt, {"source_precision": "bf16"}, small, False, "should compress"),
        (_ckpt_fp8, {"source_precision": "fp8"}, small, True, "waived"),
        (_ckpt_fp8, {"source_precision": "int8"}, small, True, "waived"),
        (_ckpt_fp8, {"source_precision": "mxfp8"}, small, True, "waived"),
        (_ckpt_fp8, {"source_precision": "bf16"}, small, False, "should compress"),
        (_ckpt, {"accept_size_growth": True}, huge, True, "accepted explicitly"),
    ]
    for helper, kw, out, expect_pass, needle in cases:
        r = evaluate_checkpoint(helper(output_bytes=out, **kw))
        label = f"{kw} out={out}"
        assert r["pass"] is expect_pass, label
        text = " ".join(r["notes"]) if expect_pass else r["detail"]
        assert needle in text, f"{label}: {text}"


def test_ptq_waiver_rejects_non_canonical_values():
    """A JSON string "false" is truthy and "not_mxfp4" contains "mxfp4"."""
    grew = {"output_bytes": 16_800_000_000}
    assert not evaluate_checkpoint(_ckpt(**grew, accept_size_growth="false"))["pass"]
    assert not evaluate_checkpoint(_ckpt(**grew, source_precision="not_mxfp4"))["pass"]
    assert evaluate_checkpoint(_ckpt(**grew, source_precision=" MXFP4 "))["pass"]


def test_ptq_rejection_messages_diagnose_without_leaking_the_waiver_list():
    """Every rejection branch names its own cause and points at the reference.

    Replaces three tests that each checked one property of these strings.
    """
    grew = {"output_bytes": 32_000_000_000}
    known = evaluate_checkpoint(_ckpt(**grew, source_precision="bf16"))["detail"]
    unknown = evaluate_checkpoint(_ckpt(**grew, source_precision="mxfp4 experts / bf16 attn"))
    bounded = evaluate_checkpoint(_ckpt(**grew, source_precision="mxfp4"))["detail"]
    absent = evaluate_checkpoint(_ckpt(**grew))["detail"]

    assert "should compress" in known  # recognised, but should have shrunk
    assert "not a recognised" in unknown["detail"]
    assert "mxfp4 experts / bf16 attn" in unknown["detail"]  # echoes what it read
    assert "None" in absent  # distinguishes absent from rejected
    assert "explains at most" in bounded
    for text in (known, unknown["detail"], bounded, absent):
        assert "checkpoint-validation.md" in text  # pointer on every branch
        assert "int4" not in text  # never enumerate the values that would waive


def test_exclude_matches_whole_token_runs():
    """Substring matching drops unrelated dirs; single-token matching makes
    multi-token values inert, which fails open."""
    assert _matches_exclude("eval_sglang_gpqa_high", "_high")
    assert _matches_exclude("eval_high_effort_run", "high_effort")
    for d, e in [
        ("eval_highctx", "_high"),
        ("eval_qwen_highmem", "_high"),
        ("eval_sglang_gpqa", "_high"),
        ("eval_sglang_gpqa", "high_effort"),
    ]:
        assert not _matches_exclude(d, e), (d, e)


def test_collapse_guard_recognises_repeats_in_both_key_shapes(tmp_path):
    """Repeats must never read as collapse -- that blocks a mandatory gate.

    Dotted "<harness>.<task>" keys and bare keys with a sibling at the key name are
    both run-indexed tasks; only indexed dirs with no such sibling are ambiguous.
    """
    for leaves in (["h.task.1", "h.task.2"], ["ifbench", "ifbench.1"]):
        root = tmp_path / leaves[0].replace(".", "_")
        for leaf in leaves:
            _mk_run(root, leaf)
        diag = {}
        out = harvest(str(root), diagnostics=diag)
        assert len(out) == 1 and len(next(iter(out.values()))) == 2, leaves
        assert "collapsed_keys" not in diag, leaves


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
