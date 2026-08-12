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

import json

from modelopt.torch.puzzletron.stages.diagnostics import (
    _diagnostic_checkpoint_needs_rebuild,
    _hidden_only_diagnostic_ready,
    _hidden_width_ranking_verdict,
    _hidden_width_result_metrics,
    _merge_reused_sort_equivalence,
    _near_teacher_axis_targets,
    _parent_sweep_sanity_verdict,
    _ratio_aligned_hidden_widths,
    _select_diagnostic_hidden_width,
    _select_layers,
    _write_hidden_only_diagnostic_artifacts,
)


def test_near_teacher_axis_targets_selects_two_largest_legal_values():
    config = {
        "search_space": {
            "axes": {
                "ffn_intermediate": {
                    "teacher_value": 12288,
                    "values": [6144, 10240, 8192, 12288],
                },
                "binary_axis": {"teacher_value": 2, "values": [1]},
            }
        }
    }

    assert _near_teacher_axis_targets(
        config, ["ffn_intermediate", "binary_axis"], count=2
    ) == {"ffn_intermediate": [10240, 8192], "binary_axis": [1]}


def test_hidden_width_targets_apply_requested_ratios_and_alignment():
    assert _ratio_aligned_hidden_widths(4096, [0.875, 0.25], alignment=256) == [
        3584,
        1024,
    ]


def test_hidden_width_diagnostic_selects_nearest_legal_seven_eighths_width():
    assert _select_diagnostic_hidden_width(4096, [4096, 3840, 3584]) == 3584
    assert _select_diagnostic_hidden_width(2688, [2688, 2496, 2304]) == 2304


def test_hidden_width_diagnostic_tie_prefers_larger_reduced_width():
    assert _select_diagnostic_hidden_width(800, [800, 690, 710]) == 710


def test_hidden_width_verdict_can_treat_original_prefix_as_diagnostic_only():
    values = {"activation": 38.0, "random": 31.0, "reverse": 95.0}

    diagnostic = _hidden_width_ranking_verdict(
        values,
        tolerance=0.0,
        require_beats_random=False,
    )
    strict = _hidden_width_ranking_verdict(
        values,
        tolerance=0.0,
        require_beats_random=True,
    )

    assert diagnostic == {
        "passed": True,
        "beats_random": False,
        "beats_reverse": True,
        "require_beats_random": False,
    }
    assert strict["passed"] is False


def test_hidden_only_diagnostic_finishes_without_empty_parent_sweep(tmp_path):
    artifacts = tmp_path / "artifacts"
    temporary = tmp_path / "diagnostics" / "temporary"
    temporary.mkdir(parents=True)
    (temporary / "reverse.bin").write_bytes(b"temporary")
    hidden = {
        "hidden_width": 1344,
        "teacher_hidden_width": 2688,
        "primary_metric": "raw_replacement_loss",
        "passed": True,
        "beats_random": True,
        "beats_reverse": True,
        "rows": [],
    }

    summary = _write_hidden_only_diagnostic_artifacts(
        artifacts_dir=artifacts,
        temporary_root=temporary,
        hidden_width_summary=hidden,
        cleanup_reverse=True,
    )

    assert summary["status"] == "complete"
    assert summary["axes"] == ["hidden_width"]
    assert summary["parent_sweep"] == {"status": "not_applicable"}
    assert not temporary.exists()
    assert json.loads((artifacts / "activation_diagnostic_summary.json").read_text())[
        "hidden_width"
    ]["passed"] is True
    assert (artifacts / "activation_diagnostic_table.md").is_file()
    assert (artifacts / "activation_diagnostic_scores.csv").is_file()
    assert json.loads((artifacts / "diagnostic_cleanup.json").read_text())[
        "reverse_checkpoint_removed"
    ] is True


def test_hidden_only_guard_allows_nonmaster_rank_without_summary():
    assert _hidden_only_diagnostic_ready(
        axes=["hidden_width"], hidden_width_summary=None, is_master=False
    )

    try:
        _hidden_only_diagnostic_ready(
            axes=["hidden_width"], hidden_width_summary=None, is_master=True
        )
    except RuntimeError as error:
        assert "rank 0" in str(error)
    else:
        raise AssertionError("master rank without a width verdict should fail")


def test_diagnostic_retry_rebuilds_partial_indexed_checkpoint(tmp_path):
    checkpoint = tmp_path / "reverse"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "sorted_permutations.json").write_text('{"x": [0]}')
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")

    assert _diagnostic_checkpoint_needs_rebuild(checkpoint) is True

    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
    assert _diagnostic_checkpoint_needs_rebuild(checkpoint) is False


def test_random_diagnostic_layers_are_deterministic_and_axis_specific():
    eligible = list(range(24))

    ffn = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="ffn_intermediate",
    )
    repeated = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="ffn_intermediate",
    )
    gdn = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="gdn_key_groups",
    )

    assert len(ffn) == 3
    assert ffn == repeated
    assert ffn != gdn
    assert ffn == sorted(ffn)


def test_hidden_width_diagnostic_preserves_all_available_solution_metrics():
    metric_names = (
        "raw_replacement_loss",
        "cosine_embedding_loss_hidden_states",
        "normalized_mse_loss_hidden_states",
        "mse_loss_hidden_states",
        "mae_loss_hidden_states",
        "kl_div",
        "lm_loss",
        "token_accuracy_top_1",
        "token_accuracy_top_1_consistency",
        "token_accuracy_top_5",
        "token_accuracy_top_5_consistency",
        "token_accuracy_top_10",
        "token_accuracy_top_10_consistency",
    )
    raw = {name: {"avg": index / 10} for index, name in enumerate(metric_names, 1)}

    metrics = _hidden_width_result_metrics(raw)

    assert list(metrics)[: len(metric_names)] == list(metric_names)
    assert all(metrics[name] == raw[name]["avg"] for name in metric_names)


def test_reused_parent_sweep_preserves_existing_sort_diagnosis_metrics():
    existing = {
        "passed": True,
        "teacher": {"lm_loss": 1.2},
        "sorted_teacher": {"lm_loss": 1.2001},
        "reverse_sorted": {"lm_loss": 1.5},
    }
    reuse = {
        "passed": True,
        "reused_parent_sweep": True,
        "equivalence": {"passed": True},
    }

    merged = _merge_reused_sort_equivalence(existing, reuse)

    assert merged["teacher"] == existing["teacher"]
    assert merged["sorted_teacher"] == existing["sorted_teacher"]
    assert merged["reverse_sorted"] == existing["reverse_sorted"]
    assert merged["reused_parent_sweep"] is True


def test_parent_sweep_sort_miss_is_blocking_but_width_miss_remains_advisory():
    verdict = _parent_sweep_sanity_verdict(
        {
            "passed": False,
            "findings": [{"stage": "width_sanity", "message": "ranking regressed"}],
        },
        {
            "passed": False,
            "findings": [{"stage": "width_sanity", "message": "teacher drifted"}],
        },
    )

    assert verdict.passed is False
    assert verdict.blocking is True
    assert verdict.findings == [
        {"stage": "width_sanity", "message": "ranking regressed"},
        {"stage": "sort_sanity", "message": "teacher drifted", "severity": "error"},
    ]
