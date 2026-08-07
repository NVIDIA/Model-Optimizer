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
from pathlib import Path

import pytest

import modelopt.torch.puzzletron.diagnostics.campaign_progress_report as report_module
from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    _activation_diagnostic_summary,
    _campaign_options_data,
    _library_scenario,
    _mamba_family_hint,
    _pipeline_state,
    _replacement_section,
    _stage_artifact_present,
    _subblock_axes,
    _varying_replacement_axes,
    _vllm_data,
    _vllm_section,
    generate_campaign_progress_report,
)
from modelopt.torch.puzzletron.stages.diagnostics import _PRIMARY_METRICS
from modelopt.torch.puzzletron.stages.graph import STAGE_REGISTRY


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_replacement_axis_discovery_collects_each_record_once():
    class CountingAxes(dict):
        iterations = 0

        def items(self):
            type(self).iterations += 1
            return super().items()

    records = [
        {"axes": CountingAxes(hidden_width=2688, num_experts=value, top_k=6)}
        for value in (128, 112, 96)
    ]

    assert _varying_replacement_axes(records) == ["num_experts"]
    assert CountingAxes.iterations == len(records)


def test_pipeline_state_uses_artifacts_and_completed_takes_precedence(tmp_path: Path):
    convert = STAGE_REGISTRY["convert"]
    bypass = STAGE_REGISTRY["bypass"]
    _write(tmp_path / "ckpts/teacher/config.json", {})
    _write(tmp_path / "artifacts/bypass/dp_observations.jsonl", {})

    assert _stage_artifact_present(tmp_path, convert)
    assert _pipeline_state(tmp_path, convert, {}) == "completed"
    assert _pipeline_state(tmp_path, bypass, {"bypass": {"enabled": False}}) == "completed"


def _disabled_bypass_report(tmp_path: Path) -> str:
    config = {
        "display_name": "Example Experiment",
        "bypass": {"enabled": False, "granularity": "subblock"},
        "vllm_stats": {"enabled": True, "granularity": "subblock"},
    }
    _write(tmp_path / "manifests/convert.json", {"status": "failed", "config": config})
    _write(tmp_path / "ckpts/teacher/config.json", {})
    _write(tmp_path / "artifacts/bypass/dp_observations.jsonl", {"observations": []})

    result = generate_campaign_progress_report(tmp_path, model_name="Example Experiment")
    return Path(result["html"]).read_text(encoding="utf-8")


def test_report_has_clean_header_and_completed_required_artifacts(tmp_path: Path):
    document = _disabled_bypass_report(tmp_path)

    assert "<h1>Example Experiment</h1>" in document
    assert '<p class="subtitle">Incremental Puzzletron campaign report</p>' in document
    assert "updated from content-addressed" not in document
    assert "Experiment summary" not in document
    assert "Campaign options" not in document
    assert "Granularity and artifact coverage" not in document
    assert "Merged experiment config" not in document
    assert ">Pipeline<" in document
    assert 'data-stage="convert" data-status="completed"' in document
    assert 'data-stage="bypass"' not in document
    assert 'class="dag-node required completed" data-stage="convert"' in document
    assert "<span class='required-node'>Required</span>" in document
    assert "<span class='optional-node'>Optional</span>" in document
    assert "dagre.min.js" in document
    assert 'data-source="convert" data-target="tokenize_data"' in document


def test_pipeline_dag_only_contains_configured_stages(tmp_path: Path):
    config = {
        "bypass": {"enabled": False},
        "vllm_stats": {"enabled": False},
        "zero_shot_evaluation": {"enabled": False},
        "aiperf": {"enabled": False},
    }

    document = report_module._stage_dag(tmp_path, config, {}, {})

    assert 'data-stage="convert"' in document
    assert 'data-stage="bypass"' not in document
    assert 'data-stage="vllm_stats"' not in document
    assert 'data-stage="zero_shot_evaluation"' not in document
    assert 'data-stage="aiperf"' not in document


def test_progress_report_renders_canonical_sort_sanity_metrics(tmp_path: Path):
    merged_config = {
        "data": {"modality": "text", "layout": "fixed"},
        "parallel": {"tp": 2, "cp": 2, "pp": 2, "dp": 2},
        "sort_sanity": {"enabled": True},
    }
    for stage in ("convert", "width_importance", "sort"):
        _write(
            tmp_path / f"manifests/{stage}.json",
            {"stage": stage, "status": "success", "config": merged_config},
        )
    _write(
        tmp_path / "manifests/sort_sanity.json",
        {
            "stage": "sort_sanity",
            "status": "success",
            "config": merged_config,
        },
    )
    _write(
        tmp_path / "artifacts/sort_sanity/summary.json",
        {
            "passed": True,
            "teacher": {
                "lm_loss": 1.25,
                "kl_div": 0.0,
                "token_accuracy_top_1": 0.42,
                "token_accuracy_top_1_consistency": 1.0,
            },
            "sorted_teacher": {
                "lm_loss": 1.2501,
                "kl_div": 0.0002,
                "token_accuracy_top_1": 0.41,
                "token_accuracy_top_1_consistency": 0.99,
            },
            "reverse_sorted": {
                "lm_loss": 1.5,
                "kl_div": 0.2,
                "token_accuracy_top_1": 0.2,
                "token_accuracy_top_1_consistency": 0.1,
            },
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")

    report_path = Path(result["html"])
    assert report_path == tmp_path / "artifacts/campaign_report/campaign_report.html"
    document = report_path.read_text(encoding="utf-8")
    assert "Merged experiment config" not in document
    assert "Sort Sanity Check" in document
    assert "token_accuracy_top_1" in document
    assert "token_accuracy_top_1_consistency" in document
    assert "Reverse sorted" in document
    assert "<tr><th>kl_div</th><td>N/A</td><td>0.0002</td><td>0.2</td></tr>" in document
    assert (
        "<tr><th>token_accuracy_top_1_consistency</th><td>N/A</td>"
        "<td>0.99</td><td>0.1</td></tr>" in document
    )
    assert "<tr><th>lm_loss</th><td>1.25</td><td>1.2501</td><td>1.5</td></tr>" in document
    assert "1.2501" in document
    assert 'data-stage="width_importance" data-status="completed"' not in document
    assert 'data-stage="sort_sanity" data-status="completed"' in document


def test_sort_sanity_failure_renders_blocking_failure_and_failed_dag_node(tmp_path: Path):
    message = "sorted teacher loss drift exceeded tolerance"
    _write(
        tmp_path / "manifests/sort_sanity.json",
        {
            "status": "failed",
            "config": {"sort_sanity": {"enabled": True}},
            "outputs": {"passed": False, "verdict": "failed", "blocking": True},
        },
    )
    _write(
        tmp_path / "artifacts/sort_sanity/summary.json",
        {
            "passed": False,
            "teacher": {"lm_loss": 1.0},
            "sorted_teacher": {"lm_loss": 1.2},
            "findings": [
                {
                    "stage": "sort_sanity",
                    "severity": "error",
                    "message": message,
                    "evidence": {"metric": "lm_loss"},
                }
            ],
        },
    )

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Equivalence gate: failed (blocking correctness)" in document
    assert "warning-value" in document
    assert f"data-warning='{message}'" in document
    assert 'data-stage="sort_sanity" data-status="failed"' in document


def test_progress_report_uses_pending_instead_of_transient_running_state(tmp_path: Path):
    _write(
        tmp_path / "manifests/convert.json",
        {"stage": "convert", "status": "success", "config": {}},
    )

    result = generate_campaign_progress_report(
        tmp_path,
        model_name="Qwen3.5-0.8B",
        running_stage="width_importance",
    )

    document = Path(result["html"]).read_text(encoding="utf-8")
    assert 'data-stage="width_importance" data-status="pending"' in document
    assert ">Pending<" in document
    assert "Running" not in document


def test_width_and_slicing_findings_render_on_affected_cells(tmp_path: Path):
    _write(
        tmp_path / "manifests/slicing_sanity.json",
        {
            "config": {
                "sort_sanity": {"enabled": True},
                "width_sanity": {"enabled": True},
                "slicing_sanity": {"enabled": True},
            }
        },
    )
    common = {
        "axis": "arbitrary_axis",
        "layer_idx": 3,
        "ratio": 0.5,
        "teacher_value": 16,
        "target_value": 8,
    }
    _write(
        tmp_path / "artifacts/width_sanity/summary.json",
        {
            "passed": False,
            "rows": [
                {**common, "method": "sorted", "raw_replacement_loss": 1.2},
                {**common, "method": "original", "raw_replacement_loss": 1.1},
                {**common, "method": "reverse", "raw_replacement_loss": 1.3},
            ],
            "findings": [
                {
                    "stage": "width_sanity",
                    "message": "sorted ranking is worse than original.",
                    "severity": "warning",
                    "evidence": {
                        "group": {
                            "axis": "arbitrary_axis",
                            "layer_idx": 3,
                            "target_value": 8,
                        },
                        "metric": "raw_replacement_loss",
                        "preferred_method": "sorted",
                        "comparison_method": "original",
                    },
                }
            ],
        },
    )
    message = "sorted and physical differ for raw_replacement_loss."
    _write(
        tmp_path / "artifacts/slicing_sanity/summary.json",
        {
            "passed": False,
            "rows": [
                {**common, "method": "sorted", "raw_replacement_loss": 1.2},
                {**common, "method": "physical", "raw_replacement_loss": 1.0},
            ],
            "findings": [
                {
                    "stage": "slicing_sanity",
                    "message": message,
                    "severity": "error",
                    "evidence": {
                        "group": {
                            "axis": "arbitrary_axis",
                            "layer_idx": 3,
                            "target_value": 8,
                        },
                        "metric": "raw_replacement_loss",
                        "left_method": "sorted",
                        "right_method": "physical",
                    },
                }
            ],
        },
    )

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Width ranking: quality warning" in document
    assert "Dynamic/physical equivalence: failed (blocking correctness)" in document
    assert "it does not mean the dynamic and physical implementations disagree" in document
    assert "Campaign qualification may still require the ranking warning to pass" in document
    assert 'data-stage="slicing_sanity" data-status="failed"' in document
    assert "class='warning-cell'" in document
    assert "class='warning-value'" in document
    assert "tabindex='0'" in document
    assert f"data-warning='{message}'" in document
    assert f"title='{message}'" not in document
    assert "<div class='finding-list'>" not in document
    assert "Slicing Sanity · axis=arbitrary_axis" not in document
    assert message in document
    assert "row.warning" in document
    assert "warningSymbol" in document


def test_report_promotes_legacy_descriptor_realization_failure(tmp_path: Path):
    _write(
        tmp_path / "artifacts/slicing_sanity/summary.json",
        {
            "rows": [],
            "findings": [],
            "axis_summaries": {
                "hidden_width": {
                    "cases": [
                        {
                            "hidden_width": 3840,
                            "primary_metric": "raw_replacement_loss",
                            "realization_delta": 6.6e-4,
                            "realization_passed": False,
                        }
                    ]
                }
            },
        },
    )

    summary = _activation_diagnostic_summary(tmp_path)

    assert len(summary["slicing_findings"]) == 1
    assert (
        summary["slicing_findings"][0]["evidence"]["kind"]
        == "descriptor_realization_gate"
    )


def test_library_uses_active_semantic_subblock_family(tmp_path: Path):
    library_path = tmp_path / "replacement_library.json"
    _write(
        library_path,
        {
            "hidden_width": 4096,
            "entries": [
                {
                    "parent_layer_indices": [0],
                    "child_block_configs": [
                        {
                            "subblock_configs": [
                                {"kind": "attention", "name": "attention", "no_op": True},
                                {"kind": "moe", "name": "moe", "no_op": False},
                            ]
                        }
                    ],
                },
                {
                    "parent_layer_indices": [1],
                    "child_block_configs": [
                        {
                            "subblock_configs": [
                                {
                                    "kind": "mamba",
                                    "name": "mamba",
                                    "no_op": False,
                                    "num_heads": 128,
                                    "head_dim": 64,
                                },
                                {"kind": "ffn", "name": "ffn", "no_op": True},
                            ]
                        }
                    ],
                },
            ],
        },
    )

    labels = {term["label"] for term in _library_scenario(library_path)["terms"]}

    assert labels == {"Mamba blocks", "MoE blocks"}


def test_mamba_and_gdn_use_distinct_semantic_axis_labels():
    common = {"kind": "mamba", "num_groups": 8, "num_heads": 128, "head_dim": 64}

    assert _subblock_axes({**common, "name": "mamba"}) == {
        "mamba_groups": 8,
        "mamba_heads": 128,
        "mamba_head_dim": 64,
    }
    assert _subblock_axes({**common, "name": "gdn"}) == {
        "gdn_key_groups": 8,
        "gdn_value_heads_per_group": 16,
        "gdn_value_head_dim": 64,
    }


def test_legacy_mamba_name_uses_declared_axis_namespace(tmp_path: Path):
    _write(
        tmp_path / "manifests/vllm_stats.json",
        {
            "config": {
                "search_space": {
                    "axes": {
                        "gdn_key_groups": {"enabled": True},
                        "gdn_value_head_dim": {"enabled": True},
                    }
                }
            }
        },
    )
    legacy = {
        "kind": "mamba",
        "name": "mamba",
        "num_groups": 8,
        "num_heads": 128,
        "head_dim": 64,
    }

    hint = _mamba_family_hint(tmp_path)

    assert hint == "gdn"
    assert _subblock_axes(legacy, mamba_family_hint=hint) == {
        "gdn_key_groups": 8,
        "gdn_value_heads_per_group": 16,
        "gdn_value_head_dim": 64,
    }


def test_vllm_report_rejects_negative_native_phase_measurements(tmp_path: Path):
    scenario = tmp_path / "scenarios/width-4096/depth-00"
    stats_path = scenario / "subblock_stats.json"
    _write(
        stats_path,
        [
            {
                "args": {
                    "runtime_stats": True,
                    "prefill_seq_len": 256,
                    "generation_seq_len": 32,
                    "batch_size": 1,
                },
                "subblocks": [
                    {
                        "subblock_config": {
                            "kind": "moe",
                            "name": "moe",
                            "no_op": False,
                            "num_experts": 128,
                        },
                        "runtime_ms": 0.001,
                        "prefill_runtime_ms": 0.04,
                        "decode_runtime_ms": -0.039,
                        "latency_difference_negative": True,
                    }
                ],
            }
        ],
    )
    library = {
        "scenarios": [
            {
                "hidden_width": 4096,
                "path": str(scenario / "replacement_library.json"),
                "unique_runtime_configs": 1,
            }
        ]
    }

    data = _vllm_data(tmp_path, library)

    assert data["scenarios"][0]["complete"] is False
    assert data["scenarios"][0]["invalid"] == 1
    assert data["warnings"][0]["kind"] == "negative_runtime_phase"
    assert data["records"][0]["warning"].startswith(
        "Native runtime measurement has a negative marginal phase"
    )
    assert "<article class='finding warning'>" not in _vllm_section(data)


def test_vllm_report_reads_canonical_root_stats_without_built_library(tmp_path: Path):
    runtime_args = {
        "runtime_stats": True,
        "runtime_backend": "vllm",
        "prefill_seq_len": 8192,
        "generation_seq_len": 1024,
        "max_num_seqs": 4,
    }
    _write(
        tmp_path / "subblock_stats.json",
        [
            {
                "args": {**runtime_args, "n_embd": width},
                "subblocks": [
                    {
                        "subblock_config": {
                            "kind": "moe",
                            "name": "moe",
                            "num_experts": experts,
                        },
                        "runtime_ms": runtime_ms,
                        "prefill_runtime_ms": runtime_ms - 1.0,
                        "decode_runtime_ms": 1.0,
                    }
                    for experts, runtime_ms in ((128, 10.0), (96, 8.0))
                ]
                + [
                    {
                        "subblock_config": {
                            "kind": "ffn",
                            "name": "ffn",
                            "no_op": True,
                        },
                        "runtime_ms": 0.0,
                        "prefill_runtime_ms": 0.0,
                        "decode_runtime_ms": 0.0,
                    }
                ],
            }
            for width in (2688, 2560)
        ],
    )

    data = _vllm_data(tmp_path, {"scenarios": [], "mamba_family_hint": "mamba"})

    assert data["widths"] == [2560, 2688]
    assert len(data["records"]) == 4
    assert [scenario["measured"] for scenario in data["scenarios"]] == [2, 2]
    assert "Measured configurations" in _vllm_section(data)


def test_vllm_sweep_explorer_uses_independent_axis_filters():
    document = _vllm_section(
        {
            "scenarios": [],
            "records": [{"hidden_width": 2688}],
            "metrics": ["runtime_ms"],
            "axes": ["moe_num_experts"],
            "axis_labels": {"moe_num_experts": "MoE experts"},
            "widths": [2688],
            "profiles": [{"id": "profile", "label": "profile"}],
        }
    )

    assert "id='vllm-axis-filters'" in document
    assert "vllm-family-select" not in document
    assert "vllm-config-select" not in document


def test_campaign_options_mark_parameter_selection_not_latency_verified(tmp_path: Path):
    _write(
        tmp_path / "manifests/campaign_options.json",
        {"optional_stages": {"vllm_stats": False}},
    )

    data = _campaign_options_data(tmp_path)

    assert data["selection_mode"] == "parameter_constrained"
    assert data["latency_verified"] is False


def test_sort_diagnosis_collects_accuracy_and_consistency_metrics():
    assert "token_accuracy_top_1" in _PRIMARY_METRICS
    assert "token_accuracy_top_1_consistency" in _PRIMARY_METRICS
    accuracy_metrics = [
        metric for metric in _PRIMARY_METRICS if metric.startswith("token_accuracy")
    ]
    assert accuracy_metrics == [
        "token_accuracy_top_1",
        "token_accuracy_top_1_consistency",
        "token_accuracy_top_5",
        "token_accuracy_top_5_consistency",
        "token_accuracy_top_10",
        "token_accuracy_top_10_consistency",
    ]


def test_progress_report_renders_axis_selectable_activation_diagnostic_tables(tmp_path: Path):
    _write(
        tmp_path / "manifests/width_sanity.json",
        {
            "config": {
                "sort_sanity": {"enabled": True},
                "width_sanity": {"enabled": True},
            }
        },
    )
    rows = []
    metric_values = {
        "token_accuracy_top_1": 0.40,
        "token_accuracy_top_1_consistency": 0.90,
        "token_accuracy_top_5": 0.60,
        "token_accuracy_top_5_consistency": 0.95,
        "token_accuracy_top_10": 0.70,
        "token_accuracy_top_10_consistency": 0.97,
        "lm_loss": 1.2,
        "kl_div": 0.03,
    }
    for method, delta in (
        ("sorted", 0.0),
        ("original", 0.1),
        ("reverse", 0.2),
        ("physical", 0.0),
    ):
        rows.append(
            {
                "axis": "ffn_intermediate",
                "layer_idx": 3,
                "ratio": 0.5,
                "teacher_value": 3584,
                "target_value": 1792,
                "method": method,
                **{metric: value + delta for metric, value in metric_values.items()},
            }
        )
    _write(
            tmp_path / "artifacts/width_sanity/summary.json",
        {"rows": rows, "primary_metric": "normalized_mse_loss_hidden_states"},
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'id="activation-axis-select"' in document
    assert 'id="activation-metric-select"' in document
    assert document.count('id="activation-diagnostic-table"') == 1
    assert "ffn_intermediate" in document
    assert "layer_3@50%" in document
    assert all(
        label in document
        for label in ("Sorted runtime", "Original runtime", "Reverse runtime", "Physical sorted")
    )
    ordered = [
        "token_accuracy_top_1",
        "token_accuracy_top_1_consistency",
        "token_accuracy_top_5",
        "token_accuracy_top_5_consistency",
        "token_accuracy_top_10",
        "token_accuracy_top_10_consistency",
    ]
    positions = [document.index(f"value='{metric}'") for metric in ordered]
    assert positions == sorted(positions)
    assert "unadjusted hidden-state mean squared error" in document


def test_progress_report_recovers_sort_table_from_compact_reuse_summary(tmp_path: Path):
    _write(
        tmp_path / "manifests/sort_sanity.json",
        {"config": {"sort_sanity": {"enabled": True}}},
    )
    _write(
        tmp_path / "artifacts/sort_sanity/summary.json",
        {
            "passed": True,
            "reused_parent_sweep": True,
            "equivalence": {"passed": True},
        },
    )
    result_dir = tmp_path / "diagnostics/sort_sanity"
    _write(
        result_dir / "single_sequence_replacement_solutions--validation/teacher.json",
        {
            "lm_loss": {"avg": 1.25},
            "kl_div": {"avg": 0.0},
        },
    )
    _write(
        result_dir / "single_sequence_replacement_solutions--validation/solution_0.json",
        {
            "i_solution": 0,
            "lm_loss": {"avg": 1.2501},
            "kl_div": {"avg": 0.0002},
        },
    )
    _write(
        result_dir / "reverse/single_sequence_replacement_solutions--validation/solution_0.json",
        {
            "lm_loss": {"avg": 1.5},
            "kl_div": {"avg": 0.2},
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Pending sort diagnosis" not in document
    assert "<tr><th>lm_loss</th><td>1.25</td><td>1.2501</td><td>1.5</td></tr>" in document
    assert "<th>i_solution</th>" not in document


def test_progress_report_renders_single_layer_selectable_bypass_overfit_plot(tmp_path: Path):
    _write(
        tmp_path / "manifests/bypass_sanity.json",
        {"config": {"bypass_sanity": {"enabled": True}}},
    )
    _write(
        tmp_path / "artifacts/bypass_sanity/summary.json",
        {"status": "complete"},
    )
    records = [
        {
            "step": step,
            "loss": 1.0 / step,
            "per_layer_loss": {"0": 2.0 / step, "3": 3.0 / step},
        }
        for step in range(1, 65)
    ]
    _write(
        tmp_path / "artifacts/bypass/overfit_probe/local_kd_loss_history.json",
        {
            "max_steps": 64,
            "loss_name": "normalized_mse_loss",
            "records": records,
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'data-stage="bypass_sanity" data-status="completed"' in document
    assert 'id="bypass-overfit-unit-select"' in document
    assert document.count("id='bypass-diverse-plot'") == 1
    assert document.count("id='bypass-fixed-plot'") == 1
    assert "normalized_mse_loss" in document
    assert '"step": 64' in document
    assert "name:'Mean'" in document


def test_progress_report_uses_subblock_losses_for_subblock_bypass_overfit(tmp_path: Path):
    _write(
        tmp_path / "manifests/bypass_sanity.json",
        {"config": {"bypass_sanity": {"enabled": True}}},
    )
    _write(
        tmp_path / "artifacts/bypass_sanity/summary.json",
        {"status": "complete"},
    )
    _write(
        tmp_path
        / "artifacts/bypass/overfit_probe/smallest_fixed/local_kd_loss_history.json",
        {
            "granularity": "subblock",
            "records": [
                {
                    "step": 1,
                    "loss": 0.2,
                    "per_layer_loss": {"0": 0.2},
                    "per_subblock_loss": {"0:attention:self_attn": 0.3},
                }
            ],
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert '<label class=\'selector-label\' for=\'bypass-overfit-unit-select\'>Subblock</label>' in document
    assert "layer_0:attention:self_attn" in document
    assert "per_subblock_loss" in document
    assert "per_layer_loss||" not in document.split("function renderLossChart", maxsplit=1)[1].split(
        "</script>", maxsplit=1
    )[0]


def test_nested_bypass_report_exposes_ema_and_display_only_outlier_controls(tmp_path: Path):
    document = Path(generate_campaign_progress_report(tmp_path)["html"]).read_text(encoding="utf-8")
    section = report_module._nested_bypass_section(
        {
            "records": [{"step": 1}],
            "units": [{"layer_idx": 0, "subblock_kind": "attention", "subblock_name": "self_attn"}],
            "observations": [{"step": 1, "layer_idx": 0, "loss": 1.0}],
            "granularity": "subblock",
        }
    )

    assert 'id="nested-bypass-ema-alpha"' in section
    assert 'id="nested-bypass-exclude-outliers"' in section
    assert "emaByStep" in document
    assert "tukeyInliers" in document


def test_progress_report_renders_profile_evaluation_and_aiperf_explorers(tmp_path: Path):
    _write(
        tmp_path / "manifests/aiperf.json",
        {
            "config": {
                "zero_shot_evaluation": {"enabled": True},
                "aiperf": {"enabled": True},
            }
        },
    )
    registry = {
        "profile_id": "params-080",
        "solutions": [
            {
                "solution_id": "teacher",
                "label": "Teacher",
                "color": "#f5c451",
                "marker": "star",
                "always_enabled": True,
            },
            {
                "solution_id": "h0512-d0",
                "label": "H=512, Drop=0",
                "color": "#ff6577",
                "marker": "circle",
                "always_enabled": False,
            },
        ],
    }
    _write(tmp_path / "mip/profiles/params-080/selected_solutions.json", registry)
    _write(
        tmp_path
        / "artifacts/zero_shot_evaluation/profiles/params-080/text-s8-l32/evaluation_summary.json",
        {
            "profile_id": "params-080",
            "eval_samples": 8,
            "block_size": 32,
            "solutions": [
                {
                    **style,
                    "parameter_count": 100 if style["solution_id"] == "teacher" else 60,
                    "total_costs": {"stats.runtime_ms": 2.0},
                    "metrics": {"lm_loss": 1.0, "kl_div": 0.1},
                }
                for style in registry["solutions"]
            ],
        },
    )
    aiperf_rows = []
    for style in registry["solutions"]:
        for concurrency in (1, 4):
            aiperf_rows.append(
                {
                    "solution_id": style["solution_id"],
                    "profile_id": "params-080",
                    "topology_id": "tp2-pp1-dp1-ep1-pcp1-dcp1",
                    "concurrency": concurrency,
                    "metrics": {
                        "output_token_throughput": 10.0 * concurrency,
                        "ttft_mean_ms": 4.0,
                        "tpot_mean_ms": 2.0,
                        "request_latency_mean_ms": 20.0,
                        "output_token_throughput_per_user_mean": 5.0,
                    },
                }
            )
    _write(
        tmp_path / "artifacts/aiperf/profiles/params-080/isl-1024-osl-128/aiperf_results.json",
        {
            "profile_id": "params-080",
            "workload": {"input_tokens": 1024, "output_tokens": 128},
            "results": aiperf_rows,
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "id='evaluation-x-select'" in document
    assert "id='evaluation-y-select'" in document
    assert "id='evaluation-scatter-plot'" in document
    assert "id='aiperf-workload-select'" in document
    assert "id='aiperf-topology-select'" in document
    assert '<option value="PARETO">PARETO</option>' in document
    assert 'data-aiperf-solution="teacher"' in document
    assert "#f5c451" in document


def test_progress_report_renders_partial_aiperf_leaf_results(tmp_path: Path):
    _write(
        tmp_path / "manifests/aiperf.json",
        {"config": {"aiperf": {"enabled": True}}},
    )
    registry = {
        "profile_id": "params-080",
        "solutions": [
            {
                "solution_id": "teacher",
                "label": "Teacher",
                "color": "#f5c451",
                "marker": "star",
                "always_enabled": True,
            },
            {
                "solution_id": "h0512-d0",
                "label": "H=512, Drop=0",
                "color": "#ff6577",
                "marker": "circle",
                "always_enabled": False,
            },
        ],
    }
    _write(tmp_path / "mip/profiles/params-080/selected_solutions.json", registry)
    for style in registry["solutions"]:
        for concurrency in (1, 4):
            _write(
                tmp_path
                / "artifacts/aiperf/profiles/params-080/isl-1024-osl-128"
                / style["solution_id"]
                / "tp2-pp1-dp1-ep1-pcp1-dcp1"
                / f"concurrency_{concurrency}/puzzletron_aiperf_result.json",
                {
                    "solution_id": style["solution_id"],
                    "profile_id": "params-080",
                    "topology_id": "tp2-pp1-dp1-ep1-pcp1-dcp1",
                    "concurrency": concurrency,
                    "failures": 0,
                    "workload": {"input_tokens": 1024, "output_tokens": 128},
                    "metrics": {
                        "output_token_throughput": 10.0 * concurrency,
                        "ttft_mean_ms": 4.0,
                        "tpot_mean_ms": 2.0,
                        "request_latency_mean_ms": 20.0,
                        "output_token_throughput_per_user_mean": 5.0,
                    },
                },
            )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'id="aiperf-benchmarks"' in document
    assert "Partial AIPerf coverage: 4 valid measurements" in document
    assert 'data-aiperf-solution="teacher"' in document
    assert 'data-stage="aiperf" data-status="pending"' in document
    assert "id='aiperf-ttft-throughput-plot'" in document
    assert "id='aiperf-latency-throughput-plot'" in document
    assert "id='aiperf-interactivity-throughput-plot'" in document
    assert "id='aiperf-tpot-throughput-plot'" in document


def test_evaluation_report_adds_teacher_styles_solution_kinds_and_pareto_front(tmp_path: Path):
    _write(
        tmp_path / "manifests/zero_shot_evaluation.json",
        {"config": {"zero_shot_evaluation": {"enabled": True}}},
    )
    _write(
        tmp_path
        / "artifacts/zero_shot_evaluation/profiles/params-075/text-s128-l8192/evaluation_summary.json",
        {
            "profile_id": "params-075",
            "teacher": {
                "solution_id": "teacher",
                "label": "Teacher",
                "parameter_count": 100,
                "metrics": {"lm_loss": 1.0},
            },
            "solutions": [
                {
                    "solution_id": "mixed-h2560-d1",
                    "parameter_count": 75,
                    "metrics": {"lm_loss": 1.1},
                },
                {
                    "solution_id": "homogeneous-h2432-d2",
                    "homogeneous_assignment": {"attention.q_per_group": 8},
                    "parameter_count": 73,
                    "metrics": {"lm_loss": 1.2},
                },
            ],
        },
    )

    data = report_module._evaluation_data(tmp_path)
    rows = data["profiles"][0]["rows"]
    document = Path(generate_campaign_progress_report(tmp_path)["html"]).read_text(encoding="utf-8")

    assert [row["solution_id"] for row in rows] == [
        "teacher",
        "mixed-h2560-d1",
        "homogeneous-h2432-d2",
    ]
    assert [row["kind"] for row in rows] == ["teacher", "heterogeneous", "homogeneous"]
    assert [row["marker"] for row in rows] == ["star", "circle", "diamond"]
    assert len({row["color"] for row in rows}) == 3
    assert "id='evaluation-best-across-profiles'" in document
    assert "bestRowsAcrossProfiles" in document
    assert "bestAcrossProfiles.checked" in document
    assert "paretoFront" in document
    assert "Pareto frontier" in document


def test_report_separates_plot_titles_from_horizontal_legends(tmp_path: Path):
    document = Path(generate_campaign_progress_report(tmp_path)["html"]).read_text(
        encoding="utf-8"
    )

    assert (
        "legend:{orientation:'h',x:0,xanchor:'left',y:1.18,yanchor:'bottom'}" in document
    )
    assert "margin:{l:62,r:18,t:96,b:55}" in document
    assert "function chartTitle(text,size=15)" in document
    assert "title:chartTitle(" in document


def test_progress_report_renders_proper_distillation_terms_and_before_after_eval(
    tmp_path: Path,
):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _proper_distillation_data,
    )

    _write(
        tmp_path / "manifests/post_distillation_evaluation.json",
        {
            "config": {
                "global_distillation": {"enabled": True},
                "post_distillation_evaluation": {"enabled": True},
            }
        },
    )
    run = (
        tmp_path
        / "artifacts/distillation/profiles/params-080"
        / "text-l2048-g32-b4-s2500-seed445/h1024-d0"
    )
    _write(
        run / "distillation_summary.json",
        {
            "profile_id": "params-080",
            "solution_id": "h1024-d0",
            "label": "H=1024, Drop=0",
            "max_steps": 2,
            "sequence_length": 2048,
            "global_batch_size": 32,
            "records": [
                {
                    "step": 0,
                    "loss": 9.0,
                    "ce_loss": 2.0,
                    "kd_loss": 3.0,
                    "main_ce": 2.0,
                    "main_kd": 3.0,
                    "mtp_ce": 1.5,
                    "mtp_kd": 2.5,
                },
                {
                    "step": 1,
                    "loss": 4.0,
                    "ce_loss": 1.0,
                    "kd_loss": 1.0,
                    "main_ce": 1.0,
                    "main_kd": 1.0,
                    "mtp_ce": 1.0,
                    "mtp_kd": 1.0,
                },
            ],
        },
    )
    _write(
        tmp_path
        / "artifacts/zero_shot_evaluation/profiles/params-080/text-s1024-l2048"
        / "evaluation_summary.json",
        {
            "solutions": [
                {
                    "solution_id": "h1024-d0",
                    "metrics": {"lm_loss": 3.5, "kl_div": 2.0},
                }
            ]
        },
    )
    _write(
        run / "evaluation/result.json",
        {
            "solution_id": "h1024-d0-post-kd",
            "metrics": {"lm_loss": 2.5, "kl_div": 1.0},
        },
    )
    _write(
        tmp_path
        / "artifacts/post_distillation_evaluation/profiles/params-080/text-s1024-l2048"
        / "evaluation_summary.json",
        {
            "solutions": [
                {
                    "solution_id": "h1024-d0-post-kd",
                    "metrics": {"lm_loss": 2.5, "kl_div": 1.0},
                }
            ]
        },
    )

    loaded = _proper_distillation_data(tmp_path)
    assert len(loaded["runs"]) == 1, loaded
    assert loaded["runs"][0]["metrics"] == [
        "loss",
        "main_ce",
        "mtp_ce",
        "main_kd",
        "mtp_kd",
    ]

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text()

    assert 'id="global-distillation"' in document
    assert 'id="proper-distillation-plots"' in document
    assert "main_ce" in document and "mtp_kd" in document
    global_section = document.split('id="global-distillation"', maxsplit=1)[1].split(
        "</details>", maxsplit=1
    )[0]
    post_section = document.split(
        'id="post-distillation-evaluation"', maxsplit=1
    )[1].split("</details>", maxsplit=1)[0]
    assert 'id="proper-distillation-plots"' in global_section
    assert "Before KD" not in global_section and "After KD" not in global_section
    assert 'id="proper-distillation-plots"' not in post_section
    assert "Before KD" in post_section and "After KD" in post_section
    assert "3.5" in document and "2.5" in document


def test_proper_distillation_data_renders_canonical_partial_training_log(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _proper_distillation_data,
    )

    run = (
        tmp_path
        / "artifacts/global_distillation/profiles/latency-095"
        / "text-n4096-l16384-s256-b16-seed444/h4096-d4"
    )
    _write(
        run / "global_kd_recipe.yaml",
        {
            "step_scheduler": {
                "global_batch_size": 16,
                "local_batch_size": 2,
                "max_steps": 257,
            },
            "dataset": {"num_samples": 4096, "seq_length": 16384},
        },
    )
    training_log = run / "checkpoints/training.jsonl"
    training_log.parent.mkdir(parents=True)
    training_log.write_text(
        json.dumps({"step": 1, "loss": 1.5, "main_kd": 0.7})
        + "\n"
        + json.dumps({"step": 2, "loss": 1.1, "main_kd": 0.4})
        + "\n",
        encoding="utf-8",
    )

    loaded = _proper_distillation_data(tmp_path)

    assert len(loaded["runs"]) == 1
    assert loaded["runs"][0]["profile_id"] == "latency-095"
    assert loaded["runs"][0]["solution_id"] == "h4096-d4"
    assert loaded["runs"][0]["partial"] is True
    assert loaded["runs"][0]["max_steps"] == 256
    assert loaded["runs"][0]["records"][-1]["loss"] == 1.1


def test_distillation_overfit_data_renders_partial_training_log(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _distillation_overfit_data,
    )

    solution = (
        tmp_path
        / "artifacts/global_distillation_sanity/profiles/latency-095"
        / "text-n8-l16384-s128-b8-seed444/h4096-d4"
    )
    _write(
        solution / "global_kd_recipe.yaml",
        {
            "step_scheduler": {
                "global_batch_size": 8,
                "local_batch_size": 8,
                "max_steps": 129,
            },
            "dataset": {"num_samples": 8, "seq_length": 16384},
        },
    )
    training_log = solution / "checkpoints/training.jsonl"
    training_log.parent.mkdir(parents=True)
    training_log.write_text(
        json.dumps({"step": 1, "loss": 4.0, "mtp_kd": 1.5})
        + "\n"
        + json.dumps({"step": 2, "loss": 2.0, "mtp_kd": 0.7})
        + "\n",
        encoding="utf-8",
    )

    loaded = _distillation_overfit_data(tmp_path)

    assert len(loaded["profiles"]) == 1
    profile = loaded["profiles"][0]
    assert profile["workload_id"] == "text-n8-l16384-s128-b8-seed444"
    assert profile["partial"] is True
    assert profile["solutions"][0]["solution_id"] == "h4096-d4"
    assert profile["solutions"][0]["records"][-1]["loss"] == 2.0


def test_progress_report_labels_latency_constrained_mip_profiles(tmp_path: Path):
    _write(
        tmp_path / "mip/profiles/latency-075/mip_grid.json",
        {
            "profile": {
                "id": "latency-075",
                "label": "75% latency",
                "constraint_type": "latency_ratio",
                "latency_ratio": 0.75,
                "latency_limit_ms": 12.5,
            },
            "expected_scenario_count": 1,
            "teacher": {"hidden_width": 4096, "total_costs": {"stats.runtime_ms": 16.0}},
            "scenarios": [
                {
                    "status": "feasible",
                    "hidden_width": 3584,
                    "removed_sublayers": 0,
                    "score": 0.25,
                    "solver_objective_sum": 1.5,
                    "total_costs": {"stats.runtime_ms": 12.0},
                }
            ],
        },
    )

    document = Path(
        generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-9B")["html"]
    ).read_text()

    assert "Latency limit" in document
    assert "12.5 ms" in document
    assert "parameter percentage is resolved" not in document
    assert "Solver Objective Sum" in document
    assert ">Score</button>" not in document


def test_mip_report_separates_homogeneous_solutions_and_simplifies_costs(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _mip_data

    _write(
        tmp_path / "mip/profiles/params-075/mip_grid.json",
        {
            "profile": {
                "id": "params-075",
                "label": "75% params",
                "constraint_type": "named_profile",
                "constraints": {"stats.num_params": [73.0, 75.0]},
            },
            "expected_scenario_count": 2,
            "teacher": {
                "hidden_width": 4096,
                "status": "teacher",
                "sliced_teacher_baseline": 0.0,
                "total_costs": {
                    "stats.num_params": 100.0,
                    "stats.memory_mib": 200.0,
                    "stats.has_attention": 8.0,
                    "stats.has_moe": 0.0,
                    "stats.num_experts": 0.0,
                    "stats.top_k": 0.0,
                    "stats.not_no_op": 32.0,
                },
            },
            "scenarios": [
                {
                    "status": "feasible",
                    "hidden_width": 3584,
                    "removed_sublayers": 1,
                    "sliced_teacher_baseline": 0.0,
                    "solver_objective_sum": 1.5,
                    "chosen_replacement_count": 31,
                    "total_costs": {
                        "stats.num_params": 75.0,
                        "stats.memory_mib": 150.0,
                        "stats.has_attention": 7.0,
                        "stats.has_moe": 0.0,
                        "stats.num_experts": 0.0,
                        "stats.top_k": 0.0,
                        "stats.not_no_op": 31.0,
                    },
                    "homogeneous_solutions": [
                        {
                            "rank": 0,
                            "score": 2.0,
                            "solver_objective_sum": 2.0,
                            "homogeneous_assignment": {
                                "attention.num_kv_heads": 2,
                                "attention.q_per_group": 8,
                                "moe.num_experts": 96,
                            },
                            "total_costs": {
                                "stats.num_params": 70.0,
                                "stats.memory_mib": 140.0,
                            },
                        }
                    ],
                },
                {
                    "status": "infeasible",
                    "hidden_width": 3072,
                    "removed_sublayers": 2,
                    "reason": "lower resource bound cannot be met",
                },
            ],
        },
    )

    data = _mip_data(tmp_path)
    document = Path(generate_campaign_progress_report(tmp_path)["html"]).read_text()

    assert "sliced_teacher_baseline" not in data["columns"]
    assert "parameter_ratio" not in data["columns"]
    assert "has_attention" not in data["columns"]
    assert "not_no_op" not in data["columns"]
    assert "status" not in data["profiles"][0]["columns"]
    assert "chosen_replacement_count" not in data["profiles"][0]["columns"]
    assert "num_experts" not in data["profiles"][0]["columns"]
    assert "top_k" not in data["profiles"][0]["columns"]
    assert len(data["profiles"][0]["rows"]) == 2
    assert data["profiles"][0]["infeasible_rows"] == [
        {
            "label": "H=3072, Drop=2",
            "hidden_width": 3072,
            "removed_sublayers": 2,
            "reason": "lower resource bound cannot be met",
        }
    ]
    assert data["profiles"][0]["homogeneous_rows"][0]["assignment"] == {
        "attention.num_kv_heads": 2,
        "attention.q_per_group": 8,
        "moe.num_experts": 96,
    }
    assert "Homogeneous solutions" in document
    assert "id='mip-homogeneous-table'" in document
    assert "id='mip-homogeneous-empty'" in document
    assert "(% of teacher)" in document
    assert ">Sliced Teacher Baseline</button>" not in document
    assert ">Has Attention</button>" not in document
    assert ">Not No Op</button>" not in document
    assert "Chosen Replacement Count" not in document
    assert "lower resource bound cannot be met" in document
    assert "Resource band" in document
    assert "stats.num_params: 73–75" in document
    assert (
        "homogeneousHead.innerHTML=`<tr><th>Solution</th><th>Hidden Width</th>"
        "<th>Removed Sublayers</th>"
        not in document
    )


def test_replacement_data_omits_empty_width_scenarios(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _replacement_data

    (tmp_path / "scenarios/width-4096/depth-00").mkdir(parents=True)

    data = _replacement_data(tmp_path)

    assert data["scenarios"] == []


def test_replacement_data_reads_width_local_subblock_scores(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _replacement_data

    scenario = tmp_path / "scenarios/width-3840/depth-00"
    _write(scenario / "scenario_manifest.json", {"hidden_width": 3840})
    _write(scenario / "single_subblock_replacement_solutions.json", [{}])
    _write(
        scenario / "single_subblock_replacement_solutions--validation/solution_0.json",
        {
            "i_solution": 0,
            "args": {"eval_samples": 2},
            "lm_loss": {"avg": 1.25},
            "puzzle_solution": {
                "single_sequence_replacement": {
                    "parent_layer_indices": [3],
                    "child_block_configs": [
                        {
                            "subblock_configs": [
                                {
                                    "kind": "moe",
                                    "name": "moe",
                                    "num_experts": 128,
                                    "no_op": False,
                                }
                            ]
                        }
                    ],
                },
                "subblock_replacement": {"kind": "moe", "name": "moe"},
            },
        },
    )

    data = _replacement_data(tmp_path)

    assert data["granularity"] == "subblock"
    assert data["scenarios"] == [
        {
            "hidden_width": 3840,
            "path": str(scenario / "single_subblock_replacement_solutions--validation"),
            "expected": 1,
            "measured": 1,
            "complete": True,
            "granularity": "subblock",
        }
    ]
    assert data["records"][0]["layer_idx"] == 3
    assert data["records"][0]["metrics"]["lm_loss"] == 1.25


def test_replacement_explorer_uses_independent_axis_filters():
    document = _replacement_section(
        {
            "records": [{"hidden_width": 2688}],
            "scenarios": [],
            "metrics": ["lm_loss"],
            "axes": ["moe_num_experts"],
            "axis_labels": {"moe_num_experts": "MoE experts"},
            "widths": [2688],
            "layers": [0],
            "eval_samples": [128],
        }
    )

    assert "id='replacement-axis-filters'" in document
    assert "replacement-family-select" not in document
    assert "replacement-config-select" not in document


def test_report_shows_partial_replacement_scores_while_stage_is_pending(tmp_path: Path):
    scenario = tmp_path / "scenarios/width-2688/depth-00"
    _write(scenario / "scenario_manifest.json", {"hidden_width": 2688})
    _write(scenario / "single_subblock_replacement_solutions.json", [{}, {}])
    _write(
        scenario / "single_subblock_replacement_solutions--validation/solution_0.json",
        {
            "i_solution": 0,
            "args": {"eval_samples": 2},
            "lm_loss": {"avg": 1.25},
            "puzzle_solution": {
                "single_sequence_replacement": {
                    "parent_layer_indices": [3],
                    "child_block_configs": [
                        {
                            "subblock_configs": [
                                {
                                    "kind": "moe",
                                    "name": "moe",
                                    "num_experts": 128,
                                    "no_op": False,
                                }
                            ]
                        }
                    ],
                },
                "subblock_replacement": {"kind": "moe", "name": "moe"},
            },
        },
    )

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'data-stage="replacement_scoring" data-status="pending"' in document
    assert "Replacement-score explorer" in document
    assert "replacement-family-select" not in document
    assert "id='replacement-axis-filters'" in document


def test_report_explorers_use_per_axis_filter_javascript(tmp_path: Path):
    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "populateVllmAxisFilters" in document
    assert "populateReplacementAxisFilters" in document
    assert "axisName!==axis.value" in document
    assert "let compatible=[]" not in document
    assert ".vllm-controls>span{display:contents}" in document
    assert "legendgroup:kind" in document
    assert "showlegend:true" in document
    assert "delete charts.theme.legend" not in document


def test_diverse_bypass_gate_uses_decreasing_per_width_trends_and_diversity():
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _overfit_summary_card

    card = _overfit_summary_card(
        "diverse_resampled",
        {
            "records": [{"step": 1}],
            "summary": {
                "diversity_passed": True,
                "loss_trend": {
                    "decreased": True,
                    "hard_gate_passed": None,
                    "per_hidden_width": {"4096": {"decreased": True}},
                },
            },
        },
    )

    assert "probe-summary passed" in card
    assert "Trend + diversity gate</dt><dd>passed" in card


def test_bypass_gate_findings_render_warning_tooltip():
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _overfit_summary_card

    message = "local KD acceptance loss did not decrease"
    card = _overfit_summary_card(
        "smallest_fixed",
        {
            "records": [{"step": 1}],
            "summary": {
                "passed": False,
                "findings": [{"message": message, "severity": "warning"}],
                "loss_trend": {"sufficient_evidence": True, "hard_gate_passed": False},
            },
        },
    )

    assert "probe-summary failed warning-value" in card
    assert f"data-warning='{message}'" in card


def test_distillation_sanity_findings_render_warning_tooltip():
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _distillation_overfit_section,
    )

    message = "ending loss did not improve"
    section = _distillation_overfit_section(
        {
            "profiles": [
                {
                    "sample_count": 8,
                    "findings": [{"message": message, "severity": "warning"}],
                    "solutions": [],
                }
            ]
        }
    )

    assert "Trend verdict: warning" in section
    assert "warning-value" in section
    assert f"data-warning='{message}'" in section


def test_report_cache_reuses_sections_and_targets_changed_inputs(tmp_path: Path, monkeypatch):
    _write(tmp_path / "artifacts/replacement_scoring/summary.json", {"value": 1})
    _write(tmp_path / "artifacts/mip/summary.json", {"value": 1})
    calls = {"replacement": 0, "mip": 0}

    def replacement_data(_root: Path):
        calls["replacement"] += 1
        return {
            "records": [],
            "scenarios": [],
            "metrics": [],
            "axes": [],
            "axis_labels": {},
            "widths": [],
            "layers": [],
            "eval_samples": [],
        }

    def mip_data(_root: Path):
        calls["mip"] += 1
        return {"profiles": [], "columns": []}

    monkeypatch.setattr(report_module, "_replacement_data", replacement_data)
    monkeypatch.setattr(report_module, "_mip_data", mip_data)

    first = generate_campaign_progress_report(tmp_path)
    second = generate_campaign_progress_report(tmp_path)

    assert int(first["cache_misses"]) > 0
    assert second["cache_misses"] == "0"
    assert second["cache_hits"] == first["cache_misses"]
    assert calls == {"replacement": 1, "mip": 1}

    _write(tmp_path / "artifacts/mip/summary.json", {"value": 200})
    changed = generate_campaign_progress_report(tmp_path)

    assert int(changed["cache_misses"]) >= 1
    assert calls["replacement"] == 1
    assert calls["mip"] == 2

    generate_campaign_progress_report(tmp_path, use_cache=False)
    assert calls["replacement"] == 2
    assert calls["mip"] == 3

    generate_campaign_progress_report(tmp_path, rebuild_sections=("replacement",))
    assert calls["replacement"] == 3
    assert calls["mip"] == 4


def test_report_cli_maps_cache_controls(tmp_path: Path):
    from examples.puzzletron.generate_campaign_progress_report import build_parser

    args = build_parser().parse_args(
        [
            "--puzzle-dir",
            str(tmp_path),
            "--model-name",
            "Example",
            "--no-cache",
            "--rebuild-section",
            "replacement",
            "--rebuild-section",
            "mip",
        ]
    )

    assert args.puzzle_dir == tmp_path
    assert args.model_name == "Example"
    assert args.no_cache is True
    assert args.rebuild_section == ["replacement", "mip"]


def test_report_verification_failure_preserves_previous_html_and_manifest(
    tmp_path: Path, monkeypatch
):
    first = generate_campaign_progress_report(tmp_path)
    html_path = Path(first["html"])
    manifest_path = Path(first["manifest"])
    old_html = html_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    _write(tmp_path / "artifacts/mip/summary.json", {"changed": True})

    def reject(_path: Path) -> None:
        raise RuntimeError("semantic verification failed")

    monkeypatch.setattr(report_module, "_verify_report_candidate", reject)
    with pytest.raises(RuntimeError, match="semantic verification failed"):
        generate_campaign_progress_report(tmp_path)

    assert html_path.read_bytes() == old_html
    assert manifest_path.read_bytes() == old_manifest
