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

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    _campaign_options_data,
    _pipeline_state,
    _stage_artifact_present,
    generate_campaign_progress_report,
)
from modelopt.torch.puzzletron.stages.graph import STAGE_REGISTRY
from modelopt.torch.puzzletron.stages.diagnostics import _PRIMARY_METRICS


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pipeline_state_uses_artifacts_and_completed_takes_precedence(tmp_path: Path):
    convert = STAGE_REGISTRY["convert"]
    bypass = STAGE_REGISTRY["bypass"]
    _write(tmp_path / "ckpts/teacher/config.json", {})
    _write(tmp_path / "artifacts/bypass/dp_observations.jsonl", {})

    assert _stage_artifact_present(tmp_path, convert)
    assert _pipeline_state(tmp_path, convert, {}) == "completed"
    assert _pipeline_state(tmp_path, bypass, {"bypass": {"enabled": False}}) == "completed"


def test_report_has_clean_header_and_only_artifact_backed_sections(tmp_path: Path):
    config = {
        "display_name": "Example Experiment",
        "bypass": {"enabled": False, "granularity": "subblock"},
        "vllm_stats": {"enabled": True, "granularity": "subblock"},
    }
    _write(tmp_path / "manifests/convert.json", {"status": "failed", "config": config})
    _write(tmp_path / "ckpts/teacher/config.json", {})
    _write(tmp_path / "artifacts/bypass/dp_observations.jsonl", {"observations": []})

    result = generate_campaign_progress_report(tmp_path, model_name="Example Experiment")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "<h1>Example Experiment</h1>" in document
    assert '<p class="subtitle">Incremental Puzzletron campaign report</p>' in document
    assert "updated from content-addressed" not in document
    assert "Experiment summary" not in document
    assert "Campaign options" not in document
    assert "Granularity and artifact coverage" not in document
    assert "Merged experiment config" not in document
    assert ">Pipeline<" in document
    assert 'data-stage="convert" data-status="completed"' in document
    assert 'data-stage="bypass" data-status="completed"' in document
    assert 'class="dag-node required completed" data-stage="convert"' in document
    assert 'class="dag-node optional completed" data-stage="bypass"' in document
    assert "<span class='required-node'>Required</span>" in document
    assert "<span class='optional-node'>Optional</span>" in document
    assert "dagre.min.js" in document
    assert 'data-source="convert" data-target="tokenize_data"' in document
    assert "<tspan x='26' dy='0'>Global Distillation</tspan>" in document
    assert "<tspan x='26' dy='15'>Sanity Check</tspan>" in document
    assert "Subblock Bypass" in document
    assert "Nested bypass" not in document
    assert "MIP solutions" not in document


def test_progress_report_renders_canonical_sort_sanity_metrics(tmp_path: Path):
    merged_config = {
        "data": {"modality": "text", "layout": "fixed"},
        "parallel": {"tp": 2, "cp": 2, "pp": 2, "dp": 2},
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
            "rows": [
                {**common, "method": "sorted", "raw_replacement_loss": 1.2},
                {**common, "method": "original", "raw_replacement_loss": 1.1},
                {**common, "method": "reverse", "raw_replacement_loss": 1.3},
            ],
            "findings": [],
        },
    )
    message = "sorted and physical differ for raw_replacement_loss."
    _write(
        tmp_path / "artifacts/slicing_sanity/summary.json",
        {
            "rows": [
                {**common, "method": "sorted", "raw_replacement_loss": 1.2},
                {**common, "method": "physical", "raw_replacement_loss": 1.0},
            ],
            "findings": [
                {
                    "stage": "slicing_sanity",
                    "message": message,
                    "severity": "warning",
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

    assert "class='warning-cell'" in document
    assert "class='warning-value'" in document
    assert "tabindex='0'" in document
    assert f"data-warning='{message}'" in document
    assert f"title='{message}'" not in document
    assert "<div class='finding-list'>" not in document


def test_campaign_options_mark_parameter_selection_not_latency_verified(tmp_path: Path):
    _write(
        tmp_path / "manifests/campaign_options.json",
        {"optional_stages": {"vllm_stats": False}},
    )

    data = _campaign_options_data(tmp_path)

    assert data["selection_mode"] == "parameter_constrained"
    assert data["latency_verified"] is False


def test_progress_report_marks_disabled_vllm_stage_as_disabled(tmp_path: Path):
    _write(tmp_path / "manifests/convert.json", {"config": {"vllm_stats": {"enabled": False}}})

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'data-stage="vllm_stats" data-status="disabled"' in document
    assert "Parameter-constrained" not in document


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
    for method, delta in (("activation", 0.0), ("random", 0.1), ("reverse", 0.2)):
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
        tmp_path / "artifacts/activation_diagnostic/activation_diagnostic_summary.json",
        {"rows": rows, "primary_metric": "normalized_mse_loss_hidden_states"},
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'id="activation-axis-select"' in document
    assert 'id="activation-metric-select"' in document
    assert document.count('id="activation-diagnostic-table"') == 1
    assert "ffn_intermediate" in document
    assert "layer_3@50%" in document
    assert "Sorted" in document and "Random (teacher order)" in document and "Reverse" in document
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


def test_progress_report_recovers_failed_hidden_width_diagnostic_without_parent_summary(
    tmp_path: Path,
):
    rows = [
        {
            "role": role,
            "hidden_width": 3584,
            "metrics": {"raw_replacement_loss": loss},
        }
        for role, loss in (
            ("random", 5.1187304),
            ("activation", 5.1188848),
            ("reverse", 5.1189997),
        )
    ]
    _write(
        tmp_path / "artifacts/activation_diagnostic/hidden_width_diagnostic_summary.json",
        {
            "teacher_hidden_width": 4096,
            "hidden_width": 3584,
            "primary_metric": "raw_replacement_loss",
            "rows": rows,
            "passed": False,
            "beats_random": False,
            "beats_reverse": True,
            "require_beats_random": True,
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Pending activation/ranking diagnosis" not in document
    assert "Embedding-width ranking: failed" in document
    assert "activation did not beat random" in document
    assert "hidden_width" in document
    assert "global@88%" in document


def test_progress_report_recovers_sort_table_from_compact_reuse_summary(tmp_path: Path):
    _write(
        tmp_path / "artifacts/sort_equivalence/sort_equivalence_summary.json",
        {
            "passed": True,
            "reused_parent_sweep": True,
            "equivalence": {"passed": True},
        },
    )
    result_dir = tmp_path / "diagnostics/sort_equivalence"
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
        tmp_path / "manifests/bypass_overfit.json",
        {"stage": "bypass_overfit", "status": "success", "config": {}},
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

    assert 'data-stage="bypass_overfit" data-status="complete"' in document
    assert 'id="bypass-overfit-unit-select"' in document
    assert document.count("id='bypass-diverse-plot'") == 1
    assert document.count("id='bypass-fixed-plot'") == 1
    assert "normalized_mse_loss" in document
    assert '"step": 64' in document
    assert "name:'Mean'" in document


def test_progress_report_uses_subblock_losses_for_subblock_bypass_overfit(tmp_path: Path):
    _write(
        tmp_path / "manifests/bypass_overfit.json",
        {"stage": "bypass_overfit", "status": "success", "config": {}},
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


def test_progress_report_renders_profile_evaluation_and_aiperf_explorers(tmp_path: Path):
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
    assert "id='aiperf-ttft-throughput-plot'" in document
    assert "id='aiperf-latency-throughput-plot'" in document
    assert "id='aiperf-interactivity-throughput-plot'" in document
    assert "id='aiperf-tpot-throughput-plot'" in document


def test_progress_report_preserves_results_and_adds_collapsible_artifact_navigation(
    tmp_path: Path,
):
    _write(
        tmp_path / "manifests/activation_diagnostic.json",
        {"stage": "activation_diagnostic", "status": "success", "config": {}},
    )
    _write(
        tmp_path / "manifests/bypass_overfit.json",
        {"stage": "bypass_overfit", "status": "success", "config": {}},
    )
    rows = [
        {
            "axis": "ffn_intermediate",
            "layer_idx": 3,
            "ratio": 0.5,
            "teacher_value": 3584,
            "target_value": 1792,
            "method": method,
            "lm_loss": value,
        }
        for method, value in (("activation", 1.0), ("random", 1.1), ("reverse", 1.2))
    ]
    _write(
        tmp_path / "artifacts/activation_diagnostic/activation_diagnostic_summary.json",
        {"rows": rows},
    )
    _write(
        tmp_path / "artifacts/bypass/overfit_probe/local_kd_loss_history.json",
        {
            "max_steps": 2,
            "loss_name": "normalized_mse_loss",
            "records": [
                {"step": 1, "loss": 0.2, "per_layer_loss": {"0": 0.3}},
                {"step": 2, "loss": 0.1, "per_layer_loss": {"0": 0.15}},
            ],
        },
    )

    result = generate_campaign_progress_report(tmp_path, model_name="Qwen3.5-0.8B")
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert document.index('id="stage-progress"') < document.index('id="merged-config"')
    assert '<details class="report-section" id="stage-progress" open>' in document
    assert '<details class="report-section" id="merged-config">' in document
    assert '<details class="report-section" id="activation-ranking" open>' in document
    assert '<details class="report-section" id="bypass-overfit" open>' in document
    assert '<details class="report-section" id="nested-bypass">' in document
    assert 'data-stage="activation_diagnostic"' in document
    assert 'href="#activation-ranking"' in document
    assert 'data-stage="bypass_overfit"' in document
    assert 'href="#bypass-overfit"' in document
    depth_card = document.split('data-stage="depth"', maxsplit=1)[1].split("</div>", maxsplit=1)[0]
    assert "href=" not in depth_card
    assert 'id="activation-axis-select"' in document
    assert 'id="bypass-overfit-unit-select"' in document
    assert "target.open=true" in document


def test_progress_report_renders_proper_distillation_terms_and_before_after_eval(
    tmp_path: Path,
):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _proper_distillation_data,
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


def test_replacement_data_omits_empty_width_scenarios(tmp_path: Path):
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import _replacement_data

    (tmp_path / "scenarios/width-4096/depth-00").mkdir(parents=True)

    data = _replacement_data(tmp_path)

    assert data["scenarios"] == []


def test_diverse_bypass_gate_uses_decreasing_per_width_trends_and_diversity():
    from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
        _overfit_summary_card,
    )

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
