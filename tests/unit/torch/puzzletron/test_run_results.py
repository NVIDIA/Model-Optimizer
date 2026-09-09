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

"""Behavior tests for the single-file Puzzletron result contract."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from examples.puzzletron import puzzletron as public_cli
from puzzletron_orchestrator.dashboard import StageView
from puzzletron_orchestrator.result_render import render_run_html, render_run_text
from puzzletron_orchestrator.run_reporting import (
    RESULT_SCHEMA,
    ResultValidationError,
    canonical_json_bytes,
    export_run_result,
    inspect_run,
    publish_controller_result,
    refresh_run_report,
    validate_result,
)

if TYPE_CHECKING:
    from pathlib import Path

NOW = "2026-09-08T00:02:00+00:00"


def _metric(role: str, value: float) -> dict:
    return {
        "metric_id": f"metric:{role}:lm-loss",
        "name": "quality.lm_loss",
        "value": value,
        "value_state": "present",
        "missing_reason": None,
        "unit": "nats_per_target_token",
        "direction": "lower_is_better",
        "subject_id": f"subject:{role}",
        "checkpoint_id": f"checkpoint:{role}",
        "producer_execution_id": "evaluation-1",
        "workload": {
            "workload_id": "loss-screen-1",
            "task": "image-text-loss",
            "row_manifest_id": "rows-1",
            "prompt_template_id": "prompt-1",
            "decoding_id": "teacher-forcing-1",
        },
        "aggregation": "target_token_weighted_mean",
        "dimensions": {
            "denominator": "unmasked_target_tokens",
            "repetition_index": 0,
        },
    }


def _result(*, running: bool = True) -> dict:
    status = "running" if running else "completed"
    subjects = [
        {
            "subject_id": f"subject:{role}",
            "role": role,
            "checkpoint": {"checkpoint_id": f"checkpoint:{role}"},
            "architecture": {
                "architecture_id": f"architecture:{role}",
                "axis_assignments": [
                    {
                        "axis": "ffn_width",
                        "scope": "language_model",
                        "selector": "block:0",
                        "value": width,
                        "unit": "features",
                    }
                ],
            },
        }
        for role, width in (("teacher", 3584), ("candidate", 3072))
    ]
    return {
        "schema": RESULT_SCHEMA,
        "run": {
            "identity": {
                "run_id": "resolved_bundle_123",
                "workflow_id": "workflow-1",
                "model_family": "Qwen",
                "modality": "vlm",
            },
            "status": {
                "execution": status,
                "attachment": "detached" if running else "not_running",
                "evidence": "partial" if running else "complete",
                "support": "unreviewed",
            },
            "timing": {
                "created_at": "2026-09-08T00:00:00+00:00",
                "started_at": "2026-09-08T00:00:30+00:00",
                "updated_at": NOW,
                "ended_at": None if running else NOW,
                "finalized_at": None if running else NOW,
                "last_executor_observation_at": NOW,
            },
            "freshness": {
                "as_of": NOW,
                "stale_after_seconds": 120,
                "state": "fresh",
                "reason": None,
            },
        },
        "subjects": subjects,
        "stages": [
            {
                "stage_id": "prepare",
                "stage_type": "prepare",
                "phase_id": "setup",
                "parent_stage_ids": [],
                "required": True,
                "state": "completed",
                "elapsed_seconds": 30,
                "attempts": [],
                "progress": {
                    "status": "completed",
                    "scope": {"stage_id": "prepare"},
                    "measures": [
                        {
                            "name": "stage_completion",
                            "completed": 1,
                            "total": 1,
                            "unit": "stage",
                            "total_kind": "exact",
                            "primary": True,
                        }
                    ],
                    "eta": {
                        "seconds": None,
                        "qualified": False,
                        "method": "observed_controller_rate",
                        "unavailable_reason": "completed",
                    },
                    "updated_at": NOW,
                },
            },
            {
                "stage_id": "evaluate",
                "stage_type": "evaluation",
                "phase_id": "validation",
                "parent_stage_ids": ["prepare"],
                "required": True,
                "state": status,
                "elapsed_seconds": 90,
                "attempts": [{"attempt_id": "attempt-1", "number": 1}],
                "progress": {
                    "status": status,
                    "scope": {
                        "stage_id": "evaluate",
                        "evaluator_iteration": 3,
                        "repetition_index": 0,
                        "task": "realworldqa",
                    },
                    "measures": [
                        {
                            "name": "processed_samples",
                            "completed": 38 if running else 64,
                            "total": 64,
                            "unit": "samples",
                            "total_kind": "discovered",
                            "primary": True,
                        },
                        {
                            "name": "unmasked_target_tokens",
                            "completed": 4096,
                            "total": 4096,
                            "unit": "tokens",
                            "total_kind": "exact",
                            "primary": False,
                        },
                    ],
                    "eta": {
                        "seconds": 12 if running else None,
                        "qualified": running,
                        "method": "observed_controller_rate",
                        "unavailable_reason": None if running else "completed",
                    },
                    "updated_at": NOW,
                },
            },
        ],
        "metrics": [_metric("teacher", 0.5), _metric("candidate", 0.7)],
        "artifacts": [
            {
                "artifact_id": "artifact:evaluator-log",
                "role": "evaluator_log",
                "path": "artifacts/evaluator.log",
                "availability": "available",
                "validation": "valid",
                "digest": {"algorithm": "sha256", "value": "a" * 64},
            }
        ],
        "provenance": {
            "resolved_bundle": {
                "bundle_id": "resolved_bundle_123",
                "producer_revision": "revision-123",
                "manifest": "orchestration/resolved_bundles/resolved_bundle_123/manifest.json",
                "provenance": "orchestration/resolved_bundles/resolved_bundle_123/provenance.json",
            },
            "controller_state": "orchestration/compiled_plan.json",
        },
        "limitations": ["Single evaluator repetition."],
    }


def _write_result(run_root: Path, result: dict | None = None) -> Path:
    path = run_root / "results/result.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(result or _result()))
    return path


def test_result_supports_detached_inspection_comparison_and_export(tmp_path: Path) -> None:
    result = _result()
    path = _write_result(tmp_path, result)

    view = inspect_run(tmp_path, viewed_at="2026-09-08T00:05:00+00:00")

    assert view["run"]["status"]["attachment"] == "detached"
    assert view["run"]["freshness"]["state"] == "stale"
    assert view["stage_counts"] == {"completed": 1, "running": 1}
    progress = view["active_progress"][0]
    assert progress["scope"] == {
        "stage_id": "evaluate",
        "evaluator_iteration": 3,
        "repetition_index": 0,
        "task": "realworldqa",
    }
    assert progress["measures"][0]["completed"] == 38
    assert progress["eta"] == {
        "seconds": 12,
        "qualified": True,
        "method": "observed_controller_rate",
        "unavailable_reason": None,
    }
    assert view["comparisons"][0]["delta"] == pytest.approx(0.2)
    assert export_run_result(tmp_path) == path
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["stages"][1]["progress"]["measures"][0].update(
                total=None, total_kind="unavailable"
            ),
            "qualified ETA requires seconds and a stable total",
        ),
        (
            lambda result: result["metrics"][0].update(unit="nats_per_token"),
            "quality.lm_loss uses an unsupported measurement contract",
        ),
        (
            lambda result: result["metrics"][0].update(
                value=None, value_state="missing", missing_reason=None
            ),
            "missing value requires a reason",
        ),
        (
            lambda result: result["artifacts"][0].update(path="../outside"),
            "normalized relative POSIX path",
        ),
    ],
)
def test_result_rejects_ambiguous_or_unsafe_evidence(mutate, message: str) -> None:
    result = _result()
    mutate(result)

    with pytest.raises(ResultValidationError, match=message):
        validate_result(result)


def test_html_is_traceable_complete_and_contains_no_unique_evidence(tmp_path: Path) -> None:
    result = _result()
    result["limitations"] = ["bounded<script>"]
    _write_result(tmp_path, result)

    rendered = refresh_run_report(tmp_path, generated_at="2026-09-08T00:03:00+00:00")
    repeated = render_run_html(
        inspect_run(tmp_path, viewed_at="2026-09-08T00:03:00+00:00"),
        generated_at="2026-09-08T00:03:00+00:00",
        renderer_revision="modelopt.puzzletron.html-summary/v1",
    )

    assert rendered == repeated
    assert rendered.manifest["source_result_path"] == "results/result.json"
    assert rendered.manifest["source_result_digest"] == inspect_run(tmp_path)["result_digest"]
    assert rendered.manifest["source_producer_revision"] == "revision-123"
    assert rendered.manifest["source_validation"] == "passed"
    assert b"bounded&lt;script&gt;" in rendered.content
    assert b"bounded<script>" not in rendered.content
    for expected in (
        b"Subjects and heterogeneous configurations",
        b"DAG stages and phases",
        b"Operational progress",
        b"quality.lm_loss",
        b"Teacher and candidate comparisons",
        b"artifacts/evaluator.log",
        b"Provenance",
        b"contains no unique evidence",
    ):
        assert expected in rendered.content
    assert "controller: detached" in render_run_text(inspect_run(tmp_path))


def test_controller_projects_existing_state_into_one_result(monkeypatch, tmp_path: Path) -> None:
    bundle_root = tmp_path / "orchestration/resolved_bundles/resolved_bundle_123"
    bundle_root.mkdir(parents=True)
    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_id": "resolved_bundle_123",
                "code": {"controller": {"revision": "revision-123"}},
            }
        )
    )
    (bundle_root / "provenance.json").write_text("{}")
    (tmp_path / "orchestration/current_bundle.json").write_text(
        json.dumps({"bundle_id": "resolved_bundle_123"})
    )
    checkpoint = tmp_path / "artifacts/post_mip/checkpoints/candidate"
    checkpoint.mkdir(parents=True)
    node_root = tmp_path / "artifacts/post_mip/nodes/evaluate"
    execution_root = node_root / "executions/evaluation-1"
    execution_root.mkdir(parents=True)
    (tmp_path / "artifacts/post_mip/candidate_registry.json").write_text(
        json.dumps(
            {
                "version": 2,
                "architectures": {
                    "architecture-1": {
                        "architecture_id": "architecture-1",
                        "block_configs": [{"layer": 0, "ffn_width": 3072}],
                        "origins": [{"kind": "heterogeneous", "profile_id": "profile-1"}],
                    }
                },
                "revisions": {
                    "revision-1": {
                        "architecture_id": "architecture-1",
                        "artifact": {"checkpoint": str(checkpoint)},
                    }
                },
            }
        )
    )
    (node_root / "current.json").write_text(json.dumps({"execution_identity": "evaluation-1"}))
    comparison = execution_root / "comparison.json"
    comparison.write_text("{}")
    aiperf = execution_root / "raw/candidate/puzzletron_aiperf_result.json"
    aiperf.parent.mkdir(parents=True)
    aiperf.write_text(
        json.dumps(
            {
                "engine": "aiperf",
                "architecture_id": "architecture-1",
                "workload_id": "workload-1",
                "cache_identity": "benchmark-1",
                "repetition": 1,
                "concurrency": 2,
                "topology_id": "tp2",
                "gpu_count": 2,
                "workload": {"input_tokens": 128, "output_tokens": 32},
                "measurement_contract": {"aggregate": "request_mean"},
                "metrics": {
                    "input_sequence_length": 127,
                    "output_token_throughput": 42,
                },
                "raw_artifacts": {},
            }
        )
    )
    (execution_root / "candidate_set.json").write_text("{}")
    (execution_root / "observations.json").write_text(
        json.dumps(
            [
                {
                    "input_revision_id": "revision-1",
                    "source_revision_id": "revision-1",
                    "metrics": {
                        "candidate.lm_loss": 0.7,
                        "reference.lm_loss": 0.5,
                        "delta.lm_loss": 0.2,
                    },
                    "artifacts": {"comparison_path": str(comparison)},
                }
            ]
        )
    )
    monkeypatch.setattr(
        "puzzletron_orchestrator.run_reporting.bundle_for_run_root",
        lambda _run_root: bundle_root,
    )
    plan = SimpleNamespace(
        puzzle_dir=tmp_path,
        contract_hash="workflow-1",
        experiment_config={"display_name": "Qwen", "modality": "vlm"},
        stages=(SimpleNamespace(stage_id="evaluate", parents=()),),
    )
    stage = StageView(
        stage_id="evaluate",
        display_name="Evaluation",
        status="running",
        nodes=1,
        tasks=1,
        gpus=1,
        progress="realworldqa 38/64 samples",
        elapsed_seconds=90,
        eta_seconds=12,
        current=38,
        total=64,
    )

    path = publish_controller_result(
        plan,
        [stage],
        execution_status="running",
        attachment_status="detached",
        now=NOW,
    )
    stored = json.loads(path.read_text())

    assert path == tmp_path / "results/result.json"
    assert path.read_bytes() == canonical_json_bytes(stored)
    assert stored["provenance"]["resolved_bundle"]["producer_revision"] == "revision-123"
    assert stored["stages"][0]["progress"]["measures"][0]["completed"] == 38
    assert stored["stages"][0]["progress"]["eta"]["qualified"] is True
    assert {subject["role"] for subject in stored["subjects"]} == {"candidate", "teacher"}
    assert {metric["name"] for metric in stored["metrics"]} == {
        "quality.lm_loss",
        "serving.observed_input_sequence_length",
        "serving.output_token_throughput",
    }
    loss_metrics = [metric for metric in stored["metrics"] if metric["name"] == "quality.lm_loss"]
    assert {metric["aggregation"] for metric in loss_metrics} == {"mean_of_sample_token_means"}
    throughput = next(
        metric
        for metric in stored["metrics"]
        if metric["name"] == "serving.output_token_throughput"
    )
    assert throughput["dimensions"]["repetition_index"] == 1
    assert throughput["workload"]["requested"] == {"input_tokens": 128, "output_tokens": 32}
    assert stored["subjects"][0]["architecture"]["block_configs"] == [
        {"ffn_width": 3072, "layer": 0}
    ]
    assert any(artifact["path"].endswith("comparison.json") for artifact in stored["artifacts"])
    assert any(
        artifact["path"].endswith("puzzletron_aiperf_result.json")
        for artifact in stored["artifacts"]
    )


def test_public_results_commands_use_the_authoritative_result(capsys, tmp_path: Path) -> None:
    path = _write_result(tmp_path)

    assert public_cli.main(["results", "inspect", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["identity"]["run_id"] == "resolved_bundle_123"
    assert public_cli.main(["results", "export", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == str(path)
    assert public_cli.main(["results", "refresh", str(tmp_path)]) == 0
    assert "campaign_report.html" in capsys.readouterr().out
