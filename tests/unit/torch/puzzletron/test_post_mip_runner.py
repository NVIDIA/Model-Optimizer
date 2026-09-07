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

"""Tests for post-MIP execution, including managed downstream evaluation."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.stages.future as future_stages
import puzzletron_orchestrator.post_mip as orchestration_post_mip
from examples.puzzletron import run_post_mip_node as post_mip_entrypoint
from modelopt.torch.puzzletron.post_mip import runner
from modelopt.torch.puzzletron.post_mip.builtin import ResultManifestNode
from modelopt.torch.puzzletron.post_mip.evidence import (
    collect_kd_exposure,
    exact_checkpoint_evidence,
    kd_exposure_metrics,
)
from modelopt.torch.puzzletron.post_mip.records import (
    ArchitectureCandidate,
    ArtifactKind,
    CandidateLedger,
    CandidateRevision,
    CandidateSet,
    NodeObservation,
)
from modelopt.torch.puzzletron.post_mip.runner import (
    _exception_diagnostics,
    _needs_puzzletron_process_group,
    _post_mip_kd_settings,
    _worker_group,
)


def test_worker_group_uses_torchrun_world_size(monkeypatch):
    monkeypatch.setenv("PUZZLETRON_GROUP_RANK", "0")
    monkeypatch.setenv("PUZZLETRON_GROUP_SIZE", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("PUZZLETRON_TASK_LAUNCHER", "torchrun")

    assert _worker_group() == (1, 2)


def test_worker_group_uses_puzzletron_identity_for_direct_tasks(monkeypatch):
    monkeypatch.setenv("PUZZLETRON_GROUP_RANK", "0")
    monkeypatch.setenv("PUZZLETRON_GROUP_SIZE", "1")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("PUZZLETRON_TASK_LAUNCHER", "direct")

    assert _worker_group() == (0, 1)


def test_exception_diagnostics_preserve_traceback():
    try:
        raise RuntimeError()
    except RuntimeError as error:
        diagnostics = _exception_diagnostics(error)

    assert diagnostics["error"] == "RuntimeError"
    assert "raise RuntimeError()" in diagnostics["traceback"]


def test_result_manifest_config_does_not_require_unused_row_manifest():
    ResultManifestNode.validate_config(
        {
            "type": "result_manifest",
            "config": {
                "pre_kd_source": "materialized",
                "pre_kd_evaluation": "pre_kd_evaluation",
                "profile": "evaluation_profile",
                "reference_checkpoint": "/checkpoint",
                "milestones": [{"steps": 64, "kd": "kd_64", "evaluation": "evaluation_64"}],
            },
        }
    )


def test_global_kd_lets_automodel_initialize_its_nccl_process_group():
    assert _needs_puzzletron_process_group("evaluation")
    assert not _needs_puzzletron_process_group("global_kd")


def test_post_mip_kd_always_requests_a_consolidated_output():
    settings = _post_mip_kd_settings(
        {"global_distillation": {"save_consolidated": False}},
        {"max_steps": 8},
    )

    assert settings["save_consolidated"] is True
    assert settings["max_steps"] == 8


def test_worker_entrypoint_registers_configured_vlm_evaluation_profile(monkeypatch):
    # Keep the examples-layer VLM dependencies out of core test collection.
    from examples.puzzletron.evaluation.vlm import post_mip as vlm_post_mip

    calls = []
    monkeypatch.setattr(vlm_post_mip, "register_profiles", lambda: calls.append(True))

    post_mip_entrypoint._register_evaluation_profiles(
        {
            "post_mip": {
                "flows": {
                    "params": {
                        "nodes": {
                            "checkpoint_eval": {
                                "type": "downstream_evaluation",
                                "config": {
                                    "profile": (
                                        "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3"
                                    )
                                },
                            },
                        }
                    },
                }
            }
        }
    )

    assert calls == [True]


def test_worker_entrypoint_leaves_unknown_vlm_profile_to_fail_closed(monkeypatch, tmp_path):
    from examples.puzzletron.evaluation.vlm import post_mip as vlm_post_mip

    calls = []
    monkeypatch.setattr(vlm_post_mip, "register_profiles", lambda: calls.append(True))
    monkeypatch.setattr(runner, "_DOWNSTREAM_EVALUATION_PROFILES", {})
    post_mip_entrypoint._register_evaluation_profiles(
        {
            "post_mip": {
                "flows": {
                    "params": {
                        "nodes": {
                            "checkpoint_eval": {
                                "type": "downstream_evaluation",
                                "config": {"profile": "qwen35_vlm_unknown"},
                            }
                        }
                    }
                }
            }
        }
    )
    assert calls == [True]

    node = SimpleNamespace(
        node_id="checkpoint_eval",
        config={"config": {"profile": "qwen35_vlm_unknown"}},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    with pytest.raises(ValueError, match="unsupported downstream evaluation profile"):
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")


def test_aggregate_entrypoint_uses_resolved_config(monkeypatch, tmp_path):
    resolved = {
        "evaluation": {"evaluator_revision": "compiled-revision"},
        "puzzle_dir": str(tmp_path),
    }
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved))
    received = []
    monkeypatch.setenv("PUZZLETRON_SOURCE_REVISION", "ambient-revision")
    monkeypatch.setattr(
        orchestration_post_mip,
        "aggregate_post_mip_node",
        lambda config, stage_id: received.append((config, stage_id)) or {"status": "success"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_post_mip_node.py",
            "--resolved-config",
            str(resolved_path),
            "--stage-id",
            "post.params.select",
            "--aggregate",
        ],
    )

    post_mip_entrypoint.main()

    assert received == [(resolved, "post.params.select")]


def test_aggregate_entrypoint_preserves_authored_config_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("puzzle_dir: /campaign\n")
    resolved = {"puzzle_dir": "/campaign", "value": 2}
    loaded = []
    received = []
    monkeypatch.setattr(
        post_mip_entrypoint,
        "load_experiment_config",
        lambda path, *, overrides: loaded.append((path, overrides)) or resolved,
    )
    monkeypatch.setattr(
        orchestration_post_mip,
        "aggregate_post_mip_node",
        lambda config, stage_id: received.append((config, stage_id)) or {"status": "success"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_post_mip_node.py",
            "--config",
            str(config_path),
            "--stage-id",
            "post.params.select",
            "--aggregate",
            "--override",
            "value=2",
        ],
    )

    post_mip_entrypoint.main()

    assert loaded == [(str(config_path), ["value=2"])]
    assert received == [(resolved, "post.params.select")]


def test_online_eval_settings_deep_merge_automodel_overrides():
    scoring = OmegaConf.create(
        {
            "eval_samples": 32,
            "automodel": {
                "force_hf": False,
                "use_puzzletron_dataloader": True,
                "parallel": {"tp": 1, "pp": 1, "dp_shard": 1},
            },
        }
    )

    merged = runner._merge_scoring_settings(
        scoring,
        {
            "eval_samples": 128,
            "automodel": {
                "teacher_cache_device": "cuda",
                "parallel": {"pp": 2, "dp_shard": 2},
            },
        },
    )

    assert merged.eval_samples == 128
    assert merged.automodel.force_hf is False
    assert merged.automodel.use_puzzletron_dataloader is True
    assert merged.automodel.teacher_cache_device == "cuda"
    assert dict(merged.automodel.parallel) == {"tp": 1, "pp": 2, "dp_shard": 2}


def test_online_eval_injects_resolved_hidden_width_into_solution(monkeypatch):
    source = SimpleNamespace(
        artifact={"hidden_width": 1792},
    )
    monkeypatch.setattr(
        runner,
        "_raw_solution",
        lambda _source: {"chosen_replacements": [{"layer_replacement": {}}]},
    )
    monkeypatch.setattr(
        runner,
        "_scenario_checkpoint_roles",
        lambda scenario, width: (Path("/sorted"), None),
    )

    work = runner._config_evaluation_work(
        {"puzzle_dir": "/puzzle"},
        "revision-1",
        source,
    )

    assert work.hidden_width == 1792
    assert work.raw_solution["hidden_width"] == 1792


def test_checkpoint_evaluation_manifest_uses_candidate_effective_config(monkeypatch, tmp_path):
    observed = {}
    checkpoint = tmp_path / "checkpoint"
    teacher = tmp_path / "teacher"
    legacy_teacher = tmp_path / "legacy-teacher"
    node = SimpleNamespace(
        node_id="evaluation",
        stage_id="post.params.evaluation",
        config={
            "config": {
                "tasks": ["candidate-task"],
                "reference_checkpoint": str(teacher),
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(checkpoint)},
    )
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {"teacher_dir": str(legacy_teacher)},
        "zero_shot_evaluation": {"enabled": False},
        "_runtime": {
            "authored_config": {
                "puzzle_dir": str(tmp_path),
                "zero_shot_evaluation": {"enabled": False},
            }
        },
    }

    def _evaluation_stage(candidate, manifest):
        observed["semantic_config"] = manifest.semantic_config
        output = Path(candidate["zero_shot_evaluation"]["output_dir"])
        output.mkdir(parents=True)
        (output / "evaluation_summary.json").write_text(
            json.dumps(
                [
                    {
                        "checkpoint": str(checkpoint),
                        "metrics": {"score": 1.0},
                        "result_path": str(output / "result.json"),
                    },
                    {
                        "checkpoint": str(teacher),
                        "metrics": {"score": 1.25},
                        "result_path": str(output / "teacher.json"),
                    },
                ]
            )
        )

    monkeypatch.setattr(future_stages, "evaluation_stage", _evaluation_stage)

    result = runner._evaluate_checkpoint(config, node, source, "execution")

    assert observed["semantic_config"]["zero_shot_evaluation"] == {
        "enabled": True,
        "checkpoints": [str(checkpoint), str(teacher)],
        "output_dir": str(
            tmp_path / "artifacts/post_mip/nodes/evaluation/executions/execution/raw/architecture"
        ),
        "tasks": ["candidate-task"],
    }
    assert result["metrics"] == {
        "score": 1.0,
        "candidate.score": 1.0,
        "reference.score": 1.25,
        "delta.score": -0.25,
    }
    assert result["reference_result_path"] == str(
        tmp_path
        / "artifacts/post_mip/nodes/evaluation/executions/execution/raw/architecture"
        / "teacher.json"
    )


def test_checkpoint_evaluation_requires_configured_reference_result(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    reference = tmp_path / "reference"
    node = SimpleNamespace(
        node_id="evaluation",
        stage_id="post.params.evaluation",
        config={"config": {"reference_checkpoint": str(reference)}},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(checkpoint)},
    )
    config = {"puzzle_dir": str(tmp_path)}

    def _evaluation_stage(candidate, manifest):
        del manifest
        output = Path(candidate["zero_shot_evaluation"]["output_dir"])
        output.mkdir(parents=True)
        (output / "evaluation_summary.json").write_text(
            json.dumps([{"checkpoint": str(checkpoint), "metrics": {"score": 1.0}}])
        )

    monkeypatch.setattr(future_stages, "evaluation_stage", _evaluation_stage)

    with pytest.raises(RuntimeError, match="reference checkpoint is missing"):
        runner._evaluate_checkpoint(config, node, source, "execution")


def test_checkpoint_evaluation_does_not_add_implicit_reference(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    teacher = tmp_path / "teacher"
    observed = {}
    node = SimpleNamespace(
        node_id="evaluation",
        stage_id="post.params.evaluation",
        config={"config": {"eval_samples": 8}},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(checkpoint)},
    )
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {"teacher_dir": str(teacher)},
    }

    def _evaluation_stage(candidate, manifest):
        observed["semantic_config"] = manifest.semantic_config
        output = Path(candidate["zero_shot_evaluation"]["output_dir"])
        output.mkdir(parents=True)
        (output / "evaluation_summary.json").write_text(
            json.dumps([{"checkpoint": str(checkpoint), "metrics": {"score": 1.0}}])
        )

    monkeypatch.setattr(future_stages, "evaluation_stage", _evaluation_stage)

    result = runner._evaluate_checkpoint(config, node, source, "execution")

    assert observed["semantic_config"]["zero_shot_evaluation"]["checkpoints"] == [str(checkpoint)]
    assert result["metrics"] == {"score": 1.0}


def test_aiperf_consumes_request_count_without_forwarding_setup_only_keys(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_run_aiperf_sweep(checkpoint, **settings):
        captured["checkpoint"] = checkpoint
        captured.update(settings)
        return [
            SimpleNamespace(
                concurrency=8,
                workload={"image_batch_size": 12},
                metrics={"output_token_throughput": throughput},
                raw_artifacts={},
            )
            for throughput in (10.0, 14.0)
        ]

    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.run_aiperf_sweep",
        fake_run_aiperf_sweep,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    node = SimpleNamespace(
        node_id="serving",
        flow_id="params",
        config={
            "config": {
                "concurrency": [8],
                "request_count": 23,
                "minimum_request_count": 4,
                "requests_per_concurrency": 2,
                "best_selection_mode": "individual_best",
                "allow_aiperf_v011_online_tokenizer_resolution": True,
                "input_tokens": 1024,
                "output_tokens": 128,
                "image_batch_sizes": [1, 6, 12],
                "image_width_mean": 1280,
                "image_height_mean": 720,
                "topology": {"gpu_group_size": 1},
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    result = runner._aiperf(
        {"puzzle_dir": str(tmp_path), "model": {"trust_remote_code": True}},
        node,
        source,
        "execution",
    )

    assert captured["checkpoint"] == str(tmp_path / "checkpoint")
    assert captured["concurrencies"] == (8,)
    assert captured["request_counts"] == {8: 23}
    assert captured["trust_remote_code"] is True
    assert captured["allow_aiperf_v011_online_tokenizer_resolution"] is True
    assert captured["image_batch_sizes"] == [1, 6, 12]
    assert captured["image_width_mean"] == 1280
    assert captured["image_height_mean"] == 720
    assert "request_count" not in captured
    assert "minimum_request_count" not in captured
    assert "requests_per_concurrency" not in captured
    assert "best_selection_mode" not in captured
    assert result["metrics"] == {
        "output_token_throughput": 12.0,
        "images_12.concurrency_8.output_token_throughput": 12.0,
    }


def test_aiperf_rejects_repetitions_with_different_metric_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.benchmarks.run_aiperf_sweep",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                concurrency=1,
                workload={},
                metrics={"latency": 1.0, "throughput": 2.0},
                raw_artifacts={},
            ),
            SimpleNamespace(
                concurrency=1,
                workload={},
                metrics={"latency": 1.1},
                raw_artifacts={},
            ),
        ],
    )
    node = SimpleNamespace(
        node_id="serving",
        flow_id="params",
        config={"config": {"concurrency": [1], "topology": {"gpu_group_size": 1}}},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact={"checkpoint": str(tmp_path / "checkpoint")},
    )

    with pytest.raises(RuntimeError, match="repetitions produced different metrics"):
        runner._aiperf({"puzzle_dir": str(tmp_path)}, node, source, "execution")


def test_downstream_evaluation_delegates_to_generic_checkpoint_evaluator(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = {}

    def fake_evaluate(checkpoint_path, *, output_root, settings):
        captured.update(
            checkpoint=checkpoint_path,
            output_root=output_root,
            settings=settings,
        )
        return {"metrics": {"ifeval.accuracy": 0.5}}

    monkeypatch.setattr(runner, "run_lmms_eval_checkpoint", fake_evaluate)
    node = SimpleNamespace(
        node_id="lmms_eval",
        config={"config": {"tasks": ["ifeval"], "limit": 4}},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(checkpoint)},
    )

    result = runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")

    assert result == {"metrics": {"ifeval.accuracy": 0.5}}
    assert captured == {
        "checkpoint": str(checkpoint),
        "output_root": (
            tmp_path
            / "artifacts/post_mip/nodes/lmms_eval/executions/execution/raw/architecture/lmms_eval"
        ),
        "settings": {"tasks": ["ifeval"], "limit": 4},
    }


def test_downstream_evaluation_routes_the_pinned_vlm_profile(monkeypatch, tmp_path):
    # Keep the examples-layer VLM dependencies out of core test collection.
    from examples.puzzletron.evaluation.vlm import post_mip

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = {}

    def fake_evaluate(checkpoint_path, *, output_root, settings):
        captured.update(
            checkpoint=checkpoint_path,
            output_root=output_root,
            settings=settings,
        )
        return {"metrics": {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5}}

    monkeypatch.setattr(runner, "_DOWNSTREAM_EVALUATION_PROFILES", {})
    monkeypatch.setattr(post_mip, "evaluate_realworldqa_checkpoint", fake_evaluate)
    post_mip.register_profiles()
    node = SimpleNamespace(
        node_id="checkpoint_eval",
        config={
            "config": {
                "profile": "qwen35_vlm_realworldqa",
                "batch_size": 1,
                "timeout_seconds": 600,
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(checkpoint)},
    )

    result = runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")

    assert result["metrics"] == {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5}
    assert captured["checkpoint"] == str(checkpoint)
    assert captured["settings"] == {"batch_size": 1, "timeout_seconds": 600}


def test_short_v1_profile_binds_the_exact_row_manifest_digest(monkeypatch, tmp_path):
    from examples.puzzletron.evaluation.vlm import post_mip

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "short-v1.json"
    manifest.write_text("{}")
    captured = {}
    monkeypatch.setattr(post_mip.suites, "load_quick_manifest", lambda path: {"path": str(path)})
    monkeypatch.setattr(post_mip.suites, "manifest_sha256", lambda _manifest: "a" * 64)

    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        captured.update(args=args, settings=settings_overrides)
        profile = dict.fromkeys(post_mip._PROFILE_CONTRACT_FIELDS)
        profile.update(
            profile="fixture",
            source_tasks=["fixture"],
            quick_selected_rows=1,
            quick_row_identities={"fixture": [{}]},
            quick_task_denominators={"fixture": {"selected_rows": 1}},
            repetitions=1,
        )
        preflight_callback(profile)
        result_path = tmp_path / "output" / "result.json"
        result_path.write_text(json.dumps({"sample_counts": {"modelopt_vlm_benchmark_fixture": 1}}))
        return {"runs": [{"metrics": {"accuracy": 0.5}, "result_path": str(result_path)}]}

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)
    with pytest.warns(DeprecationWarning, match="qwen35_vlm_short_v1 is deprecated"):
        result = post_mip.evaluate_short_v1_checkpoint(
            checkpoint,
            output_root=tmp_path / "output",
            settings={
                "row_manifest": str(manifest),
                "row_manifest_sha256": "a" * 64,
                "batch_size": 1,
            },
        )

    assert captured["args"].suite == "quick"
    assert captured["args"].quick_manifest == manifest
    assert captured["settings"] == {}
    assert result["checkpoint"] == str(checkpoint)

    with (
        pytest.warns(DeprecationWarning, match="qwen35_vlm_short_v1 is deprecated"),
        pytest.raises(ValueError, match="differs from the profile identity"),
    ):
        post_mip.evaluate_short_v1_checkpoint(
            checkpoint,
            output_root=tmp_path / "mismatch",
            settings={
                "row_manifest": str(manifest),
                "row_manifest_sha256": "b" * 64,
            },
        )

    for incomplete_settings in (
        {"row_manifest": str(manifest)},
        {"row_manifest_sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="requires row_manifest and row_manifest_sha256"):
            post_mip.evaluate_frozen_campaign_checkpoint(
                checkpoint,
                output_root=tmp_path / "missing-manifest-setting",
                settings=incomplete_settings,
            )


def test_short_v3_profile_uses_vllm_and_binds_the_embedded_row_manifest_digest(
    monkeypatch, tmp_path
):
    from examples.puzzletron.evaluation.vlm import contracts, post_mip

    monkeypatch.setattr(runner, "_DOWNSTREAM_EVALUATION_PROFILES", {})
    post_mip.register_profiles()
    assert (
        runner._DOWNSTREAM_EVALUATION_PROFILES[
            "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3"
        ]
        is post_mip.evaluate_frozen_campaign_v3_checkpoint
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    exact_rows = contracts.load_profile("core-3_344-examples_r1-vllm").exact_rows
    assert exact_rows is not None
    expected_digest = post_mip.suites.manifest_sha256(exact_rows)
    captured = {}

    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        captured.update(args=args, settings=settings_overrides)
        profile = dict.fromkeys(post_mip._PROFILE_CONTRACT_FIELDS)
        profile.update(
            profile="fixture",
            quick_manifest_sha256=expected_digest,
            source_tasks=["fixture"],
            quick_selected_rows=1,
            quick_row_identities={"fixture": [{}]},
            quick_task_denominators={"fixture": {"selected_rows": 1}},
            repetitions=1,
        )
        preflight_callback(profile)
        result_path = tmp_path / "output" / "result.json"
        result_path.write_text(json.dumps({"sample_counts": {"modelopt_vlm_benchmark_fixture": 1}}))
        return {"runs": [{"metrics": {"accuracy": 0.5}, "result_path": str(result_path)}]}

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)
    result = post_mip.evaluate_frozen_campaign_v3_checkpoint(
        checkpoint,
        output_root=tmp_path / "output",
        settings={"row_manifest_sha256": expected_digest, "batch_size": 1},
    )

    assert captured["args"].profile == "core-3_344-examples_r1-vllm"
    assert captured["args"].quick_manifest is None
    assert captured["settings"] == {}
    assert result["checkpoint"] == str(checkpoint)

    with pytest.raises(ValueError, match="differs from the campaign identity"):
        post_mip.evaluate_frozen_campaign_v3_checkpoint(
            checkpoint,
            output_root=tmp_path / "mismatch",
            settings={"row_manifest_sha256": "a" * 64},
        )


def test_downstream_evaluation_compares_candidate_with_reference(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "teacher"
    candidate.mkdir()
    reference.mkdir()
    calls = []

    def fake_evaluate(checkpoint_path, *, output_root, settings):
        calls.append((Path(checkpoint_path), output_root, settings))
        score = 0.4 if Path(checkpoint_path) == candidate else 0.5
        result_path = tmp_path / f"{Path(checkpoint_path).name}.json"
        result_path.write_text(json.dumps({"raw": "adapter-owned"}))
        return {
            "metrics": {"ifeval.accuracy": score},
            "result_path": str(result_path),
            "contract": {
                "schema": "fixture.evaluator-contract/v1",
                "dataset_revision": "dataset-revision",
            },
            "evidence": {
                "schema": "fixture.evaluation-evidence/v1",
                "sample_ids": ["example-1"],
            },
        }

    monkeypatch.setattr(runner, "run_lmms_eval_checkpoint", fake_evaluate)
    source = CandidateRevision(
        revision_id="revision",
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(candidate)},
        producer_node="kd_256",
    )
    identity = runner._downstream_evaluation_identity(
        source=source,
        reference_checkpoint=reference,
        profile=None,
        evaluator_revision="source-revision",
        candidate=fake_evaluate(candidate, output_root=tmp_path, settings={"tasks": ["ifeval"]}),
        reference=fake_evaluate(reference, output_root=tmp_path, settings={"tasks": ["ifeval"]}),
    )
    assert identity["architecture_id"] == "architecture"
    assert identity["kd"] == {"producer_node": "kd_256", "exposure": None}
    assert identity["evaluator"]["revision"] == "source-revision"
    assert identity["evaluator"]["contract"] == {
        "schema": "fixture.evaluator-contract/v1",
        "dataset_revision": "dataset-revision",
    }
    assert identity["evaluation_evidence"] == {
        "schema": "fixture.evaluation-evidence/v1",
        "sample_ids": ["example-1"],
    }
    assert identity["reference_evaluation_evidence"] == identity["evaluation_evidence"]
    invalid_candidate = fake_evaluate(candidate, output_root=tmp_path, settings={})
    del invalid_candidate["evidence"]["schema"]
    with pytest.raises(ValueError, match="evidence must declare a non-empty schema"):
        runner._downstream_evaluation_identity(
            source=source,
            reference_checkpoint=reference,
            profile=None,
            evaluator_revision="source-revision",
            candidate=invalid_candidate,
            reference=fake_evaluate(reference, output_root=tmp_path, settings={}),
        )
    calls.clear()
    node = SimpleNamespace(
        node_id="full_benchmarks",
        config={
            "config": {
                "tasks": ["ifeval"],
                "evaluator_revision": "source-revision",
                "reference_checkpoint": str(reference),
                "recorded_observation": {
                    "repeat_count": 2,
                    "identity": identity,
                    "metrics": {
                        "candidate.ifeval.accuracy": 0.35,
                        "reference.ifeval.accuracy": 0.5,
                    },
                },
            }
        },
    )
    result = runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")

    assert [call[0] for call in calls] == [candidate, reference]
    assert all(call[2] == {"tasks": ["ifeval"]} for call in calls)
    assert result["metrics"] == {
        "ifeval.accuracy": 0.4,
        "candidate.ifeval.accuracy": 0.4,
        "reference.ifeval.accuracy": 0.5,
        "delta.ifeval.accuracy": pytest.approx(-0.1),
        "observation_delta.candidate.ifeval.accuracy": pytest.approx(0.05),
        "observation_delta.reference.ifeval.accuracy": 0.0,
    }
    comparison = json.loads(Path(result["comparison_path"]).read_text())
    assert comparison["candidate"]["metrics"] == {"ifeval.accuracy": 0.4}
    assert comparison["reference"]["metrics"] == {"ifeval.accuracy": 0.5}
    assert comparison["delta"]["ifeval.accuracy"] == pytest.approx(-0.1)
    assert comparison["recorded_observation"] == {
        "repeat_count": 2,
        "status": "matched",
        "identity": identity,
        "metrics": {
            "candidate.ifeval.accuracy": 0.35,
            "reference.ifeval.accuracy": 0.5,
        },
        "difference_from_recorded": {
            "candidate.ifeval.accuracy": pytest.approx(0.05),
            "reference.ifeval.accuracy": 0.0,
        },
    }
    assert comparison["identity"] == identity
    assert comparison["evidence"] == {
        "candidate_result_path": str(tmp_path / "candidate.json"),
        "reference_result_path": str(tmp_path / "teacher.json"),
    }


def test_recorded_observation_differences_are_suppressed_on_identity_mismatch():
    comparison, metrics = runner._compare_recorded_observation(
        {"identity": {"manifest": "old"}, "metrics": {"candidate.accuracy": 0.4}},
        {"candidate.accuracy": 0.5},
        {"manifest": "current"},
    )

    assert comparison == {
        "status": "identity_mismatch",
        "identity": {"manifest": "old"},
        "actual_identity": {"manifest": "current"},
    }
    assert metrics == {}


@pytest.mark.parametrize(
    ("changed_identity", "expected_reference_calls"),
    [
        (None, 1),
        ("evaluator_revision", 2),
        ("checkpoint_fingerprint", 2),
        ("evaluator_contract", 2),
    ],
)
def test_downstream_evaluation_reuses_only_matching_reference_cache(
    monkeypatch, tmp_path, changed_identity, expected_reference_calls
):
    reference = tmp_path / "teacher"
    reference.mkdir()
    candidates = [tmp_path / "candidate-a", tmp_path / "candidate-b"]
    for candidate in candidates:
        candidate.mkdir()
    calls = []
    fingerprints = {reference: "teacher-a"}
    contract_revision = {"value": "contract-a"}

    def fake_evaluate(checkpoint_path, *, output_root, settings):
        calls.append(Path(checkpoint_path))
        result_path = Path(output_root) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"sample_counts": {"fixture": 1}}))
        return {
            "metrics": {"accuracy": 0.5 if Path(checkpoint_path) == reference else 0.4},
            "result_path": str(result_path),
            "contract": {
                "schema": "fixture.evaluator-contract/v1",
                "revision": contract_revision["value"],
            },
        }

    def fake_fingerprint(checkpoint_path):
        path = Path(checkpoint_path)
        return fingerprints.get(path, f"candidate-{path.name}")

    monkeypatch.setattr(runner, "run_lmms_eval_checkpoint", fake_evaluate)
    monkeypatch.setattr(runner, "_checkpoint_fingerprint", fake_fingerprint)
    node = SimpleNamespace(
        node_id="short_v1",
        config={
            "config": {
                "reference_checkpoint": str(reference),
                "reference_once": True,
                "reference_cache_id": "short-v1-teacher",
                "evaluator_revision": "revision-a",
                "tasks": ["fixture"],
            }
        },
    )
    config = {"puzzle_dir": str(tmp_path)}
    for index, candidate in enumerate(candidates):
        if index == 1 and changed_identity == "evaluator_revision":
            node.config["config"]["evaluator_revision"] = "revision-b"
        if index == 1 and changed_identity == "checkpoint_fingerprint":
            fingerprints[reference] = "teacher-b"
        if index == 1 and changed_identity == "evaluator_contract":
            contract_revision["value"] = "contract-b"
        source = SimpleNamespace(
            architecture_id=f"architecture-{index}",
            artifact_kind=ArtifactKind.CHECKPOINT,
            artifact={"checkpoint": str(candidate)},
            producer_node="kd_64",
        )
        runner._downstream_evaluation(config, node, source, f"execution-{index}")

    assert calls.count(reference) == expected_reference_calls


def test_downstream_evaluation_rejects_candidate_reference_contract_mismatch(
    monkeypatch, tmp_path
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "teacher"
    candidate.mkdir()
    reference.mkdir()

    def fake_evaluate(checkpoint_path, *, output_root, settings):
        del settings
        result_path = Path(output_root) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}")
        return {
            "metrics": {"accuracy": 0.5},
            "result_path": str(result_path),
            "contract": {
                "schema": "fixture.evaluator-contract/v1",
                "revision": Path(checkpoint_path).name,
            },
        }

    monkeypatch.setattr(runner, "run_lmms_eval_checkpoint", fake_evaluate)
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(candidate)},
        producer_node="materialized",
    )
    node = SimpleNamespace(
        node_id="evaluation",
        config={"config": {"reference_checkpoint": str(reference)}},
    )

    with pytest.raises(RuntimeError, match="candidate and reference evaluator contracts differ"):
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")


def test_reference_cache_requires_evaluator_revision(monkeypatch, tmp_path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "teacher"
    candidate.mkdir()
    reference.mkdir()
    monkeypatch.setattr(
        runner,
        "run_lmms_eval_checkpoint",
        lambda checkpoint, **kwargs: {"metrics": {}, "result_path": str(checkpoint)},
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(candidate)},
        producer_node="materialized",
    )
    node = SimpleNamespace(
        node_id="evaluation",
        config={
            "config": {
                "reference_checkpoint": str(reference),
                "reference_once": True,
                "reference_cache_id": "teacher",
            }
        },
    )

    with pytest.raises(ValueError, match="reference_once requires evaluator_revision"):
        runner._downstream_evaluation({"puzzle_dir": str(tmp_path)}, node, source, "execution")


def test_global_kd_resume_reports_durable_incremental_gpu_hours(tmp_path):
    output = tmp_path / "trajectory"
    training_log = output / "checkpoints" / "training.jsonl"
    training_log.parent.mkdir(parents=True)
    training_log.write_text(json.dumps({"num_label_tokens": 128}) + "\n")
    exposure_path = output / "exposure" / "step_000064.json"
    exposure_path.parent.mkdir()
    exposure_path.write_text(
        json.dumps(
            {
                "actual_incremental_gpu_hours": 1.25,
                "actual_cumulative_gpu_hours": 2.5,
            }
        )
    )
    exposure = collect_kd_exposure(
        output,
        {
            "global_batch_size": 4,
            "cumulative_examples": 256,
            "max_sample_length": 512,
            "estimated_cumulative_gpu_hours": 3.0,
        },
        max_steps=64,
        elapsed_gpu_hours=1.0,
        resumed_completed_milestone=True,
    )

    assert exposure["actual_incremental_gpu_hours"] == 1.25
    assert exposure["actual_cumulative_gpu_hours"] == 2.5
    assert kd_exposure_metrics(exposure)["exposure.actual_incremental_gpu_hours"] == 1.25


def test_exact_checkpoint_evidence_reads_physical_shapes_and_counts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "block_configs": [{"subblock_configs": []}],
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                },
            }
        )
    )
    header = json.dumps(
        {"model.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    ).encode()
    (checkpoint / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(24)
    )

    evidence = exact_checkpoint_evidence(checkpoint)

    assert evidence["geometry"] == {
        "block_configs": [{"subblock_configs": []}],
        "hidden_size": 8,
        "num_hidden_layers": 1,
    }
    assert evidence["parameter_count"] == 6
    assert evidence["tensor_count"] == 1
    assert evidence["tensor_shapes"] == {"model.weight": {"dtype": "F32", "shape": [2, 3]}}


def _result_manifest_case(monkeypatch, tmp_path):
    block_configs = [
        {
            "subblock_configs": [
                {
                    "kind": "attention",
                    "name": "attention",
                    "num_kv_heads": 1,
                    "num_query_heads": 3,
                },
                {
                    "kind": "ffn",
                    "name": "ffn",
                    "intermediate_size": 3328,
                },
                {
                    "kind": "mamba",
                    "name": "mamba",
                    "head_dim": 112,
                    "num_groups": 14,
                    "num_heads": 14,
                    "state_dim": 112,
                },
            ]
        }
    ]

    def checkpoint_evidence(checkpoint):
        # KD may persist numerically sensitive parameters in a wider dtype without
        # changing the student's physical geometry.
        dtype = "BF16" if Path(checkpoint).name == "pre-kd" else "F32"
        return {
            "content_manifest_sha256": Path(checkpoint).name,
            "geometry": {
                "block_configs": block_configs,
                "hidden_size": 8,
                "num_hidden_layers": 1,
            },
            "parameter_count": 6,
            "tensor_count": 1,
            "tensor_shapes": {"model.weight": {"dtype": dtype, "shape": [2, 3]}},
        }

    monkeypatch.setattr(
        runner,
        "_exact_checkpoint_evidence",
        checkpoint_evidence,
    )
    ledger = CandidateLedger(tmp_path / "ledger")
    architecture_id = "architecture"
    ledger.architectures[architecture_id] = ArchitectureCandidate(
        architecture_id=architecture_id,
        block_configs=block_configs,
        mip_metrics={"parameter_ratio": 0.9},
    )
    parent = None
    revisions = {}
    for node_id, checkpoint in (
        ("materialized", "pre-kd"),
        ("kd_64", "step-64"),
        ("kd_128", "step-128"),
        ("kd_256", "step-256"),
    ):
        revision_id = f"revision-{node_id}"
        ledger.revisions[revision_id] = CandidateRevision(
            revision_id=revision_id,
            architecture_id=architecture_id,
            artifact_kind=ArtifactKind.CHECKPOINT,
            artifact={"checkpoint": str(tmp_path / checkpoint)},
            parent_revision_id=parent,
            producer_node=node_id,
        )
        revisions[node_id] = revision_id
        parent = revision_id
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    reference_fingerprint = runner._checkpoint_fingerprint(teacher)
    profile = "fixture_profile"
    evaluator_contract = {
        "schema": "fixture.evaluator-contract/v1",
        "dataset": {"revision": "dataset-revision", "selection_sha256": "a" * 64},
        "generation": {"do_sample": False},
    }

    def evaluation_evidence(checkpoint):
        return {
            "schema": "fixture.evaluation-evidence/v1",
            "checkpoint": checkpoint,
            "outcomes": {"completed": 24, "failed": 0},
        }

    def evaluation_identity(steps):
        return {
            "candidate_checkpoint_fingerprint": f"student-{steps}",
            "reference_checkpoint_fingerprint": reference_fingerprint,
            "architecture_id": architecture_id,
            "kd": {"producer_node": f"kd_{steps}", "exposure": {"cumulative_steps": steps}},
            "evaluation_evidence": evaluation_evidence(f"student-{steps}"),
            "reference_evaluation_evidence": evaluation_evidence("teacher"),
            "reference_evaluator_contract": copy.deepcopy(evaluator_contract),
            "evaluator": {
                "profile": profile,
                "revision": "source-revision",
                "contract": copy.deepcopy(evaluator_contract),
            },
        }

    for steps in (64, 128, 256):
        kd_node = f"kd_{steps}"
        eval_node = f"short_v1_{steps}"
        kd_input = revisions["materialized"] if steps == 64 else revisions[f"kd_{steps // 2}"]
        ledger.observations[kd_node] = {
            kd_input: NodeObservation(
                node_id=kd_node,
                input_revision_id=kd_input,
                source_revision_id=revisions["materialized"],
                output_revision_id=revisions[kd_node],
                status="success",
                metrics={"exposure.effective_tokens": float(steps * 100)},
            )
        }
        ledger.observations[eval_node] = {
            revisions[kd_node]: NodeObservation(
                node_id=eval_node,
                input_revision_id=revisions[kd_node],
                source_revision_id=revisions[kd_node],
                output_revision_id=revisions[kd_node],
                status="success",
                metrics={"accuracy": steps / 1000},
                artifacts={"comparison_path": str(tmp_path / f"comparison-{steps}.json")},
            )
        }
        (tmp_path / f"comparison-{steps}.json").write_text(
            json.dumps({"identity": evaluation_identity(steps)})
        )
    ledger.observations["materialized"] = {
        revisions["materialized"]: NodeObservation(
            node_id="materialized",
            input_revision_id=revisions["materialized"],
            source_revision_id=revisions["materialized"],
            output_revision_id=revisions["materialized"],
            status="success",
        )
    }
    pre_kd_comparison = tmp_path / "comparison-pre-kd.json"
    pre_kd_comparison.write_text(json.dumps({"identity": evaluation_identity(0)}))
    ledger.observations["pre_kd_short_v1"] = {
        revisions["materialized"]: NodeObservation(
            node_id="pre_kd_short_v1",
            input_revision_id=revisions["materialized"],
            source_revision_id=revisions["materialized"],
            output_revision_id=revisions["materialized"],
            status="success",
            metrics={"accuracy": 0.1},
            artifacts={"comparison_path": str(pre_kd_comparison)},
        )
    }
    node = SimpleNamespace(
        node_id="bounded_result",
        flow_id="campaign",
        config={
            "config": {
                "pre_kd_source": "materialized",
                "pre_kd_evaluation": "pre_kd_short_v1",
                "profile": profile,
                "reference_checkpoint": str(teacher),
                "reference_cache_id": "teacher",
                "milestones": [
                    {
                        "steps": steps,
                        "kd": f"kd_{steps}",
                        "evaluation": f"short_v1_{steps}",
                    }
                    for steps in (64, 128, 256)
                ],
            }
        },
    )
    input_set = CandidateSet.create(
        "campaign",
        "selected",
        [revisions["kd_256"]],
        producer_execution_identity="selected-execution",
    )

    return SimpleNamespace(
        architecture_id=architecture_id,
        block_configs=block_configs,
        checkpoint_evidence=checkpoint_evidence,
        evaluation_evidence=evaluation_evidence,
        evaluation_identity=evaluation_identity,
        evaluator_contract=evaluator_contract,
        input_set=input_set,
        ledger=ledger,
        node=node,
        profile=profile,
        reference_fingerprint=reference_fingerprint,
        revisions=revisions,
    )


def test_result_manifest_freezes_pre_kd_and_learning_curve(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)
    architecture_id = case.architecture_id
    block_configs = case.block_configs
    evaluation_evidence = case.evaluation_evidence
    evaluation_identity = case.evaluation_identity
    evaluator_contract = case.evaluator_contract
    input_set = case.input_set
    ledger = case.ledger
    node = case.node
    profile = case.profile
    reference_fingerprint = case.reference_fingerprint
    revisions = case.revisions

    observations, output_set = runner._aggregate_result_manifest(
        {"puzzle_dir": str(tmp_path)}, ledger, node, input_set, "manifest-execution"
    )

    manifest = json.loads(Path(observations[0].artifacts["result_manifest_path"]).read_text())
    assert output_set.revision_ids == (revisions["kd_256"],)
    assert manifest["pre_kd"]["checkpoint"] == str(tmp_path / "pre-kd")
    assert manifest["pre_kd"]["evaluation_identity"] == evaluation_identity(0)
    assert manifest["pre_kd"]["evaluation_metrics"] == {"accuracy": 0.1}
    assert [row["steps"] for row in manifest["milestones"]] == [64, 128, 256]
    assert manifest["evaluation_contract"]["evaluator"]["contract"] == evaluator_contract
    assert [row["evaluation_identity"] for row in manifest["milestones"]] == [
        evaluation_identity(64),
        evaluation_identity(128),
        evaluation_identity(256),
    ]

    def expected_evaluation_result(accuracy):
        return {
            "candidate_evidence": evaluation_evidence(
                "student-0" if accuracy == 0.1 else f"student-{int(accuracy * 1000)}"
            ),
            "metrics": {"accuracy": accuracy},
            "reference_evidence": evaluation_evidence("teacher"),
        }

    assert manifest["exact_result"] == {
        "axis_inventory": block_configs,
        "checkpoint_and_lineage_identities": {
            "architecture_id": architecture_id,
            "milestones": [
                {
                    "checkpoint_fingerprint": f"student-{steps}",
                    "content_manifest_sha256": f"step-{steps}",
                    "producer_node": f"kd_{steps}",
                    "steps": steps,
                }
                for steps in (64, 128, 256)
            ],
            "pre_kd_content_manifest_sha256": "pre-kd",
            "pre_kd_checkpoint_fingerprint": "student-0",
            "reference_checkpoint_fingerprint": reference_fingerprint,
        },
        "evaluation_results": {
            "milestones": [
                {"steps": steps, **expected_evaluation_result(steps / 1000)}
                for steps in (64, 128, 256)
            ],
            "pre_kd": expected_evaluation_result(0.1),
        },
        "evaluator_contract": {
            "contract": evaluator_contract,
            "profile": profile,
            "revision": "source-revision",
        },
        "kd_exposure": [{"cumulative_steps": steps} for steps in (64, 128, 256)],
        "parameter_counts": {
            "materialized_checkpoint": 6,
            "mip_estimates": {"parameter_ratio": 0.9},
        },
        "realized_geometry_and_tensor_shapes": {
            "geometry": {
                "block_configs": block_configs,
                "hidden_size": 8,
                "num_hidden_layers": 1,
            },
            "tensor_count": 1,
            "tensor_shapes": {"model.weight": {"dtype": "BF16", "shape": [2, 3]}},
        },
        "stage_completion": {
            "milestones": [
                {"evaluation": "success", "kd": "success", "steps": steps}
                for steps in (64, 128, 256)
            ],
            "pre_kd": "success",
        },
    }


def test_result_manifest_rejects_changed_checkpoint_geometry(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)

    def mismatched_checkpoint_evidence(checkpoint):
        evidence = case.checkpoint_evidence(checkpoint)
        if Path(checkpoint).name == "step-128":
            evidence["tensor_shapes"]["model.weight"]["shape"] = [2, 4]
        return evidence

    monkeypatch.setattr(runner, "_exact_checkpoint_evidence", mismatched_checkpoint_evidence)

    with pytest.raises(RuntimeError, match="checkpoint geometry differs"):
        runner._aggregate_result_manifest(
            {"puzzle_dir": str(tmp_path)},
            case.ledger,
            case.node,
            case.input_set,
            "wrong-geometry-execution",
        )


def test_result_manifest_requires_candidate_evaluation_evidence(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)
    missing_evidence = case.evaluation_identity(128)
    del missing_evidence["evaluation_evidence"]
    (tmp_path / "comparison-128.json").write_text(json.dumps({"identity": missing_evidence}))

    with pytest.raises(RuntimeError, match="missing evaluator-owned candidate evidence"):
        runner._aggregate_result_manifest(
            {"puzzle_dir": str(tmp_path)},
            case.ledger,
            case.node,
            case.input_set,
            "missing-audit-execution",
        )


def test_result_manifest_rejects_changed_milestone_evaluator_contract(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)
    mismatched = case.evaluation_identity(128)
    mismatched["evaluator"]["contract"]["dataset"]["revision"] = "different-revision"
    mismatched["reference_evaluator_contract"]["dataset"]["revision"] = "different-revision"
    (tmp_path / "comparison-128.json").write_text(json.dumps({"identity": mismatched}))

    with pytest.raises(RuntimeError, match="128-step evaluation contract differs from pre-KD"):
        runner._aggregate_result_manifest(
            {"puzzle_dir": str(tmp_path)},
            case.ledger,
            case.node,
            case.input_set,
            "mismatch-execution",
        )


def test_result_manifest_requires_reference_comparison(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)
    milestone = case.ledger.observations["short_v1_128"][case.revisions["kd_128"]]
    milestone.artifacts.pop("comparison_path")

    with pytest.raises(RuntimeError, match="produced no reference comparison"):
        runner._aggregate_result_manifest(
            {"puzzle_dir": str(tmp_path)},
            case.ledger,
            case.node,
            case.input_set,
            "missing-comparison",
        )


def test_result_manifest_rejects_mixed_candidate_evaluator_contracts(monkeypatch, tmp_path):
    case = _result_manifest_case(monkeypatch, tmp_path)
    second_architecture_id = "architecture-second"
    case.ledger.architectures[second_architecture_id] = ArchitectureCandidate(
        architecture_id=second_architecture_id,
        block_configs=[],
        mip_metrics={"parameter_ratio": 0.9},
    )
    second_materialized = "revision-second-materialized"
    second_selected = "revision-second-kd-256"
    case.ledger.revisions[second_materialized] = CandidateRevision(
        revision_id=second_materialized,
        architecture_id=second_architecture_id,
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "second-pre-kd")},
        parent_revision_id=None,
        producer_node="materialized",
    )
    case.ledger.revisions[second_selected] = CandidateRevision(
        revision_id=second_selected,
        architecture_id=second_architecture_id,
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(tmp_path / "second-step-256")},
        parent_revision_id=second_materialized,
        producer_node="kd_256",
    )
    case.ledger.observations["materialized"][second_materialized] = NodeObservation(
        node_id="materialized",
        input_revision_id=second_materialized,
        source_revision_id=second_materialized,
        output_revision_id=second_materialized,
        status="success",
    )
    second_identity = case.evaluation_identity(0)
    second_identity["architecture_id"] = second_architecture_id
    second_identity["evaluator"]["contract"]["dataset"]["revision"] = "other-revision"
    second_identity["reference_evaluator_contract"]["dataset"]["revision"] = "other-revision"
    second_comparison = tmp_path / "comparison-second-pre-kd.json"
    second_comparison.write_text(json.dumps({"identity": second_identity}))
    case.ledger.observations["pre_kd_short_v1"][second_materialized] = NodeObservation(
        node_id="pre_kd_short_v1",
        input_revision_id=second_materialized,
        source_revision_id=second_materialized,
        output_revision_id=second_materialized,
        status="success",
        artifacts={"comparison_path": str(second_comparison)},
    )
    mixed_input_set = CandidateSet.create(
        "campaign",
        "selected",
        [case.revisions["kd_256"], second_selected],
        producer_execution_identity="selected-execution",
    )

    with pytest.raises(RuntimeError, match="evaluation contract differs across candidates"):
        runner._aggregate_result_manifest(
            {"puzzle_dir": str(tmp_path)},
            case.ledger,
            case.node,
            mixed_input_set,
            "mixed-contract",
        )
