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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.stages.future as future_stages
from examples.puzzletron import run_post_mip_node as post_mip_entrypoint
from modelopt.torch.puzzletron.post_mip import runner
from modelopt.torch.puzzletron.post_mip.records import ArtifactKind
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


@pytest.mark.parametrize(
    "profile",
    ["qwen35_vlm_realworldqa", "qwen35_vlm_e2e_full_eval"],
)
def test_worker_entrypoint_registers_configured_vlm_evaluation_profile(monkeypatch, profile):
    # Keep the examples-layer VLM dependencies out of core test collection.
    from examples.puzzletron.evaluation.vlm import post_mip as vlm_post_mip

    calls = []
    monkeypatch.setattr(vlm_post_mip, "register_profiles", lambda: calls.append(True))

    post_mip_entrypoint._register_evaluation_profiles({"post_mip": None})
    post_mip_entrypoint._register_evaluation_profiles(
        {
            "post_mip": {
                "flows": {
                    "null_flow": None,
                    "invalid_flow": [],
                    "params": {
                        "nodes": {
                            "null_node": None,
                            "invalid_node": [],
                            "generic_eval": {
                                "type": "downstream_evaluation",
                                "config": None,
                            },
                            "invalid_config": {
                                "type": "downstream_evaluation",
                                "config": [],
                            },
                            "invalid_list_profile": {
                                "type": "downstream_evaluation",
                                "config": {"profile": []},
                            },
                            "invalid_mapping_profile": {
                                "type": "downstream_evaluation",
                                "config": {"profile": {}},
                            },
                            "checkpoint_eval": {
                                "type": "downstream_evaluation",
                                "config": {"profile": profile},
                            },
                        }
                    },
                }
            }
        }
    )

    assert calls == [True]


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
                metrics={},
                raw_artifacts={},
            )
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
    assert result["metrics"] == {}


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
        result_path.write_text("{}")
        return {"metrics": {"ifeval.accuracy": score}, "result_path": str(result_path)}

    monkeypatch.setattr(runner, "run_lmms_eval_checkpoint", fake_evaluate)
    node = SimpleNamespace(
        node_id="full_benchmarks",
        config={
            "config": {
                "tasks": ["ifeval"],
                "reference_checkpoint": str(reference),
                "recorded_observation": {
                    "repeat_count": 2,
                    "metrics": {
                        "candidate.ifeval.accuracy": 0.35,
                        "reference.ifeval.accuracy": 0.5,
                    },
                },
            }
        },
    )
    source = SimpleNamespace(
        architecture_id="architecture",
        artifact_kind=ArtifactKind.CHECKPOINT,
        artifact={"checkpoint": str(candidate)},
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
        "metrics": {
            "candidate.ifeval.accuracy": 0.35,
            "reference.ifeval.accuracy": 0.5,
        },
        "difference_from_recorded": {
            "candidate.ifeval.accuracy": pytest.approx(0.05),
            "reference.ifeval.accuracy": 0.0,
        },
    }
