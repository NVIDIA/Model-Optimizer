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

"""Tests for the dependency-light orchestration entrypoint."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete
from puzzletron_orchestrator.config import load_experiment_config
from puzzletron_orchestrator.logging import OrchestratorLogger

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_lightweight_package_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from puzzletron_orchestrator import normalize_vllm_topology; "
            "from puzzletron_orchestrator.stages import STAGE_SPECS, semantic_stage_config; "
            "assert normalize_vllm_topology({})['gpu_count'] == 1; "
            "assert STAGE_SPECS; "
            "assert semantic_stage_config({'convert': {'mode': 'hf'}}, 'convert'); "
            "assert 'torch' not in sys.modules",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_config_import_does_not_cycle_through_post_mip() -> None:
    environment = dict(os.environ)
    # This subprocess checks cold importability, not subprocess coverage collection.
    environment.pop("COVERAGE_PROCESS_START", None)
    environment.pop("COVERAGE_FILE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path; "
            "assert callable(pipeline_config_from_path)",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_named_vllm_measurement_gpu_group_includes_data_parallelism() -> None:
    from puzzletron_orchestrator.vllm_measurements import normalize_vllm_measurements

    measurements = normalize_vllm_measurements(
        {
            "vllm_stats": {
                "enabled": True,
                "measurements": {
                    "multigpu": {
                        "prefill_seq_len": 128,
                        "generation_seq_len": 32,
                        "batch_size": 1,
                        "max_num_seqs": 1,
                        "granularity": "subblock",
                        "runtime_stats": {
                            "topology": {
                                "tensor_parallel_size": 2,
                                "data_parallel_size": 2,
                                "prefill_context_parallel_size": 2,
                                "enable_expert_parallel": True,
                                "gpu_group_size": 8,
                            }
                        },
                    }
                },
            }
        }
    )

    assert measurements["multigpu"].gpu_group_size == 8


def test_cpu_stage_command_omits_main_gpu_count(tmp_path: Path) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import StageCompatAdapter
    from puzzletron_orchestrator.schema import (
        CampaignPlan,
        ExecutionContract,
        ExecutionStrategy,
        FailurePolicy,
        RunnerEnvironment,
        StagePlanNode,
    )

    runner = RunnerEnvironment(
        kind="local",
        contract=ExecutionContract(repository=str(tmp_path), venv=".venv"),
    )
    plan = CampaignPlan(
        experiment_config_path=str(tmp_path / "experiment.yaml"),
        puzzle_dir=tmp_path / "run",
        experiment_config={},
        runner=runner,
        execution_defaults={},
        stages=(),
        contract_hash="contract",
    )
    node = StagePlanNode(
        stage_id="convert",
        strategy=ExecutionStrategy.SINGLE,
        instances=1,
        failure_policy=FailurePolicy.STRICT,
        mesh={},
        gpus_per_instance=0,
        gpus_per_node=0,
        nodes=1,
        total_gpus=0,
        exclusive=False,
        parents=(),
        distributed=False,
        resource="cpu",
    )
    adapter = StageCompatAdapter()
    item = adapter.plan(plan, node).items[0]

    attempt = adapter.command(
        plan=plan,
        node=node,
        item=item,
        attempt_id="attempt",
        runner=runner,
    )

    assert "--gpus-per-node" not in attempt.command.argv
    assert attempt.metadata["gpus_per_node"] == 0


def test_orchestrator_logger_color_modes() -> None:
    colored = io.StringIO()
    OrchestratorLogger(color="always", stream=colored).success("stage complete")
    assert "\033[32m" in colored.getvalue()
    assert "stage complete" in colored.getvalue()

    plain = io.StringIO()
    OrchestratorLogger(color="never", stream=plain).error("stage failed")
    assert "\033[" not in plain.getvalue()
    assert "stage failed" in plain.getvalue()


def test_load_experiment_config_composes_defaults_and_interpolation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "run"))
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": ["_self_"],
                "puzzle_dir": "${oc.env:RUN_ROOT,unused}",
                "pruning": {"automodel": {"parallel": {"pp": 2, "dp_shard": 4}}},
                "teacher_dir": "${puzzle_dir}/ckpts/teacher",
                "hook_class": "${get_object:package.module.Hook}",
            }
        )
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "defaults": ["/base@_global_", "_self_"],
                "pruning": {"automodel": {"parallel": {"ep": 2}}},
                "copy": "${to_path:${teacher_dir}}",
            },
            sort_keys=False,
        )
    )

    config = load_experiment_config(experiment, overrides=["pruning.automodel.parallel.pp=1"])

    assert config["puzzle_dir"] == str(tmp_path / "run")
    assert config["teacher_dir"] == str(tmp_path / "run" / "ckpts" / "teacher")
    assert config["copy"] == config["teacher_dir"]
    assert config["hook_class"] == {"__type__": "package.module.Hook"}
    assert config["pruning"]["automodel"]["parallel"] == {
        "pp": 1,
        "dp_shard": 4,
        "ep": 2,
    }
    assert config["_runtime"]["config_path"] == str(experiment)


def test_load_experiment_config_matches_hydra_scientific_number_semantics(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """\
defaults: [_self_]
bypass:
  best_val_loss: 1e+9
  training:
    learning_rate: 1e-4
    min_lr_factor: 1e-5
  schedule: [1e-4, 1e-5, \"1e-4\"]
quoted: \"1e-4\"
"""
    )

    config = load_experiment_config(experiment, overrides=["+threshold=1e-4"])

    assert config["bypass"]["best_val_loss"] == 1e9
    assert config["bypass"]["training"] == {
        "learning_rate": 1e-4,
        "min_lr_factor": 1e-5,
    }
    assert config["bypass"]["schedule"] == [1e-4, 1e-5, "1e-4"]
    assert config["quoted"] == "1e-4"
    assert config["threshold"] == 1e-4
    assert all(
        isinstance(value, float)
        for value in (
            config["bypass"]["best_val_loss"],
            config["bypass"]["training"]["learning_rate"],
            config["bypass"]["training"]["min_lr_factor"],
            config["threshold"],
        )
    )


def test_load_experiment_config_rejects_deletion_overrides(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("value: 1\n")

    with pytest.raises(ValueError, match="^Deletion overrides are not supported"):
        load_experiment_config(experiment, overrides=["~value"])


def test_load_experiment_config_distinguishes_hydra_addition_modes(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("value: 1\n")

    added = load_experiment_config(experiment, overrides=["+added.value=2"])
    with pytest.raises(ValueError, match="^Addition override already exists"):
        load_experiment_config(experiment, overrides=["+value=2"])
    with pytest.raises(ValueError, match="^Override path does not exist"):
        load_experiment_config(experiment, overrides=["missing.value=2"])
    replaced = load_experiment_config(experiment, overrides=["++value=2"])

    assert added["added"] == {"value": 2}
    assert replaced["value"] == 2


def test_convert_completeness_requires_runtime_subblock_library(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import (
        stage_is_complete,
        stage_output_patterns,
    )

    config = {
        "experiment": {"dir": str(tmp_path)},
        "puzzle_dir": str(tmp_path),
        "vllm_stats": {"enabled": True},
    }
    assert stage_output_patterns(config, "convert") == (
        "ckpts/teacher/config.json",
        "subblock_library.json",
    )
    write_terminal_manifest(tmp_path, "convert", config=config)
    teacher = tmp_path / "ckpts" / "teacher"
    teacher.mkdir(parents=True)
    (teacher / "config.json").write_text("{}")
    assert not stage_is_complete(config, "convert")
    (tmp_path / "subblock_library.json").write_text("[]\n")
    assert stage_is_complete(config, "convert")
    assert stage_output_patterns({"experiment": {"dir": str(tmp_path)}}, "convert") == (
        "ckpts/teacher/config.json",
    )


def test_non_elastic_bypass_completeness_does_not_require_dp_observations(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path), "bypass": {"elastic": False}}
    write_terminal_manifest(tmp_path, "bypass", config=config)
    history = tmp_path / "artifacts" / "bypass" / "local_kd_loss_history.json"
    history.parent.mkdir(parents=True)
    history.write_text("{}\n")

    assert stage_is_complete(config, "bypass")

    config["bypass"]["elastic"] = True
    assert not stage_is_complete(config, "bypass")
    write_terminal_manifest(tmp_path, "bypass", config=config)
    assert not stage_is_complete(config, "bypass")
    (history.parent / "dp_observations.jsonl").write_text("{}\n")
    assert stage_is_complete(config, "bypass")


def test_width_completeness_requires_success_manifest_and_complete_passes(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path)}
    output = tmp_path / "pruning" / "scores"
    pass_dir = output / "attention"
    pass_dir.mkdir(parents=True)
    (pass_dir / "args.json").write_text("{}")
    (output / "activation_passes_manifest.json").write_text(json.dumps({"passes": ["attention"]}))
    assert not stage_is_complete(config, "width_importance")

    write_terminal_manifest(
        tmp_path,
        "width_importance",
        config=config,
        outputs={"activations_log_dir": str(output)},
    )
    assert stage_is_complete(config, "width_importance")


def test_sort_completeness_rejects_early_config_and_requires_final_outputs(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path)}
    sorted_teacher = tmp_path / "ckpts" / "sorted_teacher"
    sorted_teacher.mkdir(parents=True)
    (sorted_teacher / "config.json").write_text("{}")
    assert not stage_is_complete(config, "sort")

    shard = "model-00001-of-00001.safetensors"
    (sorted_teacher / shard).write_text("weights")
    (sorted_teacher / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": shard}})
    )
    (sorted_teacher / "sorted_permutations.json").write_text('{"layer": [0]}')
    (sorted_teacher / "parallel_sort_manifest.json").write_text('{"status": "complete"}')
    assert not stage_is_complete(config, "sort")

    write_terminal_manifest(tmp_path, "sort", config=config)
    assert stage_is_complete(config, "sort")


def test_depth_completeness_requires_matching_complete_trajectory(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {
        "puzzle_dir": str(tmp_path),
        "depth_importance": {"enabled": True, "max_removals": 2},
    }
    write_terminal_manifest(tmp_path, "depth_importance", config=config)
    output = tmp_path / "depth" / "iterative"
    output.mkdir(parents=True)
    trajectory = output / "trajectory.json"
    trajectory.write_text(json.dumps({"status": "running", "max_removals": 5, "selected": [{}]}))
    assert not stage_is_complete(config, "depth_importance")

    trajectory.write_text(
        json.dumps({"status": "complete", "max_removals": 2, "selected": [{}, {}]})
    )
    assert stage_is_complete(config, "depth_importance")


def test_build_library_requires_its_own_complete_outputs(
    tmp_path: Path, write_terminal_manifest
) -> None:
    config = {"puzzle_dir": str(tmp_path)}
    write_terminal_manifest(tmp_path, "build_library", config=config)
    (tmp_path / "subblock_stats.json").write_text("{}")
    assert not stage_is_complete(config, "build_library")

    (tmp_path / "replacement_library.json").write_text("{}")
    assert not stage_is_complete(config, "build_library")
    (tmp_path / "candidate_library.json").write_text("{}")
    assert stage_is_complete(config, "build_library")


def test_build_library_completion_accepts_equivalent_loader_and_worker_configs(
    tmp_path: Path, write_terminal_manifest
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""\
defaults: [_self_]
puzzle_dir: {tmp_path}
build_library:
  enabled: true
bypass:
  best_val_loss: 1e+9
  training:
    learning_rate: 1e-4
    min_lr_factor: 1e-5
"""
    )
    controller_config = load_experiment_config(experiment)
    worker_config = deepcopy(controller_config)
    worker_config["library"] = {}

    write_terminal_manifest(tmp_path, "build_library", config=worker_config)
    for name in (
        "replacement_library.json",
        "candidate_library.json",
        "subblock_stats.json",
    ):
        (tmp_path / name).write_text("{}")

    assert stage_is_complete(controller_config, "build_library")
    controller_config["bypass"]["best_val_loss"] = 2e9
    assert not stage_is_complete(controller_config, "build_library")


def test_embedding_build_library_requires_every_width_scenario(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {
        "puzzle_dir": str(tmp_path),
        "embedding_pruning": {"enabled": True, "widths": [1024, 768]},
    }
    write_terminal_manifest(tmp_path, "build_library", config=config)
    for name in ("replacement_library.json", "candidate_library.json", "subblock_stats.json"):
        (tmp_path / name).write_text("{}")
    assert not stage_is_complete(config, "build_library")

    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "width_scenarios.json").write_text("{}")
    for width in (1024, 768):
        scenario = tmp_path / "scenarios" / f"width-{width:04d}" / "depth-00"
        (scenario / "manifests").mkdir(parents=True)
        for name in (
            "replacement_library.json",
            "candidate_library.json",
            "subblock_stats.json",
        ):
            (scenario / name).write_text("{}")
        (scenario / "scenario_manifest.json").write_text(json.dumps({"status": "complete"}))
        (scenario / "manifests" / "build_library.json").write_text(
            json.dumps({"status": "success"})
        )

    assert stage_is_complete(config, "build_library")


def test_successful_manifest_stales_when_semantic_config_changes(
    tmp_path: Path, write_terminal_manifest
) -> None:
    config = {
        "puzzle_dir": str(tmp_path),
        "convert": {"teacher_dir": "teacher-v1"},
    }
    artifact = tmp_path / "ckpts" / "teacher" / "config.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    write_terminal_manifest(tmp_path, "convert", config=config)

    assert stage_is_complete(config, "convert")
    changed = {**config, "convert": {"teacher_dir": "teacher-v2"}}
    assert not stage_is_complete(changed, "convert")


def test_tokenize_data_completeness_rejects_mismatched_cache_metadata(
    tmp_path: Path, write_terminal_manifest, write_token_cache
) -> None:
    output = tmp_path / "dataset_cache" / "train.tokens"
    config = {
        "puzzle_dir": str(tmp_path),
        "dataset_path": str(tmp_path / "dataset"),
        "convert": {"teacher_dir": str(tmp_path / "ckpts" / "teacher")},
        "tokenize_data": {
            "enabled": True,
            "caches": [
                {
                    "output": str(output),
                    "split": "train",
                    "num_samples": 1,
                    "seq_length": 8,
                    "shuffle_seed": 1,
                }
            ],
        },
    }
    receipt = write_token_cache(config, config["tokenize_data"]["caches"][0])
    write_terminal_manifest(
        tmp_path,
        "tokenize_data",
        config=config,
        outputs={"caches": [receipt]},
    )
    metadata_path = Path(receipt["metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["split"] = "validation"
    metadata_path.write_text(json.dumps(metadata))

    assert not stage_is_complete(config, "tokenize_data")

    write_token_cache(config, config["tokenize_data"]["caches"][0])
    assert stage_is_complete(config, "tokenize_data")

    config["model"] = {"trust_remote_code": True}
    assert not stage_is_complete(config, "tokenize_data")


def test_tokenize_data_completeness_requires_exact_receipts_and_cache_set(
    tmp_path: Path, write_terminal_manifest, write_token_cache
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    caches = [
        {
            "output": str(tmp_path / "dataset_cache" / f"{split}.tokens"),
            "split": split,
            "num_samples": index + 1,
            "seq_length": 3,
            "shuffle_seed": 100 + index,
        }
        for index, split in enumerate(("train", "validation"))
    ]
    config = {
        "puzzle_dir": str(tmp_path),
        "dataset_path": str(tmp_path / "dataset"),
        "convert": {"teacher_dir": str(tmp_path / "ckpts" / "teacher")},
        "tokenize_data": {"enabled": True, "caches": caches},
    }
    receipts = [write_token_cache(config, cache) for cache in caches]
    write_terminal_manifest(
        tmp_path,
        "tokenize_data",
        config=config,
        outputs={"caches": receipts},
    )

    assert stage_is_complete(config, "tokenize_data")

    manifest_path = tmp_path / "manifests" / "tokenize_data.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["caches"][0]["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest))
    assert not stage_is_complete(config, "tokenize_data")

    write_terminal_manifest(
        tmp_path,
        "tokenize_data",
        config=config,
        outputs={"caches": receipts},
    )
    Path(caches[1]["output"]).unlink()
    assert not stage_is_complete(config, "tokenize_data")


def test_stage_completeness_rejects_skip_without_reason(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path)}
    write_terminal_manifest(
        tmp_path,
        "tokenize_data",
        config=config,
        status="skipped",
        outputs={"enabled": False},
    )

    assert not stage_is_complete(config, "tokenize_data")


def test_stage_completeness_accepts_only_current_disabled_skip(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path), "aiperf": {"enabled": False}}
    write_terminal_manifest(
        tmp_path,
        "aiperf",
        config=config,
        status="skipped",
        skip_reason="disabled",
    )

    assert stage_is_complete(config, "aiperf")

    config["aiperf"]["enabled"] = True
    assert not stage_is_complete(config, "aiperf")


def test_vllm_completeness_requires_nonempty_canonical_stats(
    tmp_path: Path, write_terminal_manifest
) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {"puzzle_dir": str(tmp_path), "vllm_stats": {"enabled": True}}
    write_terminal_manifest(tmp_path, "vllm_stats", config=config)
    summary = tmp_path / "artifacts" / "vllm_stats" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}")
    assert not stage_is_complete(config, "vllm_stats")

    (tmp_path / "subblock_stats.json").write_text("[]")
    assert not stage_is_complete(config, "vllm_stats")
    (tmp_path / "subblock_stats.json").write_text('[{"args": {}}]')
    assert stage_is_complete(config, "vllm_stats")


def test_legacy_mip_and_evaluation_completeness(tmp_path: Path, write_terminal_manifest) -> None:
    from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete

    config = {
        "puzzle_dir": str(tmp_path),
        "mip": {"profiles": {"params": {}, "runtime": {}}},
        "zero_shot_evaluation": {"profile_ids": ["params", "runtime"]},
    }
    write_terminal_manifest(tmp_path, "mip", config=config)
    write_terminal_manifest(tmp_path, "zero_shot_evaluation", config=config)
    params_grid = tmp_path / "mip" / "profiles" / "params" / "mip_grid.json"
    params_grid.parent.mkdir(parents=True)
    params_grid.write_text("{}")
    assert stage_is_complete(config, "mip")

    params_eval = (
        tmp_path
        / "artifacts"
        / "zero_shot_evaluation"
        / "profiles"
        / "params"
        / "evaluation_summary.json"
    )
    params_eval.parent.mkdir(parents=True)
    params_eval.write_text("{}")
    assert not stage_is_complete(config, "zero_shot_evaluation")
    runtime_eval = (
        tmp_path
        / "artifacts"
        / "zero_shot_evaluation"
        / "profiles"
        / "runtime"
        / "evaluation_summary.json"
    )
    runtime_eval.parent.mkdir(parents=True)
    runtime_eval.write_text("{}")
    assert stage_is_complete(config, "zero_shot_evaluation")


def test_qwen_production_dry_run_uses_worker_python(tmp_path: Path) -> None:
    output = tmp_path / "dry-run.json"
    environment = dict(os.environ)
    environment["PUZZLETRON_RUN_ROOT"] = str(tmp_path / "qwen-moe")
    with output.open("w") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "examples/puzzletron/orchestrate.py",
                "--experiment",
                "examples/puzzletron/configs/families/qwen3_5/qwen3p6_35b_a3b/runs/production.yaml",
                "--runner",
                "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml",
                "--execution",
                "examples/puzzletron/configs/orchestration/qwen_moe/execution.production.yaml",
                "--stage",
                "full",
                "--dry-run",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr

    payload = yaml.safe_load(output.read_text())
    assert all(submission["argv"][0] in {"python", "bash"} for submission in payload["submissions"])
    vllm = [item for item in payload["submissions"] if item["stage_id"] == "vllm_stats"]
    assert len(vllm) == 1
    assert all(item["gpus"] == 8 and item["nodes"] == 1 for item in vllm)
