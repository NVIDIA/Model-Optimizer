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

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from puzzletron_orchestrator.adapters.stage_compat import stage_is_complete
from puzzletron_orchestrator.config import load_experiment_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_orchestrator_help_explains_first_run_contracts() -> None:
    result = subprocess.run(
        [sys.executable, "examples/puzzletron/orchestrate.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "Experiment YAML: model, data, enabled stages, and output directory." in help_text
    assert "Runner YAML: where and how worker jobs run" in help_text
    assert "Execution YAML: how each stage runs" in help_text
    assert "requires its parent artifacts" in help_text
    assert "jobs keep running" in help_text


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

    config = load_experiment_config(experiment, overrides=["++threshold=1e-4"])

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


def test_load_experiment_config_applies_deletion_overrides(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("quotas:\n  retained-95: 1\n  retained-85: 1\n")

    config = load_experiment_config(experiment, overrides=["~quotas.retained-95"])

    assert config["quotas"] == {"retained-85": 1}


@pytest.mark.parametrize("override", ["~value=1", "~missing.value"])
def test_load_experiment_config_rejects_invalid_deletion_overrides(
    tmp_path: Path,
    override: str,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("value: 1\n")

    with pytest.raises(ValueError, match="^Deletion override"):
        load_experiment_config(experiment, overrides=[override])


def test_load_experiment_config_distinguishes_hydra_addition_modes(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("value: 1\n")

    added = load_experiment_config(experiment, overrides=["added.value=2"])
    replaced = load_experiment_config(experiment, overrides=["++value=2"])
    created = load_experiment_config(experiment, overrides=["++created.value=3"])

    assert added["added"] == {"value": 2}
    assert replaced["value"] == 2
    assert created["created"] == {"value": 3}


@pytest.mark.parametrize("override", ["+experiment.dir=other"])
def test_load_experiment_config_rejects_unsupported_hydra_operators(
    tmp_path: Path,
    override: str,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("experiment:\n  dir: run\n")

    with pytest.raises(ValueError, match="not supported by the dependency-light controller"):
        load_experiment_config(experiment, overrides=[override])


def test_load_experiment_config_rejects_unknown_interpolation(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("experiment:\n  dir: ${missing.value}\n")

    with pytest.raises(ValueError, match="Unknown config interpolation 'missing.value'"):
        load_experiment_config(experiment)


def test_orchestrator_cli_reports_config_errors_without_traceback(tmp_path: Path) -> None:
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "test", "partition_name": "gpu"},
                }
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/puzzletron/orchestrate.py",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--runner",
            str(runner),
            "--execution",
            str(tmp_path / "execution.yaml"),
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert "cannot build campaign plan" in result.stderr
    assert "partition" in result.stderr
    assert "Traceback" not in result.stderr


def test_orchestrator_cli_rejects_unresolved_runner_template(tmp_path: Path) -> None:
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "REPLACE_WITH_SLURM_ACCOUNT", "partition": "gpu"},
                }
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/puzzletron/orchestrate.py",
            "--experiment",
            str(tmp_path / "experiment.yaml"),
            "--runner",
            str(runner),
            "--execution",
            str(tmp_path / "execution.yaml"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert "unresolved REPLACE_WITH_ placeholders" in result.stderr
    assert "runner.slurm.account" in result.stderr
    assert "Traceback" not in result.stderr


def test_orchestrator_cli_reports_dry_run_adapter_errors_without_traceback(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    runner = tmp_path / "runner.yaml"
    execution = tmp_path / "execution.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "experiment": {"dir": str(tmp_path / "run")},
                "embedding_pruning": {"enabled": True, "widths": []},
                "replacement_scoring": {"enabled": True},
            }
        )
    )
    runner.write_text(
        yaml.safe_dump(
            {
                "runner": {
                    "kind": "slurm",
                    "slurm": {"account": "test", "partition": "gpu"},
                }
            }
        )
    )
    execution.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "defaults": {"gpus_per_node": 1},
                    "stages": {
                        "replacement_scoring": {
                            "strategy": "persistent_pool",
                            "instances": 1,
                        }
                    },
                }
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "examples/puzzletron/orchestrate.py",
            "--experiment",
            str(experiment),
            "--runner",
            str(runner),
            "--execution",
            str(execution),
            "--stage",
            "replacement_scoring",
            "--dry-run",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 2
    assert "embedding replacement scoring requires at least one width" in result.stderr
    assert "Traceback" not in result.stderr


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
