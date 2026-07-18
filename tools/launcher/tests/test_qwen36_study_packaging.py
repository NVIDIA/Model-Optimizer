# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused packaging checks for the Qwen3.6 FP8 granularity study."""

import os

import launch
import pytest
import yaml

EXPECTED_QWEN36_IMAGE = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/viraatc/"
    "tensorrt-llm-release-1.3.0rc11.sqsh"
)
EXPECTED_QWEN36_PYDEPS_MOUNT = (
    "/lustre/fsw/portfolios/coreai/users/weimingc/qwen36_ptq/pydeps:"
    "/qwen36-pydeps:ro"
)
EXPECTED_QWEN36_STUDY_MOUNT = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_numerics_edge/users/weimingc/"
    "qwen36_fp8_granularity_study:/study"
)


def test_launcher_project_declares_build_backend():
    """Managed uv runs install this checkout instead of resolving a global command."""
    with open(os.path.join(launch.LAUNCHER_DIR, "pyproject.toml")) as file:
        pyproject = file.read()

    assert "[build-system]" in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject


def test_managed_source_root_overrides_installed_launcher_package(monkeypatch, tmp_path):
    """MCP launches package the exact managed checkout, not wheel-only assets."""
    source_root = tmp_path / "Model-Optimizer"
    launcher_dir = source_root / "tools" / "launcher"
    launcher_dir.mkdir(parents=True)
    (launcher_dir / "launch.py").touch()
    (launcher_dir / "common").mkdir()
    (source_root / "modelopt").mkdir()
    monkeypatch.setenv("MODELOPT_MCP_SOURCE_ROOT", str(source_root))

    assert launch._resolve_launcher_dir("/installed/modelopt_launcher") == str(launcher_dir)


def test_incomplete_managed_source_root_fails_closed(monkeypatch, tmp_path):
    """A stale managed-source contract must not silently create an incomplete tarball."""
    monkeypatch.setenv("MODELOPT_MCP_SOURCE_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="complete ModelOpt checkout"):
        launch._resolve_launcher_dir("/installed/modelopt_launcher")


def test_qwen36_study_is_packaged_at_runner_relative_path():
    """The focused study subtree is shipped where its YAML script path expects it."""
    study_path = os.path.join(
        launch.LAUNCHER_DIR,
        "modules/Model-Optimizer/experimental/qwen36_fp8_granularity_study",
    )
    assert study_path in launch._include_pattern
    index = launch._include_pattern.index(study_path)
    assert launch._relative_path[index] == launch.LAUNCHER_DIR
    runner = os.path.join(study_path, "launcher", "run_study.sh")
    assert os.path.isfile(runner)
    assert os.access(runner, os.X_OK)
    assert os.path.isfile(os.path.join(study_path, "study.py"))


def test_qwen36_pipelines_have_staging_plus_four_sequential_candidates():
    """Each model pipeline has one staging task and four full-node GPU slots."""
    launcher_dir = os.path.join(
        launch.LAUNCHER_DIR,
        "modules/Model-Optimizer/experimental/qwen36_fp8_granularity_study/launcher",
    )
    expected_candidates = [
        "per_tensor_fp8,per_tensor_fp8_weight_only_control",
        "block128_static_weight_only",
        "block128_dynamic_w8a8_research,block128_dynamic_weight_only_control",
        "mxfp8,mxfp8_weight_only_control",
    ]
    for filename in ("qwen3.6-35b-a3b_aws-cmh.yaml", "qwen3.6-27b_aws-cmh.yaml"):
        with open(os.path.join(launcher_dir, filename)) as file:
            config = yaml.safe_load(file)

        tasks = [config["pipeline"][f"task_{index}"] for index in range(5)]
        assert tasks[0]["args"][0] == "stage"
        assert [task["args"][-1] for task in tasks[1:]] == expected_candidates
        assert all(
            task["slurm_config"]["container"] == EXPECTED_QWEN36_IMAGE
            for task in tasks
        )
        assert all(
            EXPECTED_QWEN36_PYDEPS_MOUNT in task["slurm_config"]["container_mounts"]
            for task in tasks
        )
        assert all(
            EXPECTED_QWEN36_STUDY_MOUNT in task["slurm_config"]["container_mounts"]
            for task in tasks
        )
        assert tasks[0]["slurm_config"]["partition"] == "cpu_datamover"
        assert tasks[0]["slurm_config"]["gpus_per_node"] == 0
        for task in tasks[1:]:
            slurm = task["slurm_config"]
            assert slurm["account"] == "coreai_numerics_edge"
            assert slurm["partition"] == "batch_long"
            assert slurm["nodes"] == 1
            assert slurm["ntasks_per_node"] == 1
            assert slurm["gpus_per_node"] == 4
            assert slurm["time"] == "24:00:00"


def test_qwen36_runner_stages_and_enforces_offline_dataset_snapshot():
    """GPU tasks use only the exact, verified 1,056-row local JSONL prefix."""
    runner = os.path.join(
        launch.LAUNCHER_DIR,
        "modules/Model-Optimizer/experimental/qwen36_fp8_granularity_study/launcher/run_study.sh",
    )
    with open(runner) as file:
        script = file.read()

    assert "readonly DATASET_SOURCE_ROW_COUNT=1056" in script
    assert '"cnn_dailymail_train_first_1056.jsonl"' in script
    assert '"source_row_count": dataset_source_row_count' in script
    assert '"sha256": dataset_hash.hexdigest()' in script
    assert "export HF_DATASETS_OFFLINE=1" in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert "export TRANSFORMERS_OFFLINE=1" in script
    assert "export STUDY_CONTAINER_IMAGE" in script
    assert "export QWEN36_PYDEPS" in script
    assert 'PYTHON_BIN="$(command -v python3 || command -v python || true)"' in script
    assert "Qwen3_5MoeForConditionalGeneration" in script
    assert "Qwen3_5ForConditionalGeneration" in script
    assert "get_cuda_ext_mx(raise_if_failed=True)" in script
    assert "invalidate_staging_manifest" in script
    assert "require_staging_manifest" in script
    assert 'previous="${manifest}.previous.${SLURM_JOB_ID:-$$}"' in script
    early_guard = script.index("# Invalidate stale staging state before any checks that can fail.")
    python_lookup = script.index('PYTHON_BIN="$(command -v python3 || command -v python || true)"')
    runtime_validation = script.rindex('case "${MODE}" in')
    assert early_guard < python_lookup < runtime_validation
    guard_block = script[early_guard:python_lookup]
    assert guard_block.index("invalidate_staging_manifest") < guard_block.index(
        "require_staging_manifest"
    )
    assert 'TORCH_EXTENSIONS_DIR="${REMOTE_STUDY_ROOT}/cache/torch_extensions/${MODEL_SLUG}"' in script
    assert '"status": "launcher_preflight"' in script
    assert "temporary.replace(path)" in script
    assert '--calib-dataset "${dataset_path}"' in script
    assert '--eval-dataset "${dataset_path}"' in script
    assert "--activation-mse-size 32" in script
