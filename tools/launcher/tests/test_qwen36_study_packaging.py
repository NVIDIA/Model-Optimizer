# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused packaging checks for the Qwen3.6 FP8 granularity study."""

import os

import launch
import yaml


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
        assert tasks[0]["slurm_config"]["partition"] == "cpu_datamove"
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
    assert '"status": "launcher_preflight"' in script
    assert "temporary.replace(path)" in script
    assert '--calib-dataset "${dataset_path}"' in script
    assert '--eval-dataset "${dataset_path}"' in script
    assert "--activation-mse-size 32" in script
