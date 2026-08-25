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

"""Regression tests for portable Puzzletron example configuration."""

import re
from pathlib import Path

import pytest
import yaml

from puzzletron_orchestrator.compiler import load_execution_config, load_runner_config
from puzzletron_setup import WORKER_REPOSITORY_PLACEHOLDER, WORKER_VENV_PLACEHOLDER
from puzzletron_setup.v2.defaults import load_defaults

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SLURM_RUNNER_CONFIGS = (
    "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
    "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml",
    "examples/puzzletron/configs/orchestration/qwen3p5_0p8b/runner.slurm.yaml",
)
NAMED_SLURM_RUNNER_CONFIGS = SLURM_RUNNER_CONFIGS[1:]
NEMOTRON3_NANO_30B_MODEL_CONFIG = (
    "examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/model.yaml"
)
QWEN3P5_0P8B_MODEL_CONFIG = "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
QWEN3P5_9B_MODEL_CONFIG = "examples/puzzletron/configs/families/qwen3_5/qwen3p5_9b/model.yaml"
QWEN3P6_35B_A3B_MODEL_CONFIG = (
    "examples/puzzletron/configs/families/qwen3_5/qwen3p6_35b_a3b/model.yaml"
)


def test_slurm_runner_example_is_portable() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml"
    slurm = load_runner_config(path)
    assert slurm.contract.repository == WORKER_REPOSITORY_PLACEHOLDER
    assert slurm.contract.venv == WORKER_VENV_PLACEHOLDER
    assert slurm.contract.container is None
    assert slurm.contract.container_mounts is None
    assert not slurm.contract.prerun_commands
    assert slurm.slurm is not None
    assert slurm.slurm.account.startswith("REPLACE_WITH_")
    assert slurm.slurm.partition == (
        "REPLACE_WITH_PRIMARY_SLURM_PARTITION,REPLACE_WITH_ALTERNATE_SLURM_PARTITION"
    )
    assert slurm.slurm.log_dir == "puzzle_runs/logs"


def test_baremetal_runner_example_is_portable() -> None:
    baremetal = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.baremetal.example.yaml"
    )
    assert baremetal.contract.repository == WORKER_REPOSITORY_PLACEHOLDER
    assert baremetal.contract.venv == WORKER_VENV_PLACEHOLDER
    assert baremetal.contract.setup_env is None
    assert baremetal.baremetal is not None
    hostnames = [host.hostname for host in baremetal.baremetal.hosts]
    assert hostnames
    assert baremetal.baremetal.rendezvous_host in hostnames
    assert all(hostname.startswith("REPLACE_WITH_") for hostname in hostnames)


def test_qwen_slurm_runner_preserves_portable_environment_contract() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml"
    runner = load_runner_config(path)

    contract_values = (
        runner.contract.repository,
        runner.contract.venv,
        runner.contract.container,
        runner.contract.container_mounts,
    )
    assert contract_values[:2] == (
        WORKER_REPOSITORY_PLACEHOLDER,
        WORKER_VENV_PLACEHOLDER,
    )
    assert all(value and value.startswith("REPLACE_WITH_") for value in contract_values[2:])
    assert runner.contract.prerun_commands
    assert all("REPLACE_WITH_" in command for command in runner.contract.prerun_commands)
    assert runner.slurm is not None
    assert runner.slurm.account.startswith("REPLACE_WITH_")
    assert runner.slurm.partition.startswith("REPLACE_WITH_")


@pytest.mark.parametrize("relative_path", SLURM_RUNNER_CONFIGS)
def test_checked_in_slurm_runners_only_emit_generic_partition(
    relative_path: str,
) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())

    assert "partition" in payload["runner"]["slurm"]
    assert not any(key.startswith("partition_") for key in payload["runner"]["slurm"])


@pytest.mark.parametrize("relative_path", NAMED_SLURM_RUNNER_CONFIGS)
def test_named_slurm_runners_keep_logs_below_the_campaign_root(relative_path: str) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())

    assert "log_dir" not in payload["runner"]["slurm"]


def test_execution_example_is_loadable() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.example.yaml"

    execution = load_execution_config(path)

    assert set(execution) >= {"defaults", "stages"}


def test_setup_defaults_example_is_portable() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/setup/defaults.example.yaml"

    defaults = load_defaults(path)

    contract = defaults["infrastructure"]["execution_contract"]
    assert contract["repository"] == WORKER_REPOSITORY_PLACEHOLDER
    assert contract["venv"] == WORKER_VENV_PLACEHOLDER
    assert contract["container"] is None
    assert contract["container_mounts"] is None
    assert not contract["prerun_commands"]

    slurm = defaults["infrastructure"]["runner"]["slurm"]
    assert "account" not in slurm
    assert slurm["partition"] is None


def test_model_examples_use_public_hugging_face_identities() -> None:
    paths = (
        NEMOTRON3_NANO_30B_MODEL_CONFIG,
        QWEN3P5_0P8B_MODEL_CONFIG,
        QWEN3P5_9B_MODEL_CONFIG,
        QWEN3P6_35B_A3B_MODEL_CONFIG,
    )

    for relative_path in paths:
        config = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())

        assert config["input_hf_model_path"] == config["model_info"]["hf_repo"]
        assert not config["input_hf_model_path"].startswith("REPLACE_WITH_")
        assert re.fullmatch(r"[0-9a-f]{40}", config["model_info"]["hf_revision"])


def test_qwen_dense_model_metadata_matches_public_checkpoint() -> None:
    path = REPOSITORY_ROOT / QWEN3P5_9B_MODEL_CONFIG

    model_info = yaml.safe_load(path.read_text())["model_info"]

    assert model_info["model_type"] == "qwen3_5"
    assert model_info["architectures"] == ["Qwen3_5ForConditionalGeneration"]
