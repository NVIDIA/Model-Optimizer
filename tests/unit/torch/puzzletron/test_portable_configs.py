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

import yaml

from puzzletron_orchestrator.compiler import load_execution_config, load_runner_config
from puzzletron_setup.v2.defaults import load_defaults

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_slurm_runner_example_is_portable() -> None:
    slurm = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml"
    )
    assert not Path(slurm.contract.repository).is_absolute()
    assert not Path(slurm.contract.venv).is_absolute()
    assert slurm.contract.container is None
    assert slurm.contract.container_mounts is None
    assert not slurm.contract.prerun_commands
    assert slurm.slurm is not None
    assert slurm.slurm.account.startswith("REPLACE_WITH_")
    assert slurm.slurm.partition_cpu is None


def test_baremetal_runner_example_is_portable() -> None:
    baremetal = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.baremetal.example.yaml"
    )
    assert not Path(baremetal.contract.repository).is_absolute()
    assert not Path(baremetal.contract.venv).is_absolute()
    assert baremetal.contract.setup_env is None
    assert baremetal.baremetal is not None
    hostnames = [host.hostname for host in baremetal.baremetal.hosts]
    assert hostnames
    assert baremetal.baremetal.rendezvous_host in hostnames
    assert all(hostname.startswith("REPLACE_WITH_") for hostname in hostnames)


def test_qwen_slurm_runner_preserves_portable_environment_contract() -> None:
    runner = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml"
    )

    contract_values = (
        runner.contract.repository,
        runner.contract.venv,
        runner.contract.container,
        runner.contract.container_mounts,
    )
    assert all(value and value.startswith("REPLACE_WITH_") for value in contract_values)
    assert runner.contract.prerun_commands
    assert all("REPLACE_WITH_" in command for command in runner.contract.prerun_commands)
    assert runner.slurm is not None
    assert runner.slurm.account.startswith("REPLACE_WITH_")
    assert runner.slurm.partition_cpu is None


def test_execution_example_is_loadable() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.example.yaml"

    execution = load_execution_config(path)

    assert set(execution) >= {"defaults", "stages"}


def test_setup_defaults_example_is_portable() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/setup/defaults.example.yaml"

    defaults = load_defaults(path)

    contract = defaults["infrastructure"]["execution_contract"]
    assert not Path(contract["repository"]).is_absolute()
    assert not Path(contract["venv"]).is_absolute()
    assert contract["container"] is None
    assert contract["container_mounts"] is None
    assert not contract["prerun_commands"]

    slurm = defaults["infrastructure"]["runner"]["slurm"]
    assert "account" not in slurm
    assert slurm["partition_cpu"] is None


def test_model_examples_use_public_hugging_face_identities() -> None:
    paths = (
        "examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/model.yaml",
        "examples/puzzletron/configs/families/qwen3_5/qwen3p5_9b/model.yaml",
        "examples/puzzletron/configs/families/qwen3_5/qwen3p6_35b_a3b/model.yaml",
    )

    for relative_path in paths:
        config = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())

        assert config["input_hf_model_path"] == config["model_info"]["hf_repo"]
        assert not config["input_hf_model_path"].startswith("REPLACE_WITH_")
        assert re.fullmatch(r"[0-9a-f]{40}", config["model_info"]["hf_revision"])
