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

from pathlib import Path

import yaml
from puzzletron_orchestrator.compiler import load_runner_config

from puzzletron_setup.v2.defaults import load_defaults

REPOSITORY_ROOT = Path(__file__).parents[4]


def test_runner_examples_use_repository_relative_defaults() -> None:
    slurm = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml"
    )
    assert slurm.contract.repository == "."
    assert slurm.contract.venv == ".venv"
    assert slurm.contract.container is None
    assert slurm.contract.container_mounts is None
    assert not slurm.contract.prerun_commands
    assert slurm.slurm is not None
    assert slurm.slurm.account == "REPLACE_WITH_SLURM_ACCOUNT"
    assert slurm.slurm.partition_cpu is None

    baremetal = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.baremetal.example.yaml"
    )
    assert baremetal.contract.repository == "."
    assert baremetal.contract.venv == ".venv"
    assert baremetal.contract.setup_env is None
    assert baremetal.baremetal is not None
    assert baremetal.baremetal.rendezvous_host == "REPLACE_WITH_PRIMARY_HOST"
    assert [host.hostname for host in baremetal.baremetal.hosts] == [
        "REPLACE_WITH_PRIMARY_HOST",
        "REPLACE_WITH_SECONDARY_HOST",
    ]


def test_qwen_slurm_runner_preserves_portable_environment_contract() -> None:
    runner = load_runner_config(
        REPOSITORY_ROOT
        / "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml"
    )

    assert runner.contract.repository == "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_CHECKOUT"
    assert runner.contract.venv == "REPLACE_WITH_WORKER_VISIBLE_MODELOPT_VENV"
    assert runner.contract.container == "REPLACE_WITH_SLURM_CONTAINER_IMAGE"
    assert (
        runner.contract.container_mounts
        == "REPLACE_WITH_HOST_PATH:REPLACE_WITH_CONTAINER_PATH"
    )
    assert runner.contract.prerun_commands == (
        "source REPLACE_WITH_SITE_SETUP_SCRIPT",
        "export VLLM_ROOT=REPLACE_WITH_WORKER_VISIBLE_VLLM_CHECKOUT",
        "export AUTOMODEL_ROOT=REPLACE_WITH_WORKER_VISIBLE_AUTOMODEL_CHECKOUT",
    )
    assert runner.slurm is not None
    assert runner.slurm.account == "REPLACE_WITH_SLURM_ACCOUNT"
    assert runner.slurm.partition_cpu is None


def test_execution_example_is_valid_yaml() -> None:
    path = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.example.yaml"

    config = yaml.safe_load(path.read_text())

    assert config["execution"]["defaults"]["gpus_per_node"] == 8


def test_setup_defaults_are_portable_and_valid() -> None:
    path = REPOSITORY_ROOT / "nv-internal/puzzletron_defaults.example.yaml"

    defaults = load_defaults(path)

    assert defaults["infrastructure"]["execution_contract"] == {
        "repository": ".",
        "venv": ".venv",
        "container": None,
        "container_mounts": None,
        "prerun_commands": [],
    }


def test_model_examples_label_runnable_hugging_face_identities() -> None:
    paths = (
        "examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/model.yaml",
        "examples/puzzletron/configs/families/qwen3_5/qwen3p5_9b/model.yaml",
        "examples/puzzletron/configs/families/qwen3_5/qwen3p6_35b_a3b/model.yaml",
    )

    for relative_path in paths:
        content = (REPOSITORY_ROOT / relative_path).read_text()
        config = yaml.safe_load(content)

        assert config["input_hf_model_path"] == config["model_info"]["hf_repo"]
        assert not config["input_hf_model_path"].startswith("REPLACE_WITH_")
        assert "Runnable public Hugging Face repository ID" in content
        assert "Setup-generated bundles replace it" in content


def test_active_examples_use_portable_value_shapes() -> None:
    slurm_paths = (
        "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
        "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml",
    )

    for relative_path in slurm_paths:
        runner = load_runner_config(REPOSITORY_ROOT / relative_path)
        assert not Path(runner.contract.repository).is_absolute()
        assert not Path(runner.contract.venv).is_absolute()
        assert runner.slurm is not None
        assert runner.slurm.account.startswith("REPLACE_WITH_")

    baremetal = load_runner_config(
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.baremetal.example.yaml"
    )
    assert baremetal.baremetal is not None
    assert baremetal.baremetal.rendezvous_host.startswith("REPLACE_WITH_")
    assert all(host.hostname.startswith("REPLACE_WITH_") for host in baremetal.baremetal.hosts)

    defaults_paths = sorted((REPOSITORY_ROOT / "nv-internal").glob("*_defaults*.yaml"))
    assert [path.name for path in defaults_paths] == ["puzzletron_defaults.example.yaml"]


def test_optional_yaml_values_are_explicit() -> None:
    paths = (
        "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
        "nv-internal/puzzletron_defaults.example.yaml",
    )

    for relative_path in paths:
        content = (REPOSITORY_ROOT / relative_path).read_text()
        assert "partition_cpu: null" in content
        assert "container: null" in content
        assert "container_mounts: null" in content


def test_required_example_values_use_visible_placeholders() -> None:
    slurm_paths = (
        "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
        "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml",
    )

    for relative_path in slurm_paths:
        content = (REPOSITORY_ROOT / relative_path).read_text()
        assert "account: REPLACE_WITH_SLURM_ACCOUNT" in content

    defaults = (REPOSITORY_ROOT / "nv-internal/puzzletron_defaults.example.yaml").read_text()
    assert "# account: REPLACE_WITH_SLURM_ACCOUNT" in defaults

    baremetal = (
        REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.baremetal.example.yaml"
    ).read_text()
    assert baremetal.count("REPLACE_WITH_PRIMARY_HOST") == 2
    assert baremetal.count("REPLACE_WITH_SECONDARY_HOST") == 1


def test_execution_contract_examples_explain_runnable_and_optional_values() -> None:
    slurm_paths = (
        "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
        "nv-internal/puzzletron_defaults.example.yaml",
    )

    for relative_path in slurm_paths:
        content = (REPOSITORY_ROOT / relative_path).read_text()
        assert "checkout visible on every worker" in content
        assert "Sourced as <venv>/bin/activate" in content
        assert "accepted by the cluster's srun container plugin" in content
        assert "/data:/data,/models:/models" in content
        assert "before virtualenv activation" in content
