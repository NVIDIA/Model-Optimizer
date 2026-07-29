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
    runner_paths = (
        "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml",
        "examples/puzzletron/configs/orchestration/qwen_moe/runner.slurm.yaml",
    )

    for relative_path in runner_paths:
        runner = load_runner_config(REPOSITORY_ROOT / relative_path)
        assert runner.contract.repository == "."
        assert runner.contract.venv == ".venv"
        assert runner.contract.container is None
        assert runner.contract.container_mounts is None
        assert not runner.contract.prerun_commands


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


def test_nemotron_model_uses_hugging_face_identity() -> None:
    path = (
        REPOSITORY_ROOT
        / "examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/model.yaml"
    )

    config = yaml.safe_load(path.read_text())

    assert config["input_hf_model_path"] == config["model_info"]["hf_repo"]
