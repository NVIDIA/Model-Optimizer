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

from puzzletron_orchestrator.recipe_config import ROUTES, recipe_template, site_template

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
NEMOTRON3_NANO_30B_MODEL_CONFIG = (
    "examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/model.yaml"
)
QWEN3P5_0P8B_MODEL_CONFIG = "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
QWEN3P5_9B_MODEL_CONFIG = "examples/puzzletron/configs/families/qwen3_5/qwen3p5_9b/model.yaml"
QWEN3P6_35B_A3B_MODEL_CONFIG = (
    "examples/puzzletron/configs/families/qwen3_5/qwen3p6_35b_a3b/model.yaml"
)


def test_site_example_is_the_only_portable_environment_contract() -> None:
    checked_in = yaml.safe_load(
        (REPOSITORY_ROOT / "examples/puzzletron/configs/site.example.yaml").read_text()
    )

    assert checked_in == site_template()
    environment = checked_in["site"]["environment"]
    assert environment["repository"].startswith("REPLACE_WITH_")
    assert environment["venv"].startswith("REPLACE_WITH_")
    assert environment["container"] is None
    assert environment["container_mounts"] is None
    assert not environment["prerun_commands"]
    assert checked_in["site"]["slurm"]["account"].startswith("REPLACE_WITH_")
    assert checked_in["site"]["slurm"]["partition"].startswith("REPLACE_WITH_")
    assert checked_in["resources"]["multinode"] == {
        "mode": "per_attempt",
        "gpus_per_node": 8,
        "max_nodes": 64,
    }


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.route_id)
def test_recipe_template_is_small_and_has_no_inheritance(route) -> None:
    recipe = recipe_template(
        model=route.model,
        workflow=route.workflow,
        mode=route.mode,
    )

    expected = {
        "schema_version",
        "name",
        "model",
        "workflow",
        "mode",
        "run_root",
        "resource_profile",
    }
    if route.requires_data:
        expected.add("data")
        assert recipe["data"]["path"].startswith("REPLACE_WITH_")
        assert recipe["data"]["revision"].startswith("REPLACE_WITH_")
    assert set(recipe) == expected
    assert (recipe["model"], recipe["workflow"], recipe["mode"]) == (
        route.model,
        route.workflow,
        route.mode,
    )


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
