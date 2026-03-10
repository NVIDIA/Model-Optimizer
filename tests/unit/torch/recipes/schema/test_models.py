# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for recipe schema validation."""

import pytest
import yaml

from modelopt.torch.recipes.schema.models import RecipeConfig


def test_minimal_recipe():
    raw = {"version": "1.0", "quantization": {"preset": "fp8"}}
    recipe = RecipeConfig.model_validate(raw)
    assert recipe.quantization.preset == "fp8"


def test_exclusive_sections():
    raw = {
        "quantization": {"preset": "fp8"},
        "auto_quantize": {"effective_bits": 4.5, "formats": [{"preset": "fp8"}]},
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        RecipeConfig.model_validate(raw)


def test_preset_or_custom():
    raw = {
        "quantization": {
            "preset": "fp8",
            "weights": {"format": "int8"},
        }
    }
    with pytest.raises(ValueError, match="Cannot specify both"):
        RecipeConfig.model_validate(raw)


def test_qat_requires_training():
    raw = {"quantization": {"mode": "qat", "preset": "fp8"}}
    with pytest.raises(ValueError, match="QAT mode requires"):
        RecipeConfig.model_validate(raw)


def test_all_example_recipes_valid(examples_dir):
    for yaml_file in sorted(examples_dir.rglob("*.yaml")):
        if "experiments" in yaml_file.parts:
            continue  # skip sweep/experiment configs
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        recipe = RecipeConfig.model_validate(raw)
        assert recipe.version == "1.0", f"Failed: {yaml_file.name}"
