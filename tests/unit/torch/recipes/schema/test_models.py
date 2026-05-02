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

"""Tests for recipe schema validation.

Most schema tests are now YAML fixture-driven (see test_recipe_fixtures.py).
This file keeps tests that don't fit the fixture pattern (e.g., file iteration).
"""

from pathlib import Path

import yaml

from modelopt.torch.recipes.schema.models import RecipeConfig

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


def test_all_fixture_recipes_parseable():
    """All non-error YAML fixtures parse as valid RecipeConfig."""
    for yaml_file in sorted(FIXTURES_DIR.glob("*.yaml")):
        if yaml_file.stem.startswith("error_"):
            continue
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        recipe_dict = {k: v for k, v in raw.items() if k != "_test"}
        recipe = RecipeConfig.model_validate(recipe_dict)
        assert recipe.version == "1.0", f"Failed: {yaml_file.name}"
