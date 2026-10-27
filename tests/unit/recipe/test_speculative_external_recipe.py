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

"""Schema tests for the ``speculative_external`` recipe."""

from modelopt.recipe.config import ModelOptExternalDraftRecipe


def test_recipe_validates_without_a_data_path_so_overrides_stay_reachable():
    """The loader validates the recipe file *before* splicing CLI overrides.

    A schema-level raise on the missing data path would therefore fire on the shipped
    recipe, which carries none, and make `data.offline_data_path=...` on the command line
    impossible to pass. The gate lives in main.py, after the overrides are in.
    """
    bare = ModelOptExternalDraftRecipe.model_validate({"metadata": {"description": "d"}})
    assert bare.data.mode == "online"

    ok = ModelOptExternalDraftRecipe.model_validate(
        {"metadata": {"description": "d"}, "data": {"offline_data_path": "/tmp/hidden"}}
    )
    assert ok.data.mode == "offline"
