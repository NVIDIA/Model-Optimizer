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

"""Unit tests for LAQ PTQ recipes."""

from pathlib import Path

import pytest

from modelopt.recipe.loader import load_recipe

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_qat" / "configs" / "quantize"

_LAQ_RECIPES = [
    ("nvfp4_laq_post-mse_init-fp8_kv.yml", ["post"]),
    ("nvfp4_laq_pre_post-mse_init-fp8_kv.yml", ["pre", "post"]),
]


def _load_laq_recipe(filename):
    return load_recipe(CONFIGS_DIR / filename).quantize


@pytest.mark.parametrize(("filename", "expected_learnable"), _LAQ_RECIPES)
def test_laq_recipe_loads_with_expected_algorithm(filename, expected_learnable):
    quantize = _load_laq_recipe(filename)
    algorithm = quantize.algorithm

    assert algorithm["method"] == "laq"
    assert algorithm["learnable_amax"] == expected_learnable
    assert algorithm["tied_amax"] is False
    assert algorithm["quantize_pre_scale"] is False
    assert algorithm["scale_algorithm"] == {"method": "mse", "fp8_scale_sweep": True}


@pytest.mark.parametrize(("filename", "_"), _LAQ_RECIPES)
def test_laq_recipe_resolves_modular_quant_cfg(filename, _):
    quantize = _load_laq_recipe(filename)
    entries = {entry.quantizer_name: entry for entry in quantize.quant_cfg}

    weight_cfg = entries["*weight_quantizer"].cfg.model_dump(exclude_unset=True)
    input_cfg = entries["*input_quantizer"].cfg.model_dump(exclude_unset=True)
    kv_cfg = entries["*[kv]_bmm_quantizer"].cfg.model_dump(exclude_unset=True)

    assert weight_cfg["block_sizes"]["type"] == "static"
    assert weight_cfg["num_bits"] == (2, 1)
    assert input_cfg["block_sizes"]["type"] == "dynamic"
    assert input_cfg["num_bits"] == (2, 1)
    assert kv_cfg["num_bits"] == (4, 3)
