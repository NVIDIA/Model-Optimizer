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

"""Recipe-level automatic planning without model allocation."""

import copy
import hashlib

import pytest

import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptAutoQuantizeRecipe, load_recipe, plan_auto_quantize
from modelopt.torch.quantization.algorithms import QuantRecipe
from modelopt.torch.quantization.mode import GPTQModeDescriptor


def test_plan_preserves_group_methods_and_reuses_budget_statistics():
    iq = load_recipe("general/ptq/iq2_xs").quantize.model_dump()
    iq["algorithm"] = [{"method": "imatrix", "layerwise": {"enable": False}}, "max"]
    recipe = ModelOptAutoQuantizeRecipe(
        quantize=mtq.FP8_DEFAULT_CFG,
        auto_quantize={
            "constraints": {"effective_bits": 3.0},
            "module_search_spaces": [
                {
                    "module_name_patterns": ["*.experts.*"],
                    "candidate_formats": [iq, mtq.NVFP4_DEFAULT_CFG],
                    "allow_no_quant": False,
                }
            ],
            "disabled_layers": ["*lm_head*"],
            "cost_excluded_layers": ["*vision*"],
        },
    )
    before = recipe.model_dump()
    plan = plan_auto_quantize(recipe)
    plan.require_supported()
    snapshot = plan.as_dict()
    signature = QuantRecipe(iq, name="any_display_name").checkpoint_signature
    iq_id = hashlib.sha256(signature.encode()).hexdigest()
    iq_plan = next(f for f in snapshot["search"]["formats"] if f["id"] == iq_id)
    assert iq_plan["signature"] == signature
    assert [a["settings"]["method"] for a in iq_plan["algorithms"]] == ["imatrix", "max"]
    assert snapshot["search"]["module_search_spaces"][0]["format_ids"][0] == iq_id
    assert (
        len(snapshot["search"]["formats"]) == 3
    )  # Includes the fixed FP8 baseline, no BF16 worker.
    assert recipe.model_dump() == before
    snapshot["search"]["formats"].clear()
    assert len(plan.as_dict()["search"]["formats"]) == 3
    other_budget = recipe.model_copy(deep=True)
    other_budget.auto_quantize.constraints.effective_bits = 3.25
    assert (
        plan_auto_quantize(other_budget).as_dict()["search_digest"]
        == plan.as_dict()["search_digest"]
    )
    changed = recipe.model_copy(deep=True)
    changed.auto_quantize.score_size += 1
    assert plan_auto_quantize(changed).as_dict()["search_digest"] != plan.as_dict()["search_digest"]


@pytest.mark.parametrize(
    ("method", "layerwise"), [("gptq", True), ("max", True), ("local_hessian", False)]
)
def test_plan_refuses_undeclared_estimators_without_replacing_algorithms(method, layerwise):
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    algorithm = {"method": method, "layerwise": {"enable": layerwise}}
    if method == "gptq":
        algorithm.update(block_size=256, perc_damp=0.3)
    config["algorithm"] = algorithm
    recipe = ModelOptAutoQuantizeRecipe(
        auto_quantize={"constraints": {"effective_bits": 8}, "candidate_formats": [config]}
    )
    plan = plan_auto_quantize(recipe)
    with pytest.raises(ValueError):
        plan.require_supported()
    candidate = plan.as_dict()["search"]["formats"][0]
    assert candidate["final_layerwise"] is layerwise
    settings = candidate["algorithms"][0]["settings"]
    assert settings["method"] == method
    assert settings["layerwise"]["enable"] is layerwise
    if method == "gptq":
        assert (settings["block_size"], settings["perc_damp"]) == (256, 0.3)


def test_plan_uses_declared_layerwise_requirement(monkeypatch):
    config = load_recipe("general/ptq/iq2_xs").quantize.model_dump()
    config["algorithm"] = {"method": "gptq", "layerwise": {"enable": False}}
    recipe = ModelOptAutoQuantizeRecipe(
        auto_quantize={"constraints": {"effective_bits": 3}, "candidate_formats": [config]}
    )
    plan_auto_quantize(recipe).require_supported()
    monkeypatch.setattr(GPTQModeDescriptor, "_requires_final_layerwise", True)
    with pytest.raises(ValueError):
        plan_auto_quantize(recipe).require_supported()
