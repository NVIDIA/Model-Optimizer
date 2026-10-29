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

"""Exercise frozen recipe assignments through the native AutoQuant solver."""

import copy

import pytest
import torch
import yaml

import modelopt.torch.quantization as mtq
from modelopt.recipe import (
    ModelOptAutoQuantizeRecipe,
    freeze_auto_quantize_assignment,
    plan_auto_quantize,
)
from modelopt.torch.quantization.algorithms import QuantRecipe
from modelopt.torch.quantization.config import QuantizeConfig


@pytest.mark.parametrize("corruption", [None, "missing_group", "overlap", "attributes", "method"])
def test_assignment_roundtrip_retains_fixed_and_selected_methods(corruption):
    torch.manual_seed(42)
    fixed = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    fixed["algorithm"] = "mse"
    selected = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    selected["algorithm"] = ["max", "max"]
    recipe = ModelOptAutoQuantizeRecipe(
        quantize=fixed,
        auto_quantize={
            "constraints": {"effective_bits": 8},
            "module_search_spaces": [
                {"module_name_patterns": ["1"], "candidate_formats": [selected]}
            ],
        },
    )
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    _, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8},
        fixed_quantization_config=fixed,
        module_search_spaces=[{"module_name_patterns": ["1"], "quantization_formats": [selected]}],
        data_loader=[torch.randn(2, 8)],
        forward_step=lambda m, x: m(x),
        loss_func=lambda output, batch: output.square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
    )
    plan = plan_auto_quantize(recipe)
    provenance = {"source": "source-digest", "inputs": "data-digest", "code": "code-digest"}
    frozen = freeze_auto_quantize_assignment(state, plan, provenance)
    payload = frozen.as_dict()
    assert payload["requested_constraints"] == plan.as_dict()["constraints"]
    groups = {g["module_names"][0]: g for g in payload["groups"]}
    assert yaml.safe_load(groups["0"]["configuration_yaml"])["algorithm"] == "mse"
    assert groups["0"]["is_fixed"]
    assert yaml.safe_load(groups["1"]["configuration_yaml"])["algorithm"] == ["max", "max"]
    for group in groups.values():
        restored = QuantizeConfig(**yaml.safe_load(group["configuration_yaml"]))
        assert (
            restored.model_dump_json()
            == state["best"]["recipe"][group["name"]].checkpoint_signature
        )
    digest = frozen.digest
    payload["groups"].clear()
    assert frozen.digest == digest
    before_weights = {name: value.clone() for name, value in model.named_parameters()}
    if corruption == "missing_group":
        state["best"]["recipe"].pop("1.quant_recipe")
    elif corruption == "overlap":
        state["candidate_stats"]["1.quant_recipe"]["module_names"] = ["0"]
    elif corruption == "attributes":
        state["candidate_stats"]["1.quant_recipe"]["quantizer_attrs"]["1"] = ("absent",)
    elif corruption == "method":
        changed = copy.deepcopy(selected)
        changed["algorithm"] = "max"
        state["best"]["recipe"]["1.quant_recipe"] = QuantRecipe(changed, name="changed")
    if corruption is not None:
        with pytest.raises(ValueError):
            freeze_auto_quantize_assignment(state, plan, provenance)
    else:
        assert freeze_auto_quantize_assignment(state, plan, provenance).digest == digest
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before_weights[name], rtol=0, atol=0)
