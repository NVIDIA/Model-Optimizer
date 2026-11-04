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


"""GSQ rejects unsupported formats and invalid options before calibration."""

from copy import deepcopy

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.config import GSQCalibConfig
from modelopt.torch.quantization.gsq import gsq


@pytest.mark.parametrize("recipe", ["gptq", "max", "mse"])
def test_weight_only_gsq_recipes(recipe):
    config = load_recipe(f"general/ptq/nvfp4_gsq-{recipe}").quantize
    algorithm = GSQCalibConfig(**config.algorithm)
    assert algorithm.initializer == {"method": recipe}
    assert algorithm.layerwise.enable
    assert algorithm.reconstruction == "staged"
    assert algorithm.layerwise.activation_cache_device == "cpu"
    model = torch.nn.Linear(64, 16)
    mtq.quantize(model, {**config.model_dump(), "algorithm": None})
    assert model.weight_quantizer.is_nvfp4_static
    assert model.weight_quantizer.block_sizes[-1] == 16
    assert not model.input_quantizer.is_enabled


@pytest.mark.parametrize(
    "config", [mtq.FP8_DEFAULT_CFG, mtq.W4A16_NVFP4_CFG, mtq.NVFP4_DEFAULT_CFG]
)
def test_gsq_rejects_incompatible_quantizers_without_mutation(config):
    model = torch.nn.Linear(64, 16)
    mtq.quantize(model, {**deepcopy(config), "algorithm": None})
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        gsq(model, lambda m: m(torch.ones(4, 64)))
    assert before.keys() == model.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("temperature", 0),
        ("scale_lr", float("nan")),
        ("max_tokens", 0),
        ("qk_steps", 0),
        ("epochs", 0),
    ],
)
def test_gsq_invalid_optimization_options(field, value):
    with pytest.raises(ValueError):
        GSQCalibConfig(**{field: value})


@pytest.mark.parametrize("initializer", ["awq_lite", "awq_full", "local_hessian", "gsq", "lsq"])
def test_gsq_rejects_unsupported_initializers(initializer):
    with pytest.raises(ValueError):
        GSQCalibConfig(initializer={"method": initializer})


def test_gsq_rejects_weight_shared_with_unquantized_linear():
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    model[1].weight = model[0].weight
    config = load_recipe("general/ptq/nvfp4_gsq-gptq").quantize.model_dump()
    config["quant_cfg"].append({"quantizer_name": "1.weight_quantizer", "enable": False})
    mtq.quantize(model, {**config, "algorithm": None})
    with pytest.raises(ValueError, match="tied"):
        gsq(model, lambda m: m(torch.ones(4, 64)))


def test_staged_gsq_rejects_unsupported_layout_without_mutation():
    model = torch.nn.Linear(64, 16)
    config = load_recipe("general/ptq/nvfp4_gsq-gptq").quantize.model_dump()
    mtq.quantize(model, {**config, "algorithm": None})
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        mtq.calibrate(
            model, {"method": "gsq", "reconstruction": "staged"}, lambda m: m(torch.ones(1, 4, 64))
        )
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
