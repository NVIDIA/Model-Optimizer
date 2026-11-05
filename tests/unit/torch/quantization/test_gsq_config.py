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
from transformers import AutoModelForCausalLM, Gemma2Config, Qwen2MoeConfig

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
        ("temperature", 0),
        ("scale_lr", float("nan")),
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
        mtq.calibrate(model, {"method": "gsq"}, lambda m: m(torch.ones(1, 4, 64)))
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field", ["reconstruction", "steps", "max_tokens", "eval_every", "gradient_accumulation_steps"]
)
def test_gsq_rejects_removed_linear_options(field):
    with pytest.raises(ValueError):
        GSQCalibConfig(**{field: 1})


@pytest.mark.parametrize("field", ["checkpoint_dir", "export_dir"])
def test_gsq_rejects_restart_and_streaming_export(field, tmp_path):
    with pytest.raises(ValueError):
        GSQCalibConfig(layerwise={"enable": True, field: str(tmp_path)})


@pytest.mark.parametrize("config_type", [Gemma2Config, Qwen2MoeConfig])
def test_gsq_rejects_unsupported_dense_and_moe_decoders_without_mutation(config_type):
    config = config_type(
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    quant_config = load_recipe("general/ptq/nvfp4_gsq-max").quantize.model_dump()
    mtq.quantize(model, {**quant_config, "algorithm": None})
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        mtq.calibrate(
            model,
            quant_config["algorithm"],
            lambda m: m(torch.ones(1, 8, dtype=torch.long), use_cache=False),
        )
    assert before.keys() == model.state_dict().keys()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert all(not hasattr(layer, "_layerwise_calib") for layer in model.model.layers)
