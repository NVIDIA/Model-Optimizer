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

"""Keep the Flash IQ2 recipe scoped to routed SequentialMLP experts."""

import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe


def _mlp():
    module = nn.Module()
    module.linear_fc1 = nn.Linear(256, 512, bias=False)
    module.linear_fc2 = nn.Linear(256, 256, bias=False)
    return module


def test_flash_iq2_recipe_targets_only_routed_experts():
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.decoder.layers:
        layer.inner_layer = nn.Module()
        inner = layer.inner_layer
        inner.self_attention = nn.Module()
        inner.self_attention.linear_proj = nn.Linear(256, 256, bias=False)
        inner.mlp = nn.Module()
        inner.mlp.experts = nn.Module()
        inner.mlp.experts.local_experts = nn.ModuleList([_mlp(), _mlp()])
        inner.mlp.shared_experts = _mlp()
        inner.mlp.router = nn.Linear(256, 2, bias=False)
    model.mtp = nn.Module()
    model.mtp.mlp = model.decoder.layers[0].inner_layer.mlp.__class__()
    model.mtp.mlp.experts = nn.Module()
    model.mtp.mlp.experts.local_experts = nn.ModuleList([_mlp()])
    config = load_recipe(
        "models/deepseek-ai/DeepSeek-V4-Flash-0731/ptq/iq2_xxs_experts-gptq"
    ).quantize.model_dump()
    algorithm = config.pop("algorithm")
    assert algorithm["method"] == "gptq"
    assert algorithm["importance_weighted"]
    assert algorithm["block_size"] == 256
    assert algorithm["perc_damp"] == 0.3
    assert algorithm["layerwise"]["enable"]
    mtq.quantize(model, {**config, "algorithm": None})
    active = []
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer") and module.is_enabled:
            active.append(name)
            assert name.startswith("decoder.")
            assert ".mlp.experts.local_experts." in name
            assert module.num_bits == "iq2_xxs"
        elif name.endswith("input_quantizer"):
            assert not module.is_enabled
    assert len(active) == 8
