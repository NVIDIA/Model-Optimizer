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

"""Shared helpers for accelerate-offloaded and layerwise-calibration quantization tests."""

import copy

import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM


def make_tiny_llama_and_inputs(tmp_path, num_hidden_layers=3):
    """Tiny LLaMA checkpoint dir + its config + a GPU token batch sized for its vocab."""
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_hidden_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()
    return tiny_llama_dir, config, inputs


def make_cpu_offloaded_model(tmp_path, num_hidden_layers=3):
    """Tiny LLaMA with layer 0 offloaded to CPU via accelerate.

    Returns ``(model, config, tiny_llama_dir, inputs)``; ``inputs`` is a GPU token batch
    sized for the model's vocab.
    """
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=num_hidden_layers)
    config = AutoConfig.from_pretrained(tiny_llama_dir)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = {
        n: 0
        for n, m in model.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"

    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)
    inputs = torch.randint(0, config.vocab_size, (1, 4)).cuda()
    return model, config, tiny_llama_dir, inputs


def make_layerwise_cfg(base_cfg):
    """Copy of ``base_cfg`` with layerwise calibration enabled on its algorithm field."""
    cfg = copy.deepcopy(base_cfg)
    algo = cfg.get("algorithm", "max")
    if isinstance(algo, str):
        cfg["algorithm"] = {"method": algo, "layerwise": {"enable": True}}
    else:
        algo["layerwise"] = {"enable": True}
    return cfg


def make_layerwise_checkpoint_cfg(base_cfg, checkpoint_dir):
    """``make_layerwise_cfg`` plus a ``layerwise.checkpoint_dir``."""
    cfg = make_layerwise_cfg(base_cfg)
    cfg["algorithm"]["layerwise"]["checkpoint_dir"] = checkpoint_dir
    return cfg
