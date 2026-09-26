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

"""Unit tests for EAGLE + LoRA co-training (eagle_base_lora feature)."""

from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama
from peft.tuners.lora import LoraLayer

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.eagle.default_config import default_eagle_config

TINY_EAGLE_CFG = {
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 2,
    "use_last_layernorm": True,
    "use_aux_hidden_state": False,
    "eagle_aux_hidden_state_layer_ids": [],
}

EAGLE_LORA_CONFIG = {
    "eagle_architecture_config": {**default_eagle_config, **TINY_EAGLE_CFG},
    "eagle_base_lora": True,
    "eagle_base_lora_rank": 4,
    "eagle_base_lora_alpha": 8.0,
    "eagle_base_lora_target_modules": ["q_proj", "v_proj"],
    "eagle_base_lora_preservation_loss_weight": 0.1,
}


@pytest.fixture
def lora_eagle_model():
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, mode=[("eagle", deepcopy(EAGLE_LORA_CONFIG))])
    return model


def test_lora_layers_injected(lora_eagle_model):
    """LoRA adapters should be present in the base model after conversion."""
    lora_layers = [m for m in lora_eagle_model._base_model.modules() if isinstance(m, LoraLayer)]
    assert len(lora_layers) > 0, "No LoRA layers found in base model"


def test_trainable_params(lora_eagle_model):
    """Only LoRA and eagle_module params should be trainable; base model weights frozen."""
    for name, param in lora_eagle_model.named_parameters():
        is_lora = "lora_" in name
        is_eagle = "eagle_module" in name
        if is_lora or is_eagle:
            assert param.requires_grad, f"Expected {name} to be trainable"
        else:
            assert not param.requires_grad, f"Expected {name} to be frozen"


def test_forward_returns_loss(lora_eagle_model):
    """Forward pass should return a scalar loss containing preservation + eagle components."""
    lora_eagle_model.train()
    seq_len = 8
    input_ids = torch.randint(0, lora_eagle_model.config.vocab_size, (1, seq_len))
    output = lora_eagle_model(input_ids=input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.loss.ndim == 0, "Loss should be a scalar"
    assert output.loss.item() > 0


def test_eagle_offline_incompatible():
    """eagle_base_lora=True should raise when combined with eagle_offline=True."""
    model = get_tiny_llama(num_hidden_layers=4)
    config = deepcopy(EAGLE_LORA_CONFIG)
    config["eagle_offline"] = True
    with pytest.raises(ValueError, match="eagle_base_lora is incompatible with eagle_offline"):
        mtsp.convert(model, mode=[("eagle", config)])


def test_export_lora_artifacts(lora_eagle_model, tmp_path):
    """export() should produce lora_adapter_model.safetensors and lora_adapter_config.json."""
    export_dir = tmp_path / "eagle_export"
    lora_eagle_model.get_exporter().export(export_dir)

    assert (export_dir / "model.safetensors").exists(), "Eagle model weights missing"
    assert (export_dir / "adapter_model.safetensors").exists(), "LoRA weights missing"
    assert (export_dir / "adapter_config.json").exists(), "LoRA config missing"
