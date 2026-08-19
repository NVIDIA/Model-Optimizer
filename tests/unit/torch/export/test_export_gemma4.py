# SPDX-FileCopyrightText: Copyright 2026 Google LLC and contributors
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

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from modelopt.torch.export.model_utils import MODEL_NAME_TO_TYPE
from modelopt.torch.export.tensorrt_llm_utils import MODEL_NAME_TO_HF_ARCH_MAP, convert_to_tensorrt_llm_config
from modelopt.torch.export.layer_utils import build_decoder_config
from modelopt.torch.export.model_config import ModelConfig, DecoderLayerConfig


class DummyRMSNorm(nn.Module):
    """Mock RMSNorm class to pass modelopt's is_layernorm() validation."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(10))
        self.bias = nn.Parameter(torch.zeros(10))
        self.variance_epsilon = 1e-6


class DummyLinear(nn.Module):
    """Mock Linear class to pass modelopt's is_linear() validation."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16, 16))
        self.bias = None


class DummyConfig:
    """Mock Config container for nested attention attributes."""
    def __init__(self):
        self.num_attention_heads = 8
        self.kv_channels = 2
        self.num_query_groups = 8


class DummyAttention(nn.Module):
    """Mock Attention class to pass modelopt's is_attention() validation."""
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.q = DummyLinear()
        self.k = DummyLinear()
        self.v = DummyLinear()
        self.dense = DummyLinear()


class DummyGemma4Layer(nn.Module):
    """Mock Gemma 4 decoder layer module with RMSNorm children."""
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = DummyRMSNorm()
        self.pre_feedforward_layernorm = DummyRMSNorm()
        self.post_feedforward_layernorm = DummyRMSNorm()
        self.self_attention = DummyAttention()


def test_gemma4_model_type_registration():
    """Verify that Gemma4 is correctly mapped to 'gemma4' model type."""
    assert MODEL_NAME_TO_TYPE["Gemma4"] == "gemma4"


def test_gemma4_hf_arch_registration():
    """Verify that gemma4 is correctly mapped to Gemma4ForCausalLM."""
    assert MODEL_NAME_TO_HF_ARCH_MAP["gemma4"] == "Gemma4ForCausalLM"


def test_gemma4_layernorm_config_mappings():
    """Verify build_decoder_config correctly maps Gemma 4 layernorm blocks."""
    layer = DummyGemma4Layer()
    
    # Build decoder config for gemma4
    config = build_decoder_config(
        layer,
        model_metadata_config={"training_tensor_parallel": 1},
        decoder_type="gemma4"
    )
    
    assert config.post_layernorm is not None
    assert config.post_layernorm.layernorm_type == "RmsNorm"
    assert config.post_layernorm.eps == 1e-6
    
    assert config.pre_feedforward_layernorm is not None
    assert config.pre_feedforward_layernorm.layernorm_type == "RmsNorm"
    assert config.pre_feedforward_layernorm.eps == 1e-6

    assert config.post_feedforward_layernorm is not None
    assert config.post_feedforward_layernorm.layernorm_type == "RmsNorm"
    assert config.post_feedforward_layernorm.eps == 1e-6
