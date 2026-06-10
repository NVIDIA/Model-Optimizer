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

import json

import pytest
from safetensors import safe_open

pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_5_dir,
    create_tiny_qwen3_5vl_dir,
    create_tiny_qwen3_dir,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.puzzletron as mtpz
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config


def test_convert_anymodel(tmp_path):
    input_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    output_dir = tmp_path / "qwen3-0.6b-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3")

    descriptor = mtpz.anymodel.ModelDescriptorFactory.get("qwen3")
    with mtpz.anymodel.deci_x_patcher(descriptor):
        _ = AutoModelForCausalLM.from_pretrained(output_dir)


def test_convert_anymodel_qwen3_5_text_preserves_mtp(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    input_dir = create_tiny_qwen3_5_dir(tmp_path, with_tokenizer=True, with_mtp=True)
    output_dir = tmp_path / "qwen3_5-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3_5_text")

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.0.norm.weight"] == "subblocks_safetensors/mtp.safetensors"
    with safe_open(output_dir / "subblocks_safetensors" / "mtp.safetensors", framework="pt") as f:
        assert "mtp.0.norm.weight" in list(f.keys())

    descriptor = mtpz.anymodel.ModelDescriptorFactory.get("qwen3_5_text")
    with mtpz.anymodel.deci_x_patcher(descriptor):
        _ = AutoModelForCausalLM.from_pretrained(output_dir)


def test_convert_anymodel_qwen3_5_vl_sets_text_layer_config(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    input_dir = create_tiny_qwen3_5vl_dir(tmp_path, with_tokenizer=True)
    output_dir = tmp_path / "qwen3_5vl-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3_5")

    config = load_model_config(output_dir)
    assert len(config.block_configs) == config.text_config.num_hidden_layers
    assert config.text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert any(
        filename.endswith("vision_encoding.safetensors")
        for filename in index["weight_map"].values()
    )
