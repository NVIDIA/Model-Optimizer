# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities for exporting LLM models to ONNX."""

import json
import os
import time
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, DynamicCache


class RopeType(Enum):
    """Rope type enum."""

    K_NONE = 0
    K_ROPE_ROTATE_GPTJ = 1
    K_ROPE_ROTATE_NEOX = 2
    K_MROPE = 3


class ModelLoader:
    """A class to handle HuggingFace model loading and configuration."""

    def __init__(self, hf_model_path: str, config_path: str):
        """Initialize the ModelLoader."""
        self.config_path = config_path
        self.hf_model_path = hf_model_path
        self.model_type = self.get_model_type()
        self.hf_model = None
        self.rope_type = RopeType.K_ROPE_ROTATE_NEOX

    def get_model_type(self):
        """Get model type from config file."""
        with open(self.config_path) as f:
            return json.load(f).get("model_type")

    def load_model(self, trust_remote_code: bool = False) -> AutoModelForCausalLM:
        """Load HuggingFace model based on model type."""
        print(f"Loading HF model from {self.hf_model_path} with model type {self.model_type}")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
        )

        return self.hf_model.eval().cuda()  # type: ignore[attr-defined]

    def get_rope_type(self):
        """Get rope type."""
        return self.rope_type


class WrapperModelForCausalLM(torch.nn.Module):
    """Wrapper Model to ensure all models have the same I/O."""

    def __init__(self, model):
        """Initialize the WrapperModelForCausalLM."""
        super().__init__()
        try:
            self.model = model.model
        except Exception:
            self.model = model
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, input_ids: torch.Tensor | None, past_key_values: tuple):
        """Forward pass."""
        # Convert tuple cache to DynamicCache for models that require it (e.g., Qwen3)
        cache = DynamicCache(config=self.config)
        cache.key_cache = [kv[0] for kv in past_key_values]
        cache.value_cache = [kv[1] for kv in past_key_values]
        past_key_values = cache

        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        hidden_states = outputs[0]
        past_key_values = outputs.past_key_values.to_legacy_cache()
        logits = self.lm_head(hidden_states)
        return logits, past_key_values


def llm_to_onnx(model, output_dir, extra_inputs={}, extra_dyn_axes={}):
    """Export the WrapperModelForCausalLM to ONNX with fixed I/O names and shape definitions and save to `output_dir`.

    Parameters:
        model: torch.Module
        output_dir: str, the output_dir of the original ONNX.
        extra_inputs: dict, append additional inputs after kv_cache. Usually for VL models
        extra_dyn_axes: dict. Usually for VL models
    """
    start_time = time.time()
    config = model.config
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    hidden_size_per_layer = hidden_size // num_attention_heads

    dummy_bs = 1
    dummy_len = 10
    dummy_input_ids = torch.randint(100, (dummy_bs, dummy_len), dtype=torch.int64).cuda()
    input_names = ["input_ids"]
    output_names = ["logits"]
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
    dummy_kv_cache = ()
    for i in range(num_layers):
        dummy_k = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer), dtype=torch.float16
        ).cuda()
        dummy_v = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer), dtype=torch.float16
        ).cuda()
        dummy_kv_cache = (*dummy_kv_cache, (dummy_k, dummy_v))
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present_key_values.{i}.key", f"present_key_values.{i}.value"])
        input_dynamic_axes = {0: "batch_size", 2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.key"] = input_dynamic_axes
        dynamic_axes[f"past_key_values.{i}.value"] = input_dynamic_axes

    torch_to_onnx(
        model,
        (dummy_input_ids, {"past_key_values": dummy_kv_cache, **extra_inputs}),
        output_dir,
        "model.onnx",
        input_names=input_names + list(extra_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes | extra_dyn_axes,
    )

    end_time = time.time()
    print(
        f"Native ONNX Export from torch completed in {end_time - start_time}s. ONNX file is saved to {output_dir}."
    )


def torch_to_onnx(model, inputs, onnx_dir, onnx_name, input_names, output_names, dynamic_axes):
    """Export the model to ONNX."""
    os.makedirs(onnx_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            f"{onnx_dir}/{onnx_name}",
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )
