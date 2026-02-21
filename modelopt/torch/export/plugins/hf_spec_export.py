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

"""Modify state_dict and config for exporting speculative decoding in official format."""

import re
from copy import copy

import torch
import torch.nn as nn

from .hf_spec_configs import kimik2_eagle_template_config, llama_eagle_template_config

ALL_SPEC_MODES = ["eagle"]

LLAMA_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
        "fc.weight",
    },
    "optional": {"d2t", "lm_head.weight"},
}

KIMIK2_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.kv_a_layernorm.weight",
        "layers.0.self_attn.q_a_layernorm.weight",
        "layers.0.self_attn.q_a_proj.weight",
        "layers.0.self_attn.q_b_proj.weight",
        "layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "layers.0.self_attn.kv_b_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
        "fc.weight",
    },
    "optional": {
        "d2t",
        "lm_head.weight",
    },
}


def has_spec_opt(model: nn.Module):
    """Check if the model has speculative decoding optimization."""
    opt_modes = getattr(model, "_modelopt_state", [])
    return any(mode[0] in ALL_SPEC_MODES for mode in opt_modes)


def has_quant_opt(model: nn.Module):
    """Check if the model has quantization optimization."""
    opt_modes = getattr(model, "_modelopt_state", [])
    return any(mode[0] == "quantize" for mode in opt_modes)


class EagleExporter:
    """Draft model exporter for Eagle."""

    def __init__(self, model: nn.Module, dtype: torch.dtype | None = None):
        """Initialize the EagleExporter."""
        self.model = model
        self.eagle_decoder_type = model.eagle_config.eagle_decoder_type
        self.num_hidden_layers = model.eagle_config.num_hidden_layers
        if has_quant_opt(model):
            from ..unified_export_hf import _export_transformers_checkpoint

            self.state_dict, self.hf_quant_config = _export_transformers_checkpoint(model, dtype)
        else:
            self.state_dict, self.hf_quant_config = model.state_dict(), None

    def _check_valid_sd(self, export_sd: dict):
        """Check the export state dict is valid, otherwise raise Exception."""
        expected_keys_single_layer = {
            "llama": LLAMA_EAGLE_SINGLE_LAYER,
            "kimik2": KIMIK2_EAGLE_SINGLE_LAYER,
        }[self.eagle_decoder_type]
        # Check that export sd has required keys
        if self.num_hidden_layers == 1:
            for key in expected_keys_single_layer["required"]:
                assert key in export_sd, f"Missing required key: {key}"
        else:
            for key in expected_keys_single_layer["required"]:
                assert key in export_sd, f"Missing required key: {key}"
            for i in range(1, self.num_hidden_layers):
                for key in expected_keys_single_layer["required"] - {
                    "layers.0.hidden_norm.weight",
                    "layers.0.input_layernorm.weight",
                    "norm.weight",
                    "fc.weight",
                }:
                    assert key.replace("layers.0", f"layers.{i}") in export_sd, (
                        f"Missing required key: {key}"
                    )

        # Check that export sd has no unexpected keys
        allowed_keys_single_layer = (
            expected_keys_single_layer["required"] | expected_keys_single_layer["optional"]
        )
        if self.num_hidden_layers == 1:
            for key in export_sd:
                assert key in allowed_keys_single_layer, f"Unexpected key: {key}"
        else:
            for key in export_sd:
                assert re.sub(r"layers\.\d+\.", "", key) in {
                    k.replace("layers.0", "") for k in allowed_keys_single_layer
                }, f"Unexpected key: {key}"

    def extract_state_dict(self):
        """Extract the state dict of the draft model in deployment format."""
        export_sd = {}
        for key in self.state_dict:
            if "eagle_module" in key or "lm_head" in key:
                export_key = key.replace("eagle_module.", "")
                export_sd[export_key] = copy(self.state_dict[key])
        # Use base model's lm head if draft model doesn't have one
        if "lm_head.weight" not in export_sd:
            export_sd["lm_head.weight"] = self.state_dict["lm_head.weight"]

        self._check_valid_sd(export_sd)

        return export_sd

    def export_config(self, model):
        """Export config.json in deployment format."""
        template_config: dict = {
            "llama": llama_eagle_template_config,
            "kimik2": kimik2_eagle_template_config,
        }[model.eagle_config.eagle_decoder_type]
        template_config = copy(template_config)

        def _get_config_from_draft_or_base(key: str, model: nn.Module):
            if getattr(model._draft_model_config, key, None) is not None:
                return getattr(model._draft_model_config, key)
            elif getattr(model.config, key, None) is not None:
                return getattr(model.config, key)
            else:
                return None

        for key in template_config:
            value = template_config[key]
            if isinstance(value, dict):
                # for eagle config, we find it in model.eagle_config
                for sub_key in value:
                    if value[sub_key] is None:
                        value[sub_key] = _get_config_from_draft_or_base(sub_key, model)
            elif value is None:
                # First, we try to load fron eagle config.
                new_value = _get_config_from_draft_or_base(key, model)
                # If the value is a torch.dtype, we convert to string for serialization.
                if isinstance(new_value, torch.dtype):
                    new_value = str(new_value).replace("torch.", "")
                template_config[key] = new_value

        if self.hf_quant_config is not None:
            template_config["quantization_config"] = self.hf_quant_config

        return template_config

    def export_quant_config(self):
        """Export hf_quant_coinfig.json."""
        return copy(self.hf_quant_config)


class EagleMedusaExporter(EagleExporter):
    """Draft model exporter for EagleMedusa."""

    def __init__(self, model: nn.Module, dtype: torch.dtype | None = None):
        """Initialize the EagleMedusaExporter."""
        super().__init__(model, dtype)
        self.parallel_draft_step = model.eagle_config.parallel_draft_step
        self.parallel_draft_heads_num_layers = model.eagle_config.parallel_draft_heads_num_layers
        # NOTE: tmp: bypassing format check for parallel draft
        self._check_valid_sd = lambda *args, **kwargs: None

    def extract_state_dict(self):
        """Extract the state dict of the draft model in deployment format."""
        export_sd = super().extract_state_dict()
        if self.parallel_draft_step <= 1:
            return export_sd

        for i in range(self.parallel_draft_step - 1):
            for j in range(self.parallel_draft_heads_num_layers):
                export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.weight"] = (
                    export_sd.pop(f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.weight")
                )
                if f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.bias" in export_sd:
                    export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.bias"] = (
                        export_sd.pop(f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.bias")
                    )

        export_sd["parallel_draft_heads.lm_head.weight"] = export_sd.pop(
            "parallel_draft_heads.lm_head.weight"
        )
        return export_sd
