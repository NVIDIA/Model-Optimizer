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

import json
import re
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .hf_spec_configs import kimik2_eagle_template_config, llama_eagle_template_config

ALL_SPEC_MODES = ["eagle"]

LLAMA_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.0.hidden_norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "norm",
        "fc",
    },
    "optional": {"d2t", "lm_head"},
}

KIMIK2_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.kv_a_layernorm",
        "layers.0.self_attn.q_a_layernorm",
        "layers.0.self_attn.q_a_proj",
        "layers.0.self_attn.q_b_proj",
        "layers.0.self_attn.kv_a_proj_with_mqa",
        "layers.0.self_attn.kv_b_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.0.hidden_norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "norm",
        "fc",
    },
    "optional": {
        "d2t",
        "lm_head",
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


class SpeculativeDecodingExporter:
    """Export an modelopt speculative decoding checkpoint to deployment format."""

    def __init__(self, model: nn.Module):
        """Initialize the SpeculativeDecodingExporter."""
        self.model = model

    @abstractmethod
    def export(self, export_dir: Path | str, dtype: torch.dtype | None = None):
        """Export the model to the deployment format."""
        raise NotImplementedError("Subclasses must implement this method.")


class EagleExporter(SpeculativeDecodingExporter):
    """Draft model exporter for Eagle."""

    def __init__(self, model: nn.Module):
        """Initialize the EagleExporter."""
        super().__init__(model)
        self.eagle_decoder_type = model.eagle_config.eagle_decoder_type
        self.num_hidden_layers = model.eagle_config.num_hidden_layers

    def _check_valid_sd(self, export_sd: dict):
        """Check the export state dict is valid, otherwise raise Exception."""
        expected_keys_single_layer = {
            "llama": LLAMA_EAGLE_SINGLE_LAYER,
            "kimik2": KIMIK2_EAGLE_SINGLE_LAYER,
        }[self.eagle_decoder_type]
        # Check that export sd has required keys
        for key in expected_keys_single_layer["required"]:
            assert f"{key}.weight" in export_sd, f"Missing required key: {key}.weight"
        for i in range(1, self.num_hidden_layers):
            for key in expected_keys_single_layer["required"] - {
                "layers.0.hidden_norm",
                "layers.0.input_layernorm",
                "norm",
                "fc",
            }:
                assert f"{key}.weight".replace("layers.0", f"layers.{i}") in export_sd, (
                    f"Missing required key: {key}.weight"
                )

        # Check that export sd has no unexpected keys
        # Note that quantized eagle are allowed to have scales
        allowed_keys_single_layer = (
            expected_keys_single_layer["required"] | expected_keys_single_layer["optional"]
        )
        for key in export_sd:
            assert (
                re.sub(r"layers\.\d+\.", "layers.0.", key.rsplit(".", 1)[0])
                in allowed_keys_single_layer
            ), f"Unexpected key: {key}"

    def _extract_state_dict(self, full_state_dict: dict):
        """Extract and return eagle state dict in deployment format."""
        export_sd = {}
        for key in full_state_dict:
            if "eagle_module" in key or "lm_head" in key:
                export_key = key.replace("eagle_module.", "")
                export_sd[export_key] = full_state_dict[key].clone()
        # Use base model's lm head if draft model doesn't have one
        if "lm_head.weight" not in export_sd:
            export_sd["lm_head.weight"] = full_state_dict["lm_head.weight"]

        self._check_valid_sd(export_sd)

        return export_sd

    def _export_config(self):
        """Export config.json in deployment format."""
        template_config: dict = {
            "llama": llama_eagle_template_config,
            "kimik2": kimik2_eagle_template_config,
        }[self.model.eagle_config.eagle_decoder_type]
        template_config = deepcopy(template_config)

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
                        value[sub_key] = _get_config_from_draft_or_base(sub_key, self.model)
            elif value is None:
                # First, we try to load fron eagle config.
                new_value = _get_config_from_draft_or_base(key, self.model)
                # If the value is a torch.dtype, we convert to string for serialization.
                if isinstance(new_value, torch.dtype):
                    new_value = str(new_value).replace("torch.", "")
                template_config[key] = new_value

        return template_config

    def export(self, export_dir: Path | str, dtype: torch.dtype | None = None):
        """Export the model to the deployment format."""
        # Make export dir
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export quantized modules
        if has_quant_opt(self.model):
            from ..unified_export_hf import _export_transformers_checkpoint

            full_sd, hf_quant_config = _export_transformers_checkpoint(self.model, dtype)
        else:
            full_sd, hf_quant_config = self.model.state_dict(), None

        # Export state dit
        drafter_sd = self._extract_state_dict(full_sd)
        save_file(drafter_sd, f"{export_dir}/model.safetensors")

        # Export config
        drafter_config = self._export_config()
        if hf_quant_config is not None:
            drafter_config["quantization_config"] = hf_quant_config
        with open(f"{export_dir}/config.json", "w") as file:
            json.dump(drafter_config, file, indent=4)

        # Export hf_quant_config for backward compatibility
        if hf_quant_config is not None:
            with open(f"{export_dir}/hf_quant_config.json", "w") as file:
                json.dump(hf_quant_config, file, indent=4)


class EagleMedusaExporter(EagleExporter):
    """Draft model exporter for EagleMedusa."""

    def __init__(self, model: nn.Module):
        """Initialize the EagleMedusaExporter."""
        super().__init__(model)
        self.parallel_draft_step = model.eagle_config.parallel_draft_step
        self.parallel_draft_heads_num_layers = model.eagle_config.parallel_draft_heads_num_layers
        # NOTE: tmp: bypassing format check for parallel draft
        self._check_valid_sd = lambda *args, **kwargs: None

    def _extract_state_dict(self, full_state_dict: dict):
        """Extract the state dict of the draft model in deployment format."""
        export_sd = super()._extract_state_dict(full_state_dict)
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
