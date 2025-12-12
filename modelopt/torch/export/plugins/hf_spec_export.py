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

import torch
import torch.nn as nn


def eagle_state_dict_key_convert(num_hidden_layers: int = 1) -> list[str]:
    """Convert our eagle model state dict key to official format key(s)."""
    assert num_hidden_layers >= 1, "num_hidden_layers should be at least 1."
    eagle_modelopt_to_official = {
        "required": {
            "norm.weight": "norm.weight",
            "fc.weight": "fc.weight",
        },
        "optional": {
            "d2t": "d2t",
            "eagle_lm_head.weight": "lm_head.weight",
        },
    }
    if num_hidden_layers == 1:
        eagle_modelopt_to_official["required"].update(
            {
                "hidden_norm.weight": "midlayer.hidden_norm.weight",
                "input_embeds_norm.weight": "midlayer.input_layernorm.weight",
            }
        )
    else:
        eagle_modelopt_to_official["required"].update(
            {
                "hidden_norm.weight": "layers.0.hidden_norm.weight",
                "input_embeds_norm.weight": "layers.0.input_layernorm.weight",
            }
        )
    for i in range(num_hidden_layers):
        if num_hidden_layers == 1:
            index = ""
        else:
            index = f".{i}"
        eagle_modelopt_to_official["required"].update(
            {
                f"layers.{i}.self_attn.q_proj.weight": "midlayer"
                + index
                + ".self_attn.q_proj.weight",
                f"layers.{i}.self_attn.k_proj.weight": "midlayer"
                + index
                + ".self_attn.k_proj.weight",
                f"layers.{i}.self_attn.v_proj.weight": "midlayer"
                + index
                + ".self_attn.v_proj.weight",
                f"layers.{i}.self_attn.o_proj.weight": "midlayer"
                + index
                + ".self_attn.o_proj.weight",
                f"layers.{i}.mlp.gate_proj.weight": "midlayer" + index + ".mlp.gate_proj.weight",
                f"layers.{i}.mlp.up_proj.weight": "midlayer" + index + ".mlp.up_proj.weight",
                f"layers.{i}.mlp.down_proj.weight": "midlayer" + index + ".mlp.down_proj.weight",
                f"layers.{i}.post_attention_layernorm.weight": "midlayer"
                + index
                + ".post_attention_layernorm.weight",
            }
        )
    return eagle_modelopt_to_official


def _check_state_dict_keys_match(draft_model: nn.Module, required_items: dict):
    """Check if the state dict keys match."""
    draft_keys = set(draft_model.state_dict().keys())
    for required_key in required_items:
        if required_key not in draft_keys:
            raise ValueError(f"State dict keys mismatch!\nMissing in draft model: {required_key}")


def spec_opt_only(model: nn.Module):
    """Check if the model have only speculative decoding optimization."""
    opt_modes = getattr(model, "_modelopt_state", None)
    return (
        isinstance(opt_modes, (list, tuple)) and len(opt_modes) == 1 and opt_modes[0][0] == "eagle"
    )


def export_spec_ckpt_state_dict(model: nn.Module):
    """Only return the state dict of the draft model in official format and ignore the base model."""
    # check the model has only speculative decoding
    assert spec_opt_only(model), "Not purely eagle model."

    eagle_modelopt_to_official = eagle_state_dict_key_convert(model.eagle_config.num_hidden_layers)
    # Check if the state dict keys match
    _check_state_dict_keys_match(model.eagle_module, eagle_modelopt_to_official["required"])

    # Convert key names and save the state dict
    eagle_state = model.eagle_module.state_dict()
    export_state_dict = {}
    for ours_key, export_key in {
        **eagle_modelopt_to_official["required"],
        **eagle_modelopt_to_official["optional"],
    }.items():
        if ours_key in eagle_state:
            export_state_dict[export_key] = eagle_state[ours_key]

    # TODO: (hg) this is a temp fix. Find cleaner way to do this.
    if "eagle_lm_head.weight" not in eagle_state:
        export_state_dict["lm_head.weight"] = model.state_dict()["lm_head.weight"]

    # Add parallel draft weights
    if model.eagle_config.parallel_draft_step > 1:
        for i in range(model.eagle_config.parallel_draft_step - 1):
            for j in range(model.eagle_config.parallel_draft_heads_num_layers):
                export_state_dict[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.weight"] = (
                    eagle_state[f"parallel_draft_heads.{i}.{j}.linear.weight"]
                )
                if f"parallel_draft_heads.{i}.{j}.linear.bias" in eagle_state:
                    export_state_dict[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.bias"] = (
                        eagle_state[f"parallel_draft_heads.{i}.{j}.linear.bias"]
                    )
            export_state_dict[f"parallel_draft_heads.{i}.lm_head.weight"] = eagle_state[
                f"parallel_draft_heads.{i}.{model.eagle_config.parallel_draft_heads_num_layers}.weight"
            ]

    return export_state_dict


def export_spec_ckpt_config(model: nn.Module):
    """Return the config of draft model in official format."""
    assert spec_opt_only(model), "Not purely eagle model."

    # This is the config keys in official checkpoint.
    template_config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "bos_token_id": None,
        "eos_token_id": None,
        "hidden_act": None,
        "hidden_size": None,
        "initializer_range": None,
        "intermediate_size": None,
        "max_position_embeddings": None,
        "model_type": "llama",
        "num_attention_heads": None,
        "num_key_value_heads": None,
        "num_hidden_layers": None,
        "pad_token_id": None,
        "rms_norm_eps": None,
        "tie_word_embeddings": False,
        "torch_dtype": None,
        "transformers_version": None,
        "use_cache": None,
        "vocab_size": None,
        "draft_vocab_size": None,
        "rope_scaling": None,
        "attention_bias": None,
        "attention_dropout": None,
        "head_dim": None,
        "mlp_bias": None,
        "pretraining_tp": None,
        "rope_theta": None,
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": None,
            "use_aux_hidden_state": None,
            "use_input_layernorm_in_first_layer": None,
            "use_last_layernorm": None,
            "use_mtp_layernorm": None,
        },
    }

    def _get_config_from_eagle_config_or_base_config(key: str, model: nn.Module):
        if getattr(model.eagle_config, key, None) is not None:
            return getattr(model.eagle_config, key)
        elif getattr(model.config, key, None) is not None:
            return getattr(model.config, key)
        else:
            return None

    for key in template_config:
        value = template_config[key]
        if isinstance(value, dict):
            # for eagle config, we find it in model.eagle_config
            for sub_key in value:
                value[sub_key] = _get_config_from_eagle_config_or_base_config(sub_key, model)
        elif value is None:
            # First, we try to load fron eagle config.
            new_value = _get_config_from_eagle_config_or_base_config(key, model)
            # If the value is a torch.dtype, we convert to string for serialization.
            if isinstance(new_value, torch.dtype):
                new_value = str(new_value).replace("torch.", "")
            template_config[key] = new_value

    return template_config
