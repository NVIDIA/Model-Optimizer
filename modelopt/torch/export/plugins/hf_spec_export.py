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

LLAMA_EAGLE_SINGLE_LAYER = {
    "required": {
        "midlayer.self_attn.q_proj.weight",
        "midlayer.self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight",
        "midlayer.self_attn.o_proj.weight",
        "midlayer.mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight",
        "midlayer.hidden_norm.weight",
        "midlayer.input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight",
        "norm.weight",
        "fc.weight",
    },
    "optional": {"d2t", "lm_head.weight"},
}

KIMIK2_EAGLE_SINGLE_LAYER = {
    "required": {
        "midlayer.self_attn.kv_a_layernorm.weight",
        "midlayer.self_attn.q_a_layernorm.weight",
        "midlayer.self_attn.q_a_proj.weight",
        "midlayer.self_attn.q_b_proj.weight",
        "midlayer.self_attn.kv_a_proj_with_mqa.weight",
        "midlayer.self_attn.kv_b_proj.weight",
        "midlayer.self_attn.o_proj.weight",
        "midlayer.mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight",
        "midlayer.hidden_norm.weight",
        "midlayer.input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight",
        "norm.weight",
        "fc.weight",
    },
    "optional": {
        "d2t",
        "lm_head.weight",
    },
}


def _check_valid_sd(state_dict: dict, eagle_decoder_type: str, num_hidden_layers: int):
    """Check the export state dict is valid, otherwise raise Exception."""
    expected_keys_single_layer = {
        "llama": LLAMA_EAGLE_SINGLE_LAYER,
        "kimik2": KIMIK2_EAGLE_SINGLE_LAYER,
    }[eagle_decoder_type]
    # Check that export sd has required keys
    if num_hidden_layers == 1:
        for key in expected_keys_single_layer["required"]:
            assert key in state_dict, f"Missing required key: {key}"
    else:
        for key in expected_keys_single_layer["required"]:
            assert key.replace("midlayer", "midlayer.0") in state_dict, (
                f"Missing required key: {key}"
            )
        for i in range(1, num_hidden_layers):
            for key in expected_keys_single_layer["required"] - {
                "midlayer.hidden_norm.weight",
                "midlayer.input_layernorm.weight",
                "norm.weight",
                "fc.weight",
            }:
                assert key.replace("midlayer", f"midlayer.{i}") in state_dict, (
                    f"Missing required key: {key}"
                )

    # Check that export sd has no unexpected keys
    allowed_keys_single_layer = (
        expected_keys_single_layer["required"] | expected_keys_single_layer["optional"]
    )
    if num_hidden_layers == 1:
        for key in state_dict:
            assert key in allowed_keys_single_layer, f"Unexpected key: {key}"
    else:
        for key in state_dict:
            assert re.sub(r"midlayers\.\d+\.", "", key) in {
                k.replace("midlayer.", "") for k in allowed_keys_single_layer
            }, f"Unexpected key: {key}"


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

    # Rename layers to midlayer
    if model.eagle_config.num_hidden_layers == 1:
        model.eagle_module.midlayer = model.eagle_module._modules.pop("layers")[0]
    else:
        model.eagle_module.midlayer = model.eagle_module._modules.pop("layers")
    export_sd = copy(model.eagle_module.state_dict())

    # Use base model's lm head if draft model doesn't have one
    if "lm_head.weight" not in export_sd:
        export_sd["lm_head.weight"] = model.state_dict()["lm_head.weight"]

    # Rename parallel draft weights
    if model.eagle_config.parallel_draft_step > 1:
        for i in range(model.eagle_config.parallel_draft_step - 1):
            for j in range(model.eagle_config.parallel_draft_heads_num_layers):
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
        # NOTE: tmp: bypassing format check for parallel draft
        return export_sd

    _check_valid_sd(
        export_sd, model.eagle_config.eagle_decoder_type, model.eagle_config.num_hidden_layers
    )

    return export_sd


def export_spec_ckpt_config(model: nn.Module):
    """Return the config of draft model in official format."""
    assert spec_opt_only(model), "Not purely eagle model."

    # This is the config keys in official checkpoint.
    llama_eagle_template_config = {
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
            "next_layer_regular": True,
            "parallel_draft_step": None,
            "parallel_draft_heads_num_layers": None,
        },
    }

    kimik2_eagle_template_config = {
        "architectures": ["Eagle3DeepseekV2ForCausalLM"],
        "attention_bias": None,
        "attention_dropout": None,
        "aux_loss_alpha": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": None,
        "diversity_penalty": None,
        "do_sample": None,
        "early_stopping": None,
        "encoder_no_repeat_ngram_size": None,
        "eos_token_id": None,
        "ep_size": None,
        "first_k_dense_replace": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": None,
        "hidden_size": None,
        "id2label": None,
        "initializer_range": None,
        "intermediate_size": None,
        "is_decoder": None,
        "is_encoder_decoder": None,
        "kv_lora_rank": None,
        "label2id": None,
        "length_penalty": None,
        "max_length": None,
        "max_position_embeddings": None,
        "min_length": None,
        "model_type": "kimi_k2",
        "moe_intermediate_size": None,
        "moe_layer_freq": None,
        "n_group": None,
        "n_routed_experts": None,
        "n_shared_experts": None,
        "no_repeat_ngram_size": None,
        "norm_topk_prob": None,
        "num_attention_heads": None,
        "num_beam_groups": None,
        "num_beams": None,
        "num_experts_per_tok": None,
        "num_hidden_layers": None,
        "num_key_value_heads": None,
        "num_nextn_predict_layers": None,
        "num_return_sequences": None,
        "output_attentions": None,
        "output_hidden_states": None,
        "output_scores": None,
        "pad_token_id": None,
        "pretraining_tp": None,
        "pruned_heads": None,
        "q_lora_rank": None,
        "qk_nope_head_dim": None,
        "qk_rope_head_dim": None,
        "remove_invalid_values": None,
        "repetition_penalty": None,
        "return_dict": None,
        "return_dict_in_generate": None,
        "rms_norm_eps": None,
        "rope_scaling": None,
        "rope_theta": None,
        "routed_scaling_factor": None,
        "scoring_func": None,
        "sep_token_id": None,
        "seq_aux": None,
        "temperature": None,
        "tf_legacy_loss": None,
        "tie_encoder_decoder": None,
        "tie_word_embeddings": None,
        "top_k": None,
        "top_p": None,
        "topk_group": None,
        "topk_method": None,
        "torch_dtype": None,
        "torchscript": None,
        "transformers_version": None,
        "typical_p": None,
        "use_bfloat16": None,
        "use_cache": None,
        "v_head_dim": None,
        "vocab_size": None,
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": None,
            "use_aux_hidden_state": None,
            "use_input_layernorm_in_first_layer": None,
            "use_last_layernorm": None,
            "use_mtp_layernorm": None,
            "next_layer_regular": True,
            "parallel_draft_step": None,
            "parallel_draft_heads_num_layers": None,
        },
    }

    template_config: dict = {
        "llama": llama_eagle_template_config,
        "kimik2": kimik2_eagle_template_config,
    }[model.eagle_config.eagle_decoder_type]

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
                if value[sub_key] is None:
                    value[sub_key] = _get_config_from_eagle_config_or_base_config(sub_key, model)
        elif value is None:
            # First, we try to load fron eagle config.
            new_value = _get_config_from_eagle_config_or_base_config(key, model)
            # If the value is a torch.dtype, we convert to string for serialization.
            if isinstance(new_value, torch.dtype):
                new_value = str(new_value).replace("torch.", "")
            template_config[key] = new_value

    return template_config
