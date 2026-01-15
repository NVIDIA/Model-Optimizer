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
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import (
    AutoTokenizer as NemoAutoTokenizer,
)
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.llama_nemotron import HFLlamaNemotronImporter
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_str
from nemo.utils.import_utils import safe_import
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelopt.torch.puzzletron.export.MCore.puzzletron_layer_specs import (
    PuzzletronAttentionConfig,
    PuzzletronHeterogeneousTransformerConfig,
    PuzzletronMLPConfig,
    get_gpt_heterogeneous_layer_spec_puzzletron,
)


def convert_attention_config_from_cfg_object(attention_config, num_attention_heads, head_dim):
    for unsupported_key in [
        "llama4",
        "num_sink_tokens",
        "sparsify",
        "unshifted_sink",
        "use_prefill_window_in_sink_attention",
    ]:
        if hasattr(attention_config, unsupported_key) and getattr(
            attention_config, unsupported_key
        ) not in [
            None,
            False,
        ]:
            #
            #        if attention_config.get(unsupported_key, None) not in [None, False]:
            raise NotImplementedError(f"{unsupported_key} is not supported")
    window_size = attention_config.window_size if hasattr(attention_config, "window_size") else None
    if window_size is not None:
        window_size = (window_size, 0)
    is_mamba = attention_config.mamba if hasattr(attention_config, "mamba") else None
    n_heads_in_group = (
        attention_config.n_heads_in_group if hasattr(attention_config, "n_heads_in_group") else 1
    )
    if n_heads_in_group is None:
        n_heads_in_group = 1
    return asdict(
        PuzzletronAttentionConfig(
            no_op=attention_config.no_op if hasattr(attention_config, "no_op") else False,
            replace_with_linear=(
                attention_config.replace_with_linear
                if hasattr(attention_config, "replace_with_linear")
                else False
            ),
            num_attention_heads=num_attention_heads,
            num_query_groups=num_attention_heads // n_heads_in_group,
            kv_channels=head_dim,
            window_size=window_size,
            multi_latent_attention=False,
            is_mamba=is_mamba,
            mamba_state_dim=(
                attention_config.mamba.state_dim
                if is_mamba and hasattr(attention_config.mamba, "state_dim")
                else 128
            ),
            mamba_head_dim=(
                attention_config.mamba.head_dim
                if is_mamba and hasattr(attention_config.mamba, "head_dim")
                else 64
            ),
            mamba_num_groups=(
                attention_config.mamba.num_groups
                if is_mamba and hasattr(attention_config.mamba, "num_groups")
                else 8
            ),
            mamba_num_heads=(
                attention_config.mamba.num_heads
                if is_mamba and hasattr(attention_config.mamba, "num_heads")
                else None
            ),
        )
    )


def convert_mlp_config_from_cfg_object(mlp_config, parallel_blocks):
    """Convert MLP config from HF format to NeMo format.

    Args:
        mlp_config: The MLP configuration object from HF
        parallel_blocks: Parallel blocks configuration (not currently supported)
    """
    if parallel_blocks is not None:
        raise NotImplementedError("parallel_blocks is not supported")
    if not hasattr(mlp_config, "gated") or mlp_config.gated is False:
        raise NotImplementedError("notgated MLP is not supported")

    # Validate this block's activation function
    if not hasattr(mlp_config, "hidden_act"):
        raise ValueError(f"MLP config must have hidden_act attribute")
    #    if mlp_config.hidden_act != block_hidden_act:
    #        raise ValueError(f"MLP config hidden_act mismatch: config has {mlp_config.hidden_act}, expected {block_hidden_act}")

    if hasattr(mlp_config, "sparsify") and mlp_config.sparsify is not None:
        raise NotImplementedError("sparsify is not supported")
    is_moe = hasattr(mlp_config, "moe") and mlp_config.moe is not None
    # Note: hidden_act is validated above but not stored in PuzzletronMLPConfig
    # It will be used at the call site for the NeMo model config
    return asdict(
        PuzzletronMLPConfig(
            no_op=mlp_config.no_op if hasattr(mlp_config, "no_op") else False,
            replace_with_linear=mlp_config.replace_with_linear
            if hasattr(mlp_config, "replace_with_linear")
            else False,
            ffn_hidden_size=mlp_config.intermediate_size
            if hasattr(mlp_config, "intermediate_size")
            else None,
            num_moe_experts=(
                mlp_config.moe.num_local_experts
                if is_moe and hasattr(mlp_config.moe, "num_local_experts")
                else None
            ),
            moe_shared_expert_intermediate_size=(
                mlp_config.moe.shared_expert_intermediate_dim
                if is_moe and hasattr(mlp_config.moe, "shared_expert_intermediate_dim")
                else None
            ),
            moe_ffn_hidden_size=(
                mlp_config.moe.expert_intermediate_dim
                if is_moe and hasattr(mlp_config.moe, "expert_intermediate_dim")
                else None
            ),
            moe_router_topk=(
                mlp_config.moe.num_experts_per_tok
                if is_moe and hasattr(mlp_config.moe, "num_experts_per_tok")
                else 2
            ),
        )
    )


def convert_nemo_config_to_hf_decilm_config(
    nemo_config: "PuzzletronNemotronModelConfig",
    vocab_size: int,
    eos_token_id: Union[int, List[int], None] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> "DeciLMConfig":
    """Convert a NeMo PuzzletronNemotronModelConfig to HF DeciLMConfig.

    This function extracts the conversion logic from the exporter so it can be
    used in unit tests without requiring file I/O.

    Args:
        nemo_config: The NeMo config to convert
        vocab_size: Vocabulary size for the HF config
        eos_token_id: EOS token ID(s). Can be int or list of ints.
        bos_token_id: BOS token ID
        pad_token_id: PAD token ID

    Returns:
        DeciLMConfig: The equivalent HF config
    """
    from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig

    # Get preserved HF config metadata (stored as direct attribute)
    # This enables lossless round-trip conversion HF → NeMo → HF
    source_hf_config_metadata = getattr(nemo_config, "source_hf_config_metadata", None) or {}

    # Parse the heterogeneous layers config from JSON
    block_configs = []

    if (
        hasattr(nemo_config, "heterogeneous_layers_config_encoded_json")
        and nemo_config.heterogeneous_layers_config_encoded_json
    ):
        try:
            heterogeneous_config = json.loads(nemo_config.heterogeneous_layers_config_encoded_json)
            raw_block_configs = heterogeneous_config.get("block_configs", [])

            for i, raw_block_config in enumerate(raw_block_configs):
                attn_block = raw_block_config.get("attention", {})
                mlp_block = raw_block_config.get("mlp", {})

                # Configure attention
                attention_config = {
                    "no_op": attn_block.get("no_op", False),
                    "replace_with_linear": attn_block.get("replace_with_linear", False),
                    "sparsify": attn_block.get("sparsify", None),
                    "n_heads_in_group": attn_block.get(
                        "num_attention_heads", nemo_config.num_attention_heads
                    )
                    // attn_block.get("num_query_groups", nemo_config.num_query_groups),
                    "window_length": attn_block.get("window_size", None),
                    "num_sink_tokens": attn_block.get("num_sink_tokens", None),
                    "use_prefill_window_in_sink_attention": attn_block.get(
                        "use_prefill_window_in_sink_attention", False
                    ),
                    "unshifted_sink": attn_block.get("unshifted_sink", False),
                }

                # Handle Mamba: convert from NeMo flat structure to HF nested structure
                if attn_block.get("is_mamba", False):
                    attention_config["mamba"] = {
                        "state_dim": attn_block.get("mamba_state_dim", 128),
                        "num_heads": attn_block.get(
                            "mamba_num_heads", nemo_config.num_attention_heads
                        ),
                        "head_dim": attn_block.get("mamba_head_dim", 64),
                        "num_groups": attn_block.get("mamba_num_groups", 8),
                    }
                else:
                    attention_config["mamba"] = None

                # Handle Llama4: pass through as dict if present
                attention_config["llama4"] = attn_block.get("llama4", None)

                # Configure FFN
                ffn_config = {
                    "no_op": mlp_block.get("no_op", False),
                    "replace_with_linear": mlp_block.get("replace_with_linear", False),
                    "sparsify": mlp_block.get("sparsify", None),
                    "intermediate_size": mlp_block.get(
                        "ffn_hidden_size", nemo_config.ffn_hidden_size
                    ),
                    "gated": True,  # Puzzletron uses gated activations
                    # Use the activation function name extracted from this block's config
                    "hidden_act": mlp_block.get("hidden_act", None),
                }

                # Handle MoE: convert from NeMo flat structure to HF nested structure
                num_moe_experts = mlp_block.get("num_moe_experts", None)
                if num_moe_experts is not None:
                    ffn_config["moe"] = {
                        "num_local_experts": num_moe_experts,
                        "num_experts_per_tok": mlp_block.get("moe_router_topk", 1),
                        "expert_intermediate_dim": mlp_block.get("moe_ffn_hidden_size", 8192),
                        "shared_expert_intermediate_dim": mlp_block.get(
                            "moe_shared_expert_intermediate_size", 8192
                        ),
                    }
                else:
                    ffn_config["moe"] = None

                block_configs.append({"attention": attention_config, "ffn": ffn_config})
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Could not parse heterogeneous config JSON: {e}")
    else:
        raise ValueError("No block configs found in source configuration")

    # Create rope scaling config
    rope_scaling = {
        "factor": nemo_config.scale_factor,
        "low_freq_factor": getattr(nemo_config, "low_freq_factor", 1.0),
        "high_freq_factor": getattr(nemo_config, "high_freq_factor", 4.0),
        "original_max_position_embeddings": getattr(nemo_config, "old_context_len", 8192),
        "rope_type": "llama3",
    }

    # Get EOS token ID(s) - prefer preserved value from source HF config metadata or provided value
    if eos_token_id is None:
        eos_token_id = source_hf_config_metadata.get("eos_token_id", None)

    # Create DeciLM config
    hf_config = DeciLMConfig(
        block_configs=block_configs,
        hidden_size=nemo_config.hidden_size,
        max_position_embeddings=nemo_config.seq_length,
        num_attention_heads=nemo_config.num_attention_heads,
        num_hidden_layers=nemo_config.num_layers,
        tie_word_embeddings=nemo_config.share_embeddings_and_output_weights,
        vocab_size=vocab_size,
        rms_norm_eps=nemo_config.layernorm_epsilon,
        attention_bias=getattr(nemo_config, "attention_bias", False),
        o_proj_bias=getattr(
            nemo_config, "o_proj_bias", getattr(nemo_config, "attention_bias", False)
        ),
        rope_theta=nemo_config.rotary_base,
        rope_scaling=rope_scaling,
        position_embedding_type="rope",
        architectures=["DeciLMForCausalLM"],
        model_type="nemotron-nas",
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        head_dim=nemo_config.kv_channels,
        # Restore auto_map from preserved metadata (needed for trust_remote_code loading)
        auto_map=source_hf_config_metadata.get(
            "auto_map",
            {
                "AutoConfig": "configuration_decilm.DeciLMConfig",
                "AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM",
            },
        ),
        # Restore dtype field from preserved metadata
        dtype=source_hf_config_metadata.get("dtype", "bfloat16"),
    )

    return hf_config


def _config_to_dict(config) -> Dict[str, Any]:
    """Convert a config object to a dictionary.

    Args:
        config: Either an object with attributes or already a dictionary

    Returns:
        Dictionary representation of the config
    """
    if isinstance(config, dict):
        return config
    return vars(config)


def _build_puzzletron_mappings_and_transforms(
    source_config: PuzzletronHeterogeneousTransformerConfig,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """Build mappings and transform specifications for Puzzletron heterogeneous models.

    Args:
        source_config: The Puzzletron heterogeneous transformer configuration

    Returns:
        Tuple containing:
        - attn_mapping: Attention layer mappings
        - ffn_mapping: FFN layer mappings
        - mamba_mapping: Mamba layer mappings
        - moe_mapping: MoE layer mappings
        - transform_specs: List of transform specifications with source_key, target_key, transform_function
    """
    attn_mapping = {}
    ffn_mapping = {}
    mamba_mapping = {}
    moe_mapping = {}
    transform_specs = []

    # Determine config type and extract block configs
    is_hf_config = hasattr(source_config, "block_configs")
    is_nemo_config = (
        hasattr(source_config, "heterogeneous_layers_config_encoded_json")
        and source_config.heterogeneous_layers_config_encoded_json
    )
    assert not (is_hf_config and is_nemo_config), "Cannot have both HF and NeMo config"

    if is_hf_config:
        # HF config case (importer)
        block_configs = source_config.block_configs
    elif is_nemo_config:
        # NeMo config case (exporter) - parse JSON
        try:
            heterogeneous_config = json.loads(
                source_config.heterogeneous_layers_config_encoded_json
            )
            block_configs = heterogeneous_config.get("block_configs", [])
        except (json.JSONDecodeError, KeyError):
            block_configs = []
    else:
        block_configs = []

    # Check if we found any block configs
    if not block_configs:
        raise ValueError(
            "No block configs found in source configuration. "
            "Expected either 'block_configs' attribute (HF config) or "
            "'heterogeneous_layers_config_encoded_json' attribute (NeMo config) with valid block configs."
        )

    # TODO it is better (more stable) to use target.config.block_configs
    for idx, block_config in enumerate(block_configs):
        # Convert block config to dictionary
        block_dict = _config_to_dict(block_config)

        # Extract attention and FFN configs (handle both HF "ffn" and NeMo "mlp" keys)
        attn = block_dict.get("attention")
        ffn = block_dict.get("ffn") or block_dict.get("mlp")

        # Convert sub-configs to dictionaries
        attn_dict = _config_to_dict(attn) if attn else {}
        ffn_dict = _config_to_dict(ffn) if ffn else {}

        # Process attention config
        # Handle both HF (mamba) and NeMo (is_mamba) keys
        is_mamba = attn_dict.get("mamba") or attn_dict.get("is_mamba")

        if not attn or attn_dict.get("no_op"):
            value = None
        elif attn_dict.get("replace_with_linear"):
            value = f"decoder.layers.{idx}.self_attention.layer_norm_weight"
        elif is_mamba is not None:
            value = f"decoder.layers.{idx}.self_attention.in_proj.layer_norm_weight"
            for mamba_key in [
                "dt_bias",
                "A_log",
                "D",
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "norm.weight",
                "out_proj.weight",
            ]:
                mamba_mapping[f"model.layers.{idx}.self_attn.mamba_mixer.{mamba_key}"] = (
                    f"decoder.layers.{idx}.self_attention.{mamba_key}"
                )
        else:
            value = f"decoder.layers.{idx}.self_attention.linear_qkv.layer_norm_weight"
            # Store transform spec for QKV merging
            transform_specs.append(
                {
                    "source_key": (
                        f"model.layers.{idx}.self_attn.q_proj.weight",
                        f"model.layers.{idx}.self_attn.k_proj.weight",
                        f"model.layers.{idx}.self_attn.v_proj.weight",
                    ),
                    "target_key": f"decoder.layers.{idx}.self_attention.linear_qkv.weight",
                    "transform_function": "merge_qkv_for_puzzletron",
                    "kwargs": {"idx": idx},
                }
            )

        if value is not None:
            attn_mapping[f"model.layers.{idx}.input_layernorm.weight"] = value

        # Process FFN config
        # Handle both HF (moe, moe.shared_expert_intermediate_dim) and NeMo (num_moe_experts, moe_shared_expert_intermediate_size) keys
        moe_config = ffn_dict.get("moe") or ffn_dict.get("num_moe_experts")
        shared_expert_size = None
        if moe_config:
            # Convert moe_config to dict if it's an object (HF case)
            moe_dict = (
                _config_to_dict(moe_config) if not isinstance(moe_config, (int, type(None))) else {}
            )
            shared_expert_size = moe_dict.get("shared_expert_intermediate_dim") or ffn_dict.get(
                "moe_shared_expert_intermediate_size"
            )

        if not ffn or ffn_dict.get("no_op"):
            value = None
        elif ffn_dict.get("replace_with_linear"):
            value = f"decoder.layers.{idx}.mlp.layer_norm_weight"
        elif moe_config is not None:
            value = f"decoder.layers.{idx}.pre_mlp_layernorm.weight"
            moe_mapping[f"model.layers.{idx}.mlp.router.weight"] = (
                f"decoder.layers.{idx}.mlp.router.weight"
            )
            # Store transform spec for MoE expert FC1 merging
            transform_specs.append(
                {
                    "source_key": (
                        f"model.layers.{idx}.mlp.experts.*.gate_proj.weight",
                        f"model.layers.{idx}.mlp.experts.*.up_proj.weight",
                    ),
                    "target_key": f"decoder.layers.{idx}.mlp.experts.local_experts.*.linear_fc1.weight",
                    "transform_function": "merge_fc1_for_moe",
                    "kwargs": {},
                }
            )
            moe_mapping[f"model.layers.{idx}.mlp.experts.*.down_proj.weight"] = (
                f"decoder.layers.{idx}.mlp.experts.local_experts.*.linear_fc2.weight"
            )
            # Check for shared expert
            if shared_expert_size not in [None, 0]:
                # Store transform spec for MoE shared expert FC1 merging
                transform_specs.append(
                    {
                        "source_key": (
                            f"model.layers.{idx}.mlp.shared_expert.gate_proj.weight",
                            f"model.layers.{idx}.mlp.shared_expert.up_proj.weight",
                        ),
                        "target_key": f"decoder.layers.{idx}.mlp.shared_experts.linear_fc1.weight",
                        "transform_function": "merge_fc1_for_moe",
                        "kwargs": {},
                    }
                )
                moe_mapping[f"model.layers.{idx}.mlp.shared_expert.down_proj.weight"] = (
                    f"decoder.layers.{idx}.mlp.shared_experts.linear_fc2.weight"
                )
        else:
            value = f"decoder.layers.{idx}.mlp.linear_fc1.layer_norm_weight"

        if value is not None:
            ffn_mapping[f"model.layers.{idx}.post_attention_layernorm.weight"] = value

    return attn_mapping, ffn_mapping, mamba_mapping, moe_mapping, transform_specs


def merge_qkv_for_puzzletron(
    ctx: io.state.TransformCTX,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: Optional[int] = None,
):
    """
    Merge q, k, v to interleave-concatenated qkv.
    - Modified version of nemo.lightning.io.state.TransformFns.merge_qkv for Puzzletron
        - idx can be provided to fetch megatron_config for a specific layer
        - heads_per_group is derived from the shape of q and k, instead of calculating (head_num // num_query_groups) from config values
        - num_query_groups is not fetched from a global config value, but calculated from head_num and heads_per_group

    Example: import HF {q|k|v}_proj to layer linear_qkv
    """
    if idx is not None:
        megatron_config = ctx.target.decoder.layers[idx].config
    else:
        megatron_config = ctx.target.config
    head_num = megatron_config.num_attention_heads
    heads_per_group = (
        q.shape[0] // k.shape[0]
    )  # NOTE: This is important to support heterogeneous attention
    num_query_groups = head_num // heads_per_group
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


def split_qkv_for_puzzletron(
    ctx: io.state.TransformCTX, qkv: torch.Tensor, idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split interleave-concatenated qkv to separate q, k, v.
    - Inverse operation of merge_qkv_for_puzzletron for Puzzletron
    - idx can be provided to fetch megatron_config for a specific layer
    - heads_per_group is derived from the shape of qkv, instead of calculating from config values
    - num_query_groups is not fetched from a global config value, but calculated from head_num and heads_per_group

    Example: export NeMo layer linear_qkv to HF {q|k|v}_proj
    """
    if idx is not None:
        megatron_config = ctx.source.decoder.layers[idx].config
    else:
        megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    head_size = megatron_config.kv_channels
    # hidden_size = megatron_config.hidden_size

    # Calculate qkv_total_dim from the actual qkv tensor shape
    # qkv shape is [head_size * (head_num + 2 * num_query_groups), hidden_size]
    qkv_total_dim = qkv.shape[0] // head_size
    num_query_groups = (qkv_total_dim - head_num) // 2
    heads_per_group = head_num // num_query_groups

    # Reshape qkv to 3D: [qkv_total_dim, head_size, hidden_size]
    qkv = qkv.reshape([qkv_total_dim, head_size, -1])

    # when converting base model (linear_qkv), hidden size = megatron_config.hidden_size
    # when converting lora (linear_qkv.adapter.linear_out), hidden size = lora_r
    actual_hidden_size = qkv.size(-1)

    # Create slice indices for q, k, v
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = qkv[q_slice].reshape(-1, actual_hidden_size).cpu()
    k_proj = qkv[k_slice].reshape(-1, actual_hidden_size).cpu()
    v_proj = qkv[v_slice].reshape(-1, actual_hidden_size).cpu()

    return q_proj, k_proj, v_proj


def dtype_from_dict(config_dict):
    """
    Extracts torch dtype from a HF config.
    Handles both 'torch_dtype' (old format) and 'dtype' (new format).
    """
    # Try torch_dtype first (old format), then dtype (new format)
    if "torch_dtype" in config_dict:
        torch_dtype = config_dict["torch_dtype"]
    elif "dtype" in config_dict:
        torch_dtype = config_dict["dtype"]
    else:
        raise ValueError("Expected config dict to have attr `torch_dtype` or `dtype`")

    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    elif isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    else:
        raise ValueError(f"dtype is not of type str/torch.dtype, got {type(torch_dtype)}")


def copy_puzzletron_python_files_to_decilm_checkpoint(output_path: str) -> None:
    """Copy custom Python files from puzzle_tools package to output directory.

    Puzzletron models require custom Python files (configuration_decilm.py,
    modeling_decilm.py, etc.) to be present in the checkpoint directory for
    loading with transformers.AutoModel.

    This function copies all Python files from puzzle_tools/deci_lm_hf_code/
    to ensure the exported checkpoint is fully functional.

    Args:
        output_path: Directory where HF model is being saved
    """
    import logging
    import shutil
    from pathlib import Path

    # Get the puzzle_tools/deci_lm_hf_code directory
    # Navigate from this file: export/MCore/llama_nemotron_utils.py -> v1/puzzle_tools/deci_lm_hf_code/
    package_dir = Path(__file__).parent.parent.parent / "puzzle_tools" / "deci_lm_hf_code"

    if not package_dir.exists():
        logging.warning(
            f"Custom files directory not found: {package_dir}. "
            f"Exported checkpoint may not be loadable without these files."
        )
        return

    # Copy all Python files from the package
    output_dir = Path(output_path)
    copied_files = []
    for src_file in package_dir.glob("*.py"):
        dest_file = output_dir / src_file.name
        shutil.copy2(src_file, dest_file)
        copied_files.append(src_file.name)

    logging.info(f"Copied {len(copied_files)} custom Python files to {output_path}")
    logging.debug(f"Custom files copied: {', '.join(sorted(copied_files)[:5])}...")  # Show first 5


def embed_chat_template_in_tokenizer_config(nemo_checkpoint_path: str, output_path: str) -> None:
    """Embed chat_template from .jinja file into tokenizer_config.json.

    NeMo's HF → NeMo import extracts chat_template to a separate .jinja file
    but doesn't preserve it in tokenizer_config.json. This causes accuracy drops
    in evaluation. This function restores it by:
    1. Reading chat_template.jinja from the NeMo checkpoint
    2. Embedding it into the exported tokenizer_config.json

    Args:
        nemo_checkpoint_path: Path to the NeMo checkpoint (.nemo file/directory)
        output_path: Directory where HF model is being saved
    """
    import logging
    from pathlib import Path

    # Path to NeMo checkpoint tokenizer files
    nemo_checkpoint = Path(nemo_checkpoint_path)
    nemo_chat_template_jinja = (
        nemo_checkpoint / "context" / "nemo_tokenizer" / "chat_template.jinja"
    )

    # Path to exported tokenizer config
    output_dir = Path(output_path)
    output_tokenizer_config = output_dir / "tokenizer_config.json"

    # Check if both files exist
    if not nemo_chat_template_jinja.exists():
        logging.debug(
            f"No chat_template.jinja found in NeMo checkpoint at {nemo_chat_template_jinja}"
        )
        return

    if not output_tokenizer_config.exists():
        logging.warning(f"tokenizer_config.json not found at {output_tokenizer_config}")
        return

    # Read chat_template from .jinja file
    chat_template_content = nemo_chat_template_jinja.read_text()

    # Load tokenizer_config.json
    with open(output_tokenizer_config, "r") as f:
        tokenizer_config = json.load(f)

    # Check if chat_template is already embedded (shouldn't be, but be safe)
    if "chat_template" in tokenizer_config:
        logging.debug("chat_template already embedded in tokenizer_config.json, skipping")
        return

    # Embed the chat_template
    tokenizer_config["chat_template"] = chat_template_content

    # Save updated tokenizer_config.json
    with open(output_tokenizer_config, "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    logging.info(f"✓ Embedded chat_template from NeMo checkpoint into tokenizer_config.json")
    logging.debug(f"  Template length: {len(chat_template_content)} characters")
