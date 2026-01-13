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

"""Convert a Llama3 model to a DeciLM model."""

#!/usr/bin/env python3
from pathlib import Path

import torch
from fire import Fire
from transformers import LlamaConfig

from modelopt.torch.puzzletron.decilm.conversion_utils import convert_model_weights_to_decilm
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.tools.checkpoint_utils import copy_tokenizer
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import copy_deci_lm_hf_code

"""
example:

python -m scripts.hf.convert_llama3_to_decilm  \
    --input_dir .../meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_dir .../meta-llama/Meta-Llama-3.1-8B-Instruct--deci-hf/
"""


def convert_llama3_config_to_decilm_config(config: LlamaConfig) -> DeciLMConfig:
    """Convert Llama3 config to DeciLM config format."""
    print("\n=== Converting Llama3 Config to DeciLM Config ===")

    # Get dtype from config - check both dtype and torch_dtype
    # Prefer dtype if it's set (not None), otherwise fall back to torch_dtype
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)

    # Convert torch.dtype to string if needed (for JSON serialization)
    if dtype is not None and isinstance(dtype, torch.dtype):
        dtype = str(dtype).replace("torch.", "")

    # Track which global values will be removed (moved to per-layer configs)
    print("\n📝 Converting global values to per-layer block_configs:")
    print(
        f"  - intermediate_size: {config.intermediate_size} → block_configs[*].ffn.intermediate_size"
    )
    print(
        f"  - num_key_value_heads: {config.num_key_value_heads} → block_configs[*].attention.n_heads_in_group (derived)"
    )
    print(f"  - hidden_act: {config.hidden_act} → block_configs[*].ffn.hidden_act")
    print(
        f"  - sliding_window: {getattr(config, 'sliding_window', None)} → block_configs[*].attention.window_length"
    )

    # Create block configs for each layer
    block_configs = []
    for i in range(config.num_hidden_layers):
        # Configure attention
        attention_config = {
            "no_op": False,
            "replace_with_linear": False,
            "sparsify": None,
            "n_heads_in_group": config.num_attention_heads // config.num_key_value_heads,
            "window_length": None,  # Llama3 doesn't use sliding window by default
            "num_sink_tokens": None,  # Llama3 doesn't use sink attention
            "use_prefill_window_in_sink_attention": False,
            "unshifted_sink": False,
            "mamba": None,
            "llama4": None,  # No Llama4 specific attention for Llama3
        }

        # Configure FFN
        ffn_config = {
            "no_op": False,
            "replace_with_linear": False,
            "sparsify": None,
            "intermediate_size": config.intermediate_size,
            "gated": True,  # Llama3 uses SwiGLU
            "hidden_act": config.hidden_act,
            "moe": None,  # Llama3 doesn't use MoE
        }

        block_configs.append({"attention": attention_config, "ffn": ffn_config})

    # Create DeciLM config
    decilm_config = DeciLMConfig(
        block_configs=block_configs,
        hidden_size=config.hidden_size,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        tie_word_embeddings=config.tie_word_embeddings,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        o_proj_bias=config.attention_bias,  # llama3 bias defined by attention_bias
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        position_embedding_type="rope",  # Llama3 uses standard RoPE
        architectures=["DeciLMForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_decilm.DeciLMConfig",
            "AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM",
        },
        eos_token_id=config.eos_token_id,
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        dtype=dtype,
    )

    print(f"\n✓ Created DeciLM config with {len(block_configs)} layers")
    print(
        "✓ Global per-layer keys (intermediate_size, num_key_value_heads, hidden_act, sliding_window)"
    )
    print("  will be removed from saved config and are only in block_configs")

    return decilm_config


def convert_configs_in_dirs(input_dir, output_dir):
    """Convert the config of a Llama3 model to a DeciLM model."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_config_path = input_dir / "config.json"
    config = LlamaConfig.from_pretrained(input_config_path)
    decilm_config = convert_llama3_config_to_decilm_config(config)
    decilm_config.save_pretrained(output_dir)


def convert_llama3_to_decilm(input_dir, output_dir):
    """Convert a Llama3 model to a DeciLM model."""
    convert_configs_in_dirs(input_dir, output_dir)
    copy_tokenizer(input_dir, output_dir)
    convert_model_weights_to_decilm(input_dir, output_dir)
    copy_deci_lm_hf_code(output_dir)


if __name__ == "__main__":
    Fire(convert_llama3_to_decilm)
