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

import os
from pathlib import Path

import torch
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase

import modelopt.torch.puzzletron as mtpz
import modelopt.torch.utils.distributed as dist
from modelopt.torch.export import copy_hf_ckpt_remote_code


def setup_test_model_and_data(
    tmp_path: Path, rank: int, hf_model_name: str, hybrid_override_pattern: str | None = None
) -> tuple[Path, Path, Path]:
    """
    Setup the test model and data for the puzzletron NAS search.

    Args:
        tmp_path: the temporary path to use for the test
        rank: the rank of the process
        hf_model_name: HuggingFace model card name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        hybrid_override_pattern: For NemotronH models, the layer type pattern

    Returns:
        tuple[Path, Path, Path]: the puzzle_dir, hf_checkpoint_path, dataset_path
    """
    # Register Hydra custom resolvers (needed for config resolution)
    mtpz.tools.register_hydra_resolvers()

    puzzle_dir = tmp_path / hf_model_name
    hf_checkpoint_path = puzzle_dir / f"hf_models/{hf_model_name}"
    dataset_path = puzzle_dir / "dummy_dataset"

    if rank == 0:
        save_dummy_dataset(dataset_path)

        # Create a small HF model
        tokenizer = get_tiny_tokenizer()
        create_and_save_small_hf_model(
            output_path=str(hf_checkpoint_path),
            tokenizer=tokenizer,
            hf_model_name=hf_model_name,
            hybrid_override_pattern=hybrid_override_pattern,
        )
    dist.barrier()

    return puzzle_dir, hf_checkpoint_path, dataset_path


def create_and_save_small_hf_model(
    output_path: str,
    tokenizer: PreTrainedTokenizerBase,
    hf_model_name: str,
    hybrid_override_pattern: str | None = None,
):
    """Create and save a small HuggingFace model for testing the conversion pipeline.

    Uses real HuggingFace config to preserve model-specific settings (like tie_word_embeddings),
    but shrinks size parameters for fast testing.

    Args:
        output_path: Where to save the model.
        tokenizer: Tokenizer to save alongside the model.
        hf_model_name: HuggingFace model card name (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        hybrid_override_pattern: For NemotronH models, the layer type pattern (e.g., "*-" for
            Attention+MLP, "M-" for Mamba+MLP). Must match num_hidden_layers.
    """
    # Load real HuggingFace config (preserves tie_word_embeddings, rope_scaling, etc.)
    config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)

    # Override size-related params to make it small for testing
    # Note: intermediate_size must be divisible by 256 per DeciLM config requirements
    # Note: hidden_size must give head_dim >= 8 for Flash Attention 2 compatibility

    # VL models have nested configs (text_config, vision_config)
    if hasattr(config, "text_config") and hasattr(config, "vision_config"):
        config.text_config.vocab_size = tokenizer.vocab_size
        config.text_config.hidden_size = 256
        config.text_config.intermediate_size = 512
        config.text_config.num_hidden_layers = 2
        config.text_config.num_attention_heads = 32
        config.text_config.num_key_value_heads = 8
        config.text_config.num_experts = 16  # Reduce from 128
        config.text_config.moe_intermediate_size = 256
        config.text_config.max_position_embeddings = 512
        config.vision_config.depth = 2  # Reduce from 27
        config.vision_config.hidden_size = 256
        config.vision_config.intermediate_size = 512
        config.vision_config.out_hidden_size = 256
        # TODO: this is hack, redesign converter to not read config.num_hidden_layers directly.
        # set top-level num_hidden_layers for converter compatibility
        config.num_hidden_layers = config.text_config.num_hidden_layers
    else:
        # Regular models have flat config
        config.vocab_size = tokenizer.vocab_size
        config.hidden_size = 256
        config.intermediate_size = 512
        config.num_hidden_layers = max(2, dist.size())
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.max_position_embeddings = 512

        # Fix layer_types to match num_hidden_layers (newer transformers validates this)
        if hasattr(config, "layer_types") and config.layer_types is not None:
            config.layer_types = config.layer_types[: config.num_hidden_layers]

        # Fix rope_scaling to be consistent with max_position_embeddings
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["original_max_position_embeddings"] = 256

        # NemotronH requires hybrid_override_pattern to match num_hidden_layers
        if hasattr(config, "hybrid_override_pattern") and hybrid_override_pattern is not None:
            config.hybrid_override_pattern = hybrid_override_pattern

        # Ensure pad_token_id is within vocab_size (nn.Embedding requires padding_idx < num_embeddings)
        if (
            getattr(config, "pad_token_id", None) is not None
            and config.pad_token_id >= tokenizer.vocab_size
        ):
            config.pad_token_id = 0

        # Ensure moe_latent_size is present: the native transformers NemotronH model (>=5.5)
        # accesses config.moe_latent_size but older trust_remote_code configs don't define it.
        if not hasattr(config, "moe_latent_size"):
            config.moe_latent_size = None

    # Set seed for reproducible weight initialization
    torch.manual_seed(42)

    # Create and save the model
    # Force CPU initialization for deterministic behavior (prevents NaN on RTX GPUs)
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # TODO: Consider using AutoModel.from_config instead.
    if hasattr(config, "text_config") and hasattr(config, "vision_config"):
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration._from_config(config)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Initialize weights to ensure all parameters are properly initialized
    # This prevents NaN values in uninitialized parameters (e.g., backbone.layers.1.mixer.gate.weight
    # in nemotron-3-nano-30b-a3b-base-bf16) that can occur with from_config on RTX GPU cards (not on H100)
    model.initialize_weights()

    # Fix any remaining NaN/Inf values that initialize_weights() might have missed
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_inf_mask = torch.isnan(param) | torch.isinf(param)
            param.data = torch.where(nan_inf_mask, torch.zeros_like(param), param)

    # Restore CUDA_VISIBLE_DEVICES after model creation and initialization
    if original_cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    model.to(dtype=torch.bfloat16)
    # save_original_format=False: skip transformers' revert_weight_conversion so weights are saved
    # with in-memory key names (e.g. backbone.embeddings.weight) rather than the on-disk "original"
    # format (e.g. backbone.embedding.weight for NemotronH). This avoids key mismatches in
    # load_and_shard_model which looks up shard keys from model.named_parameters().
    try:
        model.save_pretrained(output_path, save_original_format=False)
    except AttributeError:
        # Workaround: some trust_remote_code models define _tied_weights_keys in an older
        # format (returning a list) that is incompatible with transformers v5, which
        # expects _get_tied_weight_keys to return a dict. Clear tied weight keys and retry.
        for submodule in model.modules():
            if getattr(submodule, "_tied_weights_keys", None) is not None:
                submodule._tied_weights_keys = None
        model.save_pretrained(output_path, save_original_format=False)

    # Save tokenizer, config, and custom code files
    tokenizer.save_pretrained(output_path)
    config.save_pretrained(output_path)
    if hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
        copy_hf_ckpt_remote_code(hf_model_name, output_path)


def save_dummy_dataset(dataset_path: Path | str):
    """
    Save a dummy dataset for testing purposes.
    """
    # dummy sample
    sample = [
        {"role": "user", "content": "please cite Lorem Ipsum?"},
        {
            "role": "assistant",
            "content": (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed in blandit ante. "
                "Sed tempus erat urna, ac elementum nisl facilisis quis. Aliquam consectetur mollis massa, "
                "in elementum sem venenatis posuere. Fusce lorem arcu, egestas vel massa sollicitudin, "
                "dictum mollis purus. Proin in ullamcorper elit. Nam tellus nisi, volutpat a mattis vel, "
                "pretium in purus. Nunc at lectus facilisis risus scelerisque rhoncus eu nec ex. "
                "Maecenas semper, tellus non placerat vulputate, urna felis facilisis diam, "
                "sit amet vestibulum erat sapien nec libero. Praesent non massa velit. Donec faucibus mi eros. "
                "Nam turpis nulla, congue sit amet mi at, porttitor scelerisque elit. Nunc id sodales lorem, "
                "nec tincidunt leo. Quisque a neque nec ligula porttitor auctor. "
                "Nunc accumsan nunc ac tellus congue vehicula. Praesent tellus eros, luctus non gravida dapibus, "
                "faucibus eu ex. Quisque bibendum leo pharetra, tristique est vitae, hendrerit nunc. "
                "Duis nec congue dolor. Donec commodo ipsum non efficitur volutpat. "
                "Nulla risus nulla, efficitur et urna at, imperdiet sodales lorem. "
                "Suspendisse erat est, sollicitudin at nisl tincidunt, vehicula hendrerit lectus. "
                "Nam quis nisi ullamcorper, rhoncus massa vel, tempus purus. "
                "Duis pulvinar eros vel nulla pellentesque, at dapibus justo laoreet. "
                "Praesent tortor orci, vulputate fermentum dapibus nec, feugiat vitae tortor. "
                "Donec mollis convallis massa quis iaculis."
            ),
        },
    ]

    # Prepare train and val splits with sample repeated, 2500 samples are for
    # 128 samples with block-size 8192 and LLama3 tokenizer
    data = [{"conversation": sample}] * 2500

    # For train-val splits
    data_dict = DatasetDict({"train": Dataset.from_list(data), "valid": Dataset.from_list(data)})
    data_dict.save_to_disk(str(dataset_path))
