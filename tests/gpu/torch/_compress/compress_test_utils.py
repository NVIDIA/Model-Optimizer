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
import shutil
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress.tools.hydra_utils import register_hydra_resolvers

# Path to HF configs relative to this file
HF_CONFIGS_DIR = Path(__file__).parent / "resources/hf_configs"


def setup_test_model_and_data(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    hf_config_name: str,
    hybrid_override_pattern: str | None = None,
) -> tuple[Path, Path, Path]:
    """
    Setup the test model and data for the compress NAS search.

    Args:
        project_root_path (Path): the root path of the project
        tmp_path (Path): the temporary path to use for the test
        rank (int): the rank of the process
        hf_config_name (str): Name of the HF config directory (e.g., "llama_3_1_8b_instruct")
        hybrid_override_pattern (str): For NemotronH models, the layer type pattern

    Returns:
        tuple[Path, Path, Path]:
        the puzzle_dir, hf_checkpoint_path, dataset_path
    """

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    # The inputs for the nas.convert() step.
    #
    puzzle_dir = tmp_path / hf_config_name
    hf_checkpoint_path = puzzle_dir / f"hf_models/{hf_config_name}"
    dataset_path = puzzle_dir / "dummy_dataset"

    if rank == 0:
        # Setup puzzle_dir and dataset
        setup_puzzle_dir(puzzle_dir)
        save_dummy_dataset(dataset_path)

        # Create a small HF model
        tokenizer = create_tokenizer(project_root_path)
        create_and_save_small_hf_model(
            output_path=str(hf_checkpoint_path),
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            hf_config_name=hf_config_name,
            hybrid_override_pattern=hybrid_override_pattern,
        )
    dist.barrier()

    return (
        puzzle_dir,
        hf_checkpoint_path,
        dataset_path,
    )


def create_and_save_small_hf_model(
    output_path: str,
    vocab_size: int,
    tokenizer: PreTrainedTokenizerBase,
    hf_config_name: str,
    hybrid_override_pattern: str | None = None,
):
    """
    Create and save a small HuggingFace model for testing the conversion pipeline.
    Uses real HuggingFace config to preserve model-specific settings (like tie_word_embeddings),
    but shrinks size parameters for fast testing.

    Args:
        output_path: Where to save the model
        vocab_size: Vocabulary size (should match tokenizer)
        tokenizer: Tokenizer to save alongside the model
        hf_config_name: Name of the config directory under resources/hf_configs/
                        e.g., "llama_3_1_8b_instruct", "llama_3_2_3b_instruct", or "qwen2_5_7b_instruct"
        hybrid_override_pattern: For NemotronH models, the layer type pattern (e.g., "*-" for Attention+MLP,
                                 "M-" for Mamba+MLP). Must match num_hidden_layers. None for non-NemotronH models.
    """
    os.makedirs(output_path, exist_ok=True)

    # Load real HuggingFace config (preserves tie_word_embeddings, rope_scaling, etc.)
    config_path = HF_CONFIGS_DIR / hf_config_name
    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=True)

    # Override size-related params to make it small for testing
    # Note: intermediate_size must be divisible by 256 per DeciLM config requirements
    # Note: hidden_size must give head_dim >= 8 for Flash Attention 2 compatibility

    # VL models have nested configs (text_config, vision_config)
    if hf_config_name == "qwen3-vl-30b-a3b-instruct":
        config.text_config.vocab_size = vocab_size
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
        config.vocab_size = vocab_size
        config.hidden_size = 256
        config.intermediate_size = 512
        config.num_hidden_layers = 2
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.max_position_embeddings = 512

        # Fix layer_types to match num_hidden_layers (newer transformers validates this)
        if hasattr(config, "layer_types") and config.layer_types is not None:
            config.layer_types = config.layer_types[:2]

        # Fix rope_scaling to be consistent with max_position_embeddings
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["original_max_position_embeddings"] = 256

        # NemotronH requires hybrid_override_pattern to match num_hidden_layers
        if hasattr(config, "hybrid_override_pattern") and hybrid_override_pattern is not None:
            config.hybrid_override_pattern = hybrid_override_pattern

    # Set seed for reproducible weight initialization
    torch.manual_seed(42)

    # Create and save the model
    # TODO: Consider using AutoModel.from_config instead.
    if hf_config_name == "qwen3-vl-30b-a3b-instruct":
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration._from_config(config)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.to(dtype=torch.bfloat16).save_pretrained(output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save config
    config.save_pretrained(output_path)


def create_tokenizer(project_root_path: Path) -> PreTrainedTokenizerBase:
    """
    Create a tokenizer for the model.
    """
    tokenizer_path = project_root_path / "tests/gpu/torch/_compress/resources/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def setup_puzzle_dir(puzzle_dir: str | Path):
    """
    Setup puzzle directory by removing existing directory and creating a new one.
    """
    puzzle_dir = Path(puzzle_dir)
    if puzzle_dir.exists():
        shutil.rmtree(puzzle_dir)
    puzzle_dir.mkdir(parents=True, exist_ok=True)


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
