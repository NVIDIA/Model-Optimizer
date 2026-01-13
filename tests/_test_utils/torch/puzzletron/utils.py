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
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.hydra_utils import register_hydra_resolvers


def setup_test_model_and_data(
    project_root_path: Path, tmp_path: Path, rank: int
) -> tuple[Path, Path, Path]:
    """
    Setup the test model and data for the puzzletron NAS search.

    Args:
        project_root_path (Path): the root path of the project
        tmp_path (Path): the temporary path to use for the test
        rank (int): the rank of the process

    Returns:
        tuple[Path, Path, Path]:
        the puzzle_dir, llama_checkpoint_path, dataset_path
    """

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    # The inputs for the nas.convert() step.
    #
    puzzle_dir = tmp_path
    llama_checkpoint_path = puzzle_dir / "input_model/llama"
    dataset_path = puzzle_dir / "dummy_dataset"

    if rank == 0:
        # Setup puzzle_dir and dataset
        setup_puzzle_dir(puzzle_dir)
        save_dummy_dataset(dataset_path)

        # Create a small Llama model
        tokenizer = create_tokenizer(project_root_path)
        create_and_save_small_llama_model(
            llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
        )
    dist.barrier()

    return (
        puzzle_dir,
        llama_checkpoint_path,
        dataset_path,
    )


def create_and_save_small_llama_model(
    output_path: str, vocab_size: int, tokenizer: PreTrainedTokenizerBase
):
    """
    Create and save a small Llama model for testing the conversion pipeline.
    This mimics having a real Llama checkpoint that needs to be converted.
    """
    os.makedirs(output_path, exist_ok=True)

    # Create a minimal Llama config (small for testing)
    # Note: intermediate_size must be divisible by 256 per DeciLM config requirements
    # Note: hidden_size must give head_dim >= 8 for Flash Attention 2 compatibility
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,  # 32 heads times 8 head_dim = 256 (matches bypass config expectations)
        intermediate_size=512,  # Must be divisible by 256
        num_hidden_layers=2,
        num_attention_heads=32,  # Matches original test
        num_key_value_heads=8,  # GQA: 32÷4=8 (matches original n_heads_in_group=4)
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
    )

    # Create and save the Llama model
    model = LlamaForCausalLM(llama_config)
    model.to(dtype=torch.bfloat16).save_pretrained(output_path)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save config
    llama_config.save_pretrained(output_path)


def create_tokenizer(project_root_path: Path) -> PreTrainedTokenizerBase:
    """
    Create a tokenizer for the Llama model.
    """
    tokenizer_path = project_root_path / "tests/_test_utils/torch/puzzletron/resources/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def setup_puzzle_dir(puzzle_dir: str):
    """
    Setup puzzle directory by removing existing directory and creating a new one.
    """
    if Path(puzzle_dir).exists():
        shutil.rmtree(puzzle_dir)
        Path(puzzle_dir).mkdir(parents=True, exist_ok=True)


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
