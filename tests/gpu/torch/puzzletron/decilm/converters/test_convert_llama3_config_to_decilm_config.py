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
from pathlib import Path

from _test_utils.torch.puzzletron.utils import create_and_save_small_llama_model, create_tokenizer

from modelopt.torch.puzzletron.decilm.converters.convert_llama3_to_decilm import (
    convert_llama3_to_decilm,
)


def test_convert_llama3_config_to_decilm_config(project_root_path: Path, tmp_path: Path):
    tokenizer = create_tokenizer(project_root_path)
    llama_checkpoint_path = tmp_path / "llama_checkpoint"
    create_and_save_small_llama_model(
        llama_checkpoint_path, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
    )

    # Convert the Llama model to a DeciLM model
    decilm_checkpoint_path = tmp_path / "decilm_checkpoint"
    convert_llama3_to_decilm(
        input_dir=llama_checkpoint_path,
        output_dir=decilm_checkpoint_path,
    )

    # Assert that the converted config has the correct number of block_configs
    config_path = decilm_checkpoint_path / "config.json"
    assert config_path.exists(), f"Config file not found at {config_path}"

    with open(config_path) as f:
        decilm_config = json.load(f)

    # Verify block_configs exists and has the correct length
    assert "block_configs" in decilm_config, "block_configs not found in converted config"
    actual_num_block_configs = len(decilm_config["block_configs"])
    assert actual_num_block_configs == 2
