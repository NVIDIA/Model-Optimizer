# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from _test_utils.examples.run_command import MODELOPT_ROOT, run_example_command

DFLASH_YAML = str(
    MODELOPT_ROOT / "modelopt_recipes" / "general" / "speculative_decoding" / "dflash.yaml"
)


def test_offline_dflash_context_parallel(tiny_llama_path, tmp_path, num_gpus):
    """Train an offline DFlash draft with the context split over two GPUs."""
    if num_gpus < 2:
        pytest.skip(f"cp_size=2 requires at least 2 GPUs, but only {num_gpus} found.")

    data_dir = tmp_path / "hidden_states"
    data_dir.mkdir()
    torch.manual_seed(0)
    for i in range(8):
        sample = {
            "input_ids": torch.randint(1, 32, (128,)),
            "hidden_states": torch.randn(128, 512),
            "aux_hidden_states": torch.randn(128, 512),  # one draft layer
        }
        torch.save(sample, data_dir / f"sample_{i}.pt")

    run_example_command(
        [
            "./launch_train.sh",
            "--config",
            DFLASH_YAML,
            f"model.model_name_or_path={tiny_llama_path}",
            f"data.offline_data_path={data_dir}",
            f"training.output_dir={tmp_path / 'ckpt'}",
            "training.num_train_epochs=1",
            "training.training_seq_len=128",
            "training.answer_only_loss=false",
            "training.cp_size=2",
            "training.tf32=false",  # the recipe's tf32 needs Ampere or newer
            "dflash.dflash_mask_token_id=0",
            "dflash.dflash_architecture_config.num_hidden_layers=1",
        ],
        "speculative_decoding",
    )
