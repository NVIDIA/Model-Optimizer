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

from _test_utils.torch.puzzletron.utils import setup_test_model_and_data

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron as mtpz


def test_nas_convert_ffn_pruning(dist_workers, project_root_path: Path, tmp_path: Path):
    dist_workers.run(_test_nas_convert_ffn_pruning_multiprocess_job, project_root_path, tmp_path)


def _test_nas_convert_ffn_pruning_multiprocess_job(
    rank: int, size: int, project_root_path: Path, tmp_path: Path
):
    # NOTE: no dist.setup() — the dist_workers pool initializes the process group
    # and sets the cuda device.
    # Setup the test model and data.
    puzzle_dir, llama_checkpoint_path, dataset_path = setup_test_model_and_data(
        tmp_path, rank, "meta-llama/Llama-3.1-8B-Instruct"
    )
    hydra_config_dir = project_root_path / "tests/gpu/torch/puzzletron/resources/configs"
    hydra_config_name = "meta-llama/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"

    #
    # Run the mnt.convert() step
    #
    input_model = mtpz.puzzletron_nas_plugin.PuzzletronModel()
    mtn.convert(
        input_model,
        mode=[
            (
                "puzzletron",
                {
                    "puzzle_dir": str(puzzle_dir),
                    "input_model_path": str(llama_checkpoint_path),
                    "hydra_config_dir": str(hydra_config_dir),
                    "hydra_config_name": hydra_config_name,
                    "dataset_path": str(dataset_path),
                },
            )
        ],
    )

    #
    # Check assertions
    #
    if rank == 0:
        # assertions for the score_pruning_activations step
        rank = int(os.environ["RANK"])
        rank_filepath = (
            f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
        )
        assert (puzzle_dir / rank_filepath).is_file()

        # assertions for the pruning_ckpts step
        assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()

    # NOTE: no dist.cleanup() here — the dist_workers pool owns the process-group
    # lifecycle and reuses it across tests.


def test_nas_convert_attn_pruning(dist_workers, project_root_path: Path, tmp_path: Path):
    dist_workers.run(_test_nas_convert_attn_pruning_multiprocess_job, project_root_path, tmp_path)


def _test_nas_convert_attn_pruning_multiprocess_job(
    rank: int, size: int, project_root_path: Path, tmp_path: Path
):
    # NOTE: no dist.setup() — the dist_workers pool initializes the process group
    # and sets the cuda device.
    # Setup the test model and data.
    puzzle_dir, llama_checkpoint_path, dataset_path = setup_test_model_and_data(
        tmp_path, rank, "meta-llama/Llama-3.1-8B-Instruct"
    )
    hydra_config_dir = project_root_path / "tests/gpu/torch/puzzletron/resources/configs"
    hydra_config_name = "meta-llama/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-attn-pruning"

    #
    # Run the mnt.convert() step
    #
    input_model = mtpz.puzzletron_nas_plugin.PuzzletronModel()
    mtn.convert(
        input_model,
        mode=[
            (
                "puzzletron",
                {
                    "puzzle_dir": str(puzzle_dir),
                    "input_model_path": str(llama_checkpoint_path),
                    "hydra_config_dir": str(hydra_config_dir),
                    "hydra_config_name": hydra_config_name,
                    "dataset_path": str(dataset_path),
                },
            )
        ],
    )

    #
    # Check assertions
    #
    if rank == 0:
        # assertions for the score_pruning_activations step
        rank = int(os.environ["RANK"])
        rank_filepath = (
            f"pruning/pruning_scores/attn_independent_kv_head_contribution/"
            f"100samples_diverse_mini/rank_{rank}.pth"
        )
        assert (puzzle_dir / rank_filepath).is_file()

        # assertions for the pruning_ckpts step
        assert (puzzle_dir / "ckpts/n_heads_in_group8").exists()
        assert (puzzle_dir / "ckpts/n_heads_in_group16").exists()
        assert (puzzle_dir / "ckpts/n_heads_in_group32").exists()

    # NOTE: no dist.cleanup() here — the dist_workers pool owns the process-group
    # lifecycle and reuses it across tests.
