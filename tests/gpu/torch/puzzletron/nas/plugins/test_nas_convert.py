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
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data

import modelopt.torch.nas as mtn
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.nas.plugins.puzzletron_nas_plugin import PuzzletronModel


def test_nas_convert_ffn_pruning(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nas_convert_ffn_pruning_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_nas_convert_ffn_pruning_multiprocess_job(
    project_root_path: Path, tmp_path: Path, rank: int, size: int
):
    dist.setup(timeout=timedelta(10))
    # Setup the test model and data.
    puzzle_dir, llama_checkpoint_path, dataset_path = setup_test_model_and_data(
        project_root_path, tmp_path, rank
    )
    hydra_config_dir = project_root_path / "tests/_test_utils/torch/puzzletron/resources/configs"
    hydra_config_name = "Llama-3_1-8B-ffn-pruning"

    #
    # Run the mnt.convert() step
    #
    input_model = PuzzletronModel()
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

    dist.cleanup()


def test_nas_convert_attn_pruning(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nas_convert_attn_pruning_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_nas_convert_attn_pruning_multiprocess_job(
    project_root_path: Path, tmp_path: Path, rank: int, size: int
):
    dist.setup(timeout=timedelta(10))
    # Setup the test model and data.
    puzzle_dir, llama_checkpoint_path, dataset_path = setup_test_model_and_data(
        project_root_path, tmp_path, rank
    )
    hydra_config_dir = project_root_path / "tests/_test_utils/torch/puzzletron/resources/configs"
    hydra_config_name = "Llama-3_1-8B-attn-pruning"

    #
    # Run the mnt.convert() step
    #
    input_model = PuzzletronModel()
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

    dist.cleanup()
