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

from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from gpu.torch._compress.compress_test_utils import setup_test_model_and_data

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress import compress
from modelopt.torch._compress.decilm.converters.convert_llama3_to_decilm import (
    convert_llama3_to_decilm,
)

# The e2e test to compress a model based on Local Neural Architecture Search (Mixed Integer Programing NAS search)
# using a one-click command.
#
# Note: Bypass is disabled now in the test.


def test_compress(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=min(torch.cuda.device_count(), 2),  # assertions configured for atmost 2 GPUs
        job=partial(_test_compress_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_compress_multiprocess_job(project_root_path: Path, tmp_path: Path, rank: int, size: int):
    dist.setup(timeout=timedelta(10))
    # Setup the test model and data.
    puzzle_dir, llama_checkpoint_path, dataset_path = setup_test_model_and_data(
        project_root_path, tmp_path, rank
    )
    hydra_config_dir = project_root_path / "tests/gpu/torch/_compress/resources/configs"
    hydra_config_name = "Llama-3_1-8B-ffn-pruning"

    # Convert the Llama model to DeciLM model.
    if rank == 0:
        convert_llama3_to_decilm(
            input_dir=llama_checkpoint_path,
            output_dir=puzzle_dir / "ckpts/teacher",
        )
    dist.barrier()

    # Compress the model using a one-click approach
    compress.compress(str(hydra_config_dir), hydra_config_name, str(puzzle_dir), str(dataset_path))

    #
    # Check assertions
    #
    # assertions for the score_pruning_activations step 1
    _assert_score_pruning_activations(puzzle_dir)
    if rank == 0:
        # assertions for the pruning_ckpts step 2
        assert (puzzle_dir / "ckpts/ffn_256_attn_no_op").exists()

        # assertions for the build_library_and_stats step 4

        assert (puzzle_dir / "replacement_library.json").is_file()
        assert (puzzle_dir / "subblock_stats.json").is_file()

        # assertions for the scoring step 5
        solution_0_filepath = (
            puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
        )

        assert solution_0_filepath.exists()

        # assertions for the mip_and_realize_models step 6
        solution_0_ckpt_config_path = (
            puzzle_dir
            / "mip/puzzle_solutions/target_memory_780000MiB/solutions--checkpoints/solution_0/config.json"
        )

        assert solution_0_ckpt_config_path.exists()
        assert (puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB/solutions.json").exists()

    dist.cleanup()

    print(
        "PYTEST SUMMARY: test_compress_model() test has finished successfully. Puzzle directory: ",
        puzzle_dir,
    )


def _assert_score_pruning_activations(puzzle_dir: Path):
    """Assertions for the score_pruning_activations step 1."""
    rank = dist.rank()
    size = dist.size()
    rank_filepath = f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
    assert (puzzle_dir / rank_filepath).is_file()

    pruning_scores = torch.load(puzzle_dir / rank_filepath)

    layer_names = list(pruning_scores.keys())
    assert len(layer_names) == 2 // size

    if size == 1 or rank == 0:
        # Check specific values for layer 0
        layer_0 = pruning_scores[layer_names[0]]
        assert layer_0["score"][0].item() == 371
        assert layer_0["channels_importance_ascending"][0].item() == 140

    if size == 1 or rank == 1:
        # Check specific values for layer 1
        layer_1 = pruning_scores[layer_names[1 if size == 1 else 0]]
        assert layer_1["score"][0].item() == 269
        assert layer_1["channels_importance_ascending"][0].item() == 366
