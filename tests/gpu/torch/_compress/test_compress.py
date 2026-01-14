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
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from gpu.torch._compress.compress_test_utils import setup_test_model_and_data

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress import compress
from modelopt.torch._compress.anymodel import convert_model

# The e2e test to compress a model based on Local Neural Architecture Search (Mixed Integer Programing NAS search)
# using a one-click command.
#
# Note: Bypass is disabled now in the test.


@pytest.mark.parametrize(
    (
        "hf_config_name",
        "converter",
        "hydra_config_subdir",
        "hybrid_override_pattern",
        "has_moe_layers",
    ),
    [
        ("llama_3_1_8b_instruct", "llama", "llama_3_1_8b_instruct", None, False),
    ],
)
def test_compress(
    project_root_path: Path,
    tmp_path: Path,
    hf_config_name: str,
    converter: str,
    hydra_config_subdir: str,
    hybrid_override_pattern: str,
    has_moe_layers: bool,
):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_compress_multiprocess_job,
            project_root_path,
            tmp_path,
            hf_config_name,
            converter,
            hydra_config_subdir,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_compress_multiprocess_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_config_name: str,
    converter: str,
    hydra_config_subdir: str,
    hybrid_override_pattern: str,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    dist.setup(timeout=timedelta(10))

    # Setup the test model and data.
    puzzle_dir, hf_checkpoint_path, dataset_path = setup_test_model_and_data(
        project_root_path, tmp_path, rank, hf_config_name, hybrid_override_pattern
    )
    hydra_config_dir = (
        project_root_path / f"tests/gpu/torch/_compress/resources/configs/{hydra_config_subdir}"
    )

    # Convert the model using AnyModel converter.
    if rank == 0:
        convert_model(
            input_dir=str(hf_checkpoint_path),
            output_dir=str(puzzle_dir / "ckpts/teacher"),
            converter=converter,
        )
    dist.barrier()

    # Compress the model using a one-click approach
    compress.compress(
        str(hydra_config_dir), hydra_config_subdir, str(puzzle_dir), str(dataset_path)
    )

    #
    # Check assertions
    #
    if rank == 0:
        # assertions for the score_pruning_activations step 1
        _assert_score_pruning_activations(puzzle_dir)

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
        _assert_mip_solutions(puzzle_dir)

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_compress({hf_config_name}) test has finished successfully. "
        f"Puzzle directory: {puzzle_dir}"
    )


def _assert_score_pruning_activations(puzzle_dir: Path):
    """Assertions for the score_pruning_activations step 1."""
    rank = dist.rank()
    rank_filepath = f"pruning/pruning_scores/ffn_iterative/100samples_diverse_mini/rank_{rank}.pth"
    assert (puzzle_dir / rank_filepath).is_file()

    pruning_scores = torch.load(puzzle_dir / rank_filepath)

    layer_names = list(pruning_scores.keys())
    assert len(layer_names) == 2

    # Check specific values for layer 0
    layer_0 = pruning_scores[layer_names[0]]
    assert layer_0["score"][0].item() == 371
    assert layer_0["channels_importance_ascending"][0].item() == 140

    # Check specific values for layer 1
    layer_1 = pruning_scores[layer_names[1]]
    assert layer_1["score"][0].item() == 269
    assert layer_1["channels_importance_ascending"][0].item() == 366


def _assert_mip_solutions(puzzle_dir: Path):
    """Assertions for the mip_and_realize_models step."""
    mip_dir = puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB"

    assert (mip_dir / "solutions.json").exists()
    assert (mip_dir / "solutions--checkpoints/solution_0/config.json").exists()

    # Check lm_loss value from solution validation
    solution_0_path = (
        puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
    )
    with open(solution_0_path) as f:
        validation = json.load(f)

    expected_lm_loss = 4.53060245513916
    actual_lm_loss = validation["lm_loss"]["avg"]
    assert abs(actual_lm_loss - expected_lm_loss) < 0.01, (
        f"lm_loss mismatch: expected {expected_lm_loss}, got {actual_lm_loss}"
    )
