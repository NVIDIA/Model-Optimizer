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
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data

import modelopt.torch.nas as mtn
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.nas.plugins.puzzletron_nas_plugin import PuzzletronModel


def test_nas_search(project_root_path: Path, tmp_path: Path):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nas_search_multiprocess_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _test_nas_search_multiprocess_job(
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
    converted_model = mtn.convert(
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
    # Run the mnt.search() step
    #
    mtn.search(
        converted_model,
        constraints={},  # this is not used as the search space is defined in the hydra config
        dummy_input=None,  # Not used
        config={},  # this is not used as the search space is defined in the hydra config
    )

    #
    # Check assertions for mtn.search() step
    #
    if rank == 0:
        # assertions for the build_library_and_stats step
        assert (puzzle_dir / "replacement_library.json").is_file()
        assert (puzzle_dir / "subblock_stats.json").is_file()

        # assertions for the scoring step
        solution_0_filepath = (
            puzzle_dir / "single_sequence_replacement_solutions--validation/solution_0.json"
        )

        assert solution_0_filepath.exists()

        # assertions for the mip_and_realize_models step
        solution_0_ckpt_config_path = (
            puzzle_dir
            / "mip/puzzle_solutions/target_memory_780000MiB/solutions--checkpoints/solution_0/config.json"
        )

        assert solution_0_ckpt_config_path.exists()
        assert (puzzle_dir / "mip/puzzle_solutions/target_memory_780000MiB/solutions.json").exists()

    dist.cleanup()
