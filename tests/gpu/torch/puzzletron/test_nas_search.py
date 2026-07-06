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

from pathlib import Path

from _test_utils.torch.puzzletron.utils import setup_test_model_and_data

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron as mtpz


def test_nas_search(dist_workers, project_root_path: Path, tmp_path: Path):
    dist_workers.run(_test_nas_search_multiprocess_job, project_root_path, tmp_path)


def _test_nas_search_multiprocess_job(
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

    # NOTE: no dist.cleanup() here — the dist_workers pool owns the process-group
    # lifecycle and reuses it across tests.
