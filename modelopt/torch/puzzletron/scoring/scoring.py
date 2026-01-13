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

"""Validates and scores model compression solutions by evaluating puzzle solution candidates."""

# mypy: ignore-errors
import os
import re
from glob import glob

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.hydra_utils import register_hydra_resolvers
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.validate_puzzle_with_multi_replacements import (
    validate_puzzle_solutions,
)


def extract_solution_id(filename):
    pattern = r"solution_(\d+)\.json"
    match = re.search(pattern, filename)

    if match:
        solution_id = match.group(1)
        return int(solution_id)
    else:
        mprint(f"Couldn't extract solutions_id from file {filename}")


def find_missing_solutions(solutions_df, validation_dir):
    all_solutions = np.arange(solutions_df.shape[0])

    benchmarked_solutions = list(glob(f"{validation_dir}/solution*.json"))
    benchmarked_solutions = [
        extract_solution_id(os.path.basename(s)) for s in benchmarked_solutions
    ]
    benchmarked_solutions = [s for s in benchmarked_solutions if s is not None]

    unbenchmarked_solutions = np.setdiff1d(all_solutions, benchmarked_solutions)
    return unbenchmarked_solutions.tolist()


def get_solutions_to_validate(cfg: DictConfig):
    _solutions_to_validate = cfg.scoring.solutions_to_validate
    if _solutions_to_validate is None:
        single_block_replacement_solutions = pd.read_json(cfg.scoring.solutions_path)
        if cfg.scoring.skip_existing_solutions:
            _solutions_to_validate = find_missing_solutions(
                single_block_replacement_solutions, cfg.scoring.output_dir
            )
        else:
            _solutions_to_validate = np.arange(single_block_replacement_solutions.shape[0]).tolist()
    return _solutions_to_validate


def launch_scoring(cfg: DictConfig):
    cfg.scoring.solutions_to_validate = get_solutions_to_validate(cfg)
    mprint(f"Solutions to validate: {cfg.scoring.solutions_to_validate}")
    validate_puzzle_solutions(args=cfg.scoring)


@hydra.main("", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    mprint(cfg)
    dist.setup(timeout=cfg.nccl_timeout_minutes)
    launch_scoring(cfg)
    dist.cleanup()


if __name__ == "__main__":
    register_hydra_resolvers()
    main()
