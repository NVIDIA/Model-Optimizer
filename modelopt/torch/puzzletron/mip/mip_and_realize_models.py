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

"""Runs MIP (Mixed Integer Programming) optimization and realizes the resulting model solutions."""

# mypy: ignore-errors
from pathlib import Path

import torch
from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.mip.run_puzzle import run_puzzle
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.validate_puzzle_with_multi_replacements import (
    validate_puzzle_solutions,
)


def launch_mip(cfg: DictConfig) -> list[str]:
    solution_paths = run_puzzle(args=cfg.mip)
    return solution_paths


def launch_realize_model(cfg: DictConfig):
    validate_puzzle_solutions(args=cfg.realize_model)


def launch_mip_and_realize_model(cfg: DictConfig):
    # Determine device for distributed operations (NCCL requires CUDA tensors)
    device = "cpu"
    if dist.size() > 1:
        if torch.distributed.get_backend() == "nccl":
            device = torch.cuda.current_device()

    if dist.is_master():
        solution_paths = launch_mip(cfg)
        length_tensor = torch.tensor([len(solution_paths)], dtype=torch.long, device=device)
    else:
        solution_paths = None
        length_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if not cfg.skip_realize_model:
        if dist.size() > 1:
            torch.distributed.broadcast(length_tensor, src=0)

        list_length = length_tensor.item()

        if not dist.is_master():
            solution_paths = [None] * list_length

        if dist.size() > 1:
            torch.distributed.broadcast_object_list(solution_paths, src=0)

        for solution_path in solution_paths:
            mprint(f"Realize model for the solution: {solution_path}")
            cfg.realize_model.solutions_path = Path(solution_path)
            launch_realize_model(cfg)
            dist.barrier()
