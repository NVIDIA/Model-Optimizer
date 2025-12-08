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
from typing import List

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from modelopt.torch._compress.mip.run_puzzle import run_puzzle
from modelopt.torch._compress.tools.hydra_utils import register_hydra_resolvers
from modelopt.torch._compress.tools.logger import mprint
from modelopt.torch._compress.tools.runtime import BaseRuntime, IRuntime, NativeDdpRuntime
from modelopt.torch._compress.tools.validate_puzzle_with_multi_replacements import (
    validate_puzzle_solutions,
)
from modelopt.torch._compress.utils.dist_utils import is_distributed


def launch_mip(cfg: DictConfig) -> List[str]:
    solution_paths = run_puzzle(args=cfg.mip)
    return solution_paths


def launch_realize_model(cfg: DictConfig, runtime: IRuntime):
    validate_puzzle_solutions(args=cfg.realize_model, runtime=runtime)


def launch_mip_and_realize_model(cfg: DictConfig, runtime: IRuntime):
    # Determine device for distributed operations (NCCL requires CUDA tensors)
    device = "cpu"
    if runtime.world_size > 1 and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl":
            device = torch.cuda.current_device()

    if runtime.is_main_process:
        solution_paths = launch_mip(cfg)
        length_tensor = torch.tensor([len(solution_paths)], dtype=torch.long, device=device)
    else:
        solution_paths = None
        length_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if not cfg.skip_realize_model:
        if runtime.world_size > 1:
            dist.broadcast(length_tensor, src=0)

        list_length = length_tensor.item()

        if runtime.global_rank != 0:
            solution_paths = [None] * list_length

        if runtime.world_size > 1:
            dist.broadcast_object_list(solution_paths, src=0)

        for solution_path in solution_paths:
            mprint(f"Realize model for the solution: {solution_path}")
            cfg.realize_model.solutions_path = Path(solution_path)
            launch_realize_model(cfg, runtime=runtime)
            runtime.wait_for_everyone()


@hydra.main("", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)

    _runtime = (
        NativeDdpRuntime(
            dtype=torch.bfloat16, torch_distributed_timeout=getattr(cfg, "nccl_timeout_minutes")
        )
        if is_distributed()
        else BaseRuntime(dtype=torch.bfloat16)
    )
    with _runtime as runtime:
        launch_mip_and_realize_model(cfg, runtime)


if __name__ == "__main__":
    register_hydra_resolvers()
    main()
