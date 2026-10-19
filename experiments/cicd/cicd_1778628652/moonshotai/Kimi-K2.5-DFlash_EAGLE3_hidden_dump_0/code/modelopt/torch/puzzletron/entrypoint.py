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

"""This module provides the main compression function for a model using MIP-based NAS search algorithm."""

import hydra
from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist

from .activation_scoring import launch_score_activations
from .build_library_and_stats import launch_build_library_and_stats
from .mip import launch_mip_and_realize_model
from .pruning import launch_prune_ckpt
from .scoring import launch_scoring
from .tools.hydra_utils import initialize_hydra_config_for_dir

__all__ = ["puzzletron"]


def puzzletron(
    hydra_config_dir: str, hydra_config: str, puzzle_dir: str, dataset_path: str
) -> DictConfig:
    """Compress a model using the MIP-based NAS search algorithm from Puzzletron.

    Args:
        hydra_config_dir (str): path to a hydra_config_dir that defines the search space
        hydra_config (str): the corresponding hydra config file
        puzzle_dir (str): directory with a puzzletron model to compress
        dataset_path (str): dataset used for scoring and distillation

    Returns:
        Hydra config object after compressing the model.
        The same hydra configuration object is used across all compression steps.
        TODO: Investigate if this config object is immutable across steps and clarify
    """
    # Step 0: Load puzzletron hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config,
        overrides=[
            f"puzzle_dir={puzzle_dir}",
            f"dataset_path={dataset_path}",
        ],
    )
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    # Step 1: score_pruning_activations (distributed processing)
    launch_score_activations(hydra_cfg)

    # Step 2: pruning_ckpts (single process)
    if dist.is_master():
        launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    # Step 3: build_library_and_stats (single process)
    if dist.is_master():
        launch_build_library_and_stats(hydra_cfg)
    dist.barrier()

    # Step 4: calc_one_block_scores (distributed processing)
    launch_scoring(hydra_cfg)

    # Step 5: mip_and_realize_models (distributed processing)
    launch_mip_and_realize_model(hydra_cfg)

    return hydra_cfg
