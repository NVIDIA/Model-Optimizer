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

from omegaconf import DictConfig

import modelopt.torch.puzzletron.activation_scoring.score_pruning_activations as score_pruning_activations
import modelopt.torch.puzzletron.build_library_and_stats as build_library_and_stats
import modelopt.torch.puzzletron.mip.mip_and_realize_models as mip_and_realize_models
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.puzzletron.scoring.scoring as scoring
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.hydra_utils import initialize_hydra_config_for_dir


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

    # Step 1: score_pruning_activations (distributed processing)
    score_pruning_activations.launch_score_activations(hydra_cfg)

    # Step 2: pruning_ckpts (single process)
    if dist.is_master():
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    # Step 4: build_library_and_stats (single process)
    if dist.is_master():
        build_library_and_stats.launch_build_library_and_stats(hydra_cfg)
    dist.barrier()

    # Step 5: calc_one_block_scores (distributed processing)
    scoring.launch_scoring(hydra_cfg)

    # Step 6: mip_and_realize_models (distributed processing)
    mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg)

    return hydra_cfg
