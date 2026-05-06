# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions for bypass distillation."""

from pathlib import Path

from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist


def set_experiment_id(cfg: DictConfig) -> None:
    """Set the experiment ID based on the model config overrides.

    The ID encodes every override that affects the produced student so that
    sweeps over (FFN size × KV heads) or (num_experts × KV heads) get distinct
    directories instead of clobbering each other.
    """
    if cfg.bypass.experiment_id is not None:
        return

    overrides = cfg.bypass.model.model_config_overrides
    parts: list[str] = []

    if "ffn" in overrides:
        ffn_override = overrides.ffn[0]
        if "intermediate_size" in ffn_override and ffn_override["intermediate_size"] is not None:
            parts.append(f"ffn_{ffn_override['intermediate_size']}")
        elif "moe" in ffn_override and ffn_override["moe"] is not None:
            parts.append(f"experts_{ffn_override['moe']['num_local_experts']}")

    if "attention" in overrides:
        attn_override = overrides.attention[0]
        if (
            "num_key_value_heads" in attn_override
            and attn_override["num_key_value_heads"] is not None
        ):
            parts.append(f"heads_{attn_override['num_key_value_heads']}")

    if parts:
        cfg.bypass.experiment_id = "bypass_" + "_".join(parts)


def set_experiment_dir(cfg: DictConfig) -> None:
    """Set the experiment directory for the bypass run.

    Stores the path as a string in the OmegaConf node (OmegaConf only supports
    primitive types natively). Use sites should reconstruct ``Path(...)`` as needed.
    """
    experiment_dir = Path(cfg.puzzle_dir) / "bypass" / "bypass_runs" / cfg.bypass.experiment_id
    cfg.bypass.experiment_dir = str(experiment_dir)
    if dist.is_master():
        experiment_dir.mkdir(parents=True, exist_ok=True)


def get_distributed_modules_ownership(module_count: int, world_size: int) -> list[int]:
    """Map module (block) indices to GPU ranks for pipeline-parallel distribution."""
    modules_process_ownership: list[int] = []

    for i in range(world_size):
        num_modules_for_process = module_count // world_size
        if i < module_count % world_size:
            num_modules_for_process += 1

        modules_process_ownership.extend([i] * num_modules_for_process)

    return modules_process_ownership
