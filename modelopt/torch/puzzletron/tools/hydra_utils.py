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

"""
Utilities for hydra config initialization.
"""

import datetime
import random
from pathlib import Path

from hydra import compose, initialize, initialize_config_dir
from hydra.utils import get_object
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "register_hydra_resolvers",
    "initialize_hydra_config_for_dir",
    "initialize_hydra_config",
]


def warmup_steps(tokens: int, block: int, mbs: int, grad_accum: int, pct: float) -> int:
    """
    Calculate warmup steps in optimizer-step units.

    total_iters = tokens / (block * mbs) gives micro-batches; one optimizer step
    consumes ``grad_accum`` micro-batches, so total optimizer steps = total_iters
    / grad_accum. The LR scheduler in ``_get_lr`` is indexed by ``step_num``
    (optimizer steps), so warmup must be in the same units.
    """
    grad_accum = int(grad_accum)
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum!r}")
    iters = (int(tokens) // int(block)) // int(mbs)
    steps = max(1, iters // grad_accum)
    w = pct * steps
    return max(1, round(w))


def _warmup_steps_resolver(*args):
    if len(args) != 5:
        raise ValueError(
            "warmup_steps resolver expects exactly 5 arguments: "
            "(tokens, block, micro_batch_size, grad_accumulation_steps, warmup_ratio)"
        )
    return warmup_steps(*args)


def register_hydra_resolvers():
    OmegaConf.register_new_resolver("to_path", lambda x: Path(x))
    OmegaConf.register_new_resolver(
        "random_int", lambda low, high: random.randint(int(low), int(high))
    )
    OmegaConf.register_new_resolver(
        "timedelta_minutes", lambda x: datetime.timedelta(minutes=x) if x is not None else None
    )
    OmegaConf.register_new_resolver("warmup_steps", _warmup_steps_resolver)
    OmegaConf.register_new_resolver("get_object", lambda x: get_object(x))


def initialize_hydra_config_for_dir(
    config_dir: str, config_name: str, overrides: list[str]
) -> DictConfig:
    """Initialize a hydra config from an absolute path for a config directory

    Args:
        config_dir (str):
        config_name (str):
        overrides (List[str]):

    Returns:
        DictConfig:
    """

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        args = compose(config_name, overrides)
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args


def initialize_hydra_config(config_path: str, config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        args = compose(config_name, overrides)
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args
