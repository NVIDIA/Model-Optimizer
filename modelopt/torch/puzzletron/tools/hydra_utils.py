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
    "clone_hydra_config",
]


def warmup_steps(tokens: int, block: int, mbs: int, grad_accum: int = 1, pct: float = 0.05) -> int:
    """
    Calculate warmup steps in optimizer-step units.

    total_iters = tokens / (block * mbs) gives micro-batches; one optimizer step
    consumes ``grad_accum`` micro-batches, so total optimizer steps = total_iters
    / grad_accum. The LR scheduler in ``_get_lr`` is indexed by ``step_num``
    (optimizer steps), so warmup must be in the same units.
    """
    try:
        tokens = int(tokens)
        block = int(block)
        mbs = int(mbs)
        grad_accum = int(grad_accum)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tokens, block, mbs, and grad_accum must be integers or castable to int; "
            f"got tokens={tokens!r}, block={block!r}, mbs={mbs!r}, grad_accum={grad_accum!r}"
        ) from exc

    try:
        pct = float(pct)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pct must be a float or castable to float, got {pct!r}") from exc

    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens!r}")
    if block <= 0:
        raise ValueError(f"block must be > 0, got {block!r}")
    if mbs <= 0:
        raise ValueError(f"mbs must be > 0, got {mbs!r}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {grad_accum!r}")
    if not 0.0 <= pct <= 1.0:
        raise ValueError(f"pct must be between 0.0 and 1.0 inclusive, got {pct!r}")

    iters = (tokens // block) // mbs
    steps = max(1, iters // grad_accum)
    w = pct * steps
    return max(1, round(w))


def _warmup_steps_resolver(*args):
    if len(args) == 3:
        return warmup_steps(*args)
    if len(args) == 4:
        tokens, block, mbs, pct = args
        return warmup_steps(tokens, block, mbs, pct=pct)
    if len(args) == 5:
        return warmup_steps(*args)
    raise ValueError(
        "warmup_steps resolver expects 3, 4, or 5 arguments: "
        "(tokens, block, micro_batch_size), "
        "(tokens, block, micro_batch_size, warmup_ratio), or "
        "(tokens, block, micro_batch_size, grad_accumulation_steps, warmup_ratio)"
    )


def _register_resolver(name, resolver):
    try:
        OmegaConf.register_new_resolver(name, resolver, replace=True)
        return
    except TypeError:
        pass

    if hasattr(OmegaConf, "has_resolver") and OmegaConf.has_resolver(name):
        if hasattr(OmegaConf, "clear_resolver"):
            OmegaConf.clear_resolver(name)
        else:
            return
    try:
        OmegaConf.register_new_resolver(name, resolver)
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise


def register_hydra_resolvers():
    _register_resolver("to_path", lambda x: Path(x))
    _register_resolver("random_int", lambda low, high: random.randint(int(low), int(high)))
    _register_resolver(
        "timedelta_minutes", lambda x: datetime.timedelta(minutes=x) if x is not None else None
    )
    _register_resolver("warmup_steps", _warmup_steps_resolver)
    _register_resolver("get_object", lambda x: get_object(x))


def clone_hydra_config(config, *, resolve: bool = True) -> DictConfig:
    """Clone a Puzzletron config without dropping resolved Python objects.

    Puzzletron enables ``allow_objects`` before resolving Hydra so registered
    mixin classes, dataset callables, ``Path`` values, and similar runtime
    objects are valid configuration values. Derived stage configs must preserve
    the same contract instead of rebuilding a primitive-only ``DictConfig``.
    """
    content = (
        OmegaConf.to_container(config, resolve=resolve)
        if OmegaConf.is_config(config)
        else config
    )
    cloned = OmegaConf.create(content, flags={"allow_objects": True})
    OmegaConf.set_struct(cloned, False)
    return cloned


def _normalize_puzzletron_overrides(overrides: list[str]) -> list[str]:
    """Make Puzzletron CLI overrides tolerant of runtime-only keys.

    Hydra applies CLI overrides while the composed config is still strict. Puzzletron's
    ``--override`` flag is used mostly for runtime values produced by launch scripts
    (teacher paths, recipes, timeouts), so requiring every such key to be present in
    every config makes the clean hierarchical configs brittle. Treat plain
    ``KEY=VALUE`` overrides as Hydra ``++KEY=VALUE`` overrides, which update existing
    keys or add missing runtime keys. Explicit Hydra operators are preserved.
    """
    normalized = []
    for override in overrides or []:
        text = str(override)
        stripped = text.lstrip()
        if not stripped or stripped.startswith(("+", "~")) or "=" not in stripped:
            normalized.append(text)
            continue
        prefix_len = len(text) - len(stripped)
        normalized.append(f"{text[:prefix_len]}++{stripped}")
    return normalized


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
        args = compose(config_name, _normalize_puzzletron_overrides(overrides))
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args


def initialize_hydra_config(config_path: str, config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        args = compose(config_name, _normalize_puzzletron_overrides(overrides))
        args._set_flag("allow_objects", True)
        OmegaConf.resolve(args)  # resolve object attributes
        OmegaConf.set_struct(args, False)

    return args
