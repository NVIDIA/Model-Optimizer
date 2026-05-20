# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""YAML-driven configuration loading for fastgen distillation pipelines.

YAML is the first-class entry point for DMD-on-Wan configurations — the fastgen library
does not expect callers to hand-build Python dicts. Typical usage::

    from modelopt.torch.fastgen import DMDConfig, load_dmd_config

    # (a) Load a built-in recipe by relative path
    cfg = load_dmd_config("general/distillation/dmd2_wan22_5b")

    # (b) Load a user-provided file
    cfg = load_dmd_config("/path/to/my_dmd.yaml")

    # (c) Equivalent classmethod
    cfg = DMDConfig.from_yaml("/path/to/my_dmd.yaml")

The loader resolves paths in two places, in order:

1. ``modelopt_recipes/`` (the built-in recipes package shipped with ModelOpt) — resolved
   via :func:`importlib.resources.files`. Suffixes ``.yml`` / ``.yaml`` may be omitted.
2. The filesystem (absolute or working-directory-relative).

Suffixes ``.yml`` and ``.yaml`` are both accepted.
"""

from __future__ import annotations

import contextlib
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

# ``Traversable`` moved out of ``importlib.abc`` in Python 3.11. We only need it for
# type hints, but suppress ImportError so older runtimes can still import this module.
with contextlib.suppress(ImportError):
    from importlib.resources.abc import Traversable

import yaml

from .config import DMDConfig

if TYPE_CHECKING:
    from importlib.abc import Traversable

__all__ = ["load_config", "load_dmd_config"]


# Root to all built-in recipes shipped with modelopt.
_BUILTIN_RECIPES_LIB = files("modelopt_recipes")


_SUFFIXES = (".yml", ".yaml")


def _candidate_paths(config_file: str | Path) -> list[Path | Traversable]:
    """Return the ordered list of locations to probe for ``config_file``."""
    candidates: list[Path | Traversable] = []

    # Normalize to string for suffix probing; keep Path/Traversable behavior otherwise.
    if isinstance(config_file, str):
        base = config_file
        if base.endswith(_SUFFIXES):
            candidates.append(Path(base))
            candidates.append(_BUILTIN_RECIPES_LIB.joinpath(base))
        else:
            candidates.extend(Path(base + suffix) for suffix in _SUFFIXES)
            candidates.extend(_BUILTIN_RECIPES_LIB.joinpath(base + suffix) for suffix in _SUFFIXES)
    elif isinstance(config_file, Path):
        if config_file.suffix in _SUFFIXES:
            candidates.append(config_file)
            if not config_file.is_absolute():
                candidates.append(_BUILTIN_RECIPES_LIB.joinpath(str(config_file)))
        else:
            candidates.extend(Path(str(config_file) + suffix) for suffix in _SUFFIXES)
            if not config_file.is_absolute():
                candidates.extend(
                    _BUILTIN_RECIPES_LIB.joinpath(str(config_file) + suffix) for suffix in _SUFFIXES
                )
    else:
        raise TypeError(
            f"Expected str or Path for config_file, got {type(config_file).__name__!r}."
        )
    return candidates


def load_config(config_file: str | Path) -> dict[str, Any]:
    """Load a YAML file and return the parsed mapping.

    Mirrors :func:`modelopt.recipe._config_loader.load_config` in spirit but without
    the ExMy-num-bits post-processing that is specific to quantization recipes.

    Args:
        config_file: YAML path. Suffix is optional; resolution searches the built-in
            ``modelopt_recipes/`` package first, then the filesystem.

    Returns:
        The parsed dictionary. An empty file yields ``{}``.
    """
    for candidate in _candidate_paths(config_file):
        if candidate.is_file():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise ValueError(
                    f"Config file {candidate!s} must contain a YAML mapping, got {type(data).__name__}."
                )
            return data
    raise FileNotFoundError(
        f"Cannot locate config file {config_file!r}; searched both the built-in "
        f"recipe library and the filesystem."
    )


def load_dmd_config(config_file: str | Path) -> DMDConfig:
    """Load a YAML file and construct a :class:`DMDConfig`.

    The YAML is validated against :class:`DMDConfig`'s Pydantic schema — unknown keys
    raise ``ValidationError``.

    Example YAML::

        pred_type: flow
        guidance_scale: 5.0
        student_sample_steps: 2
        gan_loss_weight_gen: 0.03
        sample_t_cfg:
          time_dist_type: shifted
          t_list: [0.999, 0.833, 0.0]
        ema:
          decay: 0.9999
    """
    data = load_config(config_file)
    return DMDConfig(**data)
