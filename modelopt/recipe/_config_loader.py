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

"""YAML config loading utilities.

This module is intentionally free of ``modelopt.torch`` imports so that
``modelopt.torch.quantization.config`` can import :func:`load_config` without
triggering a circular import through ``modelopt.recipe.loader``.

In addition to plain ``yaml.safe_load`` semantics, :func:`load_config`
recognises two custom YAML tags that make it possible to share fragments
between recipes without copy-pasting them:

``!include <path>``
    Replace the node with the contents of another YAML config. The target is
    resolved with the same rules as :func:`load_config` itself: relative paths
    are tried first against the directory of the *including* file, then
    against the built-in recipe library (:data:`BUILTIN_RECIPES_LIB`). The
    ``.yml`` / ``.yaml`` suffix may be omitted. Cycles are detected and
    rejected.

``!concat [item, item, ...]``
    Flatten one level of nesting. Items that are themselves sequences are
    spliced into the result; items that are not (mappings, scalars) are
    appended as-is. This lets a recipe interleave ``!include``-d list
    fragments with inline mapping entries without a redundant extra ``-``
    layer, e.g.

    .. code-block:: yaml

        quant_cfg: !concat
          - !include _base/disable_all          # list, spliced
          - quantizer_name: '*weight_quantizer' # mapping, appended
            cfg: { num_bits: e4m3 }
          - !include _base/default_disabled_quantizers

    This is the natural complement of ``!include`` for ordered lists such
    as ``quant_cfg``, where simple YAML merge keys (``<<:``) do not apply
    because they only operate on mappings.
"""

from importlib.resources import files

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
import re
from pathlib import Path
from typing import Any

import yaml

# Root to all built-in recipes. Users can create own recipes.
BUILTIN_RECIPES_LIB = files("modelopt_recipes")

_EXMY_RE = re.compile(r"^[Ee](\d+)[Mm](\d+)$")
_EXMY_KEYS = frozenset({"num_bits", "scale_bits"})


def _parse_exmy_num_bits(obj: Any) -> Any:
    """Recursively convert ``ExMy`` strings in ``num_bits`` / ``scale_bits`` to ``(x, y)`` tuples."""
    if isinstance(obj, dict):
        return {
            k: (
                _parse_exmy(v)
                if k in _EXMY_KEYS and isinstance(v, str)
                else _parse_exmy_num_bits(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_parse_exmy_num_bits(item) for item in obj]
    return obj


def _parse_exmy(s: str) -> tuple[int, int] | str:
    m = _EXMY_RE.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return s


def _resolve_config_paths(
    config_file: str | Path | Traversable,
    base_dir: Path | Traversable | None = None,
) -> list[Path | Traversable]:
    """Return the ordered list of candidate paths for ``config_file``.

    When ``base_dir`` is provided, relative string / :class:`Path` targets are
    additionally probed against that directory first. This is what makes
    ``!include`` resolve files relative to the *including* recipe.
    """
    paths_to_check: list[Path | Traversable] = []

    def _append_with_suffixes(prefix: Path | Traversable, name: str) -> None:
        if name.endswith((".yml", ".yaml")):
            paths_to_check.append(prefix.joinpath(name))
        else:
            paths_to_check.append(prefix.joinpath(f"{name}.yml"))
            paths_to_check.append(prefix.joinpath(f"{name}.yaml"))

    if isinstance(config_file, str):
        if base_dir is not None:
            _append_with_suffixes(base_dir, config_file)
        if not config_file.endswith((".yml", ".yaml")):
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yml"))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yaml"))
        else:
            paths_to_check.append(Path(config_file))
            paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(config_file))
    elif isinstance(config_file, Path):
        if base_dir is not None and not config_file.is_absolute():
            _append_with_suffixes(base_dir, str(config_file))
        if config_file.suffix in (".yml", ".yaml"):
            paths_to_check.append(config_file)
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(str(config_file)))
        else:
            paths_to_check.append(Path(f"{config_file}.yml"))
            paths_to_check.append(Path(f"{config_file}.yaml"))
            if not config_file.is_absolute():
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yml"))
                paths_to_check.append(BUILTIN_RECIPES_LIB.joinpath(f"{config_file}.yaml"))
    elif isinstance(config_file, Traversable):
        paths_to_check.append(config_file)
    else:
        raise ValueError(f"Invalid config file of {config_file}")

    return paths_to_check


def _resolve_existing(
    config_file: str | Path | Traversable,
    base_dir: Path | Traversable | None = None,
) -> Path | Traversable:
    """Resolve ``config_file`` to the first existing candidate path.

    Raises :class:`ValueError` if no candidate exists, mirroring the message
    format that callers (and tests) rely on.
    """
    paths_to_check = _resolve_config_paths(config_file, base_dir=base_dir)
    for path in paths_to_check:
        if path.is_file():
            return path
    raise ValueError(f"Cannot find config file of {config_file}, paths checked: {paths_to_check}")


def _parent_of(path: Path | Traversable) -> Path | Traversable | None:
    """Return the parent directory of ``path`` for resolving ``!include`` targets.

    For a :class:`pathlib.Path`, this is the standard ``.parent``. For a
    :class:`Traversable` rooted in the built-in recipe library, we walk one
    level back via ``joinpath("..")`` when the resource exposes it; otherwise
    we return ``None`` and rely on the built-in / filesystem fallback path.
    """
    if isinstance(path, Path):
        return path.parent
    parent = getattr(path, "parent", None)
    if parent is not None:
        return parent
    try:
        return path.joinpath("..")
    except (AttributeError, ValueError):
        return None


def _make_loader(
    base_dir: Path | Traversable | None,
    stack: tuple[str, ...],
) -> type[yaml.SafeLoader]:
    """Build a one-shot ``SafeLoader`` subclass with ``!include`` / ``!concat`` support.

    A fresh subclass is created per parse so that the constructors can close
    over ``base_dir`` (for relative-path resolution) and ``stack`` (for cycle
    detection) without leaking state across unrelated parses.
    """

    class _IncludeAwareLoader(yaml.SafeLoader):
        pass

    def _include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "!include expects a scalar path argument",
                node.start_mark,
            )
        target = loader.construct_scalar(node)
        if not isinstance(target, str) or not target:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "!include path must be a non-empty string",
                node.start_mark,
            )
        resolved = _resolve_existing(target, base_dir=base_dir)
        return _load_yaml_with_includes(resolved, stack=stack)

    def _concat_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> list[Any]:
        if not isinstance(node, yaml.SequenceNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "!concat expects a sequence",
                node.start_mark,
            )
        seq = loader.construct_sequence(node, deep=True)
        out: list[Any] = []
        for item in seq:
            if isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out

    _IncludeAwareLoader.add_constructor("!include", _include_constructor)
    _IncludeAwareLoader.add_constructor("!concat", _concat_constructor)
    return _IncludeAwareLoader


def _load_yaml_with_includes(
    path: Path | Traversable,
    stack: tuple[str, ...] = (),
) -> Any:
    """Parse the YAML at ``path``, resolving ``!include`` and ``!concat`` tags.

    ``stack`` is the chain of files currently being parsed; reappearance of a
    file in this chain raises a :class:`ValueError` instead of recursing
    forever.
    """
    key = str(path)
    if key in stack:
        chain = " -> ".join((*stack, key))
        raise ValueError(f"Cycle detected while resolving !include: {chain}")
    new_stack = (*stack, key)
    base_dir = _parent_of(path)

    text = path.read_text(encoding="utf-8")
    loader_cls = _make_loader(base_dir=base_dir, stack=new_stack)
    return yaml.load(text, Loader=loader_cls)


def load_config(config_file: str | Path | Traversable) -> dict[str, Any]:
    """Load a config yaml.

    config_file: Path to a config yaml file. The path suffix can be omitted.

    The loader recognises two custom tags, ``!include`` and ``!concat``; see
    the module docstring for details. ``num_bits`` / ``scale_bits`` strings
    in ``ExMy`` form are converted to ``(E, M)`` tuples after include
    resolution.
    """
    config_path = _resolve_existing(config_file)

    _raw = _load_yaml_with_includes(config_path)
    if _raw is None:
        return {}
    if not isinstance(_raw, dict):
        raise ValueError(
            f"Config file {config_path} must contain a YAML mapping, got {type(_raw).__name__}"
        )
    return _parse_exmy_num_bits(_raw)
