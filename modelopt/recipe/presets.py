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

"""PTQ quant-config preset discovery shared by the PTQ example scripts.

The example PTQ entry points (``examples/hf_ptq/hf_ptq.py``,
``examples/hf_ptq/multinode_ptq.py``, ``examples/megatron_bridge/quantize.py``)
expose a ``--qformat`` / ``--kv_cache_qformat`` (``--quant_cfg`` /
``--kv_cache_quant`` for Megatron-Bridge) CLI vocabulary. Rather than hardcoding a
name → config table in each script, the vocabulary is discovered by listing the
YAML presets shipped under ``modelopt_recipes/configs/ptq/presets/{model,kv}/``:
every ``*.yaml`` basename is a valid format name, and the directory listing is the
single source of truth. Adding a preset YAML exposes it on all three CLIs with no
code change.

:data:`QUANT_CFG_CHOICES` and :data:`KV_QUANT_CFG_CHOICES` are the ready-to-use
mappings; :func:`load_quant_cfg_choices` builds equivalent mappings for custom
preset directories. Configs are loaded eagerly into plain dicts at import; callers
that mutate a returned config must deepcopy it first (this mirrors how the
``mtq.*_CFG`` module constants — themselves eagerly-loaded shared dicts — are used).
"""

import argparse
import warnings
from typing import Any

from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT, load_config
from modelopt.torch.quantization.config import QuantizeConfig

__all__ = [
    "KV_CACHE_NONE",
    "KV_QUANT_CFG_CHOICES",
    "KV_QUANT_PRESET_DIR",
    "MODEL_QUANT_PRESET_DIR",
    "QUANT_CFG_CHOICES",
    "RecipeSupersededAction",
    "load_quant_cfg_choices",
]

# Preset directories (relative to ``modelopt_recipes/``) that back the CLI vocabulary.
#
# Prefer NOT to add new YAMLs to these directories: the long-term direction is to
# retire ``--qformat`` / ``--kv_cache_qformat`` in favour of ``--recipe`` (a full PTQ
# recipe; see ``modelopt_recipes/general/ptq/`` and :mod:`modelopt.recipe`). New
# quantization configurations should be authored as recipes, not as preset entries.
MODEL_QUANT_PRESET_DIR = "configs/ptq/presets/model"
KV_QUANT_PRESET_DIR = "configs/ptq/presets/kv"

# Sentinel ``--kv_cache_qformat`` value meaning "no KV cache quantization". Handled by
# the scripts outside the discovered presets; guarded below against a ``none.yaml`` clash.
KV_CACHE_NONE = "none"


def load_quant_cfg_choices(subdir: str) -> dict[str, dict[str, Any]]:
    """Build a ``{qformat_name: quant_cfg_dict}`` mapping from preset YAMLs.

    Every ``*.yaml`` under ``modelopt_recipes/<subdir>/`` is loaded and keyed by its
    basename — the directory listing is the CLI vocabulary.

    Args:
        subdir: Preset directory relative to ``modelopt_recipes/`` (e.g.
            :data:`MODEL_QUANT_PRESET_DIR`).

    Returns:
        Mapping from preset basename to the loaded ``QuantizeConfig`` dict. Configs are
        loaded eagerly; callers that mutate a returned config must deepcopy it first.
    """
    basenames = sorted(
        entry.name.rsplit(".", 1)[0]
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    )
    return {
        name: load_config(f"{subdir}/{name}", schema_type=QuantizeConfig).model_dump(
            exclude_unset=True
        )
        for name in basenames
    }


QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = load_quant_cfg_choices(MODEL_QUANT_PRESET_DIR)
KV_QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = load_quant_cfg_choices(KV_QUANT_PRESET_DIR)

# Guard against a future ``none.yaml`` colliding with the disable sentinel:
# the runtime branch on ``!= KV_CACHE_NONE`` would otherwise become ambiguous.
assert KV_CACHE_NONE not in KV_QUANT_CFG_CHOICES, (
    f"KV_CACHE_NONE sentinel {KV_CACHE_NONE!r} collides with a KV preset; rename the preset."
)


class RecipeSupersededAction(argparse.Action):
    """``argparse`` action for a CLI flag that ``--recipe`` replaces.

    Warns only when the flag is actually passed: argparse invokes an action for options present on
    the command line, never for a default. That distinction matters here because several of these
    flags default to a *quantizing* value -- ``--qformat fp8``, ``--kv_cache_qformat fp8_cast`` --
    so warning unconditionally would fire on every run, including runs that correctly use
    ``--recipe`` and never mention the deprecated flag.

    Handles both value-taking flags and ``store_true`` ones; for the latter pass
    ``nargs=0, const=True``.

    The warning is a ``FutureWarning``, not a ``DeprecationWarning``. Python ignores
    ``DeprecationWarning`` by default everywhere except ``__main__``, and argparse calls this action
    from its own module, so the attributed frame is ``argparse`` and the default filters would drop
    it -- the flag would go on working with nothing said, which defeats the point. ``FutureWarning``
    is the category Python documents for deprecations aimed at end users, and it is shown by
    default. (Test suites enable all warnings, so this is invisible in tests either way.)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Warn that this flag is deprecated, then store the value as usual."""
        warnings.warn(
            f"{option_string} is deprecated and will be removed in a future release. Use "
            "--recipe with a YAML recipe instead: a recipe carries the quantization config, the "
            "calibration algorithm and the KV-cache setting together, so they cannot drift apart. "
            "See modelopt_recipes/general/ptq/ and modelopt.recipe.",
            FutureWarning,
            stacklevel=2,
        )
        setattr(namespace, self.dest, self.const if self.nargs == 0 else values)
