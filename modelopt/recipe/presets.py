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

The example PTQ entry points (``examples/llm_ptq/hf_ptq.py``,
``examples/llm_ptq/multinode_ptq.py``, ``examples/megatron_bridge/quantize.py``)
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

from collections.abc import Mapping
from typing import Any

from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT, load_config
from modelopt.torch.quantization.config import QuantizeConfig

__all__ = [
    "KV_CACHE_NONE",
    "KV_QUANT_CFG_CHOICES",
    "KV_QUANT_PRESET_DIR",
    "MODEL_QUANT_PRESET_DIR",
    "QFORMAT_ALIASES",
    "QUANT_CFG_CHOICES",
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

# Backward-compat short names → canonical preset basename. These aliases predate the
# YAML-driven discovery and remain accepted so existing scripts/docs keep working.
#
# DO NOT add new entries here. New quantization formats must be exposed via their YAML
# basename under ``modelopt_recipes/configs/ptq/presets/model/`` — the directory listing
# is the canonical CLI vocabulary. This table exists solely to keep pre-existing short
# names working through deprecation and should only ever shrink.
QFORMAT_ALIASES: dict[str, str] = {
    "int8_sq": "int8_smoothquant",
    "int8_wo": "int8_weight_only",
    "w4a8_awq": "w4a8_awq_beta",
    "nvfp4_awq": "nvfp4_awq_lite",
    "nvfp4_mse": "nvfp4_w4a4_weight_mse_fp8_sweep",
    "nvfp4_local_hessian": "nvfp4_w4a4_weight_local_hessian",
    "fp8_pb_wo": "fp8_2d_blockwise_weight_only",
    "fp8_pc_pt": "fp8_per_channel_per_token",
    "fp8_blockwise": "fp8_2d_blockwise_weight_only",
}


def load_quant_cfg_choices(
    subdir: str, aliases: Mapping[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build a ``{qformat_name: quant_cfg_dict}`` mapping from preset YAMLs.

    Every ``*.yaml`` under ``modelopt_recipes/<subdir>/`` is loaded and keyed by its
    basename — the directory listing is the CLI vocabulary. ``aliases`` adds extra
    short names pointing at canonical basenames; a stale alias raises here (at load
    time) rather than failing silently at lookup time.

    Args:
        subdir: Preset directory relative to ``modelopt_recipes/`` (e.g.
            :data:`MODEL_QUANT_PRESET_DIR`).
        aliases: Optional ``short_name -> canonical_basename`` deprecation map.

    Returns:
        Mapping from format name (preset basename or alias) to the loaded
        ``QuantizeConfig`` dict. Configs are loaded eagerly; callers that mutate a
        returned config must deepcopy it first.
    """
    aliases = aliases or {}
    basenames = sorted(
        entry.name.rsplit(".", 1)[0]
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    )
    choices: dict[str, dict[str, Any]] = {
        name: load_config(f"{subdir}/{name}", schema_type=QuantizeConfig).model_dump(
            exclude_unset=True
        )
        for name in basenames
    }
    for alias, target in sorted(aliases.items()):
        if target not in choices:
            raise ValueError(
                f"Alias {alias!r} points at preset {target!r} which is not present "
                f"under modelopt_recipes/{subdir}/."
            )
        choices[alias] = choices[target]
    return choices


QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = load_quant_cfg_choices(
    MODEL_QUANT_PRESET_DIR, QFORMAT_ALIASES
)
KV_QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = load_quant_cfg_choices(KV_QUANT_PRESET_DIR)

# Guard against a future ``none.yaml`` (or alias) colliding with the disable sentinel:
# the runtime branch on ``!= KV_CACHE_NONE`` would otherwise become ambiguous.
assert KV_CACHE_NONE not in KV_QUANT_CFG_CHOICES, (
    f"KV_CACHE_NONE sentinel {KV_CACHE_NONE!r} collides with a KV preset; rename the preset."
)
