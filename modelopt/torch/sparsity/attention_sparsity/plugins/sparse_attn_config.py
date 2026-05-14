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

"""Helpers for resolving sparse attention config in a serving context.

These helpers operate on plain dicts and ``transformers.PretrainedConfig``-like
objects — they do not depend on vLLM and can be used (and unit-tested) without
it installed.

- ``match_sparse_config`` — fnmatch a module name against a sparse_cfg dict.
- ``load_from_checkpoint_metadata`` — read ``sparse_attention_config`` from a
  HF config and map ``sparse_algo`` to an mtsa preset.
"""

import fnmatch

import modelopt.torch.sparsity.attention_sparsity as mtsa

# Maps ``sparse_algo`` values written by ``export_sparse_attention_config`` into
# the checkpoint config.json to mtsa presets. Per-layer / per-seqlen calibration
# mapping (using the (a, b) polynomial under ``threshold_scale_factor``) and N:M
# sparsity require extending ``export_sparse_attention_config`` to serialize
# per-layer method_config; deferred to a follow-up.
ALGO_TO_PRESET = {
    "softmax_skip": "SKIP_SOFTMAX_TRITON_DEFAULT",
}


def match_sparse_config(module_name: str, sparse_cfg: dict) -> dict | None:
    """Match a module name against ``sparse_cfg`` patterns (first hit wins).

    ``sparse_cfg`` is either ``{"sparse_cfg": {...}}`` (as exported by mtsa
    presets) or the bare inner dict. ``default`` and ``calibration`` keys are
    metadata and never matched as patterns.
    """
    cfg = sparse_cfg.get("sparse_cfg", sparse_cfg)
    for pattern, layer_cfg in cfg.items():
        if pattern in ("default", "calibration"):
            continue
        if fnmatch.fnmatch(module_name, pattern):
            return layer_cfg
    return None


def load_from_checkpoint_metadata(hf_config) -> tuple[dict, str] | None:
    """Resolve sparse_cfg from a HF model config object.

    Reads ``sparse_attention_config`` written by ModelOpt's HF export
    (``unified_export_hf.export_sparse_attention_config``) and maps the
    declared ``sparse_algo`` to an mtsa preset via :data:`ALGO_TO_PRESET`.

    Args:
        hf_config: A ``transformers.PretrainedConfig``-like object (or any
            namespace) whose ``sparse_attention_config`` attribute holds the
            exported metadata dict.

    Returns:
        ``(sparse_cfg, preset_name)`` on hit; ``None`` if the config has no
        recognized sparse attention metadata.
    """
    if hf_config is None:
        return None
    sparse_meta = getattr(hf_config, "sparse_attention_config", None)
    if not isinstance(sparse_meta, dict):
        return None
    config_groups = sparse_meta.get("config_groups", {})
    if not isinstance(config_groups, dict):
        return None
    algos = {grp.get("sparse_algo") for grp in config_groups.values() if isinstance(grp, dict)}
    for algo, preset_name in ALGO_TO_PRESET.items():
        if algo in algos:
            preset = getattr(mtsa, preset_name, None)
            if isinstance(preset, dict):
                return preset, preset_name
    return None
