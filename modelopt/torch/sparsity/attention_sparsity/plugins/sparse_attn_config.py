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
  HF config and resolve it to a serving sparse_cfg.
"""

import fnmatch

import modelopt.torch.sparsity.attention_sparsity as mtsa

# Maps ``algorithm`` values without calibration metadata into mtsa presets.
ALGO_TO_PRESET = {
    "skip_softmax": "SKIP_SOFTMAX_TRITON_DEFAULT",
}

DEFAULT_TARGET_SPARSE_RATIO = {"prefill": 0.5, "decode": 0.5}
SPARSE_SOFTMAX_ALGOS = {"sparse_softmax", "triton_sparse_softmax"}
SPARSE_SOFTMAX_DEFAULTS = {
    "sparsity_n": 2,
    "sparsity_m": 4,
    "dense_sink_tokens": 0,
    "dense_recent_tokens": 64,
}


def _normalize_target_sparse_ratio(value) -> dict[str, float]:
    """Normalize exported target sparsity metadata, defaulting old checkpoints."""
    if isinstance(value, (float, int)):
        ratio = float(value)
        return {"prefill": ratio, "decode": ratio}
    if isinstance(value, dict):
        return {
            "prefill": float(value.get("prefill", DEFAULT_TARGET_SPARSE_RATIO["prefill"])),
            "decode": float(value.get("decode", DEFAULT_TARGET_SPARSE_RATIO["decode"])),
        }
    return DEFAULT_TARGET_SPARSE_RATIO.copy()


def _has_calibrated_threshold_scale_factor(value) -> bool:
    """Return True when checkpoint metadata has usable phase calibration params."""
    if not isinstance(value, dict):
        return False
    for phase in ("prefill", "decode"):
        params = value.get(phase)
        if isinstance(params, dict) and "a" in params and "b" in params:
            return True
    return False


def _has_sparse_softmax_algo(algos: set[str | None]) -> bool:
    """Return True if checkpoint metadata requests N:M sparse softmax."""
    return bool(algos & SPARSE_SOFTMAX_ALGOS)


def _sparse_softmax_params(sparse_meta: dict, config_groups: dict) -> dict:
    """Resolve N:M sparse-softmax params from checkpoint metadata."""
    params = SPARSE_SOFTMAX_DEFAULTS.copy()
    exported_params = sparse_meta.get("sparse_softmax")
    if isinstance(exported_params, dict):
        params.update(
            {key: exported_params[key] for key in SPARSE_SOFTMAX_DEFAULTS if key in exported_params}
        )
        return params

    for group in config_groups.values():
        if not isinstance(group, dict) or group.get("algorithm") not in SPARSE_SOFTMAX_ALGOS:
            continue
        params.update({key: group[key] for key in SPARSE_SOFTMAX_DEFAULTS if key in group})
        return params
    return params


def _add_sparse_softmax_params(layer_cfg: dict, sparse_meta: dict, config_groups: dict) -> None:
    """Add N:M sparse-softmax params to a vLLM layer config."""
    layer_cfg.update(_sparse_softmax_params(sparse_meta, config_groups))


def _build_calibrated_softmax_skip_config(skip_group: dict) -> dict:
    """Build a vLLM Triton sparse config from a skip_softmax config group."""
    return {
        "sparse_cfg": {
            "*attn*": {
                "method": "triton_skip_softmax",
                "threshold_scale_factor": skip_group["threshold_scale_factor"],
                "target_sparse_ratio": _normalize_target_sparse_ratio(
                    skip_group.get("target_sparsity")
                ),
                "backend": "triton",
                "enable": True,
            },
            "default": {"enable": False},
        },
    }


def _build_sparse_softmax_config(sparse_meta: dict, config_groups: dict) -> dict:
    """Build a vLLM Triton N:M sparse-softmax config from exported metadata."""
    layer_cfg = {
        "method": "triton_sparse_softmax",
        "backend": "triton",
        "enable": True,
    }
    _add_sparse_softmax_params(layer_cfg, sparse_meta, config_groups)
    return {
        "sparse_cfg": {
            "*attn*": layer_cfg,
            "default": {"enable": False},
        },
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
    (``unified_export_hf.export_sparse_attention_config``). Calibrated
    ``skip_softmax`` metadata is converted into a dynamic Triton config;
    uncalibrated algorithms fall back to mtsa presets via :data:`ALGO_TO_PRESET`.

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
    algos = {grp.get("algorithm") for grp in config_groups.values() if isinstance(grp, dict)}
    skip_group = next(
        (
            grp
            for grp in config_groups.values()
            if isinstance(grp, dict) and grp.get("algorithm") == "skip_softmax"
        ),
        None,
    )
    if skip_group is not None and _has_calibrated_threshold_scale_factor(
        skip_group.get("threshold_scale_factor")
    ):
        cfg = _build_calibrated_softmax_skip_config(skip_group)
        if _has_sparse_softmax_algo(algos):
            layer_cfg = cfg["sparse_cfg"]["*attn*"]
            _add_sparse_softmax_params(layer_cfg, sparse_meta, config_groups)
            return cfg, "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP_SPARSE_SOFTMAX"
        return cfg, "CHECKPOINT_CALIBRATED_SOFTMAX_SKIP"
    if _has_sparse_softmax_algo(algos):
        return _build_sparse_softmax_config(sparse_meta, config_groups), "CHECKPOINT_SPARSE_SOFTMAX"
    for algo, preset_name in ALGO_TO_PRESET.items():
        if algo in algos:
            preset = getattr(mtsa, preset_name, None)
            if isinstance(preset, dict):
                return preset, preset_name
    return None
