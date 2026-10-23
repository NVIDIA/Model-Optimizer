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

"""Spec-dec training gate: the drafter trained cleanly and its export fits the target.

Pure decision logic in ``evaluate_training`` (unit-tested without a cluster); ``main``
reads the files and prints the verdict. Inputs:

  regression_check.json  written by tools/launcher/common/check_regression.py --json. Its
                         thresholds belong to the recipe (launcher YAML); this gate requires
                         that they were set and passed, and does not apply numbers of its own.
  <export dir>           exported drafter: config.json plus *.safetensors.
  target config.json     the target model's HF config (text_config/llm_config is used when
                         the target is multimodal).

Draft-to-target checks use only fields both configs carry, so a drafter family this gate
does not know degrades to notes ("not checked") instead of failing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from glob import glob

# Exported "architectures" per algorithm, from modelopt/torch/export/plugins/hf_spec_export.py
# and hf_spec_configs.py.
_ARCHITECTURES = {
    "eagle3": {"LlamaForCausalLMEagle3", "Eagle3DeepseekV2ForCausalLM"},
    "dflash": {"DFlashDraftModel"},
    "dflash2": {"DFlash2DraftModel"},
    "dspark": {"DFlashDraftModel"},
    "domino": {"DFlashDraftModel"},
    "lilicorr": {"LiLiCorrDraftModel"},
}
# Metric each check_regression.py threshold is compared against.
_THRESHOLD_METRICS = {
    "MAX_FINAL_LOSS": "loss",
    "MIN_FINAL_ACC": "accuracy",
    "MAX_FINAL_PERPLEXITY": "perplexity",
}
# dflash_config.projector_type each algorithm's export may carry. DFlash, DSpark and Domino
# all export as DFlashDraftModel, so this is what tells them apart; plain DFlash writes none.
_PROJECTOR_TYPES: dict[str, set[str | None]] = {
    "dflash": {None, "dflash"},
    "dflash2": {"dflash2"},
    "dspark": {"dspark"},
    "domino": {"domino"},
    "lilicorr": {"lilicorr"},
}

# Highest priority first: report the most fundamental problem.
_RANKING = (
    "USER_CONFIG_ERROR",
    "EXPORT_MISSING",
    "DRAFT_CONFIG_MISMATCH",
    "METRICS_MISSING",
    "TRAIN_REGRESSION",
)


def _target_text_config(target):
    return target.get("text_config") or target.get("llm_config") or target


def _token_ids(value):
    if value is None:
        return set()
    return set(value) if isinstance(value, list) else {value}


def _check_compatibility(draft, target, problems, notes):
    """Draft <-> target checks on fields both configs carry."""
    tgt = _target_text_config(target)
    for key in ("hidden_size", "vocab_size"):
        if draft.get(key) is None or tgt.get(key) is None:
            notes.append(f"{key} not checked (absent from draft or target config)")
        elif draft[key] != tgt[key]:
            problems.append(
                ("DRAFT_CONFIG_MISMATCH", f"{key}: draft {draft[key]} != target {tgt[key]}")
            )

    vocab = tgt.get("vocab_size")
    special = {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        for tid in _token_ids(tgt.get(key)):
            special[tid] = key
    sub = draft.get("dflash_config") or {}
    mask = sub.get("mask_token_id")
    if mask is None:
        notes.append("mask_token_id not checked (absent from draft config)")
    elif not isinstance(mask, int) or isinstance(mask, bool) or mask < 0:
        problems.append(
            ("DRAFT_CONFIG_MISMATCH", f"mask_token_id {mask!r} is not a token id (int >= 0)")
        )
    elif vocab is not None and mask >= vocab:
        problems.append(
            ("DRAFT_CONFIG_MISMATCH", f"mask_token_id {mask} >= target vocab_size {vocab}")
        )
    elif mask in special:
        problems.append(
            ("DRAFT_CONFIG_MISMATCH", f"mask_token_id {mask} is the target's {special[mask]}")
        )

    n_layers = tgt.get("num_hidden_layers")
    layer_ids = sub.get("target_layer_ids")
    if layer_ids is None:
        layer_ids = (draft.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    if layer_ids is None or n_layers is None:
        notes.append("target layer ids not checked (absent from draft or target config)")
    elif not isinstance(layer_ids, list):
        problems.append(("DRAFT_CONFIG_MISMATCH", f"target layer ids {layer_ids!r} is not a list"))
    else:
        bad = [
            i
            for i in layer_ids
            if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n_layers
        ]
        if bad:
            problems.append(
                ("DRAFT_CONFIG_MISMATCH", f"target layer ids {bad} outside [0, {n_layers})")
            )
    if draft.get("num_target_layers") not in (None, n_layers) and n_layers is not None:
        problems.append(
            (
                "DRAFT_CONFIG_MISMATCH",
                f"num_target_layers {draft['num_target_layers']} != target {n_layers}",
            )
        )


def evaluate_training(
    regression, export_config, export_weights, target_config, algorithm, expect_architecture=None
):
    """Decide whether training ran cleanly and the export fits the target.

    Args:
        regression: dict from regression_check.json, or None if the file is absent.
        export_config: dict from the export's config.json, or None if absent or unreadable.
        export_weights: list of (filename, size_bytes) for the export's *.safetensors.
        target_config: dict from the target's config.json.
        algorithm: drafter algorithm (key of ``_ARCHITECTURES``), case-insensitive.
        expect_architecture: required when ``algorithm`` is not in ``_ARCHITECTURES``.

    Returns:
        dict ``{pass, failure_class, detail, notes, recipe_thresholds, metrics}``.
    """
    algorithm = (algorithm or "").lower()
    problems, notes = [], []
    expected = {expect_architecture} if expect_architecture else _ARCHITECTURES.get(algorithm)
    if not expected:
        problems.append(
            (
                "USER_CONFIG_ERROR",
                f"unknown algorithm {algorithm!r}: pass --expect-architecture "
                f"(known: {sorted(_ARCHITECTURES)})",
            )
        )

    if regression is None:
        problems.append(("METRICS_MISSING", "no regression_check.json"))
        regression = {}
    elif not isinstance(regression, dict) or not regression:
        problems.append(("METRICS_MISSING", "regression_check.json holds no verdict"))
        regression = {}
    elif not regression.get("thresholds"):
        problems.append(
            ("METRICS_MISSING", "regression check ran without thresholds (recipe sets none)")
        )
    elif regression.get("pass") is not True:
        cls = regression.get("failure_class") or "TRAIN_REGRESSION"
        if cls not in ("METRICS_MISSING", "TRAIN_REGRESSION"):
            cls = "TRAIN_REGRESSION"
        problems.append((cls, "; ".join(regression.get("messages") or []) or "check failed"))
    else:
        # A pass must carry the metric behind every threshold. The comparison itself stays
        # check_regression.py's; this only refuses a verdict with nothing behind it.
        metrics = regression.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        for name in regression["thresholds"]:
            key = _THRESHOLD_METRICS.get(name)
            value = metrics.get(key) if key else None
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                problems.append(
                    ("METRICS_MISSING", f"passed with no finite '{key or name}' for {name}")
                )

    if export_config is None:
        problems.append(("EXPORT_MISSING", "export has no readable config.json"))
    elif not any(size > 0 for _, size in export_weights):
        problems.append(("EXPORT_MISSING", "export has no non-empty *.safetensors"))
    else:
        archs = export_config.get("architectures") or []
        if expected and (len(archs) != 1 or archs[0] not in expected):
            problems.append(
                ("DRAFT_CONFIG_MISMATCH", f"architectures {archs} not one of {sorted(expected)}")
            )
        sub = export_config.get("dflash_config") or {}
        projectors = _PROJECTOR_TYPES.get(algorithm)
        if projectors and sub.get("projector_type") not in projectors:
            problems.append(
                (
                    "DRAFT_CONFIG_MISMATCH",
                    f"projector_type {sub.get('projector_type')!r} not one of "
                    f"{sorted(projectors, key=str)}",
                )
            )
        top, inner = export_config.get("block_size"), sub.get("block_size")
        if top is not None and inner is not None and top != inner:
            problems.append(("DRAFT_CONFIG_MISMATCH", f"block_size {top} != dflash_config {inner}"))
        _check_compatibility(export_config, target_config, problems, notes)

    result = {
        "recipe_thresholds": regression.get("thresholds"),
        "metrics": regression.get("metrics"),
        "notes": notes,
    }
    if not problems:
        return {
            "pass": True,
            "failure_class": None,
            "detail": "training and export valid",
            **result,
        }
    fc = min((c for c, _ in problems), key=_RANKING.index)
    return {
        "pass": False,
        "failure_class": fc,
        "detail": "; ".join(f"{c}: {m}" for c, m in problems),
        **result,
    }


def _load_json(path, missing_ok=False):
    if missing_ok and not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_object(path):
    """A JSON object from path; anything else is a bad input (ValueError)."""
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return obj


def main(argv=None):
    """CLI entry point: read the training artifacts and print the verdict."""
    p = argparse.ArgumentParser(description="Spec-dec training gate")
    p.add_argument("--regression", required=True, help="regression_check.json")
    p.add_argument("--export", required=True, help="exported drafter directory")
    p.add_argument("--target-config", required=True, help="target model config.json")
    p.add_argument("--algorithm", required=True, help=f"one of {sorted(_ARCHITECTURES)}")
    p.add_argument("--expect-architecture", help="required for an algorithm not listed above")
    args = p.parse_args(argv)
    inputs = {
        "regression": args.regression,
        "export": args.export,
        "target_config": args.target_config,
    }

    # A missing export or verdict inside an existing training output dir is what a failed
    # run leaves behind; a missing output dir is a mistyped path, and "re-run training" is
    # the wrong advice for a typo.
    for flag, path in (("--export", args.export), ("--regression", args.regression)):
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            error = {
                "pass": False,
                "failure_class": "USER_CONFIG_ERROR",
                "detail": f"directory of {flag} {parent} does not exist",
            }
            print(json.dumps({**error, "notes": [], "inputs": inputs}, indent=2))
            return 2
    try:
        target = _load_object(args.target_config)
        regression = _load_json(args.regression, missing_ok=True)
    except (OSError, ValueError) as e:
        error = {"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}
        print(json.dumps({**error, "notes": [], "inputs": inputs}, indent=2))
        return 2
    try:
        # A missing export dir reads as a missing config.json: EXPORT_MISSING.
        export_config = _load_json(os.path.join(args.export, "config.json"), missing_ok=True)
    except (OSError, ValueError):
        export_config = None  # a truncated config.json is a broken export, not a bad input
    if not isinstance(export_config, dict):
        export_config = None
    weights = [
        (os.path.basename(w), os.path.getsize(w))
        for w in glob(os.path.join(args.export, "*.safetensors"))
    ]

    result = evaluate_training(
        regression, export_config, weights, target, args.algorithm, args.expect_architecture
    )
    result["inputs"] = inputs
    print(json.dumps(result, indent=2))
    if result["failure_class"] == "USER_CONFIG_ERROR":
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
