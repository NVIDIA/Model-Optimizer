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

"""Adapt the pinned Qwen 3.5 VLM profile to a post-MIP checkpoint node."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import evaluate, suites
from modelopt.torch.puzzletron.distributed_eval.storage import atomic_write_json

__all__ = [
    "evaluate_e2e_full_eval_checkpoint",
    "evaluate_frozen_campaign_checkpoint",
    "evaluate_realworldqa_checkpoint",
    "evaluate_short_v1_checkpoint",
    "register_profiles",
]

_RUNNER_OVERRIDES = frozenset(
    {
        "dtype",
        "gpu_memory_utilization",
        "limit_mm_per_prompt",
        "max_model_len",
        "topology",
    }
)
_MANIFEST_SETTINGS = frozenset({"row_manifest", "row_manifest_sha256"})
_REALWORLDQA_PROFILE = "qwen35_vlm_realworldqa2_prefix2"
_BOUNDED_REPEATED_PROFILE = "qwen35_vlm_realworldqa100_mmmu100_prefix100_repeat2"
_FROZEN_CAMPAIGN_PROFILE = "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v1"


def _run_profile(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
    suite: str,
) -> tuple[argparse.Namespace, dict[str, object], Path]:
    settings = dict(settings)
    unexpected = (
        set(settings) - _RUNNER_OVERRIDES - _MANIFEST_SETTINGS - {"batch_size", "timeout_seconds"}
    )
    if unexpected:
        raise ValueError(f"unsupported Qwen 3.5 VLM profile settings: {sorted(unexpected)}")
    output_dir = Path(output_root).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_manifest = settings.pop("row_manifest", None)
    expected_manifest_sha256 = settings.pop("row_manifest_sha256", None)
    quick_manifest = Path(row_manifest).expanduser().absolute() if row_manifest else None
    if quick_manifest is not None:
        if (
            not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
        ):
            raise ValueError(
                "frozen 344-row campaign manifest SHA256 must be 64 lowercase hex characters"
            )
        actual_manifest_sha256 = suites.manifest_sha256(suites.load_quick_manifest(quick_manifest))
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "frozen 344-row campaign manifest SHA256 differs from the campaign identity: "
                f"{actual_manifest_sha256} != {expected_manifest_sha256}"
            )
    args = argparse.Namespace(
        checkpoint=Path(checkpoint_path).expanduser().absolute(),
        output_dir=output_dir,
        suite=suite,
        batch_size=int(settings.pop("batch_size", 1)),
        seed=42,
        timeout_seconds=settings.pop("timeout_seconds", None),
        hf_home=Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None,
        quick_manifest=quick_manifest,
        mmvu_judge_api_type=None,
        mmvu_judge_model=None,
        allow_judge_calls=False,
        preflight_only=False,
    )
    profile_path = output_dir / "profile.json"

    def write_preflight(report: dict[str, object]) -> None:
        checkpoint.write_generated(
            profile_path,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )

    result = evaluate(
        args,
        settings_overrides=settings,
        preflight_callback=write_preflight,
    )
    return args, result, profile_path


def register_profiles() -> None:
    """Install the example-owned profile into the generic post-MIP runner."""

    from modelopt.torch.puzzletron.post_mip.runner import register_downstream_evaluation_profile

    register_downstream_evaluation_profile(
        _REALWORLDQA_PROFILE,
        evaluate_realworldqa_checkpoint,
    )
    register_downstream_evaluation_profile(
        _BOUNDED_REPEATED_PROFILE,
        evaluate_e2e_full_eval_checkpoint,
    )
    register_downstream_evaluation_profile(
        _FROZEN_CAMPAIGN_PROFILE,
        evaluate_frozen_campaign_checkpoint,
    )
    # Deprecated compatibility aliases. New recipes must use explicit task and
    # row-selection identities above.
    register_downstream_evaluation_profile(
        "qwen35_vlm_realworldqa",
        evaluate_realworldqa_checkpoint,
    )
    register_downstream_evaluation_profile(
        "qwen35_vlm_e2e_full_eval",
        evaluate_e2e_full_eval_checkpoint,
    )
    register_downstream_evaluation_profile(
        "qwen35_vlm_short_v1",
        evaluate_short_v1_checkpoint,
    )


def evaluate_frozen_campaign_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint on the identity-bound frozen campaign rows."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="quick",
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM frozen 344-row profile returned an invalid run count")
    return {
        **runs[0],
        "profile_path": str(profile_path),
        "checkpoint": str(args.checkpoint),
    }


def evaluate_short_v1_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for the explicit frozen-row campaign profile."""

    warnings.warn(
        f"qwen35_vlm_short_v1 is deprecated; use {_FROZEN_CAMPAIGN_PROFILE}",
        DeprecationWarning,
        stacklevel=2,
    )
    return evaluate_frozen_campaign_checkpoint(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
    )


def evaluate_realworldqa_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the two-sample pinned RealWorldQA profile for one saved checkpoint."""

    _args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="realworldqa-smoke",
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned RealWorldQA profile returned an invalid run count")
    return {**runs[0], "profile_path": str(profile_path)}


def evaluate_e2e_full_eval_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run and average the repeated, bounded final-evaluation contract."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="e2e-full-eval",
    )
    runs = result["runs"]
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or not all(isinstance(item, dict) for item in runs)
    ):
        raise RuntimeError("pinned VLM final-evaluation profile returned an invalid run count")
    metric_names = set(runs[0].get("metrics") or {})
    if not metric_names or any(set(item.get("metrics") or {}) != metric_names for item in runs[1:]):
        raise RuntimeError("pinned VLM final-evaluation repetitions produced different metrics")
    metrics = {
        name: sum(float(item["metrics"][name]) for item in runs) / len(runs)
        for name in sorted(metric_names)
    }
    result_paths = [str(item["result_path"]) for item in runs]
    summary_path = args.output_dir / "e2e_full_eval_summary.json"
    atomic_write_json(
        summary_path,
        {
            "checkpoint": str(args.checkpoint),
            "metrics": metrics,
            "result_paths": result_paths,
            "suite": args.suite,
        },
    )
    return {
        "metrics": metrics,
        "profile_path": str(profile_path),
        "result_path": str(summary_path),
        "run_result_paths": result_paths,
    }
