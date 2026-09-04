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
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import evaluate, suites
from modelopt.torch.puzzletron.distributed_eval.storage import atomic_write_json

__all__ = [
    "TASK_PREFIX100_REPEAT2_PROFILE",
    "evaluate_e2e_full_eval_checkpoint",
    "evaluate_frozen_campaign_checkpoint",
    "evaluate_frozen_campaign_v2_checkpoint",
    "evaluate_frozen_campaign_v3_checkpoint",
    "evaluate_realworldqa_checkpoint",
    "evaluate_realworldqa_mmmu_prefix100_checkpoint",
    "evaluate_reproducibility_smoke_checkpoint",
    "evaluate_reproducibility_smoke_v2_checkpoint",
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
TASK_PREFIX100_REPEAT2_PROFILE = _BOUNDED_REPEATED_PROFILE
_FROZEN_CAMPAIGN_PROFILE_V1 = "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v1"
_FROZEN_CAMPAIGN_PROFILE_V2 = "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v2"
_FROZEN_CAMPAIGN_PROFILE_V3 = "qwen35_vlm_realworldqa64_mmmu120_mvbench160_frozen_rows_v3"
_REPRODUCIBILITY_SMOKE_PROFILE = "qwen35_vlm_core3_24row_smoke_v1"
_REPRODUCIBILITY_SMOKE_PROFILE_V2 = "qwen35_vlm_core3_24row_smoke_v2"


def _run_profile(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
    suite: str,
    evaluation_profile: str | None = None,
    require_manifest: bool = False,
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
    if (
        require_manifest
        and evaluation_profile is None
        and (not row_manifest or not expected_manifest_sha256)
    ):
        raise ValueError(
            "frozen 344-row campaign profile requires row_manifest and row_manifest_sha256"
        )
    if require_manifest and evaluation_profile is not None and not expected_manifest_sha256:
        raise ValueError("frozen 344-row campaign evaluation profile requires row_manifest_sha256")
    if row_manifest is not None and evaluation_profile is not None:
        raise ValueError("an embedded evaluation profile cannot be overridden by row_manifest")
    if expected_manifest_sha256 is not None and (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        raise ValueError(
            "frozen 344-row campaign manifest SHA256 must be 64 lowercase hex characters"
        )
    quick_manifest = Path(row_manifest).expanduser().absolute() if row_manifest else None
    if quick_manifest is not None:
        actual_manifest_sha256 = suites.manifest_sha256(suites.load_quick_manifest(quick_manifest))
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "frozen 344-row campaign manifest SHA256 differs from the campaign identity: "
                f"{actual_manifest_sha256} != {expected_manifest_sha256}"
            )
    args = argparse.Namespace(
        checkpoint=Path(checkpoint_path).expanduser().absolute(),
        output_dir=output_dir,
        profile=evaluation_profile,
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
        if evaluation_profile is not None and (
            expected_manifest_sha256 is not None
            and report.get("quick_manifest_sha256") != expected_manifest_sha256
        ):
            raise ValueError(
                "frozen 344-row campaign manifest SHA256 differs from the campaign identity: "
                f"{report.get('quick_manifest_sha256')} != {expected_manifest_sha256}"
            )
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
        evaluate_realworldqa_mmmu_prefix100_checkpoint,
    )
    register_downstream_evaluation_profile(
        _FROZEN_CAMPAIGN_PROFILE_V1,
        evaluate_frozen_campaign_checkpoint,
    )
    register_downstream_evaluation_profile(
        _FROZEN_CAMPAIGN_PROFILE_V2,
        evaluate_frozen_campaign_v2_checkpoint,
    )
    register_downstream_evaluation_profile(
        _FROZEN_CAMPAIGN_PROFILE_V3,
        evaluate_frozen_campaign_v3_checkpoint,
    )
    register_downstream_evaluation_profile(
        _REPRODUCIBILITY_SMOKE_PROFILE,
        evaluate_reproducibility_smoke_checkpoint,
    )
    register_downstream_evaluation_profile(
        _REPRODUCIBILITY_SMOKE_PROFILE_V2,
        evaluate_reproducibility_smoke_v2_checkpoint,
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
        require_manifest=True,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM frozen 344-row profile returned an invalid run count")
    return {
        **runs[0],
        "profile_path": str(profile_path),
        "checkpoint": str(args.checkpoint),
    }


def evaluate_frozen_campaign_v2_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint on the current-image frozen campaign profile."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="short",
        evaluation_profile="core-3_344-examples_r1-native",
        require_manifest=True,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM frozen 344-row profile returned an invalid run count")
    return {
        **runs[0],
        "profile_path": str(profile_path),
        "checkpoint": str(args.checkpoint),
    }


def evaluate_frozen_campaign_v3_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate heterogeneous materialized checkpoints with the current vLLM profile."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="short",
        evaluation_profile="core-3_344-examples_r1-vllm",
        require_manifest=True,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM frozen 344-row profile returned an invalid run count")
    return {
        **runs[0],
        "profile_path": str(profile_path),
        "checkpoint": str(args.checkpoint),
    }


def evaluate_reproducibility_smoke_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint on the immutable 24-row lifecycle smoke."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="short",
        evaluation_profile="core-3_24-examples_r1-native",
        require_manifest=True,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM 24-row smoke returned an invalid run count")
    return {
        **runs[0],
        "profile_path": str(profile_path),
        "checkpoint": str(args.checkpoint),
    }


def evaluate_reproducibility_smoke_v2_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a heterogeneous materialized checkpoint on the 24-row smoke."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="short",
        evaluation_profile="core-3_24-examples_r1-vllm",
        require_manifest=True,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned VLM 24-row smoke returned an invalid run count")
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
        f"qwen35_vlm_short_v1 is deprecated; use {_FROZEN_CAMPAIGN_PROFILE_V1}",
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


def evaluate_realworldqa_mmmu_prefix100_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run and average the two-task prefix-100 profile repeated twice."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite=suites.TASK_PREFIX100_REPEAT2_SUITE,
    )
    runs = result["runs"]
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or not all(isinstance(item, dict) for item in runs)
    ):
        raise RuntimeError("pinned VLM 100x2 profile returned an invalid run count")
    metric_names = set(runs[0].get("metrics") or {})
    if not metric_names or any(set(item.get("metrics") or {}) != metric_names for item in runs[1:]):
        raise RuntimeError("pinned VLM 100x2 repetitions produced different metrics")
    metrics = {
        name: sum(float(item["metrics"][name]) for item in runs) / len(runs)
        for name in sorted(metric_names)
    }
    result_paths = [str(item["result_path"]) for item in runs]
    sample_counts: Counter[str] = Counter()
    parser_status_counts: Counter[str] = Counter()
    parser_sample_count = 0
    for result_path in result_paths:
        payload = json.loads(Path(result_path).read_text())
        sample_counts.update({key: int(value) for key, value in payload["sample_counts"].items()})
        parser_audit = payload.get("mmmu_parser_audit") or {}
        parser_sample_count += int(parser_audit.get("sample_count", 0))
        parser_status_counts.update(
            {key: int(value) for key, value in (parser_audit.get("status_counts") or {}).items()}
        )
    summary_path = args.output_dir / "realworldqa_mmmu_prefix100_repeat2_summary.json"
    atomic_write_json(
        summary_path,
        {
            "checkpoint": str(args.checkpoint),
            "metrics": metrics,
            "profile": _BOUNDED_REPEATED_PROFILE,
            "result_paths": result_paths,
            "sample_counts": dict(sorted(sample_counts.items())),
            "mmmu_parser_audit": {
                "sample_count": parser_sample_count,
                "status_counts": dict(sorted(parser_status_counts.items())),
            },
            "suite": args.suite,
        },
    )
    return {
        "metrics": metrics,
        "profile": _BOUNDED_REPEATED_PROFILE,
        "profile_path": str(profile_path),
        "result_path": str(summary_path),
        "run_result_paths": result_paths,
    }


def evaluate_e2e_full_eval_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for the explicit two-task prefix-100 profile."""

    warnings.warn(
        f"qwen35_vlm_e2e_full_eval is deprecated; use {_BOUNDED_REPEATED_PROFILE}",
        FutureWarning,
        stacklevel=2,
    )
    return evaluate_realworldqa_mmmu_prefix100_checkpoint(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
    )
