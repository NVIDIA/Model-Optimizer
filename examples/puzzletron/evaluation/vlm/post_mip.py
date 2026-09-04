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
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
_PROFILE_CONTRACT_FIELDS = (
    "profile",
    "profile_name",
    "profile_schema",
    "profile_fingerprint",
    "model_pin",
    "profile_population_rows",
    "suite",
    "lmms_eval_revision",
    "model_backend",
    "source_tasks",
    "profile_task",
    "profile_task_shard",
    "dataset_revisions",
    "frame_policy",
    "generation_policy",
    "backend_limitations",
    "output_budget_contract",
    "sample_limit",
    "quick_selected_rows",
    "quick_row_identities",
    "quick_task_denominators",
    "quick_manifest_sha256",
    "judge_free_mmvu_rows",
    "short_repetitions",
    "repetitions",
    "batch_size",
    "judge_policy",
    "network_policy",
    "post_mip_runner_overrides",
)
_MMMU_PARSER_STATUSES = {"fallback_random", "invalid_open", "parsed", "parsed_open"}


def _evaluation_contract(profile_path: Path) -> dict[str, Any]:
    """Project stable Qwen VLM evaluator inputs for core comparability checks."""

    payload = json.loads(profile_path.read_text())
    if not isinstance(payload, Mapping):
        raise RuntimeError("Qwen VLM evaluator profile must contain an object")
    missing = set(_PROFILE_CONTRACT_FIELDS) - payload.keys()
    if missing:
        raise RuntimeError(f"Qwen VLM evaluator profile is missing {sorted(missing)}")
    return {
        "schema": "modelopt.puzzletron.qwen35-vlm-evaluator-contract/v1",
        **{field: payload[field] for field in _PROFILE_CONTRACT_FIELDS},
    }


def _expected_sample_counts(contract: Mapping[str, Any]) -> dict[str, int]:
    """Resolve the exact generated-task counts encoded by a frozen Qwen profile."""

    source_tasks = contract.get("source_tasks")
    denominators = contract.get("quick_task_denominators")
    identities = contract.get("quick_row_identities")
    expected_rows = contract.get("quick_selected_rows")
    repetitions = contract.get("repetitions")
    if (
        not isinstance(source_tasks, list)
        or not source_tasks
        or any(not isinstance(task, str) or not task for task in source_tasks)
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise RuntimeError("Qwen VLM evaluator profile has invalid task-count evidence")
    if denominators is None and identities is None and expected_rows is None:
        sample_limit = contract.get("sample_limit")
        if (
            isinstance(sample_limit, bool)
            or not isinstance(sample_limit, int)
            or sample_limit <= 0
            or any(task in {"mvbench", "video_mmmu"} for task in source_tasks)
        ):
            raise RuntimeError("Qwen VLM evaluator profile has invalid task-count evidence")
        return {
            f"modelopt_vlm_benchmark_{task}": sample_limit * repetitions
            for task in sorted(source_tasks)
        }
    if (
        not isinstance(denominators, Mapping)
        or not isinstance(identities, Mapping)
        or set(source_tasks) != set(denominators)
        or set(source_tasks) != set(identities)
    ):
        raise RuntimeError("Qwen VLM evaluator profile has invalid exact-row task evidence")
    expected: dict[str, int] = {}
    for source_task in source_tasks:
        denominator = denominators.get(source_task)
        rows = identities.get(source_task)
        selected = denominator.get("selected_rows") if isinstance(denominator, Mapping) else None
        if (
            not isinstance(source_task, str)
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected <= 0
            or not isinstance(rows, list)
            or len(rows) != selected
        ):
            raise RuntimeError("Qwen VLM evaluator profile has invalid exact-row task evidence")
        if source_task in {"mvbench", "video_mmmu"}:
            for row in rows:
                leaf = row.get("leaf_task") if isinstance(row, Mapping) else None
                if not isinstance(leaf, str) or not leaf.startswith(f"{source_task}_"):
                    raise RuntimeError(
                        "Qwen VLM evaluator profile has invalid exact-row task evidence"
                    )
                task = f"modelopt_vlm_benchmark_{leaf}"
                expected[task] = expected.get(task, 0) + repetitions
        else:
            expected[f"modelopt_vlm_benchmark_{source_task}"] = selected * repetitions
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or sum(expected.values()) != expected_rows * repetitions
    ):
        raise RuntimeError("Qwen VLM evaluator profile has invalid exact-row task evidence")
    return dict(sorted(expected.items()))


def _evaluation_evidence(result_path: str | Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and project Qwen VLM row-completion evidence."""

    payload = json.loads(Path(result_path).read_text())
    if not isinstance(payload, Mapping):
        raise RuntimeError("Qwen VLM evaluator result must contain an object")
    counts = payload.get("sample_counts")
    if (
        not isinstance(counts, Mapping)
        or not counts
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or int(value) != value
            for value in counts.values()
        )
    ):
        raise RuntimeError("Qwen VLM evaluator result has invalid sample-count evidence")
    normalized_counts = {str(task): int(value) for task, value in sorted(counts.items())}
    if normalized_counts != _expected_sample_counts(contract):
        raise RuntimeError("Qwen VLM evaluator sample counts do not match its exact-row profile")
    evidence: dict[str, Any] = {
        "schema": "modelopt.puzzletron.qwen35-vlm-evaluation-evidence/v1",
        "sample_counts": normalized_counts,
    }
    source_tasks = contract.get("source_tasks")
    if not isinstance(source_tasks, list) or "mmmu_val" not in source_tasks:
        return evidence
    audit = payload.get("mmmu_parser_audit")
    sample_count = audit.get("sample_count") if isinstance(audit, Mapping) else None
    status_counts = audit.get("status_counts") if isinstance(audit, Mapping) else None
    expected_mmmu_samples = normalized_counts.get("modelopt_vlm_benchmark_mmmu_val")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or not isinstance(expected_mmmu_samples, int)
        or sample_count != expected_mmmu_samples
        or not isinstance(status_counts, Mapping)
        or not status_counts
        or any(
            status not in _MMMU_PARSER_STATUSES
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for status, count in status_counts.items()
        )
        or sum(status_counts.values()) != sample_count
    ):
        raise RuntimeError("Qwen VLM evaluator result has invalid MMMU parser-audit evidence")
    evidence["mmmu_parser_audit"] = {
        "sample_count": sample_count,
        "status_counts": {
            str(status): int(count) for status, count in sorted(status_counts.items())
        },
    }
    return evidence


def _with_evaluation_identity(result: Mapping[str, Any], profile_path: Path) -> dict[str, Any]:
    """Attach adapter-owned comparison inputs and observed evidence."""

    contract = _evaluation_contract(profile_path)
    result_path = result.get("result_path")
    if not isinstance(result_path, (str, Path)):
        raise RuntimeError("Qwen VLM evaluator result is missing result_path")
    return {
        **result,
        "profile_path": str(profile_path),
        "contract": contract,
        "evidence": _evaluation_evidence(result_path, contract),
    }


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
        raise ValueError("pinned VLM evaluation profile requires row_manifest_sha256")
    if row_manifest is not None and evaluation_profile is not None:
        raise ValueError("an embedded evaluation profile cannot be overridden by row_manifest")
    if expected_manifest_sha256 is not None and (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
    ):
        raise ValueError("pinned VLM manifest SHA256 must be 64 lowercase hex characters")
    quick_manifest = Path(row_manifest).expanduser().absolute() if row_manifest else None
    if quick_manifest is not None:
        actual_manifest_sha256 = suites.manifest_sha256(suites.load_quick_manifest(quick_manifest))
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "pinned VLM manifest SHA256 differs from the profile identity: "
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
        report = {
            **report,
            "post_mip_runner_overrides": {
                key: settings[key] for key in sorted(_RUNNER_OVERRIDES) if key in settings
            },
        }
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


def _evaluate_single_run(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
    suite: str,
    invalid_run_message: str,
    evaluation_profile: str | None = None,
    require_manifest: bool = False,
    include_checkpoint: bool = True,
) -> dict[str, Any]:
    """Run one profile and normalize its single-result payload."""

    args, result, profile_path = _run_profile(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite=suite,
        evaluation_profile=evaluation_profile,
        require_manifest=require_manifest,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError(invalid_run_message)
    payload = _with_evaluation_identity(runs[0], profile_path)
    if include_checkpoint:
        payload["checkpoint"] = str(args.checkpoint)
    return payload


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

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="quick",
        invalid_run_message="pinned VLM frozen 344-row profile returned an invalid run count",
        require_manifest=True,
    )


def evaluate_frozen_campaign_v2_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint on the current-image frozen campaign profile."""

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="short",
        evaluation_profile="core-3_344-examples_r1-native",
        invalid_run_message="pinned VLM frozen 344-row profile returned an invalid run count",
        require_manifest=True,
    )


def evaluate_frozen_campaign_v3_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate heterogeneous materialized checkpoints with the current vLLM profile."""

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        evaluation_profile="core-3_344-examples_r1-vllm",
        invalid_run_message="pinned VLM frozen 344-row profile returned an invalid run count",
        require_manifest=True,
    )


def evaluate_reproducibility_smoke_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one checkpoint on the immutable 24-row lifecycle smoke."""

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        evaluation_profile="core-3_24-examples_r1-native",
        invalid_run_message="pinned VLM 24-row smoke returned an invalid run count",
        require_manifest=True,
    )


def evaluate_reproducibility_smoke_v2_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a heterogeneous materialized checkpoint on the 24-row smoke."""

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        evaluation_profile="core-3_24-examples_r1-vllm",
        invalid_run_message="pinned VLM 24-row smoke returned an invalid run count",
        require_manifest=True,
    )


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

    return _evaluate_single_run(
        checkpoint_path,
        output_root=output_root,
        settings=settings,
        suite="realworldqa-smoke",
        invalid_run_message="pinned RealWorldQA profile returned an invalid run count",
        include_checkpoint=False,
    )


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
    return _with_evaluation_identity(
        {
            "metrics": metrics,
            "profile": _BOUNDED_REPEATED_PROFILE,
            "result_path": str(summary_path),
            "run_result_paths": result_paths,
        },
        profile_path,
    )


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
