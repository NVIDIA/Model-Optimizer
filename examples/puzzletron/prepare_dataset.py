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

"""Prepare an immutable dataset snapshot as an ordinary resumable stage."""

from __future__ import annotations

import os
from pathlib import Path

from examples.puzzletron.evaluation.vlm.preparation.benchmark_data import prepare_benchmark_datasets
from modelopt.torch.puzzletron.dataset.acquisition import (
    ACQUISITION_MANIFEST,
    VLM_HEADER_SUBSETS,
    VlmAcquisitionSpec,
    materialize_nemotron_vlm_dataset,
)
from modelopt.torch.puzzletron.manifest import stage_manifest_from_config, write_stage_manifest
from modelopt.torch.puzzletron.orchestration.dataset_payload import record_file_inventory
from modelopt.torch.puzzletron.stage_runner import StageResult
from modelopt.torch.puzzletron.stages.graph import StageSkipReason, stage_is_enabled

__all__ = ["prepare_dataset_stage"]


def _completion_inventories(output: Path, evaluation_data: list[dict]) -> dict:
    """Seal stage-owned payloads without exposing dataset policy to orchestration."""

    inventories = [record_file_inventory(output)]
    for row in evaluation_data:
        snapshot = Path(row["snapshot"]).expanduser().absolute()
        inventories.append(
            record_file_inventory(snapshot, allowed_symlink_root=snapshot.parent.parent)
        )
        media_root = row.get("media_root")
        if media_root:
            inventories.append(
                record_file_inventory(
                    media_root,
                    ignored_names=(".modelopt_vlm_benchmark_preparation.json",),
                )
            )
    return {
        "schema": "modelopt.puzzletron.file-inventories/v1",
        "inventories": inventories,
    }


def prepare_dataset_stage(config: dict) -> StageResult:
    """Materialize or validate the configured bounded VLM dataset."""

    stage_config = config.get("prepare_dataset") or {}
    puzzle_dir = Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])
    manifest_path = puzzle_dir / "manifests" / "prepare_dataset.json"
    manifest = stage_manifest_from_config("prepare_dataset", config)
    if not stage_is_enabled("prepare_dataset", config):
        skip_reason = StageSkipReason.DISABLED
        manifest.complete(outputs={"enabled": False}, status="skipped", skip_reason=skip_reason)
        write_stage_manifest(manifest_path, manifest)
        return StageResult(
            "prepare_dataset",
            "skipped",
            manifest_path,
            "Dataset preparation is disabled.",
            skip_reason.value,
        )

    adapter = stage_config.get("adapter", "nemotron_vlm_v2")
    if adapter != "nemotron_vlm_v2":
        raise ValueError(f"unsupported prepare_dataset adapter {adapter!r}")
    output = Path(stage_config.get("output") or config["dataset_path"])
    subsets = tuple(stage_config.get("subsets") or VLM_HEADER_SUBSETS)
    raw_subset_rows = stage_config.get("subset_rows") or {}
    if not isinstance(raw_subset_rows, dict):
        raise ValueError("prepare_dataset.subset_rows must be a mapping")
    result = materialize_nemotron_vlm_dataset(
        VlmAcquisitionSpec(
            output_dir=output,
            subsets=subsets,
            subset_rows=tuple((name, raw_subset_rows[name]) for name in subsets)
            if raw_subset_rows
            else (),
            num_samples=stage_config.get("num_samples", 512),
            seed=stage_config.get("seed", 42),
            max_shards_per_subset=stage_config.get("max_shards_per_subset", 1),
            revision=stage_config.get("revision") or (config.get("data") or {}).get("revision"),
        )
    )
    raw_evaluation_tasks = stage_config.get("evaluation_tasks") or ()
    if isinstance(raw_evaluation_tasks, str) or not isinstance(raw_evaluation_tasks, (list, tuple)):
        raise TypeError("prepare_dataset.evaluation_tasks must be a list of task names")
    evaluation_tasks = tuple(raw_evaluation_tasks)
    evaluation_data = []
    evaluation_hf_home = None
    if evaluation_tasks:
        configured_hf_home = stage_config.get("evaluation_hf_home")
        runner_hf_home = os.environ.get("HF_HOME")
        if not configured_hf_home:
            raise ValueError("prepare_dataset.evaluation_tasks requires evaluation_hf_home")
        evaluation_hf_home = Path(str(configured_hf_home)).expanduser().absolute()
        if runner_hf_home and evaluation_hf_home != Path(runner_hf_home).expanduser().absolute():
            raise ValueError("prepare_dataset.evaluation_hf_home differs from runner HF_HOME")
        raw_catalog = stage_config.get("evaluation_datasets")
        if raw_catalog is not None and not isinstance(raw_catalog, dict):
            raise TypeError("prepare_dataset.evaluation_datasets must be a mapping")
        evaluation_catalog = None
        if isinstance(raw_catalog, dict):
            evaluation_catalog = {
                task: raw_catalog[task] for task in evaluation_tasks if task in raw_catalog
            }
            if len(evaluation_catalog) != len(evaluation_tasks):
                raise ValueError("prepare_dataset.evaluation_datasets is missing a configured task")
        evaluation_data = prepare_benchmark_datasets(
            evaluation_hf_home,
            evaluation_tasks,
            max_workers=int(stage_config.get("evaluation_max_workers", 8)),
            range_resume=bool(stage_config.get("evaluation_range_resume", False)),
            verify_content=bool(stage_config.get("verify_content", False)),
            expected_catalog=evaluation_catalog,
        )
    acquisition_path = output / ACQUISITION_MANIFEST
    manifest.complete(
        outputs={
            "dataset_path": str(output),
            "acquisition_manifest": str(acquisition_path),
            "sample_count": result["sample_count"],
            "acquisition": result["acquisition"],
            "evaluation_hf_home": str(evaluation_hf_home) if evaluation_hf_home else None,
            "evaluation_data": evaluation_data,
            "completion": _completion_inventories(output, evaluation_data),
        }
    )
    write_stage_manifest(manifest_path, manifest)
    return StageResult(
        "prepare_dataset",
        "success",
        manifest_path,
        f"Prepared {result['sample_count']} immutable VLM rows at {output}.",
    )
