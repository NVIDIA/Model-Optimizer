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

"""Execution-contract identity hashing for orchestration attempts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .schema import ExecutionContract, RunnerEnvironment

__all__ = [
    "artifact_snapshot_identity",
    "canonicalize",
    "execution_contract_hash",
    "hash_payload",
    "mip_input_artifact_paths",
    "stable_hash",
]

_CORE_IDENTITY_NAME = "_puzzletron_core_identity"
_CORE_IDENTITY_PATH = Path(__file__).resolve().parents[1] / "identity.py"
_CORE_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    _CORE_IDENTITY_NAME, _CORE_IDENTITY_PATH
)
if _CORE_IDENTITY_SPEC is None or _CORE_IDENTITY_SPEC.loader is None:
    raise ImportError(f"Unable to load Puzzletron identity helpers from {_CORE_IDENTITY_PATH}")
_CORE_IDENTITY = importlib.util.module_from_spec(_CORE_IDENTITY_SPEC)
sys.modules.setdefault(_CORE_IDENTITY_NAME, _CORE_IDENTITY)
_CORE_IDENTITY_SPEC.loader.exec_module(_CORE_IDENTITY)

canonicalize = _CORE_IDENTITY.canonicalize
stable_hash = _CORE_IDENTITY.stable_hash


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA256 digest for a JSON-serializable payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def mip_input_artifact_paths(
    puzzle_dir: str | Path, widths: list[int], score_granularity: str
) -> dict[str, Path]:
    """Return the scored/statistical artifacts that determine a MIP execution."""

    root = Path(puzzle_dir)
    score_name = {
        "block": "single_sequence_replacement_solutions--validation",
        "subblock": "single_subblock_replacement_solutions--validation",
    }[str(score_granularity).lower()]
    paths = {
        f"manifest/{stage}": root / "manifests" / f"{stage}.json"
        for stage in (
            "sort",
            "width_importance",
            "depth_importance",
            "build_library",
            "vllm_stats",
            "replacement_scoring",
        )
    }
    for width in widths:
        base = root / "scenarios" / f"width-{int(width):04d}" / "depth-00"
        paths.update(
            {
                f"width/{width}/stats": base / "subblock_stats.json",
                f"width/{width}/scores": base / score_name,
                f"width/{width}/canonical": base / "single_sequence_replacement_solutions.json",
                f"width/{width}/library": base / "replacement_library.json",
                f"width/{width}/teacher_config": base / "ckpts" / "sorted_teacher" / "config.json",
                f"width/{width}/teacher_index": base
                / "ckpts"
                / "sorted_teacher"
                / "model.safetensors.index.json",
            }
        )
    return paths


def artifact_snapshot_identity(paths: Mapping[str, str | Path]) -> str:
    """Fingerprint input artifacts by path, metadata, and small control-file content."""

    rows = []
    for label, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        members = (
            sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        )
        if not path.exists():
            rows.append({"label": label, "missing": True})
            continue
        for member in members:
            stat = member.stat()
            row = {
                "label": label,
                "path": str(member.relative_to(path)) if path.is_dir() else path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if member.suffix.lower() in {".json", ".yaml", ".yml"} and stat.st_size <= 1 << 20:
                row["sha256"] = hashlib.sha256(member.read_bytes()).hexdigest()
            rows.append(row)
    return hash_payload({"artifacts": rows})


def execution_contract_hash(runner: RunnerEnvironment) -> str:
    """Hash the immutable execution contract for one campaign."""

    contract = runner.contract
    payload = {
        "repository": contract.repository,
        "venv": contract.venv,
        "container": contract.container,
        "container_mounts": contract.container_mounts,
        "setup_env": contract.setup_env,
        "prerun_commands": list(contract.prerun_commands),
        "postrun_commands": list(contract.postrun_commands),
        "runner_kind": runner.kind,
        "task_topology_contract": 1,
    }
    if runner.slurm is not None:
        payload["slurm"] = {
            "account": runner.slurm.account,
            "job_name_prefix": runner.slurm.job_name_prefix,
            "partition": runner.slurm.partition,
            "partition_interactive": runner.slurm.partition_interactive,
            "partition_batch": runner.slurm.partition_batch,
            "partition_cpu": runner.slurm.partition_cpu,
            "interactive_max_nodes": runner.slurm.interactive_max_nodes,
            "max_nodes": runner.slurm.max_nodes,
            "time_limit": runner.slurm.time_limit,
            "qos": runner.slurm.qos,
            "log_dir": runner.slurm.log_dir,
        }
    if runner.baremetal is not None:
        payload["baremetal"] = {
            "hosts": [(host.hostname, host.gpus) for host in runner.baremetal.hosts],
            "rendezvous_host": runner.baremetal.rendezvous_host,
            "rendezvous_port_base": runner.baremetal.rendezvous_port_base,
        }
    return hash_payload(payload)


def with_contract_hash(runner: RunnerEnvironment) -> ExecutionContract:
    """Attach a content hash to the runner execution contract."""

    digest = execution_contract_hash(runner)
    contract = runner.contract
    if contract.contract_hash == digest:
        return contract
    updated = ExecutionContract(
        repository=contract.repository,
        venv=contract.venv,
        container=contract.container,
        container_mounts=contract.container_mounts,
        setup_env=contract.setup_env,
        prerun_commands=contract.prerun_commands,
        postrun_commands=contract.postrun_commands,
        contract_hash=digest,
    )
    return updated
