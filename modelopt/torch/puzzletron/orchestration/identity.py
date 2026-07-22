# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-contract identity hashing for orchestration attempts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .schema import ExecutionContract, RunnerEnvironment

__all__ = ["execution_contract_hash", "hash_payload"]


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA256 digest for a JSON-serializable payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


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
            "partition": runner.slurm.partition,
            "partition_interactive": runner.slurm.partition_interactive,
            "partition_batch": runner.slurm.partition_batch,
            "interactive_max_nodes": runner.slurm.interactive_max_nodes,
            "max_nodes": runner.slurm.max_nodes,
            "time_limit": runner.slurm.time_limit,
            "qos": runner.slurm.qos,
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
