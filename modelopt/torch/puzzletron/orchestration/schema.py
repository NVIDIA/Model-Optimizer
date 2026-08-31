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

"""Public contracts for Puzzletron campaign orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, cast


def normalize_slurm_partition(value: Any, *, path: str) -> str | None:
    """Normalize one or more eligible Slurm partitions for ``--partition``."""

    if value is None:
        return None
    values = value.split(",") if isinstance(value, str) else value
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{path} must be a partition name or a sequence of names")
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{path} must contain only partition names")
    partitions = [item.strip() for item in values]
    if not partitions or any(not item for item in partitions):
        raise ValueError(f"{path} must contain at least one non-empty partition name")
    if any("," in item or any(character.isspace() for character in item) for item in partitions):
        raise ValueError(f"{path} contains an invalid partition name")
    if len(set(partitions)) != len(partitions):
        raise ValueError(f"{path} contains duplicate partition names")
    return ",".join(partitions)


class ExecutionStrategy(str, Enum):
    """How one stage maps work items onto model instances."""

    SINGLE = "single"
    SHARDED = "sharded"
    PERSISTENT_POOL = "persistent_pool"


class FailurePolicy(str, Enum):
    """Named retry/resume policy for campaign attempts."""

    STRICT = "strict"
    RESUME = "resume"


class HaltPolicy(str, Enum):
    """When to stop the campaign after a stage attempt fails."""

    DRAIN = "drain"
    FAIL_FAST = "fail_fast"


class FailureClass(str, Enum):
    """Semantic failure classification for policy decisions."""

    SUCCESS = "success"
    TRANSIENT = "transient"
    TIMEOUT_RESUMABLE = "timeout_resumable"
    TIMEOUT_FATAL = "timeout_fatal"
    OOM = "oom"
    APPLICATION = "application"
    CONFIG = "config"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class JobState(str, Enum):
    """Executor-reported job lifecycle state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class TaskLauncher(str, Enum):
    """How an executor starts processes inside one scheduler task."""

    DIRECT = "direct"
    TORCHRUN = "torchrun"


@dataclass(frozen=True)
class TaskTopology:
    """Scheduler-neutral task and distributed-group layout for one attempt."""

    task_count: int = 1
    gpus_per_task: int | None = None
    tasks_per_group: int = 1
    launcher: TaskLauncher = TaskLauncher.DIRECT
    placement: str = "block"


@dataclass(frozen=True)
class ParallelMeshOverride:
    """Optional per-stage mesh override from execution config."""

    tp: int | None = None
    cp: int | None = None
    pp: int | None = None
    ep: int | None = None
    dp_shard: int | None = None
    dp_replicate: int | None = None


@dataclass(frozen=True)
class StageExecutionSpec:
    """Execution semantics for one stage node."""

    stage_id: str
    strategy: ExecutionStrategy
    instances: int = 1
    failure_policy: FailurePolicy = FailurePolicy.STRICT
    mesh_override: ParallelMeshOverride | None = None
    gpus_per_node: int | None = None
    partition: str | None = None
    resource: str = "gpu"


@dataclass(frozen=True)
class ExecutionContract:
    """Immutable environment contract shared by every attempt."""

    repository: str
    venv: str
    container: str | None = None
    container_mounts: str | None = None
    setup_env: str | None = None
    prerun_commands: tuple[str, ...] = ()
    postrun_commands: tuple[str, ...] = ()
    contract_hash: str = ""


@dataclass(frozen=True)
class SlurmRunnerConfig:
    """Slurm-specific runner facts."""

    account: str
    job_name_prefix: str = "pt"
    partition: str | Sequence[str] | None = None
    partition_interactive: str | Sequence[str] | None = None
    partition_batch: str | Sequence[str] | None = None
    partition_cpu: str | Sequence[str] | None = None
    interactive_max_nodes: int = 2
    max_nodes: int | None = None
    time_limit: str = "4:00:00"
    qos: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        job_name_prefix = str(self.job_name_prefix)
        if not job_name_prefix or any(
            not (character.isascii() and (character.isalnum() or character in "._-"))
            for character in job_name_prefix
        ):
            raise ValueError(
                "runner.slurm.job_name_prefix must contain only ASCII letters, digits, '.', '_', "
                "or '-'"
            )
        object.__setattr__(self, "job_name_prefix", job_name_prefix)
        for field_name in (
            "partition",
            "partition_interactive",
            "partition_batch",
            "partition_cpu",
        ):
            object.__setattr__(
                self,
                field_name,
                normalize_slurm_partition(
                    getattr(self, field_name), path=f"runner.slurm.{field_name}"
                ),
            )
        if self.interactive_max_nodes <= 0:
            raise ValueError("runner.slurm.interactive_max_nodes must be positive")

    def partition_for_nodes(self, nodes: int) -> str | None:
        """Resolve the canonical partition or a deprecated role-based fallback."""

        partition = cast("str | None", self.partition)
        partition_interactive = cast("str | None", self.partition_interactive)
        partition_batch = cast("str | None", self.partition_batch)
        if self.partition_interactive is None and self.partition_batch is None:
            return partition
        interactive = partition_interactive or (partition if partition == "interactive" else None)
        batch = partition_batch or (partition if partition != "interactive" else "batch")
        if interactive and nodes <= self.interactive_max_nodes:
            return interactive
        return batch or partition


@dataclass(frozen=True)
class BareMetalHost:
    """One bare-metal host in the inventory."""

    hostname: str
    gpus: int = 8


@dataclass(frozen=True)
class BareMetalRunnerConfig:
    """Bare-metal inventory and rendezvous defaults."""

    hosts: tuple[BareMetalHost, ...]
    rendezvous_host: str | None = None
    rendezvous_port_base: int = 29500


@dataclass(frozen=True)
class RunnerEnvironment:
    """One scheduler-neutral runner environment bound to a campaign."""

    kind: str
    contract: ExecutionContract
    slurm: SlurmRunnerConfig | None = None
    baremetal: BareMetalRunnerConfig | None = None


@dataclass(frozen=True)
class StagePlanNode:
    """One compiled stage node in the campaign DAG."""

    stage_id: str
    strategy: ExecutionStrategy
    instances: int
    failure_policy: FailurePolicy
    mesh: Mapping[str, int]
    gpus_per_instance: int
    gpus_per_node: int
    nodes: int
    total_gpus: int
    exclusive: bool
    parents: tuple[str, ...]
    distributed: bool
    partition: str | None = None
    resource: str = "gpu"


@dataclass(frozen=True)
class CampaignPlan:
    """Compiled campaign plan bound to one experiment and runner."""

    experiment_config_path: str
    puzzle_dir: Path
    experiment_config: Mapping[str, Any]
    runner: RunnerEnvironment
    execution_defaults: Mapping[str, Any]
    stages: tuple[StagePlanNode, ...]
    contract_hash: str
    overrides: tuple[str, ...] = ()
    final_report_partition: str | None = None

    @property
    def log_dir(self) -> Path:
        """Return the configured shared log directory for every campaign attempt."""

        if self.runner.slurm is not None and self.runner.slurm.log_dir:
            configured = Path(self.runner.slurm.log_dir).expanduser()
            return configured if configured.is_absolute() else self.puzzle_dir / configured
        return self.puzzle_dir / "logs"


@dataclass(frozen=True)
class WorkItem:
    """One unit of work inside a stage plan."""

    work_id: str
    stage_id: str
    shard_index: int
    shard_count: int
    gpus_per_instance: int
    local_gpu_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkPlan:
    """Deterministic work plan for one stage attempt."""

    stage_id: str
    strategy: ExecutionStrategy
    items: tuple[WorkItem, ...]
    aggregate_required: bool = False


@dataclass(frozen=True)
class CommandSpec:
    """Shell command and environment for one attempt."""

    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    log_path: str | None = None
    shell: bool = False


@dataclass(frozen=True)
class AttemptSpec:
    """One submitted attempt for one work item or coordinated stage."""

    attempt_id: str
    work_id: str
    stage_id: str
    command: CommandSpec
    allocation_nodes: int = 1
    allocation_gpus: int = 1
    exclusive: bool = False
    contract_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_topology: TaskTopology = field(default_factory=TaskTopology)


@dataclass(frozen=True)
class JobHandle:
    """Opaque executor handle persisted for recovery."""

    backend: str
    handle_id: str
    attempt_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobStatus:
    """Normalized executor status."""

    handle: JobHandle
    state: JobState
    exit_code: int | None = None
    reason: str | None = None
    log_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResumeDecision:
    """Whether and how to resume one work item."""

    action: str
    reason: str = ""
    skip: bool = False


@dataclass(frozen=True)
class ValidatedResult:
    """Semantic validation outcome for one work item."""

    valid: bool
    reason: str = ""
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublishedOutput:
    """Canonical aggregate output for a multi-worker stage."""

    stage_id: str
    artifacts: tuple[str, ...]
    summary: Mapping[str, Any] = field(default_factory=dict)
