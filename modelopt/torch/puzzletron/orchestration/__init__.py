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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral Puzzletron campaign orchestration."""

from .compiler import compile_campaign_plan, plan_to_dict
from .controller import CampaignController
from .mesh import ParallelMesh, normalize_vllm_topology, pack_gpu_allocation
from .schema import (
    AttemptSpec,
    CampaignPlan,
    CommandSpec,
    ExecutionStrategy,
    FailureClass,
    FailurePolicy,
    HaltPolicy,
    JobHandle,
    JobStatus,
    ParallelMeshOverride,
    RunnerEnvironment,
    StageExecutionSpec,
    TaskLauncher,
    TaskTopology,
    WorkItem,
    WorkPlan,
)
from .terminal import ShutdownAction, TerminalControls

__all__ = [
    "AttemptSpec",
    "CampaignController",
    "CampaignPlan",
    "CommandSpec",
    "ExecutionStrategy",
    "FailureClass",
    "FailurePolicy",
    "HaltPolicy",
    "JobHandle",
    "JobStatus",
    "ParallelMesh",
    "ParallelMeshOverride",
    "RunnerEnvironment",
    "StageExecutionSpec",
    "TaskLauncher",
    "TaskTopology",
    "ShutdownAction",
    "TerminalControls",
    "WorkItem",
    "WorkPlan",
    "compile_campaign_plan",
    "normalize_vllm_topology",
    "pack_gpu_allocation",
    "plan_to_dict",
]
