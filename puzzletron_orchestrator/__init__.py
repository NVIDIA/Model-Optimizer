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

"""Dependency-light Puzzletron campaign orchestration."""

from pathlib import Path

# Keep one implementation while exposing it outside ``modelopt.torch``. Importing
# through this package bypasses ModelOpt's eager PyTorch initialization.
_PUZZLETRON_SOURCE = Path(__file__).resolve().parents[1] / "modelopt" / "torch" / "puzzletron"
_SOURCE = _PUZZLETRON_SOURCE / "orchestration"
__path__.append(str(_SOURCE))
__path__.append(str(_PUZZLETRON_SOURCE))

from .compiler import compile_campaign_plan, plan_to_dict  # noqa: E402
from .controller import CampaignController  # noqa: E402
from .mesh import (  # noqa: E402
    ParallelMesh,
    normalize_vllm_topology,
    pack_gpu_allocation,
    vllm_topology_to_mesh,
)
from .schema import (  # noqa: E402
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
from .terminal import ShutdownAction, TerminalControls  # noqa: E402

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
    "ShutdownAction",
    "StageExecutionSpec",
    "TaskLauncher",
    "TaskTopology",
    "TerminalControls",
    "WorkItem",
    "WorkPlan",
    "compile_campaign_plan",
    "normalize_vllm_topology",
    "pack_gpu_allocation",
    "plan_to_dict",
    "vllm_topology_to_mesh",
]
