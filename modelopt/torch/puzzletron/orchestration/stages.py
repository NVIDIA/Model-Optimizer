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

"""Dependency-light access to Puzzletron's canonical stage graph."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = [
    "configured_parent_stage_ids",
    "configured_stage_ids",
    "distributed_stage_ids",
    "enabled_stage_ids",
    "StageSkipReason",
    "StageStatus",
    "StageTerminalState",
    "selected_parent_stage_ids",
    "semantic_stage_config",
    "stage_display_name",
    "stage_is_enabled",
    "stage_spec",
    "stage_terminal_state",
    "topological_mapping_items",
    "topological_stage_ids",
]

_MODULE_NAME = "_puzzletron_orchestrator_stage_graph"
_GRAPH_PATH = Path(__file__).resolve().parents[1] / "stages" / "graph.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _GRAPH_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Puzzletron stage graph from {_GRAPH_PATH}")
_GRAPH = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _GRAPH)
_SPEC.loader.exec_module(_GRAPH)

_SEMANTICS_MODULE_NAME = "_puzzletron_orchestrator_stage_semantics"
_SEMANTICS_PATH = Path(__file__).resolve().parents[1] / "stage_semantics.py"
_SEMANTICS_SPEC = importlib.util.spec_from_file_location(_SEMANTICS_MODULE_NAME, _SEMANTICS_PATH)
if _SEMANTICS_SPEC is None or _SEMANTICS_SPEC.loader is None:
    raise ImportError(f"Unable to load Puzzletron stage semantics from {_SEMANTICS_PATH}")
_SEMANTICS = importlib.util.module_from_spec(_SEMANTICS_SPEC)
sys.modules.setdefault(_SEMANTICS_MODULE_NAME, _SEMANTICS)
_SEMANTICS_SPEC.loader.exec_module(_SEMANTICS)

configured_stage_ids = _GRAPH.configured_stage_ids
configured_parent_stage_ids = _GRAPH.configured_parent_stage_ids
distributed_stage_ids = _GRAPH.distributed_stage_ids
enabled_stage_ids = _GRAPH.enabled_stage_ids
StageSkipReason = _GRAPH.StageSkipReason
StageStatus = _GRAPH.StageStatus
StageTerminalState = _GRAPH.StageTerminalState
selected_parent_stage_ids = _GRAPH.selected_parent_stage_ids
semantic_stage_config = _SEMANTICS.semantic_stage_config


def stage_display_name(stage_id: str, *, granularity: str | None = None) -> str:
    if stage_id.startswith("post."):
        return stage_id.split(".", 2)[-1].replace("_", " ").title()
    return _GRAPH.stage_display_name(stage_id, granularity=granularity)


stage_spec = _GRAPH.stage_spec
stage_is_enabled = _GRAPH.stage_is_enabled
stage_terminal_state = _GRAPH.stage_terminal_state
topological_mapping_items = _GRAPH.topological_mapping_items
topological_stage_ids = _GRAPH.topological_stage_ids
