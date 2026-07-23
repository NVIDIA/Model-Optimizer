# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light access to Puzzletron's canonical stage graph."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = [
    "distributed_stage_ids",
    "enabled_stage_ids",
    "selected_parent_stage_ids",
    "stage_display_name",
    "stage_spec",
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

distributed_stage_ids = _GRAPH.distributed_stage_ids
enabled_stage_ids = _GRAPH.enabled_stage_ids
selected_parent_stage_ids = _GRAPH.selected_parent_stage_ids
def stage_display_name(stage_id: str, *, granularity: str | None = None) -> str:
    if stage_id.startswith("post."):
        return stage_id.split(".", 2)[-1].replace("_", " ").title()
    return _GRAPH.stage_display_name(stage_id, granularity=granularity)


stage_spec = _GRAPH.stage_spec
topological_mapping_items = _GRAPH.topological_mapping_items
topological_stage_ids = _GRAPH.topological_stage_ids
