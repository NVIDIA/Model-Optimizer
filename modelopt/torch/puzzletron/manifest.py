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

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .identity import canonicalize, stable_hash

__all__ = [
    "StageManifest",
    "read_stage_manifest",
    "semantic_stage_config",
    "write_stage_manifest",
]


_SHARED_SEMANTIC_CONFIG_SECTIONS = (
    "model",
    "data",
    "dataset",
    "parallel",
    "search_space",
    "embedding_pruning",
    "granularity",
    "capability_validation",
)

_STAGE_SEMANTIC_CONFIG_SECTIONS = {
    "convert": ("convert",),
    "tokenize_data": (
        "tokenize_data",
        "convert",
        "dataset_path",
        "pruning",
        "replacement_scoring",
    ),
    "width_importance": ("width_importance", "pruning"),
    "sort": ("sort", "pruning"),
    "sort_sanity": ("sort_sanity", "sanity", "sort", "pruning", "replacement_scoring"),
    "slicing_sanity": (
        "slicing_sanity",
        "sanity",
        "sort",
        "pruning",
        "replacement_scoring",
    ),
    "width_sanity": ("width_sanity", "sanity", "pruning", "replacement_scoring"),
    "bypass_sanity": ("bypass_sanity", "sanity", "bypass", "pruning"),
    "bypass": ("bypass", "pruning"),
    "depth_importance": ("depth_importance", "pruning", "replacement_scoring"),
    "vllm_stats": (
        "vllm_stats",
        "build_library",
        "library",
    ),
    "build_library": (
        "build_library",
        "vllm_stats",
        "library",
        "bypass",
    ),
    "replacement_scoring": (
        "replacement_scoring",
        "build_library",
        "library",
        "pruning",
    ),
    "mip": (
        "mip",
        "realize_model",
        "replacement_scoring",
        "vllm_stats",
        "library",
        "bypass",
    ),
    "zero_shot_evaluation": ("zero_shot_evaluation", "convert", "replacement_scoring"),
    "aiperf": ("aiperf", "zero_shot_evaluation"),
    "global_distillation_sanity": (
        "global_distillation_sanity",
        "sanity",
        "global_distillation",
        "replacement_scoring",
        "calibration",
    ),
    "global_distillation": (
        "global_distillation",
        "zero_shot_evaluation",
        "replacement_scoring",
    ),
    "post_distillation_evaluation": (
        "post_distillation_evaluation",
        "global_distillation",
        "zero_shot_evaluation",
        "replacement_scoring",
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def semantic_stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return configuration that can change the semantic result of one stage."""

    sections = dict.fromkeys(
        (*_SHARED_SEMANTIC_CONFIG_SECTIONS, *_STAGE_SEMANTIC_CONFIG_SECTIONS.get(stage, (stage,)))
    )
    return {key: config[key] for key in sections if key in config}


@dataclass
class StageManifest:
    """Durable metadata and semantic identity for one Puzzletron stage execution."""

    stage: str
    version: str = "1"
    status: str = "pending"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    capability_snapshot: dict[str, Any] | None = None
    semantic_config: dict[str, Any] | None = None
    implementation_provenance: dict[str, Any] = field(default_factory=dict)
    stale_reason: str | None = None
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None

    @property
    def config_identity(self) -> str:
        return stable_hash(self.config, prefix=f"{self.stage}_cfg")

    @property
    def semantic_config_identity(self) -> str:
        """Return the identity of configuration relevant to this stage's result."""

        config = self.semantic_config
        if config is None:
            config = semantic_stage_config(self.config, self.stage)
        return stable_hash(config, prefix=f"{self.stage}_semantic_cfg")

    @property
    def semantic_identity(self) -> str:
        """Return the compatibility identity consumed by downstream resume checks."""

        payload = {
            "stage": self.stage,
            "semantic_config_identity": self.semantic_config_identity,
            "capability_snapshot": self.capability_snapshot,
        }
        return stable_hash(payload, prefix=f"{self.stage}_semantic")

    def complete(self, *, outputs: dict[str, Any] | None = None, status: str = "success") -> None:
        """Mark the stage complete with its validated outputs and final status."""

        if outputs is not None:
            self.outputs = outputs
        self.status = status
        self.ended_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Return the backward-compatible serialized manifest payload."""

        return {
            "stage": self.stage,
            "version": self.version,
            "status": self.status,
            "inputs": canonicalize(self.inputs),
            "outputs": canonicalize(self.outputs),
            "config": canonicalize(self.config),
            "config_identity": self.config_identity,
            "semantic_config": canonicalize(
                self.semantic_config
                if self.semantic_config is not None
                else semantic_stage_config(self.config, self.stage)
            ),
            "semantic_config_identity": self.semantic_config_identity,
            "semantic_identity": self.semantic_identity,
            "implementation_provenance": canonicalize(self.implementation_provenance),
            "capability_snapshot": canonicalize(self.capability_snapshot),
            "stale_reason": self.stale_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


def write_stage_manifest(path: str | Path, manifest: StageManifest) -> None:
    """Atomically write a stage manifest from rank zero."""

    if os.environ.get("RANK") not in (None, "", "0"):
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_stage_manifest(path: str | Path) -> dict[str, Any]:
    """Read a stage manifest while preserving compatibility with older schemas."""

    return json.loads(Path(path).read_text())
