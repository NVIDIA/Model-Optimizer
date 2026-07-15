# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import modelopt.torch.utils.distributed as dist

from ..manifest import StageManifest
from ..pipeline_config import load_runtime_hydra_config
from ..scoring_parent import ensure_scoring_parent
from .common import complete_stage
from .pipeline import _distributed

__all__ = ["depth_stage"]


def _resolve_depth_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    configured = (config.get("depth_importance") or {}).get("source_checkpoint_dir")
    if configured:
        source = Path(configured).resolve()
        if not (source / "config.json").is_file():
            raise FileNotFoundError(f"depth source is not a usable checkpoint: {source}")
        from ..distributed_eval.config import checkpoint_identity

        identity = checkpoint_identity(source)
        return source, {
            "path": str(source),
            "role": "configured_depth_source",
            "fingerprint": identity["fingerprint"],
        }
    parent = ensure_scoring_parent(config, refresh=True)
    return parent.path, parent.to_dict()


def depth_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    source, parent_record = _resolve_depth_source(config)
    hydra_cfg.depth.source_checkpoint_dir = str(source)
    with _distributed(hydra_cfg):
        from ..depth import launch_iterative_depth_automodel

        outputs = launch_iterative_depth_automodel(hydra_cfg)
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={**outputs, "scoring_parent": parent_record},
    )
