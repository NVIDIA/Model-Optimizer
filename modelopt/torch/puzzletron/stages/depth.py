# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import modelopt.torch.utils.distributed as dist

from ..manifest import StageManifest
from ..pipeline_config import load_runtime_hydra_config
from ..scoring_parent import ensure_scoring_parent
from .common import complete_stage
from .pipeline import _distributed

__all__ = ["depth_stage"]


def depth_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    # Bypass publishes its checkpoint before ``complete_stage`` writes the final
    # manifest. Refresh here so the content-addressed parent captures both the
    # final checkpoint identity and the final successful upstream manifest.
    parent = ensure_scoring_parent(config, refresh=True)
    hydra_cfg.depth.source_checkpoint_dir = str(parent.path)
    with _distributed(hydra_cfg):
        from ..depth import launch_iterative_depth_automodel

        outputs = launch_iterative_depth_automodel(hydra_cfg)
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={**outputs, "scoring_parent": parent.to_dict()},
    )
