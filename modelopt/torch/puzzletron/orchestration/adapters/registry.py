# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter selection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schema import ExecutionStrategy, StagePlanNode
from .pool import PersistentPoolAdapter
from .post_mip import PostMIPAdapter
from .sharded import ShardedStageAdapter
from .stage_compat import StageCompatAdapter

if TYPE_CHECKING:
    from .base import WorkAdapter

__all__ = ["adapter_for_stage"]


def adapter_for_stage(node: StagePlanNode) -> WorkAdapter:
    if node.stage_id.startswith("post."):
        return PostMIPAdapter()
    if node.strategy is ExecutionStrategy.SHARDED:
        return ShardedStageAdapter()
    if node.strategy is ExecutionStrategy.PERSISTENT_POOL:
        return PersistentPoolAdapter()
    return StageCompatAdapter()
