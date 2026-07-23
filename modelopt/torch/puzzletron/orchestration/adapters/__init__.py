# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage work adapters for campaign orchestration."""

from .base import WorkAdapter
from .pool import PersistentPoolAdapter
from .registry import adapter_for_stage
from .sharded import ShardedStageAdapter
from .stage_compat import StageCompatAdapter

__all__ = [
    "PersistentPoolAdapter",
    "ShardedStageAdapter",
    "StageCompatAdapter",
    "WorkAdapter",
    "adapter_for_stage",
]
