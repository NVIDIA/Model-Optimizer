# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NeMo-AutoModel backend for Puzzletron activation scoring.

This package loads a converted AnyModel teacher checkpoint into a NeMo AutoModel
(parallelized with DP/FSDP/TP/SP/CP/EP/PP) and runs the pruning activation-scoring
hooks so that the aggregated per-target scores are identical regardless of the
parallel layout or the number of nodes.

The :mod:`reduction` module is the parallelism-aware core that every ported hook
builds on (see the plan, section "the core: parallelism-aware aggregation").
"""

from .hooks import (
    FFNIndependentScorer,
    FFNIterativeScorer,
    GroupedAttentionScorer,
    HiddenWidthSiteScorer,
    ScoringHook,
)
from .load import load_anymodel_for_scoring, validate_force_hf_ep
from .output import write_scores
from .patch import apply_patch, auto_detect_block_configs, load_block_configs, remove_patch
from .reduction import (
    MeshGroups,
    finalize_additive,
    full_weight,
    gather_scored_axis,
    is_writer,
    reduce_token_sum,
    to_local_with_feature_group,
    writer_shard_id,
)
from .target_resolver import build_scorers

__all__ = [
    "FFNIndependentScorer",
    "FFNIterativeScorer",
    "GroupedAttentionScorer",
    "HiddenWidthSiteScorer",
    "MeshGroups",
    "ScoringHook",
    "apply_patch",
    "auto_detect_block_configs",
    "build_scorers",
    "finalize_additive",
    "full_weight",
    "gather_scored_axis",
    "is_writer",
    "load_anymodel_for_scoring",
    "load_block_configs",
    "reduce_token_sum",
    "remove_patch",
    "to_local_with_feature_group",
    "validate_force_hf_ep",
    "write_scores",
    "writer_shard_id",
]
from .batch_adapter import (
    VisionForwardMonitor,
    validate_native_feature_config,
    validated_forward_kwargs,
)
