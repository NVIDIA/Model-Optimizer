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

"""Parallelism-aware activation-scoring hooks (FFN, attention, MoE, Mamba)."""

from .attention import GroupedAttentionScorer
from .base import ScoringHook
from .embedding import HiddenWidthSiteScorer
from .ffn import FFNIndependentScorer, FFNIterativeScorer
from .gated_delta_net import GatedDeltaNetActivationScorer
from .magnitude import ActivationMagnitudeScorer
from .mamba import MambaInProjContributionScorer
from .moe import (
    MoEExpertRemovalDiffScorer,
    MoEGroupedExpertChannelScorer,
    MoELatentCalibrationScorer,
    MoESharedExpertChannelScorer,
)

__all__ = [
    "FFNIndependentScorer",
    "FFNIterativeScorer",
    "GroupedAttentionScorer",
    "GatedDeltaNetActivationScorer",
    "HiddenWidthSiteScorer",
    "MambaInProjContributionScorer",
    "ActivationMagnitudeScorer",
    "MoEExpertRemovalDiffScorer",
    "MoEGroupedExpertChannelScorer",
    "MoELatentCalibrationScorer",
    "MoESharedExpertChannelScorer",
    "ScoringHook",
]
