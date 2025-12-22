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

"""Plugin to add NAS support for Transformer Engine modules."""

import transformer_engine as te

from ..modules import _DynamicLayerNorm
from ..registry import DMRegistry
from ..traced_hp import TracedHp

__all__ = ["_DynamicTENorm"]


@DMRegistry.register(
    {te.pytorch.LayerNorm: "te.pytorch.LayerNorm", te.pytorch.RMSNorm: "te.pytorch.RMSNorm"}
)
class _DynamicTENorm(_DynamicLayerNorm):
    """A ``te.pytorch.{Layer/RMS}Norm`` layer with dynamic hyperparams."""

    def _setup(self, *, num_features: TracedHp):
        """Setup the TENorm dynamic module with pre-defined num_features hparam."""
        self._register_hparam("num_features", num_features)
        # register dynamic attributes
        self._register_dynamic_attribute("weight", self._cut_to_active_features)
        if hasattr(self, "bias"):  # Bias is not present in RMSNorm
            self._register_dynamic_attribute("bias", self._cut_to_active_features)
