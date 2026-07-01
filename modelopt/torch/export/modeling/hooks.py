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

"""Per-model behavioral hooks for the export engine.

A family subclasses :class:`ModelHooks` and overrides only the seams it needs; every
default method declines (returns the no-op sentinel) so the engine falls back to its
generic path. The engine builds each config object and the hook only routes it, so most
hooks need nothing from :class:`ExportContext`.
"""

import torch.nn as nn

from .context import ExportContext

__all__ = ["NULL_HOOKS", "ModelHooks"]


class ModelHooks:
    """Optional per-model overrides for the export engine's structural seams."""

    def unwrap_decoder_layer(self, module: nn.Module, ctx: ExportContext) -> nn.Module | None:
        """Return the real decoder layer when a family wraps it (e.g. DBRX/ExaOne/Deci).

        Returns ``None`` to use ``module`` unchanged.
        """
        return None

    def place_submodule(
        self, name: str, module: nn.Module, built, layer_config, ctx: ExportContext
    ) -> bool:
        """Route an already-built sub-module config into a family-specific slot.

        ``built`` is the config object the engine produced for ``module`` (e.g. a
        ``LayernormConfig`` or ``AttentionConfig``); assign it onto ``layer_config`` and
        return ``True`` if handled, else return ``False`` to use the default placement.
        """
        return False

    def build_moe(self, module: nn.Module, ctx: ExportContext):
        """Build and return this family's MoE config, or ``None`` to use the default."""
        return None


# Shared no-op instance used when a model has no overrides.
NULL_HOOKS = ModelHooks()
