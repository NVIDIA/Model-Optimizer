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

"""MoE-block detection.

Whether a module *is* an MoE block is a modeling question, not an export one, so it
lives beside the per-model specs that answer it rather than in an export utility
module. Unlike ``specs.py`` this needs torch at runtime: the last resort is a
structural check on the module's children, which no amount of per-model data can
replace for a model nobody has registered yet.
"""

import torch.nn as nn

from .specs import match_moe_block

__all__ = ["is_moe"]


def is_moe(module: nn.Module, model_type: str | None = None) -> bool:
    """Return whether ``module`` is an MoE block.

    ``model_type`` (``model.config.model_type``) scopes the registry lookup to the
    model's own spec; ``None`` searches every spec.

    The model's own spec is consulted first, so per-model data always outranks the
    generic name and structural fallbacks. The fallbacks matter: a model with no spec
    still has to be recognised, which is what keeps export working for MoE
    architectures nobody has registered.
    """
    # Per-model data (modelopt/torch/models/*), which also covers non-standard names.
    if match_moe_block(module, model_type) is not None:
        return True
    # Generic fallback: the common MoE block naming conventions.
    name = type(module).__name__.lower()
    if name.endswith("sparsemoeblock") or "moelayer" in name:
        return True
    # Structural fallback: modules with router + experts (e.g. Gemma4TextDecoderLayer)
    return (
        hasattr(module, "router")
        and hasattr(module, "experts")
        and isinstance(module.experts, nn.Module)
    )
