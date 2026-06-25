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

"""Registry that maps a model (or one of its sub-modules) to its ``ModelSpec``.

Families register their specs at import time (see ``families/``). Lookups return
``None`` when nothing matches, so the generic export code can fall back to its legacy
path — this is what makes the migration incremental and safe.
"""

import torch.nn as nn

from .base import ModelSpec

__all__ = ["match_by_architecture", "match_by_decoder_type", "match_moe_block", "register"]

_SPECS: list[ModelSpec] = []


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model-family spec. Returns the spec for convenient module-level use."""
    _SPECS.append(spec)
    return spec


def match_by_architecture(architecture: str) -> ModelSpec | None:
    """Return the spec whose ``architectures`` contains an exact ``architecture`` match."""
    for spec in _SPECS:
        if architecture in spec.architectures:
            return spec
    return None


def match_by_decoder_type(decoder_type: str) -> ModelSpec | None:
    """Return the spec whose ``decoder_types`` contains an exact ``decoder_type`` match.

    ``decoder_type`` is ModelOpt's normalized family string (from ``MODEL_NAME_TO_TYPE``),
    so an exact match is appropriate (unlike raw, possibly quant-prefixed class names).
    """
    for spec in _SPECS:
        if decoder_type in spec.decoder_types:
            return spec
    return None


def match_moe_block(module: nn.Module) -> ModelSpec | None:
    """Return the spec whose ``moe_block_names`` matches ``module``'s class name.

    Mirrors the legacy ``module_match_name_list`` semantics: case-insensitive substring
    match against ``type(module).__name__``.
    """
    cls_name = type(module).__name__.lower()
    for spec in _SPECS:
        if any(name.lower() in cls_name for name in spec.moe_block_names):
            return spec
    return None
