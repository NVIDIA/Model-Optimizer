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

"""Registry that resolves a model (or one of its sub-modules) to its ``ModelSpec``.

Families register their specs at import time (see ``families/``). Lookups return
``None`` when nothing matches, so callers can fall back to their default behavior.
"""

import torch.nn as nn

from .base import ModelSpec

__all__ = [
    "iter_pqs_fuse_rules",
    "match_by_architecture",
    "match_by_decoder_type",
    "match_mlp_block",
    "match_moe_block",
    "register",
]

_SPECS: list[ModelSpec] = []


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model-family spec and return it."""
    _SPECS.append(spec)
    return spec


def match_by_architecture(architecture: str) -> ModelSpec | None:
    """Return the spec whose ``architectures`` contains an exact ``architecture`` match."""
    for spec in _SPECS:
        if architecture in spec.architectures:
            return spec
    return None


def iter_pqs_fuse_rules():
    """Yield every ``(module_class_substrings, fuse_into, fuse_from)`` AWQ fusion rule.

    Aggregated across all registered specs (the consumer matches each model module
    against the substrings, so the order across families does not matter).
    """
    for spec in _SPECS:
        yield from spec.pqs_fuse_rules


def match_by_decoder_type(decoder_type: str) -> ModelSpec | None:
    """Return the spec whose ``decoder_types`` contains ``decoder_type`` (exact match)."""
    for spec in _SPECS:
        if decoder_type in spec.decoder_types:
            return spec
    return None


def match_moe_block(module: nn.Module) -> ModelSpec | None:
    """Return the spec matching ``module``'s class name against ``moe_block_names``.

    Case-insensitive substring match against ``type(module).__name__``.
    """
    cls_name = type(module).__name__.lower()
    for spec in _SPECS:
        if any(name.lower() in cls_name for name in spec.moe_block_names):
            return spec
    return None


def match_mlp_block(module: nn.Module) -> ModelSpec | None:
    """Return the spec whose ``mlp_block_names`` exactly equals ``module``'s class name."""
    cls_name = type(module).__name__
    for spec in _SPECS:
        if cls_name in spec.mlp_block_names:
            return spec
    return None
