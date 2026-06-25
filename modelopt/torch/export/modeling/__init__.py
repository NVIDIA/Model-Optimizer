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

"""Model-family export descriptors (the "modeling library").

Holds per-model export *data* extracted from the generic export code, organized by
model family. The generic export path reads ``ModelSpec`` fields via the registry
lookups below; an unmatched lookup returns ``None`` so callers fall back to legacy
behavior. See ``MODEL_SPECIFIC_REFACTOR.md`` for scope and migration priority.
"""

from . import families
from .base import ModelSpec
from .registry import (
    iter_pqs_fuse_rules,
    match_by_architecture,
    match_by_decoder_type,
    match_mlp_block,
    match_moe_block,
    register,
)

__all__ = [
    "ModelSpec",
    "iter_pqs_fuse_rules",
    "match_by_architecture",
    "match_by_decoder_type",
    "match_mlp_block",
    "match_moe_block",
    "register",
]
