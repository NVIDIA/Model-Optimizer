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

"""The per-model descriptor that holds model-specific export *data* (not algorithms).

A ``ModelSpec`` declares how a model family differs from the generic export path. The
generic export code reads these fields instead of branching on ``decoder_type`` /
class-name strings. Fields are added one migration step at a time (see
``MODEL_SPECIFIC_REFACTOR.md``); keep this to per-model *values*, never algorithms.
"""

from dataclasses import dataclass, field

__all__ = ["ModelSpec"]


@dataclass
class ModelSpec:
    """Declarative, per-model-family export data.

    Matching keys (a spec applies to a model when any key matches):
        architectures: exact HF architecture class names (e.g. "Qwen3MoeForCausalLM").
        decoder_types: ModelOpt ``decoder_type`` strings (the normalized family name from
            ``MODEL_NAME_TO_TYPE``, e.g. "bloom", "phi3"). Exact match.
        moe_block_names: MoE block class-name substrings, matched case-insensitively
            the same way the legacy ``module_match_name_list`` did (e.g.
            "Qwen3MoeSparseMoeBlock"). Used to resolve a spec from a sub-module.

    Per-model data (grows as categories are migrated):
        expert_linear_names: the expert linear projection names for this family's MoE
            block, e.g. ("gate_proj", "down_proj", "up_proj"). ``None`` means this spec
            carries no MoE-naming override.
        has_iterable_experts: True when this family stores experts as per-expert
            iterable sub-modules (Mixtral, Qwen MoE, NemotronH, Gemma4), so the grouped
            export path (``get_experts_list``) can index them. False/unset for stacked
            or fused layouts (DBRX, GptOss, ...), which are handled by other paths.
        forced_activation: activation that overrides MLP activation detection for this
            family (e.g. Bloom/GLM → "gelu", Phi3 → "swiglu"). ``None`` = no override.
    """

    name: str
    architectures: tuple[str, ...] = ()
    decoder_types: tuple[str, ...] = ()
    moe_block_names: tuple[str, ...] = ()

    # --- P1: MoE expert naming ---
    expert_linear_names: tuple[str, ...] | None = None

    # --- P2: grouped (iterable) expert export support ---
    has_iterable_experts: bool = False

    # --- P3: non-MoE per-model flags ---
    forced_activation: str | None = None

    # Reserved for later migration steps; added when those land.
    _extra: dict = field(default_factory=dict, repr=False)
