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

"""Per-model-family export descriptor.

A ``ModelSpec`` declares how one model family differs from the generic export path,
so the export code can read these values instead of branching on model names. Each
spec holds per-model data only, not export logic.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hooks import ModelHooks

__all__ = ["ModelSpec"]


@dataclass
class ModelSpec:
    """Per-model-family export data.

    A spec is resolved from a model (or one of its sub-modules) via any of its matching
    keys:
        architectures: exact HuggingFace architecture class names (e.g. "Qwen3MoeForCausalLM").
        decoder_types: ModelOpt ``decoder_type`` strings (e.g. "bloom", "phi3"). Exact match.
        moe_block_names: MoE block class-name substrings, matched case-insensitively
            (e.g. "Qwen3MoeSparseMoeBlock").
        mlp_block_names: MLP module class names, matched by exact equality
            (e.g. "ArcticMLP", "Phi3SmallMLP").

    Per-model fields:
        expert_linear_names: expert linear projection names for this family's MoE block,
            e.g. ("gate_proj", "down_proj", "up_proj"). ``None`` if not applicable.
        has_iterable_experts: True when experts are stored as per-expert iterable
            sub-modules (Mixtral, Qwen MoE, NemotronH, Gemma4) and can be grouped by
            ``get_experts_list``. False for stacked or fused layouts (DBRX, GptOss).
        forced_activation: activation that overrides MLP activation detection
            (e.g. Bloom/GLM → "gelu", Phi3 → "swiglu"). ``None`` for no override.
        force_share_embedding_table: True for families that share the embedding/output
            table (Gemma/Gemma2/Gemma3); still gated by a weight-equality check.
        mlp_keyword_roles: overrides mapping a child-module name to its MLP role, e.g.
            {"up_proj": "fc"} (MPT/Phi3Small) or {"w1": "fc", "w2": "proj", "w3": "gate"}
            (Arctic/InternLM2). Each keyword is removed from the default role sets and
            added to its target role. ``None`` for no override.
        pqs_fuse_rules: AWQ pre_quant_scale fusion rules, each a
            ``(module_class_substrings, fuse_into, fuse_from)`` triple: for a module whose
            class name contains one of the substrings, the pre_quant_scale on ``fuse_from``
            is folded into ``fuse_into`` (e.g. attention o_proj → v_proj, MLP down_proj →
            up_proj).
        hooks: optional :class:`ModelHooks` for behavioral seams that no value can express
            (custom config-slot placement, module-tree unwrap, MoE building). ``None`` uses
            the engine defaults.
    """

    name: str

    # Matching keys.
    architectures: tuple[str, ...] = ()
    decoder_types: tuple[str, ...] = ()
    moe_block_names: tuple[str, ...] = ()
    mlp_block_names: tuple[str, ...] = ()

    # MoE expert layout.
    expert_linear_names: tuple[str, ...] | None = None
    has_iterable_experts: bool = False

    # MLP and activation.
    forced_activation: str | None = None
    mlp_keyword_roles: dict[str, str] | None = None

    # Embedding.
    force_share_embedding_table: bool = False

    # AWQ pre_quant_scale fusion.
    pqs_fuse_rules: tuple[tuple[tuple[str, ...], str, str], ...] = ()

    # Behavioral overrides for structural export seams (None = use the engine default).
    hooks: "ModelHooks | None" = None

    # Free-form extension slot for future per-model fields.
    _extra: dict = field(default_factory=dict, repr=False)
