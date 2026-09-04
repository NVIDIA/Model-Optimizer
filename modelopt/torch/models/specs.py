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

"""Per-model descriptor classes.

``ModelSpec`` is the one global descriptor of a model: everything modelopt knows
about a model type lives on a single instance, resolved by ``config.model_type``.
It holds each concern as a separate, optional section object:

- **topic sections** hold architecture facts shared across subsystems (``MoESpec``:
  what a model's MoE blocks are);
- **subsystem sections** hold one subsystem's per-model data (``ExportSpec``).

A section is ``None`` when the model has nothing to say about it -- a dense model
carries ``moe_spec=None`` rather than an empty ``MoESpec``, so "not an MoE model" and
"an MoE model whose layout is not filled in yet" stay distinguishable.

Sections hold per-model data plus trivial accessors over that data; subsystem logic
never lives here. A model registers exactly one ``ModelSpec`` in its sibling
``specs.py``, filling only the sections it customizes.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ExportSpec",
    "MoESpec",
    "MoEVariant",
    "ModelSpec",
    "match_class_names",
]


def match_class_names(module, names: tuple[str, ...]) -> bool:
    """Return True if any of ``names`` equals a class name in ``module``'s MRO.

    Case-insensitive exact-name comparison against every class in
    ``type(module).__mro__``: dynamically generated quantized classes match through
    their base class, and exact names avoid substring false positives.
    """
    mro_names = {cls.__name__.lower() for cls in type(module).__mro__}
    return any(name.lower() in mro_names for name in names)


@dataclass(kw_only=True)
class MoEVariant:
    """One concrete MoE-block layout of a model.

    A model type usually has exactly one; it has several when the same checkpoint
    materializes with different module classes and projection names (e.g. Mixtral
    across transformers generations). ``block_names`` tells the variants apart.
    """

    block_names: tuple[str, ...] = ()
    """The matching key -- MoE block class names, matched against the module's MRO
    (case-insensitive exact names, not substrings; see ``match_class_names``)."""

    expert_linear_names: tuple[str, ...] | None = None
    """Expert linear projection names, e.g. ``("gate_proj", "down_proj", "up_proj")``.
    For layouts modelopt rewrites (e.g. quantized DBRX), these are the names on the
    rewritten module."""

    has_iterable_experts: bool = False
    """True when experts are per-expert iterable sub-modules that
    ``get_experts_list`` can group (Mixtral, Qwen MoE, ...); False for stacked or
    fused layouts (DBRX, GptOss) and for layouts not yet validated on the grouped
    export path."""

    gate_up_pair: tuple[str, str] | None = None
    """The (gate, up) pair among ``expert_linear_names`` that serving engines fuse
    into a single ``gate_up_proj``, e.g. ``("gate_proj", "up_proj")`` or
    ``("w1", "w3")``. ``None`` for non-gated experts (NemotronH) and already-fused
    layouts (GptOss, DBRX)."""


@dataclass(kw_only=True)
class MoESpec:
    """Topic section: MoE architecture facts — the model's MoE-block layout(s).

    This describes what a model's MoE blocks *are* — which class, what the expert
    projections are called — so any modelopt subsystem (export, quantization,
    speculative decoding, ...) can read it instead of keeping its own per-model MoE
    table.
    """

    moe_variants: tuple[MoEVariant, ...] = ()
    """The model's MoE-block layouts; more than one when the same checkpoint
    materializes differently (see ``MoEVariant``)."""

    @property
    def gate_up_pairs(self) -> tuple[tuple[str, str], ...]:
        """The (gate, up) projection-name pairs declared by this model's variants."""
        return tuple(v.gate_up_pair for v in self.moe_variants if v.gate_up_pair is not None)

    def match_moe_variant(self, module) -> MoEVariant | None:
        """Return the variant whose ``block_names`` matches ``module``, else None."""
        for variant in self.moe_variants:
            if match_class_names(module, variant.block_names):
                return variant
        return None

    def expert_linear_names_for(self, module) -> tuple[str, ...] | None:
        """Resolve ``module``'s expert linear names within this model.

        When every variant agrees on one naming, the module's class is irrelevant
        (a spec can provide naming without the block class being known); with
        several namings, the module's class picks the variant.
        """
        namings = {
            variant.expert_linear_names
            for variant in self.moe_variants
            if variant.expert_linear_names is not None
        }
        if len(namings) == 1:
            return next(iter(namings))
        variant = self.match_moe_variant(module)
        return variant.expert_linear_names if variant is not None else None


@dataclass(kw_only=True)
class ExportSpec:
    """Subsystem section: per-model data of the unified HF export path.

    Architecture facts (MoE block classes, expert naming) live in ``MoESpec``; this
    section holds data consumed by the export algorithms only.
    """

    pqs_fuse_rules: tuple[tuple[tuple[str, ...], str, str], ...] = ()
    """AWQ ``pre_quant_scale`` fusion rules, each a ``(module_class_substrings,
    fuse_into, fuse_from)`` triple: for a module whose class name contains one of the
    substrings, the pre_quant_scale on ``fuse_from`` is folded into ``fuse_into``
    (e.g. attention ``o_proj`` -> ``v_proj``, MLP ``down_proj`` -> ``up_proj``).
    A rule asserts mathematical equivalence for that model's modules, so it is
    declared per model rather than applied generically."""

    weight_plus_one_norm_names: tuple[str, ...] = ()
    """Class names of norm layers whose stored weight is ``w - 1`` (the effective
    scale is ``weight + 1``), e.g. Gemma's RMSNorm variants and LayerNorm1P.
    Matched against a norm module's MRO (case-insensitive exact names). Export
    must account for the +1 when folding scales into the norm weight (AWQ
    pre_quant_scale fusion). A structural fallback (``zero_centered_gamma``) stays
    in the engine."""


@dataclass(kw_only=True)
class ModelSpec:
    """The one global per-model descriptor, holding each section as an attribute.

    Resolved by HF model type (see ``registry.get_spec``); a model registers exactly
    one instance, filling only the sections it customizes and leaving the rest
    ``None``. Consumers must handle an absent section; ``registry.match_moe_block``
    and ``registry.list_all_possible`` already do for the lookups they cover.
    """

    model_type: str
    """The HF model type this spec describes (``config.model_type``, e.g.
    ``"qwen3_moe"``). Unique across the registry."""

    modeling_source: Literal["transformers", "remote_code"] = "transformers"
    """Where the model's modeling code lives: shipped inside ``transformers``, or
    carried by the checkpoint and loaded with ``trust_remote_code=True``.

    A fact about the model, not about any one subsystem: it decides whether the classes
    named in this spec can be imported at all, so ``model_type`` is a remote-code
    spelling rather than a transformers one, and loading the model needs the flag."""

    moe_spec: MoESpec | None = None
    """The model's MoE architecture facts, or ``None`` for a dense model."""

    export_spec: ExportSpec | None = None
    """Per-model data of the unified HF export path, or ``None`` when export needs
    nothing model-specific."""
