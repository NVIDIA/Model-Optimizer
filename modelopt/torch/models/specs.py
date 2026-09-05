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

"""Per-model descriptors and the registry indexing them by HF model type.

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
never lives here. A model registers exactly one ``ModelSpec`` in its own ``specs.py``
at import time; importing the package registers them all. Lookups return ``None`` (or
an empty list) when nothing matches, so callers can fail loudly or fall back per their
own policy.

The registry lives here rather than beside these classes because it is not a separate
concept: every lookup takes or returns a ``ModelSpec``, and it has to know how one is
composed. Matching is by model-type and class-name strings only, so this module needs
no torch import.
"""

import functools
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal, get_args, get_type_hints

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "ExportSpec",
    "MoESpec",
    "MoEVariant",
    "ModelSpec",
    "SpecSection",
    "get_spec",
    "get_specs",
    "hf_model_type",
    "list_all_possible",
    "match_class_names",
    "match_moe_block",
    "register",
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
class SpecSection:
    """Base class for the sections composing a ``ModelSpec``.

    Marks a class as a section so ``_spec_sections`` can find ``ModelSpec``'s section
    fields from its annotations. Without it the lookups would need a hand-maintained
    list of section names, which would silently fall out of date the first time a
    section was added and the list was not.
    """


@dataclass(kw_only=True)
class MoESpec(SpecSection):
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
class ExportSpec(SpecSection):
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

    Resolved by HF model type (see ``get_spec`` below); a model registers exactly one
    instance, filling only the sections it customizes and leaving the rest ``None``.
    Consumers must handle an absent section; ``match_moe_block`` and
    ``list_all_possible`` already do for the lookups they cover.
    """

    model_type: str
    """The HF model type this spec describes (``config.model_type``, e.g.
    ``"qwen3_moe"``). Unique across the registry."""

    min_transformers_version: str | None = None
    """Earliest ``transformers`` release whose definitions match this spec, or ``None``
    when the question does not apply (``modeling_source="remote_code"``).

    Clamped below at the repo's minimum supported transformers (``tf_min`` in
    ``noxfile.py``): a model that predates the floor records the floor, since nothing
    older is ever installed or tested. A model added after the floor records its own
    release, which is what lets the test suite tell an expected absence on an older
    transformers apart from a spec that no longer matches reality."""

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


_SPECS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Register a model spec and return it. One spec per model type."""
    if spec.model_type in _SPECS:
        raise ValueError(f"ModelSpec for model type {spec.model_type!r} already registered")
    _SPECS[spec.model_type] = spec
    return spec


def get_spec(model_type: str) -> ModelSpec | None:
    """Return the spec registered for ``model_type``, or ``None``."""
    return _SPECS.get(model_type)


def get_specs() -> list[ModelSpec]:
    """Return all registered specs, in registration order."""
    return list(_SPECS.values())


@functools.cache
def _spec_sections() -> tuple[tuple[str, type], ...]:
    """(field name, section class) for every section field on ``ModelSpec``.

    Read off ``ModelSpec``'s own annotations rather than restated here, so adding a
    section is a single edit. A section field is one whose declared type includes a
    ``SpecSection`` subclass.
    """
    hints = get_type_hints(ModelSpec)
    sections: list[tuple[str, type]] = []
    for field in fields(ModelSpec):
        annotation = hints.get(field.name)
        for candidate in get_args(annotation) or (annotation,):
            if isinstance(candidate, type) and issubclass(candidate, SpecSection):
                sections.append((field.name, candidate))
                break
    return tuple(sections)


@functools.cache
def _spec_attr_names() -> frozenset[str]:
    """All field and property names readable off a ``ModelSpec`` or one of its sections."""
    names = {f.name for f in fields(ModelSpec)}
    for _, section_type in _spec_sections():
        names.update(f.name for f in fields(section_type))
        for klass in section_type.__mro__:
            names.update(name for name, attr in vars(klass).items() if isinstance(attr, property))
    return frozenset(names)


def _read_spec_attr(spec: ModelSpec, attr: str):
    """Read ``attr`` off ``spec`` or whichever section declares it.

    Returns ``None`` when the declaring section is absent on this model, which is how
    a dense model contributes nothing to an MoE vocabulary. That is distinct from a
    present-but-empty section, which contributes an empty tuple; both end up adding
    no values.
    """
    if hasattr(spec, attr):
        return getattr(spec, attr)
    for section_name, _ in _spec_sections():
        section = getattr(spec, section_name)
        if section is not None and hasattr(section, attr):
            return getattr(section, attr)
    return None


def list_all_possible(attr: str) -> tuple:
    """List a spec attribute's values across all registered specs, deduplicated in order.

    E.g. ``list_all_possible("gate_up_pairs")``. Looks the name up on ``ModelSpec``
    and then on its sections, so callers name the field (``"pqs_fuse_rules"``) without
    naming the section that holds it. Models whose declaring section is ``None``
    contribute nothing.

    The result is a global vocabulary: consumers match it against any model's modules,
    so adding a value to one spec affects all models the consumer walks — prefer
    ``get_spec(model_type)`` / ``match_moe_block`` wherever the owning model is
    identifiable.

    ``attr`` must name a tuple-valued attribute. Scalars (``model_type``) would be
    iterated character by character, so they are rejected rather than silently
    producing nonsense.
    """
    if attr not in _spec_attr_names():
        raise ValueError(
            f"{attr!r} is not a ModelSpec attribute; available: {sorted(_spec_attr_names())}"
        )
    values: list = []
    for spec in get_specs():
        value = _read_spec_attr(spec, attr)
        if value is None:
            # This model does not declare the section that holds ``attr``.
            continue
        if not isinstance(value, tuple):
            raise ValueError(
                f"list_all_possible({attr!r}) expects a tuple-valued attribute, got "
                f"{type(value).__name__} on model type {spec.model_type!r}."
            )
        # Deduplicate by equality: items need not be hashable (MoEVariant is not).
        values.extend(item for item in value if item not in values)
    return tuple(values)


def hf_model_type(model) -> str | None:
    """Return the root HF model type (``model.config.model_type``), or ``None``.

    Accepts a model or a config object (duck-typed, no transformers import). This
    is the key for ``get_spec`` / ``match_moe_block``.
    """
    config = getattr(model, "config", model)
    model_type = getattr(config, "model_type", None)
    return model_type if isinstance(model_type, str) else None


def match_moe_block(module: "nn.Module", model_type: str | None = None) -> MoEVariant | None:
    """Return the MoE layout variant for ``module``, resolved by model type.

    ``model_type`` (the root ``model.config.model_type``) is a strict filter: only
    that model's own spec is consulted, and an unregistered model type resolves to
    ``None`` even if the module's class names coincide with another model's.
    ``model_type=None`` searches all specs. A composite model whose MoE lives under
    a sub-model type registers the root type too (see ``gemma4/specs.py``). Within the
    spec, variant ``block_names`` matched against the module's MRO picks the layout.
    """
    if model_type:
        return _match_in_spec(get_spec(model_type), module)
    for spec in get_specs():
        variant = _match_in_spec(spec, module)
        if variant is not None:
            return variant
    return None


def _match_in_spec(spec: ModelSpec | None, module: "nn.Module") -> MoEVariant | None:
    """Match ``module`` against one spec's MoE section; ``None`` if either is absent."""
    if spec is None or spec.moe_spec is None:
        return None
    return spec.moe_spec.match_moe_variant(module)
