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

"""Registry indexing the per-model ``ModelSpec`` by HF model type.

Model modules register their spec at import time; one spec per model type. Lookups
return ``None`` (or an empty list) when nothing matches, so callers can fail loudly
or fall back per their own policy. Matching is by model-type and class-name strings
only, so this package needs no torch import.
"""

import functools
from dataclasses import fields
from typing import TYPE_CHECKING

from .specs import ExportSpec, ModelSpec, MoESpec, MoEVariant

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "get_spec",
    "get_specs",
    "hf_model_type",
    "list_all_possible",
    "match_moe_block",
    "register",
]

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


# The section attributes of ``ModelSpec``, searched in order when resolving an
# attribute name that a section rather than ``ModelSpec`` itself declares.
_SECTION_FIELDS = ("moe_spec", "export_spec")
_SECTION_TYPES = (MoESpec, ExportSpec)


@functools.cache
def _spec_attr_names() -> frozenset[str]:
    """All field and property names readable off a ``ModelSpec`` or one of its sections."""
    names = {f.name for f in fields(ModelSpec)}
    for section_type in _SECTION_TYPES:
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
    for section_name in _SECTION_FIELDS:
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
