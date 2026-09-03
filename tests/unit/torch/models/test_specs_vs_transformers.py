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

"""Validate registered MoE specs against the installed transformers definitions.

``test_model_specs.py`` pins spec values against a hand-written table. That catches an
accidental edit, but not a spec that is simply *wrong*: the table and the spec can be
wrong in the same way. ``gpt_oss`` shipped ``block_names=("GptOssMoE",)``, a class that
does not exist in transformers, and a mirrored table agreed with it.

These tests read the other side of the contract -- what transformers actually defines --
so a renamed block class or projection is caught at the source.

Structure follows the transformers repo's own convention: the assertions are shared and
parametrized over the registry, and a model is excluded only by an explicit entry in
``UNCHECKABLE`` with a reason (their ``utils/check_repo.py`` ignore-lists). A model that
is neither checkable nor listed fails ``test_every_moe_model_type_is_classified``, so
coverage cannot erode silently as models are added.
"""

import functools
import importlib
import inspect

import pytest

pytest.importorskip("transformers")

import torch.nn as nn

from modelopt.torch.models import get_specs

# Model types whose definition cannot be introspected from an installed transformers,
# with the reason. Mirrors transformers' own IGNORE_NON_TESTED: an exclusion is an
# explicit, reviewable line rather than a silent gap.
UNCHECKABLE = {
    "arctic": "trust_remote_code model; not shipped in transformers",
    "deepseek": "trust_remote_code model; not shipped in transformers",
}

# Model types that have been in transformers long enough to exist in every version this
# repo supports, so failing to resolve one means the lookup itself broke rather than the
# installed transformers predating the model. Everything else may legitimately be
# missing on the tf_min end of the CI matrix and only skips.
#
# Unlike a bare count, a name here states something a reader can check: "this model has
# been upstream for years." Drop an entry only when transformers actually removes it.
ALWAYS_RESOLVABLE = {
    "dbrx",
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
}


def _moe_variants():
    """(model_type, variant) for every registered MoE variant."""
    return [
        (spec.model_type, variant)
        for spec in get_specs()
        if spec.moe_spec is not None
        for variant in spec.moe_spec.moe_variants
    ]


@functools.cache
def _package_name(model_type: str) -> str:
    """The transformers package directory for ``model_type``.

    A model type is not always its own directory: sub-model types live with their parent
    (``gemma3_text`` in ``gemma3``) and a few are spelled differently (``kosmos-2`` in
    ``kosmos2``). transformers resolves this with ``model_type_to_module_name``, backed
    by its SPECIAL_MODEL_TYPE_TO_MODULE_NAME table -- ``CONFIG_MAPPING[model_type]``
    would fail to import otherwise. Deferring to it keeps that knowledge upstream
    instead of duplicating a table here that goes stale as sub-model types are added.

    Falls back to the identity spelling if the helper ever moves, which costs at most
    the coverage of a sub-model type whose parent package is checked anyway.
    """
    try:
        from transformers.models.auto.configuration_auto import model_type_to_module_name
    except ImportError:
        return model_type.replace("-", "_")
    return model_type_to_module_name(model_type)


def _modeling_module(model_type: str):
    """Import ``transformers.models.<pkg>.modeling_<pkg>``, or None if absent."""
    pkg = _package_name(model_type)
    try:
        return importlib.import_module(f"transformers.models.{pkg}.modeling_{pkg}")
    except ImportError:
        return None


def _find_block_class(module, block_names: tuple[str, ...]):
    """The first of ``block_names`` that names a real class in ``module``."""
    for name in block_names:
        cls = getattr(module, name, None)
        if isinstance(cls, type):
            return cls
    return None


def _ids(items):
    return [f"{mt}-{'/'.join(v.block_names)}" for mt, v in items]


VARIANTS = _moe_variants()


def test_excused_model_types_are_still_registered():
    """An UNCHECKABLE entry cannot outlive the spec it excuses."""
    registered = {mt for mt, _ in VARIANTS}
    stale = set(UNCHECKABLE) - registered
    assert not stale, (
        f"UNCHECKABLE names model types with no registered MoE spec: {sorted(stale)}. "
        f"Drop the entry."
    )


def test_always_resolvable_models_are_registered():
    """ALWAYS_RESOLVABLE cannot drift away from the registry."""
    registered = {mt for mt, _ in VARIANTS}
    assert registered >= ALWAYS_RESOLVABLE, (
        f"ALWAYS_RESOLVABLE names unregistered model types: "
        f"{sorted(ALWAYS_RESOLVABLE - registered)}"
    )


@pytest.mark.parametrize("model_type", sorted(UNCHECKABLE))
def test_excuses_are_still_true(model_type):
    """An excused model must really be absent from transformers.

    Keeps a stale excuse from suppressing a check forever: once transformers ships a
    model we skip today, this fires and the entry should be dropped rather than
    silently leaving the spec unvalidated.
    """
    module = _modeling_module(model_type)
    assert module is None, (
        f"{model_type!r} is excused as {UNCHECKABLE[model_type]!r}, but transformers now "
        f"provides {module.__name__}. Remove it from UNCHECKABLE so its spec is checked."
    )


@pytest.mark.parametrize(("model_type", "variant"), VARIANTS, ids=_ids(VARIANTS))
def test_block_names_name_a_real_transformers_class(model_type, variant):
    """``block_names`` must match a class transformers actually defines.

    ``block_names`` is the matching key: a name that exists nowhere silently matches
    nothing, which is how the ``GptOssMoE`` entry stayed invisible.
    """
    if model_type in UNCHECKABLE:
        pytest.skip(UNCHECKABLE[model_type])
    module = _modeling_module(model_type)
    if module is None:
        pytest.skip(f"no modeling module for {model_type!r} in this transformers")

    block_cls = _find_block_class(module, variant.block_names)
    if block_cls is None:
        pytest.skip(
            f"none of {variant.block_names} exist in {module.__name__}; this transformers "
            f"version may predate or postdate the layout the spec describes"
        )
    assert block_cls.__name__ in variant.block_names


@pytest.mark.parametrize(("model_type", "variant"), VARIANTS, ids=_ids(VARIANTS))
def test_expert_linear_names_exist_in_transformers(model_type, variant):
    """Some module class in the model's file must define every declared projection.

    Checked against the class sources rather than a constructed model: building a real
    MoE block needs a per-model config, and the failure this guards against -- a
    projection renamed upstream, so the spec's names no longer resolve -- shows up in
    the source just as well. The check is deliberately module-scoped: it asserts the
    names exist together on *a* module class, not which one holds them.
    """
    if variant.expert_linear_names is None:
        pytest.skip("variant declares no expert linear names")
    if model_type in UNCHECKABLE:
        pytest.skip(UNCHECKABLE[model_type])
    module = _modeling_module(model_type)
    if module is None:
        pytest.skip(f"no modeling module for {model_type!r} in this transformers")
    if _find_block_class(module, variant.block_names) is None:
        pytest.skip(f"none of {variant.block_names} exist in {module.__name__}")

    wanted = variant.expert_linear_names
    for obj in vars(module).values():
        if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if all(f"self.{name}" in src for name in wanted):
            return
    pytest.fail(
        f"{model_type}: no nn.Module in {module.__name__} defines all of {wanted}. "
        f"The expert projections were most likely renamed upstream; update the spec."
    )


def test_long_standing_models_still_resolve():
    """Guard against every case above degrading into a skip.

    The checks skip when a model is absent from the installed transformers, correct for
    the tf_min end of the matrix but equally able to hide a broken lookup. These models
    are old enough that not finding one means the lookup broke.

    One resolving variant per model is enough. A model can declare a variant that no
    transformers version defines -- mixtral carries ``MixtralMoeSparseMoeBlock`` with
    MCore ``linear_fc1``/``linear_fc2`` naming, migrated from a legacy branch -- and
    requiring every variant to resolve would fail on those forever.
    """
    unresolved = []
    for model_type in sorted(ALWAYS_RESOLVABLE):
        module = _modeling_module(model_type)
        variants = [v for mt, v in VARIANTS if mt == model_type]
        if module is None or not any(
            _find_block_class(module, v.block_names) is not None for v in variants
        ):
            unresolved.append((model_type, [v.block_names for v in variants]))
    assert not unresolved, (
        f"{unresolved} did not resolve against transformers "
        f"{importlib.import_module('transformers').__version__}, but they ship in every "
        f"supported version -- the module lookup or the spec's block_names is wrong."
    )
