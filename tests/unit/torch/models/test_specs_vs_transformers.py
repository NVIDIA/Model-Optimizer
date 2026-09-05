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

"""Validate registered specs against the installed transformers definitions.

``test_model_specs.py`` pins spec values against a hand-written table. That catches an
accidental edit, but not a spec that is simply *wrong*: the table and the spec can be
wrong in the same way. ``gpt_oss`` shipped ``block_names=("GptOssMoE",)``, a class that
does not exist in transformers, and a mirrored table agreed with it.

This file reads the other side of the contract -- what transformers actually defines --
so a block class naming nothing real is caught at the source.

Nothing here names a model. Which specs to check, which cannot be checked, and which
transformers releases each applies to all come from the registry:

- ``modeling_source`` says whether the classes ship in transformers at all;
- ``min_transformers_version`` says from which release a spec's definitions apply, so an
  absence on an older transformers is an expected skip rather than a failure.

That second field is what lets these checks be assertions. Without it every check
degrades to "skip when not found", and a broken lookup would leave the suite green
having verified nothing.
"""

import importlib

import pytest

pytest.importorskip("transformers")

from packaging.version import Version

from modelopt.torch.models import get_spec, get_specs


def _installed_version() -> Version:
    return Version(importlib.import_module("transformers").__version__)


def _package_name(model_type: str) -> str:
    """The transformers package directory for ``model_type``.

    A model type is not always its own directory: sub-model types live with their parent
    (``gemma4_text`` in ``gemma4``) and a few are spelled differently (``kosmos-2`` in
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
    """The first of ``block_names`` naming a class in ``module``, matched case-insensitively.

    Mirrors ``specs.match_class_names``, which lowercases both sides, so a spec that
    resolves at runtime resolves here too. The spellings do drift: ``nemotron_h``
    declares ``NemotronHMOE`` while transformers defines ``NemotronHMoE``.
    """
    by_lower = {name.lower(): obj for name, obj in vars(module).items() if isinstance(obj, type)}
    for name in block_names:
        cls = by_lower.get(name.lower())
        if cls is not None:
            return cls
    return None


def _moe_variants():
    """(model_type, variant) for every registered MoE variant."""
    return [
        (spec.model_type, variant)
        for spec in get_specs()
        if spec.moe_spec is not None
        for variant in spec.moe_spec.moe_variants
    ]


VARIANTS = _moe_variants()
MOE_MODEL_TYPES = sorted({mt for mt, _ in VARIANTS})
REMOTE_CODE_MODEL_TYPES = sorted(
    s.model_type for s in get_specs() if s.modeling_source == "remote_code"
)
VERSIONED_MOE_MODEL_TYPES = sorted(
    mt for mt in MOE_MODEL_TYPES if get_spec(mt).min_transformers_version is not None
)


def test_version_claim_matches_modeling_source():
    """A spec names a transformers release exactly when its code ships in transformers.

    Pure bookkeeping over the registry, so it means the same thing on every version in
    the CI matrix. It is what makes the checks below exhaustive: a new spec cannot land
    without either a release to check against or a declared reason there is none.
    """
    for spec in get_specs():
        has_version = spec.min_transformers_version is not None
        in_transformers = spec.modeling_source == "transformers"
        assert has_version == in_transformers, (
            f"{spec.model_type!r}: modeling_source={spec.modeling_source!r} but "
            f"min_transformers_version={spec.min_transformers_version!r}. A transformers "
            f"model needs the release its definitions come from; a remote_code model has "
            f"no release to name."
        )


@pytest.mark.parametrize("model_type", REMOTE_CODE_MODEL_TYPES)
def test_remote_code_models_are_really_absent(model_type):
    """``modeling_source="remote_code"`` must still hold for the installed transformers.

    Models do graduate from remote code into transformers, and a field left stale would
    silently suppress every other check for that model.
    """
    module = _modeling_module(model_type)
    assert module is None, (
        f"{model_type!r} declares modeling_source='remote_code', but transformers now "
        f"provides {module.__name__}. Set modeling_source='transformers' and give it a "
        f"min_transformers_version so its block names get checked."
    )


@pytest.mark.parametrize("model_type", VERSIONED_MOE_MODEL_TYPES)
def test_moe_block_resolves_from_its_declared_version(model_type):
    """From ``min_transformers_version`` on, a spec's MoE block must name a real class.

    ``block_names`` is the matching key: a name existing nowhere silently matches
    nothing, which is how the ``GptOssMoE`` entry stayed invisible.

    One resolving variant is enough. A model can declare a variant no transformers
    release defines -- mixtral carries ``MixtralMoeSparseMoeBlock`` with MCore
    ``linear_fc1``/``linear_fc2`` naming, migrated from a legacy branch -- so requiring
    every variant to resolve would fail on those permanently.
    """
    spec = get_spec(model_type)
    installed = _installed_version()
    required = Version(spec.min_transformers_version)
    if installed < required:
        pytest.skip(f"needs transformers >= {required}, installed {installed}")

    module = _modeling_module(model_type)
    assert module is not None, (
        f"{model_type!r} declares min_transformers_version="
        f"{spec.min_transformers_version!r} but transformers {installed} has no modeling "
        f"module for it. Either that version is wrong or the model moved."
    )
    variants = [v for mt, v in VARIANTS if mt == model_type]
    resolved = [v.block_names for v in variants if _find_block_class(module, v.block_names)]
    assert resolved, (
        f"{model_type!r}: none of {[v.block_names for v in variants]} name a class in "
        f"{module.__name__} on transformers {installed}. The block class was most likely "
        f"renamed upstream; update block_names, and min_transformers_version if the "
        f"rename landed in a later release."
    )
