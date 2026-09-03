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

# Model types transformers does ship, so their specs are checked against it. Listed
# literally rather than derived, so registering a spec forces a decision here -- the
# point of the classification test below. Membership does not promise the model exists
# in *every* supported transformers version: the CI matrix spans tf_min to tf_latest,
# and a model missing from the installed one skips rather than fails.
IN_TRANSFORMERS = {
    "dbrx",
    "gemma4",
    "gemma4_text",
    "gpt_oss",
    "mixtral",
    "nemotron_h",
    "qwen2_moe",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_next",
}

# model_type -> the transformers package directory holding its modeling module, when it
# differs from the model type itself (a text-only checkpoint of a VLM reuses the VLM's).
MODULE_OVERRIDES = {
    "gemma4_text": "gemma4",
}

# A floor so that a wholesale breakage -- wrong module paths, a transformers layout
# change -- fails instead of turning every case into a silent skip. Individual models
# may legitimately be missing from a given transformers version; all of them cannot.
# Lower this only with a note explaining which models stopped resolving and why.
MIN_VALIDATED_BLOCKS = 5


def _moe_variants():
    """(model_type, variant) for every registered MoE variant."""
    return [
        (spec.model_type, variant)
        for spec in get_specs()
        if spec.moe_spec is not None
        for variant in spec.moe_spec.moe_variants
    ]


def _modeling_module(model_type: str):
    """Import ``transformers.models.<pkg>.modeling_<pkg>``, or None if absent."""
    pkg = MODULE_OVERRIDES.get(model_type, model_type)
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


def test_every_moe_model_type_is_classified():
    """Every registered MoE model type is classified exactly once.

    The coverage guard, and deliberately pure bookkeeping -- it consults the registry
    and the two tables, never the installed transformers, so it means the same thing on
    every version in the CI matrix. Registering a spec without classifying it fails
    here rather than quietly testing nothing.
    """
    registered = {mt for mt, _ in VARIANTS}
    classified = IN_TRANSFORMERS | set(UNCHECKABLE)
    assert not (registered - classified), (
        f"unclassified MoE model types: {sorted(registered - classified)}. Add each to "
        f"IN_TRANSFORMERS, or to UNCHECKABLE with the reason it cannot be introspected."
    )
    assert not (classified - registered), (
        f"classified model types that are no longer registered: {sorted(classified - registered)}"
    )
    assert not (IN_TRANSFORMERS & set(UNCHECKABLE)), (
        f"model types in both tables: {sorted(IN_TRANSFORMERS & set(UNCHECKABLE))}"
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


def test_enough_blocks_actually_validated():
    """Guard against every case above degrading into a skip."""
    validated = []
    for model_type, variant in VARIANTS:
        if model_type in UNCHECKABLE:
            continue
        module = _modeling_module(model_type)
        if module is not None and _find_block_class(module, variant.block_names) is not None:
            validated.append((model_type, variant.block_names))
    assert len(validated) >= MIN_VALIDATED_BLOCKS, (
        f"only {len(validated)} MoE block classes resolved against transformers "
        f"{importlib.import_module('transformers').__version__}: {validated}. Expected at "
        f"least {MIN_VALIDATED_BLOCKS} -- the lookup itself is probably broken."
    )
