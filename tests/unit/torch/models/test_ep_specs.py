# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The expert-parallel section of the model specs."""

import pytest

from modelopt.torch.models import get_spec, get_specs

EP_MODEL_TYPES = [s.model_type for s in get_specs() if s.ep_spec is not None]


# --------------------------------------------------------------------------- the registered specs


@pytest.mark.parametrize("model_type", EP_MODEL_TYPES)
def test_every_ep_plan_is_well_formed(model_type):
    """Globs map to style names; a malformed entry matches nothing and fails silently at load."""
    plan = get_spec(model_type).ep_spec.base_model_ep_plan
    if plan is None:
        return
    assert plan, f"{model_type} declares an empty plan, which is not the same as declaring none"
    for glob, style in plan.items():
        assert isinstance(glob, str) and glob, f"{model_type}: bad glob {glob!r}"
        assert isinstance(style, str) and style, f"{model_type}: bad style for {glob!r}"


@pytest.mark.parametrize("model_type", EP_MODEL_TYPES)
def test_a_spec_declares_a_plan_a_floor_a_prepare_or_a_refusal(model_type):
    """An EP section that says nothing would claim support the model does not have."""
    ep = get_spec(model_type).ep_spec
    assert ep.base_model_ep_plan or ep.requires_transformers or ep.prepare or ep.unsupported, (
        f"{model_type} has an EP section that declares nothing"
    )


def test_the_omni_vlm_reuses_the_text_models_plan():
    """Two spellings of one architecture must not drift into two different plans, nor share one
    dict, which a change through either spec would carry into the other."""
    omni = get_spec("nemotron_h_omni").ep_spec.base_model_ep_plan
    text = get_spec("nemotron_h").ep_spec.base_model_ep_plan
    assert omni == text
    assert omni is not text


def test_the_omni_vlm_puts_its_plan_on_the_nested_config():
    """Its experts live in the language sub-model, and the globs are written against that tree."""
    assert get_spec("nemotron_h_omni").ep_spec.plan_config_attr == ("text_config", "llm_config")
    assert get_spec("nemotron_h").ep_spec.plan_config_attr is None
