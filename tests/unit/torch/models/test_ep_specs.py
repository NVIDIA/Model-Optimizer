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

"""The expert-parallel section of the model specs, and the applier that reads it."""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from modelopt.torch.models import EPSpec, ModelSpec, get_spec, get_specs
from modelopt.torch.utils.moe import apply_distributed_support

EP_MODEL_TYPES = [s.model_type for s in get_specs() if s.ep_spec is not None]


def cfg(**kw):
    return SimpleNamespace(**kw)


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
    """Two spellings of one architecture must not drift into two different plans."""
    assert (
        get_spec("nemotron_h_omni").ep_spec.base_model_ep_plan
        == get_spec("nemotron_h").ep_spec.base_model_ep_plan
    )


def test_the_omni_vlm_puts_its_plan_on_the_nested_config():
    """Its experts live in the language sub-model, and the globs are written against that tree."""
    assert get_spec("nemotron_h_omni").ep_spec.plan_config_attr == "llm_config"
    assert get_spec("nemotron_h").ep_spec.plan_config_attr is None


# ----------------------------------------------------------------------------------- the applier


def test_the_models_own_plan_always_wins():
    """Stock transformers ships a plan for some models; overwriting it would substitute our idea
    of the layout for the model's own."""
    config = cfg(base_model_ep_plan={"layers.*.mlp.experts": "stock_style"})
    apply_distributed_support("deepseek_v3", config)
    assert config.base_model_ep_plan == {"layers.*.mlp.experts": "stock_style"}


def test_the_attached_plan_is_a_copy():
    """Two models prepared in one process must not share one mutable dict."""
    a, b = cfg(), cfg()
    apply_distributed_support("deepseek_v3", a)
    apply_distributed_support("deepseek_v3", b)
    a.base_model_ep_plan["injected"] = "oops"
    assert "injected" not in b.base_model_ep_plan
    assert "injected" not in get_spec("deepseek_v3").ep_spec.base_model_ep_plan


def test_a_vlm_plan_lands_on_the_nested_config_only():
    """On the root config the globs match no parameter, and the failure is silence, not an error."""
    config = cfg(llm_config=cfg())
    apply_distributed_support("nemotron_h_omni", config)
    assert config.llm_config.base_model_ep_plan
    assert not hasattr(config, "base_model_ep_plan")


def test_a_missing_nested_config_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="llm_config"):
        apply_distributed_support("nemotron_h_omni", cfg())


@pytest.mark.parametrize("model_type", ["gpt_oss", "llama4", "llama4_text"])
def test_an_unsupported_model_is_refused_with_its_config_untouched(model_type):
    """These ship plans of their own, so only their spec's refusal keeps them off the
    expert-parallel path."""
    config = cfg()

    with pytest.raises(NotImplementedError, match="cannot be quantized expert-parallel yet"):
        apply_distributed_support(model_type, config)
    assert vars(config) == {}


def test_a_refusal_comes_before_the_rest_of_the_section(monkeypatch):
    """Nothing is attached or patched for a run that is about to be refused."""
    prepared = []
    spec = ModelSpec(
        model_type="_probe",
        ep_spec=EPSpec(
            base_model_ep_plan={"layers.*.experts": "grouped_gemm"},
            prepare=lambda hf_config, **_: prepared.append(hf_config),
            unsupported="It is not ready",
        ),
    )
    monkeypatch.setattr(
        "modelopt.torch.models.get_spec", lambda mt: spec if mt == "_probe" else None
    )
    config = cfg()

    with pytest.raises(NotImplementedError, match="It is not ready"):
        apply_distributed_support("_probe", config)
    assert not prepared
    assert vars(config) == {}


def test_prepare_runs_after_the_plan_is_attached_and_its_config_is_returned(monkeypatch):
    """A prepare hook may inspect or adjust the plan, so it must not run before there is one."""
    seen = {}

    def prepare(hf_config, **kwargs):
        seen.update(kwargs, config=hf_config, plan_present=hasattr(hf_config, "base_model_ep_plan"))
        return "replaced"

    spec = ModelSpec(
        model_type="_probe",
        ep_spec=EPSpec(base_model_ep_plan={"layers.*.experts": "grouped_gemm"}, prepare=prepare),
    )
    monkeypatch.setattr(
        "modelopt.torch.models.get_spec", lambda mt: spec if mt == "_probe" else None
    )
    config = cfg()

    returned = apply_distributed_support("_probe", config, ckpt_path="/c", trust_remote_code=True)

    assert returned == "replaced"
    assert seen == {
        "config": config,
        "ckpt_path": "/c",
        "trust_remote_code": True,
        "plan_present": True,
    }
