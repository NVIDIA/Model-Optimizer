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
from packaging.version import Version

from modelopt.torch import models
from modelopt.torch.models import EPSpec, ModelSpec, get_spec, get_specs
from modelopt.torch.utils import moe
from modelopt.torch.utils.moe import (
    _EP_MIN_TRANSFORMERS,
    _ep_config,
    apply_distributed_support,
    check_distributed_support,
)

try:
    import transformers
except ImportError:  # the optional [hf] extra: only the checks of its release need it
    transformers = None

EP_MODEL_TYPES = [s.model_type for s in get_specs() if s.ep_spec is not None]


needs_transformers = pytest.mark.skipif(transformers is None, reason="needs transformers")
# Applying a spec enforces the transformers floor the expert-parallel load needs.
needs_ep_transformers = pytest.mark.skipif(
    transformers is None or Version(transformers.__version__) < Version(_EP_MIN_TRANSFORMERS),
    reason=f"expert parallelism needs transformers >= {_EP_MIN_TRANSFORMERS}",
)


def cfg(**kw):
    return SimpleNamespace(**kw)


# --------------------------------------------------------------------------- the registered specs


# The styles the expert-parallel load implements for a plan's entries.
_KNOWN_STYLES = frozenset(moe._EXPERTS_STYLES) | {"grouped_gemm"}


@pytest.mark.parametrize("model_type", EP_MODEL_TYPES)
def test_every_ep_plan_is_well_formed(model_type):
    """Every entry uses a style the expert-parallel load implements, and the plan names a
    routed-experts container it can count and shard: a malformed plan fails silently at load."""
    ep = get_spec(model_type).ep_spec
    for plan in (ep.base_model_ep_plan, ep.prepared_ep_plan):
        if plan is None:
            continue
        assert plan, f"{model_type} declares an empty plan, which is not the same as declaring none"
        for glob, style in plan.items():
            assert isinstance(glob, str) and glob, f"{model_type}: bad glob {glob!r}"
            assert style in _KNOWN_STYLES, (
                f"{model_type}: {glob!r} uses style {style!r}, which the expert-parallel load does "
                f"not implement (known: {sorted(_KNOWN_STYLES)})"
            )
        assert moe._routed_expert_keys(plan), (
            f"{model_type}'s plan names no routed-experts container, so the load can neither count "
            "nor shard its experts"
        )


@pytest.mark.parametrize("model_type", EP_MODEL_TYPES)
def test_a_spec_declares_a_plan_a_floor_a_prepare_or_a_refusal(model_type):
    """An EP section that says nothing would claim support the model does not have."""
    ep = get_spec(model_type).ep_spec
    assert ep.base_model_ep_plan or ep.requires_transformers or ep.prepare or ep.unsupported, (
        f"{model_type} has an EP section that declares nothing"
    )


@pytest.mark.parametrize("model_type", EP_MODEL_TYPES)
def test_a_plan_goes_where_the_load_reads_one(model_type):
    """A spelling the load does not read is read as no plan at all, and the MoE planned as dense."""
    attrs = get_spec(model_type).ep_spec.plan_config_attr or ()
    attrs = (attrs,) if isinstance(attrs, str) else attrs
    assert not set(attrs) - set(moe._SUB_CONFIG_ATTRS), f"{model_type}: {attrs}"


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


# ----------------------------------------------------------------------------------- the applier


@needs_ep_transformers
@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        ("deepseek_v3", lambda own: cfg(base_model_ep_plan=own)),
        (
            "nemotron_h_omni",
            lambda own: cfg(text_config=cfg(), llm_config=cfg(base_model_ep_plan=own)),
        ),
        ("nemotron_h_omni", lambda own: cfg(base_model_ep_plan=own, text_config=cfg())),
    ],
    ids=["on-the-root", "on-another-nested-config", "on-a-vlms-root"],
)
def test_the_models_own_plan_always_wins(model_type, config):
    """Stock transformers ships a plan for some models; overwriting it would substitute our idea
    of the layout for the model's own. It wins wherever the load reads it, not only on the config
    the spec would take: ours on text_config would shadow the model's on llm_config, which the
    load reads after it, or lie unused beside a plan on the root."""
    own = {"layers.*.mlp.experts": "stock_style"}
    config = config(own)
    apply_distributed_support(model_type, config)
    assert _ep_config(config).base_model_ep_plan == own
    assert not hasattr(getattr(config, "text_config", cfg()), "base_model_ep_plan")


@needs_transformers
def test_a_model_without_a_spec_still_needs_the_loads_transformers(monkeypatch):
    """A plan that ships with transformers needs no spec, but its load needs the release every
    expert-parallel load does: auto picks no EP degree below it, rather than reach the load."""
    monkeypatch.setattr(transformers, "__version__", "5.6.2")
    with pytest.raises(RuntimeError, match=r"requires transformers >= 5\.7.*without expert"):
        check_distributed_support("no_such_model_type")
    monkeypatch.setattr(transformers, "__version__", _EP_MIN_TRANSFORMERS)
    check_distributed_support("no_such_model_type")


@needs_ep_transformers
def test_the_attached_plan_is_a_copy():
    """Two models prepared in one process must not share one mutable dict."""
    a, b = cfg(), cfg()
    apply_distributed_support("deepseek_v3", a)
    apply_distributed_support("deepseek_v3", b)
    a.base_model_ep_plan["injected"] = "oops"
    assert "injected" not in b.base_model_ep_plan
    assert "injected" not in get_spec("deepseek_v3").ep_spec.base_model_ep_plan


@needs_ep_transformers
@pytest.mark.parametrize(
    "attrs", [("llm_config",), ("text_config",), ("text_config", "llm_config")]
)
def test_a_vlm_plan_lands_on_the_nested_config_only(attrs):
    """On the root config the globs match no parameter, and the failure is silence, not an error.
    The Omni checkpoint's own code names that config llm_config, and transformers' port
    text_config: a multi-rank run of either applies the spec, whether it wants experts sharded or
    not. A config with both takes it on text_config, which the load reads the expert count from
    first."""
    config = cfg(**{attr: cfg() for attr in attrs})
    apply_distributed_support("nemotron_h_omni", config)
    assert getattr(config, attrs[0]).base_model_ep_plan
    others = [config, *(getattr(config, attr) for attr in attrs[1:])]
    assert not any(hasattr(other, "base_model_ep_plan") for other in others)


@needs_ep_transformers
@pytest.mark.parametrize("by_prepare", [False, True], ids=["by-the-spec", "by-prepare"])
def test_a_missing_nested_config_is_an_error_not_a_silent_skip(monkeypatch, by_prepare):
    model_type = "nemotron_h_omni"
    if by_prepare:
        # A plan only the spec's prepare supplies is refused the same way, before prepare runs.
        model_type = "_probe"
        spec = ModelSpec(
            model_type=model_type,
            ep_spec=EPSpec(
                plan_config_attr=("text_config", "llm_config"),
                prepare=lambda *args, **kwargs: pytest.fail("ran prepare"),
                prepared_ep_plan={"layers.*.experts": "grouped_gemm"},
            ),
        )
        monkeypatch.setattr(
            "modelopt.torch.models.get_spec", lambda mt: spec if mt == model_type else None
        )
    with pytest.raises(ValueError, match=r"config\.text_config or config\.llm_config"):
        apply_distributed_support(model_type, cfg())


@needs_ep_transformers
def test_a_plan_at_the_root_needs_no_nested_config():
    """A config carrying a plan of its own at its root keeps it, as an existing plan does wherever it
    sits, and the expert-parallel load reads it there: the nested config the spec's own plan would
    go on is not required."""
    plan = {"layers.*.experts": "grouped_gemm"}
    config = cfg(base_model_ep_plan=dict(plan))

    assert apply_distributed_support("nemotron_h_omni", config) is config
    assert config.base_model_ep_plan == plan


@needs_ep_transformers
@pytest.mark.parametrize(
    "supplied",
    [
        {"base_model_ep_plan": {"layers.*.experts": "grouped_gemm"}},
        {"prepare": lambda *args, **kwargs: pytest.fail("ran prepare")},
    ],
    ids=["by-the-spec", "by-prepare"],
)
def test_a_plan_attached_where_the_load_does_not_read_one_is_an_error(monkeypatch, supplied):
    """The load reads a VLM's plan off text_config or llm_config: attached anywhere else, it would
    be read as no plan at all, and the MoE planned as a dense model. Refused whichever supplies the
    plan, the spec or its prepare, before either touches the config."""
    spec = ModelSpec(
        model_type="_probe", ep_spec=EPSpec(plan_config_attr="language_config", **supplied)
    )
    monkeypatch.setattr(
        "modelopt.torch.models.get_spec", lambda mt: spec if mt == "_probe" else None
    )
    config = cfg(language_config=cfg())

    with pytest.raises(ValueError, match="language_config, where the expert-parallel load"):
        apply_distributed_support("_probe", config)
    assert vars(config.language_config) == {}


@pytest.mark.parametrize("model_type", ["gpt_oss", "llama4", "llama4_text"])
def test_an_unsupported_model_is_refused_with_its_config_untouched(model_type):
    """These ship plans of their own, so only their spec's refusal keeps them off the
    expert-parallel path. The refusal names a single process first: FSDP2 loads only an
    unquantized causal LM."""
    config = cfg()

    with pytest.raises(NotImplementedError, match=r"yet: .* in a single process, or FSDP2-sharded"):
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


@needs_ep_transformers
def test_prepare_runs_after_the_plan_is_attached_and_its_config_is_returned(monkeypatch):
    """A prepare hook may inspect or adjust the plan, so it must not run before there is one. A
    config it returns in place of the one it was handed is what the load gets, the plan carried
    over, as the spec's contract asks."""
    seen = {}
    replacement = cfg()

    def prepare(hf_config, **kwargs):
        seen.update(kwargs, config=hf_config, plan_present=hasattr(hf_config, "base_model_ep_plan"))
        replacement.base_model_ep_plan = hf_config.base_model_ep_plan
        return replacement

    spec = ModelSpec(
        model_type="_probe",
        ep_spec=EPSpec(base_model_ep_plan={"layers.*.experts": "grouped_gemm"}, prepare=prepare),
    )
    monkeypatch.setattr(
        "modelopt.torch.models.get_spec", lambda mt: spec if mt == "_probe" else None
    )
    config = cfg()

    returned = apply_distributed_support("_probe", config, ckpt_path="/c", trust_remote_code=True)

    assert returned is replacement
    assert returned.base_model_ep_plan == spec.ep_spec.base_model_ep_plan
    assert seen == {
        "config": config,
        "ckpt_path": "/c",
        "trust_remote_code": True,
        "plan_present": True,
    }


@needs_transformers
def test_the_plan_a_spec_would_supply(monkeypatch):
    """What the plan decision reads at an EP degree of 1, to refuse an MoE left whole on every rank:
    the spec's plan, or the plan it states its ``prepare`` attaches, read without running
    ``prepare``; nothing for a model with no spec or no expert-parallel section, and nothing for a
    spec that refuses expert parallelism even when it names a plan. A model below its spec's
    transformers floor keeps its plan: its experts are no less routed, and whole on every rank they
    are refused like any other's."""
    assert moe._spec_ep_plan("deepseek_v3") == get_spec("deepseek_v3").ep_spec.base_model_ep_plan
    assert moe._spec_ep_plan("llama") == {}
    assert moe._spec_ep_plan("no_such_model_type") == {}
    prepared = {"layers.*.mlp.experts": "moe_tp_experts"}
    specs = {
        "preparing": SimpleNamespace(
            ep_spec=EPSpec(
                prepare=lambda *a, **k: pytest.fail("ran prepare"), prepared_ep_plan=prepared
            )
        ),
        "refusing": SimpleNamespace(
            ep_spec=SimpleNamespace(unsupported="not yet", base_model_ep_plan={"layers.*.mlp": "x"})
        ),
        "below_its_floor": SimpleNamespace(
            ep_spec=EPSpec(base_model_ep_plan=prepared, requires_transformers="999")
        ),
    }
    monkeypatch.setattr(models, "get_spec", specs.get)
    assert moe._spec_ep_plan("preparing") == prepared
    assert moe._spec_ep_plan("refusing") == {}
    assert moe._spec_refuses_ep("below_its_floor")
    assert moe._spec_ep_plan("below_its_floor") == prepared
