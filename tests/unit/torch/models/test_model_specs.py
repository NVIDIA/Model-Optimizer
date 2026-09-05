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

"""Unit tests for the per-model spec registry (modelopt.torch.models)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modelopt.torch.export.layer_utils import (
    get_expert_linear_names,
    get_experts_list,
    is_moe,
    sync_moe_gate_up_amax,
)
from modelopt.torch.export.quant_utils import _layernorm_uses_weight_plus_one
from modelopt.torch.models import (
    ModelSpec,
    MoESpec,
    MoEVariant,
    get_spec,
    get_specs,
    hf_model_type,
    list_all_possible,
    match_moe_block,
)
from modelopt.torch.models.specs import _SPECS


def _variants(spec):
    """The MoE variants a spec declares; empty when it has no MoE section."""
    return spec.moe_spec.moe_variants if spec.moe_spec is not None else ()


class Qwen3MoeSparseMoeBlock(nn.Module):
    pass


class MixtralSparseMoeBlock(nn.Module):
    pass


class QuantMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    """Quantized classes subclass the original module class; matching goes through the MRO."""


class _UnknownMoeBlock(nn.Module):
    pass


def test_match_moe_block_by_class_name():
    variant = match_moe_block(Qwen3MoeSparseMoeBlock())
    assert variant is not None
    assert variant.expert_linear_names == ("gate_proj", "down_proj", "up_proj")
    assert variant.has_iterable_experts


def test_match_moe_block_matches_quantized_class_via_mro():
    variant = match_moe_block(QuantMixtralSparseMoeBlock())
    assert variant is not None
    assert variant.expert_linear_names == ("w1", "w2", "w3")


def test_match_moe_block_unmatched_returns_none():
    assert match_moe_block(_UnknownMoeBlock()) is None


def test_get_expert_linear_names_raises_when_unmatched():
    # No spec for these model types and no fused-expert structure — must fail loudly
    # instead of guessing another model's naming (the legacy w1/w2/w3 default was
    # removed).
    with pytest.raises(NotImplementedError, match="expert linear names"):
        get_expert_linear_names(_UnknownMoeBlock(), "some_unknown_model")


def test_get_expert_linear_names_from_specs():
    # Arctic keeps the w1/w2/w3 naming it previously got from the engine default.
    assert get_expert_linear_names(_UnknownMoeBlock(), "arctic") == ["w1", "w2", "w3"]
    # DBRX resolves the quantized per-expert ModuleList names (previously it fell
    # through to the w1/w2/w3 default, which never existed on the quantized module).
    assert get_expert_linear_names(_UnknownMoeBlock(), "dbrx") == [
        "w1_linear",
        "w2_linear",
        "v1_linear",
    ]


class NemotronHMOE(nn.Module):
    def __init__(self):
        super().__init__()

        class _Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(8, 16)
                self.down_proj = nn.Linear(16, 8)

        self.experts = nn.ModuleList([_Expert() for _ in range(3)])


def test_get_experts_list_groups_by_spec_linear_names():
    module = NemotronHMOE()
    groups = get_experts_list(module, "nemotron_h")
    assert len(groups) == 2  # up_proj group + down_proj group
    assert all(len(group) == 3 for group in groups)
    assert groups[0][0] is module.experts[0].up_proj
    assert groups[1][2] is module.experts[2].down_proj


def test_get_experts_list_skips_fused_expert_containers():
    """A fused experts container has nothing to group, and must not crash trying.

    transformers 5 replaced several iterable expert ModuleLists with a single module
    holding 3-D parameters (DeepseekV3Experts, MixtralExperts), while their specs still
    describe the transformers 4 iterable layout -- the spec cannot tell the two apart,
    only the module can. Without the structural guard, the AWQ/SVDQuant resmooth pass
    in requantize_resmooth_fused_llm_layers reaches ``len(module.experts)`` on a module
    with no ``__len__`` and dies with TypeError mid-export.
    """

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16))
            self.down_proj = nn.Parameter(torch.zeros(4, 16, 8))
            # Present once modelopt quantizes fused experts; makes
            # get_expert_linear_names take its structural shortcut.
            self.gate_up_proj_weight_quantizers = nn.ModuleList()

    class DeepseekV3MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()

    assert get_experts_list(DeepseekV3MoE(), "deepseek_v3") == []

    # Same shape under a model whose spec also claims iterable experts.
    class MixtralSparseMoeBlockFused(MixtralSparseMoeBlock):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()

    assert get_experts_list(MixtralSparseMoeBlockFused(), "mixtral") == []


def test_get_experts_list_rejects_non_iterable_layouts():
    # DBRX matches a spec but is not an iterable-experts layout; grouped export
    # must keep rejecting it (legacy behavior).
    class DbrxFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList()

    with pytest.raises(NotImplementedError):
        get_experts_list(DbrxFFN(), "dbrx")

    with pytest.raises(NotImplementedError):
        get_experts_list(_UnknownMoeBlock(), "some_unknown_model")


class ArcticMoE(nn.Module):
    pass


class DbrxFFN(nn.Module):
    pass


def test_is_moe_matches_registered_non_standard_names():
    # Non-standard MoE block names (no *SparseMoeBlock suffix, no router/experts
    # attributes) resolve through the model spec registry.
    assert is_moe(ArcticMoE())
    assert is_moe(DbrxFFN())
    assert not is_moe(_UnknownMoeBlock())


def test_match_is_exact_name_not_substring():
    # A class whose name merely CONTAINS a registered name must not match; only
    # exact MRO class names do (quantized wrappers match via their base class).
    class MyArcticMoEWrapper(nn.Module):
        pass

    assert match_moe_block(MyArcticMoEWrapper()) is None


def test_pqs_fuse_rules_match_legacy_mapping():
    # Aggregated per-model rules, flattened to (class_substring, fuse_into, fuse_from)
    # triples, must reproduce the legacy PQS_FUSE_MODULE_MAPPING.
    rules = {
        (substring, fuse_into, fuse_from)
        for substrings, fuse_into, fuse_from in list_all_possible("pqs_fuse_rules")
        for substring in substrings
    }
    legacy = {
        ("LlamaAttention", "v_proj", "o_proj"),
        ("LlamaMLP", "up_proj", "down_proj"),
        ("Qwen3Attention", "v_proj", "o_proj"),
        ("Qwen3MoeAttention", "v_proj", "o_proj"),
        ("Qwen3MLP", "up_proj", "down_proj"),
        ("Qwen3MoeMLP", "up_proj", "down_proj"),
    }
    assert rules == legacy


def test_gate_up_pairs_match_legacy():
    # Aggregated per-model pairs must reproduce the legacy _GATE_UP_PAIRS set.
    assert set(list_all_possible("gate_up_pairs")) == {("gate_proj", "up_proj"), ("w1", "w3")}


def test_list_all_possible_rejects_unknown_attr():
    with pytest.raises(ValueError, match="not a ModelSpec attribute"):
        list_all_possible("no_such_field")


def test_list_all_possible_rejects_scalar_attr():
    # model_type is a str: iterating it would yield characters, not values.
    with pytest.raises(ValueError, match="expects a tuple-valued attribute"):
        list_all_possible("model_type")


def test_list_all_possible_handles_unhashable_items():
    # MoEVariant is a mutable dataclass and therefore unhashable; deduplication must
    # not go through set()/dict.fromkeys().
    variants = list_all_possible("moe_variants")
    assert variants
    assert all(isinstance(v, MoEVariant) for v in variants)


def test_weight_plus_one_norm_names_cover_legacy():
    names = set(list_all_possible("weight_plus_one_norm_names"))
    assert {"GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm", "LayerNorm1P"} <= names


def test_layernorm_weight_plus_one_via_specs():
    class GemmaRMSNorm(nn.Module):
        pass

    class NemotronLayerNorm1P(nn.Module):
        pass

    class PlainRMSNorm(nn.Module):
        pass

    class ZeroCentered(nn.Module):
        zero_centered_gamma = True

    assert _layernorm_uses_weight_plus_one(GemmaRMSNorm())
    assert _layernorm_uses_weight_plus_one(NemotronLayerNorm1P())
    assert not _layernorm_uses_weight_plus_one(PlainRMSNorm())
    # Structural fallback stays in the engine.
    assert _layernorm_uses_weight_plus_one(ZeroCentered())


class _FakeQuantizer:
    def __init__(self, amax):
        self.amax = amax


def _make_gated_block(block_cls, gate_name, up_name, gate_amax, up_amax):
    class _Expert(nn.Module):
        def __init__(self):
            super().__init__()
            setattr(self, gate_name, nn.Linear(4, 8))
            setattr(self, up_name, nn.Linear(4, 8))
            getattr(self, gate_name).weight_quantizer = _FakeQuantizer(torch.tensor(gate_amax))
            getattr(self, up_name).weight_quantizer = _FakeQuantizer(torch.tensor(up_amax))

    block = block_cls()
    block.experts = nn.ModuleList([_Expert()])
    return block


def test_sync_moe_gate_up_amax_uses_own_spec():
    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    model = nn.Module()
    model.moe = _make_gated_block(
        Qwen3MoeSparseMoeBlock, "gate_proj", "up_proj", [1.0, 3.0], [2.0, 2.0]
    )
    assert sync_moe_gate_up_amax(model) == 1
    expert = model.moe.experts[0]
    assert torch.equal(expert.gate_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))
    assert torch.equal(expert.up_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))


def test_sync_moe_gate_up_amax_falls_back_for_unmatched_block():
    class UnknownSparseMoeBlock(nn.Module):
        """Passes is_moe by naming convention but has no MoESpec."""

    # Quantization admits MoE blocks structurally, so unregistered families reach
    # this function. They must still sync: skipping them would leave the two halves
    # of the fused gate_up_proj on inconsistent weight_scale_2, which is exactly the
    # corruption this function exists to prevent. Pre-registry behavior tried every
    # known gate/up naming; that fallback is preserved.
    model = nn.Module()
    model.moe = _make_gated_block(
        UnknownSparseMoeBlock, "gate_proj", "up_proj", [1.0, 3.0], [2.0, 2.0]
    )
    assert sync_moe_gate_up_amax(model) == 1
    expert = model.moe.experts[0]
    assert torch.equal(expert.gate_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))
    assert torch.equal(expert.up_proj.weight_quantizer.amax, torch.tensor([2.0, 3.0]))


def test_sync_moe_gate_up_amax_fallback_covers_w1_w3_naming():
    class UnknownSparseMoeBlock(nn.Module):
        pass

    # The fallback vocabulary is every declared pair, not just gate_proj/up_proj.
    model = nn.Module()
    model.moe = _make_gated_block(UnknownSparseMoeBlock, "w1", "w3", [1.0, 3.0], [2.0, 2.0])
    assert sync_moe_gate_up_amax(model) == 1
    assert torch.equal(model.moe.experts[0].w1.weight_quantizer.amax, torch.tensor([2.0, 3.0]))


def test_sync_moe_gate_up_amax_skips_variant_without_gate_up_pair():
    # A registered variant that declares no pair (non-gated or already-fused experts)
    # must NOT fall back to the global vocabulary -- its spec already says "no sync".
    class NemotronHMOE(nn.Module):
        pass

    model = nn.Module()
    model.moe = _make_gated_block(NemotronHMOE, "gate_proj", "up_proj", [1.0, 3.0], [2.0, 2.0])
    model.config = SimpleNamespace(model_type="nemotron_h")
    assert sync_moe_gate_up_amax(model) == 0


def test_match_moe_block_scope_prefers_own_model_type():
    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    # A hypothetical remote-code fork registering the same block class name under
    # its own model type: scope must pick the model's own spec among candidates.
    fork_variant = MoEVariant(
        block_names=("Qwen3MoeSparseMoeBlock",),
        expert_linear_names=("a_proj", "b_proj"),
    )
    fork_spec = ModelSpec(model_type="zz_fork", moe_spec=MoESpec(moe_variants=(fork_variant,)))
    _SPECS[fork_spec.model_type] = fork_spec
    try:
        assert match_moe_block(Qwen3MoeSparseMoeBlock(), "zz_fork") is fork_variant
        assert match_moe_block(Qwen3MoeSparseMoeBlock(), "qwen3_moe").expert_linear_names == (
            "gate_proj",
            "down_proj",
            "up_proj",
        )
        # No scope -> first registered class-name match wins (legacy order).
        assert match_moe_block(Qwen3MoeSparseMoeBlock()).expert_linear_names == (
            "gate_proj",
            "down_proj",
            "up_proj",
        )
    finally:
        del _SPECS[fork_spec.model_type]


def test_match_moe_block_scope_is_strict():
    class Qwen3MoeSparseMoeBlock(nn.Module):
        pass

    # A model whose model_type has no spec resolves to None even when its module
    # class name coincides with another model's — register a spec instead of
    # inheriting a neighbor's data.
    assert match_moe_block(Qwen3MoeSparseMoeBlock(), "some_unknown_vlm") is None
    # No scope (no config available) searches all specs.
    assert match_moe_block(Qwen3MoeSparseMoeBlock()) is not None


def test_get_expert_linear_names_by_model_type_only():
    # With a scope, naming resolves from the model's own spec — the block class
    # name is irrelevant (a spec need not declare block_names to provide naming).
    assert get_expert_linear_names(_UnknownMoeBlock(), "qwen3_moe") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    with pytest.raises(NotImplementedError, match="model type"):
        get_expert_linear_names(_UnknownMoeBlock(), "some_unknown_vlm")


def test_mixtral_variants_disambiguated_by_block_class():
    class MixtralMoeSparseMoeBlock(nn.Module):
        """Legacy-naming Mixtral layout — same model type, different projections."""

    assert get_expert_linear_names(MixtralMoeSparseMoeBlock(), "mixtral") == [
        "linear_fc1",
        "linear_fc2",
    ]
    # An unrecognized block class under a multi-naming model type cannot resolve.
    with pytest.raises(NotImplementedError):
        get_expert_linear_names(_UnknownMoeBlock(), "mixtral")


def test_gemma4_both_root_types_resolve():
    # A gemma4 VLM's root model_type is gemma4; a text-only checkpoint's is
    # gemma4_text (gemma3 precedent). Both register the same layout.
    assert get_expert_linear_names(_UnknownMoeBlock(), "gemma4") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]
    assert get_expert_linear_names(_UnknownMoeBlock(), "gemma4_text") == [
        "gate_proj",
        "down_proj",
        "up_proj",
    ]


def test_hf_model_type_accepts_model_or_config():
    config = SimpleNamespace(model_type="qwen3_moe")
    model = SimpleNamespace(config=config)
    assert hf_model_type(model) == "qwen3_moe"
    assert hf_model_type(config) == "qwen3_moe"
    assert hf_model_type(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Structural fused-expert shortcut
#
# get_expert_linear_names checks for fused per-expert quantizer attributes BEFORE
# consulting any spec, and that check is reachable with model_type=None. It serves
# fused Mixtral (transformers 5.0+) and any module rewritten by
# _export_fused_experts, so it is a production path that must not regress when the
# registry changes.
# ---------------------------------------------------------------------------


def _make_fused_experts_block(first_proj_attr=None, block_cls=None):
    """MoE block whose experts container carries fused per-expert quantizers."""

    class _Experts(nn.Module):
        pass

    experts = _Experts()
    if first_proj_attr is not None:
        experts._first_proj_attr = first_proj_attr
    attr = first_proj_attr or "gate_up_proj"
    # Presence of the attribute is the signal; its value is never read.
    setattr(experts, f"{attr}_weight_quantizers", [object()])

    block = (block_cls or _UnknownMoeBlock)()
    block.experts = experts
    return block


def test_fused_experts_shortcut_without_model_type():
    # Resolves with no spec and no model_type at all: the structural signal alone
    # is enough, which is what the fused-export path relies on.
    assert get_expert_linear_names(_make_fused_experts_block(), None) == [
        "gate_up_proj",
        "down_proj",
    ]


def test_fused_experts_shortcut_precedes_spec_lookup():
    # The structural hit must win over a spec that WOULD resolve this module: a
    # fused module has already been rewritten, so its spec's unfused names no
    # longer describe it. The block class is one the mixtral spec matches, so the
    # two paths disagree and precedence is actually observable.
    block = _make_fused_experts_block(block_cls=MixtralSparseMoeBlock)
    assert get_expert_linear_names(block, "mixtral") == ["gate_up_proj", "down_proj"]
    # Sanity: without the fused marker the same class resolves to the spec's names,
    # so the assertion above really is testing precedence and not a dead spec.
    assert get_expert_linear_names(MixtralSparseMoeBlock(), "mixtral") == ["w1", "w2", "w3"]


def test_fused_experts_shortcut_honors_first_proj_attr_override():
    # _first_proj_attr overrides the gate_up_proj default (e.g. modules fused under
    # gate_proj naming); the non-default branch is otherwise untested.
    assert get_expert_linear_names(_make_fused_experts_block("gate_proj"), None) == [
        "gate_proj",
        "down_proj",
    ]


def test_fused_experts_shortcut_ignores_unrelated_attributes():
    # A container with experts but no fused quantizer attribute must fall through
    # to the spec path, not silently claim the fused naming.
    block = _UnknownMoeBlock()
    block.experts = nn.ModuleList()
    with pytest.raises(NotImplementedError):
        get_expert_linear_names(block, "some_unknown_model")


# ---------------------------------------------------------------------------
# Per-model MoE data, pinned as a table
#
# The registry machinery is covered above; these pin the actual values carried by
# every registered MoE spec, so a typo in a block class name or a flipped flag
# fails here instead of at export time.
# ---------------------------------------------------------------------------

# (model_type, block_names, expert_linear_names, has_iterable_experts, gate_up_pair)
EXPECTED_MOE_VARIANTS = [
    ("arctic", ("ArcticMoE",), ("w1", "w2", "w3"), False, ("w1", "w3")),
    ("dbrx", ("DbrxFFN",), ("w1_linear", "w2_linear", "v1_linear"), False, None),
    (
        "deepseek",
        ("DeepseekMoE",),
        ("gate_proj", "down_proj", "up_proj"),
        False,
        ("gate_proj", "up_proj"),
    ),
    (
        "deepseek_v3",
        ("DeepseekV3MoE",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
    (
        "deepseek_v4",
        ("DeepseekV4SparseMoeBlock",),
        ("gate_up_proj", "down_proj"),
        False,
        None,
    ),
    (
        "gemma4",
        ("Gemma4TextDecoderLayer",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
    (
        "gemma4_text",
        ("Gemma4TextDecoderLayer",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
    # transformers names the block GptOssMLP; GptOssMoE is the legacy name.
    ("gpt_oss", ("GptOssMLP", "GptOssMoE"), ("gate_up_proj", "down_proj"), False, None),
    ("mixtral", ("MixtralSparseMoeBlock",), ("w1", "w2", "w3"), True, ("w1", "w3")),
    ("mixtral", ("MixtralMoeSparseMoeBlock",), ("linear_fc1", "linear_fc2"), False, None),
    ("nemotron_h", ("NemotronHMOE",), ("up_proj", "down_proj"), True, None),
    (
        "qwen2_moe",
        ("Qwen2MoeSparseMoeBlock",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
    (
        # has_iterable_experts is False to match pre-refactor behavior: the legacy
        # get_experts_list keyed off the root class name and "qwen3_5moeforcausallm"
        # matched none of its qwen substrings. See modelopt/torch/models/qwen3_5_moe.py.
        "qwen3_5_moe",
        ("Qwen3_5MoeSparseMoeBlock",),
        ("gate_proj", "down_proj", "up_proj"),
        False,
        ("gate_proj", "up_proj"),
    ),
    (
        "qwen3_moe",
        ("Qwen3MoeSparseMoeBlock",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
    (
        "qwen3_next",
        ("Qwen3NextSparseMoeBlock",),
        ("gate_proj", "down_proj", "up_proj"),
        True,
        ("gate_proj", "up_proj"),
    ),
]


@pytest.mark.parametrize(
    ("model_type", "block_names", "expert_linear_names", "has_iterable_experts", "gate_up_pair"),
    EXPECTED_MOE_VARIANTS,
    ids=[f"{mt}:{bn[0]}" for mt, bn, _, _, _ in EXPECTED_MOE_VARIANTS],
)
def test_moe_variant_values(
    model_type, block_names, expert_linear_names, has_iterable_experts, gate_up_pair
):
    spec = get_spec(model_type)
    assert spec is not None, f"no spec registered for {model_type!r}"
    matching = [v for v in _variants(spec) if v.block_names == block_names]
    assert len(matching) == 1, f"expected exactly one {block_names} variant on {model_type!r}"
    variant = matching[0]
    assert variant.expert_linear_names == expert_linear_names
    assert variant.has_iterable_experts is has_iterable_experts
    assert variant.gate_up_pair == gate_up_pair


def test_moe_variant_table_is_exhaustive():
    # Fails when a new MoE spec is added without a row above, so the table cannot
    # silently drift out of date.
    registered = {(s.model_type, v.block_names) for s in get_specs() for v in _variants(s)}
    tabled = {(mt, bn) for mt, bn, _, _, _ in EXPECTED_MOE_VARIANTS}
    assert registered == tabled


@pytest.mark.parametrize(
    ("model_type", "block_names", "expert_linear_names", "has_iterable_experts", "gate_up_pair"),
    EXPECTED_MOE_VARIANTS,
    ids=[f"{mt}:{bn[0]}" for mt, bn, _, _, _ in EXPECTED_MOE_VARIANTS],
)
def test_expert_linear_names_resolve_for_every_registered_block(
    model_type, block_names, expert_linear_names, has_iterable_experts, gate_up_pair
):
    # End-to-end through the public accessor, with a stand-in module of the real
    # block class name -- covers the block classes no hand-written test names.
    block = type(block_names[0], (nn.Module,), {})()
    assert get_expert_linear_names(block, model_type) == list(expert_linear_names)


def test_gpt_oss_block_name_matches_the_real_module():
    # GptOssMoE never matched a real module: transformers names the block GptOssMLP,
    # and expert naming resolved only through the single-naming shortcut in
    # expert_linear_names_for. Pin the real name so the block match works too.
    class GptOssMLP(nn.Module):
        pass

    variant = match_moe_block(GptOssMLP(), "gpt_oss")
    assert variant is not None
    assert variant.expert_linear_names == ("gate_up_proj", "down_proj")
    assert is_moe(GptOssMLP(), "gpt_oss")


def test_gate_up_pair_is_a_subset_of_expert_linear_names():
    # A gate/up pair naming projections the variant does not declare would silently
    # never sync in sync_moe_gate_up_amax.
    for spec in get_specs():
        for variant in _variants(spec):
            if variant.gate_up_pair is None or variant.expert_linear_names is None:
                continue
            missing = set(variant.gate_up_pair) - set(variant.expert_linear_names)
            assert not missing, (
                f"{spec.model_type} {variant.block_names}: gate_up_pair names "
                f"{sorted(missing)} are not in expert_linear_names"
            )


def test_iterable_experts_matches_pre_refactor_support():
    """``has_iterable_experts`` must reproduce the legacy get_experts_list support set.

    Legacy keyed off ``type(root_model).__name__.lower()`` and supported exactly the
    substrings below; everything else raised NotImplementedError. This pins the
    refactor to that set so grouped export neither gains nor loses a model silently.

    ``ADDED_AFTER_REFACTOR`` exempts specs registered later. Those are deliberate new
    coverage, not refactor drift, so comparing them against the legacy set is
    meaningless -- but they are still listed by name, so the exemption stays reviewable.
    """
    # Registered after the refactor: deepseek_v3/v4 are invisible to legacy detection
    # (DeepseekV3MoE calls its router ``gate``), so they had no legacy behavior to match.
    added_after_refactor = {"deepseek_v3", "deepseek_v4"}
    legacy_substrings = (
        "mixtralforcausallm",
        "qwenmoeforcausallm",
        "qwen2moeforcausallm",
        "qwen3moeforcausallm",
        "qwen3nextforcausallm",
        "nemotronhforcausallm",
        "gemma4",
    )
    # The real HF root class name for each registered MoE model type.
    root_class_names = {
        "arctic": "ArcticForCausalLM",
        "dbrx": "DbrxForCausalLM",
        "deepseek": "DeepseekForCausalLM",
        "deepseek_v3": "DeepseekV3ForCausalLM",
        "deepseek_v4": "DeepseekV4ForCausalLM",
        "gemma4": "Gemma4ForConditionalGeneration",
        "gemma4_text": "Gemma4TextForCausalLM",
        "gpt_oss": "GptOssForCausalLM",
        "mixtral": "MixtralForCausalLM",
        "nemotron_h": "NemotronHForCausalLM",
        "qwen2_moe": "Qwen2MoeForCausalLM",
        "qwen3_5_moe": "Qwen3_5MoeForCausalLM",
        "qwen3_moe": "Qwen3MoeForCausalLM",
        "qwen3_next": "Qwen3NextForCausalLM",
    }
    moe_specs = [s for s in get_specs() if _variants(s)]
    assert {s.model_type for s in moe_specs} == set(root_class_names), (
        "root_class_names is out of sync with the registered MoE specs"
    )
    for spec in moe_specs:
        if spec.model_type in added_after_refactor:
            continue
        root = root_class_names[spec.model_type].lower()
        legacy_supported = any(sub in root for sub in legacy_substrings)
        spec_supported = any(v.has_iterable_experts for v in _variants(spec))
        assert spec_supported is legacy_supported, (
            f"{spec.model_type}: grouped-export support changed "
            f"(legacy={legacy_supported}, spec={spec_supported})"
        )
