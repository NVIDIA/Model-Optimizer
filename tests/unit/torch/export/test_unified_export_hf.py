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

"""Tests for tied-weight helpers in unified_export_hf."""

from collections import OrderedDict

import torch
from _test_utils.torch.quantization.tied_modules import (
    make_tied_linear_pair,
    wrap_in_parent_with_tied_keys,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_utils import (
    TiedGroupResolver,
    _build_tied_alias_map,
    _collect_canonical_tied_patterns,
    _reorder_canonical_first,
)
from modelopt.torch.export.quant_utils import (
    fuse_prequant_layernorm,
    postprocess_state_dict,
    sync_tied_input_amax,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def test_collect_canonical_tied_patterns_dict_style():
    """Dict-style _tied_weights_keys yields regex patterns + canonical-side substrings."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    patterns, side_substrings = _collect_canonical_tied_patterns(parent)

    assert len(patterns) >= 1
    # "decoder" is in the canonical RHS but not the alias LHS — must auto-derive.
    # "encoder" is alias-only and must NOT be returned as canonical (would invert dedup).
    assert "decoder" in side_substrings
    assert "encoder" not in side_substrings


def test_collect_canonical_tied_patterns_list_style_yields_no_canonical_info():
    """Legacy list-style _tied_weights_keys carries no canonical/alias info — returns empty."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=False)

    patterns, side_substrings = _collect_canonical_tied_patterns(parent)

    assert patterns == []
    assert side_substrings == []


def test_build_tied_alias_map_dict_style_maps_alias_to_canonical():
    """Dict-style _tied_weights_keys yields {alias_full_name: canonical_full_name}."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    amap = _build_tied_alias_map(parent)

    assert amap == {"encoder.weight": "decoder.weight"}


def test_build_tied_alias_map_list_style_is_empty():
    """Legacy list-style _tied_weights_keys carries no canonical info — empty map."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=False)

    assert _build_tied_alias_map(parent) == {}


def test_build_tied_alias_map_skips_declared_but_unapplied_tie():
    """A class declares a dict-style tie but the two params are DISTINCT objects
    (e.g. ``tie_word_embeddings=False`` still ships a class-level ``_tied_weights_keys``).
    A name-only match would drop an independent weight; the build-time object-identity
    gate must skip it so the map stays empty.
    """

    class _Untied(torch.nn.Module):
        _tied_weights_keys = {r"^lm_head\.weight$": "embed.weight"}

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)  # separate weight object

    parent = _Untied()
    assert parent.lm_head.weight is not parent.embed.weight  # declared, but not applied
    assert _build_tied_alias_map(parent) == {}


def test_tied_group_resolver_group_key_is_shared_and_order_independent():
    """Both sides of a declared tie map to the same key; untied params map to None."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    resolver = TiedGroupResolver(parent)

    assert resolver.group_key("encoder.weight") == resolver.group_key("decoder.weight")
    assert resolver.group_key("encoder.weight") == "decoder.weight"  # canonical wins
    assert resolver.group_key("unrelated.weight") is None


def test_tied_group_resolver_per_layer_backreference():
    """Per-layer alias regex with a backreference resolves each layer independently."""

    class _Parent(torch.nn.Module):
        _tied_weights_keys = {
            r"^encoder\.layers\.(\d+)\.experts\.gate_up_proj$": r"decoder.layers.\1.experts.gate_up_proj",
        }

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.decoder = torch.nn.Module()
            for side in (self.encoder, self.decoder):
                side.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
            # Tie the fused expert Parameter per layer.
            for i in range(2):
                p = torch.nn.Parameter(torch.zeros(4, 8, 8))
                self.decoder.layers[i].experts = torch.nn.Module()
                self.decoder.layers[i].experts.gate_up_proj = p
                self.encoder.layers[i].experts = torch.nn.Module()
                self.encoder.layers[i].experts.gate_up_proj = p

    parent = _Parent()
    resolver = TiedGroupResolver(parent)

    assert (
        resolver.container_group_key("encoder.layers.0.experts", "gate_up_proj")
        == "decoder.layers.0.experts"
    )
    assert (
        resolver.container_group_key("encoder.layers.1.experts", "gate_up_proj")
        == "decoder.layers.1.experts"
    )
    # Encoder layer 0 must not collapse into decoder layer 1.
    assert resolver.container_group_key(
        "encoder.layers.0.experts", "gate_up_proj"
    ) != resolver.container_group_key("encoder.layers.1.experts", "gate_up_proj")


def test_tied_group_resolver_parallel_pattern_declaration():
    """DiffusionGemma-style {alias_regex: canonical_regex} (parallel patterns, no backrefs).

    Both sides are regexes that differ only in a leading literal head
    (``encoder.language_model.`` -> ``decoder.``); the shared trailing structure is copied
    from the concrete name. The container's fused expert Parameter is split into per-expert
    keys on export, so the alias-prefix rewrite must map those to the decoder canonical.
    """

    class _Model(torch.nn.Module):
        _tied_weights_keys = {
            r"encoder.language_model.layers\.(?:[^.]+\.)*gate_up_proj": r"decoder.layers\.(?:[^.]+\.)*gate_up_proj",
            r"encoder.language_model.layers\.(?:[^.]+\.)*down_proj": r"decoder.layers\.(?:[^.]+\.)*down_proj",
        }

        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.encoder = torch.nn.Module()
            self.encoder.language_model = torch.nn.Module()
            for root in (self.decoder, self.encoder.language_model):
                root.layers = torch.nn.ModuleList([torch.nn.Module()])
            gup = torch.nn.Parameter(torch.zeros(4, 8, 8))
            dp = torch.nn.Parameter(torch.zeros(4, 8, 8))
            for root in (self.decoder, self.encoder.language_model):
                root.layers[0].experts = torch.nn.Module()
                root.layers[0].experts.gate_up_proj = gup  # tied (same object)
                root.layers[0].experts.down_proj = dp

    class _Root(torch.nn.Module):  # ForCausalLM-style `.model` wrapper
        def __init__(self):
            super().__init__()
            self.model = _Model()

    resolver = TiedGroupResolver(_Root())

    # container group key: encoder side resolves to the decoder canonical container
    assert (
        resolver.container_group_key(
            "model.encoder.language_model.layers.0.experts", "gate_up_proj"
        )
        == "model.decoder.layers.0.experts"
    )
    # post-export per-expert split key rewrites to the decoder canonical (so it is dropped)
    prefixes = resolver.alias_prefix_pairs()
    got = resolver.canonical_state_dict_key(
        "model.encoder.language_model.layers.0.experts.3.gate_proj.weight", prefixes
    )
    assert got == "model.decoder.layers.0.experts.3.gate_proj.weight"


def test_reorder_canonical_first_puts_decoder_keys_before_encoder_keys():
    """_reorder_canonical_first moves canonical-side state_dict keys ahead of alias-side keys."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    sd = OrderedDict(
        [
            ("encoder.weight", torch.zeros(1)),
            ("unrelated.foo", torch.zeros(1)),
            ("decoder.weight", torch.zeros(1)),
        ]
    )

    reordered = _reorder_canonical_first(sd, parent)
    keys = list(reordered.keys())

    assert keys.index("decoder.weight") < keys.index("encoder.weight")
    assert set(reordered) == set(sd)  # no drops or additions


def _quantize_and_get_input_quantizers(parent):
    """Insert FP8 quantizers via no-op forward_loop and return both input_quantizers."""
    mtq.quantize(parent, mtq.FP8_DEFAULT_CFG, forward_loop=lambda m: None)
    return parent.encoder.input_quantizer, parent.decoder.input_quantizer


def test_sync_tied_input_amax_max_merges_tied_module_amaxes_in_place():
    """Tied Linears with divergent input_quantizer.amax get both sides overwritten with the max."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    enc_q, dec_q = _quantize_and_get_input_quantizers(parent)

    enc_q.amax = torch.tensor(2.0)
    dec_q.amax = torch.tensor(5.0)

    sync_tied_input_amax(parent)

    expected = torch.tensor(5.0)
    assert torch.allclose(enc_q.amax, expected)
    assert torch.allclose(dec_q.amax, expected)


def test_sync_tied_input_amax_no_op_for_untied_modules():
    """Untied Linears keep their per-side amaxes — the helper is a no-op when there's no tie."""
    parent = torch.nn.Module()
    parent.encoder = torch.nn.Linear(16, 32, bias=False)
    parent.decoder = torch.nn.Linear(16, 32, bias=False)
    enc_q, dec_q = _quantize_and_get_input_quantizers(parent)

    enc_q.amax = torch.tensor(2.0)
    dec_q.amax = torch.tensor(5.0)

    sync_tied_input_amax(parent)

    assert torch.allclose(enc_q.amax, torch.tensor(2.0))
    assert torch.allclose(dec_q.amax, torch.tensor(5.0))


def test_sync_tied_input_amax_merges_undeclared_shared_weight():
    """Two Linears that physically SHARE a weight object but declare no tie still get
    their input amaxes merged by object identity. The address backstop drops one such
    side, so the surviving input_scale must cover both activation ranges (no clipping).
    """
    parent = torch.nn.Module()
    parent.a = torch.nn.Linear(16, 32, bias=False)
    parent.b = torch.nn.Linear(16, 32, bias=False)
    parent.b.weight = parent.a.weight  # undeclared physical share (same Parameter object)

    mtq.quantize(parent, mtq.FP8_DEFAULT_CFG, forward_loop=lambda m: None)
    assert parent.a.weight is parent.b.weight  # share survives quantize
    # No _tied_weights_keys declared, so name-based grouping finds nothing to merge.
    assert TiedGroupResolver(parent).group_key("a.weight") is None

    parent.a.input_quantizer.amax = torch.tensor(2.0)
    parent.b.input_quantizer.amax = torch.tensor(8.0)

    sync_tied_input_amax(parent)

    expected = torch.tensor(8.0)
    assert torch.allclose(parent.a.input_quantizer.amax, expected)
    assert torch.allclose(parent.b.input_quantizer.amax, expected)


def test_postprocess_name_based_drops_alias_across_distinct_addresses():
    """Declared alias is dropped by name even when its tensor has a DIFFERENT address
    than the canonical -- the FSDP full_state_dict case that address dedup cannot catch.
    """
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    resolver = TiedGroupResolver(parent)

    # Distinct storages (different data_ptr): the address pass could never collapse these.
    sd = {"encoder.weight": torch.randn(4, 4), "decoder.weight": torch.randn(4, 4)}
    assert sd["encoder.weight"].data_ptr() != sd["decoder.weight"].data_ptr()

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, resolver=resolver)

    assert "decoder.weight" in out  # canonical kept
    assert "encoder.weight" not in out  # alias dropped by name


def test_postprocess_name_based_keeps_alias_when_canonical_absent():
    """An alias is NOT dropped when its canonical counterpart is missing (no orphaning)."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    resolver = TiedGroupResolver(parent)

    sd = {"encoder.weight": torch.randn(4, 4)}  # canonical decoder.weight absent
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, resolver=resolver)

    assert "encoder.weight" in out


def test_postprocess_name_based_keeps_both_sides_of_bidirectional_tie():
    """A bidirectional map (A->B and B->A) must not delete BOTH sides (missing tensor).

    Only an alias whose canonical is a *terminal* canonical (not itself an alias) is
    dropped; a pathological bidirectional declaration keeps both keys so the loader is
    never left without a tensor.
    """

    class _Bi(torch.nn.Module):
        _tied_weights_keys = {r"^a\.weight$": "b.weight", r"^b\.weight$": "a.weight"}

        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, bias=False)
            self.b = torch.nn.Linear(4, 4, bias=False)
            self.b.weight = self.a.weight  # genuinely shared, declared both ways

    resolver = TiedGroupResolver(_Bi())
    # sanity: the map really is bidirectional at the prefix level
    assert resolver.alias_prefix_pairs() == {"a": "b", "b": "a"}

    sd = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(4, 4)}
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, resolver=resolver)

    assert "a.weight" in out and "b.weight" in out  # neither side dropped


def test_postprocess_keeps_both_sides_when_tied_quant_state_differs():
    """Tied sides with different quant state (alias quantized, canonical not) must not be
    deduped: dropping ``encoder.weight`` while its ``weight_scale`` / ``input_scale`` have
    no canonical counterpart would orphan scales on a weight that's gone. The alias group
    is dropped atomically (all-or-nothing), so here nothing is dropped.
    """
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    resolver = TiedGroupResolver(parent)

    sd = {
        # alias (encoder) exported as quantized: weight + companion scales
        "encoder.weight": torch.randn(4, 4),
        "encoder.weight_scale": torch.randn(4),
        "encoder.input_scale": torch.randn(1),
        # canonical (decoder) exported unquantized: weight only, no scales
        "decoder.weight": torch.randn(4, 4),
    }

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, resolver=resolver)

    # Mismatched companion keys -> keep the whole alias group; no orphaned scales.
    assert set(out) == set(sd)


def test_postprocess_name_based_drops_tied_expert_subtree_by_name():
    """A container-level declared expert tie drops every per-expert alias key by name,
    keeping only the canonical subtree -- across distinct addresses (FSDP-safe)."""

    class _Parent(torch.nn.Module):
        _tied_weights_keys = {
            r"^encoder\.experts\.gate_up_proj$": "decoder.experts.gate_up_proj",
            r"^encoder\.experts\.down_proj$": "decoder.experts.down_proj",
        }

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.experts = torch.nn.Module()
            self.decoder = torch.nn.Module()
            self.decoder.experts = torch.nn.Module()
            gup = torch.nn.Parameter(torch.zeros(2, 4, 4))
            dp = torch.nn.Parameter(torch.zeros(2, 4, 4))
            # decoder registered first (canonical) to exercise remove_duplicate=False.
            self.decoder.experts.gate_up_proj = gup
            self.decoder.experts.down_proj = dp
            self.encoder.experts.gate_up_proj = gup
            self.encoder.experts.down_proj = dp

    parent = _Parent()
    resolver = TiedGroupResolver(parent)
    assert resolver.alias_prefix_pairs() == {"encoder.experts": "decoder.experts"}

    # Craft exported-style per-expert keys with distinct storages on both sides.
    sd = {}
    for side in ("encoder", "decoder"):
        for e in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                sd[f"{side}.experts.{e}.{proj}.weight"] = torch.randn(4, 4)
                sd[f"{side}.experts.{e}.{proj}.weight_scale"] = torch.randn(4)

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, resolver=resolver)

    assert not any(k.startswith("encoder.experts.") for k in out)  # all aliases dropped
    assert all(k.startswith("decoder.experts.") for k in out)  # only canonical remains
    assert len(out) == 2 * 3 * 2  # 2 experts * 3 projections * (weight + weight_scale)


def test_postprocess_state_dict_preserves_tensors_with_different_byte_ranges():
    storage = torch.arange(4)
    state_dict = {"short": storage[:2], "long": storage}
    assert state_dict["short"].data_ptr() == state_dict["long"].data_ptr()

    processed = postprocess_state_dict(state_dict, maxbound=448, quantization=None)

    assert set(processed) == set(state_dict)


def test_postprocess_state_dict_preserves_zero_pointer_tensors():
    state_dict = {
        "first": torch.empty(4, device="meta"),
        "second": torch.empty(4, device="meta"),
    }

    processed = postprocess_state_dict(state_dict, maxbound=448, quantization=None)

    assert set(processed) == set(state_dict)


def _linear_with_input_quantizer():
    linear = torch.nn.Linear(4, 4, bias=False)
    linear.input_quantizer = TensorQuantizer()
    return linear


def test_fuse_prequant_layernorm_skips_modules_without_pre_quant_scale():
    layernorm = torch.nn.LayerNorm(4)
    original_weight = layernorm.weight.detach().clone()
    modules = [_linear_with_input_quantizer(), _linear_with_input_quantizer()]

    fuse_prequant_layernorm(layernorm, modules)

    assert torch.allclose(layernorm.weight, original_weight)
    assert not hasattr(modules[0], "fused_with_prequant")
    assert not hasattr(modules[1], "fused_with_prequant")


def test_fuse_prequant_layernorm_fuses_and_removes_pre_quant_scale():
    layernorm = torch.nn.LayerNorm(4)
    modules = [_linear_with_input_quantizer(), _linear_with_input_quantizer()]
    pre_quant_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for module in modules:
        module.input_quantizer._pre_quant_scale = pre_quant_scale

    fuse_prequant_layernorm(layernorm, modules)

    assert torch.allclose(layernorm.weight, pre_quant_scale)
    assert torch.allclose(layernorm.bias, torch.zeros_like(pre_quant_scale))
    for module in modules:
        assert not hasattr(module.input_quantizer, "_pre_quant_scale")
        assert module.fused_with_prequant
