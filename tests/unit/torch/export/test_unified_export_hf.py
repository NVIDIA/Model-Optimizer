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

import torch
from _test_utils.torch.quantization.tied_modules import (
    make_tied_linear_pair,
    wrap_in_parent_with_tied_keys,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_utils import TiedWeightMap, _build_tied_alias_map
from modelopt.torch.export.quant_utils import (
    fuse_prequant_layernorm,
    postprocess_state_dict,
    sync_tied_input_amax,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def test_build_tied_alias_map_dict_style_maps_alias_to_canonical():
    """Dict-style _tied_weights_keys yields {alias_full_name: canonical_full_name}."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    amap = _build_tied_alias_map(parent)

    assert amap == {"encoder.weight": "decoder.weight"}


def test_build_tied_alias_map_empty_for_non_applied_ties():
    """No alias map for list-style ties (no canonical) or dict-style ties whose params are distinct objects."""
    enc, dec = make_tied_linear_pair()
    list_style = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=False)
    assert _build_tied_alias_map(list_style) == {}

    class _Untied(torch.nn.Module):
        _tied_weights_keys = {r"^lm_head\.weight$": "embed.weight"}

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)  # separate weight object

    untied = _Untied()
    assert untied.lm_head.weight is not untied.embed.weight  # declared, but not applied
    assert _build_tied_alias_map(untied) == {}


def test_build_tied_alias_map_warns_when_declared_tie_unformed_under_fsdp(monkeypatch):
    """A declared tie whose id-group did not form warns under FSDP2 (loud) but is silent off FSDP."""
    import warnings as _warnings

    import modelopt.torch.export.model_utils as mu
    from modelopt.torch.export.model_utils import _build_tied_alias_map as build

    class _Untied(torch.nn.Module):
        _tied_weights_keys = {r"^lm_head\.weight$": "embed.weight"}

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)  # distinct object

    # Off FSDP: no warning.
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert build(_Untied()) == {}
    assert not any("shared-parameter group" in str(w.message) for w in rec)

    # Under FSDP2: warn. (is_fsdp2_model is imported into model_utils, so patch it there.)
    monkeypatch.setattr(mu, "is_fsdp2_model", lambda m: True)
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert build(_Untied()) == {}
    assert any("shared-parameter group" in str(w.message) for w in rec)


def test_tied_group_resolver_group_key_is_shared_and_order_independent():
    """Both sides of a declared tie map to the same key; untied params map to None."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    tied_map = TiedWeightMap(parent)

    assert tied_map.group_key("encoder.weight") == tied_map.group_key("decoder.weight")
    assert tied_map.group_key("encoder.weight") == "decoder.weight"  # canonical wins
    assert tied_map.group_key("unrelated.weight") is None


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
    tied_map = TiedWeightMap(parent)

    assert (
        tied_map.container_group_key("encoder.layers.0.experts", "gate_up_proj")
        == "decoder.layers.0.experts"
    )
    assert (
        tied_map.container_group_key("encoder.layers.1.experts", "gate_up_proj")
        == "decoder.layers.1.experts"
    )
    # Encoder layer 0 must not collapse into decoder layer 1.
    assert tied_map.container_group_key(
        "encoder.layers.0.experts", "gate_up_proj"
    ) != tied_map.container_group_key("encoder.layers.1.experts", "gate_up_proj")


def test_tied_group_resolver_parallel_pattern_declaration():
    """DiffusionGemma-style parallel-regex ties resolve each container to its decoder canonical."""

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

    tied_map = TiedWeightMap(_Root())

    # container group key: encoder side resolves to the decoder canonical container
    assert (
        tied_map.container_group_key(
            "model.encoder.language_model.layers.0.experts", "gate_up_proj"
        )
        == "model.decoder.layers.0.experts"
    )
    # post-export per-expert split keys of the (fully tied) container are dropped by name.
    enc = "model.encoder.language_model.layers.0.experts"
    dec = "model.decoder.layers.0.experts"
    sd = {
        f"{enc}.3.gate_proj.weight": torch.randn(4, 4),
        f"{dec}.3.gate_proj.weight": torch.randn(4, 4),
    }
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)
    assert f"{enc}.3.gate_proj.weight" not in out  # alias split key dropped
    assert f"{dec}.3.gate_proj.weight" in out  # canonical kept


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
    """Two Linears sharing a weight but declaring no tie still get their input amaxes merged by identity."""
    parent = torch.nn.Module()
    parent.a = torch.nn.Linear(16, 32, bias=False)
    parent.b = torch.nn.Linear(16, 32, bias=False)
    parent.b.weight = parent.a.weight  # undeclared physical share (same Parameter object)

    mtq.quantize(parent, mtq.FP8_DEFAULT_CFG, forward_loop=lambda m: None)
    assert parent.a.weight is parent.b.weight  # share survives quantize
    # No _tied_weights_keys declared, so name-based grouping finds nothing to merge.
    assert TiedWeightMap(parent).group_key("a.weight") is None

    parent.a.input_quantizer.amax = torch.tensor(2.0)
    parent.b.input_quantizer.amax = torch.tensor(8.0)

    sync_tied_input_amax(parent)

    expected = torch.tensor(8.0)
    assert torch.allclose(parent.a.input_quantizer.amax, expected)
    assert torch.allclose(parent.b.input_quantizer.amax, expected)


def test_postprocess_name_based_drops_alias_across_distinct_addresses():
    """Declared alias dropped by name even when its tensor is at a different address (the FSDP full_state_dict case)."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    tied_map = TiedWeightMap(parent)

    # Distinct storages (different data_ptr): the address pass could never collapse these.
    sd = {"encoder.weight": torch.randn(4, 4), "decoder.weight": torch.randn(4, 4)}
    assert sd["encoder.weight"].data_ptr() != sd["decoder.weight"].data_ptr()

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert "decoder.weight" in out  # canonical kept
    assert "encoder.weight" not in out  # alias dropped by name


def test_postprocess_name_based_keeps_alias_when_canonical_absent():
    """An alias is NOT dropped when its canonical counterpart is missing (no orphaning)."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    tied_map = TiedWeightMap(parent)

    sd = {"encoder.weight": torch.randn(4, 4)}  # canonical decoder.weight absent
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert "encoder.weight" in out


def test_postprocess_name_based_keeps_both_sides_of_bidirectional_tie():
    """A bidirectional declaration (A<->B) has no clear canonical, so the map is empty and neither side is dropped."""

    class _Bi(torch.nn.Module):
        _tied_weights_keys = {r"^a\.weight$": "b.weight", r"^b\.weight$": "a.weight"}

        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, bias=False)
            self.b = torch.nn.Linear(4, 4, bias=False)
            self.b.weight = self.a.weight  # genuinely shared, declared both ways

    tied_map = TiedWeightMap(_Bi())
    # Both sides are declared aliases -> no unambiguous canonical -> not deduped.
    assert tied_map.alias_to_canonical == {}

    sd = {"a.weight": torch.randn(4, 4), "b.weight": torch.randn(4, 4)}
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert "a.weight" in out and "b.weight" in out  # neither side dropped


def test_postprocess_keeps_both_sides_when_tied_quant_state_differs():
    """Tied sides with differing quant state aren't deduped (atomic drop), so no scale is orphaned."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    tied_map = TiedWeightMap(parent)

    sd = {
        # alias (encoder) exported as quantized: weight + companion scales
        "encoder.weight": torch.randn(4, 4),
        "encoder.weight_scale": torch.randn(4),
        "encoder.input_scale": torch.randn(1),
        # canonical (decoder) exported unquantized: weight only, no scales
        "decoder.weight": torch.randn(4, 4),
    }

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

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
    tied_map = TiedWeightMap(parent)
    assert tied_map.alias_to_canonical == {
        "encoder.experts.gate_up_proj": "decoder.experts.gate_up_proj",
        "encoder.experts.down_proj": "decoder.experts.down_proj",
    }

    # Craft exported-style per-expert keys with distinct storages on both sides.
    sd = {}
    for side in ("encoder", "decoder"):
        for e in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                sd[f"{side}.experts.{e}.{proj}.weight"] = torch.randn(4, 4)
                sd[f"{side}.experts.{e}.{proj}.weight_scale"] = torch.randn(4)

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert not any(k.startswith("encoder.experts.") for k in out)  # all aliases dropped
    assert all(k.startswith("decoder.experts.") for k in out)  # only canonical remains
    assert len(out) == 2 * 3 * 2  # 2 experts * 3 projections * (weight + weight_scale)


def test_postprocess_keeps_independent_bias_under_tied_weight():
    """A weight tie must not drop an independent bias sharing the module prefix (the NVBug 6525352 failure class)."""

    class _TwoLinear(torch.nn.Module):
        _tied_weights_keys = {r"^A\.weight$": "B.weight"}

        def __init__(self):
            super().__init__()
            self.A = torch.nn.Linear(4, 4, bias=True)
            self.B = torch.nn.Linear(4, 4, bias=True)
            self.A.weight = self.B.weight  # weights tied; biases independent

    tied_map = TiedWeightMap(_TwoLinear())
    sd = {
        "A.weight": torch.randn(4, 4),
        "A.bias": torch.randn(4),
        "B.weight": torch.randn(4, 4),
        "B.bias": torch.randn(4),
    }
    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert "A.weight" not in out  # tied weight dropped
    assert "A.bias" in out  # independent bias survives
    assert "B.weight" in out and "B.bias" in out


def test_postprocess_partially_tied_container_dedups_only_tied_projections():
    """Only the tied projection's per-expert keys are deduped; an untied down_proj and a router child survive."""

    class _Parent(torch.nn.Module):
        _tied_weights_keys = {r"^encoder\.experts\.gate_up_proj$": "decoder.experts.gate_up_proj"}

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.experts = torch.nn.Module()
            self.decoder = torch.nn.Module()
            self.decoder.experts = torch.nn.Module()
            gup = torch.nn.Parameter(torch.zeros(2, 4, 4))
            self.decoder.experts.gate_up_proj = gup
            self.encoder.experts.gate_up_proj = gup  # tied
            # down_proj is a distinct Parameter on each side (untied)
            self.decoder.experts.down_proj = torch.nn.Parameter(torch.zeros(2, 4, 4))
            self.encoder.experts.down_proj = torch.nn.Parameter(torch.zeros(2, 4, 4))

    tied_map = TiedWeightMap(_Parent())
    assert tied_map.alias_to_canonical == {
        "encoder.experts.gate_up_proj": "decoder.experts.gate_up_proj"
    }

    sd = {}
    for side in ("encoder", "decoder"):
        for e in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                sd[f"{side}.experts.{e}.{proj}.weight"] = torch.randn(4, 4)
        sd[f"{side}.experts.router.weight"] = torch.randn(4, 4)  # non-projection child

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    # Tied gate_up_proj (splits to gate_proj/up_proj) is deduped on the encoder (alias) side.
    assert not any(".gate_proj." in k or ".up_proj." in k for k in out if k.startswith("encoder."))
    # Untied down_proj and the router survive on both sides.
    assert all(f"encoder.experts.{e}.down_proj.weight" in out for e in range(2))
    assert "encoder.experts.router.weight" in out and "decoder.experts.router.weight" in out
    # Decoder (canonical) side fully kept.
    assert all(
        f"decoder.experts.{e}.{p}.weight" in out
        for e in range(2)
        for p in ("gate_proj", "up_proj", "down_proj")
    )


def test_postprocess_backstop_collapses_keys_sharing_a_dataptr():
    """The address backstop drops a later key that shares a ``data_ptr`` with an earlier one (first-wins)."""
    storage = torch.arange(4)
    sd = {"short": storage[:2], "long": storage}  # both start at offset 0 -> same data_ptr
    assert sd["short"].data_ptr() == sd["long"].data_ptr()

    out = postprocess_state_dict(sd, maxbound=448, quantization=None)

    assert len(out) == 1 and "short" in out  # first-seen kept, later collision dropped


def test_postprocess_backstop_keeps_keys_with_distinct_dataptrs():
    """Two slices at different offsets have distinct ``data_ptr``s, so the backstop leaves both."""
    base = torch.arange(4)
    sd = {"first": base[:2], "second": base[2:]}  # offsets 0 and 2 -> different data_ptr
    assert sd["first"].data_ptr() != sd["second"].data_ptr()

    out = postprocess_state_dict(sd, maxbound=448, quantization=None)

    assert set(out) == {"first", "second"}  # neither dropped
    assert torch.equal(out["first"], torch.tensor([0, 1]))
    assert torch.equal(out["second"], torch.tensor([2, 3]))


def test_postprocess_dense_tie_drops_pre_quant_scale_companion():
    """An AWQ-style dense tie drops ``pre_quant_scale`` with the weight (no orphaned companion)."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    tied_map = TiedWeightMap(parent)
    sd = {
        "encoder.weight": torch.randn(4, 4),
        "encoder.pre_quant_scale": torch.randn(4),
        "decoder.weight": torch.randn(4, 4),
        "decoder.pre_quant_scale": torch.randn(4),
    }

    out = postprocess_state_dict(sd, maxbound=448, quantization=None, tied_map=tied_map)

    assert "encoder.weight" not in out and "encoder.pre_quant_scale" not in out  # both dropped
    assert "decoder.weight" in out and "decoder.pre_quant_scale" in out  # canonical kept


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
