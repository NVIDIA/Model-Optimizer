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
    _collect_canonical_tied_patterns,
    _reorder_canonical_first,
)
from modelopt.torch.export.quant_utils import fuse_prequant_layernorm, sync_tied_input_amax
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
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


def _calibrate_through_both_children(parent):
    """Insert NVFP4 quantizers and run a one-shot forward through both children for calibration."""

    def forward_loop(m):
        x = torch.randn(2, 16)
        m.encoder(x)
        m.decoder(x)

    mtq.quantize(parent, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)


def test_export_quantized_weight_aliases_packed_weight_for_tied_linears():
    """Tied Linears share data_ptr for packed .weight and scale buffers after export."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec)
    _calibrate_through_both_children(parent)

    # Per-call dedup cache (the production pattern: caller owns the cache, scoped
    # to one export invocation). Threaded through both sides of the tied pair so
    # the alias step at the end of _export_quantized_weight catches the dedup.
    tied_cache: dict = {}
    _export_quantized_weight(enc, torch.float16, "weight", _tied_cache=tied_cache)
    _export_quantized_weight(dec, torch.float16, "weight", _tied_cache=tied_cache)

    assert enc.weight.data_ptr() == dec.weight.data_ptr()
    for scale_attr in ("weight_scale", "weight_scale_2"):
        if hasattr(enc, scale_attr) and hasattr(dec, scale_attr):
            assert getattr(enc, scale_attr).data_ptr() == getattr(dec, scale_attr).data_ptr()


def test_export_quantized_weight_no_alias_for_untied_linears():
    """Untied Linears keep independent data_ptrs after export — no false-positive aliasing."""
    parent = torch.nn.Module()
    parent.encoder = torch.nn.Linear(16, 32, bias=False)
    parent.decoder = torch.nn.Linear(16, 32, bias=False)
    assert parent.encoder.weight.data_ptr() != parent.decoder.weight.data_ptr()
    _calibrate_through_both_children(parent)

    # Same fresh cache shape as the positive case — confirms that even with
    # dedup enabled, untied modules with distinct source data_ptrs do not get
    # falsely aliased.
    tied_cache: dict = {}
    _export_quantized_weight(parent.encoder, torch.float16, "weight", _tied_cache=tied_cache)
    _export_quantized_weight(parent.decoder, torch.float16, "weight", _tied_cache=tied_cache)

    assert parent.encoder.weight.data_ptr() != parent.decoder.weight.data_ptr()


def test_export_quantized_weight_skips_alias_when_one_tied_side_is_unquantized():
    """Unquantized side early-returns; its .weight stays at the original shared Parameter."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec)
    original_shared_data_ptr = enc.weight.data_ptr()

    _calibrate_through_both_children(parent)
    # is_enabled is a read-only property; .disable() is the canonical bypass.
    dec.weight_quantizer.disable()

    tied_cache: dict = {}
    _export_quantized_weight(enc, torch.float16, "weight", _tied_cache=tied_cache)
    _export_quantized_weight(dec, torch.float16, "weight", _tied_cache=tied_cache)

    assert enc.weight.data_ptr() != original_shared_data_ptr  # encoder got fresh packed
    assert dec.weight.data_ptr() == original_shared_data_ptr  # decoder untouched
    assert enc.weight.data_ptr() != dec.weight.data_ptr()


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
