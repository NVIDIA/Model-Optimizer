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
from modelopt.torch.export.model_utils import _build_tied_weight_map
from modelopt.torch.export.quant_utils import (
    fuse_prequant_layernorm,
    postprocess_state_dict,
    sync_tied_input_amax,
)
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.nn import TensorQuantizer


def test_build_tied_weight_map_uses_parameter_identity_and_declared_direction():
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)

    assert parent.encoder.weight is parent.decoder.weight
    assert _build_tied_weight_map(parent) == {"encoder.weight": "decoder.weight"}


def test_build_tied_weight_map_does_not_apply_declared_tie():
    parent = wrap_in_parent_with_tied_keys(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )

    assert parent.encoder.weight is not parent.decoder.weight
    assert _build_tied_weight_map(parent) == {}


def test_build_tied_weight_map_prefers_input_embedding():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"tie_word_embeddings": True})()
            self.lm_head = torch.nn.Linear(4, 4, bias=False)
            self.embed = torch.nn.Embedding(4, 4)
            self.embed.weight = self.lm_head.weight

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

    assert _build_tied_weight_map(Model()) == {"lm_head.weight": "embed.weight"}


def test_postprocess_tied_weights_uses_pre_export_names_after_materialization():
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    tied_weight_map = _build_tied_weight_map(parent)
    state_dict = {
        "encoder.weight": parent.encoder.weight.detach().clone(),
        "decoder.weight": parent.decoder.weight.detach().clone(),
    }

    processed = postprocess_state_dict(state_dict, 448, None, tied_weight_map=tied_weight_map)

    assert set(processed) == {"decoder.weight"}


def test_postprocess_tied_weights_drops_quantization_companions_atomically():
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    state_dict = {}
    for side in ("encoder", "decoder"):
        state_dict[f"{side}.weight"] = torch.randn(4, 4)
        state_dict[f"{side}.weight_scale"] = torch.randn(4)
        state_dict[f"{side}.weight_scale_2"] = torch.randn(1)
        state_dict[f"{side}.input_scale"] = torch.randn(1)

    processed = postprocess_state_dict(
        state_dict, 448, None, tied_weight_map=_build_tied_weight_map(parent)
    )

    assert set(processed) == {
        "decoder.weight",
        "decoder.weight_scale",
        "decoder.weight_scale_2",
        "decoder.input_scale",
    }


def test_postprocess_tied_weights_keeps_group_if_canonical_companion_is_missing():
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec, decoder_canonical=True)
    state_dict = {
        "encoder.weight": torch.randn(4, 4),
        "encoder.weight_scale": torch.randn(4),
        "decoder.weight": torch.randn(4, 4),
    }

    processed = postprocess_state_dict(
        state_dict, 448, None, tied_weight_map=_build_tied_weight_map(parent)
    )

    assert set(processed) == set(state_dict)


def test_postprocess_tied_weights_drops_expanded_moe_expert_aliases():
    parent = torch.nn.Module()
    parent.encoder = torch.nn.Module()
    parent.encoder.experts = torch.nn.Module()
    parent.decoder = torch.nn.Module()
    parent.decoder.experts = torch.nn.Module()
    parent.decoder.experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 4))
    parent.decoder.experts.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))
    parent.encoder.experts.gate_up_proj = parent.decoder.experts.gate_up_proj
    parent.encoder.experts.down_proj = parent.decoder.experts.down_proj
    parent._tied_weights_keys = {
        r"^encoder\.experts\.gate_up_proj$": "decoder.experts.gate_up_proj",
        r"^encoder\.experts\.down_proj$": "decoder.experts.down_proj",
    }
    state_dict = {}
    for side in ("encoder", "decoder"):
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{side}.experts.{expert}.{projection}"
                state_dict[f"{prefix}.weight"] = torch.randn(4, 4)
                state_dict[f"{prefix}.weight_scale"] = torch.randn(4)

    processed = postprocess_state_dict(
        state_dict, 448, None, tied_weight_map=_build_tied_weight_map(parent)
    )

    assert len(processed) == 12
    assert all(key.startswith("decoder.experts.") for key in processed)


def test_postprocess_state_dict_has_no_pointer_or_storage_fallback():
    shared = torch.randn(4)
    state_dict = {"first": shared, "second": shared, "view": shared[:2]}

    processed = postprocess_state_dict(state_dict, 448, None)

    assert set(processed) == set(state_dict)


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


def test_export_quantized_weight_packs_tied_linears_independently():
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec)
    _calibrate_through_both_children(parent)

    _export_quantized_weight(enc, torch.float16, "weight")
    _export_quantized_weight(dec, torch.float16, "weight")

    assert enc.weight is not dec.weight
    assert torch.equal(enc.weight, dec.weight)


def test_export_quantized_weight_skips_alias_when_one_tied_side_is_unquantized():
    """Unquantized side early-returns; its .weight stays at the original shared Parameter."""
    enc, dec = make_tied_linear_pair()
    parent = wrap_in_parent_with_tied_keys(enc, dec)
    original_shared_weight = enc.weight

    _calibrate_through_both_children(parent)
    # is_enabled is a read-only property; .disable() is the canonical bypass.
    dec.weight_quantizer.disable()

    _export_quantized_weight(enc, torch.float16, "weight")
    _export_quantized_weight(dec, torch.float16, "weight")

    assert enc.weight is not original_shared_weight
    assert dec.weight is original_shared_weight
    assert enc.weight is not dec.weight


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
