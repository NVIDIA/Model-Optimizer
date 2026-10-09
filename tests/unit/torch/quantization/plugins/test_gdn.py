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

import numpy as np
import pytest
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    get_linear_attention_layers,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins import gdn
from modelopt.torch.quantization.plugins.gdn import GatedDeltaNetStateQuantMixin
from modelopt.torch.quantization.plugins.kda import KimiDeltaAttentionStateQuantMixin
from modelopt.torch.quantization.plugins.linear_attention import _validate_linear_attention

GDN_STATE_FP8_DYNAMIC = {"num_bits": (4, 3), "axis": (0, 1), "type": "dynamic"}


def chunk_gated_delta_rule(q, k, v, g, beta, **kwargs):
    """CPU stand-in for the optional FLA kernel."""
    return q + k + v, None


@pytest.fixture(autouse=True)
def mock_fla_kernel(monkeypatch):
    monkeypatch.setattr(gdn, "_fla_chunk_gated_delta_rule", lambda: chunk_gated_delta_rule)


class TinyGatedDeltaNet(nn.Module):
    """A module that, like Megatron-Core's GatedDeltaNet, calls ``self.gated_delta_rule``."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.gated_delta_rule = chunk_gated_delta_rule

    def forward(self, x):
        out, _ = self.gated_delta_rule(x, x, x, x[..., 0], x[..., 0])
        return self.proj(out)


@QuantModuleRegistry.register({TinyGatedDeltaNet: "TinyGatedDeltaNet"})
class _QuantTinyGatedDeltaNet(GatedDeltaNetStateQuantMixin):
    def forward(self, x):
        gated_delta_rule = self.gated_delta_rule
        self.gated_delta_rule = lambda *a, **kw: self._state_quantized_chunk_gated_delta_rule(
            gated_delta_rule, *a, **kw
        )
        try:
            return super().forward(x)
        finally:
            self.gated_delta_rule = gated_delta_rule


def quant_cfg():
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*gdn_state_quantizer", "cfg": GDN_STATE_FP8_DYNAMIC},
        ],
        "linear_attention": [{"module_name": "*", "cfg": {"backend": "serving"}}],
        "algorithm": "max",
    }


def test_disabled_quantization_preserves_baseline():
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    expected = model(x)
    cfg = quant_cfg()
    cfg["quant_cfg"] = [{"quantizer_name": "*", "enable": False}]
    mtq.quantize(model, cfg, lambda m: m(x))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert model.gated_delta_rule is chunk_gated_delta_rule
    assert not model.linear_attention_is_enabled


def test_phase_routes_lengths_and_preserves_outer_context(monkeypatch):
    monkeypatch.setattr(
        _QuantTinyGatedDeltaNet,
        "validate_linear_attention_execution",
        lambda _: pytest.fail("Conversion must not validate runtime execution settings"),
    )
    cfg = {**quant_cfg(), "algorithm": None}
    with pytest.raises(ValueError, match="requires backend='serving'"):
        mtq.quantize(TinyGatedDeltaNet(), {k: v for k, v in cfg.items() if k != "linear_attention"})
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    calls = []

    def forward(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(gdn, "gdn_state_qat", forward)
    with (
        linear_attention_training_phase(
            model, np.array([4, 4]), sequence_lengths=[7, 6], cu_seqlens=[0, 8, 16]
        ),
        linear_attention_training_phase(model, [1, 1], preserve_existing=True),
    ):
        model(torch.randn(2, 8, 3, 4))
    assert calls[-1]["prefill_lengths"] == (4, 4)
    assert calls[-1]["sequence_lengths"] == (7, 6)
    boundaries = calls[-1]["cu_seqlens_cpu"]
    actual = torch.tensor([0, 8, 16])
    assert boundaries.resolve(actual) == (0, 8, 16)
    actual.data[1] = 7  # Alias writes do not bump the tensor version.
    with pytest.raises(ValueError, match="does not match"):
        boundaries.resolve(actual)
    assert model._linear_attention_prefill_lengths is None
    assert model._linear_attention_sequence_lengths is None
    mtq.disable_quantizer(model, "*")
    assert get_linear_attention_layers(model) == (model,)
    assert not get_linear_attention_layers(model, enabled_only=True)
    with linear_attention_training_phase(model, [4, 4]):
        assert not get_linear_attention_layers(model, unphased_only=True)
        assert model.linear_attention_is_enabled
    assert not model.linear_attention_is_enabled


def test_restore_uses_saved_quantizer_and_policy(tmp_path, monkeypatch):
    model = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    mtq.quantize(model, {**quant_cfg(), "algorithm": None})
    # Change both after conversion: restore must not validate one against stale recipe defaults.
    model[0].gdn_state_quantizer.set_from_attribute_config(
        {"num_bits": 8, "axis": (0, 1), "type": "dynamic", "narrow_range": True}
    )
    model[0].linear_attention_config = LinearAttentionConfig(
        backend="serving", precision="replayssm", replay_window=4
    )
    sample = torch.randn(2, 4, 19)
    expected = model[0].gdn_state_quantizer(sample)
    path = tmp_path / "gdn.pth"
    mto.save(model, path)

    def no_rematching(*args):
        pytest.fail("Restore must use saved resolved policies")

    monkeypatch.setattr(
        "modelopt.torch.quantization.plugins.linear_attention._apply_linear_attention_policy",
        no_rematching,
    )
    restored = mto.restore(nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4)), path)
    assert restored[0].linear_attention_config == model[0].linear_attention_config
    assert restored[0].gdn_state_quantizer.is_enabled
    assert not restored[0].gdn_w_quantizer.is_enabled
    assert not hasattr(restored[1], "gdn_state_quantizer")
    torch.testing.assert_close(restored[0].gdn_state_quantizer(sample), expected)


@pytest.mark.parametrize("mixin", [GatedDeltaNetStateQuantMixin, KimiDeltaAttentionStateQuantMixin])
def test_state_qat_requires_serving_policy(mixin):
    model = mixin.convert(TinyGatedDeltaNet())
    model._linear_attn_state.set_from_attribute_config(GDN_STATE_FP8_DYNAMIC)
    model._linear_attn_state.enable()
    with pytest.raises(ValueError, match="requires backend='serving'"):
        model.validate_linear_attention()
    model.linear_attention_config = LinearAttentionConfig(backend="serving")
    if mixin is KimiDeltaAttentionStateQuantMixin:
        with pytest.warns(UserWarning, match="standalone FLA/Triton"):
            _validate_linear_attention(model)
        with pytest.warns(UserWarning, match="standalone FLA/Triton"):
            model.modelopt_post_restore()


@pytest.mark.parametrize("overrides", [{"pass_through_bwd": False}, {"type": "static"}])
def test_unsupported_quantizer_fails_at_conversion(overrides):
    cfg = quant_cfg()
    cfg["algorithm"] = None
    cfg["quant_cfg"][-1]["cfg"] = {**GDN_STATE_FP8_DYNAMIC, **overrides}
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(TinyGatedDeltaNet(), cfg)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefill_lengths": [-1]}, "Prefill lengths"),
        ({"prefill_lengths": [0], "sequence_lengths": [True]}, "Sequence lengths"),
        ({"prefill_lengths": [2], "sequence_lengths": [1]}, "prefill length <= sequence length"),
        ({"prefill_lengths": [0, 0], "sequence_lengths": [1]}, "per sequence"),
    ],
)
def test_invalid_phase_lengths(kwargs, message):
    with (
        pytest.raises(ValueError, match=message),
        linear_attention_training_phase(nn.Identity(), **kwargs),
    ):
        pass


def test_restore_legacy_gdn_without_new_quantizer_handles():
    config = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    model = mtq.quantize(TinyGatedDeltaNet(), config)
    state = mto.modelopt_state(model)
    for _, mode_state in state["modelopt_state_dict"]:
        mode_state["config"].pop("linear_attention", None)
        metadata = mode_state["metadata"]
        metadata.pop("linear_attention", None)
        for name in ("gdn_state_quantizer", "gdn_w_quantizer"):
            metadata["quantizer_state"].pop(name, None)
    restored = TinyGatedDeltaNet()
    mto.restore_from_modelopt_state(restored, state)
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 8, 3, 4)
    torch.testing.assert_close(restored(x), model(x))
    assert not restored.gdn_state_quantizer.is_enabled
    assert not restored.gdn_w_quantizer.is_enabled


def test_standard_projection_recipe_leaves_gdn_emulation_disabled():
    model = mtq.quantize(
        TinyGatedDeltaNet(), mtq.FP8_DEFAULT_CFG, lambda m: m(torch.randn(2, 8, 3, 4))
    )
    assert not model.gdn_state_quantizer.is_enabled
    assert not model.gdn_w_quantizer.is_enabled


def test_empty_stage_conversion_and_offline_restore(monkeypatch):
    cfg = {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}],
        "linear_attention": [{"module_name": "layer", "cfg": {"backend": "serving"}}],
        "algorithm": None,
    }
    model = mtq.quantize(nn.ModuleDict({"layer": nn.Linear(4, 4)}), cfg)
    with linear_attention_training_phase(model, [3]):
        pass
    state = mto.modelopt_state(model)

    # Saved resolved policies must restore without evaluating the original global recipe.
    def no_rematching(*args):
        pytest.fail("Restore must use saved resolved policies")

    monkeypatch.setattr(
        "modelopt.torch.quantization.plugins.linear_attention._apply_linear_attention_policy",
        no_rematching,
    )
    restored = nn.ModuleDict({"layer": nn.Linear(4, 4)})
    mto.restore_from_modelopt_state(restored, state)
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored["layer"](x), model["layer"](x))


def test_enabled_legacy_w_quantizer_is_rejected_on_restore():
    cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    model.gdn_w_quantizer.enable()
    with pytest.raises(ValueError, match="GDN W quantization is no longer supported"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNet(), mto.modelopt_state(model))


def test_quant_cfg_refinement_and_policy_precedence():
    cfg = {**quant_cfg(), "algorithm": None}
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    cfg["linear_attention"] = [
        {"module_name": "*", "cfg": {"backend": "serving", "state_block_v": 128}},
        {"module_name": "", "cfg": {"backend": "serving", "state_block_v": 32}},
    ]
    assert mtq.quantize(model, cfg) is model
    assert model.gdn_state_qdq_block_v == 32
    cfg["linear_attention"].reverse()
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 128
    cfg["quant_cfg"][-1]["cfg"] = {**GDN_STATE_FP8_DYNAMIC, "pass_through_bwd": False}
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(model, cfg)
