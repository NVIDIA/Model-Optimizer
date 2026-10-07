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

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionConfig,
    linear_attention_training_phase,
)
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins import gdn
from modelopt.torch.quantization.plugins.gdn import GatedDeltaNetStateQuantMixin
from modelopt.torch.quantization.plugins.kda import KimiDeltaAttentionStateQuantMixin

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


@pytest.mark.parametrize(
    "attributes",
    [
        {"num_bits": (4, 3), "axis": (0, 1)},  # static
        {"num_bits": (4, 3), "type": "dynamic"},  # per tensor
        {"num_bits": 8, "axis": (0, 1), "type": "dynamic"},  # int8
        {"num_bits": (4, 3), "type": "dynamic", "block_sizes": {-1: 16}},  # blockwise
    ],
)
def test_validate_state_quantizer_rejects_unsupported(attributes):
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(
            TinyGatedDeltaNet(),
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {"quantizer_name": "*gdn_state_quantizer", "cfg": attributes},
                ],
                "algorithm": None,
            },
        )


def test_dynamic_export_removes_linear_attention_attributes():
    model = _QuantTinyGatedDeltaNet.convert(TinyGatedDeltaNet())
    model.export()
    assert type(model) is TinyGatedDeltaNet
    for name in (
        "gdn_state_quantizer",
        "gdn_w_quantizer",
        "linear_attention_config",
        "_linear_attention_prefill_lengths",
    ):
        assert not hasattr(model, name)


def test_disabled_state_quantizer_calls_original_kernel():
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    expected = model(x)

    disable_all = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "max"}
    mtq.quantize(model, disable_all, lambda m: m(x))

    assert isinstance(model, _QuantTinyGatedDeltaNet)
    assert not model.gdn_state_quantizer.is_enabled and not model.gdn_w_quantizer.is_enabled
    assert torch.equal(model(x), expected)
    assert model.gated_delta_rule is chunk_gated_delta_rule, "the kernel swap must be undone"


@pytest.mark.parametrize("mixin", [GatedDeltaNetStateQuantMixin, KimiDeltaAttentionStateQuantMixin])
def test_state_qat_requires_explicit_serving_policy(mixin):
    model = mixin.convert(TinyGatedDeltaNet())
    model._linear_attn_state.set_from_attribute_config(GDN_STATE_FP8_DYNAMIC)
    model._linear_attn_state.enable()
    with pytest.raises(ValueError, match="requires backend='serving'"):
        model.validate_linear_attention()


def test_training_phase_routes_prefill_lengths(monkeypatch):
    cfg = {**quant_cfg(), "algorithm": None}
    missing_policy = {key: value for key, value in cfg.items() if key != "linear_attention"}
    with pytest.raises(ValueError, match="requires backend='serving'"):
        mtq.quantize(TinyGatedDeltaNet(), missing_policy)
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    x = torch.randn(2, 8, 3, 4)
    calls = []

    def forward(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(gdn, "gdn_state_qat", forward)
    with linear_attention_training_phase(model, [4, 4]):
        model(x)
    assert calls[-1]["prefill_lengths"] == (4, 4)


@pytest.mark.parametrize("phase", ["convert", "forward", "restore"])
def test_enabled_legacy_w_quantizer_is_rejected(phase):
    cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    with pytest.raises(ValueError, match="GDN W quantization is no longer supported"):
        if phase == "convert":
            cfg["quant_cfg"].append({"quantizer_name": "*gdn_w_quantizer", "enable": True})
            mtq.quantize(TinyGatedDeltaNet(), cfg)
        else:
            model.gdn_w_quantizer.enable()
            if phase == "forward":
                model(torch.randn(2, 8, 3, 4))
            else:
                mto.restore_from_modelopt_state(TinyGatedDeltaNet(), mto.modelopt_state(model))


@pytest.mark.parametrize(
    "overrides",
    [{"pass_through_bwd": False}, {"type": "static"}, {"fake_quant": False}, {"rotate": True}],
)
def test_unsupported_quantizer_fails_during_conversion(overrides):
    cfg = deepcopy(quant_cfg())
    cfg["quant_cfg"][-1]["cfg"].update(overrides)
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(TinyGatedDeltaNet(), cfg)


def test_quantizer_roundtrip_and_hybrid_selection(tmp_path):
    model = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    cfg = deepcopy(quant_cfg())
    cfg["algorithm"] = None
    cfg["quant_cfg"][-1]["cfg"].update(
        num_bits=8, axis=None, block_sizes={-1: 32}, unsigned=False, narrow_range=True
    )
    cfg["linear_attention"] = [
        {"module_name": "*", "cfg": {"backend": "serving", "state_block_v": 128}},
        {"module_name": "0", "cfg": {"backend": "serving", "state_block_v": 32}},
    ]
    mtq.quantize(model, cfg)
    assert model[0].gdn_state_qdq_block_v == 32
    model[0].linear_attention_config.state_block_v = 16
    sample = torch.randn(2, 4, 19)
    expected = model[0].gdn_state_quantizer(sample)
    path = tmp_path / "gdn.pth"
    mto.save(model, path)
    restored = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    mto.restore(restored, path)
    assert restored[0].linear_attention_config == model[0].linear_attention_config
    assert restored[0].gdn_state_qdq_block_v == 16
    for name, enabled in (("gdn_state_quantizer", True), ("gdn_w_quantizer", False)):
        original = getattr(model[0], name)
        quantizer = getattr(restored[0], name)
        assert quantizer.is_enabled == enabled
        assert quantizer.axis == original.axis
        assert quantizer.num_bits == original.num_bits
        assert quantizer.block_sizes == original.block_sizes
        assert quantizer._dynamic == original._dynamic
        assert not hasattr(restored[1], name)
    torch.testing.assert_close(restored[0].gdn_state_quantizer(sample), expected)


@pytest.mark.parametrize("reverse", [False, True], ids=["fp8-to-replay", "replay-to-fp8"])
def test_restore_changed_policy_and_quantizer_together(tmp_path, reverse):
    states = [
        (GDN_STATE_FP8_DYNAMIC, LinearAttentionConfig(backend="serving")),
        (
            {"num_bits": 8, "axis": (0, 1), "type": "dynamic", "narrow_range": True},
            LinearAttentionConfig(backend="serving", precision="replayssm"),
        ),
    ]
    if reverse:
        states.reverse()
    (original_quantizer, original_policy), (saved_quantizer, saved_policy) = states
    cfg = quant_cfg()
    cfg["algorithm"] = None
    cfg["quant_cfg"][-1]["cfg"] = original_quantizer
    cfg["linear_attention"][0]["cfg"] = original_policy.model_dump()
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    model.gdn_state_quantizer.set_from_attribute_config(saved_quantizer)
    model.linear_attention_config = saved_policy
    model.validate_linear_attention()

    path = tmp_path / "changed-policy.pth"
    mto.save(model, path)
    restored = mto.restore(TinyGatedDeltaNet(), path)
    assert restored.linear_attention_config == saved_policy
    assert restored.gdn_state_quantizer.num_bits == saved_quantizer["num_bits"]
    assert restored.gdn_state_quantizer.is_enabled
    restored.validate_linear_attention()


def test_quant_cfg_refinement_updates_and_validates_existing_quantized_module():
    cfg = quant_cfg()
    cfg["algorithm"] = None
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    assert model.gdn_state_quantizer.is_enabled
    assert not model.gdn_w_quantizer.is_enabled

    cfg = deepcopy(quant_cfg())
    cfg["algorithm"] = None
    cfg["linear_attention"] = [
        {"module_name": "", "cfg": {"backend": "serving", "state_block_v": 32}}
    ]
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 32
    cfg["linear_attention"].append({"module_name": "", "cfg": {"backend": "serving"}})
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 64
    assert model.gdn_state_quantizer.is_enabled
    assert not model.gdn_w_quantizer.is_enabled

    cfg = deepcopy(cfg)
    cfg["quant_cfg"][-1]["cfg"]["pass_through_bwd"] = False
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(model, cfg)


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


def test_policy_rejects_unmatched_and_unimplemented_modes():
    with pytest.raises(ValueError, match="matches no supported"):
        mtq.quantize(
            nn.Linear(4, 4), {"linear_attention": [{"module_name": "*"}], "algorithm": None}
        )
    for policy in (
        {"chunk_size": 32},
        {"solve": {"method": "neumann"}},
        {"state": {"mode": "token"}},
    ):
        with pytest.raises(ValueError):
            QuantizeConfig(linear_attention=[{"module_name": "*", "cfg": policy}])
