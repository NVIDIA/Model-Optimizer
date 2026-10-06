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


GDN_W_FP8_DYNAMIC = {"num_bits": (4, 3), "axis": (0, 1, 2), "type": "dynamic"}


def quant_cfg(state=True, w=False):
    entries = [{"quantizer_name": "*", "enable": False}]
    if state:
        entries.append({"quantizer_name": "*gdn_state_quantizer", "cfg": GDN_STATE_FP8_DYNAMIC})
    if w:
        entries.append({"quantizer_name": "*gdn_w_quantizer", "cfg": GDN_W_FP8_DYNAMIC})
    return {"quant_cfg": entries, "algorithm": "max"}


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
        "replay_key_quantizer",
        "replay_update_quantizer",
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


def test_state_qat_requires_explicit_serving_policy(monkeypatch):
    model = mtq.quantize(TinyGatedDeltaNet(), {**quant_cfg(), "algorithm": None})
    x = torch.randn(2, 8, 3, 4)
    with pytest.raises(ValueError, match="Chunk-only state QAT is retired"):
        model(x)
    calls = []

    def forward(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(gdn, "matmul_gdn", forward)
    model.linear_attention_config = LinearAttentionConfig(backend="serving", decode={})
    with linear_attention_training_phase(model, [4, 4]):
        model(x)
    assert calls[-1]["prefill_lengths"] == (4, 4)


def test_w_quantizer_is_passed_to_the_kernel(monkeypatch):
    """Pass the configured W TensorQuantizer to the FLA wrapper."""
    calls = []

    def fake_w_qdq_kernel(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(gdn, "_w_qdq_chunk_gated_delta_rule", lambda: fake_w_qdq_kernel)
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    mtq.quantize(model, quant_cfg(state=False, w=True), lambda m: m(x))
    assert model.gdn_w_quantizer.is_enabled and not model.gdn_state_quantizer.is_enabled

    model(x)
    assert calls[-1]["w_quantizer"] is model.gdn_w_quantizer

    # The w quantizer really quantizes: 256 random values per token collapse onto the E4M3 grid,
    # which has at most 127 distinct magnitudes per (row-specific) scale.
    w = torch.randn(1, 1, 1, 256)
    quantized = model.gdn_w_quantizer(w)
    assert not torch.equal(quantized, w)
    assert torch.unique(quantized.abs()).numel() <= 127 < torch.unique(w.abs()).numel()


@pytest.mark.parametrize("site", ["state", "w"])
@pytest.mark.parametrize(
    "overrides",
    [{"pass_through_bwd": False}, {"type": "static"}, {"fake_quant": False}, {"rotate": True}],
)
def test_unsupported_quantizer_fails_during_conversion(site, overrides):
    cfg = quant_cfg(state=site == "state", w=site == "w")
    cfg = deepcopy(cfg)
    cfg["quant_cfg"][-1]["cfg"].update(overrides)
    with pytest.raises(ValueError, match="supports only"):
        mtq.quantize(TinyGatedDeltaNet(), cfg)


@pytest.mark.parametrize("axis", [None, (0, 1), (0, 1, 2)])
def test_w_grouping_is_preserved_during_conversion(axis):
    cfg = deepcopy(quant_cfg(state=False, w=True))
    cfg["algorithm"] = None
    cfg["quant_cfg"][-1]["cfg"]["axis"] = axis
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    assert model.gdn_w_quantizer.axis == axis


@pytest.mark.parametrize(("state", "w"), [(True, False), (False, True), (True, True)])
def test_quantizer_roundtrip_and_hybrid_selection(tmp_path, state, w):
    model = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    cfg = deepcopy(quant_cfg(state=state, w=w))
    cfg["algorithm"] = None
    if state and not w:
        cfg["quant_cfg"][-1]["cfg"].update(
            num_bits=8, axis=None, block_sizes={-1: 32}, unsigned=False, narrow_range=True
        )
    cfg["linear_attention"] = [
        {"module_name": "*", "cfg": {"state": {"block_v": 128}}},
        {"module_name": "0", "cfg": {"state": {"block_v": 32}}},
    ]
    mtq.quantize(model, cfg)
    assert model[0].gdn_state_qdq_block_v == 32
    model[0].linear_attention_config.state.block_v = 16
    if state and not w:
        sample = torch.randn(2, 4, 19)
        expected = model[0].gdn_state_quantizer(sample)
    if w:
        # Save the actual quantizer settings, including edits after conversion.
        model[0].gdn_w_quantizer.axis = None
    path = tmp_path / "gdn.pth"
    mto.save(model, path)
    # Load policies saved before fixed, non-configurable fields were removed.
    checkpoint = torch.load(path)
    for _, mode_state in checkpoint["modelopt_state"]["modelopt_state_dict"]:
        for policy in mode_state["metadata"]["linear_attention"].values():
            policy["state"].update(mode="chunk", quantize_initial=True)
            policy["solve"] = {"method": "exact"}
    torch.save(checkpoint, path)
    restored = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    mto.restore(restored, path)
    assert restored[0].linear_attention_config == model[0].linear_attention_config
    assert restored[0].gdn_state_qdq_block_v == 16
    for name, enabled in (("gdn_state_quantizer", state), ("gdn_w_quantizer", w)):
        original = getattr(model[0], name)
        quantizer = getattr(restored[0], name)
        assert quantizer.is_enabled == enabled
        assert quantizer.axis == original.axis
        assert quantizer.num_bits == original.num_bits
        assert quantizer.block_sizes == original.block_sizes
        assert quantizer._dynamic == original._dynamic
        assert not hasattr(restored[1], name)
    if state and not w:
        torch.testing.assert_close(restored[0].gdn_state_quantizer(sample), expected)


def test_quant_cfg_refinement_updates_and_validates_existing_quantized_module():
    cfg = quant_cfg()
    cfg["algorithm"] = None
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    assert model.gdn_state_quantizer.is_enabled
    assert not model.gdn_w_quantizer.is_enabled

    cfg = quant_cfg(state=False, w=True)
    cfg["algorithm"] = None
    cfg["linear_attention"] = [{"module_name": "", "cfg": {"state": {"block_v": 32}}}]
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 32
    cfg["linear_attention"].append({"module_name": "", "cfg": {}})
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 64
    assert not model.gdn_state_quantizer.is_enabled
    assert model.gdn_w_quantizer.is_enabled

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
    assert not model.replay_key_quantizer.is_enabled
    assert not model.replay_update_quantizer.is_enabled


def test_replay_quantizers_use_standard_controls_and_restore():
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*replay_key_quantizer",
                "cfg": {"num_bits": (4, 3), "type": "dynamic", "axis": (0,)},
            },
            {
                "quantizer_name": "*replay_update_quantizer",
                "cfg": {"num_bits": (4, 3), "type": "dynamic", "block_sizes": {-1: 16}},
            },
        ],
        "linear_attention": [
            {
                "module_name": "*",
                "cfg": {
                    "backend": "matmul",
                    "state": {"block_v": 16},
                    "decode": {"mode": "replay", "replay": {"window": 3}},
                },
            }
        ],
        "algorithm": None,
    }
    torch.manual_seed(53)
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    x = torch.randn(1, 7, 1, 4) * 0.1
    with linear_attention_training_phase(model, [2]):
        quantized = model(x)
    saved = deepcopy(mto.modelopt_state(model))
    weights = deepcopy(model.state_dict())
    mtq.disable_quantizer(model, "*")
    with linear_attention_training_phase(model, [2]):
        plain = model(x)
    assert not torch.equal(plain, quantized)
    # Current checkpoints and pre-handle replay checkpoints reproduce the same computation.
    for legacy in (None, False, True):
        checkpoint = deepcopy(saved)
        if legacy is not None:
            for _, mode_state in checkpoint["modelopt_state_dict"]:
                policy = mode_state["metadata"]["linear_attention"][""]
                policy["schema_version"] = 1
                if not legacy:
                    policy["decode"]["replay"]["factor_qdq"] = False
                for name in model.replay_quantizer_names:
                    mode_state["metadata"]["quantizer_state"].pop(name)
        restored = mto.restore_from_modelopt_state(TinyGatedDeltaNet(), checkpoint)
        restored.load_state_dict(weights)
        assert restored.replay_key_quantizer.is_enabled == (legacy is not False)
        assert restored.replay_update_quantizer.is_enabled == (legacy is not False)
        assert "factor_qdq" not in restored.linear_attention_config.decode.replay.model_dump()
        with linear_attention_training_phase(restored, [2]):
            torch.testing.assert_close(
                restored(x), plain if legacy is False else quantized, rtol=0, atol=0
            )


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
