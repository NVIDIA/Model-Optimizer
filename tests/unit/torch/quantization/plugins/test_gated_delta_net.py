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
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins import gated_delta_net
from modelopt.torch.quantization.plugins.gated_delta_net import GatedDeltaNetStateQuantMixin

GDN_STATE_INT8_DYNAMIC = {
    "num_bits": 8,
    "unsigned": False,
    "narrow_range": True,
    "type": "dynamic",
    "axis": (0, 1),
}

GDN_STATE_FP8_DYNAMIC = {"num_bits": (4, 3), "axis": (0, 1), "type": "dynamic"}


def chunk_gated_delta_rule(q, k, v, g, beta, **kwargs):
    """Stand-in with the fla kernel's name; the real kernel needs a GPU."""
    return q + k + v, None


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


def test_disabled_state_quantizer_calls_original_kernel():
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    expected = model(x)

    disable_all = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "max"}
    mtq.quantize(model, disable_all, lambda m: m(x))

    assert isinstance(model, _QuantTinyGatedDeltaNet)
    assert not model.gdn_state_quantizer.is_enabled and not model.gdn_w_quantizer.is_enabled
    assert model.gdn_state_qdq_block_v == 64
    assert torch.equal(model(x), expected)
    assert model.gated_delta_rule is chunk_gated_delta_rule, "the kernel swap must be undone"


@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_enabled_state_quantizer_uses_state_qdq_kernel(monkeypatch, state_format):
    calls = []

    def fake_state_qdq_kernel(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(
        gated_delta_net, "_state_qdq_chunk_gated_delta_rule", lambda: fake_state_qdq_kernel
    )
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    cfg = quant_cfg()
    if state_format == "int8":
        cfg["quant_cfg"][1]["cfg"] = GDN_STATE_INT8_DYNAMIC
    mtq.quantize(model, cfg, lambda m: m(x))

    model(x)
    assert calls and calls[-1] == {
        "chunk_size": 64,
        "state_qdq": 2 if state_format == "int8" else 1,
        "state_qdq_block_v": 64,
        "w_quantizer": None,
    }

    # The deterministic torch kernel has no quantized counterpart.
    model.gated_delta_rule = lambda *a, **kw: chunk_gated_delta_rule(*a, **kw)
    with pytest.raises(NotImplementedError, match="deterministic torch kernel"):
        model(x)


@pytest.mark.parametrize("state", [False, True])
def test_w_quantizer_is_passed_to_the_kernel(monkeypatch, state):
    """``*gdn_w_quantizer`` in the config hands the module's TensorQuantizer to the kernel, with
    or without the state quantizer."""
    calls = []

    def fake_state_qdq_kernel(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(
        gated_delta_net, "_state_qdq_chunk_gated_delta_rule", lambda: fake_state_qdq_kernel
    )
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    mtq.quantize(model, quant_cfg(state=state, w=True), lambda m: m(x))
    assert model.gdn_w_quantizer.is_enabled and model.gdn_state_quantizer.is_enabled == state

    model(x)
    assert calls[-1]["state_qdq"] == int(state)
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


def test_policy_roundtrip_last_match_and_hybrid_selection(tmp_path):
    model = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    cfg = quant_cfg()
    cfg["algorithm"] = None
    cfg["linear_attention"] = [
        {"module_name": "*", "cfg": {"state": {"block_v": 128}}},
        {"module_name": "0", "cfg": {"state": {"block_v": 32}}},
    ]
    mtq.quantize(model, cfg)
    assert model[0].gdn_state_qdq_block_v == 32
    # A post-conversion policy edit must also survive save/load.
    model[0].linear_attention_config.state.block_v = 16
    path = tmp_path / "gdn.pth"
    mto.save(model, path)
    restored = nn.Sequential(TinyGatedDeltaNet(), nn.Linear(4, 4))
    mto.restore(restored, path)
    assert restored[0].linear_attention_config == model[0].linear_attention_config
    assert restored[0].gdn_state_quantizer.is_enabled
    assert restored[0].gdn_state_qdq_block_v == 16
    assert not hasattr(restored[1], "linear_attention_config")


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


def test_restore_old_config_without_policy_field():
    model = mtq.quantize(
        nn.Linear(4, 4),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    state = mto.modelopt_state(model)
    for _, mode_state in state["modelopt_state_dict"]:
        config = mode_state["config"]
        if isinstance(config, dict):
            config.pop("linear_attention", None)
        else:
            config.__dict__.pop("linear_attention", None)
    restored = nn.Linear(4, 4)
    mto.restore_from_modelopt_state(restored, state)
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored(x), model(x))


def test_policy_refinement_updates_existing_quantized_module():
    cfg = quant_cfg()
    cfg["algorithm"] = None
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    cfg["linear_attention"] = [{"module_name": "", "cfg": {"state": {"block_v": 32}}}]
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 32
    # A later complete rule resets omitted fields instead of merging the previous policy.
    cfg["linear_attention"].append({"module_name": "", "cfg": {}})
    mtq.quantize(model, cfg)
    assert model.gdn_state_qdq_block_v == 64


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


def test_int8_state_conversion_and_checkpoint(tmp_path):
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*gdn_state_quantizer", "cfg": GDN_STATE_INT8_DYNAMIC},
        ],
        "algorithm": None,
    }
    model = mtq.quantize(TinyGatedDeltaNet(), cfg)
    assert model._linear_attn_state_format == "int8"
    path = tmp_path / "int8.pt"
    mto.save(model, path)
    restored = mto.restore(TinyGatedDeltaNet(), path)
    assert restored._linear_attn_state_format == "int8"
    assert restored.gdn_state_quantizer.narrow_range
    assert not restored.gdn_state_quantizer.unsigned
    assert not restored.gdn_w_quantizer.is_enabled
