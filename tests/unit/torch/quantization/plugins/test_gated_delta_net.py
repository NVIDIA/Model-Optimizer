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

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins import gated_delta_net
from modelopt.torch.quantization.plugins.gated_delta_net import (
    GatedDeltaNetStateQuantMixin,
    _validate_state_quantizer,
)

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
        _validate_state_quantizer(TensorQuantizer(QuantizerAttributeConfig(**attributes)))
    _validate_state_quantizer(TensorQuantizer(QuantizerAttributeConfig(**GDN_STATE_FP8_DYNAMIC)))


def test_disabled_state_quantizer_calls_original_kernel():
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    expected = model(x)

    disable_all = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": "max"}
    mtq.quantize(model, disable_all, lambda m: m(x))

    assert isinstance(model, _QuantTinyGatedDeltaNet)
    assert not model.gdn_state_quantizer.is_enabled and not model.gdn_w_quantizer.is_enabled
    assert model.gdn_state_qdq_block_v is None
    assert torch.equal(model(x), expected)
    assert model.gated_delta_rule is chunk_gated_delta_rule, "the kernel swap must be undone"


def test_enabled_state_quantizer_uses_state_qdq_kernel(monkeypatch):
    calls = []

    def fake_state_qdq_kernel(*args, **kwargs):
        calls.append(kwargs)
        return chunk_gated_delta_rule(*args)

    monkeypatch.setattr(
        gated_delta_net, "_state_qdq_chunk_gated_delta_rule", lambda: fake_state_qdq_kernel
    )
    model = TinyGatedDeltaNet()
    x = torch.randn(2, 8, 3, 4)
    mtq.quantize(model, quant_cfg(), lambda m: m(x))
    model.gdn_state_qdq_block_v = 64

    model(x)
    assert calls and calls[-1] == {"state_qdq": 1, "state_qdq_block_v": 64, "w_quantizer": None}

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
