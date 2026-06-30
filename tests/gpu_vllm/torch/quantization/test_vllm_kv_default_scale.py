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

from types import SimpleNamespace

import torch
import torch.nn as nn

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.plugins import vllm as vllm_plugin
from modelopt.torch.quantization.plugins.vllm import _set_vllm_attention_kv_runtime_amax


def _nvfp4_quantizer(*, block_size=16, enabled=True, amax=None, use_constant_amax=False):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
            enable=enabled,
            use_constant_amax=use_constant_amax,
        )
    )
    if amax is not None:
        quantizer.amax = torch.tensor(amax)
    return quantizer


def test_uncalibrated_nvfp4_kv_use_global_scale_one():
    layer = SimpleNamespace(
        q_bmm_quantizer=_nvfp4_quantizer(),
        k_bmm_quantizer=_nvfp4_quantizer(use_constant_amax=True),
        v_bmm_quantizer=_nvfp4_quantizer(use_constant_amax=True),
        p_bmm_quantizer=_nvfp4_quantizer(),
    )
    _set_vllm_attention_kv_runtime_amax(layer, torch.device("cpu"))
    inputs = torch.tensor([-3.0, 5.0])
    for name in ("k_bmm_quantizer", "v_bmm_quantizer"):
        quantizer = getattr(layer, name)
        assert quantizer._get_amax(inputs).item() == 2688.0
        assert quantizer._get_amax(inputs).item() / (6.0 * 448.0) == 1.0
        assert "_runtime_default_amax" not in quantizer.state_dict()
    assert not hasattr(layer.q_bmm_quantizer, "_runtime_default_amax")
    assert not hasattr(layer.p_bmm_quantizer, "_runtime_default_amax")


def test_calibrated_kv_amax_overrides_runtime_default():
    layer = SimpleNamespace(
        k_bmm_quantizer=_nvfp4_quantizer(amax=7.25, use_constant_amax=True),
        v_bmm_quantizer=_nvfp4_quantizer(use_constant_amax=True),
    )
    _set_vllm_attention_kv_runtime_amax(layer, torch.device("cpu"))
    # Simulate optional modelopt_state_weights loading after metadata restore.
    layer.v_bmm_quantizer.amax = torch.tensor(9.5)
    _set_vllm_attention_kv_runtime_amax(layer, torch.device("cpu"))
    inputs = torch.tensor([-3.0, 5.0])
    assert layer.k_bmm_quantizer._get_amax(inputs).item() == 7.25
    assert layer.v_bmm_quantizer._get_amax(inputs).item() == 9.5


def test_sequential_kv_quantizers_are_left_unchanged():
    sequential = SequentialQuantizer(_nvfp4_quantizer(), _nvfp4_quantizer())
    layer = SimpleNamespace(k_bmm_quantizer=sequential, v_bmm_quantizer=sequential)
    _set_vllm_attention_kv_runtime_amax(layer, torch.device("cpu"))
    assert all(not hasattr(q, "_runtime_default_amax") for q in sequential)


def test_unsupported_kv_quantizers_do_not_get_runtime_default():
    fp8 = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
    block32 = _nvfp4_quantizer(block_size=32)
    disabled = _nvfp4_quantizer(enabled=False)
    for quantizer in (fp8, block32, disabled):
        layer = SimpleNamespace(k_bmm_quantizer=quantizer, v_bmm_quantizer=quantizer)
        _set_vllm_attention_kv_runtime_amax(layer, torch.device("cpu"))
        assert not hasattr(quantizer, "_runtime_default_amax")


def test_post_restore_vllm_attentions_visits_only_attention_modules(monkeypatch):
    class BaseAttention(nn.Module):
        pass

    class QuantAttention(BaseAttention):
        def __init__(self):
            super().__init__()
            self.post_restore_calls = []

        def modelopt_post_restore(self, prefix=""):
            self.post_restore_calls.append(prefix)

    class NativeAttention(BaseAttention):
        pass

    class UnrelatedModule(nn.Module):
        @property
        def modelopt_post_restore(self):
            raise AssertionError("non-attention hook lookup")

    attention = QuantAttention()
    native_attention = NativeAttention()
    model = nn.ModuleList([attention, native_attention, UnrelatedModule(), nn.Linear(2, 2)])
    monkeypatch.setattr(vllm_plugin, "_ATTENTION_TYPES", (BaseAttention,))

    vllm_plugin.post_restore_vllm_attentions(model)

    assert attention.post_restore_calls == [""]
