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

from modelopt.torch.quantization.config import QuantizeConfig, QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig, kda_state_qat
from modelopt.torch.quantization.linear_attention import training as state_training
from modelopt.torch.quantization.linear_attention.utils import _state_qdq, state_quantizer_config
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("state_format", ["fp8_e4m3", "int8"])
def test_state_qdq_matches_tensor_quantizer(state_format):
    torch.manual_seed(762)
    # A partial value tile checks grouping and padding against TensorQuantizer.
    value = torch.randn(2, 3, 5, 19, requires_grad=True)
    cfg = {"num_bits": (4, 3), "type": "dynamic", "axis": (0, 1)}
    if state_format == "int8":
        cfg.update(num_bits=8, unsigned=False, narrow_range=True)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**cfg))
    expected = torch.cat(
        [quantizer(tile.flatten(-2)).reshape_as(tile) for tile in value.split(16, -1)], -1
    )
    actual = _state_qdq(value, quantizer, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((actual * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_fp8_state_blocks_match_tensor_quantizer(block_size):
    value = torch.randn(1, 2, 3, 133, requires_grad=True)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(4, 3), type="dynamic", block_sizes={-1: block_size}, pass_through_bwd=True
        )
    )
    assert state_quantizer_config(quantizer) == ("fp8_e4m3", block_size)
    actual = _state_qdq(value, quantizer)
    torch.testing.assert_close(actual, quantizer(value), rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((actual * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"chunk_size": 32},
        {"solve": {"method": "neumann"}},
        {"state": {"mode": "token"}},
    ],
)
def test_linear_attention_policy_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        QuantizeConfig(
            linear_attention=[{"module_name": "*", "cfg": {"backend": "serving", **settings}}]
        )


@pytest.mark.parametrize("raw_gates", [False, True])
def test_kda_rejects_incompatible_negative_eigenvalue_gates(raw_gates):
    q = torch.zeros(1, 1, 1, 32)
    beta = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="allow_neg_eigval"):
        kda_state_qat(
            q,
            q,
            q,
            q,
            beta,
            policy=LinearAttentionConfig(backend="serving"),
            allow_neg_eigval=True,
            use_beta_sigmoid_in_kernel=raw_gates,
            gate_inputs=(q, beta, torch.zeros(1), torch.zeros(32)) if raw_gates else None,
        )


@pytest.mark.parametrize(("major", "triton_version"), [(12, "3.6.0"), (12, "3.7.0"), (8, "3.6.0")])
def test_state_training_device_guard(monkeypatch, major, triton_version):
    # CI cannot safely run the faulting kernels; simulate only the absent hardware/version.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    monkeypatch.setattr(state_training, "version", lambda name: triton_version)
    state_training._validate_training_device.cache_clear()
    try:
        if major >= 12 and triton_version == "3.6.0":
            with pytest.raises(NotImplementedError, match=r"sm_120.*Triton < 3\.7"):
                state_training._validate_training_device(torch.device("cuda:0"))
        else:
            state_training._validate_training_device(torch.device("cuda:0"))
    finally:
        state_training._validate_training_device.cache_clear()
