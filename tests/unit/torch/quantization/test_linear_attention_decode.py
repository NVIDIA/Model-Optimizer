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

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import LinearAttentionConfig
from modelopt.torch.quantization.linear_attention.decode import _encode
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
    encoded = _encode(
        value, True, 16, state=True, state_format=state_format, state_quantizer=quantizer
    )
    torch.testing.assert_close(encoded.values, expected, rtol=0, atol=0)
    probe = torch.randn_like(value)
    (gradient,) = torch.autograd.grad((encoded.values * probe).sum(), value)
    torch.testing.assert_close(gradient, probe, rtol=0, atol=0)
    assert encoded.scales.shape == (2, 3, 2)
    assert not encoded.scales.requires_grad


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "reference", "decode": {}},
        {"backend": "matmul", "decode": {}},
        {"backend": "serving", "decode": {"precision": "full"}},
    ],
)
def test_reference_training_is_rejected(config):
    with pytest.raises(ValueError, match="Reference training is retired"):
        LinearAttentionConfig(**config)


@pytest.mark.parametrize(("precision", "window"), [("vllm_0_15", 1), ("replayssm", 4)])
def test_native_legacy_policy_preserves_precision(precision, window, tmp_path):
    decode = {"precision": precision, "readout": "working"}
    if precision == "replayssm":
        decode.update(state_codec="int8_hadamard32", mode="replay", replay={"window": window})
    policy = LinearAttentionConfig(schema_version=2, backend="matmul", decode=decode)
    assert policy == LinearAttentionConfig(
        backend="serving", precision=precision, replay_window=window
    )
    assert "decode" not in policy.model_dump()
    assert "state_codec" not in policy.model_dump()
    # Reproduce the field layout saved by the old class, bypassing new validation.
    legacy = LinearAttentionConfig()
    legacy.__dict__.clear()
    legacy.__dict__.update(schema_version=2, backend="matmul", decode=decode)
    path = tmp_path / "legacy-policy.pt"
    torch.save(legacy, path)
    assert torch.load(path, weights_only=True) == policy


@pytest.mark.parametrize(
    "settings", [{"readout": "stored"}, {"prefill_state_qdq": True}, {"decay_log_step": 0.02}]
)
def test_serving_precision_requires_native_schedule(settings):
    with pytest.raises(ValueError, match="Native state QAT requires"):
        LinearAttentionConfig(backend="serving", decode=settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"replay_window": 4},
        {"precision": "replayssm", "replay_window": 0},
        {"precision": "replayssm", "replay_window": 65},
        {"precision": "replayssm", "state_block_v": 16},
        {"decode": {}, "precision": "replayssm"},
    ],
)
def test_unified_policy_rejects_incompatible_settings(settings):
    with pytest.raises(ValueError):
        LinearAttentionConfig(backend="serving", **settings)
