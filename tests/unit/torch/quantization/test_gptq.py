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

"""CPU unit tests for GPTQ utilities."""

from unittest.mock import Mock, create_autospec

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.model_calib as model_calib_module
from modelopt.torch.quantization.config import (
    GPTQCalibConfig,
    LocalHessianCalibConfig,
    MaxCalibConfig,
    MseCalibConfig,
)
from modelopt.torch.quantization.model_calib import gptq
from modelopt.torch.quantization.utils.calib_utils import update_hessian


class TestGPTQScaleAlgorithmConfig:
    """Tests for GPTQ scale-calibration configuration."""

    def test_default_config(self):
        cfg = GPTQCalibConfig()
        assert cfg.scale_algorithm is None
        assert cfg.model_dump()["scale_algorithm"] is None
        assert cfg.perc_damp == 0.01
        assert cfg.block_size == 128
        assert cfg.fused is False

    def test_old_style_config_dict_restores(self):
        cfg = GPTQCalibConfig(method="gptq", perc_damp=0.02)
        assert cfg.scale_algorithm is None

    @pytest.mark.parametrize(
        ("method", "config_type"),
        [
            ("max", MaxCalibConfig),
            ("mse", MseCalibConfig),
            ("local_hessian", LocalHessianCalibConfig),
        ],
    )
    def test_scale_algorithm(self, method, config_type):
        cfg = GPTQCalibConfig(scale_algorithm={"method": method})
        assert isinstance(cfg.scale_algorithm, config_type)

    def test_unsupported_scale_algorithm(self):
        with pytest.raises(ValueError):
            GPTQCalibConfig(scale_algorithm={"method": "smoothquant"})

    def test_local_hessian_activation_error_coupling(self):
        cfg = GPTQCalibConfig(
            scale_algorithm={"method": "local_hessian", "activation_error_coupling": True}
        )
        assert cfg.model_dump()["scale_algorithm"]["activation_error_coupling"] is True

    def test_scale_algorithm_preserves_sparse_dict(self, monkeypatch):
        cfg = GPTQCalibConfig(scale_algorithm={"method": "mse", "fp8_scale_sweep": True})
        assert cfg.model_dump()["scale_algorithm"] == {
            "method": "mse",
            "fp8_scale_sweep": True,
        }

        calibrate = create_autospec(model_calib_module.mse_calibrate)
        monkeypatch.setattr(model_calib_module, "mse_calibrate", calibrate)
        model = Mock()
        model_calib_module._run_scale_calibration(model, None, cfg.scale_algorithm)
        calibrate.assert_called_once_with(model, forward_loop=None, fp8_scale_sweep=True)


@pytest.mark.parametrize(
    ("scale_algorithm", "expected_func", "expected_kwargs"),
    [
        (None, "max", {}),
        ({"method": "mse", "fp8_scale_sweep": True}, "mse", {"fp8_scale_sweep": True}),
        (
            {"method": "local_hessian", "activation_error_coupling": True},
            "local_hessian",
            {"activation_error_coupling": True},
        ),
    ],
)
def test_gptq_scale_algorithm_dispatch(
    monkeypatch, scale_algorithm, expected_func, expected_kwargs
):
    calls = []

    def record(name):
        def calibrate(model, forward_loop, **kwargs):
            calls.append((name, model, forward_loop, kwargs))

        return calibrate

    monkeypatch.setattr(model_calib_module, "max_calibrate", record("max"))
    monkeypatch.setattr(model_calib_module, "mse_calibrate", record("mse"))
    monkeypatch.setattr(model_calib_module, "local_hessian_calibrate", record("local_hessian"))
    model = nn.Linear(4, 4)

    def forward_loop(model):
        pass

    gptq(model, forward_loop=forward_loop, scale_algorithm=scale_algorithm)

    assert calls == [(expected_func, model, forward_loop, expected_kwargs)]


@pytest.mark.parametrize("sequential", [False, True])
def test_gptq_fused_rejects_four_over_six(sequential):
    model = nn.Linear(32, 32)
    weight_quantizer_cfg = {
        "num_bits": (2, 1),
        "block_sizes": {
            -1: 16,
            "type": "static",
            "scale_bits": (4, 3),
            "four_over_six": True,
        },
    }
    if sequential:
        weight_quantizer_cfg = [weight_quantizer_cfg, {"num_bits": (4, 3), "axis": None}]
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": weight_quantizer_cfg,
            },
        ],
        "algorithm": None,
    }
    mtq.quantize(model, quant_cfg)

    with pytest.raises(ValueError, match="four_over_six"):
        gptq(model, forward_loop=lambda m: None, fused=True)


def test_update_hessian():
    """Test for update_hessian function with both random and known inputs."""
    # Test 1: Random input - general functionality test
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 3
    features = 4
    input_tensor = torch.randn(batch_size, seq_len, features, dtype=torch.float32)

    hessian = torch.zeros(features, features, dtype=torch.float32)
    n_samples = 0

    updated_hessian, new_n_samples = update_hessian(input_tensor, hessian, n_samples)

    # Verify output shape
    assert updated_hessian.shape == (features, features), (
        f"Expected hessian shape ({features}, {features}), got {updated_hessian.shape}"
    )

    # Verify sample count is updated correctly (incremented by total tokens = batch * seq_len)
    expected_n_samples = batch_size * seq_len
    assert new_n_samples == expected_n_samples, (
        f"Expected n_samples={expected_n_samples}, got {new_n_samples}"
    )

    # Verify hessian is not all zeros after update
    assert not torch.allclose(updated_hessian, torch.zeros_like(updated_hessian)), (
        "Hessian should not be all zeros after update"
    )

    # Verify hessian is symmetric (should be for outer product X @ X.T)
    assert torch.allclose(updated_hessian, updated_hessian.t()), "Hessian should be symmetric"

    # Test 2: Known input - verify correct hessian calculation
    batch_size = 6
    seq_len = 2
    features = 2
    input_tensor = torch.ones(batch_size, seq_len, features, dtype=torch.float32)

    hessian = torch.zeros(features, features, dtype=torch.float32)
    n_samples = 0

    updated_hessian, new_n_samples = update_hessian(input_tensor, hessian, n_samples)

    # Manual calculation:
    # input_flat shape: (features, batch*seq) = (2, 12), all ones
    # n_samples = batch * seq = 12 (token count after flattening)
    # scaled_input = sqrt(2/12) * ones(2, 12)
    # outer_product = (2/12) * ones(2,12) @ ones(12,2) = [[2,2], [2,2]]
    expected_n_samples = batch_size * seq_len  # 12 tokens
    expected_hessian = torch.ones(features, features, dtype=torch.float32) * 2.0

    assert torch.allclose(updated_hessian, expected_hessian, atol=1e-5), (
        f"Expected hessian {expected_hessian}, got {updated_hessian}"
    )
    assert new_n_samples == expected_n_samples

    # Test 3: Accumulated hessians - verify equivalence
    # Processing [6,2,2] in one step should equal processing [2,2,2] three times
    seq_len = 2
    features = 2

    # Process in 3 steps of batch_size=2 (4 tokens each, 12 total)
    hessian_accumulated = torch.zeros(features, features, dtype=torch.float32)
    n_samples_accumulated = 0

    for _ in range(3):
        input_batch = torch.ones(2, seq_len, features, dtype=torch.float32)
        hessian_accumulated, n_samples_accumulated = update_hessian(
            input_batch, hessian_accumulated, n_samples_accumulated
        )

    # Verify that accumulated result matches single-step result from Test 2
    assert torch.allclose(hessian_accumulated, updated_hessian, atol=1e-5), (
        f"Accumulated hessian should match single-step: expected {updated_hessian}, got {hessian_accumulated}"
    )
    assert torch.allclose(hessian_accumulated, expected_hessian, atol=1e-5), (
        f"Accumulated hessian should match expected: expected {expected_hessian}, got {hessian_accumulated}"
    )
    # 3 batches * 2 batch_size * 2 seq_len = 12 tokens
    assert n_samples_accumulated == 12, f"Expected n_samples=12, got {n_samples_accumulated}"


def test_update_hessian_zero_token_input_noops():
    """Empty activations must not mutate the Hessian or sample count."""
    features = 4
    hessian = torch.eye(features, dtype=torch.float32)
    expected_hessian = hessian.clone()
    n_samples = 7

    updated_hessian, new_n_samples = update_hessian(
        torch.empty(0, features, dtype=torch.float32), hessian, n_samples
    )

    assert updated_hessian is hessian
    assert new_n_samples == n_samples
    torch.testing.assert_close(hessian, expected_hessian)
