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

"""Tests of tensor quantization function and module."""

import numpy as np
import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.tensor_quant_common import FakeTensorQuantTester

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn.modules.tensor_quantizer as tensor_quantizer_module
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import (
    TensorQuantizer,
    register_quant_backend,
    unregister_quant_backend,
)
from modelopt.torch.quantization.nn.functional import _random_signs


class TestFakeTensorQuantCPU(FakeTensorQuantTester):
    device = "cpu"


class TestQuantizerAttributeConfig:
    def test_scaled_mode(self):
        num_bits = np.random.randint(1, 16)

        test_quant_attr_cfg = QuantizerAttributeConfig(num_bits=num_bits)
        assert test_quant_attr_cfg.num_bits == num_bits
        assert test_quant_attr_cfg.axis is None

        axis = (0, 1, 3)
        test_quant_attr_cfg = QuantizerAttributeConfig(axis=axis)
        assert test_quant_attr_cfg.num_bits == 8  # default value
        assert test_quant_attr_cfg.axis == axis

    def test_from_to_dict(self, verbose):
        quant_attr_cfg_1 = QuantizerAttributeConfig(
            num_bits=2,
            fake_quant=True,
            axis=(1, 2),
        )
        quant_attr_cfg_2 = QuantizerAttributeConfig(**quant_attr_cfg_1.dict())
        assert quant_attr_cfg_1 == quant_attr_cfg_2

        quant_attr_cfg_1 = QuantizerAttributeConfig(num_bits=2, unsigned=True)
        quant_attr_cfg_2 = QuantizerAttributeConfig(**quant_attr_cfg_1.dict())
        assert quant_attr_cfg_1 == quant_attr_cfg_2

    def test_rotate_mode_serialization(self):
        quant_attr_cfg = QuantizerAttributeConfig(
            rotate={
                "enable": True,
                "mode": "rotate_back",
                "rotate_fp32": True,
                "block_size": 8,
                "seed": 123,
            }
        )

        assert quant_attr_cfg.model_dump(exclude_unset=True)["rotate"] == {
            "enable": True,
            "mode": "rotate_back",
            "rotate_fp32": True,
            "block_size": 8,
            "seed": 123,
        }

        with pytest.raises(ValueError, match="seed must be a non-negative int"):
            QuantizerAttributeConfig(rotate={"enable": True, "seed": -1})

    def test_num_bits(self):
        """Test num_bits for both integer and tuple cases."""

        with pytest.raises(
            ValueError,
            match=r"Invalid quantizer config: Cannot specify only {'enable': True}. "
            r"Additional parameters are required when enabling quantization.",
        ):
            QuantizerAttributeConfig(enable=True)

        with pytest.raises(
            ValueError,
            match=r"num_bits must be a positive integer or a tuple of positive integers.",
        ):
            QuantizerAttributeConfig(enable=True, num_bits=0)

        with pytest.raises(
            ValueError,
            match=r"num_bits must be a positive integer or a tuple of positive integers.",
        ):
            QuantizerAttributeConfig(enable=True, num_bits=-1)

        # # Test positive tuple validation
        with pytest.raises(
            ValueError,
            match=r"num_bits must be a positive integer or a tuple of positive integers.",
        ):
            QuantizerAttributeConfig(enable=True, num_bits=(0, 3))

        with pytest.raises(
            ValueError,
            match=r"num_bits must be a positive integer or a tuple of positive integers.",
        ):
            QuantizerAttributeConfig(enable=True, num_bits=(-1, 2))


def _run_rotated_backend(monkeypatch, rotate):
    calls = []

    def rotate_fn(inputs, rotate_fp32=False, block_size=None, random_sign_seed=None, inverse=False):
        calls.append((rotate_fp32, block_size, random_sign_seed, inverse))
        return inputs + 10

    def backend(inputs, _tq):
        return inputs * 2

    monkeypatch.setattr(tensor_quantizer_module, "normalized_hadamard_transform", rotate_fn)
    register_quant_backend("test_rotate_mode_backend", backend)
    try:
        quantizer = TensorQuantizer(
            QuantizerAttributeConfig(rotate=rotate, backend="test_rotate_mode_backend")
        )
        inputs = torch.tensor([[1.0, 2.0]])
        return quantizer(inputs), inputs, calls, quantizer
    finally:
        unregister_quant_backend("test_rotate_mode_backend")


def test_tensor_quantizer_rotate_mode_preserves_default_path(monkeypatch):
    outputs, inputs, calls, quantizer = _run_rotated_backend(monkeypatch, rotate={"enable": True})

    assert not quantizer.rotate_back_is_enabled
    assert torch.equal(outputs, (inputs + 10) * 2)
    assert calls == [(False, None, None, False)]


def test_tensor_quantizer_rotate_mode_can_rotate_back(monkeypatch):
    outputs, inputs, calls, quantizer = _run_rotated_backend(
        monkeypatch,
        rotate={
            "enable": True,
            "mode": "rotate_back",
            "rotate_fp32": True,
            "block_size": 8,
            "seed": 123,
        },
    )

    assert quantizer.rotate_back_is_enabled
    assert torch.equal(outputs, ((inputs + 10) * 2) + 10)
    assert calls == [(True, 8, 123, False), (True, 8, 123, True)]


def test_tensor_quantizer_rotate_back_rejects_real_quant(monkeypatch):
    def fail_if_rotated(
        inputs, rotate_fp32=False, block_size=None, random_sign_seed=None, inverse=False
    ):
        raise AssertionError("rotate_back with fake_quant=False should fail before rotation")

    monkeypatch.setattr(
        tensor_quantizer_module,
        "normalized_hadamard_transform",
        fail_if_rotated,
    )
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8,
            fake_quant=False,
            rotate={"enable": True, "mode": "rotate_back"},
        )
    )

    with pytest.raises(ValueError, match="rotate_back mode is only supported with fake_quant=True"):
        quantizer(torch.tensor([[1.0, 2.0]]))


def test_random_signs_are_cached_by_seed_dim_device_and_dtype():
    _random_signs.cache_clear()

    try:
        signs = _random_signs(123, 8, torch.device("cpu"), torch.float32)
        cached_signs = _random_signs(123, 8, torch.device("cpu"), torch.float32)
        fp16_signs = _random_signs(123, 8, torch.device("cpu"), torch.float16)
        different_seed_signs = _random_signs(124, 8, torch.device("cpu"), torch.float32)
        different_dim_signs = _random_signs(123, 16, torch.device("cpu"), torch.float32)

        assert cached_signs.data_ptr() == signs.data_ptr()
        assert torch.equal(cached_signs, signs)
        assert fp16_signs.dtype == torch.float16
        assert fp16_signs.data_ptr() != signs.data_ptr()
        assert different_seed_signs.data_ptr() != signs.data_ptr()
        assert different_dim_signs.data_ptr() != signs.data_ptr()
    finally:
        _random_signs.cache_clear()


WINT4INT8_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": [
                {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                {"num_bits": 8, "axis": 0},
            ],
            "enable": True,
        },
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8}, "enable": True},
    ],
    "algorithm": "awq_full",
}


def test_set_quantizer_cxt():
    model = SimpleLinear()
    model.eval()
    inputs = model.get_input()
    mtq.quantize(model, WINT4INT8_CFG, lambda model: model(inputs))
    state_dict = model.state_dict()
    output_ref = model(inputs)

    mtq.set_quantizer_by_cfg(model, [{"quantizer_name": "*output_quantizer", "enable": True}])

    with mtq.set_quantizer_by_cfg_context(
        model,
        [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*output_quantizer", "enable": True},
        ],
    ):
        for name, module in model.named_modules():
            if not isinstance(module, TensorQuantizer):
                continue
            if name.endswith("output_quantizer"):
                assert module.is_enabled
            else:
                assert not module.is_enabled
        mtq.calibrate(model, "max", lambda model: model(inputs * 10))

    mtq.set_quantizer_by_cfg(model, [{"quantizer_name": "*output_quantizer", "enable": False}])

    output_test = model(inputs)
    assert torch.allclose(output_ref, output_test)

    state_dict_test = model.state_dict()
    for k, v in state_dict_test.items():
        if "output_quantizer" in k:
            continue
        assert torch.allclose(v, state_dict[k])


def test_set_quantizer_cxt_restores_on_exception():
    """Quantizer properties must be fully restored when the context body raises.

    Guards the try/finally in `set_quantizer_by_cfg_context`: regressing it
    silently corrupts state for every caller (AWQ-lite, GPTQ Hessian, etc.)
    when calibration / forward_loop bodies raise. The snapshot uses the same
    `get_modelopt_state(properties_only=True)` API the context manager itself
    uses for save/restore, so a regression in any tracked property is caught.
    """
    model = SimpleLinear()
    model.eval()
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(model.get_input()))

    pre_state = {
        name: module.get_modelopt_state(properties_only=True)
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer)
    }
    assert any(not s.get("_disabled", True) for s in pre_state.values()), (
        "expected at least one enabled quantizer after quantize()"
    )

    class _BoomError(RuntimeError):
        pass

    with (
        pytest.raises(_BoomError),
        mtq.set_quantizer_by_cfg_context(model, [{"quantizer_name": "*", "enable": False}]),
    ):
        for module in model.modules():
            if isinstance(module, TensorQuantizer):
                assert not module.is_enabled
        raise _BoomError("simulate calibration failure inside context body")

    post_state = {
        name: module.get_modelopt_state(properties_only=True)
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer)
    }
    assert post_state == pre_state
