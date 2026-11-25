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

import contextlib
import sys
import weakref
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.tensor_quant_common import FakeTensorQuantTester
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.utils import parametrize
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.conversion as quant_conversion
import modelopt.torch.quantization.nn.modules.tensor_quantizer as tensor_quantizer_module
from modelopt.torch.quantization import QuantModuleRegistry, tensor_quant
from modelopt.torch.quantization.calib import MaxCalibrator
from modelopt.torch.quantization.config import (
    QuantizerAttributeConfig,
    QuantizerCfgEntry,
    RotateConfig,
)
from modelopt.torch.quantization.conversion import create_and_replace_svdquant_linear_on_the_fly
from modelopt.torch.quantization.ggml.common import _CacheKey
from modelopt.torch.quantization.nn import (
    SequentialQuantizer,
    TensorQuantizer,
    register_quant_backend,
    unregister_quant_backend,
)


class TestFakeTensorQuantCPU(FakeTensorQuantTester):
    device = "cpu"


def test_custom_op_schemas_keep_legacy_positional_calls():
    inputs = torch.randn(4, 32)
    amax = torch.tensor(1.0)
    fp8 = torch.ops.tensorrt.quantize_op(inputs, amax, 8, 4, False, False)
    mxfp8 = torch.ops.tensorrt.dynamic_block_quantize_op.overload(inputs, 32, None, 8, 4, 9, 8)

    assert (fp8.shape, fp8.dtype) == (inputs.shape, inputs.dtype)
    assert (mxfp8.shape, mxfp8.dtype) == (inputs.shape, inputs.dtype)


@pytest.mark.parametrize("num_bits", [8, 4])
def test_cpu_integer_carrier_strict_capture_preserves_custom_op(num_bits):
    class QuantizeCarrier(torch.nn.Module):
        def forward(self, inputs, amax):
            return torch.ops.tensorrt.quantize_op(inputs, amax, num_bits, 0, False, False, "Float")

    inputs = torch.randn(4, 32)
    amax = torch.tensor(1.0)
    exported = torch.export.export(QuantizeCarrier(), (inputs, amax), strict=True)

    assert any(
        node.target == torch.ops.tensorrt.quantize_op.default for node in exported.graph.nodes
    )
    expected = tensor_quant.fake_tensor_quant(inputs, amax, None, num_bits, False, False)
    torch.testing.assert_close(exported.module()(inputs, amax), expected)


def test_cpu_fp8_quantizer_strict_capture_preserves_custom_op():
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), narrow_range=False))
    quantizer.amax = torch.tensor(1.0)
    inputs = (torch.randn(4, 32),)

    exported = torch.export.export(quantizer, inputs, strict=True)

    assert any(
        node.target == torch.ops.tensorrt.quantize_op.default for node in exported.graph.nodes
    )
    torch.testing.assert_close(exported.module()(*inputs), quantizer(*inputs))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("amax_shape", "num_bits", "block_size", "axis"),
    [([], 8, None, None), ([2, 1], 8, None, 0), ([4, 1], 4, 2, 1)],
    ids=["scalar", "per_axis", "block"],
)
def test_registered_cpu_quantize_matches_eager(amax_shape, num_bits, block_size, axis, dtype):
    inputs = torch.linspace(-3, 3, 8, dtype=dtype).reshape(2, 4)
    if block_size is not None:
        inputs = inputs.reshape(-1, block_size)
    amax = torch.full(amax_shape, 2.0, dtype=dtype)
    if amax.numel() > 1:
        amax[0] = 0
    expected = tensor_quant.fake_tensor_quant(inputs, amax, None, num_bits, False, True)

    legacy = torch.ops.tensorrt.quantize_op(inputs, amax, num_bits, 0, False, True)
    with_metadata = torch.ops.tensorrt.quantize_op(
        inputs, amax, num_bits, 0, False, True, "Float", block_size, axis
    )

    torch.testing.assert_close(legacy, expected, rtol=0, atol=0)
    torch.testing.assert_close(with_metadata, expected, rtol=0, atol=0)


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
            rotate={"enable": True, "mode": "rotate_back", "rotate_fp32": True, "block_size": 8}
        )

        assert quant_attr_cfg.model_dump(exclude_unset=True)["rotate"] == {
            "enable": True,
            "mode": "rotate_back",
            "rotate_fp32": True,
            "block_size": 8,
        }

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

    def rotate_fn(inputs, rotate_fp32=False, block_size=None):
        calls.append((rotate_fp32, block_size))
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


@pytest.mark.parametrize(
    ("rotate", "rotate_back_enabled", "expected_calls", "expected_fn"),
    [
        ({"enable": True}, False, [(False, None)], lambda inputs: (inputs + 10) * 2),
        (
            {"enable": True, "mode": "rotate_back", "rotate_fp32": True, "block_size": 8},
            True,
            [(True, 8), (True, 8)],
            lambda inputs: ((inputs + 10) * 2) + 10,
        ),
    ],
)
def test_tensor_quantizer_rotate_modes(
    monkeypatch, rotate, rotate_back_enabled, expected_calls, expected_fn
):
    outputs, inputs, calls, quantizer = _run_rotated_backend(
        monkeypatch,
        rotate=rotate,
    )

    assert quantizer.rotate_back_is_enabled is rotate_back_enabled
    assert torch.equal(outputs, expected_fn(inputs))
    assert calls == expected_calls


def test_tensor_quantizer_rotate_back_rejects_real_quant(monkeypatch):
    def fail_if_rotated(inputs, rotate_fp32=False, block_size=None):
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


@pytest.mark.parametrize(
    ("rotate", "rotate_back_enabled", "expected_call_count", "expected_fn"),
    [
        ({"enable": True}, False, 1, lambda inputs: inputs + 10),
        ({"enable": True, "mode": "rotate_back"}, True, 2, lambda inputs: inputs + 20),
    ],
)
def test_tensor_quantizer_disabled_rotate_modes_roundtrip(
    monkeypatch, rotate, rotate_back_enabled, expected_call_count, expected_fn
):
    calls = []

    def rotate_fn(inputs, rotate_fp32=False, block_size=None):
        calls.append((rotate_fp32, block_size))
        return inputs + 10

    monkeypatch.setattr(tensor_quantizer_module, "normalized_hadamard_transform", rotate_fn)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(rotate=rotate, enable=False))
    inputs = torch.tensor([[1.0, 2.0]])

    outputs = quantizer(inputs)

    assert quantizer.rotate_back_is_enabled is rotate_back_enabled
    assert torch.equal(outputs, expected_fn(inputs))
    assert len(calls) == expected_call_count


def test_disable_only_update_clears_regular_quantizer_rotate_state():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(rotate={"enable": True, "mode": "rotate_back", "block_size": 8})
    )
    assert quantizer.rotate_is_enabled
    assert quantizer.rotate_back_is_enabled

    quantizer.set_from_attribute_config({"enable": False})

    assert not quantizer.is_enabled
    assert not quantizer.rotate_is_enabled
    assert isinstance(quantizer._rotate, RotateConfig)
    assert quantizer._rotate.mode == "rotate_back"
    assert quantizer._rotate.block_size == 8


def test_disable_rotate_preserves_type():
    # RotateConfig: enable off, other fields retained.
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(rotate={"enable": True, "mode": "rotate_back", "block_size": 8})
    )
    assert isinstance(quantizer._rotate, RotateConfig)
    quantizer.disable_rotate()
    assert isinstance(quantizer._rotate, RotateConfig)
    assert quantizer._rotate.enable is False
    assert quantizer._rotate.mode == "rotate_back"
    assert quantizer._rotate.block_size == 8
    assert not quantizer.rotate_is_enabled
    quantizer.disable_rotate()  # idempotent
    assert quantizer._rotate.enable is False

    # Raw dict (old checkpoints).
    quantizer._rotate = {"enable": True, "mode": "rotate", "block_size": 4}
    quantizer.disable_rotate()
    assert quantizer._rotate == {"enable": False, "mode": "rotate", "block_size": 4}

    # Bool.
    quantizer._rotate = True
    quantizer.disable_rotate()
    assert quantizer._rotate is False


def test_sequential_quantizer_disable_rotate_delegates():
    q0 = TensorQuantizer(QuantizerAttributeConfig(rotate={"enable": True}))
    q1 = TensorQuantizer(QuantizerAttributeConfig(rotate={"enable": True, "mode": "rotate_back"}))
    seq = SequentialQuantizer(q0, q1)

    seq.disable_rotate()

    assert not q0.rotate_is_enabled
    assert not q1.rotate_is_enabled


def _make_qlinear_with_backend(monkeypatch, calls, backend_name, rotate=False):
    def rotate_fn(inputs, rotate_fp32=False, block_size=None):
        calls.append((rotate_fp32, block_size))
        return inputs + 10

    def backend(inputs, _tq):
        return inputs * 2

    monkeypatch.setattr(tensor_quantizer_module, "normalized_hadamard_transform", rotate_fn)
    register_quant_backend(backend_name, backend)
    qlinear = QuantModuleRegistry.convert(torch.nn.Linear(4, 3))
    qlinear.input_quantizer.disable()
    qlinear.output_quantizer.disable()
    qlinear.weight_quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(rotate=rotate, backend=backend_name)
    )
    return qlinear


@pytest.mark.parametrize(
    ("rotate", "expected_weight_fn"),
    [
        ({"enable": True}, lambda weight: (weight + 10) * 2),
        ({"enable": True, "mode": "rotate_back"}, lambda weight: ((weight + 10) * 2) + 10),
    ],
)
def test_fold_weight_disables_quantizer_without_extra_transform(
    monkeypatch, rotate, expected_weight_fn
):
    calls = []
    backend_name = "test_fold_backend"
    qlinear = _make_qlinear_with_backend(monkeypatch, calls, backend_name, rotate=rotate)
    try:
        qlinear.weight_quantizer.amax = torch.tensor(1.0)
        weight0 = qlinear.weight.detach().clone()
        x = torch.randn(2, 4)
        out_before = qlinear(x)

        qlinear.fold_weight()

        assert torch.allclose(qlinear.weight, expected_weight_fn(weight0))
        assert not qlinear.weight_quantizer.is_enabled
        assert not qlinear.weight_quantizer.rotate_is_enabled
        assert not hasattr(qlinear.weight_quantizer, "_amax")

        calls_after_fold = len(calls)
        out_after = qlinear(x)
        assert torch.allclose(out_after, out_before)
        assert len(calls) == calls_after_fold

        # Second fold is a no-op: quantizer is disabled.
        weight_after = qlinear.weight.detach().clone()
        qlinear.fold_weight()
        assert torch.allclose(qlinear.weight, weight_after)
    finally:
        unregister_quant_backend(backend_name)


def test_fold_weight_keep_attrs_keeps_calibration_attrs_inactive(monkeypatch):
    calls = []
    backend_name = "test_fold_backend_keep"
    qlinear = _make_qlinear_with_backend(monkeypatch, calls, backend_name)
    try:
        qlinear.weight_quantizer.amax = torch.tensor(1.0)
        qlinear.weight_quantizer.pre_quant_scale = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        inputs = torch.randn(2, 4)
        output_before = qlinear(inputs)

        qlinear.fold_weight(keep_attrs=True)

        assert hasattr(qlinear.weight_quantizer, "_amax")
        assert hasattr(qlinear.weight_quantizer, "_pre_quant_scale")
        assert qlinear.weight_quantizer.pre_quant_scale is None
        assert not qlinear.weight_quantizer.is_enabled
        assert torch.allclose(qlinear(inputs), output_before)
    finally:
        unregister_quant_backend(backend_name)


def test_fold_weight_folds_svdquant_lora():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    inputs = torch.randn(2, 4)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda m: m(inputs))
    create_and_replace_svdquant_linear_on_the_fly(model)
    model[0].weight_quantizer.svdquant_lora_a = torch.randn(2, 4)
    model[0].weight_quantizer.svdquant_lora_b = torch.randn(3, 2)
    output_before = model(inputs)

    mtq.fold_weight(model)

    assert not hasattr(model[0].weight_quantizer, "_svdquant_lora_a")
    assert torch.allclose(model(inputs), output_before, atol=1e-6)


@torch.no_grad()
def test_fold_weight_folds_sequential_weight_quantizer():
    model = SimpleLinear()
    inputs = model.get_input()
    mtq.quantize(model, {**WINT4INT8_CFG, "algorithm": "max"}, lambda m: m(inputs))
    expected = {
        module: module.weight_quantizer(module.weight)
        for module in model.modules()
        if isinstance(getattr(module, "weight_quantizer", None), SequentialQuantizer)
    }
    assert expected

    mtq.fold_weight(model)

    for module, weight in expected.items():
        assert torch.equal(module.weight, weight)
        assert not any(q.is_enabled or hasattr(q, "_amax") for q in module.weight_quantizer)


def test_temporarily_fold_weights_restores_after_exception(monkeypatch):
    calls = []
    backend_name = "test_temporary_fold_backend"
    qlinear = _make_qlinear_with_backend(
        monkeypatch,
        calls,
        backend_name,
        rotate={"enable": True},
    )
    try:
        quantizer = qlinear.weight_quantizer
        quantizer.amax = torch.tensor(1.0)
        quantizer.pre_quant_scale = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        original_weight = qlinear.weight.detach().clone()
        original_storage = qlinear.weight.data_ptr()
        original_rotate = quantizer._rotate
        original_pre_quant_scale = quantizer.pre_quant_scale.detach().clone()
        original_input_dtype = quantizer._input_dtype
        inputs = torch.randn(2, 4)
        output_before = qlinear(inputs)

        with (
            pytest.raises(RuntimeError, match="test error"),
            mtq.temporarily_fold_weights(qlinear, snapshot_device="cpu"),
        ):
            assert not torch.equal(qlinear.weight, original_weight)
            assert qlinear.weight.data_ptr() == original_storage
            assert not quantizer.is_enabled
            assert not quantizer.rotate_is_enabled
            assert hasattr(quantizer, "_pre_quant_scale")
            assert quantizer.pre_quant_scale is None
            assert torch.allclose(qlinear(inputs), output_before)
            raise RuntimeError("test error")

        assert torch.equal(qlinear.weight, original_weight)
        assert qlinear.weight.data_ptr() == original_storage
        assert quantizer.is_enabled
        assert quantizer._rotate == original_rotate
        assert torch.equal(quantizer.pre_quant_scale, original_pre_quant_scale)
        assert quantizer._input_dtype == original_input_dtype
        assert torch.allclose(qlinear(inputs), output_before)
    finally:
        unregister_quant_backend(backend_name)


def test_temporarily_fold_weights_handles_disabled_transform_and_none_weight():
    disabled = QuantModuleRegistry.convert(torch.nn.Linear(4, 3, bias=False))
    pre_quant_scale = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
    disabled.weight_quantizer.pre_quant_scale = pre_quant_scale
    disabled.weight_quantizer.disable()
    original_weight = disabled.weight.detach().clone()

    tied_output = QuantModuleRegistry.convert(torch.nn.Linear(4, 3, bias=False))
    tied_output.weight = None

    with mtq.temporarily_fold_weights(torch.nn.ModuleList([disabled, tied_output])):
        assert torch.equal(disabled.weight, original_weight * pre_quant_scale)
        assert not disabled.weight_quantizer.is_enabled
        assert disabled.weight_quantizer.pre_quant_scale is None
        assert tied_output.weight_quantizer.is_enabled

    assert torch.equal(disabled.weight, original_weight)
    assert not disabled.weight_quantizer.is_enabled
    assert torch.equal(disabled.weight_quantizer.pre_quant_scale, pre_quant_scale)
    assert tied_output.weight_quantizer.is_enabled


def test_temporarily_fold_weights_restores_when_folding_fails(monkeypatch):
    first_calls = []
    second_calls = []
    first_backend = "test_temporary_fold_first_backend"
    second_backend = "test_temporary_fold_failing_backend"
    first = _make_qlinear_with_backend(monkeypatch, first_calls, first_backend)
    second = _make_qlinear_with_backend(monkeypatch, second_calls, second_backend)

    def failing_backend(inputs, _quantizer):
        raise RuntimeError("fold failed")

    unregister_quant_backend(second_backend)
    register_quant_backend(second_backend, failing_backend)
    model = torch.nn.ModuleList([first, second])
    original_weights = [module.weight.detach().clone() for module in model]
    try:
        with pytest.raises(RuntimeError, match="fold failed"), mtq.temporarily_fold_weights(model):
            pass

        for module, original_weight in zip(model, original_weights):
            assert torch.equal(module.weight, original_weight)
            assert module.weight_quantizer.is_enabled
    finally:
        unregister_quant_backend(first_backend)
        unregister_quant_backend(second_backend)


def test_temporarily_fold_weights_rejects_sequential_quantizer():
    qlinear = QuantModuleRegistry.convert(torch.nn.Linear(4, 3, bias=False))
    qlinear.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())

    with (
        pytest.raises(NotImplementedError, match="does not support SequentialQuantizer"),
        mtq.temporarily_fold_weights(qlinear),
    ):
        pass


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


def test_preserve_quantizer_attributes_context_restores_type_and_properties():
    model = QuantModuleRegistry.convert(torch.nn.Linear(4, 3, bias=False))
    model.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
    original_states = [
        quantizer.get_modelopt_state(properties_only=True) for quantizer in model.weight_quantizer
    ]

    with mtq.preserve_quantizer_attributes_context(model):
        mtq.set_quantizer_by_cfg(
            model,
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}}],
        )
        assert isinstance(model.weight_quantizer, TensorQuantizer)

    assert isinstance(model.weight_quantizer, SequentialQuantizer)
    restored_states = [
        quantizer.get_modelopt_state(properties_only=True) for quantizer in model.weight_quantizer
    ]
    assert restored_states == original_states


@pytest.mark.parametrize(
    ("case_sensitive", "layout"),
    [
        (True, "plain"),
        (True, "cached-weight"),
        (True, "shared-layer"),
        (True, "dotted-name"),
        (True, "empty-name"),
        (True, "custom-traversal"),
        (True, "equal-modules"),
        (True, "changing-equality"),
        (True, "attribute-lookup"),
        (True, "class-property"),
        (True, "bound-traversal"),
        (True, "newline-name"),
        (True, "scripted"),
        (True, "served-modules"),
        (True, "modules-property"),
        (True, "dict-subclass"),
        (True, "state-subclass"),
        (True, "dict-property"),
        (True, "state-key"),
        (True, "calibrator-setattr"),
        (True, "class-metaclass"),
        (True, "late-wildcard"),
        (True, "patched-traversal"),
        (True, "str-name"),
        (True, "foreign-member"),
        (True, "parametrized"),
        (True, "tensor-attribute"),
        (True, "held-tuple"),
        (True, "offload-hooks"),
        (True, "calibrator-state"),
        (True, "device-context"),
        (True, "torch-mode"),
        (True, "python-dispatcher"),
        (False, "plain"),
    ],
    ids=[
        "subtree",
        "cached-weight",
        "shared-layer",
        "dotted-name",
        "empty-name",
        "custom-traversal",
        "equal-modules",
        "changing-equality",
        "attribute-lookup",
        "class-property",
        "bound-traversal",
        "newline-name",
        "scripted",
        "served-modules",
        "modules-property",
        "dict-subclass",
        "state-subclass",
        "dict-property",
        "state-key",
        "calibrator-setattr",
        "class-metaclass",
        "late-wildcard",
        "patched-traversal",
        "str-name",
        "foreign-member",
        "parametrized",
        "tensor-attribute",
        "held-tuple",
        "offload-hooks",
        "calibrator-state",
        "device-context",
        "torch-mode",
        "python-dispatcher",
        "case-insensitive",
    ],
)
def test_set_quantizer_by_cfg_finds_named_quantizers_in_one_walk(
    monkeypatch, case_sensitive, layout
):
    """Entries naming one quantizer each, as a finalized AutoQuantize search's, act as wildcards do."""
    monkeypatch.setattr(quant_conversion, "_CASE_SENSITIVE_NAMES", case_sensitive)
    monkeypatch.setattr(quant_conversion, "_EXACT_ENTRIES_FOR_CHECK", 1)

    class FourBitHidingSequential(torch.nn.Sequential):
        hiding = False

        def named_modules(self, *args, **kwargs):
            # A traversal that leaves out 4-bit quantizers, so what it lists changes as entries
            # apply: all its paths are listed at the first exact entry.
            for name, module in super().named_modules(*args, **kwargs):
                if not (
                    self.hiding and isinstance(module, TensorQuantizer) and module.num_bits == 4
                ):
                    yield name, module

    class EqualHolder(torch.nn.Module):
        def __init__(self, num_bits=8):
            super().__init__()
            self.weight_quantizer = TensorQuantizer()
            self.weight_quantizer.num_bits = num_bits

        def __eq__(self, other):
            # Holders of quantizers of one width are equal, which entries can change.
            return (
                isinstance(other, EqualHolder)
                and other.weight_quantizer.num_bits == self.weight_quantizer.num_bits
            )

        def __hash__(self):
            return 0

    class LookupHolder(torch.nn.Module):
        state_reads = 0

        def __getattribute__(self, name):
            if name == "__dict__":
                type(self).state_reads += 1
            return super().__getattribute__(name)

    class State(dict):
        gets = 0

        def get(self, *args):
            type(self).gets += 1
            return super().get(*args)

    class DictHolder(torch.nn.Module):
        reads = 0

        @property
        def __dict__(self):
            type(self).reads += 1
            return torch.nn.Module.__dict__["__dict__"].__get__(self)

    class RecordingMeta(type):
        hooks = 0

        def __hash__(cls):
            RecordingMeta.hooks += 1
            return type.__hash__(cls)

        def __getattribute__(cls, name):
            if name == "__mro__":
                RecordingMeta.hooks += 1
            return type.__getattribute__(cls, name)

    class MetaHolder(torch.nn.Module, metaclass=RecordingMeta):
        pass

    class Key(str):
        compared = 0

        __hash__ = str.__hash__

        def __eq__(self, other):
            type(self).compared += 1
            return str.__eq__(self, other)

    class ClassHolder(torch.nn.Module):
        @property
        def __class__(self):
            return ClassHolder

    class ServedHolder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.__dict__["_served"] = self.__dict__.pop("_modules")

        def __getattr__(self, name):
            if name == "_modules":
                return self.__dict__["_served"]
            return super().__getattr__(name)

    class ModulesHolder(torch.nn.Module):
        @property
        def _modules(self):
            return self.__dict__["_modules"]

        @_modules.setter
        def _modules(self, value):
            self.__dict__["_modules"] = value

    class ForeignMember(torch.nn.Module):
        def set_from_attribute_config(self, attribute_cfg):
            pass

    stock_named_modules = torch.nn.Module.named_modules

    @contextlib.contextmanager
    def root_filtered():
        # An integration's wrap of nn.Module.named_modules that leaves a quantizer out of walks of
        # the whole model.
        def named_modules(self, *args, **kwargs):
            for name, module in stock_named_modules(self, *args, **kwargs):
                if args or kwargs or name != "7.weight_quantizer":
                    yield name, module

        torch.nn.Module.named_modules = named_modules
        try:
            yield
        finally:
            torch.nn.Module.named_modules = stock_named_modules

    contexts = {
        "patched-traversal": root_filtered,
        "device-context": lambda: torch.device("cpu"),
        "torch-mode": lambda: FlopCounterMode(display=False),
        "python-dispatcher": enable_python_dispatcher,
    }
    context = contexts.get(layout, contextlib.nullcontext)

    def quantized():
        layers = (torch.nn.Linear(4, 4) for _ in range(8))
        traversal = layout == "custom-traversal"
        model = (FourBitHidingSequential if traversal else torch.nn.Sequential)(*layers)
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
        if layout == "cached-weight":
            # A weak reference, as a GGML quantizer's packed-weight cache holds, keeps the subtree walk.
            model[0].weight_quantizer._quantizer_cache = weakref.ref(model[0].weight)
        elif layout == "shared-layer":
            # A second name for the last layer, which named_modules skips.
            model.add_module("8", model[7])
        elif layout == "dotted-name":
            # setattr, unlike add_module, registers a name with a dot, which named_modules lists.
            setattr(model, "8.weight_quantizer", TensorQuantizer())
        elif layout == "empty-name":
            # Or an empty name, below which named_modules lists names with no dot before them.
            holder = torch.nn.Module()
            holder.weight_quantizer = TensorQuantizer()
            setattr(model, "", holder)
        elif layout == "equal-modules":
            # Two modules that compare equal, the second of which named_modules skips.
            model.add_module("8", EqualHolder())
            model.add_module("9", EqualHolder())
        elif layout == "changing-equality":
            # Or that come to compare equal once an entry makes the first one's quantizer 4-bit.
            model.add_module("8", EqualHolder())
            model.add_module("9", EqualHolder(num_bits=4))
        elif layout == "attribute-lookup":
            # A class's own attribute lookup runs as the walks look its modules up, and its own
            # __class__ as isinstance checks them.
            model.add_module("8", LookupHolder())
        elif layout == "class-property":
            model.add_module("8", ClassHolder())
        elif layout == "bound-traversal":
            # Or an instance whose named_modules lists another module's paths, here without the
            # "inner" level.
            holder = torch.nn.Module()
            holder.inner = torch.nn.Module()
            holder.inner.weight_quantizer = TensorQuantizer()
            holder.named_modules = holder.inner.named_modules
            model.add_module("8", holder)
        elif layout == "newline-name":
            # Or a name ending in a newline, which normalizing a fused-experts name keeps.
            experts = torch.nn.Module()
            experts.weight_quantizers = torch.nn.Module()
            setattr(experts.weight_quantizers, "0\n", TensorQuantizer())
            model.add_module("8", experts)
        elif layout == "scripted":
            # TorchScript keeps its children in a dict of its own.
            model.add_module("8", torch.jit.script(torch.nn.Linear(4, 4)))
        elif layout == "served-modules":
            # Or a class serves them, or looks them up, through code of its own.
            model.add_module("8", ServedHolder())
        elif layout == "modules-property":
            model.add_module("8", ModulesHolder())
        elif layout == "dict-subclass":
            model.add_module("8", torch.nn.Module())
            model[8].__dict__["_modules"] = type("Modules", (dict,), {})()
        elif layout == "state-subclass":
            model.add_module("8", torch.nn.Module())
            model[8].__dict__ = State(model[8].__dict__)
        elif layout == "dict-property":
            model.add_module("8", DictHolder())
        elif layout == "state-key":
            # A key equal to a name the check, but no walk, would look up.
            model.add_module("8", torch.nn.Module())
            model[8].weight_quantizer = TensorQuantizer()
            model[8].weight_quantizer.__dict__[Key("_hf_hook")] = None
        elif layout == "class-metaclass":
            # A class whose metaclass's hash and attribute lookup no walk runs.
            model.add_module("8", MetaHolder())
        elif layout == "calibrator-setattr":
            # Configuring a quantizer sets its calibrator's axis through the calibrator's class.
            model.add_module("8", torch.nn.Module())
            model[8].weight_quantizer = TensorQuantizer()
            model[8].weight_quantizer.__dict__["_calibrator"] = TensorQuantizer()
        elif layout == "str-name":
            # A name's own hash and equality run as the subtree walk looks it up.
            model.add_module("8", torch.nn.Module())
            model[8]._modules[type("Name", (str,), {})("weight_quantizer")] = TensorQuantizer()
        elif layout == "foreign-member":
            # A SequentialQuantizer configures its members, here one of the caller's class.
            model.add_module("8", torch.nn.Module())
            model[8].weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
            model[8].weight_quantizer[1] = ForeignMember()
        elif layout == "parametrized":
            # A parametrization runs the caller's module as the quantizer reads its amax.
            parametrize.register_parametrization(
                model[7].weight_quantizer, "_amax", torch.nn.Identity()
            )
        elif layout == "tensor-attribute":
            # Code can read what a tensor holds.
            model[7].weight_quantizer._amax.read = print
        elif layout == "held-tuple":
            # Or a modelopt tuple holds.
            model[7].weight_quantizer._quantizer_cache = _CacheKey(
                print, (), (), torch.float32, torch.device("cpu"), 0
            )
        elif layout == "offload-hooks":
            # Accelerate's offloading wraps forward, which configuring a quantizer never runs.
            model[0].weight_quantizer._hf_hook = SimpleNamespace(offload=True)
            model[0].weight_quantizer._old_forward = len
            model[0].weight_quantizer.forward = print
        elif layout == "calibrator-state":
            # Configuring a quantizer only replaces its calibrator or sets its axis.
            model[7].weight_quantizer._calibrator._quant_func = print
        model.hiding = traversal
        return model

    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    for layer in range(8):
        quant_cfg.append({"quantizer_name": f"{layer}.weight_quantizer", "cfg": {"num_bits": 4}})
        quant_cfg.append({"quantizer_name": f"{layer}.input_quantizer", "enable": True})
    if layout == "changing-equality":
        quant_cfg.append({"quantizer_name": "8.weight_quantizer", "cfg": {"num_bits": 4}})
        quant_cfg.append({"quantizer_name": "9.weight_quantizer", "enable": True})
    elif layout == "bound-traversal":
        # The name the delegated named_modules lists, which the holder's _modules lack.
        quant_cfg.append({"quantizer_name": "8.weight_quantizer", "enable": True})
    elif layout == "late-wildcard":
        # A wildcard entry with a cfg walks the model once. When it replaces no quantizer, the named
        # entry after it walks its subtree; when it replaces some, the named entry walks the model.
        quant_cfg.append(
            {"quantizer_name": "*output_quantizer", "cfg": {"num_bits": 8}, "enable": False}
        )
        quant_cfg.append({"quantizer_name": "0.output_quantizer", "enable": True})
        quant_cfg.append(
            {"quantizer_name": "*output_quantizer", "cfg": [{"num_bits": 8}] * 2, "enable": False}
        )
        quant_cfg.append({"quantizer_name": "1.output_quantizer", "enable": True})
    # Replacing a quantizer with a SequentialQuantizer, and back, changes the names to find.
    first_replacing = len(quant_cfg)
    quant_cfg.append({"quantizer_name": "2.weight_quantizer", "cfg": [{"num_bits": 4}] * 2})
    quant_cfg.append({"quantizer_name": "2.weight_quantizer.1", "enable": False})
    quant_cfg.append({"quantizer_name": "3.weight_quantizer", "cfg": [{"num_bits": 8}] * 2})
    quant_cfg.append({"quantizer_name": "3.weight_quantizer", "cfg": {"num_bits": 8}})
    if layout == "shared-layer":
        quant_cfg.append({"quantizer_name": "8.weight_quantizer", "enable": False})
    elif layout == "dotted-name":
        quant_cfg.append({"quantizer_name": "8.weight_quantizer", "enable": True})
    elif layout == "empty-name":
        quant_cfg.append({"quantizer_name": "weight_quantizer", "enable": True})
    elif layout == "equal-modules":
        quant_cfg.append({"quantizer_name": "9.weight_quantizer", "enable": False})
    elif layout == "newline-name":
        quant_cfg.append({"quantizer_name": "8.weight_quantizer\n", "enable": True})
    # The same names as one-character wildcards, matched by walking the model.
    walked_cfg = [
        entry
        if quant_conversion._WILDCARD_RE.search(name := entry["quantizer_name"])
        else {**entry, "quantizer_name": f"[{name[0]}]{name[1:]}"}
        for entry in quant_cfg
    ]

    def states(model):
        return {
            name: (
                type(module).__name__,
                isinstance(module, TensorQuantizer)
                and module.get_modelopt_state(properties_only=True),
            )
            # Every registered module, hidden and skipped ones too.
            for name, module in torch.nn.Module.named_modules(model, remove_duplicate=False)
            if isinstance(module, (TensorQuantizer, SequentialQuantizer))
        }

    walked = quantized()
    with context():
        mtq.set_quantizer_by_cfg(walked, walked_cfg)
    named = quantized()
    walks = 0

    def count_walks(frame, event, arg):
        # A walk of the whole model: torch's named_modules called on it, outside any other walk.
        nonlocal walks
        if event == "call" and frame.f_code is stock_named_modules.__code__:
            local = frame.f_locals
            walks += local.get("self") is named and local.get("memo") is None

    sys.setprofile(count_walks)
    try:
        with context():
            mtq.set_quantizer_by_cfg(named, quant_cfg)
    finally:
        sys.setprofile(None)
    # The check refuses a class's own lookups, a state dict of a subclass or a key of a str
    # subclass without running them, and reads classes without their metaclass. (Comparing the
    # states compares the keys.)
    assert LookupHolder.state_reads == State.gets == DictHolder.reads == Key.compared == 0
    assert RecordingMeta.hooks == 0
    assert states(named) == states(walked)
    fast = (
        "plain",
        "cached-weight",
        "offload-hooks",
        "calibrator-state",
        "device-context",
        "dict-property",
        "class-metaclass",
        "late-wildcard",
    )
    if case_sensitive and layout in fast:
        # Only the "*" entry walks the model, and the entries from the first replacement on.
        late_walks = 3 if layout == "late-wildcard" else 0
        assert walks == 1 + late_walks + len(quant_cfg) - first_replacing
    else:
        # Every entry walks the model: where fnmatch ignores case, or named_modules can list other
        # paths than the model registers, or the model keeps them out of a plain dict, or a
        # quantizer, or a mode or the Python dispatcher, can run the caller's code.
        assert walks == len(quant_cfg)


@pytest.mark.usefixtures("check_at_the_first_named_entry")
@pytest.mark.parametrize("weight_quantizer", ["*weight_quantizer", "0.weight_quantizer"])
@pytest.mark.parametrize("value", ["unhashable-class", "no-module-class"])
def test_set_quantizer_by_cfg_stores_an_opaque_config_value(monkeypatch, weight_quantizer, value):
    """An entry stores a value whose class cannot be hashed, or has no module, as it did before: a
    wildcard call never scans its entries, and a named one counts such a value as the caller's code."""

    class UnhashableMeta(type):
        __hash__ = None

    namespace: dict = {}
    exec("policy_class = type('Policy', (), {})", namespace)  # where no __name__ names a module
    policy_class = (
        UnhashableMeta("Policy", (), {})
        if value == "unhashable-class"
        else namespace["policy_class"]
    )

    scans = []
    scan = quant_conversion._runs_callers_code
    monkeypatch.setattr(
        quant_conversion,
        "_runs_callers_code",
        lambda obj, seen: scans.append(obj) or scan(obj, seen),
    )
    policy = policy_class()
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
    scans.clear()
    cfg = {"num_bits": 8, "backend_extra_args": {"policy": policy}}
    mtq.set_quantizer_by_cfg(model, [{"quantizer_name": weight_quantizer, "cfg": cfg}])
    assert model[0].weight_quantizer.backend_extra_args["policy"] is policy
    assert bool(scans) == (weight_quantizer == "0.weight_quantizer")


@pytest.mark.usefixtures("check_at_the_first_named_entry")
def test_set_quantizer_by_cfg_reads_an_entry_without_its_getattr():
    """An entry whose pydantic state is incomplete applies as it did before: the check reads its
    state without pydantic's __getattr__, which reads the entry's extra state."""

    class Extra(dict):
        def __bool__(self):
            raise AssertionError("the entry's extra state was read")

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
    entry = QuantizerCfgEntry(quantizer_name="0.weight_quantizer", enable=False)
    object.__setattr__(entry, "__pydantic_extra__", Extra())
    object.__delattr__(entry, "__pydantic_private__")
    mtq.set_quantizer_by_cfg(model, [entry])
    assert not model[0].weight_quantizer.is_enabled


def test_set_quantizer_by_cfg_reads_each_entry_after_the_entries_before_it():
    """An entry of the caller's own class is read once the entries before it have applied, as
    before: counting a call's named entries reads only modelopt's own."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))

    class Late(QuantizerCfgEntry):
        def __getitem__(self, key):
            if key == "quantizer_name":
                assert not model[0].input_quantizer.is_enabled, (
                    "read before the first entry applied"
                )
            return super().__getitem__(key)

    mtq.set_quantizer_by_cfg(
        model,
        [
            {"quantizer_name": "*input_quantizer", "enable": False},
            Late(quantizer_name="0.weight_quantizer", enable=False),
        ],
    )
    assert not model[0].weight_quantizer.is_enabled


def test_set_quantizer_by_cfg_matches_a_filter_function_name():
    """An entry whose name is a filter function, which only an unvalidated entry can hold, matches
    the quantizers it accepts, as it did before."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
    entry = QuantizerCfgEntry.model_construct(
        quantizer_name=lambda name: name.endswith("weight_quantizer"),
        cfg=None,
        enable=False,
        parent_class=None,
    )
    mtq.set_quantizer_by_cfg(model, [entry])
    assert not model[0].weight_quantizer.is_enabled
    assert model[0].input_quantizer.is_enabled


@pytest.mark.usefixtures("check_at_the_first_named_entry")
def test_set_quantizer_by_cfg_reads_class_namespaces_only_with_str_keys():
    """A named entry acts as its wildcard does on a model whose module class keeps a key of a str
    subclass in its namespace: looking a name up there compares the keys of equal hash, so the
    check looks only in namespaces of plain-str keys."""
    armed = []

    def configured(weight_quantizer, sub_quantizer):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))

        class Key(str):
            # Equal in hash to a name only the check looks up.
            def __hash__(self):
                return hash("__eq__")

            def __eq__(self, other):
                if armed and model[1] is not model[0]:
                    model[1] = model[0]  # named_modules skips the second name
                return False

        model.add_module("2", type("Holder", (torch.nn.Module,), {Key("holder"): None})())
        armed.append(True)
        try:
            mtq.set_quantizer_by_cfg(
                model,
                [
                    {"quantizer_name": weight_quantizer, "cfg": {"num_bits": 4}},
                    {"quantizer_name": sub_quantizer, "enable": False},
                ],
            )
        finally:
            armed.clear()
        return {
            name: module.get_modelopt_state(properties_only=True)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    walked = configured("[0].weight_quantizer", "[1].weight_quantizer")
    assert configured("0.weight_quantizer", "1.weight_quantizer") == walked


@pytest.mark.usefixtures("check_at_the_first_named_entry")
def test_set_quantizer_by_cfg_walks_a_model_the_check_cannot_read():
    """A named entry acts as its wildcard does on a model holding a non-module child: both apply
    the entry's earlier matches, then fail as named_modules reaches that child."""

    def configured(weight_quantizer):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
        model._modules["2"] = object()
        with pytest.raises(AttributeError):
            mtq.set_quantizer_by_cfg(model, [{"quantizer_name": weight_quantizer, "enable": False}])
        return model[0].weight_quantizer.is_enabled

    assert configured("0.weight_quantizer") == configured("[0].weight_quantizer")


@pytest.mark.parametrize("below", [1, 0], ids=["below", "at"])
def test_set_quantizer_by_cfg_checks_the_model_only_for_enough_named_entries(monkeypatch, below):
    """A call checks the model, at its first named entry, only when it names enough quantizers to
    repay the check; one with fewer walks the model for each, as before."""
    monkeypatch.setattr(quant_conversion, "_CASE_SENSITIVE_NAMES", True)
    checks = []
    check = quant_conversion._allows_subtree_walk
    monkeypatch.setattr(
        quant_conversion, "_allows_subtree_walk", lambda model: checks.append(model) or check(model)
    )
    named = quant_conversion._EXACT_ENTRIES_FOR_CHECK - below
    model = torch.nn.Sequential(*(torch.nn.Linear(2, 2) for _ in range(named)))
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(1, 2)))
    checks.clear()
    quant_cfg = [{"quantizer_name": f"{i}.weight_quantizer", "enable": False} for i in range(named)]
    mtq.set_quantizer_by_cfg(model, [{"quantizer_name": "*", "enable": True}, *quant_cfg])
    assert len(checks) == (not below)
    assert not any(layer.weight_quantizer.is_enabled for layer in model)


@pytest.fixture
def check_at_the_first_named_entry(monkeypatch):
    # These calls name one or two quantizers, in lowercase, which fnmatch matches alike everywhere.
    monkeypatch.setattr(quant_conversion, "_EXACT_ENTRIES_FOR_CHECK", 1)
    monkeypatch.setattr(quant_conversion, "_CASE_SENSITIVE_NAMES", True)


@pytest.mark.usefixtures("check_at_the_first_named_entry")
@pytest.mark.parametrize(
    ("trigger", "nested"),
    [
        ("calibrator", False),
        ("calibrator", True),
        ("hook", False),
        ("setter", False),
        ("quantizer", False),
        ("getter", False),
        ("config", False),
        ("wrapped", False),
        ("fields", False),
        ("method", False),
        ("tensor", False),
        ("function-mode", False),
        ("dispatch-mode", False),
        ("calibrator-key", False),
    ],
    ids=[
        "later-entry",
        "nested-call",
        "registration-hook",
        "parent-setter",
        "quantizer-class",
        "parent-getter",
        "config-class",
        "wrapped-value",
        "fields-set",
        "instance-method",
        "tensor-subclass",
        "function-mode",
        "dispatch-mode",
        "calibrator-key",
    ],
)
def test_set_quantizer_by_cfg_finds_named_quantizers_after_a_callback_shares_a_layer(
    trigger, nested
):
    """A named entry acts as its wildcard does after code run mid-call shares a layer: a calibrator
    the caller supplies, for a later entry of that call or in a call the calibrator makes, a buffer
    registration hook, the __setattr__ of a parent whose quantizer an entry replaces, a quantizer of
    the caller's class, the __getattr__ a parent_class entry runs to find a match's parent, an
    attribute config of the caller's class, a value wrapped in a builtin or kept in pydantic's
    fields set, a method or tensor subclass a quantizer holds, a key of a str subclass in its
    calibrator's dict, or a torch function or dispatch mode."""

    def configured(weight_quantizer, sub_quantizer):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
        sub_entry = {"quantizer_name": sub_quantizer, "enable": False}

        def share():
            if model[1] is not model[0]:
                model[1] = model[0]  # named_modules skips the second name
                if nested:
                    mtq.set_quantizer_by_cfg(model, [sub_entry])

        class SharingCalibrator(MaxCalibrator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                share()

        def sharing_hook(module, name, buffer):
            share()

        class SharingQuantizer(TensorQuantizer):
            def set_from_attribute_config(self, *args, **kwargs):
                super().set_from_attribute_config(*args, **kwargs)
                share()

        class SharingSequential(torch.nn.Sequential):
            def __getattr__(self, name):
                if name == "0":
                    share()
                return super().__getattr__(name)

        class SharingConfig(QuantizerAttributeConfig):
            def model_copy(self, *args, **kwargs):
                share()
                return super().model_copy(*args, **kwargs)

        class SharingKey(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                share()
                return str.__eq__(self, other)

        class SharingDict(dict):
            def __len__(self):
                share()
                return super().__len__()

        class SharingSet(set):
            def __copy__(self):
                share()
                return set(self)

        class SharingTensor(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                share()
                return super().__torch_function__(func, types, args, kwargs or {})

        class SharingFunctionMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                share()
                return func(*args, **(kwargs or {}))

        class SharingDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                share()
                return func(*args, **(kwargs or {}))

        entry = {"quantizer_name": weight_quantizer}
        if trigger == "calibrator":
            # The calibrator as a class, or as a constructor tuple.
            cfg = {"calibrator": SharingCalibrator if nested else (SharingCalibrator, (), {})}
        elif trigger in ("hook", "function-mode", "dispatch-mode"):
            # Pinning amax registers it as a buffer, under the hook, with tensor operations.
            del model[0].weight_quantizer._amax
            cfg = {"num_bits": 8, "axis": None, "constant_amax": 1.0}
        elif trigger == "calibrator-key":
            # Setting the axis compares the calibrator dict's keys of the same hash.
            calibrator = model[0].weight_quantizer._calibrator
            calibrator.__dict__[SharingKey("_axis")] = calibrator.__dict__.pop("_axis")
            cfg = {"num_bits": 4, "axis": 0}
        elif trigger == "tensor":
            # Pinning amax copies it into the buffer the quantizer holds.
            model[0].weight_quantizer._amax = torch.tensor(1.0).as_subclass(SharingTensor)
            cfg = {"num_bits": 8, "axis": None, "constant_amax": 1.0}
        elif trigger == "wrapped":
            # Unvalidated, so the proxy stays, and the quantizer reads its length.
            cfg = QuantizerAttributeConfig.model_construct(
                num_bits=4, backend_extra_args=MappingProxyType(SharingDict())
            )
        elif trigger == "fields":
            # model_copy() copies the fields set.
            cfg = QuantizerAttributeConfig.model_construct(
                _fields_set=SharingSet({"num_bits"}), num_bits=4
            )
        elif trigger == "method":
            quantizer = model[0].weight_quantizer
            configure = quantizer.set_from_attribute_config

            def sharing_configure(*args, **kwargs):
                configure(*args, **kwargs)
                share()

            quantizer.set_from_attribute_config = sharing_configure
            cfg = {"num_bits": 4}
        elif trigger == "quantizer":
            model[0].weight_quantizer.__class__ = SharingQuantizer
            cfg = {"num_bits": 4}
        elif trigger == "getter":
            # get_submodule looks the parent up through the model's __getattr__.
            model.__class__ = SharingSequential
            entry["parent_class"] = "nn.Linear"
            cfg = {"num_bits": 4}
        elif trigger == "config":
            cfg = SharingConfig(num_bits=4)
        else:

            class SharingLinear(type(model[0])):
                def __setattr__(self, name, value):
                    super().__setattr__(name, value)
                    if isinstance(value, SequentialQuantizer):
                        share()

            model[0].__class__ = SharingLinear
            # A list replaces the quantizer with a SequentialQuantizer, through the parent.
            cfg = [{"num_bits": 4}, {"num_bits": 8}]
        register = torch.nn.modules.module.register_module_buffer_registration_hook
        handle = register(sharing_hook) if trigger == "hook" else None
        mode = {"function-mode": SharingFunctionMode, "dispatch-mode": SharingDispatchMode}
        try:
            with mode[trigger]() if trigger in mode else contextlib.nullcontext():
                mtq.set_quantizer_by_cfg(
                    model,
                    [{**entry, "cfg": cfg}] + ([] if nested else [sub_entry]),
                )
        finally:
            if handle is not None:
                handle.remove()
        # The states compare every attribute a quantizer holds.
        vars(model[0].weight_quantizer).pop("set_from_attribute_config", None)
        return {
            name: module.get_modelopt_state(properties_only=True)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    walked = configured("[0].weight_quantizer", "[1].weight_quantizer")
    assert configured("0.weight_quantizer", "1.weight_quantizer") == walked


@pytest.mark.usefixtures("check_at_the_first_named_entry")
def test_set_quantizer_by_cfg_finds_named_quantizers_after_a_calibrator_aliases_a_later_match():
    """A named entry acts as its wildcard does when the calibrator it builds puts a quantizer the
    walk has passed in the place of one of the entry's later matches."""

    def configured(weight_quantizer):
        model = torch.nn.Module()
        model.prior = TensorQuantizer()
        # Both slots' names normalize to ``weight_quantizer``, so one entry matches both.
        model.weight_quantizers = torch.nn.ModuleList([TensorQuantizer(), TensorQuantizer()])

        class AliasingCalibrator(MaxCalibrator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                model.weight_quantizers[1] = model.prior  # named_modules skips the second name

        cfg = {"num_bits": 4, "calibrator": AliasingCalibrator}
        mtq.set_quantizer_by_cfg(model, [{"quantizer_name": weight_quantizer, "cfg": cfg}])
        return {
            name: module.get_modelopt_state(properties_only=True)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    assert configured("weight_quantizer") == configured("[w]eight_quantizer")


@pytest.mark.usefixtures("check_at_the_first_named_entry")
@pytest.mark.parametrize(
    ("kind", "same_call"),
    [("calibrator", False), ("axis-property", False), ("rotation", False), ("rotation", True)],
    ids=["installed-calibrator", "axis-property", "installed-rotation", "rotation-entry"],
)
def test_set_quantizer_by_cfg_finds_named_quantizers_after_a_configured_object_shares_a_layer(
    kind, same_call
):
    """A named entry acts as its wildcard does after a calibrator or rotation config of the caller's
    class, set by an earlier call or entry, shares a layer when a later entry calls into it."""
    model = None

    def share():
        model[1] = model[0]  # named_modules skips the second name

    class SharingCalibrator(MaxCalibrator):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "_axis" and value == 0:
                share()

    class AxisSharingCalibrator(MaxCalibrator):
        @property
        def _axis(self):
            return self.__dict__.get("axis")

        @_axis.setter
        def _axis(self, value):
            self.__dict__["axis"] = value
            if value == 0:
                share()

    class SharingRotation(RotateConfig):
        def model_copy(self, *args, **kwargs):
            share()
            return super().model_copy(*args, **kwargs)

    def configured(weight_quantizer, sub_quantizer):
        nonlocal model
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 4)))
        if kind in ("calibrator", "axis-property"):
            setting = {
                "calibrator": SharingCalibrator if kind == "calibrator" else AxisSharingCalibrator
            }
            # The quantizer keeps its calibrator until after an entry sets its axis.
            trigger = {"cfg": {"num_bits": 4, "axis": 0}}
        else:
            setting = {"rotate": SharingRotation(enable=True)}
            # Disabling a quantizer copies its rotation config.
            trigger = {"enable": False}
        setting_entries = [{"quantizer_name": weight_quantizer, "cfg": setting}]
        entries = [
            {"quantizer_name": weight_quantizer, **trigger},
            {"quantizer_name": sub_quantizer, "cfg": {"num_bits": 6}},
        ]
        if same_call:
            mtq.set_quantizer_by_cfg(model, setting_entries + entries)
        else:
            mtq.set_quantizer_by_cfg(model, setting_entries)
            mtq.set_quantizer_by_cfg(model, entries)
        assert model[1] is model[0]
        return {
            name: module.get_modelopt_state(properties_only=True)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    walked = configured("[0].weight_quantizer", "[1].weight_quantizer")
    assert configured("0.weight_quantizer", "1.weight_quantizer") == walked


@pytest.mark.parametrize(
    "attribute_cfg",
    [
        {
            "num_bits": (2, 1),
            "axis": None,
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        },
        {"num_bits": (4, 3), "axis": None, "use_constant_amax": True},
        {"num_bits": 8, "axis": 0, "unsigned": True, "narrow_range": True, "type": "dynamic"},
        {"num_bits": "custom", "backend": "custom_backend", "backend_extra_args": {"offset": 0.25}},
        {"enable": False},
    ],
)
def test_attribute_config_from_modelopt_state_round_trips_yaml(attribute_cfg):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(**attribute_cfg))
    state = quantizer.get_modelopt_state(properties_only=True)
    restored_cfg = TensorQuantizer.attribute_config_from_modelopt_state(
        yaml.safe_load(yaml.safe_dump(state))
    )
    restored = TensorQuantizer(QuantizerAttributeConfig(**restored_cfg))
    assert restored.get_modelopt_state(properties_only=True) == state
