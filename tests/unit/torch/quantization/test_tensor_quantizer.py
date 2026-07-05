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

"""Direct unit tests for the TensorQuantizer contract.

Complements ``test_tensor_quantizer_cpu.py`` (which runs the shared
``TensorQuantizerTester`` suite) with direct, hand-computed tests of the class's own
contract: enable/disable states, amax get/set semantics, pre_quant_scale, exact
fake-quant forward values for INT8/FP8 on tiny known tensors, state serialization
round-trips, flag/error paths, repr stability and dtype handling. All tests are
CPU-only, hermetic and deterministic.
"""

import warnings

import pytest
import torch
from torch import nn

from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    HardDisabledTensorQuantizer,
    is_registered_quant_backend,
    register_quant_backend,
    unregister_quant_backend,
)


def _tq(**kwargs):
    """Build a TensorQuantizer from QuantizerAttributeConfig kwargs."""
    return TensorQuantizer(QuantizerAttributeConfig(**kwargs))


class TestConstructionDefaults:
    def test_default_attributes(self):
        tq = TensorQuantizer()
        assert tq.num_bits == 8
        assert tq.axis is None
        assert tq.block_sizes is None
        assert tq.amax is None
        assert tq.fake_quant is True
        assert tq.unsigned is False
        assert tq.narrow_range is False
        assert tq.is_enabled
        assert tq._if_quant is True
        assert tq._if_calib is False
        assert isinstance(tq._calibrator, calib.MaxCalibrator)
        assert tq.pre_quant_scale is None
        assert tq.bias is None

    def test_constructor_amax_argument(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(num_bits=8), amax=2.5)
        assert isinstance(tq.amax, torch.Tensor)
        assert tq.amax.item() == pytest.approx(2.5)
        # amax passed as array-like is converted to a tensor buffer
        tq2 = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)), amax=[1.0, 2.0])
        assert torch.equal(tq2.amax, torch.tensor([1.0, 2.0]))

    def test_constructor_if_flags(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(), if_quant=False, if_calib=True)
        assert tq._if_quant is False
        assert tq._if_calib is True

    def test_set_from_attribute_config_rejects_unknown_attribute(self):
        tq = TensorQuantizer()
        with pytest.raises(AssertionError, match="not a valid"):
            tq.set_from_attribute_config({"not_a_real_attribute": 1})

    def test_setters_propagate_to_calibrator(self):
        tq = TensorQuantizer()
        tq.num_bits = 4
        assert tq.num_bits == 4
        assert tq._calibrator._num_bits == 4

        tq.axis = 1
        assert tq.axis == 1
        assert tq._calibrator._axis == 1

        tq.unsigned = True
        assert tq.unsigned is True
        assert tq._calibrator._unsigned is True

    def test_block_sizes_setter_clears_axis(self):
        tq = _tq(num_bits=8, axis=0)
        assert tq.axis == 0
        tq.block_sizes = {-1: 16}
        assert tq.block_sizes == {-1: 16}
        assert tq._axis is None

    def test_set_from_attribute_config_block_sizes_clears_axis(self):
        tq = _tq(num_bits=8, axis=0)
        tq.set_from_attribute_config({"block_sizes": {-1: 16}})
        assert tq._axis is None
        assert tq._calibrator._axis is None


class TestEnableDisable:
    def test_enable_disable_roundtrip(self):
        tq = TensorQuantizer()
        assert tq.is_enabled
        tq.disable()
        assert not tq.is_enabled
        tq.enable()
        assert tq.is_enabled

    def test_disabled_forward_is_identity(self):
        tq = _tq(enable=False)
        x = torch.tensor([0.123, -4.56, 789.0])
        out = tq(x)
        # Disabled quantizer must return the input tensor unmodified (same object).
        assert out is x
        # Disabled forward still records the input dtype (used for model saving).
        assert tq._input_dtype == torch.float32

    def test_disable_quant_forward_is_identity(self):
        tq = TensorQuantizer()
        tq.disable_quant()
        assert tq._if_quant is False
        x = torch.tensor([1.0, 2.0, 3.0])
        assert tq(x) is x
        tq.enable_quant()
        assert tq._if_quant is True
        assert not torch.equal(tq(torch.tensor([0.4, 1.6, 127.0])), torch.tensor([0.4, 1.6, 127.0]))

    def test_empty_tensor_forward_early_return(self):
        tq = TensorQuantizer()
        x = torch.empty(0)
        assert tq(x) is x

    def test_disabled_forward_still_applies_pre_quant_scale(self):
        # NOTE: documents current behavior; arguably a bug because a disabled
        # quantizer is documented to "bypass the module", yet pre_quant_scale is
        # applied before the `self._disabled` check in `forward`, so the input is
        # still scaled. (Intentional for smoothquant folding, but surprising for
        # a module reporting `is_enabled == False`.)
        tq = TensorQuantizer()
        tq.pre_quant_scale = 2.0
        tq.disable()
        x = torch.tensor([1.0, -3.0])
        out = tq(x)
        assert torch.equal(out, torch.tensor([2.0, -6.0]))


class TestAmaxSemantics:
    def test_scalar_amax_set_get(self):
        tq = TensorQuantizer()
        tq.amax = 1.5
        assert isinstance(tq.amax, torch.Tensor)
        assert tq.amax.item() == pytest.approx(1.5)
        # In-place update through the setter keeps the same buffer
        buf = tq._amax
        tq.amax = 3.0
        assert tq._amax is buf
        assert tq.amax.item() == pytest.approx(3.0)

    def test_per_axis_amax_set_get(self):
        tq = _tq(num_bits=8, axis=0)
        amax = torch.tensor([[1.0], [2.0], [3.0]])
        tq.amax = amax
        assert torch.equal(tq.amax, amax)

    def test_amax_cannot_be_set_to_none(self):
        tq = TensorQuantizer()
        tq.amax = 1.0
        with pytest.raises(AssertionError, match="amax cannot be set to None"):
            tq.amax = None

    def test_amax_shape_change_raises(self):
        tq = TensorQuantizer()
        tq.amax = torch.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError, match="Changing shape"):
            tq.amax = torch.tensor([1.0, 2.0, 3.0])

    def test_amax_appears_in_state_dict_only_when_set(self):
        tq = TensorQuantizer()
        assert "_amax" not in tq.state_dict()
        tq.amax = 1.0
        assert "_amax" in tq.state_dict()

    def test_reset_amax(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(), if_calib=True, if_quant=False)
        tq(torch.tensor([1.0, -5.0]))
        tq.load_calib_amax()
        assert tq.amax is not None
        tq.reset_amax()
        assert tq.amax is None
        assert "_amax" not in tq.state_dict()
        # Calibrator statistics are also reset
        assert tq._calibrator.compute_amax() is None

    def test_step_size(self):
        tq = TensorQuantizer()
        tq.amax = 1.27
        # step_size = amax / (2^(8-1) - 1) = 1.27 / 127 = 0.01
        assert tq.step_size.item() == pytest.approx(0.01)

    def test_step_size_without_amax_warns_and_returns_none(self):
        tq = TensorQuantizer()
        with pytest.warns(UserWarning, match="step_size is undefined"):
            assert tq.step_size is None

    def test_step_size_undefined_for_float_quant(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)), amax=448.0)
        with pytest.raises(AssertionError, match="not defined for non-integer"):
            _ = tq.step_size

    def test_dynamic_quantizer_amax_contract(self):
        tq = _tq(type="dynamic")
        assert tq._dynamic is True
        # Without a buffer the property reports None
        assert tq.amax is None
        # Calibration is a no-op for dynamic quantization
        tq.enable_calib()
        assert tq._if_calib is False
        with pytest.raises(AssertionError, match="does not need calibration"):
            tq.load_calib_amax()

    def test_dynamic_quantizer_amax_getter_asserts_after_set(self):
        # NOTE: documents current behavior; arguably a bug because the amax
        # *setter* happily registers a buffer on a dynamic quantizer, after which
        # the *getter* raises AssertionError instead of returning None or refusing
        # the set in the first place.
        tq = _tq(type="dynamic")
        tq.amax = 1.0
        with pytest.raises(AssertionError, match="Dynamic quantization does not have fixed amax"):
            _ = tq.amax


class TestPreQuantScale:
    def test_set_get_and_buffer_registration(self):
        tq = TensorQuantizer()
        tq.pre_quant_scale = 2.0
        assert isinstance(tq.pre_quant_scale, torch.Tensor)
        assert tq.pre_quant_scale.item() == pytest.approx(2.0)
        assert "_pre_quant_scale" in tq.state_dict()

    def test_cannot_set_none(self):
        tq = TensorQuantizer()
        with pytest.raises(AssertionError, match="cannot be set to None"):
            tq.pre_quant_scale = None

    def test_shape_change_raises(self):
        tq = TensorQuantizer()
        tq.pre_quant_scale = torch.tensor([1.0, 2.0])
        with pytest.raises(RuntimeError, match="Changing shape"):
            tq.pre_quant_scale = torch.tensor([1.0, 2.0, 3.0])

    def test_disable_pre_quant_scale_context_manager(self):
        tq = TensorQuantizer()
        tq.amax = 127.0
        tq.pre_quant_scale = 2.0

        x = torch.tensor([0.4, 1.6])
        # With scale: quantize(2 * x) = round([0.8, 3.2]) = [1, 3]
        assert torch.equal(tq(x), torch.tensor([1.0, 3.0]))

        with tq.disable_pre_quant_scale():
            assert tq.pre_quant_scale is None
            # Without scale: round([0.4, 1.6]) = [0, 2]
            assert torch.equal(tq(x), torch.tensor([0.0, 2.0]))

        # Restored on exit
        assert tq.pre_quant_scale is not None
        assert torch.equal(tq(x), torch.tensor([1.0, 3.0]))

    def test_setter_asserts_when_pre_quant_scale_disabled(self):
        tq = TensorQuantizer()
        tq._enable_pre_quant_scale = False
        with pytest.raises(AssertionError):
            tq.pre_quant_scale = 1.0


class TestExactInt8FakeQuant:
    """Hand-computed exact-value tests for the INT8 fake-quant forward path.

    CPU path: out = clamp(round(x * 127/amax), -128, 127) / (127/amax)
    with torch.round using round-half-to-even (ties avoided below).
    """

    def test_per_tensor_amax_127(self):
        # amax=127 -> scale = 127/127 = 1
        # x      = [ 0.0, 0.4, 1.6, -1.6, 126.7, -300.0]
        # round  = [ 0,   0,   2,   -2,   127,   -300  ]
        # clamp  = [ 0,   0,   2,   -2,   127,   -128  ]  (default narrow_range=False)
        tq = TensorQuantizer()
        tq.amax = 127.0
        x = torch.tensor([0.0, 0.4, 1.6, -1.6, 126.7, -300.0])
        expected = torch.tensor([0.0, 0.0, 2.0, -2.0, 127.0, -128.0])
        assert torch.equal(tq(x), expected)

    def test_per_tensor_amax_scaled(self):
        # amax=1.27 -> scale = 127/1.27 = 100
        # x*scale = [ 0.4, 1.1, -1.6, 127, 200 ]
        # round   = [ 0,   1,   -2,   127, 200 ]; clamp -> [0, 1, -2, 127, 127]
        # /scale  = [ 0.0, 0.01, -0.02, 1.27, 1.27 ]
        tq = TensorQuantizer()
        tq.amax = 1.27
        x = torch.tensor([0.004, 0.011, -0.016, 1.27, 2.0])
        expected = torch.tensor([0.0, 0.01, -0.02, 1.27, 1.27])
        assert torch.allclose(tq(x), expected, rtol=0, atol=1e-7)

    def test_narrow_range_clamps_to_minus_127(self):
        # narrow_range=True -> clamp range is [-127, 127] instead of [-128, 127]
        # amax=127 -> scale=1; x = [-128.0, -127.4, 127.0]
        # round = [-128, -127, 127]; clamp(min=-127) -> [-127, -127, 127]
        tq = _tq(num_bits=8, narrow_range=True)
        tq.amax = 127.0
        x = torch.tensor([-128.0, -127.4, 127.0])
        expected = torch.tensor([-127.0, -127.0, 127.0])
        assert torch.equal(tq(x), expected)

    def test_unsigned_range(self):
        # unsigned=True -> max_bound = 2^8 - 1 = 255, min_bound = 0
        # amax=255 -> scale=1; x = [0.0, 0.4, 1.6, 254.7, 300.0]
        # round+clamp = [0, 0, 2, 255, 255]
        tq = _tq(num_bits=8, unsigned=True)
        tq.amax = 255.0
        x = torch.tensor([0.0, 0.4, 1.6, 254.7, 300.0])
        expected = torch.tensor([0.0, 0.0, 2.0, 255.0, 255.0])
        assert torch.equal(tq(x), expected)

    def test_unsigned_rejects_negative_inputs(self):
        tq = _tq(num_bits=8, unsigned=True)
        tq.amax = 1.0
        with pytest.raises(TypeError, match="Negative values encountered"):
            tq(torch.tensor([0.5, -0.5]))

    def test_per_channel_amax(self):
        # axis=0 with amax [[127], [254]] -> row scales [1, 0.5]
        # row0: round([0.4, 1.6]*1)   = [0, 2]           -> /1   = [0, 2]
        # row1: round([300, -3.4]*.5) = [150->127, -2]   -> /0.5 = [254, -4]
        tq = _tq(num_bits=8, axis=0)
        tq.amax = torch.tensor([[127.0], [254.0]])
        x = torch.tensor([[0.4, 1.6], [300.0, -3.4]])
        expected = torch.tensor([[0.0, 2.0], [254.0, -4.0]])
        assert torch.equal(tq(x), expected)

    def test_dynamic_amax_computed_from_input(self):
        # No stored amax: amax = max|x| = 127 -> scale=1
        # round([0.4, 1.6, -127.0]) = [0, 2, -127]
        tq = TensorQuantizer()
        assert tq.amax is None
        x = torch.tensor([0.4, 1.6, -127.0])
        expected = torch.tensor([0.0, 2.0, -127.0])
        assert torch.equal(tq(x), expected)
        # Nothing is persisted: quantizer stays amax-free
        assert tq.amax is None

    def test_dynamic_type_quantizer_same_math(self):
        # type="dynamic" computes amax on the fly with identical math
        tq = _tq(type="dynamic")
        x = torch.tensor([0.4, 1.6, -127.0])
        expected = torch.tensor([0.0, 2.0, -127.0])
        assert torch.equal(tq(x), expected)

    def test_zero_amax_quantizes_to_zero(self):
        # amax=0 -> scale forced to 0 => all outputs quantize to 0 (then scale=1 for dequant)
        tq = TensorQuantizer()
        tq.amax = 0.0
        x = torch.tensor([1.0, -2.0, 0.5])
        assert torch.equal(tq(x), torch.zeros(3))

    def test_int4_per_tensor(self):
        # num_bits=4 -> max_bound = 2^3 - 1 = 7; amax=7 -> scale=1
        # x = [0.4, 1.6, 6.7, -8.9] -> round [0, 2, 7, -9] -> clamp(-8, 7) -> [0, 2, 7, -8]
        tq = _tq(num_bits=4)
        tq.amax = 7.0
        x = torch.tensor([0.4, 1.6, 6.7, -8.9])
        expected = torch.tensor([0.0, 2.0, 7.0, -8.0])
        assert torch.equal(tq(x), expected)


class TestExactFP8FakeQuant:
    """Hand-computed exact-value tests for FP8 E4M3 fake quantization.

    CPU path: clamp(x * 448/amax, +-448) -> cast to float8_e4m3fn (round-to-
    nearest-even) -> multiply by amax/448.
    """

    def test_amax_448_identity_scale(self):
        # amax=448 -> scale=1. E4M3 in [16, 32) has step 2:
        #   17 is a tie between 16 (mantissa 000) and 18 (mantissa 001) -> RNE -> 16
        # E4M3 in [2, 4) has step 0.25: 3.3 -> 3.25
        # 0.5, 1.5, 448 are exactly representable; 500 clamps to 448.
        tq = _tq(num_bits=(4, 3))
        tq.amax = 448.0
        x = torch.tensor([0.5, 1.5, 17.0, 3.3, 448.0, 500.0, -500.0])
        expected = torch.tensor([0.5, 1.5, 16.0, 3.25, 448.0, 448.0, -448.0])
        assert torch.equal(tq(x), expected)

    def test_amax_2_scaled(self):
        # amax=2 -> scale = 448/2 = 224.
        #   1.0  -> 224    (= 14*16, exactly representable) -> 224/224  = 1.0
        #   2.0  -> 448    (max normal)                     -> 448/224  = 2.0
        #   1.03 -> 230.72 -> nearest of {224, 240} is 224  -> 224/224  = 1.0
        #  -2.5  -> -560   -> clamp -448                    -> -448/224 = -2.0
        tq = _tq(num_bits=(4, 3))
        tq.amax = 2.0
        x = torch.tensor([1.0, 2.0, 1.03, -2.5])
        expected = torch.tensor([1.0, 2.0, 1.0, -2.0])
        assert torch.equal(tq(x), expected)

    def test_per_channel_fp8(self):
        # amax [[448], [896]] -> row scales [1, 0.5]
        # row0: 1.5 exactly representable -> 1.5
        # row1: 3.0*0.5 = 1.5 -> representable -> 1.5/0.5 = 3.0
        tq = _tq(num_bits=(4, 3), axis=0)
        tq.amax = torch.tensor([[448.0], [896.0]])
        x = torch.tensor([[1.5], [3.0]])
        expected = torch.tensor([[1.5], [3.0]])
        assert torch.equal(tq(x), expected)

    def test_is_fp8_property(self):
        assert _tq(num_bits=(4, 3)).is_fp8
        assert not _tq(num_bits=(4, 3), axis=0).is_fp8
        assert not _tq(num_bits=8).is_fp8


class TestExactBlockQuant:
    def test_static_block_quant_last_axis_exact(self):
        # block_sizes={-1: 2} on shape (1, 4): rows of the (2, 2) reshaped view
        # get their own amax.
        # block0 = [127.0, 0.4]  -> amax=127,  scale=1: round -> [127, 0]
        # block1 = [63.5, 0.9]   -> amax=63.5, scale=2: round([127, 1.8]) = [127, 2] -> [63.5, 1.0]
        tq = _tq(num_bits=8, block_sizes={-1: 2})
        x = torch.tensor([[127.0, 0.4, 63.5, 0.9]])
        expected = torch.tensor([[127.0, 0.0, 63.5, 1.0]])
        out = tq(x)
        assert out.shape == x.shape
        assert torch.equal(out, expected)

    def test_static_block_quant_requires_fixed_shape(self):
        tq = _tq(num_bits=8, block_sizes={-1: 2})
        tq(torch.randn(4, 4))
        with pytest.raises(ValueError, match="Block-quantization requires a fixed input shape"):
            tq(torch.randn(2, 8))

    def test_block_sizes_none_values_convert_to_axis(self):
        # {-1: None, -2: None} on a 2D tensor means amax shared over both dims
        # -> converted to per-tensor (axis=None), block_sizes cleared.
        tq = _tq(num_bits=8, block_sizes={-1: None, -2: None})
        x = torch.tensor([[0.4, 1.6], [-127.0, 3.4]])
        out = tq(x)
        assert tq.block_sizes is None
        assert tq.axis is None
        # per-tensor: amax=127, scale=1 -> [[0, 2], [-127, 3]]
        assert torch.equal(out, torch.tensor([[0.0, 2.0], [-127.0, 3.0]]))

    def test_is_static_block_quant_property(self):
        assert _tq(num_bits=8, block_sizes={-1: 16}).is_static_block_quant
        assert not _tq(
            num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
        ).is_static_block_quant
        assert not _tq(num_bits=8).is_static_block_quant


class TestDtypeHandling:
    def test_fp16_input_roundtrip(self):
        # amax=127 -> scale=1; fp16 inputs, math done in fp32 internally:
        # round([0.4, 1.6, -2.7]) = [0, 2, -3]; output dtype must stay fp16
        tq = TensorQuantizer()
        tq.amax = 127.0
        x = torch.tensor([0.4, 1.6, -2.7], dtype=torch.float16)
        out = tq(x)
        assert out.dtype == torch.float16
        assert torch.equal(out, torch.tensor([0.0, 2.0, -3.0], dtype=torch.float16))

    def test_bf16_input_roundtrip(self):
        tq = TensorQuantizer()
        tq.amax = 127.0
        x = torch.tensor([0.4, 1.6, -2.7], dtype=torch.bfloat16)
        out = tq(x)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, torch.tensor([0.0, 2.0, -3.0], dtype=torch.bfloat16))

    def test_fp8_preserves_input_dtype(self):
        tq = _tq(num_bits=(4, 3))
        tq.amax = 448.0
        x = torch.tensor([0.5, 1.5], dtype=torch.float16)
        out = tq(x)
        assert out.dtype == torch.float16
        assert torch.equal(out, torch.tensor([0.5, 1.5], dtype=torch.float16))

    def test_module_dtype_conversion_converts_amax_buffer(self):
        tq = TensorQuantizer()
        tq.amax = 127.0
        tq.half()
        assert tq._amax.dtype == torch.float16
        # Still quantizes fp32 inputs correctly (amax upcast internally)
        out = tq(torch.tensor([0.4, 1.6]))
        assert torch.equal(out, torch.tensor([0.0, 2.0]))

    def test_non_contiguous_input(self):
        tq = TensorQuantizer()
        tq.amax = 127.0
        base = torch.tensor([[0.4, 1.6], [-2.7, 126.7]])
        x = base.t()
        assert not x.is_contiguous()
        out = tq(x)
        assert torch.equal(out, tq(x.contiguous()))


class TestCalibrationContract:
    def test_calib_collect_and_load_amax_exact(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(), if_calib=True, if_quant=False)
        x1 = torch.tensor([1.0, -3.0])
        x2 = torch.tensor([5.0, -2.0])
        # if_quant=False: forward passes inputs through unchanged while collecting
        assert tq(x1) is x1
        assert tq(x2) is x2
        tq.load_calib_amax()
        # max calibrator: amax = max(|x1|, |x2|) = 5.0
        assert tq.amax.item() == pytest.approx(5.0)

    def test_load_calib_amax_strict_false_sets_nan(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(), if_calib=True, if_quant=False)
        with pytest.warns(UserWarning):
            tq.load_calib_amax(strict=False)
        assert torch.isnan(tq._amax).all()

    def test_enable_calib_without_calibrator_raises(self):
        tq = TensorQuantizer()
        tq._calibrator = None
        with pytest.raises(ValueError, match="Calibrator was not created"):
            tq.enable_calib()

    def test_forward_calib_without_calibrator_raises(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(), if_calib=True)
        tq._calibrator = None
        with pytest.raises(RuntimeError, match="Calibrator was not created"):
            tq(torch.tensor([1.0]))

    def test_collect_is_noop_when_calib_disabled(self):
        tq = TensorQuantizer()
        assert tq._if_calib is False
        tq.collect(torch.tensor([1.0, 2.0]))
        assert tq._calibrator.compute_amax() is None

    def test_enable_disable_calib_flags(self):
        tq = TensorQuantizer()
        tq.enable_calib()
        assert tq._if_calib is True
        tq.disable_calib()
        assert tq._if_calib is False


class TestStateSerialization:
    def test_state_dict_roundtrip_through_fresh_quantizer(self):
        ref = _tq(num_bits=8, axis=0)
        ref.amax = torch.tensor([[1.27], [2.54]])
        ref.pre_quant_scale = torch.tensor([2.0, 0.5])

        test = TensorQuantizer(QuantizerAttributeConfig())
        test.set_from_modelopt_state(ref.get_modelopt_state())
        test.load_state_dict(ref.state_dict())

        assert torch.equal(test.amax, ref.amax)
        assert torch.equal(test.pre_quant_scale, ref.pre_quant_scale)
        assert test.num_bits == ref.num_bits
        assert test.axis == ref.axis

        x = torch.tensor([[0.4, 1.6], [300.0, -3.4]])
        assert torch.equal(test(x), ref(x))

    def test_modelopt_state_properties_only(self):
        tq = TensorQuantizer(QuantizerAttributeConfig(num_bits=4), amax=10.0)
        state = tq.get_modelopt_state(properties_only=True)
        assert "_pytorch_state_metadata" not in state
        assert state["_num_bits"] == 4
        # No buffer values, only properties
        assert "_amax" not in state

    def test_modelopt_state_metadata_recreates_buffers(self):
        ref = TensorQuantizer(QuantizerAttributeConfig(num_bits=8), amax=1.0)
        state = ref.get_modelopt_state()
        meta = state["_pytorch_state_metadata"]
        assert meta["buffers"]["_amax"]["shape"] == torch.Size([])
        assert meta["buffers"]["_amax"]["dtype"] == torch.float32

        test = TensorQuantizer(QuantizerAttributeConfig())
        test.set_from_modelopt_state(state)
        # Buffer exists with the right shape/dtype (values come from load_state_dict)
        assert hasattr(test, "_amax")
        assert test._amax.shape == torch.Size([])
        assert test._amax.dtype == torch.float32

    def test_modelopt_state_restores_calibrator_attributes(self):
        ref = _tq(num_bits=4, axis=1, unsigned=True)
        test = TensorQuantizer(QuantizerAttributeConfig())
        test.set_from_modelopt_state(ref.get_modelopt_state())
        assert test._calibrator._num_bits == 4
        assert test._calibrator._axis == 1
        assert test._calibrator._unsigned is True

    def test_skip_properties_not_in_modelopt_state(self):
        tq = TensorQuantizer()
        state = tq.get_modelopt_state()
        for skipped in ("_calibrator", "_bias_calibrator", "_quantizer_cache"):
            assert skipped not in state

    def test_shared_quant_tied_attr_reassignment_blocked(self):
        tq = TensorQuantizer()
        tq._shared_quant_tied_attrs = {"_foo"}
        with pytest.raises(RuntimeError, match="tied shared quant state"):
            tq._foo = 1


class TestReprAndFormatProperties:
    def test_default_extra_repr_stable(self):
        tq = TensorQuantizer()
        assert (
            tq.extra_repr() == "8 bit fake per-tensor amax=dynamic calibrator=MaxCalibrator quant"
        )

    def test_extra_repr_with_scalar_amax(self):
        tq = TensorQuantizer()
        tq.amax = 1.27
        assert "amax=1.27e+00" in tq.extra_repr()

    def test_extra_repr_with_per_channel_amax(self):
        tq = _tq(num_bits=8, axis=0)
        tq.amax = torch.tensor([1.0, 2.0])
        assert "amax=[1.00e+00, 2.00e+00](2)" in tq.extra_repr()
        assert "axis=0" in tq.extra_repr()

    def test_extra_repr_flags(self):
        assert _tq(num_bits=8, unsigned=True).extra_repr().startswith("unsigned 8 bit")
        assert " narrow" in _tq(num_bits=8, narrow_range=True).extra_repr()
        assert "(4, 3) bit" in _tq(num_bits=(4, 3)).extra_repr()

    def test_disabled_extra_repr(self):
        tq = _tq(enable=False)
        assert tq.extra_repr() == "disabled"
        # NOTE: documents current behavior; arguably a bug because the disabled
        # branch of extra_repr builds a string including pre_quant_scale info and
        # then discards it by returning the literal "disabled".
        tq2 = TensorQuantizer()
        tq2.pre_quant_scale = 2.0
        tq2.disable()
        assert tq2.extra_repr() == "disabled"
        assert "pre_quant_scale" not in tq2.extra_repr()

    def test_maxbound_values(self):
        assert _tq(num_bits=8).maxbound == 127
        assert _tq(num_bits=8, unsigned=True).maxbound == 255
        assert _tq(num_bits=4).maxbound == 7
        assert _tq(num_bits=(4, 3)).maxbound == 448.0
        nvfp4 = _tq(num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)})
        assert nvfp4.maxbound == 6.0

    def test_mx_format_properties(self):
        mxfp4 = _tq(num_bits=(2, 1), block_sizes={-1: 32, "type": "dynamic", "scale_bits": (8, 0)})
        assert mxfp4.is_mx_format
        assert mxfp4.is_mxfp(4)
        assert not mxfp4.is_mxfp(8)
        # MX formats force pass-through backward
        assert mxfp4._pass_through_bwd is True
        # MX formats have no amax
        assert mxfp4.amax is None

        nvfp4 = _tq(num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)})
        assert not nvfp4.is_mx_format
        assert nvfp4.is_nvfp4_dynamic
        assert not nvfp4.is_nvfp4_static

    def test_rotate_disabled_by_default(self):
        tq = TensorQuantizer()
        assert tq.rotate_is_enabled is False
        assert tq.rotate_is_fp32 is False
        assert tq.rotate_block_size is None


class TestValidateAndExportAmax:
    def test_validate_attr(self):
        tq = TensorQuantizer()
        assert tq.validate_attr()  # missing amax counts as valid
        tq.amax = 1.0
        assert tq.validate_attr(attr_name="_amax")
        assert not tq.validate_attr(attr_value=torch.tensor(-1.0))
        assert not tq.validate_attr(attr_value=torch.tensor(float("nan")))
        assert not tq.validate_attr(attr_value=torch.tensor(float("inf")))
        with pytest.raises(ValueError, match="invalid values"):
            tq.validate_attr(attr_value=torch.tensor(-1.0), raise_error=True)
        with pytest.warns(UserWarning, match="invalid values"):
            tq.validate_attr(attr_value=torch.tensor(-1.0), warn_error=True)

    def test_export_amax_none_when_unset(self):
        assert TensorQuantizer().export_amax() is None

    def test_export_amax_per_tensor_unsqueezed(self):
        tq = TensorQuantizer()
        tq.amax = 1.5
        amax = tq.export_amax()
        assert amax.shape == (1,)
        assert amax.item() == pytest.approx(1.5)

    def test_export_amax_replaces_zero_and_nan_with_maxbound(self):
        tq = _tq(num_bits=8, axis=0)
        tq.amax = torch.tensor([0.0, float("nan"), 2.0])
        amax = tq.export_amax()
        assert torch.equal(amax, torch.tensor([127.0, 127.0, 2.0]))

    def test_export_amax_per_channel_squeezed(self):
        tq = _tq(num_bits=8, axis=0)
        tq.amax = torch.tensor([[1.0], [2.0]])
        assert tq.export_amax().shape == (2,)


class TestBackward:
    def test_pass_through_bwd_default_ste(self):
        # Default pass_through_bwd=True: plain STE, gradient of 1 everywhere,
        # including for clipped inputs.
        tq = TensorQuantizer()
        tq.amax = 1.0
        x = torch.tensor([0.5, 2.0], requires_grad=True)
        tq(x).sum().backward()
        assert torch.equal(x.grad, torch.tensor([1.0, 1.0]))

    def test_clipped_ste_zeroes_outlier_gradients(self):
        # pass_through_bwd=False: STE with zeroed gradients where |x| > amax.
        # x = [0.5, 2.0], amax = 1.0 -> grads [1, 0]
        tq = _tq(num_bits=8, pass_through_bwd=False)
        tq.amax = 1.0
        x = torch.tensor([0.5, 2.0], requires_grad=True)
        tq(x).sum().backward()
        assert torch.equal(x.grad, torch.tensor([1.0, 0.0]))


class TestHardDisabledTensorQuantizer:
    def test_construction_is_disabled_and_forward_passthrough(self):
        tq = HardDisabledTensorQuantizer()
        assert not tq.is_enabled
        x = torch.tensor([1.0, -2.0])
        assert tq(x) is x

    def test_enable_methods_raise(self):
        tq = HardDisabledTensorQuantizer()
        with pytest.raises(RuntimeError, match="hard-disabled"):
            tq.enable()
        with pytest.raises(RuntimeError, match="hard-disabled"):
            tq.enable_quant()
        with pytest.raises(RuntimeError, match="hard-disabled"):
            tq.enable_calib()

    def test_wildcard_config_absorbed_but_stays_disabled(self):
        tq = HardDisabledTensorQuantizer()
        tq.set_from_attribute_config({"num_bits": 4, "enable": True})
        assert tq.num_bits == 4  # config is absorbed for introspection
        assert not tq.is_enabled  # ... but the quantizer stays disabled


class TestQuantBackendRegistry:
    def test_register_validation(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_quant_backend("", lambda x, tq: x)
        with pytest.raises(TypeError, match="must be callable"):
            register_quant_backend("bad_entrypoint", "not callable")
        with pytest.raises(ValueError, match="non-empty string"):
            unregister_quant_backend("")

    def test_register_forward_unregister(self):
        name = "_tq_unit_test_backend"
        received = {}

        def entrypoint(inputs, tq):
            received["quantizer"] = tq
            offset = (tq.backend_extra_args or {}).get("offset", 0.0)
            return inputs + offset

        register_quant_backend(name, entrypoint)
        try:
            assert is_registered_quant_backend(name)
            tq = _tq(num_bits=8, backend=name, backend_extra_args={"offset": 2.5})
            x = torch.tensor([1.0, -1.0])
            out = tq(x)
            # Backend fully replaces the fake-quant math
            assert torch.equal(out, torch.tensor([3.5, 1.5]))
            # Entrypoint receives the quantizer itself
            assert received["quantizer"] is tq
        finally:
            unregister_quant_backend(name)
        assert not is_registered_quant_backend(name)

    def test_reregister_warns(self):
        name = "_tq_unit_test_backend_dup"
        register_quant_backend(name, lambda x, tq: x)
        try:
            with pytest.warns(UserWarning, match="Overwriting existing backend"):
                register_quant_backend(name, lambda x, tq: x)
        finally:
            unregister_quant_backend(name)

    def test_unregistered_backend_forward_raises(self):
        tq = _tq(num_bits=8, backend="_tq_never_registered")
        with pytest.raises(KeyError, match="not registered"):
            tq(torch.tensor([1.0]))


class TestSequentialQuantizerContract:
    def test_rejects_non_tensor_quantizer(self):
        with pytest.raises(AssertionError, match="must be a TensorQuantizer"):
            SequentialQuantizer(TensorQuantizer(), nn.Identity())

    def test_delegated_amax_set_fans_out(self):
        sq = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
        sq.amax = 2.0
        assert sq[0].amax.item() == pytest.approx(2.0)
        assert sq[1].amax.item() == pytest.approx(2.0)
        # Getter returns the first quantizer's value
        sq[0].amax = 5.0
        assert sq.amax.item() == pytest.approx(5.0)


class TestUseConstantAmax:
    def test_constant_amax_overrides_stored_amax(self):
        # The 448 constant itself is covered by the shared tester
        # (tensor_quantizer_common.py::test_use_constant_amax); the delta here
        # is that an explicitly STORED amax is ignored by _get_amax.
        tq = _tq(num_bits=8, use_constant_amax=True)
        tq.amax = 1.0  # should be ignored by _get_amax
        got = tq._get_amax(torch.tensor([1.0]))
        assert got.item() == torch.finfo(torch.float8_e4m3fn).max

    def test_constant_amax_forward_exact(self):
        # int8 with constant amax 448: scale = 127/448
        # x = 448 -> 127 -> back to 448 exactly
        tq = _tq(num_bits=8, use_constant_amax=True)
        out = tq(torch.tensor([448.0]))
        assert out.item() == pytest.approx(448.0)


class TestWarningHygiene:
    def test_plain_forward_emits_no_warnings(self):
        tq = TensorQuantizer()
        tq.amax = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tq(torch.tensor([0.25, -0.75]))
