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

"""Tests of QuantConv module."""

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules import quant_conv
from modelopt.torch.quantization.nn.modules.quant_conv import (
    _is_nvfp4_quantizer,
    _nvfp4_quantize_weight_along_k,
)
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

NUM_IN_CHANNELS = 3
NUM_OUT_CHANNELS = 5

_NVFP4_CFG = QuantizerAttributeConfig(
    num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
)


class TestQuantConvND:
    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_no_quant(self, conv_cls, f_conv, input_shape):
        kernel_size = 8

        quant_conv_object = conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=False)
        quant_conv_object.input_quantizer.disable()
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_copy

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_weight_fake_quant_per_tensor(self, conv_cls, f_conv, input_shape):
        kernel_size = 8

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(),
        )
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, weight_copy.abs().amax())

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_weight_fake_quant_per_channel(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        quant_conv_object.input_quantizer.disable()
        test_input = torch.randn(input_shape)

        weight_copy = quant_conv_object.weight.clone()
        amax = weight_copy.abs().amax(dim=(1, 2), keepdim=True)
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, amax)

        out1 = f_conv(test_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_input(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=False)
        quant_conv_object.weight_quantizer.disable()
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = f_conv(quant_input, quant_conv_object.weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_tensor(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(weight_copy, weight_copy.abs().amax())

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy, weight_copy.abs().amax(dim=list(range(1, len(input_shape))), keepdim=True)
        )

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel_other_prec(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_desc_input = QuantizerAttributeConfig(num_bits=4)
        quant_desc_weight = QuantizerAttributeConfig(num_bits=3, axis=(0))

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=False,
            quant_desc_input=quant_desc_input,
            quant_desc_weight=quant_desc_weight,
        )
        test_input = torch.randn(input_shape)

        test_input_quantizer = TensorQuantizer(quant_desc_input)
        weight_quantizer = TensorQuantizer(quant_desc_weight)

        quant_input = test_input_quantizer(test_input)

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = weight_quantizer(weight_copy)

        out1 = f_conv(quant_input, quant_weight)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "f_conv", "input_shape"),
        [
            (quant_conv.QuantConv1d, F.conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, F.conv_transpose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, F.conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, F.conv_transpose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, F.conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, F.conv_transpose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel_bias(self, conv_cls, f_conv, input_shape):
        kernel_size = 3

        quant_conv_object = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_weight=QuantizerAttributeConfig(axis=(0)),
        )
        test_input = torch.randn(input_shape)

        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        weight_copy = quant_conv_object.weight.clone()
        quant_weight = tensor_quant.fake_tensor_quant(
            weight_copy, weight_copy.abs().amax(dim=list(range(1, len(input_shape))), keepdim=True)
        )

        out1 = f_conv(quant_input, quant_weight, bias=quant_conv_object.bias)
        out2 = quant_conv_object(test_input)
        assert torch.allclose(out1, out2, rtol=1e-0, atol=1e-0)

    @pytest.mark.parametrize(
        ("conv_cls", "nn_conv_cls", "input_shape"),
        [
            (quant_conv.QuantConv1d, nn.Conv1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConvTranspose1d, nn.ConvTranspose1d, (2, NUM_IN_CHANNELS, 8)),
            (quant_conv.QuantConv2d, nn.Conv2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConvTranspose2d, nn.ConvTranspose2d, (2, NUM_IN_CHANNELS, 8, 8)),
            (quant_conv.QuantConv3d, nn.Conv3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
            (quant_conv.QuantConvTranspose3d, nn.ConvTranspose3d, (2, NUM_IN_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_against_unquantized(self, conv_cls, nn_conv_cls, input_shape):
        kernel_size = 3
        test_input = torch.randn(input_shape)

        quant_conv = conv_cls(
            NUM_IN_CHANNELS,
            NUM_OUT_CHANNELS,
            kernel_size,
            bias=True,
            quant_desc_input=QuantizerAttributeConfig(num_bits=16),
            quant_desc_weight=QuantizerAttributeConfig(num_bits=16, axis=(0)),
        )

        conv = nn_conv_cls(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size, bias=True)
        conv.load_state_dict(quant_conv.state_dict())

        quant_conv.input_quantizer.disable()
        quant_conv.weight_quantizer.disable()

        quant_output = quant_conv(test_input)
        output = conv(test_input)

        assert torch.allclose(quant_output, output)


class TestQuantConv3dNVFP4:
    """Tests for the NVFP4 implicit GEMM dispatch path in ``_QuantConv3d``.

    The CUDA kernel itself is GPU-only (covered in ``tests/gpu``); here we only
    exercise the CPU-side predicate and fallback branches that currently lack
    coverage.
    """

    @staticmethod
    def _make_nvfp4_conv3d(groups: int = 1, bias: bool = False) -> quant_conv.QuantConv3d:
        return quant_conv.QuantConv3d(
            NUM_IN_CHANNELS * groups,
            NUM_OUT_CHANNELS * groups,
            kernel_size=3,
            groups=groups,
            bias=bias,
            quant_desc_input=_NVFP4_CFG,
            quant_desc_weight=_NVFP4_CFG,
        )

    def test_is_nvfp4_quantizer_true(self):
        q = TensorQuantizer(_NVFP4_CFG)
        assert _is_nvfp4_quantizer(q)

    def test_is_nvfp4_quantizer_false_for_int8(self):
        q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        assert not _is_nvfp4_quantizer(q)

    def test_is_nvfp4_quantizer_false_for_fp8(self):
        # FP8 E4M3: num_bits == (4, 3), block_sizes is None
        q = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
        assert not _is_nvfp4_quantizer(q)

    def test_is_nvfp4_quantizer_false_for_static_block(self):
        static_cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        )
        q = TensorQuantizer(static_cfg)
        assert not _is_nvfp4_quantizer(q)

    def test_should_use_implicit_gemm_true(self):
        m = self._make_nvfp4_conv3d()
        assert m._should_use_implicit_gemm()

    def test_should_use_implicit_gemm_false_groups_gt_1(self):
        m = self._make_nvfp4_conv3d(groups=NUM_IN_CHANNELS)
        # Groups > 1 disqualifies the fused kernel even with NVFP4 quantizers.
        assert not m._should_use_implicit_gemm()

    def test_should_use_implicit_gemm_false_non_nvfp4(self):
        # INT8 per-tensor config — default Conv3D quantizers, no NVFP4.
        m = quant_conv.QuantConv3d(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size=3, bias=False)
        assert not m._should_use_implicit_gemm()

    def test_nvfp4_quantize_weight_along_k_reshape(self):
        """Verify weight is flattened/restored along the K (input) dimension.

        Uses a stub quantizer so this test stays CPU-only (the real NVFP4 dynamic
        quantizer requires a CUDA tensor).
        """

        def identity_quantizer(x):
            # K-dim must be the last axis when passed to the quantizer so NVFP4's
            # block-wise scaling aligns with the GEMM reduction axis.
            assert x.dim() == 2 and x.shape[0] == NUM_OUT_CHANNELS
            return x

        w = torch.randn(NUM_OUT_CHANNELS, NUM_IN_CHANNELS, 3, 3, 3)
        qw = _nvfp4_quantize_weight_along_k(w, identity_quantizer)
        assert qw.shape == w.shape
        assert torch.equal(qw, w)

    def test_forward_non_nvfp4_matches_unquantized(self):
        # Disabled quantizers: forward must match plain conv3d exactly.
        m = quant_conv.QuantConv3d(NUM_IN_CHANNELS, NUM_OUT_CHANNELS, kernel_size=3, bias=False)
        m.input_quantizer.disable()
        m.weight_quantizer.disable()
        x = torch.randn(1, NUM_IN_CHANNELS, 4, 4, 4)
        out = m(x)
        ref = F.conv3d(x, m.weight)
        assert torch.allclose(out, ref)

    def test_forward_nvfp4_training_warns_and_falls_back(self):
        """Training mode must fall back to the default (cuDNN) path with a warning.

        The implicit-GEMM kernel is inference-only; this exercises the CPU-visible
        training-fallback branch. We disable the quantizers so the default-path
        ``super().forward()`` does not try to run NVFP4 dynamic quantization
        (which requires CUDA).
        """
        m = self._make_nvfp4_conv3d()
        # NVFP4 predicate reads configuration (num_bits/block_sizes), not enable
        # state, so disabling the quantizers still routes through the NVFP4 branch.
        m.input_quantizer.disable()
        m.weight_quantizer.disable()
        assert m._should_use_implicit_gemm()
        m.train()
        x = torch.randn(1, NUM_IN_CHANNELS, 4, 4, 4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = m(x)
        assert out.shape == (1, NUM_OUT_CHANNELS, 2, 2, 2)
        assert any("inference-only" in str(w.message) for w in caught), (
            f"Expected an 'inference-only' warning, got: {[str(w.message) for w in caught]}"
        )

    def test_forward_nvfp4_calib_only_uses_default_path(self):
        """Calibration-only mode must NOT try to invoke the CUDA kernel.

        When the input quantizer is in calibration mode without quant enabled,
        the forward must fall back to the default path. This exercises the
        calib-only early-return branch; we assert that the output matches the
        default-path output exactly (i.e. the implicit-GEMM path wasn't used).
        """
        m = self._make_nvfp4_conv3d()
        m.eval()
        # Match the state toggled by TensorQuantizer.disable_quant()/enable_calib().
        m.input_quantizer.disable_quant()
        m.input_quantizer.enable_calib()
        m.weight_quantizer.disable_quant()
        m.weight_quantizer.enable_calib()
        assert m.input_quantizer._if_calib and not m.input_quantizer._if_quant

        x = torch.randn(1, NUM_IN_CHANNELS, 4, 4, 4)
        out = m(x)
        # Default path with quant disabled should equal plain conv3d.
        ref = F.conv3d(x, m.weight.detach())
        assert torch.allclose(out, ref, atol=1e-5)
