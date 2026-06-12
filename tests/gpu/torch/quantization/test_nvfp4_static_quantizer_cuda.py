# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for NVFP4StaticQuantizer and NVFP4MSECalibrator."""

import pytest
import torch

from modelopt.torch.quantization.calib import NVFP4MSECalibrator
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer
from modelopt.torch.quantization.tensor_quant import (
    scaled_e4m3_impl,
    static_blockwise_fp4_fake_quant,
)


@pytest.mark.parametrize("device", ["cuda"])
class TestNVFP4StaticQuantizer:
    def test_from_tensor_quantizer(self, device):
        """Test creating NVFP4StaticQuantizer from TensorQuantizer."""
        cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        )
        tq = TensorQuantizer(quant_attribute_cfg=cfg).to(device)
        tq.amax = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

        nvfp4_quantizer = NVFP4StaticQuantizer.from_tensor_quantizer(tq)

        assert nvfp4_quantizer.global_amax is None
        assert nvfp4_quantizer._num_bits == (2, 1)
        assert torch.allclose(nvfp4_quantizer._amax, tq._amax)

    def test_global_amax_property(self, device):
        """Test global_amax property getter/setter."""
        cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        )
        quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=cfg).to(device)

        assert quantizer.global_amax is None

        quantizer.global_amax = torch.tensor(5.0, device=device)
        assert quantizer.global_amax is not None
        assert torch.isclose(quantizer.global_amax, torch.tensor(5.0, device=device))

        quantizer.global_amax = 10.0
        assert torch.isclose(quantizer.global_amax, torch.tensor(10.0, device=device))

        quantizer.global_amax = None
        assert quantizer.global_amax is None

    def test_fake_quantize_with_both_amaxs(self, device):
        """Test _fake_quantize uses both _amax and _global_amax."""
        num_blocks = 4
        block_size = 16

        cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: block_size, "type": "static", "scale_bits": (4, 3)},
        )
        quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=cfg).to(device)

        x = torch.randn(num_blocks, block_size, device=device)
        per_block_amax = x.abs().amax(dim=-1)
        global_amax = per_block_amax.max()

        quantizer.amax = per_block_amax
        quantizer.global_amax = global_amax

        output = quantizer._fake_quantize(x)

        expected = static_blockwise_fp4_fake_quant(
            x,
            per_block_amax,
            global_amax,
        )

        assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", ["cuda"])
class TestNVFP4MSECalibrator:
    def test_basic_initialization(self, device):
        """Test NVFP4MSECalibrator initialization."""
        num_blocks = 4
        amax = torch.ones(num_blocks, device=device)
        global_amax = torch.tensor(10.0, device=device)
        cal = NVFP4MSECalibrator(amax=amax, global_amax=global_amax)

        assert cal._losses_sum is None
        assert cal._amax is None

    def test_fp8_candidates_generation(self, device):
        """Test that 126 valid FP8 candidates are generated."""
        num_blocks = 4
        amax = torch.ones(num_blocks, device=device)
        global_amax = torch.tensor(10.0, device=device)
        cal = NVFP4MSECalibrator(amax=amax, global_amax=global_amax)

        candidates = cal._generate_candidates(device)

        assert candidates.shape[0] == 126
        assert torch.all(torch.isfinite(candidates))
        assert torch.all(candidates > 0)

    def test_collect_and_compute_amax(self, device):
        """Test collect and compute_amax workflow."""
        num_blocks = 8
        block_size = 16
        per_block_amax = torch.ones(num_blocks, device=device)
        global_amax = torch.tensor(10.0, device=device)

        def quant_func(x, amax):
            return static_blockwise_fp4_fake_quant(x, amax, global_amax)

        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            global_amax=global_amax,
            quant_func=quant_func,
        )

        x = torch.randn(num_blocks, block_size, device=device)
        cal.collect(x)

        assert cal._losses_sum is not None
        assert len(cal._losses_sum) == 126

        amax = cal.compute_amax()

        assert amax is not None
        assert amax.shape[0] == num_blocks
        assert torch.all(torch.isfinite(amax))
        assert torch.all(amax > 0)

    def test_multiple_collections(self, device):
        """Test that multiple collections accumulate correctly."""
        num_blocks = 4
        block_size = 16
        per_block_amax = torch.ones(num_blocks, device=device)
        global_amax = torch.tensor(5.0, device=device)

        def quant_func(x, amax):
            return static_blockwise_fp4_fake_quant(x, amax, global_amax)

        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            global_amax=global_amax,
            quant_func=quant_func,
        )

        x1 = torch.randn(num_blocks, block_size, device=device)
        x2 = torch.randn(num_blocks, block_size, device=device)

        cal.collect(x1)
        losses_after_first = [loss.clone() for loss in cal._losses_sum]

        cal.collect(x2)
        losses_after_second = cal._losses_sum

        for loss1, loss2 in zip(losses_after_first, losses_after_second):
            assert torch.all(loss2 >= loss1 - 1e-6)

    def test_per_block_independent_optimization(self, device):
        """Test that each block is optimized independently.

        Uses constant values per block to ensure deterministic behavior.
        """
        num_blocks = 4
        block_size = 16

        # Create blocks with constant values (all elements in a block are the same)
        # This ensures deterministic behavior for the test
        x = torch.zeros(num_blocks, block_size, device=device)
        x[0, :] = 0.5
        x[1, :] = 2.0
        x[2, :] = 5.0
        x[3, :] = 10.0

        per_block_amax = x.abs().amax(dim=-1)
        global_amax = per_block_amax.max()

        def quant_func(x, amax):
            return static_blockwise_fp4_fake_quant(x, amax, global_amax)

        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            axis=0,  # reduce_axis = -1
            global_amax=global_amax,
            quant_func=quant_func,
        )

        cal.collect(x)
        amax = cal.compute_amax()

        # With constant values per block, the optimal amax should scale with the block values
        assert amax[1] > amax[0]
        assert amax[2] > amax[1]
        assert amax[3] > amax[2]

    def test_fp8_sweep_generates_quantized_scales(self, device):
        """Test that the fp8 sweep produces scales that are already FP8-quantized."""
        num_blocks = 8
        block_size = 16

        x = torch.randn(num_blocks, block_size, device=device)
        per_block_amax = x.abs().amax(dim=-1)
        global_amax = per_block_amax.max()

        def quant_func(x, amax):
            return static_blockwise_fp4_fake_quant(x, amax, global_amax)

        cal = NVFP4MSECalibrator(
            amax=per_block_amax,
            global_amax=global_amax,
            quant_func=quant_func,
        )

        cal.collect(x)
        amax = cal.compute_amax()

        # The calibrator sweeps over FP8 candidates, so the resulting scales
        # should already be representable in FP8 (i.e., quantize-dequantize is a no-op).
        scale = amax.float() / 6.0
        scale_fp8_quant_amax = global_amax.float() / 6.0
        scale_qdq = scaled_e4m3_impl(scale, scale_fp8_quant_amax)
        assert torch.allclose(scale_qdq, scale)
