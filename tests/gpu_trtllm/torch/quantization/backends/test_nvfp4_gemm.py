# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.torch.quantization.models import OneLayerLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.qtensor import NVFP4QTensor


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("shape", [(128, 64), (3, 16)])
def test_nvfp4_quantization(shape):
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
        cutlass_fp4_scale_to_modelopt_fp4_scale,
    )

    block_sizes = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    weight = torch.randn(shape).to(torch.float16).cuda()

    weight_fp4_trtllm, wsf_trtllm, wsf2_trtllm = NVFP4QTensor.quantize(
        weight, block_sizes[-1], try_tensorrt=True
    )
    weight_fp4, wsf, wsf2 = NVFP4QTensor.quantize(weight, block_sizes[-1], try_tensorrt=False)
    mask = weight_fp4_trtllm._quantized_data != weight_fp4._quantized_data
    diff = (
        weight_fp4_trtllm._quantized_data.to(torch.int16)
        - weight_fp4._quantized_data.to(torch.int16)
    ).abs()
    # two quantized_data are combined in an unin8.
    # 1, 16, and 17 mean the number in uint4 is only different by 1.
    assert (mask.sum((0, 1)) <= 30) and (set(diff[mask].unique().tolist()) <= {1, 16, 17}), (
        f"quantized data mismatch:\n"
        f"{weight_fp4_trtllm._quantized_data[mask]} != {weight_fp4._quantized_data[mask]}\n"
        f"{diff[mask]}"
    )
    wsf_converted = cutlass_fp4_scale_to_modelopt_fp4_scale(wsf_trtllm, weight.shape[-2:])
    mask = wsf_converted != wsf
    assert torch.equal(wsf_converted, wsf), (
        f"wsf mismatch: {wsf_converted.float()[mask]} != {wsf.float()[mask]}"
    )
    assert torch.equal(wsf2_trtllm, wsf2), f"wsf2 mismatch: {wsf2_trtllm} != {wsf2}"

    weight_dequantized_trtllm = weight_fp4_trtllm.dequantize(
        scale=wsf_trtllm, double_scale=wsf2_trtllm, block_sizes=block_sizes
    )
    weight_dequantized = weight_fp4.dequantize(
        scale=wsf, double_scale=wsf2, block_sizes=block_sizes
    )
    diff = (weight_dequantized_trtllm - weight_dequantized).abs()
    mask = diff > 0.1
    assert (mask.sum((0, 1)) <= 10) and (diff.amax() < 0.5), (
        f"dequantized data mismatch:\n"
        f"{weight_dequantized_trtllm[mask]} != {weight_dequantized[mask]}\n"
        f"{(weight_dequantized_trtllm[mask] - weight_dequantized[mask]).abs().amax()}"
    )
    diff = (weight_dequantized_trtllm - weight).abs()
    assert torch.allclose(weight_dequantized_trtllm, weight, atol=0.5, rtol=0.1), (
        f"dequantized data mismatch: {diff.amax()}\n"
        f"{weight_dequantized_trtllm[diff > 0.4]} != {weight[diff > 0.4]}\n"
    )


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
@pytest.mark.parametrize("model", [OneLayerLinear(in_features=64, out_features=32)])
@pytest.mark.parametrize("config", [mtq.NVFP4_DEFAULT_CFG])
def test_forward(model, config):
    model = model.to(torch.float16).cuda()
    calib_data = [model.get_input(2).to(torch.float16).cuda() for _ in range(8)]

    def forward_loop(model, run_backward=False):
        for batch in calib_data:
            output = model(batch)
            if run_backward:
                output.sum().backward()

    mtq.quantize(model, config, forward_loop)

    input_tensor = calib_data[0].clone()
    output_fake = model(input_tensor)

    mtq.compress(model)
    output_compressed_real = model(input_tensor)
    diff = (output_fake - output_compressed_real).abs()
    # The difference is due to the scale factor.
    # The reciprocal of reciprocal is not the same as the original scale factor.
    assert torch.allclose(output_fake, output_compressed_real, atol=0.2), (
        "Difference between real and compressed real quantization: "
        f"{diff.amax()}, {diff.mean()}, {diff.std()}"
    )
