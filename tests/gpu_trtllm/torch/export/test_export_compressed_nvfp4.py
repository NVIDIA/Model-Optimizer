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

import copy

import pytest
import torch
from _test_utils.torch.export.utils import ToyModel

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.backends.utils import fp4_compatible
from modelopt.torch.quantization.utils import quantizer_attr_names

BLOCK_SIZE = 16


@pytest.mark.skipif(not fp4_compatible(), reason="FP4 is not supported on this GPU")
def test_export_compressed_nvfp4_weight_trtllm_scale():
    """Export a compressed NVFP4 weight whose block scale is cutlass-swizzled.

    With TensorRT-LLM importable on an FP4-capable device, ``TensorQuantizer._real_quantize``
    routes through ``torch.ops.trtllm.fp4_quantize`` and stores a 1-D uint8 swizzled scale in
    ``weight_quantizer._scale`` instead of the modelopt 2-D E4M3 layout. The export must
    un-swizzle it; using it as-is would write a scale of raw byte values.
    """
    in_features = 256
    calib = lambda x: x(torch.randn(1, 4, in_features).cuda().half())  # noqa: E731

    model = ToyModel(dims=[in_features] * 4).cuda().half()
    reference = mtq.quantize(copy.deepcopy(model), mtq.NVFP4_DEFAULT_CFG, calib)
    compressed = mtq.quantize(copy.deepcopy(model), mtq.NVFP4_DEFAULT_CFG, calib)
    mtq.compress(compressed)

    quantizer_attrs = quantizer_attr_names("weight")
    ref_module, compressed_module = reference.linears[2], compressed.linears[2]

    # Precondition: this environment really does produce the swizzled layout, otherwise the
    # test would silently degrade into the dense-scale case already covered in tests/gpu.
    stored_scale = getattr(compressed_module, quantizer_attrs.weight_quantizer)._scale
    assert stored_scale.dtype == torch.uint8 and stored_scale.ndim == 1, (
        f"expected a cutlass-swizzled scale, got {stored_scale.dtype} with ndim {stored_scale.ndim}"
    )

    _export_quantized_weight(ref_module, torch.float16, "weight")
    _export_quantized_weight(compressed_module, torch.float16, "weight")

    ref_scale = getattr(ref_module, quantizer_attrs.weight_scale)
    compressed_scale = getattr(compressed_module, quantizer_attrs.weight_scale)

    # Un-swizzled back to the checkpoint layout, not left as 1-D bytes.
    assert compressed_scale.shape == ref_scale.shape
    assert compressed_scale.shape[-1] == in_features // BLOCK_SIZE
    assert compressed_scale.dtype == ref_scale.dtype

    # weight_scale * weight_scale_2 is what dequantization consumes.
    ref_2 = getattr(ref_module, quantizer_attrs.weight_scale_2)
    compressed_2 = getattr(compressed_module, quantizer_attrs.weight_scale_2)
    assert torch.allclose(
        compressed_scale.float() * compressed_2.float(),
        ref_scale.float() * ref_2.float(),
        rtol=0.05,
    )
