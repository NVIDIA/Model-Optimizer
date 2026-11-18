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

"""Native IQ payload placement, mesh adoption and real four-GPU projection checks."""

import pytest
import torch
from torch.nn import functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("name", sorted(IQ_FORMAT_REGISTRY))
@pytest.mark.parametrize("dim", [0, 1])
def test_native_iq_partition_preserves_pinned_projection(name, dim):
    from modelopt.torch.quantization.repair_partition import (
        NativeWeightPartition,
        native_weight_partition,
    )

    torch.manual_seed(59)
    fmt = IQ_FORMAT_REGISTRY[name]
    weight = torch.randn(4, 512)
    importance = 1 + torch.rand(512) if fmt.weighted else None
    packed, shape = fmt.quantize(
        weight, **({"importance": importance} if importance is not None else {})
    )
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=name, backend="ggml", block_sizes={-1: fmt.block_size})
    )
    search = fmt.payload_scales(packed.reshape(-1, fmt.block_bytes)) if fmt.payload_scales else None
    pin_packed_weight(quantizer, name, packed, importance, source=weight, search_scales=search)
    expected = fmt.dequantize(packed, shape, dtype=torch.float32)
    view = NativeWeightPartition((4, 512), dim, 1, 2, split_factor=2 if dim == 0 else 1)
    local = view.select(weight)
    fields, _, _ = native_weight_partition(quantizer, view, source=local)
    for key, value in fields.items():
        quantizer._set_buffer(key, value) if isinstance(value, torch.Tensor) else setattr(
            quantizer, key, value
        )
    x = torch.randn(3, local.shape[-1], requires_grad=True)
    actual = F.linear(x, quantizer(local))
    reference = F.linear(x, view.select(expected))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.square().sum().backward()
    actual_grad = x.grad.clone()
    x.grad = None
    reference.square().sum().backward()
    torch.testing.assert_close(actual_grad, x.grad, rtol=0, atol=0)
    before = {n: v.clone() for n, v in quantizer.state_dict().items()}
    invalid = NativeWeightPartition((4, 512), 1, 0, 2 * (512 // fmt.block_size))
    with pytest.raises(ValueError):
        native_weight_partition(quantizer, invalid)
    assert all(torch.equal(v, before[n]) for n, v in quantizer.state_dict().items())


def test_packed_frozen_projection_staging_and_restore(tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
    from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

    torch.manual_seed(60)
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model[0]
    native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
    layer.weight_quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=False)
    )
    layer.weight_quantizer._set_buffer("_scale", scale)
    layer.weight = QTensorWrapper(native)
    decoded = native.dequantize(scale=scale)
    prepare_packed_weights_for_tp(model)
    x = torch.randn(3, 256, requires_grad=True)
    expected = F.linear(x, decoded)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)
    assert not layer.weight.requires_grad
    path = tmp_path / "staged.pt"
    mto.save(model, path)
    restored = mto.restore(torch.nn.Sequential(torch.nn.Linear(256, 16, bias=False)), path)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert torch.equal(
        restored[0].weight_quantizer._native_weight_codes, native._quantized_data.view(torch.uint8)
    )
