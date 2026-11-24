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

"""Native payload partitions and frozen-weight staging with CPU reference codecs."""

import copy

import pytest
import torch
from torch.nn import functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.repair_partition import (
    NativeWeightPartition,
    native_weight_partition,
)


@pytest.mark.parametrize("name", [*sorted(IQ_FORMAT_REGISTRY), "q8_0"])
@pytest.mark.parametrize("dim", [0, 1])
def test_native_partition_preserves_pinned_projection(name, dim):
    from modelopt.torch.quantization.repair_partition import (
        NativeWeightPartition,
        native_weight_partition,
    )

    torch.manual_seed(59)
    fmt = GGML_FORMAT_REGISTRY[name]
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
    for bad_dim, bad_rank, bad_size, bad_split in (
        (0, -1, 2, 1),
        (0, 2, 2, 1),
        (-1, 0, 2, 1),
        (2, 0, 2, 1),
        (0, 0, 0, 1),
        (0, 0, 2, 0),
    ):
        with pytest.raises(ValueError):
            NativeWeightPartition((4, 512), bad_dim, bad_rank, bad_size, bad_split)
    assert all(torch.equal(v, before[n]) for n, v in quantizer.state_dict().items())


@pytest.mark.parametrize(
    "format_name", ["fp8", "nvfp4", "fp8_block", "fp8_block_padded", "nvfp4_swizzled"]
)
@pytest.mark.parametrize("inference_staging", [False, True])
def test_packed_frozen_projection_staging_and_restore(format_name, inference_staging, tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper
    from modelopt.torch.quantization.repair_partition import (
        NativeWeightPartition,
        native_weight_partition,
        prepare_packed_weights_for_tp,
    )
    from modelopt.torch.quantization.utils.native_weight import (
        get_pinned_native_weight,
        pin_native_weight,
    )

    torch.manual_seed(60)
    columns = 258 if format_name == "fp8_block_padded" else 256
    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(columns, 16, bias=False)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    layer = model[0]
    if format_name == "fp8":
        native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
        double_scale = None
        attributes = {"num_bits": (4, 3), "axis": 0}
    elif format_name.startswith("fp8_block"):
        native, scale = FP8QTensor.quantize(layer.weight.detach(), block_sizes={-1: 128})
        double_scale = None
        attributes = {"num_bits": (4, 3), "block_sizes": {-1: 128}}
    else:
        native, scale, double_scale = NVFP4QTensor.quantize(
            layer.weight.detach(), 16, try_tensorrt=False
        )
        attributes = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    layer.weight_quantizer.set_from_attribute_config(
        QuantizerAttributeConfig(**attributes, fake_quant=False)
    )
    if format_name == "nvfp4_swizzled":
        padded = torch.zeros(128, 16, dtype=torch.uint8)
        padded[:16] = scale.view(torch.uint8)
        scale = padded.reshape(1, 4, 32, 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(128, 16)
    layer.weight_quantizer._set_buffer("_scale", scale)
    if double_scale is not None:
        layer.weight_quantizer._set_buffer("_double_scale", double_scale)
    layer.weight = QTensorWrapper(native)
    decoded = native.dequantize(
        scale=scale.flatten() if format_name == "nvfp4_swizzled" else scale,
        double_scale=double_scale,
        block_sizes=layer.weight_quantizer.block_sizes,
    )
    original_rotate = copy.deepcopy(layer.weight_quantizer._rotate)
    original_pre_scale = layer.weight_quantizer._enable_pre_quant_scale
    with torch.inference_mode(inference_staging):
        metadata = prepare_packed_weights_for_tp(model)
    assert metadata["0"]["rotate"] == original_rotate
    assert metadata["0"]["enable_pre_quant_scale"] == original_pre_scale
    layer.weight_quantizer.disable()
    x = torch.randn(3, columns, requires_grad=True)
    expected = F.linear(x, decoded)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)
    assert not layer.weight.requires_grad
    for dim in (0, 1):
        view = NativeWeightPartition(tuple(decoded.shape), dim, 1, 2)
        local = view.select(decoded)
        quantizer = copy.deepcopy(layer.weight_quantizer)
        if format_name == "fp8_block_padded" and dim == 1:
            with pytest.raises(ValueError):
                native_weight_partition(quantizer, view, source=local)
            continue
        fields, _, _ = native_weight_partition(quantizer, view, source=local)
        for name, value in fields.items():
            quantizer._set_buffer(name, value) if isinstance(value, torch.Tensor) else setattr(
                quantizer, name, value
            )
        pinned = get_pinned_native_weight(quantizer, local)
        assert pinned is not None
        codes = native._quantized_data.reshape(decoded.shape[0], -1)
        expected_codes = view.select(
            codes, 2 if format_name.startswith("nvfp4") and dim == 1 else 1
        )
        assert torch.equal(
            pinned._quantized_data.view(torch.uint8).reshape(expected_codes.shape),
            expected_codes.view(torch.uint8),
        )
        quantizer.enable()
        torch.testing.assert_close(quantizer(local), local, rtol=0, atol=0)
        assert get_pinned_native_weight(quantizer, local) is not None
        if format_name.startswith("fp8_block"):
            quantizer.axis = None
            assert get_pinned_native_weight(quantizer, local) is None
    layer.weight_quantizer.enable()
    path = tmp_path / "staged.pt"
    mto.save(model, path)
    restored = mto.restore(torch.nn.Sequential(torch.nn.Linear(columns, 16, bias=False)), path)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert get_pinned_native_weight(restored[0].weight_quantizer, restored[0].weight) is not None
    assert torch.equal(
        restored[0].weight_quantizer._native_weight_codes, native._quantized_data.view(torch.uint8)
    )
    # Re-pinning is supported after native field edits and must retain safe miss behavior.
    q = restored[0].weight_quantizer
    pin_native_weight(q, get_pinned_native_weight(q, restored[0].weight), restored[0].weight)
    # A changed master cannot silently switch an uncalibrated staged layer to dynamic amax.
    with torch.no_grad():
        restored[0].weight.add_(0.125)
    with pytest.raises(ValueError, match="calibrat"):
        restored(x)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_packed_staging_rolls_back_all_modules(monkeypatch, failure):
    from modelopt.torch.quantization import repair_partition
    from modelopt.torch.quantization.qtensor import FP8QTensor, QTensorWrapper
    from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

    model = mtq.quantize(
        torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)),
        {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None},
    )
    snapshots = []
    for layer in model:
        native, scale = FP8QTensor.quantize(layer.weight.detach(), axis=0)
        q = layer.weight_quantizer
        q.set_from_attribute_config(
            QuantizerAttributeConfig(num_bits=(4, 3), axis=0, fake_quant=False)
        )
        q._set_buffer("_scale", scale)
        layer.weight = QTensorWrapper(native)
        snapshots.append(
            (layer.weight, dict(q._buffers), q._fake_quant, q._rotate, q._enable_pre_quant_scale)
        )
    original = repair_partition.pin_native_weight
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure("staging interrupted")
        return original(*args)

    monkeypatch.setattr(repair_partition, "pin_native_weight", fail_second)
    with pytest.raises(failure, match="staging interrupted"):
        prepare_packed_weights_for_tp(model)
    for layer, (weight, buffers, fake_quant, rotate, pre_scale) in zip(model, snapshots):
        q = layer.weight_quantizer
        assert layer.weight is weight
        assert q._buffers.keys() == buffers.keys()
        assert all(q._buffers[name] is value for name, value in buffers.items())
        assert (q._fake_quant, q._rotate, q._enable_pre_quant_scale) == (
            fake_quant,
            rotate,
            pre_scale,
        )
        assert not hasattr(q, "_native_weight_meta")


@pytest.mark.parametrize("policy", ["uncalibrated", "constant", "dynamic"])
def test_native_partition_preserves_threshold_policy(policy):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(4, 3),
            axis=0,
            use_constant_amax=policy == "constant",
            type="dynamic" if policy == "dynamic" else "static",
        )
    )
    view = NativeWeightPartition((16, 32), 0, 0, 2)
    assert getattr(quantizer, "_amax", None) is None
    if policy == "uncalibrated":
        with pytest.raises(ValueError, match="require calibrated thresholds"):
            native_weight_partition(quantizer, view)
    else:
        fields, _, _ = native_weight_partition(quantizer, view)
        assert "_amax" not in fields
