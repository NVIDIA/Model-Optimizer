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
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.nn import functional as F

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill._bypass_parallel import validate_tp_quantizer_aliases
from modelopt.torch.quantization import repair_partition
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, IQ_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pin_packed_weight
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper
from modelopt.torch.quantization.repair_partition import (
    NativeWeightPartition,
    activation_partition,
    native_tp_weight_view,
    native_weight_partition,
    prepare_packed_weights_for_tp,
    validate_native_weight_partition,
    validate_tp_output_quantizer,
)
from modelopt.torch.quantization.utils.native_weight import (
    get_pinned_native_weight,
    pin_native_weight,
)


@pytest.mark.parametrize("name", [*sorted(IQ_FORMAT_REGISTRY), "q8_0"])
@pytest.mark.parametrize("dim", [0, 1])
def test_native_partition_preserves_pinned_projection(name, dim):

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


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, "clipped", "lsq", "learned"])
def test_packed_staging_rolls_back_all_modules(monkeypatch, failure):

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
    if isinstance(failure, str):
        q = model[1].weight_quantizer
        q.enable()
        if failure == "clipped":
            q._pass_through_bwd = False
        elif failure == "lsq":
            q._lsq = True
        else:
            q.register_parameter("learned_scale", torch.nn.Parameter(torch.ones(())))
    original = repair_partition.pin_native_weight
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure("staging interrupted")
        return original(*args)

    if not isinstance(failure, str):
        monkeypatch.setattr(repair_partition, "pin_native_weight", fail_second)
    with pytest.raises(
        ValueError if isinstance(failure, str) else failure,
        match="native backward policy" if isinstance(failure, str) else "staging interrupted",
    ):
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


@pytest.mark.parametrize("policy", ["uncalibrated", "scale_only", "constant", "dynamic"])
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
    if policy == "scale_only":
        quantizer._set_buffer("_scale", torch.ones(16, 1))
    if policy in ("uncalibrated", "scale_only"):
        with pytest.raises(ValueError, match="require calibrated thresholds"):
            native_weight_partition(quantizer, view)
    else:
        fields, _, _ = native_weight_partition(quantizer, view)
        assert "_amax" not in fields


@pytest.mark.parametrize("axis", [0, 1, -1, -2, (0,), (-2,), (0, 1), (-1, -2), None])
@pytest.mark.parametrize("dim", [0, 1])
def test_dynamic_partition_requires_local_threshold_reductions(axis, dim):
    torch.manual_seed(7)
    weight = torch.randn(4, 32) * torch.tensor([1.0] * 16 + [100.0] * 16)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=axis, type="dynamic")
    )
    view = NativeWeightPartition(tuple(weight.shape), dim, 0, 2)
    axes = (axis,) if isinstance(axis, int) else axis or ()
    before = copy.deepcopy(quantizer.state_dict())
    retained = {value % 2 for value in axes}
    if dim not in retained or retained == {0, 1}:
        with pytest.raises(ValueError, match="require calibrated thresholds"):
            native_weight_partition(quantizer, view)
    else:
        fields, _, _ = native_weight_partition(quantizer, view)
        assert not fields
        torch.testing.assert_close(
            quantizer(view.select(weight)), view.select(quantizer(weight)), rtol=0, atol=0
        )
    assert quantizer.state_dict().keys() == before.keys()
    assert all(torch.equal(value, before[name]) for name, value in quantizer.state_dict().items())


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("provenance", ["global", "missing", "format_drift", "local_stale"])
def test_native_fp8_partition_requires_matching_pin_provenance(dim, provenance):
    weight = torch.randn(4, 32)
    native, scale = FP8QTensor.quantize(weight, axis=0)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=0))
    quantizer._set_buffer("_scale", scale)
    quantizer.amax = weight.abs().amax(dim=1, keepdim=True)
    if provenance == "format_drift":
        weight = native.dequantize(scale=scale)
    pin_native_weight(quantizer, native, weight)
    view = NativeWeightPartition(tuple(weight.shape), dim, 0, 2)
    if provenance == "local_stale":
        local = view.select(weight)
        local_native, local_scale = FP8QTensor.quantize(local, axis=0)
        quantizer._set_buffer("_scale", local_scale)
        pin_native_weight(quantizer, local_native, local)
        with pytest.raises(ValueError):
            native_weight_partition(quantizer, view, source=local + 1)
        return
    if provenance != "global":
        if provenance == "format_drift":
            quantizer.num_bits = (5, 2)
        with pytest.raises(ValueError):
            native_weight_partition(
                quantizer, view, source=None if provenance == "missing" else view.select(weight)
            )
        return
    fields, _, _ = native_weight_partition(quantizer, view, source=weight)
    for name, value in fields.items():
        quantizer._set_buffer(name, value) if isinstance(value, torch.Tensor) else setattr(
            quantizer, name, value
        )
    local = view.select(weight)
    assert get_pinned_native_weight(quantizer, local) is not None
    expected = view.select(native.dequantize(scale=scale))
    torch.testing.assert_close(quantizer(local), expected, rtol=0, atol=0)
    stale = copy.deepcopy(quantizer)
    with torch.no_grad():
        local.add_(1)
    assert get_pinned_native_weight(stale, local) is None


def test_dynamic_nvfp4_partition_refuses_unavailable_global_maximum():
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            type="dynamic",
            block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        )
    )
    quantizer._set_buffer("_scale", torch.ones(4, 2))
    with pytest.raises(ValueError):
        native_weight_partition(quantizer, NativeWeightPartition((4, 32), 1, 0, 2))


@pytest.mark.parametrize("mixed", [False, True])
def test_compressed_staging_preserves_public_save_restore(tmp_path, mixed):
    torch.manual_seed(63)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 16, bias=False), torch.nn.Linear(16, 8, bias=False)
    )
    inputs = torch.randn(3, 32)
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "enable": True,
                    "cfg": {"num_bits": (4, 3), "axis": 0},
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(inputs),
    )
    mtq.compress(model)
    assert isinstance(model[0].weight, QTensorWrapper)
    expected = model(inputs)
    packed_path = tmp_path / "compressed.pt"
    mto.save(model, packed_path)
    packed_restored = mto.restore(
        torch.nn.Sequential(
            torch.nn.Linear(32, 16, bias=False), torch.nn.Linear(16, 8, bias=False)
        ),
        packed_path,
    )
    assert isinstance(packed_restored[0].weight, QTensorWrapper)
    torch.testing.assert_close(packed_restored(inputs), expected, rtol=0, atol=0)
    sibling_codes = model[1].weight.get_qtensor()._quantized_data.view(torch.uint8).clone()
    prepare_packed_weights_for_tp(model[0] if mixed else model)
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    codes = model[0].weight_quantizer._native_weight_codes.clone()
    quantizer = model[0].weight_quantizer
    pin_native_weight(
        quantizer, get_pinned_native_weight(quantizer, model[0].weight), model[0].weight
    )
    path = tmp_path / "compressed-staged.pt"
    mto.save(model, path)
    restored = mto.restore(
        torch.nn.Sequential(
            torch.nn.Linear(32, 16, bias=False), torch.nn.Linear(16, 8, bias=False)
        ),
        path,
    )
    assert isinstance(restored[1].weight, QTensorWrapper) == mixed
    if mixed:
        assert torch.equal(
            restored[1].weight.get_qtensor()._quantized_data.view(torch.uint8), sibling_codes
        )
    assert not isinstance(restored[0].weight, QTensorWrapper)
    assert not restored[0].weight.requires_grad
    assert get_pinned_native_weight(restored[0].weight_quantizer, restored[0].weight) is not None
    assert torch.equal(restored[0].weight_quantizer._native_weight_codes, codes)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduced", [(-1,), (-2,), (-1, -2)])
@pytest.mark.parametrize("dim", [0, 1])
def test_dynamic_partition_derives_unconsumed_block_axes(reduced, dim):
    blocks = dict.fromkeys(reduced)
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), type="dynamic", block_sizes=blocks)
    )
    weight = torch.randn(4, 32)
    view = NativeWeightPartition(tuple(weight.shape), dim, 0, 2)
    retained = {0, 1} - {value % 2 for value in reduced}
    if dim not in retained or retained == {0, 1}:
        with pytest.raises(ValueError, match="require calibrated thresholds"):
            native_weight_partition(quantizer, view)
    else:
        fields, _, _ = native_weight_partition(quantizer, view)
        assert not fields
        torch.testing.assert_close(
            copy.deepcopy(quantizer)(view.select(weight)),
            view.select(copy.deepcopy(quantizer)(weight)),
            rtol=0,
            atol=0,
        )
    assert quantizer.axis is None
    assert quantizer.block_sizes == blocks


@pytest.mark.parametrize("format_name", ["fp8", "nvfp4"])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("failure", [None, "missing_codes", "incomplete_row", "wrong_shape"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_partition_preflight_uses_only_metadata(format_name, dim, failure, dynamic):
    weight = torch.randn(8, 32)
    if format_name == "fp8":
        native, scale = FP8QTensor.quantize(weight, axis=0)
        attributes = {"num_bits": (4, 3), "axis": 0}
        double_scale = None
    else:
        native, scale, double_scale = NVFP4QTensor.quantize(weight, 16, try_tensorrt=False)
        attributes = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        }
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            **attributes, fake_quant=False, type="dynamic" if dynamic else "static"
        )
    )
    quantizer._set_buffer("_scale", scale)
    if double_scale is not None:
        quantizer._set_buffer("_double_scale", double_scale)
    source = native.dequantize(
        scale=scale, double_scale=double_scale, block_sizes=quantizer.block_sizes
    )
    pin_native_weight(quantizer, native, source)
    quantizer.to("meta")
    if failure == "missing_codes":
        del quantizer._native_weight_codes
    elif failure == "incomplete_row":
        quantizer._native_weight_codes = quantizer._native_weight_codes.flatten()[:1]
    elif failure == "wrong_shape":
        quantizer._native_weight_meta["source_shape"] = (6, 32)
    buffers = dict(quantizer.named_buffers())
    parameters = dict(quantizer.named_parameters())
    metadata = copy.deepcopy(quantizer._native_weight_meta)
    view = NativeWeightPartition((8, 32), dim, 1, 2)
    accepted = failure is None and (not dynamic or (format_name == "fp8" and dim == 0))
    if accepted:
        validate_native_weight_partition(quantizer, view)
        validate_tp_quantizer_aliases(quantizer, view, {})
    else:
        with pytest.raises(ValueError):
            validate_native_weight_partition(quantizer, view)
    if failure is None and not accepted:
        with pytest.raises(ValueError):
            validate_tp_quantizer_aliases(quantizer, view, {})
    assert quantizer._native_weight_meta == metadata
    assert not quantizer.fake_quant
    assert quantizer._buffers.keys() == buffers.keys()
    assert all(quantizer._buffers[name] is value for name, value in buffers.items())
    assert quantizer._parameters.keys() == parameters.keys()
    assert all(quantizer._parameters[name] is value for name, value in parameters.items())


@pytest.mark.parametrize(
    ("name", "shape", "placement", "allow_expert_axis", "expected"),
    [
        ("weight", (8, 32), Shard(0), False, ((4, 32), [(4, 4)])),
        ("weight", (8, 32), Shard(-1), False, ((8, 16), [(16, 16)])),
        ("gate_up_proj", (3, 8, 32), Shard(1), False, ((4, 32), [(4, 4)])),
        ("down_proj", (3, 8, 32), Shard(2), False, ((8, 16), [(16, 16)])),
        (
            "gate_up_proj",
            (3, 8, 32),
            _StridedShard(1, split_factor=2),
            False,
            ((4, 32), [(2, 2), (6, 2)]),
        ),
        ("gate_up_proj", (3, 8, 32), Shard(0), True, None),
        ("weight", (8, 32), Replicate(), False, None),
        ("weight", (3, 8, 32), Shard(1), False, ValueError),
        ("gate_up_proj", (3, 8, 32), Shard(0), False, ValueError),
        ("gate_up_proj", (3, 8, 32), _StridedShard(0, split_factor=2), True, ValueError),
        ("weight", (6, 32), _StridedShard(0, split_factor=2), False, ValueError),
    ],
)
def test_native_tp_view_respects_physical_weight_ownership(
    name, shape, placement, allow_expert_axis, expected
):
    # Geometry resolution needs only mesh metadata, without initializing collectives.
    mesh = SimpleNamespace(get_local_rank=lambda: 1, size=lambda: 2)
    if expected is ValueError:
        with pytest.raises(ValueError):
            native_tp_weight_view(name, shape, placement, mesh, allow_expert_axis=allow_expert_axis)
    else:
        view = native_tp_weight_view(
            name, shape, placement, mesh, allow_expert_axis=allow_expert_axis
        )
        if expected is None:
            assert view is None
        else:
            local_shape, intervals = expected
            assert view.local_shape == local_shape
            assert view.intervals() == intervals


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("quantizing", [False, True])
@pytest.mark.parametrize("style", ["colwise_output", "rowwise_input"])
def test_tp_refuses_rank_local_calibration(rank, quantizing, style):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None))
    quantizer.amax = torch.tensor(8.0)
    if not quantizing:
        quantizer.disable_quant()
    quantizer.enable_calib()
    # Existing scalar amax is insufficient: actual calibration replaces it with
    # different rank-local maxima, instead of the unsharded maximum of eight.
    quantizer(torch.tensor([1.0 if rank == 0 else 8.0]))
    quantizer.load_calib_amax()
    assert quantizer.amax.item() == (1.0 if rank == 0 else 8.0)
    view = NativeWeightPartition((4, 16), int(style == "rowwise_input"), rank, 2)
    validate = validate_tp_output_quantizer if style == "colwise_output" else activation_partition
    with pytest.raises(ValueError, match=r"calibration.*disabled"):
        validate(quantizer, view)
    quantizer.disable_calib()
    validate(quantizer, view)


@pytest.mark.parametrize("field", ["_amax", "_global_amax", "_pre_quant_scale", "_bias_value"])
def test_rowwise_activation_partition_refuses_tied_fields(field):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=-1))
    quantizer.amax = torch.ones(1, 1, 16)
    original = torch.arange(1.0, 17.0).reshape(1, 1, 16)
    quantizer._set_buffer(field, original)
    quantizer.__dict__["_shared_quant_tied_attrs"] = {field}
    before = dict(quantizer._buffers)
    with pytest.raises(ValueError, match="tied shared quantizer field"):
        activation_partition(quantizer, NativeWeightPartition((4, 16), 1, 0, 2))
    assert quantizer._buffers.keys() == before.keys()
    assert all(quantizer._buffers[name] is value for name, value in before.items())
    assert not hasattr(quantizer, "_native_tp_activation_layout")


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("rotate_back", [False, True])
def test_rowwise_output_rotation_preserves_bias_contract(biased, rotate_back):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            enable=False,
            rotate={
                "enable": True,
                "mode": "rotate_back" if rotate_back else "rotate",
                "block_size": 8,
            },
        )
    )
    view = NativeWeightPartition((8, 16), 1, 0, 2)
    if biased and not rotate_back:
        with pytest.raises(ValueError, match="after bias addition"):
            validate_tp_output_quantizer(quantizer, view, has_bias=biased)
    else:
        validate_tp_output_quantizer(quantizer, view, has_bias=biased)
