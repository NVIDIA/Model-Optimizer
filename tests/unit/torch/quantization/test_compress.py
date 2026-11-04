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

"""CPU tests of mtq.compress."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.compress import is_real_quantized
from modelopt.torch.quantization.nn import GroupedQuantizer


@pytest.mark.parametrize("four_over_six", [False, True])
def test_static_nvfp4_repair_preserves_calibrated_initialization(four_over_six):
    from modelopt.torch.distill import LocalDistillationConfig
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import StaticBlockScaleQuantizer
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy

    torch.manual_seed(12)
    model = mtq.quantize(
        nn.Sequential(nn.Linear(256, 8, bias=False)),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {
                        "num_bits": (2, 1),
                        "block_sizes": {
                            -1: 16,
                            "type": "static",
                            "scale_bits": (4, 3),
                            "four_over_six": four_over_six,
                        },
                    },
                },
            ],
            "algorithm": "max",
        },
    )
    quantizer = StaticBlockScaleQuantizer.from_tensor_quantizer(
        model[0].weight_quantizer, global_amax=model[0].weight.detach().abs().max()
    )
    quantizer._amax.mul_(0.7)
    scales, global_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
        quantizer, model[0].weight
    )
    native, _, _ = NVFP4QTensor.quantize(
        model[0].weight, 16, weights_scaling_factor=scales, weights_scaling_factor_2=global_scale
    )
    inputs = torch.randn(3, 256)
    expected = torch.nn.functional.linear(
        inputs,
        native.dequantize(
            scale=scales, double_scale=global_scale, block_sizes=quantizer.block_sizes
        ),
    )
    policy = QuantizationRepairPolicy()
    try:
        policy.prepare(model, LocalDistillationConfig(updates=1))
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    finally:
        policy.close()


@pytest.mark.parametrize(
    "weight_cfg",
    [
        {"num_bits": (4, 3), "axis": 0},
        {"num_bits": (4, 3), "block_sizes": {-2: 128, -1: 128}},
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
        {"num_bits": (2, 1), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
        {"num_bits": (4, 3), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
    ],
    ids=["fp8", "fp8_2d", "nvfp4", "mxfp4", "mxfp8"],
)
def test_fake_ptq_native_scale_repair_and_reload(weight_cfg, tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.distill import LocalDistillationConfig
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy
    from modelopt.torch.quantization.utils.native_weight import get_pinned_native_weight

    torch.manual_seed(12)
    tail = weight_cfg["num_bits"] == (4, 3) and weight_cfg.get("block_sizes", {}).get(
        "scale_bits"
    ) == (8, 0)
    in_features = 33 if tail else 256
    out_features = 2 if tail else (128 if -2 in weight_cfg.get("block_sizes", {}) else 8)
    model = mtq.quantize(
        nn.Sequential(nn.Linear(in_features, out_features, bias=False)),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": weight_cfg},
            ],
            "algorithm": "max",
        },
    )
    if weight_cfg["num_bits"] == (2, 1) and weight_cfg.get("block_sizes", {}).get("scale_bits") == (
        4,
        3,
    ):
        model[0].input_quantizer.pre_quant_scale = torch.linspace(0.8, 1.2, 256)
    reference = copy.deepcopy(model)
    mtq.compress(reference)
    inputs = torch.randn(3, in_features)
    expected = reference(inputs).detach()
    original_payload = reference[0].weight.detach().view(torch.uint8).clone()
    quantizer = model[0].weight_quantizer
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(model, LocalDistillationConfig(updates=1))
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    optimizer = torch.optim.Adagrad([{"params": g.params, "lr": g.lr} for g in groups])
    model(inputs).square().mean().backward()
    optimizer.step()
    policy.project_()
    changed = model(inputs).detach()
    policy.finalize_()
    torch.testing.assert_close(model(inputs), changed, rtol=0, atol=0)
    native = get_pinned_native_weight(quantizer, model[0].weight)
    assert torch.equal(native._quantized_data.view(torch.uint8), original_payload)
    from modelopt.torch.export.quantized_weight_export import (
        capture_quantized_weight_export_state,
        export_quantized_weight_tensors,
    )

    export_state = capture_quantized_weight_export_state(model[0])
    exported = export_quantized_weight_tensors(model[0].weight, export_state, dtype=torch.float32)
    assert torch.equal(exported["weight"].view(torch.uint8), original_payload)
    torch.testing.assert_close(
        exported["weight_scale"].float().reshape(-1),
        quantizer._scale.float().reshape(-1),
        rtol=0,
        atol=0,
    )
    if hasattr(quantizer, "_double_scale"):
        torch.testing.assert_close(
            exported["weight_scale_2"], quantizer._double_scale.squeeze(), rtol=0, atol=0
        )
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    deployment = copy.deepcopy(model[0])
    _export_quantized_weight(deployment, torch.float32)
    assert torch.equal(deployment.weight.view(torch.uint8), original_payload)
    torch.testing.assert_close(
        deployment.weight_scale.float().reshape(-1),
        quantizer._scale.float().reshape(-1),
        rtol=0,
        atol=0,
    )
    mto.save(model, tmp_path / "repaired.pt")
    restored = mto.restore(
        nn.Sequential(nn.Linear(in_features, out_features, bias=False)), tmp_path / "repaired.pt"
    )
    torch.testing.assert_close(restored(inputs), changed, rtol=0, atol=0)
    restored_native = get_pinned_native_weight(restored[0].weight_quantizer, restored[0].weight)
    assert torch.equal(restored_native._quantized_data.view(torch.uint8), original_payload)
    native_fields = {
        name: value.clone()
        for name, value in restored[0].weight_quantizer.named_buffers()
        if name in ("_scale", "_double_scale", "_scale_zeros")
    }
    restored.to(torch.bfloat16)
    for name, expected in native_fields.items():
        actual = getattr(restored[0].weight_quantizer, name)
        assert actual.dtype == expected.dtype
        assert torch.equal(
            actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
        )
    assert get_pinned_native_weight(restored[0].weight_quantizer, restored[0].weight) is not None
    inputs = inputs.bfloat16()
    before = restored(inputs).detach()
    policy = QuantizationRepairPolicy()
    policy.prepare(restored, LocalDistillationConfig(updates=1))
    torch.testing.assert_close(restored(inputs), before, rtol=0, atol=0)
    policy.finalize_()
    torch.testing.assert_close(restored(inputs), before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "weight_cfg",
    [
        {"num_bits": 8, "axis": 0},
        {"num_bits": 4, "block_sizes": {-1: 32}},
        {"num_bits": (4, 3), "axis": 0},
        {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
        {"num_bits": (2, 1), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
        {"num_bits": (4, 3), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
        {"num_bits": 4, "block_sizes": {-1: 64, "scale_bits": 8, "scale_block_sizes": {-1: 16}}},
    ],
    ids=["int8", "int4", "fp8", "nvfp4", "mxfp4", "mxfp8", "nf4"],
)
def test_compressed_native_scale_repair_preserves_payload(weight_cfg, tmp_path):
    import modelopt.torch.opt as mto
    from modelopt.torch.distill import LocalDistillationConfig
    from modelopt.torch.quantization.repair import QuantizationRepairPolicy

    torch.manual_seed(12)
    in_features, out_features = 256, 8
    if weight_cfg["num_bits"] == 4:
        in_features, out_features = (
            (66 if weight_cfg["block_sizes"].get("scale_bits") == 8 else 34),
            1,
        )
    elif weight_cfg["num_bits"] == (4, 3) and weight_cfg.get("block_sizes", {}).get(
        "scale_bits"
    ) == (8, 0):
        in_features, out_features = 33, 2
    layer = mtq.quantize(
        nn.Sequential(nn.Linear(in_features, out_features, bias=False)),
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": weight_cfg},
            ],
            "algorithm": "max",
        },
    )
    mtq.compress(layer)
    quantizer = layer[0].weight_quantizer
    if weight_cfg["num_bits"] == (2, 1) and weight_cfg["block_sizes"]["scale_bits"] == (4, 3):
        from modelopt.torch.export.diffusers_utils import swizzle_nvfp4_scales

        fields = swizzle_nvfp4_scales(
            {
                "probe.weight": layer[0].weight.detach().view(torch.uint8),
                "probe.weight_scale": quantizer._scale,
                "probe.weight_scale_2": quantizer._double_scale,
            }
        )
        quantizer._scale = fields["probe.weight_scale"].view(torch.uint8).reshape(-1)
    scale_layout = (quantizer._scale.dtype, quantizer._scale.shape)
    inputs = torch.randn(3, in_features)
    original = layer(inputs).detach()
    original_payload = layer[0].weight.detach().view(torch.uint8).clone()
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(layer, LocalDistillationConfig(updates=1))
    assert not layer[0].allow_real_quant_gemm
    torch.testing.assert_close(layer(inputs), original, rtol=0, atol=0)
    optimizer = torch.optim.Adagrad([{"params": g.params, "lr": g.lr} for g in groups])
    layer(inputs).square().mean().backward()
    optimizer.step()
    policy.project_()
    changed = layer(inputs).detach()
    policy.finalize_()
    assert layer[0].allow_real_quant_gemm
    assert (quantizer._scale.dtype, quantizer._scale.shape) == scale_layout
    assert torch.equal(layer[0].weight.view(torch.uint8), original_payload)
    torch.testing.assert_close(layer(inputs), changed, rtol=0, atol=0)
    mto.save(layer, tmp_path / "repaired.pt")
    restored = mto.restore(
        nn.Sequential(nn.Linear(in_features, out_features, bias=False)), tmp_path / "repaired.pt"
    )
    assert torch.equal(restored[0].weight.view(torch.uint8), original_payload)
    torch.testing.assert_close(restored(inputs), changed, rtol=0, atol=0)


@pytest.mark.parametrize("grouped", [False, True], ids=["linear", "grouped"])
def test_compress_refuses_w4a8_weights_before_changing_the_model(grouped):
    """A compressed W4A8 (SequentialQuantizer) weight can neither run forward nor be restored."""
    model = nn.Sequential(nn.Linear(128, 128, bias=False))  # 128 wide for W4A8's INT4 blocks
    mtq.quantize(model, mtq.W4A8_AWQ_BETA_CFG, lambda m: m(torch.randn(2, 128)))
    if grouped:  # one per expert, as Transformer Engine's grouped linear holds them
        model[0].weight_quantizer = GroupedQuantizer(model[0].weight_quantizer)

    with pytest.raises(NotImplementedError, match="W4A8"):
        mtq.compress(model)

    assert not is_real_quantized(model)
