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

"""Native quantization repair codecs, gradients and persistence on CUDA."""

import copy
import os

import pytest
import torch
from torch import nn

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.repair import QuantizationRepairPolicy
from modelopt.torch.quantization.repair_formats import grouped_native_repair, native_scale_execution


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_iq_scale_kernel_matches_native_torch_gradients(dtype, monkeypatch):
    """Fused scale application retains projected values and local scale gradients."""
    # Defer optional Triton imports so CUDA tests can be collected on CPU.
    from modelopt.torch.kernels.quantization.repair import apply_block_scales, decode_iq_levels

    for name in ("iq1_s", "iq1_m", "iq2_s", "iq2_xs", "iq2_xxs"):
        fmt = GGML_FORMAT_REGISTRY[name]
        packed = torch.randint(0, 256, (47, fmt.block_bytes), device="cuda", dtype=torch.uint8)
        expected_levels, _ = fmt.decode_int8(packed, torch.tensor([47, 256]))
        actual_levels = decode_iq_levels(packed, name, (47, 256))
        torch.testing.assert_close(actual_levels, expected_levels, rtol=0, atol=0)
        layer = nn.Linear(256, 2, bias=False).to(device="cuda", dtype=dtype)
        mtq.quantize(
            layer,
            {
                "quant_cfg": [
                    {"quantizer_name": "*", "enable": False},
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": name, "backend": "ggml"},
                    },
                ],
                "algorithm": "max",
            },
        )
        policy = QuantizationRepairPolicy()
        groups = policy.prepare(layer, mtd.LocalDistillationConfig())
        parameters = [p for group in groups for p in group.params]
        with torch.no_grad():
            for parameter in parameters:
                parameter.uniform_(-0.1, 0.1)
        inputs = torch.randn(3, 256, device="cuda", dtype=dtype)
        reference = layer(inputs)
        reference.float().square().sum().backward()
        gradients = [parameter.grad.clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = None

        def reference_unpack_forbidden(*args, **kwargs):
            raise AssertionError("Native IQ replay must avoid the reference unpacker.")

        with monkeypatch.context() as patch, native_scale_execution():
            patch.setattr(type(fmt), "decode_int8", reference_unpack_forbidden)
            actual = layer(inputs)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            actual.float().square().sum().backward()
        for parameter, expected in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=1e-6)
        policy.finalize_()
        torch.testing.assert_close(layer(inputs), reference, rtol=0, atol=0)
    torch.manual_seed(43)
    levels = torch.randint(-9, 10, (513, 16), device="cuda", dtype=torch.int8)
    scales = torch.randn(513, device="cuda", requires_grad=True)
    gradient = torch.randn_like(levels, dtype=dtype)
    expected = (levels.float() * scales[:, None]).to(dtype)
    expected.backward(gradient)
    expected_gradient = scales.grad.clone()
    scales.grad = None
    actual = apply_block_scales(levels, scales, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(gradient)
    torch.testing.assert_close(scales.grad, expected_gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compressed_nf4_native_rounding(dtype, tmp_path):
    """Native CUDA BF16 decode rounding survives scale repair and reload."""
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(256, 16, bias=False)).to(device="cuda", dtype=dtype)
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {
                        "num_bits": 4,
                        "block_sizes": {-1: 64, "scale_bits": 8, "scale_block_sizes": {-1: 16}},
                    },
                },
            ],
            "algorithm": "max",
        },
    )
    mtq.compress(model)
    inputs = torch.randn(2, 256, device="cuda", dtype=dtype)
    expected = model(inputs).detach()
    codes = model[0].weight.detach().view(torch.uint8).clone()
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(model, mtd.LocalDistillationConfig(updates=1))
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    optimizer = torch.optim.Adagrad([{"params": g.params, "lr": g.lr} for g in groups])
    model(inputs).float().square().mean().backward()
    optimizer.step()
    policy.project_()
    repaired = model(inputs).detach()
    policy.finalize_()
    torch.testing.assert_close(model(inputs), repaired, rtol=0, atol=0)
    assert torch.equal(model[0].weight.view(torch.uint8), codes)
    mto.save(model, tmp_path / "nf4.pt")
    restored = mto.restore(
        nn.Sequential(nn.Linear(256, 16, bias=False)).to("cuda", dtype),
        tmp_path / "nf4.pt",
    ).cuda()
    restored_quantizer = restored[0].weight_quantizer
    assert all(
        getattr(restored_quantizer, field).device == inputs.device
        for field in ("_scale", "_double_scale", "_scale_zeros")
    )
    torch.testing.assert_close(restored(inputs), repaired, rtol=0, atol=0)


@pytest.mark.timeout(600)
def test_native_repair_cuda(tmp_path):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    workers = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cases = [
        ("int8", {"num_bits": 8, "axis": 0}),
        (
            "nvfp4",
            {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}},
        ),
        (
            "mxfp4",
            {"num_bits": (2, 1), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}},
        ),
        ("iq2_s", {"num_bits": "iq2_s", "backend": "ggml"}),
        *[
            (
                name,
                {
                    "num_bits": bits,
                    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)},
                },
            )
            for name, bits in [("nvfp6_e3m2", (3, 2)), ("nvfp8_e5m2", (5, 2))]
        ],
        *[
            (
                name,
                {
                    "num_bits": bits,
                    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
                },
            )
            for name, bits in [
                ("mxfp6_e3m2", (3, 2)),
                ("mxint8", 8),
                ("mxfp8_e5m2", (5, 2)),
                ("mxfp6_e2m3", (2, 3)),
                ("mx4_e1m2", (1, 2)),
            ]
        ],
    ]
    for name, weight_config in cases[rank::workers]:
        torch.manual_seed(13)
        teacher = (
            nn.Sequential(nn.Linear(256, 32, bias=False), nn.SiLU(), nn.Linear(32, 256, bias=False))
            .to(device, torch.bfloat16)
            .eval()
        )
        student = copy.deepcopy(teacher)
        batches = [torch.randn(8, 16, 256, device=device, dtype=torch.bfloat16) for _ in range(2)]
        quantizers = [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "0.weight_quantizer" if name == "iq2_s" else "*weight_quantizer",
                "cfg": weight_config,
            },
        ]
        if name == "nvfp4":
            quantizers.append({"quantizer_name": "*input_quantizer", "cfg": weight_config})
        mtq.quantize(
            student, {"quant_cfg": quantizers, "algorithm": "max"}, lambda model: model(batches[0])
        )
        initial_mse = (
            (student(batches[0]).float() - teacher(batches[0]).float()).square().mean().item()
        )
        result = mtd.local_distill(
            student,
            teacher,
            batches,
            {
                "units": [""],
                "updates": 2,
                "global_batch_size": 8,
                "micro_batch_size": 2,
                "scheduler": "constant",
                "warmup_fraction": 0,
            },
            validation_dataloader=batches[:1],
        )
        if name.startswith(("nvfp6", "nvfp8")):
            torch.testing.assert_close(
                result.metrics["units"][""]["baseline_validation_mse"],
                initial_mse,
                rtol=2.5e-2,
                atol=1e-7,
            )
        assert torch.isfinite(result.student(batches[0])).all()
        assert all(row["units"][""]["gradient_norm"] > 0 for row in result.metrics["history"])
        path = tmp_path / (name + ".pt")
        mto.save(result.student, path)
        restored = mto.restore(copy.deepcopy(teacher), path)
        torch.testing.assert_close(restored(batches[0]), result.student(batches[0]), rtol=0, atol=0)


@pytest.mark.manual
def test_iq_scale_kernels_large_expert_bank():
    """Native decode, application and gradients must address banks beyond 2^31 elements."""
    # Defer optional Triton imports so CUDA tests can be collected on CPU.
    from modelopt.torch.kernels.quantization.repair import apply_block_scales, decode_iq_levels

    elements = 2**31 + 256
    packed = torch.zeros((elements // 256, 50), device="cuda", dtype=torch.uint8)
    reference, _ = GGML_FORMAT_REGISTRY["iq1_s"].decode_int8(packed[:1], torch.tensor([1, 256]))
    levels = decode_iq_levels(packed, "iq1_s", (elements // 256, 256)).flatten()
    indices = torch.cat(
        [
            torch.arange(256, device="cuda"),
            torch.arange(2**31 - 256, 2**31 + 256, device="cuda"),
        ]
    )
    expected = reference.flatten().repeat(3)
    torch.testing.assert_close(levels[indices], expected, rtol=0, atol=0)
    scales = torch.ones(elements // 32, device="cuda", requires_grad=True)
    output = apply_block_scales(levels, scales, torch.bfloat16)
    torch.testing.assert_close(output[indices], expected.bfloat16(), rtol=0, atol=0)
    output.backward(torch.ones_like(output))
    groups = torch.cat(
        [
            torch.arange(8, device="cuda"),
            torch.arange(2**31 // 32 - 8, elements // 32, device="cuda"),
        ]
    )
    expected_gradient = reference.reshape(8, 32).float().sum(-1).repeat(3)
    torch.testing.assert_close(scales.grad[groups], expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("format_name", "axis"), [("fp8", None), ("fp8", 0), ("nvfp4", None), ("iq2_s", None)]
)
def test_grouped_native_repair_matches_scalar_values_and_gradients(format_name, axis):

    cfg = (
        {"num_bits": (4, 3), "axis": axis}
        if format_name == "fp8"
        else (
            {"num_bits": "iq2_s", "backend": "ggml"}
            if format_name == "iq2_s"
            else {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            }
        )
    )
    if format_name == "nvfp4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip(
            "Grouped native FP4 decode requires Blackwell; scalar repair remains supported."
        )
    torch.manual_seed(79)
    model = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(3)]).cuda().bfloat16()
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": cfg},
            ],
            "algorithm": "max",
        },
    )
    policy = QuantizationRepairPolicy()
    groups = policy.prepare(model, mtd.LocalDistillationConfig())
    quantizers = [layer.weight_quantizer for layer in model]
    parameters = [p for group in groups for p in group.params]
    try:
        with torch.no_grad():
            for parameter in parameters:
                parameter.uniform_(-0.05, 0.05)

        def compare(layers):
            with native_scale_execution():
                actual = grouped_native_repair(
                    [layer.weight_quantizer for layer in layers], torch.bfloat16
                )
            assert actual is not None
            expected = torch.stack([layer.weight_quantizer(layer.weight) for layer in layers])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            upstream = torch.randn_like(actual)
            actual_grad = torch.autograd.grad(actual, parameters, upstream)
            expected_grad = torch.autograd.grad(expected, parameters, upstream)
            for observed, reference in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(observed, reference, rtol=2e-5, atol=2e-5)

        compare(model)
        state = quantizers[0]._local_repair_state
        constants = state._grouped_constants.values
        payload_name = "packed" if format_name == "iq2_s" else "packed_codes"
        original_codes = getattr(state, payload_name).view(torch.uint8).clone()
        compare(model)
        assert state._grouped_constants.values is constants
        torch.testing.assert_close(
            getattr(state, payload_name).view(torch.uint8), original_codes, rtol=0, atol=0
        )
        assert sum(value.numel() * value.element_size() for value in constants.values()) <= 262144
        assert all(
            not value.requires_grad and value.grad_fn is None for value in constants.values()
        )
        with torch.no_grad():
            for parameter in parameters:
                parameter.add_(0.01)
            payload = getattr(state, payload_name)
            payload.view(torch.uint8).zero_()
            base = state.global_base if format_name == "iq2_s" else state.base_scale
            base.mul_(0.5)
        compare(model)
        assert state._grouped_constants.values is not constants
        constants = state._grouped_constants.values
        setattr(state, payload_name, getattr(state, payload_name).clone())
        compare(model)
        assert state._grouped_constants.values is not constants
        compare(list(reversed(model)))
        state.load_state_dict(state.state_dict())
        assert state._grouped_constants is None
        compare(model)
        state.to(dtype=torch.float64)
        assert state._grouped_constants is None
        with native_scale_execution():
            assert grouped_native_repair([quantizers[0], None], torch.bfloat16) is None
    finally:
        policy.close()
