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

"""CUDA-only latent initialization formats and initialization memory/time."""

import copy
import time
from contextlib import nullcontext

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.extensions import (
    get_cuda_ext,
    get_cuda_ext_fp8,
    get_cuda_ext_ggml,
    get_cuda_ext_mx,
)
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.utils import enable_weight_cache, weight_cache_scope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def compiled_quantization_extensions(request):
    # Cold JIT builds belong in setup, outside the numerical tests' call-only timeout.
    params = request.node.callspec.params if hasattr(request.node, "callspec") else {}
    config = params.get("config")
    if config is None:
        num_bits = params.get("num_bits")
        config = (
            mtq.NVFP4_DEFAULT_CFG
            if num_bits == "nvfp4"
            else {"quant_cfg": [{"cfg": {"backend": "ggml"}}]}
            if num_bits == "iq2_xxs"
            else mtq.INT8_WEIGHT_ONLY_CFG
        )
    loaders = set()
    for rule in config["quant_cfg"]:
        if rule.get("enable") is False:
            continue
        attributes = rule.get("cfg", {})
        for attr in attributes if isinstance(attributes, list) else [attributes]:
            if attr.get("backend") == "ggml":
                loaders.add(get_cuda_ext_ggml)
            elif (attr.get("block_sizes") or {}).get("type") == "dynamic":
                loaders.add(get_cuda_ext_mx)
            elif attr.get("num_bits") == (4, 3):
                loaders.add(get_cuda_ext_fp8)
            else:
                loaders.add(get_cuda_ext)
    for loader in loaders:
        assert loader(raise_if_failed=True) is not None


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(mtq.INT8_WEIGHT_ONLY_CFG, id="int8"),
        pytest.param(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, id="int4_blockwise"),
        pytest.param(mtq.FP8_DEFAULT_CFG, id="fp8"),
        pytest.param(mtq.NVFP4_DEFAULT_CFG, id="nvfp4"),
        pytest.param(mtq.MXFP4_DEFAULT_CFG, id="mxfp4"),
        pytest.param(mtq.MXFP6_DEFAULT_CFG, id="mxfp6"),
        pytest.param(mtq.MXFP8_DEFAULT_CFG, id="mxfp8"),
        pytest.param(mtq.MXINT8_DEFAULT_CFG, id="mxint8"),
        pytest.param(mtq.W4A8_MXFP4_FP8_CFG, id="w4a8_mxfp4_fp8"),
        pytest.param(mtq.W4A8_NVFP4_FP8_CFG, id="w4a8_nvfp4_fp8"),
        pytest.param(mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG, id="fp8_2d_blockwise"),
        *[
            pytest.param(
                {
                    "quant_cfg": [
                        {"quantizer_name": "*", "enable": False},
                        {
                            "quantizer_name": "*weight_quantizer",
                            "cfg": {"num_bits": name, "backend": "ggml"},
                            "enable": True,
                        },
                    ],
                    "algorithm": "max"
                    if algorithm == "ptq"
                    else {
                        "method": "gptq",
                        "block_size": 256,
                        "perc_damp": 0.3,
                        "importance_weighted": True,
                    },
                },
                id=f"{name}-{algorithm}",
            )
            for name in GGML_FORMAT_REGISTRY
            for algorithm in ["ptq", "gptq"]
        ],
    ],
)
def test_initialize_cuda_weight_formats(config, record_property):
    torch.manual_seed(0)
    model = nn.Linear(256, 256, bias=False).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    mtq.quantize(model, copy.deepcopy(config), lambda m: m(inputs))
    weight = model.weight
    output_before = model(inputs).detach()
    expected = model.weight_quantizer(weight).detach().clone()
    reference = copy.deepcopy(model)
    with torch.no_grad():
        reference.weight.copy_(expected)
    expected_output = reference(inputs).detach()
    del reference
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    mtq.initialize_qdq_weights(model)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    assert model.weight is weight and model.weight_quantizer.is_enabled
    output_after = model(inputs)
    torch.testing.assert_close(output_after, expected_output, rtol=0, atol=0)
    relative_error = (
        output_after.float() - output_before.float()
    ).norm() / output_before.float().norm()
    record_property("initialization_forward_relative_l2", relative_error.item())
    effective_after = model.weight_quantizer(model.weight).detach()
    record_property("effective_weight_changed_elements", (effective_after != expected).sum().item())
    if getattr(model.weight_quantizer, "backend", None) == "ggml":
        torch.testing.assert_close(effective_after, expected, rtol=0, atol=0)
        torch.testing.assert_close(output_after, output_before, rtol=0, atol=0)
    for name, value in buffers.items():
        torch.testing.assert_close(dict(model.named_buffers())[name], value, rtol=0, atol=0)
    model(inputs).float().square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


def test_initialize_bf16_transformed_weights():
    torch.manual_seed(42)
    model = nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG, lambda m: m(inputs))
    model.weight_quantizer.pre_quant_scale = torch.linspace(
        0.5, 2, 256, device="cuda", dtype=torch.bfloat16
    )
    model.weight_quantizer.set_from_attribute_config({"rotate": {"enable": True}})
    before = model(inputs).detach().float()
    mtq.initialize_qdq_weights(model)
    relative_error = (model(inputs).float() - before).norm() / before.norm()
    assert relative_error < 0.03


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize(
    ("num_bits", "retain_initialization"),
    [("nvfp4", False), ("iq2_xxs", False), ("iq2_xxs", True)],
)
def test_initialize_weight_memory_and_time(
    record_property, capsys, cache_enabled, num_bits, retain_initialization
):
    model = nn.Sequential(*(nn.Linear(4096, 4096, bias=False) for _ in range(4)))
    model = model.to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
    config = (
        mtq.NVFP4_DEFAULT_CFG
        if num_bits == "nvfp4"
        else {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": {"num_bits": num_bits, "backend": "ggml"},
                    "enable": True,
                },
            ],
            "algorithm": "max",
        }
    )
    mtq.quantize(model, config, lambda m: m(inputs))
    if retain_initialization:
        for layer in model:
            layer.weight_quantizer._ggml_retain_initialization = True
    if cache_enabled:
        enable_weight_cache(model)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with weight_cache_scope(model) if cache_enabled else nullcontext():
        start.record()
        mtq.initialize_qdq_weights(model)
        end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    extra_peak_bytes = torch.cuda.max_memory_allocated() - baseline
    assert extra_peak_bytes < sum(p.numel() * p.element_size() for p in model.parameters())
    record_property("initialization_ms", elapsed_ms)
    record_property("initialization_extra_peak_bytes", extra_peak_bytes)
    packed_bytes = sum(
        buffer.numel() * buffer.element_size()
        for name, buffer in model.named_buffers()
        if name.endswith("_qdq_initialization")
    )
    record_property("retained_initialization_payload_bytes", packed_bytes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    for phase in ("unchanged", "first_change", "subsequent_change"):
        if phase != "unchanged":
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.001)
        optimizer.step()  # Invalidates GGML caches even at zero LR.
        with weight_cache_scope(model) if cache_enabled else nullcontext() as stats:
            for access in ("cold", "warm"):
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start.record()
                wall_start = time.perf_counter()
                with torch.no_grad():
                    model(inputs)
                end.record()
                torch.cuda.synchronize()
                record_property(f"{phase}_{access}_forward_ms", start.elapsed_time(end))
                record_property(
                    f"{phase}_{access}_forward_wall_ms", (time.perf_counter() - wall_start) * 1000
                )
                record_property(
                    f"{phase}_{access}_forward_extra_peak_bytes",
                    torch.cuda.max_memory_allocated() - baseline,
                )
            if cache_enabled:
                assert stats is not None
                assert stats.misses == 4 and stats.hits == 4
        record_property(
            f"{phase}_initialization_state_bytes",
            sum(
                buffer.numel() * buffer.element_size()
                for name, buffer in model.named_buffers()
                if "_qdq_initialization" in name
            ),
        )
    with capsys.disabled():
        print(
            f"QDQ initialization: device={torch.cuda.get_device_name()}, "
            f"format={num_bits}, dtype=BF16, weights=4x4096x4096, cache_enabled={cache_enabled}, "
            f"retain_initialization={retain_initialization}, "
            f"elapsed_ms={elapsed_ms:.3f}, extra_peak_allocated_bytes={extra_peak_bytes}"
        )
