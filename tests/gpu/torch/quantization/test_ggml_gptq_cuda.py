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
"""GPTQ on a GGML format through ``mtq.quantize``, with the CUDA encoders."""

import copy

import pytest
import torch
from _test_utils.torch.quantization.models import TOY_MOE_HIDDEN, ToyMoE

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.nn import QuantModule
from modelopt.torch.quantization.utils import enable_weight_access_and_writeback


def _iq1_s_config(algorithm):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _correlated_inputs(cols=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(256, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    return inputs.cuda()


def _weighted_error(quantized, weight, inputs):
    return ((quantized - weight) @ inputs.T).square().sum()


def test_gptq_on_a_ggml_format_pins_the_payload_it_chose():
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    original = model.weight.detach().clone()
    inputs = _correlated_inputs()
    plain = copy.deepcopy(model)
    mtq.quantize(plain, _iq1_s_config("max"), forward_loop=lambda m: m(inputs))

    mtq.quantize(
        model,
        _iq1_s_config({"method": "gptq", "block_size": 256}),
        forward_loop=lambda m: m(inputs),
    )

    iq1_s = GGML_FORMAT_REGISTRY["iq1_s"]
    packed = pinned_packed_weight(model.weight_quantizer, "iq1_s")
    decoded = iq1_s.dequantize(packed, torch.tensor(model.weight.shape), dtype=torch.float32)
    torch.testing.assert_close(model.weight.detach(), decoded)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)
    assert torch.equal(iq1_s.pack(model.weight, model.weight_quantizer).flatten(), packed.flatten())
    plain_weight = plain.weight_quantizer(plain.weight).detach()
    assert _weighted_error(decoded, original, inputs) < _weighted_error(
        plain_weight, original, inputs
    )


def test_gptq_on_a_ggml_format_rejects_blocks_smaller_than_a_ggml_block():
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    inputs = _correlated_inputs()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        mtq.quantize(
            model,
            _iq1_s_config({"method": "gptq", "block_size": 128}),
            forward_loop=lambda m: m(inputs),
        )


@pytest.mark.parametrize("layout", ["linear", "fused", "cpu_offload", "disk_offload", "fsdp2"])
def test_autoquant_searches_iq_gptq_with_nvfp4_and_fp8(layout, request, tmp_path):
    if layout == "fsdp2":
        request.getfixturevalue("dist_workers").run(_autoquant_mixed_iq, layout, tmp_path)
    else:
        _autoquant_mixed_iq(0, 1, layout, tmp_path)


def _autoquant_mixed_iq(rank, size, layout, tmp_path):
    torch.manual_seed(0)
    fused_experts = layout == "fused"
    model = (
        ToyMoE()
        if fused_experts
        else torch.nn.Sequential(
            torch.nn.Linear(512, 512, bias=False), torch.nn.Linear(512, 512, bias=False)
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    weights = {name: weight.detach().clone() for name, weight in model.named_parameters()}
    inputs = torch.randn(
        8, TOY_MOE_HIDDEN if fused_experts else 512, device="cuda", dtype=torch.bfloat16
    )
    if layout == "fsdp2":
        # FSDP2 is optional for the single-GPU cases in this file.
        from torch.distributed.fsdp import fully_shard

        for layer in model:
            fully_shard(layer)
        fully_shard(model)
    elif layout in {"cpu_offload", "disk_offload"}:
        # Accelerate is only needed by the offloaded layouts.
        from accelerate import cpu_offload, disk_offload

        if layout == "cpu_offload":
            cpu_offload(model, execution_device=inputs.device)
        else:
            disk_offload(model, tmp_path / "offload", execution_device=inputs.device)
    formats = []
    for name in ("iq2_xxs", "iq2_xs", "iq2_s"):
        config = _iq1_s_config({"method": "gptq", "block_size": 256})
        config["quant_cfg"][1]["cfg"].update(
            num_bits=name, effective_bits=GGML_FORMAT_REGISTRY[name].effective_bits
        )
        formats.append((config, name))

    def forward_step(model, batch):
        for name, module in model.named_modules():
            if not isinstance(module, QuantModule):
                continue
            with enable_weight_access_and_writeback(module, model):
                for attr, weight in module.named_parameters(recurse=False):
                    key = f"{name}.{attr}" if name else attr
                    torch.testing.assert_close(weight, weights[key], rtol=0, atol=0)
                del weight
        return model(batch)

    model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 3.25},
        quantization_formats=[*formats, mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
        data_loader=[inputs],
        forward_step=forward_step,
        loss_func=lambda output, batch: output.float().square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
        verbose=False,
    )
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 3.25
    assert torch.isfinite(model(inputs)).all()
    for module in model.modules():
        if isinstance(module, QuantModule):
            with enable_weight_access_and_writeback(module, model):
                for weight, quantizer in module.iter_weights_for_calibration():
                    if quantizer.is_enabled and quantizer.backend == "ggml":
                        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
                        packed = ggml_format.pack(weight, quantizer)
                        decoded = ggml_format.dequantize(
                            packed, torch.tensor(weight.shape), dtype=weight.dtype
                        )
                        torch.testing.assert_close(weight, decoded, rtol=0, atol=0)
                del weight
