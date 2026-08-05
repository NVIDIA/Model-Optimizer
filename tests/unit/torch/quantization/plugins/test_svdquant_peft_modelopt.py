# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the ModelOpt checkpoint contract of PEFT-backed SVDQuant."""

import copy
import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

peft = pytest.importorskip("peft", minversion="0.17.0")

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.svdquant_peft import (
    _SVDQUANT_ADAPTER_NAME,
    _SVDQuantPeftLinear,
)


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.svdquant = nn.Linear(8, 8, bias=False)
        self.skipped = nn.Linear(8, 8, bias=False)
        self.disabled = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = F.silu(self.svdquant(x))
        x = F.silu(self.skipped(x))
        return self.disabled(x)


def _svdquant_config(*, select_one_target=False):
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    config["algorithm"] = {
        "method": "svdquant",
        "lowrank": 4,
        "skip_layers": ["skipped"] if select_one_target else None,
    }
    if select_one_target:
        # A disabled weight quantizer must not cause PEFT injection either.
        config["quant_cfg"].append({"quantizer_name": "disabled.weight_quantizer", "enable": False})
    return config


def _quantize(model, *, select_one_target=False):
    reference = next(model.parameters())
    calibration_input = torch.randn(4, 8, device=reference.device, dtype=reference.dtype)
    return mtq.quantize(
        model,
        _svdquant_config(select_one_target=select_one_target),
        forward_loop=lambda current: current(calibration_input),
    )


def _factor_state(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def _svdquant_metadata(model):
    matches = [
        mode_state["metadata"]["svdquant_peft"]
        for mode_name, mode_state in mto.modelopt_state(model)["modelopt_state_dict"]
        if mode_name == "svdquant_calibrate"
    ]
    assert len(matches) == 1
    return matches[0]


def test_svdquant_peft_uses_exact_targets_and_preserves_unquantized_forward():
    torch.manual_seed(17)
    original = _TinyMLP()
    model = copy.deepcopy(original)
    model = _quantize(model, select_one_target=True)

    assert isinstance(model.svdquant, _SVDQuantPeftLinear)
    assert not isinstance(model.skipped, _SVDQuantPeftLinear)
    assert not isinstance(model.disabled, _SVDQuantPeftLinear)
    assert _svdquant_metadata(model)["target_modules"] == ["svdquant"]

    adapter = model.svdquant
    base_layer = adapter.get_base_layer()
    assert base_layer.weight_quantizer.svdquant_lora_a is None
    assert base_layer.weight_quantizer.svdquant_lora_b is None
    assert adapter.scaling[_SVDQUANT_ADAPTER_NAME] == 1.0
    assert list(adapter.active_adapters) == [_SVDQUANT_ADAPTER_NAME]
    assert adapter.lora_A[_SVDQUANT_ADAPTER_NAME].weight.requires_grad
    assert adapter.lora_B[_SVDQUANT_ADAPTER_NAME].weight.requires_grad
    assert not model.disabled.weight_quantizer.is_enabled

    # Both branches see the same AWQ-scaled input, while output quantization applies only
    # to the residual-weight branch, matching the historical SVDQuant contract.
    probe = torch.randn(3, 8)
    scaled_probe = base_layer._apply_pre_quant_scale(probe)
    with base_layer.input_quantizer.disable_pre_quant_scale():
        raw_base_output = base_layer(scaled_probe)
    lora_output = F.linear(
        F.linear(scaled_probe, adapter.lora_A[_SVDQUANT_ADAPTER_NAME].weight),
        adapter.lora_B[_SVDQUANT_ADAPTER_NAME].weight,
    )
    base_layer.output_quantizer.amax = raw_base_output.detach().abs().amax().mul(0.5)
    base_layer.output_quantizer.enable()
    expected = base_layer.output_quantizer(raw_base_output) + lora_output
    combined_quantized = base_layer.output_quantizer(raw_base_output + lora_output)
    actual = adapter(probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, combined_quantized)

    # With Q/DQ disabled, W_residual + BA must reproduce the original BF16/FP32 model.
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.disable()
    torch.testing.assert_close(model(probe), original(probe), rtol=2e-5, atol=2e-5)

    bf16_model = _quantize(_TinyMLP().to(torch.bfloat16), select_one_target=True)
    bf16_adapter = bf16_model.svdquant
    assert bf16_adapter.lora_A[_SVDQUANT_ADAPTER_NAME].weight.dtype == torch.bfloat16
    assert bf16_adapter.lora_B[_SVDQUANT_ADAPTER_NAME].weight.dtype == torch.bfloat16


def test_mto_save_restore_preserves_qat_mutated_factors():
    torch.manual_seed(19)
    model = _quantize(_TinyMLP())
    initial_factors = _factor_state(model)
    assert initial_factors

    factor_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    assert factor_parameters and all(parameter.requires_grad for parameter in factor_parameters)
    optimizer = torch.optim.SGD(factor_parameters, lr=5e-2)
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(5, 8)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in factor_parameters)
    optimizer.step()

    trained_factors = _factor_state(model)
    assert any(
        not torch.equal(initial_factors[name], trained_factors[name])
        for name in trained_factors
        if ".lora_A." in name
    )
    assert any(
        not torch.equal(initial_factors[name], trained_factors[name])
        for name in trained_factors
        if ".lora_B." in name
    )

    probe = torch.randn(2, 8)
    expected = model(probe).detach().clone()
    checkpoint = io.BytesIO()
    mto.save(model, checkpoint)
    checkpoint.seek(0)
    restored = mto.restore(_TinyMLP(), checkpoint)

    restored_factors = _factor_state(restored)
    assert trained_factors.keys() == restored_factors.keys()
    for name in trained_factors:
        assert torch.equal(trained_factors[name], restored_factors[name]), name
    assert all(
        parameter.requires_grad
        for name, parameter in restored.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )
    assert restored.peft_config[_SVDQUANT_ADAPTER_NAME].inference_mode is False
    for module in restored.modules():
        if isinstance(module, _SVDQuantPeftLinear):
            assert list(module.active_adapters) == [_SVDQUANT_ADAPTER_NAME]
    torch.testing.assert_close(restored(probe), expected, rtol=0, atol=0)

    # Optimizers are application-owned, but restored PEFT parameters must be discoverable
    # before a resume optimizer is constructed.
    resume_optimizer = torch.optim.SGD(
        (parameter for parameter in restored.parameters() if parameter.requires_grad), lr=1e-3
    )
    optimizer_ids = {
        id(parameter) for group in resume_optimizer.param_groups for parameter in group["params"]
    }
    assert all(
        id(parameter) in optimizer_ids
        for name, parameter in restored.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )

    # The split ModelOpt API follows the same topology-then-state lifecycle.
    modelopt_state = copy.deepcopy(mto.modelopt_state(model))
    model_state = copy.deepcopy(model.state_dict())
    manually_restored = mto.restore_from_modelopt_state(_TinyMLP(), modelopt_state)
    manually_restored.load_state_dict(model_state)
    for name, tensor in trained_factors.items():
        assert torch.equal(_factor_state(manually_restored)[name], tensor), name
    assert all(
        parameter.requires_grad
        for name, parameter in manually_restored.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )
    torch.testing.assert_close(manually_restored(probe), expected, rtol=0, atol=0)
