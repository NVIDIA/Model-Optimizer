# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quantized_weight import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight,
    quantized_weight_export_states_compatible,
    quantized_weight_export_states_equal,
    replicate_quantized_weight_export_state,
    synchronize_quantized_weight_export_state,
)
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.nn import GroupedQuantizer


def _quantized_linear(quant_cfg, weight_amax=2.0, input_amax=3.0):
    module = nn.Linear(16, 2, bias=False)
    mtq.quantize(module, copy.deepcopy(quant_cfg), lambda m: m(torch.ones(1, 16)))
    with torch.no_grad():
        module.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(2, 16) / 8 - 2)
    module.weight_quantizer.amax = torch.tensor(weight_amax)
    if module.input_quantizer.is_enabled:
        module.input_quantizer.amax = torch.tensor(input_amax)
    return module


@pytest.mark.parametrize(
    ("quant_cfg", "has_input_scale"),
    [
        (mtq.NVFP4_DEFAULT_CFG, True),
        (mtq.W4A16_NVFP4_CFG, False),
    ],
)
def test_export_matches_offline_export_without_mutating_source(quant_cfg, has_input_scale):
    module = _quantized_linear(quant_cfg)
    source_weight = module.weight.detach().clone()
    source_parameter = module.weight
    source_buffers = {
        name: (buffer, buffer.detach().clone()) for name, buffer in module.named_buffers()
    }

    state = capture_quantized_weight_export_state(module)
    exported = export_quantized_weight(module.weight, state)
    repeated = export_quantized_weight(module.weight, state)

    assert torch.equal(exported.weight, repeated.weight)
    assert torch.equal(exported.weight_scale, repeated.weight_scale)
    assert torch.equal(exported.weight_scale_2, repeated.weight_scale_2)
    assert module.weight is source_parameter
    assert torch.equal(module.weight, source_weight)
    assert set(dict(module.named_buffers())) == set(source_buffers)
    for name, buffer in module.named_buffers():
        original, value = source_buffers[name]
        assert buffer is original
        assert torch.equal(buffer, value)

    _export_quantized_weight(module, torch.float32)

    assert torch.equal(module.weight, exported.weight)
    assert torch.equal(module.weight_scale, exported.weight_scale)
    assert torch.equal(module.weight_scale_2, exported.weight_scale_2)
    assert hasattr(module, "input_scale") is has_input_scale
    if has_input_scale:
        assert torch.equal(module.input_scale, exported.input_scale)


def test_capture_preserves_per_expert_state():
    modules = [
        _quantized_linear(mtq.NVFP4_DEFAULT_CFG, weight_amax, input_amax)
        for weight_amax, input_amax in ((2.0, 3.0), (4.0, 5.0))
    ]
    grouped = nn.Module()
    grouped.weight0 = modules[0].weight
    grouped.weight1 = modules[1].weight
    grouped.weight_quantizer = GroupedQuantizer(
        modules[0].weight_quantizer,
        modules[1].weight_quantizer,
    )

    states = [
        capture_quantized_weight_export_state(
            grouped,
            f"weight{index}",
            weight_quantizer=grouped.weight_quantizer[index],
            input_quantizer=modules[index].input_quantizer,
        )
        for index in range(2)
    ]

    assert torch.equal(states[0].weight_amax, torch.tensor(2.0))
    assert torch.equal(states[1].weight_amax, torch.tensor(4.0))
    assert torch.equal(states[0].input_amax, torch.tensor(3.0))
    assert torch.equal(states[1].input_amax, torch.tensor(5.0))


def test_state_synchronization_replication_and_equality(monkeypatch):
    state = capture_quantized_weight_export_state(_quantized_linear(mtq.NVFP4_DEFAULT_CFG))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    calls = []

    def _all_reduce(tensor, op, group):
        calls.append((op, group))
        tensor.add_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    synchronized = synchronize_quantized_weight_export_state(state, group="tp")

    assert len(calls) == 2
    assert all(op == torch.distributed.ReduceOp.MAX and group == "tp" for op, group in calls)
    assert torch.equal(state.weight_amax, torch.tensor(2.0))
    assert torch.equal(synchronized.weight_amax, torch.tensor(3.0))
    assert torch.equal(synchronized.input_amax, torch.tensor(4.0))

    replicas = replicate_quantized_weight_export_state(state, 2)
    assert quantized_weight_export_states_compatible(replicas[0], replicas[1])
    assert quantized_weight_export_states_equal(replicas[0], replicas[1])
    assert replicas[0].weight_amax.data_ptr() != replicas[1].weight_amax.data_ptr()
    replicas[0].weight_amax.add_(1)
    assert not quantized_weight_export_states_equal(replicas[0], replicas[1])


def test_hf_config_supports_mixed_nvfp4_and_exclusions():
    states = {
        "model.layers.0.mlp.up_proj": capture_quantized_weight_export_state(
            _quantized_linear(mtq.NVFP4_DEFAULT_CFG)
        ),
        "model.layers.1.mlp.up_proj": capture_quantized_weight_export_state(
            _quantized_linear(mtq.W4A16_NVFP4_CFG)
        ),
        "model.layers.0.self_attn.q_proj": None,
    }

    config = build_hf_quantization_config(states)

    assert config["quant_algo"] == "MIXED_PRECISION"
    assert len(config["config_groups"]) == 2
    assert config["ignore"] == ["model.layers.0.self_attn*"]
    assert config["quant_method"] == "modelopt"


def test_capture_rejects_unsupported_nvfp4_variant():
    module = _quantized_linear(mtq.W4A8_NVFP4_FP8_CFG)

    with pytest.raises(NotImplementedError, match="input quantizers"):
        capture_quantized_weight_export_state(module)
