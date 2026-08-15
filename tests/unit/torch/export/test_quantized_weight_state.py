# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight_tensors,
    merge_quantized_weight_export_states,
    permute_quantized_weight_export_state,
    select_quantized_weight_export_state,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, TensorQuantizer


def _fp8_linear() -> nn.Linear:
    module = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8 - 1)
    return mtq.quantize(module, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m(torch.ones(2, 4)))


def _static_w4a16_linear(
    weight: torch.Tensor,
    per_block_amax: torch.Tensor,
    global_amax: torch.Tensor,
) -> nn.Linear:
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    module.weight.data.copy_(weight)
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
    )
    quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=cfg)
    quantizer.amax = per_block_amax.clone()
    quantizer.global_amax = global_amax.clone()
    module.weight_quantizer = quantizer
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()
    return module


def test_capture_does_not_modify_zero_amax():
    module = _fp8_linear()
    module.weight_quantizer._amax.zero_()
    module.input_quantizer._amax.zero_()
    weight_amax = module.weight_quantizer._amax.clone()
    input_amax = module.input_quantizer._amax.clone()

    capture_quantized_weight_export_state(module)

    torch.testing.assert_close(module.weight_quantizer._amax, weight_amax)
    torch.testing.assert_close(module.input_quantizer._amax, input_amax)


def test_functional_fp8_export_is_repeatable_and_supports_permutation():
    module = _fp8_linear()
    original_weight = module.weight
    original_buffers = {name: value.clone() for name, value in module.named_buffers()}
    state = capture_quantized_weight_export_state(module)

    base = export_quantized_weight_tensors(module.weight, state, torch.float32)
    repeated = export_quantized_weight_tensors(module.weight, state, torch.float32)
    transposed = export_quantized_weight_tensors(
        module.weight.T,
        permute_quantized_weight_export_state(state, (1, 0)),
        torch.float32,
    )

    assert module.weight is original_weight
    assert set(dict(module.named_buffers())) == set(original_buffers)
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, original_buffers[name])
    for name in base:
        torch.testing.assert_close(base[name], repeated[name])
    torch.testing.assert_close(transposed["weight"], base["weight"].T)


def test_static_nvfp4_merge_recomputes_scales_from_merged_amax():
    left_weight = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32
    right_weight = torch.arange(32, 64, dtype=torch.float32).reshape(2, 16) / 16
    left_amax = left_weight.abs().amax(dim=1, keepdim=True)
    right_amax = right_weight.abs().amax(dim=1, keepdim=True)
    left = _static_w4a16_linear(left_weight, left_amax, torch.tensor(1.0))
    right = _static_w4a16_linear(right_weight, right_amax, torch.tensor(4.0))

    state = merge_quantized_weight_export_states(
        (
            capture_quantized_weight_export_state(left),
            capture_quantized_weight_export_state(right),
        ),
        0,
    )
    weight = torch.cat((left_weight, right_weight), dim=0)
    actual = export_quantized_weight_tensors(weight, state, torch.float32)

    reference = _static_w4a16_linear(
        weight,
        torch.cat((left_amax, right_amax), dim=0),
        torch.tensor(4.0),
    )
    expected = export_quantized_weight_tensors(
        reference.weight,
        capture_quantized_weight_export_state(reference),
        torch.float32,
    )
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_static_nvfp4_noncontiguous_selection_tracks_block_amax():
    weight = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 32
    per_block_amax = weight.abs().amax(dim=1, keepdim=True)
    module = _static_w4a16_linear(weight, per_block_amax, torch.tensor(2.0))
    indices = torch.tensor([0, 2])

    state = select_quantized_weight_export_state(
        capture_quantized_weight_export_state(module), 0, indices
    )
    actual = export_quantized_weight_tensors(weight.index_select(0, indices), state, torch.float32)

    reference = _static_w4a16_linear(
        weight.index_select(0, indices),
        per_block_amax.index_select(0, indices),
        torch.tensor(2.0),
    )
    expected = export_quantized_weight_tensors(
        reference.weight,
        capture_quantized_weight_export_state(reference),
        torch.float32,
    )
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_state_rejects_invalid_dimensions():
    state = capture_quantized_weight_export_state(_fp8_linear())
    with pytest.raises(IndexError, match="invalid"):
        merge_quantized_weight_export_states((state, state), 2)
    with pytest.raises(IndexError, match="invalid"):
        select_quantized_weight_export_state(state, -3, (0,))


def test_mixed_config_groups_moe_experts_by_projection_family():
    fp8 = capture_quantized_weight_export_state(_fp8_linear())
    weight = torch.ones(4, 16)
    w4a16 = capture_quantized_weight_export_state(
        _static_w4a16_linear(weight, torch.ones(4, 1), torch.tensor(1.0))
    )
    config = build_hf_quantization_config(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": fp8,
            "model.layers.0.mlp.experts.1.gate_proj.weight": fp8,
            "model.layers.0.mlp.experts.0.down_proj.weight": w4a16,
            "model.layers.0.mlp.experts.1.down_proj.weight": w4a16,
            "lm_head.weight": None,
        }
    )

    assert set(config["quantized_layers"]) == {
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    assert "lm_head" in config["ignore"]
