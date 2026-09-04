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

import copy
import pickle

import pytest
import torch
import torch.nn as nn

import modelopt.torch.export.quantized_weight_export as weight_export
import modelopt.torch.quantization as mtq
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.quantized_weight_export import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight_tensors,
    get_quantized_weight_export_spec,
    merge_quantized_weight_export_states,
    restore_quantized_weight_export_state,
    select_quantized_weight_export_state,
    split_quantized_weight_export_state,
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


class _SquareTransposedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.arange(512, dtype=torch.float32).reshape(2, 16, 16))
        self.down_proj = nn.Parameter(torch.arange(512, dtype=torch.float32).reshape(2, 16, 16))

        fp8_cfg = QuantizerAttributeConfig(num_bits=(4, 3))
        self.gate_up_proj_weight_quantizer = TensorQuantizer(fp8_cfg)
        self.gate_up_proj_weight_quantizer._amax = torch.tensor(2.0)
        self.gate_up_proj_input_quantizer = TensorQuantizer(fp8_cfg)
        self.gate_up_proj_input_quantizer._amax = torch.tensor(1.0)

        nvfp4_cfg = QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
        )
        self.down_proj_weight_quantizer = NVFP4StaticQuantizer(quant_attribute_cfg=nvfp4_cfg)
        self.down_proj_weight_quantizer.amax = torch.arange(32, dtype=torch.float32) + 1
        self.down_proj_weight_quantizer.global_amax = torch.tensor(32.0)
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer.disable()

    def iter_weights_for_calibration(self):
        for name in ("gate_up_proj", "down_proj"):
            yield getattr(self, name).transpose(-1, -2), getattr(self, f"{name}_weight_quantizer")


class _GroupedWeights(nn.Module):
    """Minimal TEGroupedLinear-style numbered-weight layout."""

    def __init__(self):
        super().__init__()
        self.weight0 = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        self.weight1 = nn.Parameter(torch.arange(16, 32, dtype=torch.float32).reshape(4, 4))
        cfg = QuantizerAttributeConfig(num_bits=(4, 3))
        self.quantizers = nn.ModuleList([TensorQuantizer(cfg), TensorQuantizer(cfg)])
        for index, quantizer in enumerate(self.quantizers, start=1):
            quantizer._amax = torch.tensor(float(index))
        self.input_quantizer = TensorQuantizer()
        self.input_quantizer.disable()

    def iter_weights_for_calibration(self):
        yield self.weight0, self.quantizers[0]
        yield self.weight1, self.quantizers[1]


def _blockwise_export_state(width: int, *, transposed: bool = False):
    return weight_export._QuantizedWeightExportState(
        quantization_format="fp8_pb_wo",
        block_size=128,
        weight_shape=(width, 128) if transposed else (128, width),
        tensors=(
            weight_export._ExportStateTensor(
                "weight_scale",
                torch.ones(1, (width + 127) // 128),
                axes=(0, 1),
                block_sizes=(128, 128),
            ),
        ),
        packing_permutation=(1, 0) if transposed else (0, 1),
    )


def test_capture_does_not_modify_zero_amax():
    module = _fp8_linear()
    module.weight_quantizer._amax.zero_()
    module.input_quantizer._amax.zero_()
    weight_amax = module.weight_quantizer._amax.clone()
    input_amax = module.input_quantizer._amax.clone()

    capture_quantized_weight_export_state(module)

    torch.testing.assert_close(module.weight_quantizer._amax, weight_amax)
    torch.testing.assert_close(module.input_quantizer._amax, input_amax)


def test_functional_fp8_export_is_repeatable():
    module = _fp8_linear()
    original_weight = module.weight
    original_buffers = {name: value.clone() for name, value in module.named_buffers()}
    state = capture_quantized_weight_export_state(module)

    base = export_quantized_weight_tensors(module.weight, state, torch.float32)
    repeated = export_quantized_weight_tensors(module.weight, state, torch.float32)
    assert module.weight is original_weight
    assert set(dict(module.named_buffers())) == set(original_buffers)
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, original_buffers[name])
    for name in base:
        torch.testing.assert_close(base[name], repeated[name])


def test_export_state_round_trips_through_object_transport():
    module = _fp8_linear()
    state = pickle.loads(pickle.dumps(capture_quantized_weight_export_state(module)))

    exported = export_quantized_weight_tensors(module.weight, state, torch.float32)

    assert exported["weight"].shape == module.weight.shape


def test_capture_resolves_numbered_grouped_weights_by_storage():
    module = _GroupedWeights()

    state0 = capture_quantized_weight_export_state(module, "weight0")
    state1 = capture_quantized_weight_export_state(module, "weight1")

    assert state0 is not None and state1 is not None
    assert state0.quantization_format == state1.quantization_format == "fp8"
    assert state0.tensors[0].value.item() != state1.tensors[0].value.item()


def test_unquantized_weight_has_no_export_state_or_spec():
    module = nn.Linear(4, 4, bias=False)

    assert capture_quantized_weight_export_state(module) is None
    assert get_quantized_weight_export_spec(module) is None


def test_export_spec_rejects_unsupported_format():
    module = nn.Linear(4, 4, bias=False)
    module.weight_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()

    with pytest.raises(NotImplementedError, match="int8_wo"):
        get_quantized_weight_export_spec(module)


def test_export_spec_rejects_weight_only_mxfp4():
    module = nn.Linear(32, 32, bias=False)
    module.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(2, 1),
            block_sizes={-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        )
    )
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()

    with pytest.raises(NotImplementedError, match="weight-only MXFP4"):
        get_quantized_weight_export_spec(module)


def test_export_spec_rejects_unsupported_fp8_2d_block_shape():
    module = nn.Linear(128, 128, bias=False)
    module.weight_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), block_sizes={-2: 64, -1: 128})
    )
    module.input_quantizer = TensorQuantizer()
    module.input_quantizer.disable()

    with pytest.raises(NotImplementedError, match="only with 128x128 blocks"):
        get_quantized_weight_export_spec(module)


def test_export_state_split_restore_preserves_output():
    module = _fp8_linear()
    state = capture_quantized_weight_export_state(module)
    assert state is not None
    metadata, tensors = split_quantized_weight_export_state(state)

    restored = restore_quantized_weight_export_state(metadata, tensors)

    expected = export_quantized_weight_tensors(module.weight, state, torch.float32)
    actual = export_quantized_weight_tensors(module.weight, restored, torch.float32)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_export_spec_builds_config_without_tensor_state():
    spec = get_quantized_weight_export_spec(_fp8_linear())
    assert spec is not None

    config = build_hf_quantization_config({"model.layers.0.proj.weight": spec})

    assert config["quant_algo"] == "FP8"


_FUNCTIONAL_FORMAT_CASES = (
    ("fp8", "FP8", 0, 8, True),
    ("fp8_pc_pt", "FP8_PER_CHANNEL_PER_TOKEN", 0, 8, True),
    ("fp8_pb_wo", "FP8_PB_WO", 128, 8, False),
    ("mxfp8", "MXFP8", 32, 8, True),
    ("mxfp4", "MXFP4", 32, 4, True),
    ("nvfp4", "NVFP4", 16, 4, True),
    ("w4a16_nvfp4", "W4A16_NVFP4", 16, 4, False),
    ("w4a8_nvfp4_fp8", "W4A8_NVFP4_FP8", 16, 4, True),
    ("w4a8_mxfp4_fp8", "W4A8_MXFP4_FP8", 32, 4, True),
)


@pytest.mark.parametrize(
    ("quantization_format", "quant_algo", "block_size", "weight_bits", "has_input"),
    _FUNCTIONAL_FORMAT_CASES,
)
def test_functional_format_builds_canonical_hf_config(
    quantization_format, quant_algo, block_size, weight_bits, has_input
):
    spec = weight_export._QuantizedWeightExportSpec(quantization_format, block_size)

    config = build_hf_quantization_config({"model.layers.0.proj.weight": spec})

    group = config["config_groups"]["group_0"]
    assert config["quant_algo"] == quant_algo
    assert group["weights"]["num_bits"] == weight_bits
    assert ("input_activations" in group) is has_input
    if quant_algo == "FP8_PB_WO":
        assert group["weights"]["strategy"] == "block"
        assert group["weights"]["block_structure"] == [128, 128]
    elif block_size:
        assert group["weights"]["group_size"] == block_size
    if quant_algo == "FP8_PER_CHANNEL_PER_TOKEN":
        assert group["input_activations"]["dynamic"] is True
        assert group["input_activations"]["strategy"] == "token"
    elif quant_algo in {"MXFP4", "MXFP8"}:
        assert group["input_activations"]["dynamic"] is True
        assert group["input_activations"]["strategy"] == "group"


def test_all_functional_formats_have_canonical_hf_config_coverage():
    assert {case[0] for case in _FUNCTIONAL_FORMAT_CASES} == set(
        weight_export._FUNCTIONAL_WEIGHT_EXPORT_FORMATS
    )


def test_export_spec_does_not_materialize_static_scale_state(monkeypatch):
    weight = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32
    module = _static_w4a16_linear(weight, weight.abs().amax(dim=1, keepdim=True), torch.tensor(1.0))
    monkeypatch.setattr(
        weight_export,
        "_state_tensor",
        lambda *args, **kwargs: pytest.fail("export spec materialized tensor state"),
    )

    spec = get_quantized_weight_export_spec(module)

    assert spec is not None
    assert spec.quantization_format == "w4a16_nvfp4"
    assert spec.block_size == 16


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


def test_blockwise_state_merge_rejects_misaligned_shard_boundaries():
    with pytest.raises(ValueError, match=r"boundary 192.*block sizes.*128"):
        merge_quantized_weight_export_states(
            (_blockwise_export_state(192), _blockwise_export_state(192)), 1
        )


def test_blockwise_state_merge_accepts_aligned_cumulative_boundaries_after_permutation():
    merged = merge_quantized_weight_export_states(
        tuple(_blockwise_export_state(width, transposed=True) for width in (128, 128, 64)),
        0,
    )

    assert merged.weight_shape == (320, 128)
    assert merged.tensors[0].value.shape == (1, 3)


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


def test_static_nvfp4_single_block_preserves_scale_shape():
    weight = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    module = _static_w4a16_linear(weight, weight.abs().amax().reshape(1, 1), torch.tensor(1.0))

    exported = export_quantized_weight_tensors(
        module.weight,
        capture_quantized_weight_export_state(module),
        torch.float32,
    )

    assert exported["weight_scale"].shape == (1, 1)


def test_static_nvfp4_selection_rejects_duplicate_block_members():
    weight = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    module = _static_w4a16_linear(
        weight,
        weight.abs().amax(dim=1, keepdim=True),
        torch.tensor(1.0),
    )

    with pytest.raises(ValueError, match="complete quantization blocks"):
        select_quantized_weight_export_state(
            capture_quantized_weight_export_state(module),
            1,
            (0,) * 16,
        )


def test_state_rejects_invalid_dimensions():
    state = capture_quantized_weight_export_state(_fp8_linear())
    with pytest.raises(IndexError, match="invalid"):
        merge_quantized_weight_export_states((state, state), 2)
    with pytest.raises(IndexError, match="invalid"):
        select_quantized_weight_export_state(state, -3, (0,))


def test_capture_uses_the_requested_weight_format_and_transposed_layout():
    module = _SquareTransposedExperts()
    gate_state = capture_quantized_weight_export_state(module, "gate_up_proj")
    down_state = capture_quantized_weight_export_state(module, "down_proj")

    config = build_hf_quantization_config(
        {
            "model.layers.0.mlp.gate_proj.weight": gate_state,
            "model.layers.0.mlp.down_proj.weight": down_state,
        }
    )
    assert config["quantized_layers"]["model.layers.0.mlp.gate_proj"]["quant_algo"] == "FP8"
    assert config["quantized_layers"]["model.layers.0.mlp.down_proj"]["quant_algo"] == "W4A16_NVFP4"

    with pytest.raises(ValueError, match="complete quantization blocks"):
        select_quantized_weight_export_state(down_state, 1, (0,))
    select_quantized_weight_export_state(down_state, 2, (0, 2))


def test_converted_hf_config_preserves_nvfp4_group_size():
    config = convert_hf_quant_config_format(
        {
            "quantization": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 32,
            }
        }
    )

    assert config["group_size"] == 32
    assert config["config_groups"]["group_0"]["weights"]["group_size"] == 32


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
