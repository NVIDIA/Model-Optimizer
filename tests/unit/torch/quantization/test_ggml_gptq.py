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
"""GPTQ for GGML block formats: the group update, the helper, and the payload pin."""

import copy
import io

import pytest
import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gptq import gptq_group_update
from modelopt.torch.quantization.utils.calib_utils import (
    compute_hessian_inverse,
    gptq_blockwise_update,
)


def _problem(rows=6, cols=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator)
    inputs = torch.randn(64, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    hessian = inputs.T @ inputs / inputs.shape[0]
    return weight, hessian, compute_hessian_inverse(hessian, weight, 0.01)


def _round_to_tenths(weight):
    return torch.round(weight * 10) / 10


def _round_in_groups_of_4(group):
    scale = group.abs().amax(dim=-1, keepdim=True) / 2
    return torch.round(group / scale) * scale


def test_single_column_groups_match_columnwise_gptq():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    gptq_blockwise_update(expected, h_inv, 8, _round_to_tenths)

    gptq_group_update(weight, h_inv, 8, 1, _round_to_tenths)

    torch.testing.assert_close(weight, expected)


def test_group_update_folds_each_group_error_through_its_inverse_hessian_block():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    for start in range(0, expected.shape[1], 4):
        group, rest = slice(start, start + 4), slice(start + 4, None)
        qdq = _round_in_groups_of_4(expected[:, group])
        err = (expected[:, group] - qdq) @ torch.linalg.inv(h_inv[group, group])
        expected[:, group] = qdq
        expected[:, rest] -= err @ h_inv[group, rest]

    gptq_group_update(weight, h_inv, 8, 4, _round_in_groups_of_4)

    torch.testing.assert_close(weight, expected, rtol=1e-4, atol=1e-5)


def test_group_update_rejects_blocks_that_split_a_group():
    weight, _, h_inv = _problem()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        gptq_group_update(weight, h_inv, 6, 4, _round_in_groups_of_4)


IQ1_S = GGML_FORMAT_REGISTRY["iq1_s"]
IQ2_XXS = GGML_FORMAT_REGISTRY["iq2_xxs"]
GPTQ = {"method": "gptq", "block_size": 256, "perc_damp": 0.3}


def _iq_config(algorithm, num_bits="iq1_s"):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _inputs():
    generator = torch.Generator().manual_seed(0)
    return torch.randn(128, 512, generator=generator) @ torch.randn(512, 512, generator=generator)


def _gptq_model():
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False)
    inputs = _inputs()
    mtq.quantize(model, _iq_config(GPTQ), forward_loop=lambda m: m(inputs))
    return model, inputs


def _pin(model):
    return pinned_packed_weight(model.weight_quantizer, "iq1_s")


def _decoded(packed, weight):
    return IQ1_S.dequantize(packed, torch.tensor(weight.shape), dtype=torch.float32)


def test_gptq_on_a_ggml_format_pins_the_payload_it_chose():
    torch.manual_seed(0)
    original = torch.nn.Linear(512, 8, bias=False)
    plain = copy.deepcopy(original)
    inputs = _inputs()
    mtq.quantize(plain, _iq_config("max"), forward_loop=lambda m: m(inputs))
    model, _ = _gptq_model()

    pinned = _pin(model)
    assert pinned.shape == (8, 2, IQ1_S.block_bytes)
    torch.testing.assert_close(model.weight.detach(), _decoded(pinned, model.weight))
    torch.testing.assert_close(model(inputs), inputs @ _decoded(pinned, model.weight).T)
    assert torch.equal(IQ1_S.pack(model.weight, model.weight_quantizer), pinned)

    def weighted_error(weight):
        return ((weight - original.weight) @ inputs.T).square().sum()

    assert weighted_error(model.weight) < weighted_error(plain.weight_quantizer(plain.weight))


def test_pin_outlives_the_weight_tensor():
    model, _ = _gptq_model()
    # Same values in a new tensor, as an offload round trip or a layerwise resume leaves them.
    model.weight.data = model.weight.data.clone()

    assert torch.equal(IQ1_S.pack(model.weight, model.weight_quantizer), _pin(model))
    # Encoding the GPTQ'd weight again would not have returned GPTQ's codes.
    assert not torch.equal(IQ1_S.quantize(model.weight.detach())[0], _pin(model))


def test_pin_survives_save_and_restore():
    model, inputs = _gptq_model()
    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)

    restored = mto.restore(torch.nn.Linear(512, 8, bias=False), buffer)

    assert torch.equal(IQ1_S.pack(restored.weight, restored.weight_quantizer), _pin(model))
    torch.testing.assert_close(restored(inputs), model(inputs))


def test_pin_survives_accelerate_buffer_offload():
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False)
    inputs = _inputs()
    mtq.quantize(model, _iq_config("max"), forward_loop=lambda m: m(inputs))
    weights_map = {name: value.detach().clone() for name, value in model.state_dict().items()}
    hook = AlignDevicesHook(
        execution_device="cpu",
        offload=True,
        offload_buffers=True,
        place_submodules=True,
        weights_map=weights_map,
    )
    add_hook_to_module(model, hook)

    mtq.calibrate(model, algorithm=GPTQ, forward_loop=lambda m: m(inputs))

    (pin_key,) = [key for key in weights_map if key.endswith("_ggml_pinned_iq1_s")]
    decoded = IQ1_S.dequantize(weights_map[pin_key], torch.tensor((8, 512)), dtype=torch.float32)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)


def test_pin_is_dropped_once_the_weight_changes():
    model, _ = _gptq_model()
    with torch.no_grad():
        model.weight.add_(0.01)

    assert torch.equal(
        IQ1_S.pack(model.weight, model.weight_quantizer), IQ1_S.quantize(model.weight.detach())[0]
    )


@pytest.mark.parametrize("restore", [False, True])
def test_pin_is_ignored_once_the_quantizer_changes_format(restore):
    model, inputs = _gptq_model()
    if restore:
        buffer = io.BytesIO()
        mto.save(model, buffer)
        buffer.seek(0)
        model = mto.restore(torch.nn.Linear(512, 8, bias=False), buffer)

    mtq.set_quantizer_by_cfg(model, _iq_config(None, "iq2_xxs")["quant_cfg"])

    weight = model.weight.detach()
    expected = IQ2_XXS.dequantize(*IQ2_XXS.quantize(weight), dtype=torch.float32)
    torch.testing.assert_close(model(inputs), inputs @ expected.T)
    assert torch.equal(
        IQ2_XXS.pack(model.weight, model.weight_quantizer), IQ2_XXS.quantize(weight)[0]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_megatron_export_writes_the_pinned_rows_of_split_projections(dtype):
    model, _ = _gptq_model()
    weight, quantizer, pinned = model.weight.detach().to(dtype), model.weight_quantizer, _pin(model)

    def exported(rows):
        return GPTModelExporter._get_iq_weight_state("w", rows, "iq1_s", quantizer)["w"]

    heads = [0, 2]  # a QKV-style gather of whole 2-row heads
    assert torch.equal(exported(weight), pinned)
    assert torch.equal(exported(weight[:4]), pinned[:4])  # gate half of a fused gate/up
    assert torch.equal(exported(weight[4:]), pinned[4:])  # up half
    assert torch.equal(
        exported(weight.view(4, 2, 512)[heads].reshape(-1, 512)),
        pinned.view(4, 2, 2, IQ1_S.block_bytes)[heads].reshape(-1, 2, IQ1_S.block_bytes),
    )
