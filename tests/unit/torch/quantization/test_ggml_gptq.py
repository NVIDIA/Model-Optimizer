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
import dataclasses
import io
from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import _row_keys, pinned_packed_weight
from modelopt.torch.quantization.ggml.gptq import gptq_group_update
from modelopt.torch.quantization.nn.modules.tensor_quantizer import GroupedQuantizer
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


def _round_to_tenths(weight, columns=None):
    return torch.round(weight * 10) / 10


def _round_in_groups_of_4(group, columns=None):
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


def _gptq_model(rows=8):
    torch.manual_seed(0)
    model = torch.nn.Linear(512, rows, bias=False)
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


@pytest.mark.parametrize(
    ("num_bits", "importance_weighted", "weighted"),
    [("iq2_xxs", True, True), ("iq2_xxs", False, False), ("iq1_s", True, False)],
)
def test_gptq_weights_the_search_by_the_root_of_the_hessian_diagonal(
    monkeypatch, num_bits, importance_weighted, weighted
):
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    calls = []

    def recording_quantize(weight, **kwargs):
        calls.append(kwargs.get("importance"))
        return ggml_format.quantize(weight, **kwargs)

    monkeypatch.setitem(
        GGML_FORMAT_REGISTRY,
        num_bits,
        dataclasses.replace(ggml_format, quantize=recording_quantize),
    )
    model = torch.nn.Linear(512, 8, bias=False)
    inputs = _inputs()
    algorithm = {**GPTQ, "importance_weighted": importance_weighted}
    mtq.quantize(model, _iq_config(algorithm, num_bits), forward_loop=lambda m: m(inputs))

    gptq_calls = calls[-2:]  # one call per 256-column group
    if not weighted:
        assert gptq_calls == [None, None]
        return
    root = inputs.square().sum(dim=0).sqrt()
    importance = torch.cat(gptq_calls)
    torch.testing.assert_close(importance / importance.mean(), root / root.mean())


def test_a_module_calibration_never_reached_gets_the_plain_search():
    # An MoE expert that no calibration token reaches keeps a zero Hessian. Weighting by its
    # diagonal would make every code score zero error, so GPTQ must fall back to the plain search.
    torch.manual_seed(0)
    model = torch.nn.ModuleDict(
        {"used": torch.nn.Linear(512, 8, bias=False), "unused": torch.nn.Linear(512, 8, bias=False)}
    )
    original = model["unused"].weight.detach().clone()
    inputs = _inputs()
    algorithm = {**GPTQ, "importance_weighted": True}
    mtq.quantize(model, _iq_config(algorithm, "iq2_xxs"), forward_loop=lambda m: m["used"](inputs))

    packed, shape = IQ2_XXS.quantize(original)
    plain = IQ2_XXS.dequantize(packed, shape, dtype=original.dtype)
    torch.testing.assert_close(model["unused"].weight.detach(), plain, rtol=0, atol=0)


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


def _megatron_exporter(weight):
    """A GPTModelExporter whose quantized-state lookup hands back ``weight`` as an IQ1_S tensor."""
    exporter = object.__new__(GPTModelExporter)
    exporter.dtype = torch.bfloat16
    exporter._state_dict = {}
    exporter.exclude_modules = []
    exporter.layer_config_dict = {}
    exporter._get_quantized_state = lambda *args, **kwargs: ({"weight": weight}, "iq1_s", 256)
    return exporter


def _pinned_export(rows, dtype):
    """A GPTQ'd weight in the export dtype, its quantizer, and its pinned payload."""
    model, _ = _gptq_model(rows)
    return model.weight.detach().to(dtype), model.weight_quantizer, _pin(model)


EXPORT_DTYPES = pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])


@EXPORT_DTYPES
def test_megatron_gated_mlp_export_writes_the_pinned_rows(dtype):
    weight, quantizer, pinned = _pinned_export(8, dtype)
    module = SimpleNamespace(config=SimpleNamespace(ffn_hidden_size=4), weight_quantizer=quantizer)
    exporter = _megatron_exporter(weight)

    exporter._gated_mlp_slicing(module, "mlp.")

    assert torch.equal(exporter._state_dict["mlp.gate_proj.weight"], pinned[:4])
    assert torch.equal(exporter._state_dict["mlp.up_proj.weight"], pinned[4:])


@EXPORT_DTYPES
def test_megatron_grouped_mlp_export_writes_the_pinned_rows(dtype):
    weight, quantizer, pinned = _pinned_export(8, dtype)
    module = SimpleNamespace(
        num_gemms=1,
        weight0=weight,
        local_expert_indices=[0],
        state_dict=lambda: {"weight0": weight},
        weight_quantizer=GroupedQuantizer(quantizer),
    )
    exporter = _megatron_exporter(weight)

    exporter._grouped_mlp_slicing(
        module, "mlp.experts.{}", gate_proj_name="gate_proj", up_proj_name="up_proj"
    )

    assert torch.equal(exporter._state_dict["mlp.experts.0.gate_proj.weight"], pinned[:4])
    assert torch.equal(exporter._state_dict["mlp.experts.0.up_proj.weight"], pinned[4:])


@EXPORT_DTYPES
def test_megatron_qkv_export_writes_the_pinned_rows(dtype):
    # 4 query groups of [2 q heads, k, v], 4 rows per head: K and V are gathered from scattered
    # rows, which is where a lookup keyed on rounded projections went wrong.
    weight, quantizer, pinned = _pinned_export(64, dtype)
    config = SimpleNamespace(
        hidden_size=512,
        num_query_groups=4,
        num_attention_heads=8,
        kv_channels=4,
        attention_output_gate=False,
    )
    exporter = _megatron_exporter(weight)

    exporter._qkv_slicing(SimpleNamespace(config=config, weight_quantizer=quantizer), "attn.")

    heads = pinned.view(4, 4, 4, 2, IQ1_S.block_bytes)  # [group, q q k v, row, block, byte]
    assert torch.equal(exporter._state_dict["attn.q_proj.weight"], heads[:, :2].flatten(0, 2))
    assert torch.equal(exporter._state_dict["attn.k_proj.weight"], heads[:, 2].flatten(0, 1))
    assert torch.equal(exporter._state_dict["attn.v_proj.weight"], heads[:, 3].flatten(0, 1))


@EXPORT_DTYPES
def test_megatron_gated_delta_net_export_writes_the_pinned_rows(dtype):
    weight, quantizer, pinned = _pinned_export(12, dtype)
    module = SimpleNamespace(
        in_proj=SimpleNamespace(weight_quantizer=quantizer),
        in_proj_split_names=("query", "key", "value", "z", "beta", "alpha"),
        in_proj_split_sections=(2, 2, 2, 2, 2, 2),
    )
    exporter = _megatron_exporter(weight)

    exporter._gated_delta_net_slicing(module, "mixer.")

    assert torch.equal(exporter._state_dict["mixer.in_proj_qkv.weight"], pinned[:6])
    assert torch.equal(exporter._state_dict["mixer.in_proj_z.weight"], pinned[6:8])


@EXPORT_DTYPES
def test_row_keys_do_not_depend_on_the_batch_a_row_is_hashed_in(dtype):
    # Wide-range values make any floating-point key sensitive to how its reduction is ordered;
    # the key must still be identical for a row hashed in a full table, a gathered subset, or alone.
    generator = torch.Generator().manual_seed(0)
    table = (
        torch.randn(64, 512, generator=generator)
        * torch.randn(64, 512, generator=generator).mul(8).exp()
    ).to(dtype)
    index = torch.tensor([56, 3, 52, 17])

    keys = _row_keys(table)

    assert torch.equal(_row_keys(table[index]), keys[index])
    assert all(torch.equal(_row_keys(table[i : i + 1]), keys[i : i + 1]) for i in index.tolist())
    assert len(keys.unique()) == len(table)
