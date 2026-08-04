# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test of quantization with FSDP2."""

import copy
from functools import partial

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import synchronize_state_dict
from torch.distributed._composable.fsdp.fully_shard import fully_shard

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.dynamic import _pytorch_managed


def _test_fsdp2_simple_linear(rank, size):
    dim = 32
    model = nn.Linear(dim, dim).cuda(rank)
    inputs = torch.randn(2, 2, dim).cuda(rank)

    synchronize_state_dict(model)
    fsdp_model_after = copy.deepcopy(model)
    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    manager = model._get_dm_attribute_manager()
    assert "weight" in manager.da_keys()
    assert model._get_dm_attribute_manager().get_da_value("weight") is _pytorch_managed

    out_ref = model(inputs)

    fsdp_model = fully_shard(model)
    assert "weight" in manager.da_keys()
    out_test = fsdp_model(inputs)

    assert torch.allclose(out_ref, out_test)

    # quantize after fsdp2
    fsdp_model_after = fully_shard(fsdp_model_after)
    fsdp_model_after = mtq.quantize(
        fsdp_model_after, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs)
    )
    out_fsdp_model_after = fsdp_model_after(inputs)
    assert torch.allclose(out_ref, out_fsdp_model_after)


def _test_nested_fsdp2_backward(rank, size, quant_cfg):
    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Linear(dim, dim),
    ).cuda()
    inputs = torch.randn(2, 2, dim).cuda()
    inputss = inputs.detach().clone()
    synchronize_state_dict(model)
    # test for quantization after fsdp2
    fsdp_model_quant_after = copy.deepcopy(model)

    def forward_loop(model):
        model(inputs)

    forward_loop = forward_loop if quant_cfg != mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG else None

    model = mtq.quantize(model, quant_cfg, forward_loop)
    fsdp_model = copy.deepcopy(model)

    optimizer_ref = torch.optim.SGD(model.parameters(), lr=0.1)
    out_ref = model(inputs)
    out_ref.sum().backward()

    fully_shard(fsdp_model[0])
    fully_shard(fsdp_model[1])
    fsdp_model = fully_shard(fsdp_model)

    optimizer_test = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    out_test = fsdp_model(inputs)
    out_test.sum().backward()

    fully_shard(fsdp_model_quant_after[0])
    fully_shard(fsdp_model_quant_after[1])
    fsdp_model_quant_after = fully_shard(fsdp_model_quant_after)
    fsdp_model_quant_after = mtq.quantize(fsdp_model_quant_after, quant_cfg, forward_loop)
    optimizer_quant_after = torch.optim.SGD(fsdp_model_quant_after.parameters(), lr=0.1)
    out_quant_after = fsdp_model_quant_after(inputs)
    out_quant_after.sum().backward()

    assert torch.allclose(out_ref, out_test)
    assert torch.allclose(out_ref, out_quant_after)

    optimizer_ref.step()
    optimizer_ref.zero_grad()

    optimizer_test.step()
    optimizer_test.zero_grad()

    optimizer_quant_after.step()
    optimizer_quant_after.zero_grad()

    out_ref_1 = model(inputss)
    out_test_1 = fsdp_model(inputss)
    out_quant_after_1 = fsdp_model_quant_after(inputss)
    assert torch.allclose(out_ref_1, out_test_1, rtol=1e-4)
    assert torch.allclose(out_ref_1, out_quant_after_1, rtol=1e-4)


def test_fsdp_simple_linear(dist_workers):
    dist_workers.run(_test_fsdp2_simple_linear)


@pytest.mark.parametrize(
    "quant_cfg", [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG]
)
def test_nested_fsdp2_backward(quant_cfg, dist_workers):
    dist_workers.run(partial(_test_nested_fsdp2_backward, quant_cfg=quant_cfg))


class _DecoderBlock(nn.Module):
    """Minimal decoder block for FSDP2 sequential tests."""

    def __init__(self, dim=32):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.ReLU(), nn.Linear(dim, dim, bias=False)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(x)
        return x


class _SimpleTransformerModel(nn.Module):
    """Model with ``model.layers`` for layerwise calibration discovery."""

    def __init__(self, n_layers=3, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _test_layerwise_calibrate_fsdp2(rank, size):
    """Layerwise calibration on FSDP2-wrapped model matches non-FSDP reference."""
    from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

    dim = 32
    torch.manual_seed(1)
    model = _SimpleTransformerModel(n_layers=3, dim=dim).cuda()
    inputs = torch.randn(2, 2, dim).cuda()
    synchronize_state_dict(model)

    # Register discoverer for our simple model
    old_support = LayerActivationCollector._decoder_layer_support[:]
    LayerActivationCollector._decoder_layer_support = [
        (
            lambda m: hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList),
            lambda m: m.layers,
        ),
        *old_support,
    ]

    try:
        # Reference: non-FSDP layerwise calibration
        ref_model = copy.deepcopy(model)
        seq_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        seq_cfg["algorithm"] = {"method": "max", "layerwise": True}
        mtq.quantize(ref_model, seq_cfg, lambda m: m(inputs))
        output_ref = ref_model(inputs)

        # Test: FSDP2-wrapped layerwise calibration
        for layer in model.layers:
            fully_shard(layer)
        model = fully_shard(model)
        mtq.quantize(model, seq_cfg, lambda m: m(inputs))
        output_test = model(inputs)

        assert torch.allclose(output_ref, output_test)
    finally:
        LayerActivationCollector._decoder_layer_support = old_support


def test_layerwise_calibrate_fsdp2(dist_workers):
    dist_workers.run(_test_layerwise_calibrate_fsdp2)


def _test_persistent_materialization(rank, size):
    """persistent_materialization keeps weights accessible and writes back modifications."""
    from torch.distributed.tensor import DTensor

    from modelopt.torch.quantization.utils import (
        enable_weight_access_and_writeback,
        persistent_materialization,
    )

    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
    ).cuda(rank)
    synchronize_state_dict(model)

    fully_shard(model[0])
    fully_shard(model[1])
    model = fully_shard(model)

    layer = model[0]
    inputs = torch.randn(2, dim).cuda(rank)

    # Warmup forward to trigger FSDP2's lazy_init (mirrors real usage where
    # layerwise_calibrate always runs get_first_layer_inputs first).
    model(inputs)

    # Save reference weight (gathered)
    with enable_weight_access_and_writeback(layer[0], model):
        ref_weight = layer[0].weight.clone()

    # Verify sharded before context
    assert isinstance(next(iter(layer.parameters())), DTensor)

    with persistent_materialization(layer):
        # Params are local tensors (not DTensors)
        assert not isinstance(layer[0].weight, DTensor)
        assert layer[0].weight.device.type == "cuda"

        # Run multiple forward passes (FSDP hooks fire, unshard/reshard are no-ops)
        for _ in range(3):
            layer(inputs)

        # Modify a weight
        layer[0].weight.data.add_(1.0)

    # After context: params restored to DTensors (sharded)
    assert isinstance(next(iter(layer.parameters())), DTensor)

    # Verify modification persisted
    with enable_weight_access_and_writeback(layer[0], model):
        assert torch.allclose(layer[0].weight, ref_weight + 1.0)


def test_persistent_materialization(dist_workers):
    dist_workers.run(_test_persistent_materialization)
