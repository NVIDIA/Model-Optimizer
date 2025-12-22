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
from _test_utils.torch.distributed.utils import (
    get_device_counts,
    spawn_multiprocess_job,
    synchronize_state_dict,
)
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


@pytest.mark.parametrize("device_count", get_device_counts())
def test_fsdp_simple_linear(device_count):
    spawn_multiprocess_job(size=device_count, job=_test_fsdp2_simple_linear, backend="nccl")


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize(
    "quant_cfg", [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG]
)
def test_nested_fsdp2_backward(device_count, quant_cfg):
    spawn_multiprocess_job(
        size=device_count,
        job=partial(_test_nested_fsdp2_backward, quant_cfg=quant_cfg),
        backend="nccl",
    )
