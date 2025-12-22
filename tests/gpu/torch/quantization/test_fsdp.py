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

from functools import partial

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import (
    get_device_counts,
    spawn_multiprocess_job,
    synchronize_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.dynamic import _pytorch_managed


def _test_fsdp_simple_linear(rank, size):
    model = nn.Linear(128, 128).cuda(rank)
    inputs = torch.randn(2, 2, 128).cuda(rank)

    synchronize_state_dict(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    manager = model._get_dm_attribute_manager()
    assert "weight" in manager.da_keys()
    assert model._get_dm_attribute_manager().get_da_value("weight") is _pytorch_managed

    out_ref = model(inputs)

    fsdp_model = FSDP(model)

    assert "weight" in manager.da_keys()
    assert isinstance(fsdp_model._get_dm_attribute_manager().get_da_value("weight"), torch.Tensor)

    # Lets make sure backward works fine
    fsdp_model(inputs).sum().backward()

    # Lets compare outputs before and after wrapping in FSDP
    out_test = fsdp_model(inputs)

    assert torch.allclose(out_ref, out_test)


def _test_nested_fsdp(use_orig_params, rank, size):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
    ).cuda()
    inputs = torch.randn(2, 2, 128).cuda()
    synchronize_state_dict(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))
    out_ref = model(inputs)

    fsdp_model = FSDP(
        model, auto_wrap_policy=ModuleWrapPolicy((nn.Sequential,)), use_orig_params=use_orig_params
    )

    fsdp_model(inputs).sum().backward()
    out_test = fsdp_model(inputs)

    assert torch.allclose(out_ref, out_test)


@pytest.mark.parametrize("device_count", get_device_counts())
def test_fsdp_simple_linear(device_count):
    spawn_multiprocess_job(size=device_count, job=_test_fsdp_simple_linear, backend="nccl")


@pytest.mark.parametrize("use_orig_params", [True, False])
@pytest.mark.parametrize("device_count", get_device_counts())
def test_nested_fsdp(use_orig_params, device_count):
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_nested_fsdp, use_orig_params), backend="nccl"
    )
