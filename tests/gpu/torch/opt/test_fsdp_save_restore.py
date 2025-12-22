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
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def _test_fsdp_save_restore(mode, rank, size):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
    ).cuda()
    inputs = torch.randn(2, 2, 128).cuda()
    synchronize_state_dict(model)

    if mode == "quantize":
        model = mtq.quantize(model, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, lambda model: model(inputs))
    elif mode == "autonas":
        model = mtn.convert(model, mode=mode)
        mtn.sample(model)

    fsdp_model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy((nn.Sequential,)))
    fsdp_model.eval()
    out_ref = fsdp_model(inputs)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        state_dict = model.state_dict()

    modelopt_state = mto.modelopt_state(fsdp_model)

    model_test = nn.Sequential(
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
        nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)),
    ).cuda()
    mto.restore_from_modelopt_state(model_test, modelopt_state)
    model_test.load_state_dict(state_dict)

    assert modelopt_state == mto.modelopt_state(model_test)

    model_test.eval()
    out_test = model_test(inputs)

    assert torch.allclose(out_ref, out_test)


@pytest.mark.parametrize("mode", ["quantize", "autonas", "fastnas"])
@pytest.mark.parametrize("device_count", get_device_counts())
def test_fsdp_save_restore(mode, device_count):
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_fsdp_save_restore, mode), backend="nccl"
    )
