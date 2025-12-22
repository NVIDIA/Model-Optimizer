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
from _test_utils.torch.distributed.fsdp_test import run_fsdp_test
from _test_utils.torch.distributed.utils import get_device_counts, spawn_multiprocess_job

from modelopt.torch.nas.search_space import SearchSpace
from modelopt.torch.opt.conversion import apply_mode


def _get_test_case():
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
    model = apply_mode(model, "sparse_magnitude")
    model = model.cuda()
    dummy_input = torch.rand(1, 1, 32, device="cuda")
    return model, dummy_input


def _sample_subnet(model):
    model[0].set_mask(torch.rand_like(model[0].weight) > 0.5)
    SearchSpace(model).sample(sample_func=min)


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp(device_count, use_orig_params):
    # run test
    spawn_multiprocess_job(
        size=device_count,
        job=partial(
            run_fsdp_test,
            _get_test_case,
            "0",
            _sample_subnet,
            fsdp_kwargs={"use_orig_params": use_orig_params},
        ),
        backend="nccl",
    )
