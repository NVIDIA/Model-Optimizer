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
import torch.distributed
from _test_utils.torch.distributed.fsdp_test import run_fsdp2_test, run_fsdp_test
from _test_utils.torch.distributed.utils import (
    get_device_counts,
    spawn_multiprocess_job,
    synchronize_state_dict,
)
from torch import nn
from torchvision.models.resnet import Bottleneck

from modelopt.torch.nas.search_space import SearchSpace, generate_search_space


def _get_test_case():
    model = Bottleneck(32, 8)
    model = generate_search_space(model).model
    model = model.cuda()

    if torch.distributed.is_initialized():
        synchronize_state_dict(model)

    dummy_input = torch.randn(1, 32, 16, 16).cuda()
    return model, dummy_input


def test_replicate():
    model, dummy_input = _get_test_case()

    replica = nn.parallel.replicate(model, ["cuda:0"], not torch.is_grad_enabled())[0]
    assert replica.forward != model.forward
    assert replica.forward.__self__ == replica

    assert torch.allclose(model(dummy_input), replica(dummy_input))


def test_replicate_nested():
    model, dummy_input = _get_test_case()
    model = nn.Sequential(model)

    replica = nn.parallel.replicate(model, ["cuda:0"], not torch.is_grad_enabled())[0]
    assert replica[0].forward != model[0].forward
    assert replica[0].forward.__self__ == replica[0]

    assert torch.allclose(model(dummy_input), replica(dummy_input))


def _sample_subnet(model):
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
            "conv1",
            _sample_subnet,
            fsdp_kwargs={"use_orig_params": use_orig_params},
        ),
        backend="nccl",
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_fsdp2(device_count):
    # run test
    spawn_multiprocess_job(
        size=device_count,
        job=partial(run_fsdp2_test, _get_test_case, "conv1", _sample_subnet),
        backend="nccl",
    )
