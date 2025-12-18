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
import os

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import init_process

import modelopt.torch.opt as mto


@pytest.fixture
def distributed_setup_size_1():
    init_process(rank=0, size=1, backend="nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture
def need_2_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs to run this test")


@pytest.fixture
def need_8_gpus():
    if torch.cuda.device_count() < 8:
        pytest.skip("Need at least 8 GPUs to run this test")


@pytest.fixture
def need_4_gpus():
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs to run this test")


@pytest.fixture(scope="module")
def set_torch_dtype(request):
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield
    torch.set_default_dtype(orig_dtype)


@pytest.fixture(scope="session", autouse=True)
def enable_hf_checkpointing():
    mto.enable_huggingface_checkpointing()


@pytest.fixture
def use_minitron_channel_div_4():
    """NOTE: This only works for tests that spawn processes."""
    os.environ["MINITRON_SS_CHANNEL_DIVISOR"] = "4"
    yield
    del os.environ["MINITRON_SS_CHANNEL_DIVISOR"]
