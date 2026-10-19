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
import contextlib

import pytest
import torch
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from megatron.core.parallel_state import destroy_model_parallel

import modelopt.torch.utils.distributed as dist

apex_destroy = None
with contextlib.suppress(ImportError):
    from apex.transformer.parallel_state import destroy_model_parallel as apex_destroy


def megatron_worker_teardown(rank, world_size):
    """Clean up model-parallel state between tests in persistent workers."""
    if dist.is_initialized():
        dist.barrier()
    try:
        destroy_model_parallel()
    except Exception as e:
        print(f"Error destroying model parallel: {e}")
    if apex_destroy is not None:
        try:
            apex_destroy()
        except Exception as e:
            print(f"Error destroying model parallel with Apex: {e}")
    torch.cuda.empty_cache()


def _make_pool(world_size):
    return DistributedWorkerPool(
        world_size=world_size,
        backend="nccl",
        teardown_fn=megatron_worker_teardown,
    )


@pytest.fixture(scope="module")
def dist_workers():
    """Module-scoped pool with world_size=torch.cuda.device_count()."""
    pool = _make_pool(torch.cuda.device_count())
    yield pool
    pool.shutdown()


@pytest.fixture(scope="module")
def dist_workers_size_1():
    """Module-scoped pool with world_size=1 for tests that require a single process."""
    pool = _make_pool(1)
    yield pool
    pool.shutdown()


@pytest.fixture(scope="module")
def dist_workers_size_2():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    pool = _make_pool(2)
    yield pool
    pool.shutdown()


@pytest.fixture(scope="module")
def dist_workers_size_4():
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs")
    pool = _make_pool(4)
    yield pool
    pool.shutdown()
