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


@pytest.fixture(scope="session", autouse=True)
def _prebuild_quant_cuda_extensions():
    """Build ModelOpt's quant CUDA extensions once, before any test runs.

    The first fake-quant op JIT-compiles these via torch.utils.cpp_extension.load()
    (~100s of ninja build). Without this, that compile lands inside the first test's
    timed `call` and trips the per-test timeout. pyproject sets ``timeout_func_only``,
    so doing it here in (untimed) session setup keeps it off the clock; worker
    subprocesses then load the cached .so from the shared TORCH_EXTENSIONS_DIR.
    """
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):  # best-effort; tests fall back to on-demand JIT
            from modelopt.torch.quantization import extensions as quant_ext

            quant_ext.get_cuda_ext(raise_if_failed=False)
            quant_ext.get_cuda_ext_fp8(raise_if_failed=False)


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
def _pool_cache():
    """Module-scoped cache of worker pools keyed by world_size.

    Spinning up a pool cold-imports the full torch/megatron/modelopt stack per worker
    (~tens of seconds), so fixtures that request the same world_size share one pool
    instead of spawning duplicates — e.g. on a 2-GPU runner ``dist_workers`` and
    ``dist_workers_size_2`` are both size 2. The cache is module-scoped and torn down at
    module end, so workers are never reused across modules (avoids cross-test
    state contamination).
    """
    pools: dict[int, DistributedWorkerPool] = {}
    yield pools
    for pool in pools.values():
        pool.shutdown()


def _get_pool(cache, world_size):
    if world_size not in cache:
        cache[world_size] = _make_pool(world_size)
    return cache[world_size]


@pytest.fixture(scope="module")
def dist_workers(_pool_cache):
    """Module-scoped pool with world_size=torch.cuda.device_count()."""
    return _get_pool(_pool_cache, torch.cuda.device_count())


@pytest.fixture(scope="module")
def dist_workers_size_1(_pool_cache):
    """Module-scoped pool with world_size=1 for tests that require a single process."""
    return _get_pool(_pool_cache, 1)


@pytest.fixture(scope="module")
def dist_workers_size_2(_pool_cache):
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    return _get_pool(_pool_cache, 2)


@pytest.fixture(scope="module")
def dist_workers_size_4(_pool_cache):
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs")
    return _get_pool(_pool_cache, 4)
