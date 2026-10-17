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
import gc
import queue

import pytest
import torch
from _test_utils.fs_utils import assert_unmodified_tree
from _test_utils.torch.distributed.utils import (
    DistributedWorkerPool,
    get_live_pool,
    reset_worker_state,
)
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from megatron.core.parallel_state import destroy_model_parallel

import modelopt.torch.quantization.extensions as ext
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.plugins.megatron_preprocess_data import _prestart_forkserver


@pytest.fixture(scope="session")
def tiny_tokenizer_path(tmp_path_factory):
    tokenizer_path = tmp_path_factory.mktemp("tiny_tokenizer")
    get_tiny_tokenizer().save_pretrained(tokenizer_path)
    with assert_unmodified_tree(tokenizer_path) as path:
        yield str(path)


apex_destroy = None
with contextlib.suppress(ImportError):
    from apex.transformer.parallel_state import destroy_model_parallel as apex_destroy


@pytest.fixture(scope="session", autouse=True)
def _prebuild_quant_cuda_extensions():
    """Prebuild quant CUDA extensions before per-test timeouts start.

    First-use JIT compilation can take minutes in CI, so build the base, FP8, and MX
    extensions during session setup and let tests fall back to on-demand JIT if needed.

    Doing it here in session setup (``pyproject`` sets ``timeout_func_only``) keeps the
    build off the per-test clock, runs regardless of test selection/ordering (e.g. ``-k``
    filters) and is not itself capped by a per-test timeout. Worker subprocesses then load
    the cached .so from the shared ``TORCH_EXTENSIONS_DIR``. (The extension tests themselves
    live in ``tests/gpu/_extensions``.)
    """
    ext.precompile()


@pytest.fixture(scope="session", autouse=True)
def _prestart_preprocess_forkserver():
    """Start the forkserver that ``megatron_preprocess_data`` workers fork from, at session start.

    Its first start cold-imports torch/megatron, which would otherwise land inside the first
    multi-worker preprocess test and its per-test timeout. Starting it here, without waiting,
    overlaps that import with the earlier tests. The server's environment is fixed at start.
    """
    _prestart_forkserver()


def megatron_worker_teardown(rank, world_size):
    """Clean up model-parallel state between tests in persistent workers."""
    # Surface asynchronous CUDA errors at the test that caused them, not at a later one
    torch.cuda.synchronize()
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
    # destroy_model_parallel() only drops Megatron's references to its NCCL groups, so every test's
    # communicators (and their GPU buffers) piled up in the persistent workers (~20 groups and
    # ~0.4 GiB per test), slowing later tests until they timed out. Destroy all but the default.
    for group in list(torch.distributed.distributed_c10d._world.pg_map):
        if group is not torch.distributed.GroupMember.WORLD:
            torch.distributed.destroy_process_group(group)
    # The finished test's modules can still reference the destroyed groups. Collect them here, with
    # no kernels in flight, rather than at an arbitrary point of the next job.
    gc.collect()
    torch.cuda.empty_cache()


def _make_pool(world_size):
    return DistributedWorkerPool(
        world_size=world_size,
        backend="nccl",
        teardown_fn=megatron_worker_teardown,
    )


@pytest.fixture(scope="session")
def _pool_cache():
    """Session-scoped cache of worker pools keyed by world_size.

    Spinning up a pool cold-imports the full torch/megatron/modelopt stack per worker
    (75-100 s in CI), so every module that requests the same world_size shares one pool,
    e.g. on a 2-GPU runner ``dist_workers`` and ``dist_workers_size_2`` are both size 2.
    Isolation between tests comes from ``megatron_worker_teardown`` (model-parallel state and
    process groups after every job), from ``_reset_shared_pools`` (process-global torch settings
    at each module boundary) and from ``get_live_pool`` replacing a pool that a failed, hung,
    timed-out or crashed job retired (a sticky CUDA error would otherwise fail every later
    test). The ``dist_workers*`` fixtures are deliberately function-scoped so
    ``get_live_pool`` can swap in a new pool after one was killed; they therefore cannot be
    requested from a module- or session-scoped fixture.
    """
    pools: dict[int, DistributedWorkerPool] = {}
    yield pools
    for pool in pools.values():
        pool.shutdown()


@pytest.fixture(scope="module", autouse=True)
def _reset_shared_pools(_pool_cache):
    """Workers outlive test modules: undo what the previous module did to process-global torch state."""
    for pool in _pool_cache.values():
        if not pool.broken:
            # A pool whose reset fails is retired by run(); the next test that needs one gets a new
            # pool, and the module's tests, which may not use a pool at all, are not errored.
            with contextlib.suppress(RuntimeError, queue.Empty):
                pool.run(reset_worker_state)


def _get_pool(cache, world_size):
    # A hung test kills its pool; a fresh one keeps a single hang from timing out every later
    # test (that cascade cost the nightly 20+ min).
    return get_live_pool(cache, world_size, lambda: _make_pool(world_size))


@pytest.fixture
def dist_workers(_pool_cache):
    """Shared pool with world_size=torch.cuda.device_count()."""
    return _get_pool(_pool_cache, torch.cuda.device_count())


@pytest.fixture
def dist_workers_size_1(_pool_cache):
    """Shared pool with world_size=1 for tests that require a single process."""
    return _get_pool(_pool_cache, 1)


@pytest.fixture
def dist_workers_size_2(_pool_cache):
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs")
    return _get_pool(_pool_cache, 2)


@pytest.fixture
def dist_workers_size_4(_pool_cache):
    if torch.cuda.device_count() < 4:
        pytest.skip("Need at least 4 GPUs")
    return _get_pool(_pool_cache, 4)
