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

import pytest
import torch
from _test_utils.fs_utils import assert_unmodified_tree
from _test_utils.torch.calib_data import write_calib_jsonl
from _test_utils.torch.distributed.utils import (
    DistributedWorkerPool,
    default_worker_teardown,
    get_live_pool,
)


@pytest.fixture(scope="session")
def _dist_pools():
    pools = {}
    yield pools
    for pool in pools.values():
        pool.shutdown()


@pytest.fixture
def dist_workers(_dist_pools):
    """The session-wide pool; replaced if an earlier test hung and killed it."""
    world_size = torch.cuda.device_count()
    return get_live_pool(
        _dist_pools,
        world_size,
        lambda: DistributedWorkerPool(
            world_size=world_size, backend="nccl", teardown_fn=default_worker_teardown
        ),
    )


@pytest.fixture(scope="session")
def calib_jsonl(tmp_path_factory):
    """Local calibration texts for tests that only need some text, not a particular Hub dataset.

    Shared by the whole session: a test that writes into its directory fails at teardown.
    """
    calib_dir = tmp_path_factory.mktemp("calib")
    path = write_calib_jsonl(calib_dir / "calib.jsonl")
    with assert_unmodified_tree(calib_dir):
        yield path
