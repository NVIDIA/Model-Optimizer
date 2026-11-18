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
from _test_utils.torch.distributed.utils import DistributedWorkerPool, default_worker_teardown


@pytest.fixture(scope="session")
def dist_workers():
    pool = DistributedWorkerPool(
        world_size=torch.cuda.device_count(),
        backend="nccl",
        teardown_fn=default_worker_teardown,
    )
    yield pool
    pool.shutdown()
