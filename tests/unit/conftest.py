# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import platform

import pytest

# Enforce no HuggingFace Hub network access for unit tests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

with contextlib.suppress(ImportError):
    import huggingface_hub.constants as _hf_constants

    _hf_constants.HF_HUB_OFFLINE = True


@pytest.fixture(scope="session")
def cpu_dist_workers():
    """Two persistent gloo ranks shared by every multi-rank unit test.

    A fresh spawn per test re-imports torch and transformers in each child (~10-20s on CI), so the
    workers start once. Jobs must leave the process group intact and undo any global state they set.
    """
    if platform.system() == "Windows":
        pytest.skip("Multi-rank tests are not run on Windows")
    # Lazy: the onnx-only partial-install job has no torch.
    from _test_utils.torch.distributed.utils import DistributedWorkerPool

    pool = DistributedWorkerPool(world_size=2, backend="gloo", teardown_fn=None)
    yield pool
    pool.shutdown()
