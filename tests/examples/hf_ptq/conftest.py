# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.examples.gpu_lock import gpu_lock


def _hold_gpu(tmp_path_factory, *, shared: bool):
    if "PYTEST_XDIST_WORKER" not in os.environ:  # tests already run one at a time
        yield
        return
    # The parent of each worker's base temp directory is shared by all workers of the session.
    # The lock is taken in fixture setup, which the per-test timeout does not cover.
    with gpu_lock(tmp_path_factory.getbasetemp().parent, shared=shared):
        yield


@pytest.fixture
def shared_gpu(tmp_path_factory):
    """Share the GPU with other ``shared_gpu`` tests under pytest-xdist; small models only."""
    yield from _hold_gpu(tmp_path_factory, shared=True)


@pytest.fixture
def exclusive_gpu(tmp_path_factory):
    """Run alone on the GPU under pytest-xdist."""
    yield from _hold_gpu(tmp_path_factory, shared=False)


@pytest.fixture(autouse=True)
def _per_test_save_root(tmp_path, monkeypatch):
    # huggingface_example.sh names a checkpoint directory from the model and format alone, so runs
    # that differ only in KV format, TP or PP would otherwise share it and skip PTQ, or race.
    monkeypatch.setenv("ROOT_SAVE_PATH", str(tmp_path))
