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

import collections
import os
import shutil
from pathlib import Path

import pytest
from _test_utils.examples.gpu_lock import gpu_lock


def pytest_configure(config):
    config.addinivalue_line("markers", "deploy_empty_group: no cases for the selected backend")


def pytest_collection_modifyitems(config, items):
    # Empty parameter placeholders are outside an explicit backend selection.
    deselected = [item for item in items if item.get_closest_marker("deploy_empty_group")]
    if deselected:
        items[:] = [item for item in items if not item.get_closest_marker("deploy_empty_group")]
        config.hook.pytest_deselected(items=deselected)


def _deploy_hub_dir(model_id):
    """Hub repository id to its HF cache directory; local paths do not map."""
    if not model_id or model_id.startswith("/") or "/" not in model_id:
        return None
    return "models--" + model_id.replace("/", "--")


def pytest_collection_finish(session):
    # Runs here rather than in collection_modifyitems so the item list is final and
    # deselected deploy cases no longer count as snapshot consumers.
    consumers = {}
    for item in session.items:
        param = getattr(getattr(item, "callspec", None), "params", {}).get("command")
        if not isinstance(param, tuple) or not param or not isinstance(param[0], dict):
            continue
        repos = {
            name
            for name in (
                _deploy_hub_dir(param[0].get("model_id")),
                _deploy_hub_dir(param[0].get("base_model")),
            )
            if name
        }
        if repos:
            consumers[item.nodeid] = repos
    config = session.config
    config._deploy_hf_consumers = consumers
    config._deploy_hf_remaining = collections.Counter(
        repo for repos in consumers.values() for repo in repos
    )
    config._deploy_hf_hub = Path(
        os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    )
    # Snapshots that predate the run may be a warm cache shared with other jobs; only
    # directories this run downloaded are evicted once their last consumer finishes.
    config._deploy_hf_baseline = {
        path.name for path in config._deploy_hf_hub.glob("models--*") if path.is_dir()
    }


def pytest_runtest_teardown(item):
    consumers = getattr(item.config, "_deploy_hf_consumers", None)
    if not consumers or item.nodeid not in consumers:
        return
    for repo in consumers[item.nodeid]:
        item.config._deploy_hf_remaining[repo] -= 1
        snapshot = item.config._deploy_hf_hub / repo
        if (
            item.config._deploy_hf_remaining[repo] <= 0
            and repo not in item.config._deploy_hf_baseline
            and snapshot.is_dir()
        ):
            print(f"Evicting downloaded checkpoint {repo} after its last selected case")
            shutil.rmtree(snapshot, ignore_errors=True)


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
