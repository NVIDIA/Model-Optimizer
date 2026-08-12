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

"""Tests for bounded cleanup in shared distributed test helpers."""

import os
from types import SimpleNamespace

import pytest
from _test_utils.torch.distributed import utils


@pytest.mark.parametrize("raises", [False, True], ids=("success", "failure"))
def test_init_process_destroys_one_shot_process_group(monkeypatch, raises):
    events = []
    initialized = False

    def init_process_group(*_args, **_kwargs):
        nonlocal initialized
        initialized = True
        events.append("initialized")

    def destroy_process_group():
        nonlocal initialized
        initialized = False
        events.append("destroyed")

    def job(_rank, _size):
        events.append("job")
        if raises:
            raise RuntimeError("worker failed")

    for name in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "WANDB_DISABLED",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(utils.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(utils.dist, "destroy_process_group", destroy_process_group)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(utils.torch, "manual_seed", lambda _seed: None)

    if raises:
        with pytest.raises(RuntimeError, match="worker failed"):
            utils.init_process(0, 1, job=job)
    else:
        utils.init_process(0, 1, job=job)

    assert events == ["initialized", "job", "destroyed"]


def test_init_process_leaves_fixture_owned_process_group_initialized(monkeypatch):
    initialized = False

    def init_process_group(*_args, **_kwargs):
        nonlocal initialized
        initialized = True

    for name in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "WANDB_DISABLED",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(utils.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        utils.dist,
        "destroy_process_group",
        lambda: pytest.fail("fixture owner must tear down its process group"),
    )
    monkeypatch.setattr(utils.torch, "manual_seed", lambda _seed: None)

    utils.init_process(0, 1)

    assert initialized


@pytest.mark.parametrize(
    "terminate_reaps",
    [True, False],
    ids=("terminate", "kill"),
)
def test_spawn_multiprocess_job_reaps_children_when_join_is_interrupted(
    monkeypatch,
    terminate_reaps,
):
    class Process:
        def __init__(self):
            self.exitcode = None
            self.join_timeouts = []
            self.killed = False
            self.terminated = False
            self._alive = False

        def start(self):
            self._alive = True

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)
            if timeout is None:
                raise TimeoutError("join interrupted")
            if self.killed or (self.terminated and terminate_reaps):
                self._alive = False
                self.exitcode = -9 if self.killed else -15

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self._alive = False

    processes = []

    def make_process(*, target, args):
        assert target is utils.init_process
        assert args[-2:] == ("gloo", 12345)
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=make_process),
    )
    monkeypatch.setattr(utils, "monotonic", lambda: 0)

    with pytest.raises(TimeoutError, match="join interrupted"):
        utils.spawn_multiprocess_job(2, lambda *_args: None)

    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    assert all(process.killed is not terminate_reaps for process in processes)
    expected_cleanup_joins = [10] if terminate_reaps else [10, 10]
    assert processes[0].join_timeouts == [None, *expected_cleanup_joins]
    assert processes[1].join_timeouts == expected_cleanup_joins


@pytest.mark.parametrize("coverage_config", ["/tmp/coverage-config.toml", None])
def test_spawn_multiprocess_job_disables_child_coverage_and_restores_parent(
    monkeypatch,
    coverage_config,
):
    inherited_coverage = []

    class Process:
        exitcode = 0

        def start(self):
            inherited_coverage.append(os.environ.get("COVERAGE_PROCESS_START"))

        def join(self, timeout=None):
            assert os.environ.get("COVERAGE_PROCESS_START") == coverage_config

        def is_alive(self):
            return False

    if coverage_config is None:
        monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    else:
        monkeypatch.setenv("COVERAGE_PROCESS_START", coverage_config)
    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: Process()),
    )

    utils.spawn_multiprocess_job(2, lambda *_args: None)

    assert inherited_coverage == [None, None]
    assert os.environ.get("COVERAGE_PROCESS_START") == coverage_config


def test_spawn_multiprocess_job_preserves_child_coverage_for_other_backends(monkeypatch):
    coverage_config = "/tmp/coverage-config.toml"
    inherited_coverage = []

    class Process:
        exitcode = 0

        def start(self):
            inherited_coverage.append(os.environ.get("COVERAGE_PROCESS_START"))

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    monkeypatch.setenv("COVERAGE_PROCESS_START", coverage_config)
    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: Process()),
    )

    utils.spawn_multiprocess_job(2, lambda *_args: None, backend="nccl")

    assert inherited_coverage == [coverage_config, coverage_config]
    assert os.environ["COVERAGE_PROCESS_START"] == coverage_config


@pytest.mark.parametrize("coverage_config", ["/tmp/coverage-config.toml", None])
def test_spawn_multiprocess_job_restores_coverage_after_partial_start(
    monkeypatch,
    coverage_config,
):
    processes = []

    class Process:
        def __init__(self, index):
            self.exitcode = None
            self.index = index
            self.terminated = False

        def start(self):
            assert "COVERAGE_PROCESS_START" not in os.environ
            if self.index == 1:
                raise RuntimeError("start failed")

        def join(self, timeout=None):
            assert os.environ.get("COVERAGE_PROCESS_START") == coverage_config
            if timeout is not None and self.terminated:
                self.exitcode = -15

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.terminated = True

        def kill(self):
            pytest.fail("terminated child should not need kill")

    def make_process(**_kwargs):
        process = Process(len(processes))
        processes.append(process)
        return process

    if coverage_config is None:
        monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    else:
        monkeypatch.setenv("COVERAGE_PROCESS_START", coverage_config)
    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=make_process),
    )
    monkeypatch.setattr(utils, "monotonic", lambda: 0)

    with pytest.raises(RuntimeError, match="start failed"):
        utils.spawn_multiprocess_job(2, lambda *_args: None)

    assert processes[0].terminated
    assert os.environ.get("COVERAGE_PROCESS_START") == coverage_config


def test_spawn_multiprocess_job_reaps_siblings_after_worker_failure(monkeypatch):
    class Process:
        def __init__(self, index):
            self.exitcode = 1 if index == 0 else None
            self.index = index
            self.terminated = False

        def start(self):
            pass

        def join(self, timeout=None):
            if timeout is not None and self.terminated:
                self.exitcode = -15

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.terminated = True

        def kill(self):
            pytest.fail("terminated sibling should not need kill")

    processes = []

    def make_process(**_kwargs):
        process = Process(len(processes))
        processes.append(process)
        return process

    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=make_process),
    )
    monkeypatch.setattr(utils, "monotonic", lambda: 0)

    with pytest.raises(RuntimeError, match="distributed worker exited with code 1"):
        utils.spawn_multiprocess_job(2, lambda *_args: None)

    assert processes[1].terminated
    assert processes[1].exitcode == -15


def test_spawn_multiprocess_job_preserves_primary_error_when_cleanup_fails(monkeypatch):
    class Process:
        exitcode = None

        def start(self):
            pass

        def join(self, timeout=None):
            if timeout is None:
                raise TimeoutError("primary join failure")

        def is_alive(self):
            return True

        def terminate(self):
            raise RuntimeError("cleanup failure")

    monkeypatch.setattr(utils, "get_free_port", lambda: 12345)
    monkeypatch.setattr(
        utils.mp,
        "get_context",
        lambda _method: SimpleNamespace(Process=lambda **_kwargs: Process()),
    )

    with pytest.raises(TimeoutError, match="primary join failure") as exc_info:
        utils.spawn_multiprocess_job(1, lambda *_args: None)

    if hasattr(exc_info.value, "__notes__"):
        assert exc_info.value.__notes__ == [
            "distributed child cleanup also failed: RuntimeError('cleanup failure')"
        ]
    else:
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "cleanup failure"
