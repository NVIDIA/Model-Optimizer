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

"""Tests for the engine-shutdown helpers in specdec_bench/utils.py.

run.py must shut the engine down once it is done.
"""

import asyncio

import pytest
from specdec_bench.models import AutoDeployModel, SGLANGModel, TRTLLMPYTModel, VLLMModel
from specdec_bench.models.vllm import ENGINE_SHUTDOWN_TIMEOUT_S
from specdec_bench.utils import shutdown_engine, stop_runner


class _SyncEngine:
    """vLLM v1 shape: synchronous shutdown(timeout=...)."""

    def __init__(self):
        self.calls = []

    def shutdown(self, timeout=None):
        self.calls.append(timeout)


class _LegacySyncEngine:
    def __init__(self):
        self.calls = 0

    def shutdown(self):
        self.calls += 1


class _AsyncEngine:
    def __init__(self):
        self.done = False

    async def shutdown(self):
        self.done = True


def test_sync_shutdown_gets_timeout():
    engine = _SyncEngine()
    loop = asyncio.new_event_loop()
    try:
        # vLLM v1 returns None here; the old VLLMModel.stop() handed that to
        # run_until_complete, which raised a TypeError before the loop was closed.
        shutdown_engine(engine, loop, timeout=60)
    finally:
        loop.close()
    assert engine.calls == [60]


def test_legacy_shutdown_without_timeout():
    engine = _LegacySyncEngine()
    shutdown_engine(engine, loop=None, timeout=60)
    assert engine.calls == 1


def test_coroutine_shutdown_runs_on_loop():
    engine = _AsyncEngine()
    loop = asyncio.new_event_loop()
    try:
        shutdown_engine(engine, loop)
    finally:
        loop.close()
    assert engine.done


def test_stop_runner_calls_stop():
    class _Runner:
        stopped = False

        def stop(self):
            self.stopped = True

    runner = _Runner()
    stop_runner(runner)
    assert runner.stopped


def test_stop_runner_swallows_shutdown_errors(capsys):
    class _Runner:
        def stop(self):
            raise RuntimeError("boom")

    stop_runner(_Runner())
    assert "engine shutdown failed" in capsys.readouterr().out


@pytest.mark.parametrize("model_cls", [SGLANGModel, TRTLLMPYTModel, AutoDeployModel])
def test_model_stop_shuts_engine_down(model_cls):
    engine = _LegacySyncEngine()
    model = model_cls.__new__(model_cls)
    model.model = engine
    model.stop()
    assert engine.calls == 1


def test_vllm_model_stop_shuts_engine_down_and_closes_loop():
    engine = _SyncEngine()
    model = VLLMModel.__new__(VLLMModel)
    model.model = engine
    model.loop = asyncio.new_event_loop()
    model.stop()
    assert engine.calls == [ENGINE_SHUTDOWN_TIMEOUT_S]
    assert model.loop.is_closed()
