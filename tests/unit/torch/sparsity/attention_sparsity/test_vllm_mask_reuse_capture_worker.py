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

"""Tests for the policy-free vLLM capture worker bootstrap and RPC wrappers."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def vllm_mask_reuse_capture(monkeypatch):
    """Import the worker against a minimal optional-vLLM boundary."""

    class Worker:
        pass

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.worker": ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_worker": ModuleType("vllm.v1.worker.gpu_worker"),
    }
    modules["vllm.v1.worker.gpu_worker"].Worker = Worker
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    target = "modelopt.torch.sparsity.attention_sparsity.plugins.vllm_mask_reuse_capture"
    sys.modules.pop(target, None)
    module = importlib.import_module(target)
    yield module
    sys.modules.pop(target, None)


def _api(calls):
    return SimpleNamespace(
        configure_capture_runtime=lambda plan: calls.append(("configure", plan)),
        capture_status=lambda: {
            "capture_schema_version": 1,
            "available": True,
            "rank": 0,
            "world_size": 1,
            "reason": None,
        },
        begin_capture=lambda invocation: calls.append(("begin", invocation)) or {"armed": True},
        drain_capture=lambda: calls.append(("drain",)) or {"records": []},
    )


def test_bootstrap_requires_gate_and_plan(monkeypatch, vllm_mask_reuse_capture):
    monkeypatch.delenv(vllm_mask_reuse_capture.CAPTURE_ENV, raising=False)
    monkeypatch.setenv(vllm_mask_reuse_capture.PLAN_ENV, "qwen3_stride2")
    with pytest.raises(RuntimeError, match="CALIBRATION_CAPTURE=1"):
        vllm_mask_reuse_capture._configure_capture_before_model_load()

    monkeypatch.setenv(vllm_mask_reuse_capture.CAPTURE_ENV, "1")
    monkeypatch.delenv(vllm_mask_reuse_capture.PLAN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="must name"):
        vllm_mask_reuse_capture._configure_capture_before_model_load()


def test_bootstrap_installs_policy_free_runtime_before_load(monkeypatch, vllm_mask_reuse_capture):
    calls = []
    api = _api(calls)
    monkeypatch.setenv(vllm_mask_reuse_capture.CAPTURE_ENV, "1")
    monkeypatch.setenv(vllm_mask_reuse_capture.PLAN_ENV, "nemotron3_ultra_stride2")
    monkeypatch.setattr(vllm_mask_reuse_capture, "_capture_api", lambda: api)

    assert vllm_mask_reuse_capture._configure_capture_before_model_load() is api
    assert calls == [("configure", "nemotron3_ultra_stride2")]


def test_worker_rpc_methods_forward_only_to_capture_api(monkeypatch, vllm_mask_reuse_capture):
    calls = []
    api = _api(calls)
    monkeypatch.setattr(vllm_mask_reuse_capture, "_capture_api", lambda: api)
    worker = object.__new__(vllm_mask_reuse_capture.MaskReuseCaptureWorker)
    invocation = {"capture_schema_version": 1}

    assert worker.mask_reuse_capture_status()["available"] is True
    assert worker.mask_reuse_capture_begin(invocation) == {"armed": True}
    assert worker.mask_reuse_capture_drain() == {"records": []}
    assert calls == [("begin", invocation), ("drain",)]
