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
        def load_model(self):
            self.load_events.append("load_model")
            if hasattr(self, "override_cache_dtype"):
                self.vllm_config.cache_config.cache_dtype = self.override_cache_dtype

    monkeypatch.delenv("MASK_REUSE_CALIBRATION_PROVIDER", raising=False)

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


@pytest.mark.parametrize(
    ("provider", "page", "api_page", "valid"),
    [
        ("fa4", 16, None, True),
        ("fa4", 4160, None, False),
        ("reference", 16, None, True),
        ("reference", 4160, None, False),
        ("reference", 4160, 4160, True),
        ("reference", 4160, 16, False),
        ("reference", True, True, False),
        ("reference", 32, 32, False),
    ],
)
def test_actual_page_rpc_preserves_fa4_contract(
    monkeypatch, vllm_mask_reuse_capture, provider, page, api_page, valid
):
    api = _api([])
    if api_page is not None:
        api.runtime_geometry = lambda: {"kv_page_tokens": api_page}
    monkeypatch.setattr(vllm_mask_reuse_capture, "_capture_api", lambda: api)
    monkeypatch.setenv(vllm_mask_reuse_capture.PROVIDER_ENV, provider)
    worker = object.__new__(vllm_mask_reuse_capture.MaskReuseCaptureWorker)
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=page))
    if valid:
        assert worker.mask_reuse_capture_runtime_geometry() == {"kv_page_tokens": page}
    else:
        with pytest.raises(RuntimeError):
            worker.mask_reuse_capture_runtime_geometry()


def test_reference_installs_host_adapter_after_native_model_load(
    monkeypatch, vllm_mask_reuse_capture
):
    calls = []
    api = _api(calls)
    api.__file__ = "/pinned/src/attention_backend/src/mask_reuse_vllm/reference/capture.py"
    monkeypatch.setenv("MASK_REUSE_CALIBRATION_SOURCE_ROOT", "/pinned")
    api.install_capture = lambda runner: calls.append(("install", runner))
    monkeypatch.setenv(vllm_mask_reuse_capture.CAPTURE_ENV, "1")
    monkeypatch.setenv(vllm_mask_reuse_capture.PLAN_ENV, "qwen3_235b_topology_discovery")
    monkeypatch.setenv(vllm_mask_reuse_capture.PROVIDER_ENV, "reference")
    monkeypatch.setitem(sys.modules, "mask_reuse_vllm.reference.capture", api)
    worker = object.__new__(vllm_mask_reuse_capture.MaskReuseCaptureWorker)
    worker.load_events = calls
    worker.model_runner = object()
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="bfloat16"))

    worker.load_model()

    assert calls == [
        ("configure", "qwen3_235b_topology_discovery"),
        "load_model",
        ("install", worker.model_runner),
    ]


@pytest.mark.parametrize(
    ("cache_dtype", "after_load"), [("auto", False), ("fp8", False), ("fp8", True)]
)
def test_worker_rejects_unpinned_or_changed_cache_before_capture(
    monkeypatch, vllm_mask_reuse_capture, cache_dtype, after_load
):
    calls = []
    monkeypatch.setenv(vllm_mask_reuse_capture.CAPTURE_ENV, "1")
    monkeypatch.setenv(vllm_mask_reuse_capture.PLAN_ENV, "qwen3_235b_topology_discovery")
    monkeypatch.setattr(vllm_mask_reuse_capture, "_capture_api", lambda: _api(calls))
    worker = object.__new__(vllm_mask_reuse_capture.MaskReuseCaptureWorker)
    worker.load_events = calls
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="bfloat16" if after_load else cache_dtype)
    )
    if after_load:
        worker.override_cache_dtype = cache_dtype

    with pytest.raises(RuntimeError, match="requires explicit kv_cache_dtype='bfloat16'"):
        worker.load_model()

    assert calls == (
        [("configure", "qwen3_235b_topology_discovery"), "load_model"] if after_load else []
    )


@pytest.mark.parametrize("wrong_source", [False, True])
def test_reference_rejects_missing_install_hook_or_unwitnessed_source(
    monkeypatch, vllm_mask_reuse_capture, wrong_source
):
    monkeypatch.setenv(vllm_mask_reuse_capture.CAPTURE_ENV, "1")
    monkeypatch.setenv(vllm_mask_reuse_capture.PROVIDER_ENV, "reference")
    api = _api([])
    api.__file__ = "/pinned/src/attention_backend/src/mask_reuse_vllm/reference/capture.py"
    monkeypatch.setenv(
        "MASK_REUSE_CALIBRATION_SOURCE_ROOT", "/other" if wrong_source else "/pinned"
    )
    monkeypatch.setitem(sys.modules, "mask_reuse_vllm.reference.capture", api)
    with pytest.raises(
        RuntimeError, match="witnessed source" if wrong_source else "install_capture"
    ):
        vllm_mask_reuse_capture._capture_api()
