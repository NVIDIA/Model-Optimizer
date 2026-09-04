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

"""Focused tests for the quant+sparse vLLM worker lifecycle."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.quantization.plugins import vllm as quant_plugin

_WORKER_PATH = Path(__file__).parents[5] / "examples/vllm_serve/sparse_attn_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "shared_attention_worker_quant_test", _WORKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker_module = _load_worker_module()


def test_quant_memory_profile_uses_inference_mode_and_disables_compilation(monkeypatch):
    events = []
    model = object()

    @contextmanager
    def recorded_context(name, value=None):
        events.append(("enter", name, value))
        try:
            yield
        finally:
            events.append(("exit", name, value))

    monkeypatch.setattr(
        torch,
        "inference_mode",
        lambda: recorded_context("inference"),
    )
    monkeypatch.setattr(
        quant_plugin,
        "disable_compilation",
        lambda actual_model: recorded_context("compilation", actual_model),
    )

    instance = object.__new__(worker_module.QuantSparseAttnWorker)
    instance.model_runner = SimpleNamespace(model=SimpleNamespace(unwrap=lambda: model))

    def profile(actual_worker):
        events.append(("profile", actual_worker))
        return 73

    monkeypatch.setattr(BaseWorker, "determine_available_memory", profile)

    assert instance.determine_available_memory() == 73
    assert events == [
        ("enter", "inference", None),
        ("enter", "compilation", model),
        ("profile", instance),
        ("exit", "compilation", model),
        ("exit", "inference", None),
    ]


def _quant_worker_with_config(monkeypatch, calls, additional_config):
    module = _load_worker_module()
    monkeypatch.setattr(BaseWorker, "load_model", lambda *_a, **_k: calls.append("base"))
    monkeypatch.setattr(
        module,
        "install_vllm_nvfp4_attention",
        lambda runner, **kw: (
            calls.append(("install", kw))
            or SimpleNamespace(installed_count=0, sparse_algorithm=None, backend_counts={})
        ),
    )
    instance = object.__new__(module.QuantSparseAttnWorker)
    instance.model_runner = SimpleNamespace()
    instance.vllm_config = SimpleNamespace(additional_config=additional_config)
    return instance


def test_quant_worker_defaults_to_nvfp4(monkeypatch):
    calls = []
    instance = _quant_worker_with_config(monkeypatch, calls, None)
    instance.load_model()
    assert calls[0] == "base"
    assert calls[1][1] == {"sparse_cfg": "checkpoint"}


def test_quant_worker_reads_formats_from_additional_config(monkeypatch):
    calls = []
    instance = _quant_worker_with_config(
        monkeypatch,
        calls,
        {"modelopt_attn_quant": {"p_format": "fp8", "v_format": "fp8"}},
    )
    instance.load_model()
    assert calls[0] == "base"
    assert calls[1][1] == {"sparse_cfg": "checkpoint", "p_format": "fp8", "v_format": "fp8"}


def test_quant_worker_rejects_unknown_format_keys(monkeypatch):
    calls = []
    instance = _quant_worker_with_config(
        monkeypatch, calls, {"modelopt_attn_quant": {"o_format": "fp8"}}
    )
    with pytest.raises(ValueError, match="o_format"):
        instance.load_model()
    assert ("install", {"sparse_cfg": "checkpoint"}) not in calls
