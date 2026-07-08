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

"""Focused tests for the fixed-recipe quant+sparse vLLM worker."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import vllm
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flashinfer import FlashInferImpl
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from modelopt.torch.quantization.plugins import vllm as quant_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    get_flashinfer_sparse_impl_cls,
)

_WORKER_PATH = Path(__file__).parents[5] / "examples/vllm_serve/sparse_attn_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "shared_attention_worker_quant_test", _WORKER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker_module = _load_worker_module()


def test_quant_policy_rejects_old_vllm(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.9.0")
    worker_module._load_quant_api.cache_clear()

    try:
        with pytest.raises(RuntimeError, match=r"vLLM >= 0\.14\.0"):
            worker_module._install_attention(SimpleNamespace(), quantize=True)
    finally:
        worker_module._load_quant_api.cache_clear()


def _attention(attn_type=AttentionType.DECODER, impl_cls=FlashAttentionImpl):
    module = object.__new__(quant_plugin.vllm_attention.Attention)
    nn.Module.__init__(module)
    module.dummy = nn.Parameter(torch.zeros(1, dtype=torch.float16))
    module.impl = object.__new__(impl_cls)
    module.impl.__dict__.update(alibi_slopes=None, logits_soft_cap=None, sinks=None)
    module.__dict__.update(
        attn_type=attn_type,
        head_size=64,
        head_size_v=64,
        sliding_window=None,
        kv_cache_dtype="auto",
        kv_sharing_target_layer_name=None,
    )
    return module


def _worker(model):
    runner = SimpleNamespace(
        model=model, model_config=SimpleNamespace(hf_config=None), cascade_attn_enabled=True
    )
    return SimpleNamespace(model_runner=runner)


def _patch_conversion(monkeypatch):
    monkeypatch.setattr(
        quant_plugin,
        "create_parallel_state",
        lambda: quant_plugin.ParallelState(data_parallel_group=None),
    )
    monkeypatch.setattr(worker_module, "load_from_checkpoint_metadata", lambda _: None)
    monkeypatch.setattr(worker_module, "_global_errors", lambda _worker, _api: [])


@pytest.mark.parametrize("impl_cls", [FlashAttentionImpl, FlashInferImpl])
def test_install_converts_only_attention_and_configures_fixed_recipe(monkeypatch, impl_cls):
    _patch_conversion(monkeypatch)
    attention = _attention(impl_cls=impl_cls)
    linear = nn.Linear(4, 4)
    model = nn.ModuleDict({"attn": attention, "linear": linear})
    state = _worker(model)

    worker_module._install_attention(state, quantize=True)

    converted = model["attn"]
    assert isinstance(converted, quant_plugin._QuantVLLMAttention)
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "p_bmm_quantizer", "v_bmm_quantizer"):
        quantizer = getattr(converted, name)
        assert quantizer.is_enabled and quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    assert not hasattr(converted.q_bmm_quantizer, "_amax")
    assert converted.k_bmm_quantizer._amax == 6.0 * 448.0
    assert converted.v_bmm_quantizer._amax == 6.0 * 448.0
    assert converted._query_quant_in_kernel is True
    assert converted._value_quant_in_kernel is True
    expected_impl_cls = (
        ModelOptSparseAttentionImpl
        if impl_cls is FlashAttentionImpl
        else get_flashinfer_sparse_impl_cls()
    )
    assert isinstance(converted.impl, impl_cls)
    assert type(converted.impl) is expected_impl_cls
    assert converted.impl.quant_kw == {
        "p_qdq": "nvfp4",
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4",
        "v_qdq_amax": 6.0 * 448.0,
    }
    assert model["linear"] is linear
    assert state.model_runner.cascade_attn_enabled is False


def test_validation_of_all_layouts_precedes_mutation(monkeypatch):
    _patch_conversion(monkeypatch)
    good, bad = _attention(), _attention(AttentionType.ENCODER)
    bad.head_size_v = 32
    bad.dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    model = nn.ModuleDict({"good": good, "bad": bad})
    original_impl = good.impl

    with pytest.raises(NotImplementedError) as exc:
        worker_module._install_attention(_worker(model), quantize=True)

    assert all(needle in str(exc.value) for needle in ("bad: attn_type", "head_size_v", "float32"))
    assert model["good"] is good and good.impl is original_impl
    assert not isinstance(good, quant_plugin._QuantVLLMAttention)


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

    api = SimpleNamespace(
        torch=SimpleNamespace(inference_mode=lambda: recorded_context("inference")),
        plugin=SimpleNamespace(
            disable_compilation=lambda actual_model: recorded_context("compilation", actual_model)
        ),
    )
    monkeypatch.setattr(worker_module, "_quant_api", lambda: api)

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


@pytest.mark.parametrize(
    ("mode", "rejected"),
    [(CUDAGraphMode.FULL, True), (CUDAGraphMode.FULL_AND_PIECEWISE, False)],
)
def test_full_mixed_cudagraph_validation(mode, rejected):
    api = worker_module._quant_api()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        model_config=SimpleNamespace(dtype=torch.float16),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )
    errors = worker_module._global_errors(
        SimpleNamespace(model_runner=SimpleNamespace(vllm_config=config)), api
    )
    assert any("mixed" in error for error in errors) is rejected


def test_calibrated_decode_skip_softmax_rejects_full_decode_graphs():
    api = worker_module._quant_api()
    sparse_kw = {
        "threshold_scale_factor": {
            "prefill": {"a": 1.0, "b": 2.0},
            "decode": {"a": 0.1, "b": 1.0},
        }
    }

    assert "calibrated decode skip-softmax" in worker_module._sparse_graph_error(
        sparse_kw, CUDAGraphMode.FULL_AND_PIECEWISE, api
    )
    assert worker_module._sparse_graph_error(sparse_kw, CUDAGraphMode.PIECEWISE, api) is None
    assert (
        worker_module._sparse_graph_error(
            {"threshold_scale_factor": {"prefill": {"a": 1.0, "b": 2.0}}},
            CUDAGraphMode.FULL_AND_PIECEWISE,
            api,
        )
        is None
    )
