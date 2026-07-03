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

"""CPU tests for the ``quant_sparse_attn_worker`` install path.

Three layers of coverage, all on CPU (no vLLM engine boot):

- ``_attach_attention_quant_impl_swap`` converts only the approved vLLM ``Attention`` modules to
  ``_QuantVLLMAttention`` (dynamic block-16 NVFP4, K/V global-scale-1.0) and leaves a realquant
  Linear untouched.
- ``_preflight_quant_sparse_attn`` / ``_validate_global_runtime`` / ``_unsupported_attention_reasons``
  reject every unsupported configuration in the support matrix with a single, layer-qualified
  ``NotImplementedError`` and never mutate the model (pure resolver).
- ``_prepare_quant_sparse_attn`` / ``_commit_quant_sparse_attn`` install atomically: a clone failure
  on any layer leaves every original ``module.impl`` and ``_value_quant_in_kernel`` gate unchanged.

Only ``create_parallel_state`` is patched away (it needs a live vLLM distributed group); the
registry, the real ``_setup``, ``set_quantizer_by_cfg``, ``post_restore_vllm_attentions``, the real
``FlashAttentionImpl`` state, ``_clone_sparse_impl``, and ``select_sparse_impl_cls`` all run for real.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import vllm.model_executor.layers.linear as vllm_linear
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

import modelopt.torch.quantization.plugins.vllm as vllm_plugin
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import ModelOptSparseAttentionImpl
from modelopt.torch.utils.distributed import ParallelState

# The worker module lives under examples/vllm_serve and is imported as a top-level module there.
_VLLM_SERVE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples", "vllm_serve"
)
sys.path.insert(0, os.path.abspath(_VLLM_SERVE_DIR))

import quant_sparse_attn_worker as worker
from quant_sparse_attn_worker import (
    AttentionType,
    _attach_attention_quant_impl_swap,
    _AttentionInstallRecord,
    _commit_quant_sparse_attn,
    _install_quant_sparse_attn,
    _preflight_quant_sparse_attn,
    _prepare_quant_sparse_attn,
    _unsupported_attention_reasons,
    _validate_global_runtime,
)


@pytest.fixture
def _no_distributed_parallel_state(monkeypatch):
    """Convert attention on CPU without a live vLLM distributed group.

    ``_QuantVLLMAttention._setup`` calls ``create_parallel_state()`` (needs ``get_dp_group()`` /
    ``get_tp_group()``); on CPU we swap it for the same default ``_initialize_parallel_state`` uses.
    """
    monkeypatch.setattr(
        vllm_plugin, "create_parallel_state", lambda: ParallelState(data_parallel_group=None)
    )


@pytest.fixture
def _no_sparse_config(monkeypatch):
    """No checkpoint sparse-attention config, so candidacy is driven purely by quant."""
    monkeypatch.setattr(worker, "load_from_checkpoint_metadata", lambda hf_config: None)


def _make_flash_impl(**overrides):
    """A real vLLM ``FlashAttentionImpl`` with initialized runtime state (clone/select work on it)."""
    impl = FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        scale=0.125,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    for key, value in overrides.items():
        setattr(impl, key, value)
    return impl


class _FakeAttention(vllm_plugin.vllm_attention.Attention):
    """Real vLLM ``Attention`` subclass with ``__init__`` bypassed (no engine needed on CPU)."""

    def __init__(
        self,
        *,
        impl=None,
        attn_type=AttentionType.DECODER,
        sliding_window=None,
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="auto",
        head_size=128,
    ):
        nn.Module.__init__(self)
        # A trivial param so post-restore device/dtype detection succeeds.
        self.dummy = nn.Parameter(torch.zeros(1))
        self.impl = impl if impl is not None else _make_flash_impl()
        self.attn_type = attn_type
        self.sliding_window = sliding_window
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.kv_cache_dtype = kv_cache_dtype
        self.head_size = head_size


def _make_worker(
    model,
    *,
    dcp=1,
    dbo=False,
    ubatching=False,
    prefix_caching=False,
    kv_transfer=None,
    speculative=None,
    block_size=16,
    hf_config=None,
    enforce_eager=True,
    dtype=torch.bfloat16,
):
    # In real vLLM ``vllm_config.model_config`` and ``model_runner.model_config`` are the same object.
    model_config = SimpleNamespace(hf_config=hf_config, enforce_eager=enforce_eager, dtype=dtype)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, enable_dbo=dbo, use_ubatching=ubatching
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching, block_size=block_size),
        kv_transfer_config=kv_transfer,
        speculative_config=speculative,
        model_config=model_config,
    )
    model_runner = SimpleNamespace(
        model=model,
        vllm_config=vllm_config,
        model_config=model_config,
        cascade_attn_enabled=False,
    )
    return SimpleNamespace(model_runner=model_runner)


class _RealQuantMethod:
    """Stand-in for a realquant Linear method (e.g. ModelOptFp8LinearMethod) -- NOT unquantized."""


class _RealQuantLinear(vllm_linear.RowParallelLinear):
    """Real vLLM Linear subclass carrying a non-``UnquantizedLinearMethod`` quant method."""

    def __init__(self):
        nn.Module.__init__(self)
        self.quant_method = _RealQuantMethod()


# --- Attach: convert only approved attention modules -------------------------------------------


@pytest.mark.usefixtures("_no_distributed_parallel_state")
def test_impl_swap_converts_attention_and_configures_nvfp4():
    attention = _FakeAttention()
    model = nn.ModuleDict({"attn": attention})

    _attach_attention_quant_impl_swap(model, {"attn"})

    converted = model.attn
    assert isinstance(converted, vllm_plugin._QuantVLLMAttention)

    # All four BMM quantizers enabled and configured as dynamic block-16 NVFP4.
    for name in ("q_bmm_quantizer", "k_bmm_quantizer", "v_bmm_quantizer", "p_bmm_quantizer"):
        quantizer = getattr(converted, name)
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.num_bits == (2, 1)
        assert (quantizer.block_sizes or {}).get(-1) == 16

    # K/V pick up the global-scale-1.0 runtime default (amax == 6 * 448 == 2688); dynamic Q/P do not.
    inputs = torch.tensor([-3.0, 5.0])
    for name in ("k_bmm_quantizer", "v_bmm_quantizer"):
        assert getattr(converted, name)._get_amax(inputs).item() == 2688.0
    assert not hasattr(converted.q_bmm_quantizer, "_runtime_default_amax")
    assert not hasattr(converted.p_bmm_quantizer, "_runtime_default_amax")


@pytest.mark.usefixtures("_no_distributed_parallel_state")
def test_impl_swap_leaves_non_attention_untouched():
    linear = _RealQuantLinear()
    model = nn.ModuleDict({"proj": linear})

    # Sanity: this Linear IS a registered QuantModule but is NOT an attention type; converting it
    # (as mtq.quantize / replace_quant_module would) trips the _VLLMParallelLinear quant_method
    # assert -- the exact failure the attention-only attach exists to avoid.
    assert type(linear) in QuantModuleRegistry
    assert not isinstance(linear, vllm_plugin._ATTENTION_TYPES)
    with pytest.raises(AssertionError):
        QuantModuleRegistry.convert(_RealQuantLinear())

    # Even with the Linear's name approved, the isinstance guard skips it (not an attention type).
    _attach_attention_quant_impl_swap(model, {"proj"})

    assert model.proj is linear
    assert type(model.proj) is _RealQuantLinear
    assert not isinstance(model.proj, vllm_plugin._QuantVLLMAttention)


@pytest.mark.usefixtures("_no_distributed_parallel_state")
def test_impl_swap_converts_only_allowed_names():
    keep = _FakeAttention()
    skip = _FakeAttention()
    model = nn.ModuleDict({"keep": keep, "skip": skip})

    _attach_attention_quant_impl_swap(model, {"keep"})

    assert isinstance(model.keep, vllm_plugin._QuantVLLMAttention)
    assert not isinstance(model.skip, vllm_plugin._QuantVLLMAttention)


# --- Preflight support matrix (pure resolver) --------------------------------------------------


def test_unsupported_reasons_empty_for_supported_decoder():
    assert _unsupported_attention_reasons(_FakeAttention()) == []


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"attn_type": "encoder"}, "DECODER"),
        ({"sliding_window": 128}, "sliding_window"),
        ({"kv_sharing_target_layer_name": "model.layers.0.self_attn"}, "kv_sharing"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "kv_cache_dtype"),
        ({"head_size": 72}, "head_size"),  # not a multiple of 16 -> blocks span heads
        ({"head_size": 40}, "head_size"),
    ],
)
def test_unsupported_reasons_for_module_attrs(kwargs, needle):
    reasons = _unsupported_attention_reasons(_FakeAttention(**kwargs))
    assert any(needle in reason for reason in reasons), reasons


def test_unsupported_reasons_accepts_head_size_multiple_of_16():
    assert _unsupported_attention_reasons(_FakeAttention(head_size=128)) == []


@pytest.mark.skipif(vllm_plugin.VllmMLAAttention is None, reason="MLA attention not available")
def test_unsupported_reasons_rejects_mla_layout():
    """MLA is a distinct class (attn_type==DECODER) — must be rejected by layout, not silently run."""

    class _FakeMLA(vllm_plugin.VllmMLAAttention):
        def __init__(self):
            nn.Module.__init__(self)
            self.attn_type = AttentionType.DECODER
            self.impl = _make_flash_impl()
            self.head_size = 128

    module = _FakeMLA()
    assert not isinstance(module, vllm_plugin.vllm_attention.Attention)
    reasons = _unsupported_attention_reasons(module)
    assert any("MLA" in reason or "layout" in reason for reason in reasons), reasons


@pytest.mark.parametrize(
    ("impl_overrides", "needle"),
    [
        ({"alibi_slopes": [0.1, 0.2]}, "alibi"),
        ({"logits_soft_cap": 30.0}, "logits_soft_cap"),
        ({"sinks": object()}, "sinks"),
    ],
)
def test_unsupported_reasons_for_impl_attrs(impl_overrides, needle):
    module = _FakeAttention(impl=_make_flash_impl(**impl_overrides))
    reasons = _unsupported_attention_reasons(module)
    assert any(needle in reason for reason in reasons), reasons


def test_unsupported_reasons_for_unrecognized_backend():
    class _WeirdImpl:
        pass

    reasons = _unsupported_attention_reasons(_FakeAttention(impl=_WeirdImpl()))
    assert any("backend" in reason for reason in reasons), reasons


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"dcp": 2}, "decode_context_parallel_size"),
        ({"dbo": True}, "dual batch overlap"),
        ({"ubatching": True}, "dual batch overlap"),
        ({"prefix_caching": True}, "prefix caching"),
        ({"kv_transfer": object()}, "connector"),
        ({"speculative": object()}, "speculative"),
        ({"block_size": 24}, "multiple"),
        ({"block_size": 0}, "multiple"),
        ({"enforce_eager": False}, "enforce_eager"),
        (
            {"dtype": torch.float32},
            "dtype",
        ),  # resolved fp32 KV cache (e.g. via kv_cache_dtype=auto)
    ],
)
def test_validate_global_runtime_flags_unsupported(kwargs, needle):
    worker_obj = _make_worker(nn.ModuleDict({"attn": _FakeAttention()}), **kwargs)
    reasons = _validate_global_runtime(worker_obj)
    assert any(needle in reason for reason in reasons), reasons


def test_validate_global_runtime_ok():
    worker_obj = _make_worker(nn.ModuleDict({"attn": _FakeAttention()}))
    assert _validate_global_runtime(worker_obj) == []


# --- ATTN_QUANT_MODE validation (no silent fall-through to un-quantized) ------------------------


def test_resolve_attn_quant_mode_default_is_impl_swap(monkeypatch):
    monkeypatch.delenv("ATTN_QUANT_MODE", raising=False)
    assert worker._resolve_attn_quant_mode() == "impl_swap"


def test_resolve_attn_quant_mode_rejects_typo(monkeypatch):
    monkeypatch.setenv("ATTN_QUANT_MODE", "impl_swp")
    with pytest.raises(ValueError, match="invalid"):
        worker._resolve_attn_quant_mode()


def test_resolve_attn_quant_mode_mtq_requires_source(monkeypatch):
    monkeypatch.setenv("ATTN_QUANT_MODE", "mtq")
    for key in ("quant_cfg", "kv_quant_cfg", "modelopt_state_path", "recipe_path"):
        monkeypatch.setitem(worker.quant_config, key, None)
    with pytest.raises(ValueError, match="requires a quant source"):
        worker._resolve_attn_quant_mode()


def test_resolve_attn_quant_mode_mtq_with_source(monkeypatch):
    monkeypatch.setenv("ATTN_QUANT_MODE", "mtq")
    monkeypatch.setitem(worker.quant_config, "quant_cfg", "NVFP4_DEFAULT_CFG")
    assert worker._resolve_attn_quant_mode() == "mtq"


@pytest.mark.usefixtures("_no_sparse_config")
def test_preflight_happy_path_returns_records():
    model = nn.ModuleDict({"a": _FakeAttention(), "b": _FakeAttention()})
    records = _preflight_quant_sparse_attn(_make_worker(model), quant_will_be_configured=True)
    assert {record.name for record in records} == {"a", "b"}


@pytest.mark.usefixtures("_no_sparse_config")
def test_preflight_raises_with_layer_name_and_reason():
    model = nn.ModuleDict({"good": _FakeAttention(), "bad": _FakeAttention(sliding_window=256)})
    with pytest.raises(NotImplementedError) as exc:
        _preflight_quant_sparse_attn(_make_worker(model), quant_will_be_configured=True)
    message = str(exc.value)
    assert "bad" in message
    assert "sliding_window" in message


@pytest.mark.usefixtures("_no_sparse_config")
def test_preflight_aggregates_global_and_per_layer_reasons():
    model = nn.ModuleDict({"bad": _FakeAttention(kv_cache_dtype="fp8")})
    worker_obj = _make_worker(model, prefix_caching=True)
    with pytest.raises(NotImplementedError) as exc:
        _preflight_quant_sparse_attn(worker_obj, quant_will_be_configured=True)
    message = str(exc.value)
    assert "prefix caching" in message
    assert "kv_cache_dtype" in message


@pytest.mark.usefixtures("_no_sparse_config")
def test_preflight_is_pure_and_leaves_model_unmutated():
    bad = _FakeAttention(sliding_window=256)
    model = nn.ModuleDict({"bad": bad})
    impl_before = bad.impl
    with pytest.raises(NotImplementedError):
        _preflight_quant_sparse_attn(_make_worker(model), quant_will_be_configured=True)
    assert bad.impl is impl_before
    assert not isinstance(bad, vllm_plugin._QuantVLLMAttention)


# --- Two-pass atomic install -------------------------------------------------------------------


@pytest.fixture
def converted_two_layer_worker(_no_distributed_parallel_state):
    """A worker whose model has two NVFP4-quantized decoder attention layers (post-attach)."""
    model = nn.ModuleDict({"a": _FakeAttention(), "b": _FakeAttention()})
    _attach_attention_quant_impl_swap(model, {"a", "b"})
    # Conversion must preserve the backend impl the sparse clone re-classes.
    assert all(isinstance(model[name].impl, FlashAttentionImpl) for name in ("a", "b"))
    return _make_worker(model)


def _records_for(worker_obj):
    model = worker_obj.model_runner.model
    return tuple(_AttentionInstallRecord(name=name, sparse_kw={}) for name in model)


def test_prepare_constructs_all_without_assigning(converted_two_layer_worker):
    worker_obj = converted_two_layer_worker
    model = worker_obj.model_runner.model
    impls_before = {name: model[name].impl for name in model}

    prepared = _prepare_quant_sparse_attn(worker_obj, _records_for(worker_obj))

    assert len(prepared) == 2
    assert all(isinstance(item.new_impl, ModelOptSparseAttentionImpl) for item in prepared)
    # Prepare must not touch the live modules.
    for name in model:
        assert model[name].impl is impls_before[name]
        assert model[name]._value_quant_in_kernel is False


def test_commit_assigns_impls_then_flips_value_gate(converted_two_layer_worker):
    worker_obj = converted_two_layer_worker
    model = worker_obj.model_runner.model

    prepared = _prepare_quant_sparse_attn(worker_obj, _records_for(worker_obj))
    _commit_quant_sparse_attn(prepared)

    for name in model:
        assert isinstance(model[name].impl, ModelOptSparseAttentionImpl)
        # V is NVFP4 -> in-kernel V ownership, so the head_dim V pre-step is skipped.
        assert model[name]._value_quant_in_kernel is True


def test_install_failure_on_last_layer_leaves_model_unchanged(converted_two_layer_worker):
    worker_obj = converted_two_layer_worker
    model = worker_obj.model_runner.model
    # Make the second layer unclonable (sinks are unsupported) so prepare raises after the first
    # layer already prepared -- commit must never run.
    model["b"].impl.sinks = object()
    impls_before = {name: model[name].impl for name in model}

    with pytest.raises(NotImplementedError, match="sinks"):
        _install_quant_sparse_attn(worker_obj, _records_for(worker_obj))

    for name in model:
        assert model[name].impl is impls_before[name]
        assert model[name]._value_quant_in_kernel is False
