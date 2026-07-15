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

"""Tests for the reusable vLLM attention runtime installer."""

from types import SimpleNamespace

import pytest
import torch
import vllm
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import CrossAttention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flashinfer import FlashInferImpl

from modelopt.torch.quantization.plugins import vllm as quant_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm_runtime
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    get_flashinfer_sparse_impl_cls,
)


def _bare_attention(impl_cls=FlashAttentionImpl, module_cls=None):
    module = object.__new__(module_cls or vllm_runtime._VLLM_ATTENTION)
    nn.Module.__init__(module)
    module.attn_type = "decoder"
    module.head_size = 64
    module.device = torch.device("cpu")
    module.dtype = torch.float16
    module.impl = object.__new__(impl_cls)
    module.impl.sinks = None
    return module


def _sparse_metadata():
    return {
        "config_groups": {
            "group_0": {
                "algorithm": "sparse_softmax",
                "sparsity_n": 2,
                "sparsity_m": 4,
            }
        }
    }


def _model_runner(model, *, sparse_metadata=None):
    hf_config = SimpleNamespace(sparse_attention_config=sparse_metadata)
    model_config = SimpleNamespace(hf_config=hf_config, dtype=torch.float16)
    return SimpleNamespace(
        model=model,
        model_config=model_config,
        cascade_attn_enabled=True,
        vllm_config=SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                enable_dbo=False,
                use_ubatching=False,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
            kv_transfer_config=None,
            speculative_config=None,
        ),
    )


def test_sparse_install_from_checkpoint_is_validation_atomic():
    valid = _bare_attention()
    invalid = _bare_attention()
    invalid.sliding_window = (128, 128)
    valid_impl = valid.impl
    invalid_impl = invalid.impl
    runner = _model_runner(
        nn.ModuleDict({"valid_attn": valid, "invalid_attn": invalid}),
        sparse_metadata=_sparse_metadata(),
    )
    del runner.vllm_config

    with pytest.raises(NotImplementedError, match="sliding_window"):
        vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)

    assert valid.impl is valid_impl
    assert invalid.impl is invalid_impl
    assert runner.cascade_attn_enabled is True


def test_installer_rejects_cross_attention_layout_even_if_marked_decoder():
    attention = _bare_attention(module_cls=CrossAttention)
    original_impl = attention.impl
    runner = _model_runner(
        nn.ModuleDict({"cross_attn": attention}),
        sparse_metadata=_sparse_metadata(),
    )
    del runner.vllm_config

    with pytest.raises(NotImplementedError, match="layout CrossAttention"):
        vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)

    assert attention.impl is original_impl


@pytest.mark.parametrize("impl_cls", [FlashAttentionImpl, FlashInferImpl])
def test_sparse_install_uses_checkpoint_metadata(monkeypatch, impl_cls):
    attention = _bare_attention(impl_cls)
    runner = _model_runner(
        nn.ModuleDict({"attn": attention}),
        sparse_metadata=_sparse_metadata(),
    )
    del runner.vllm_config
    monkeypatch.setattr(
        vllm_runtime.attention_plugin, "patch_flashinfer_metadata_builder", lambda: True
    )

    report = vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)

    expected_impl_cls = (
        ModelOptSparseAttentionImpl
        if impl_cls is FlashAttentionImpl
        else get_flashinfer_sparse_impl_cls()
    )
    assert type(attention.impl) is expected_impl_cls
    assert attention.impl.sparse_kw["sparsity_n"] == 2
    assert report.installed_layers == ("attn",)
    assert report.sparse_layers == ("attn",)
    assert report.backend_counts == {expected_impl_cls.__name__: 1}
    assert runner.cascade_attn_enabled is False


def test_sparse_install_is_noop_without_checkpoint_metadata(monkeypatch):
    attention = _bare_attention()
    original_impl = attention.impl
    runner = _model_runner(nn.ModuleDict({"attn": attention}))
    monkeypatch.setattr(vllm, "__version__", "0.14.0")

    report = vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)

    assert report.installed_count == 0
    assert not report.transforms_active
    assert attention.impl is original_impl
    assert runner.cascade_attn_enabled is True


@pytest.mark.parametrize("quantize", [False, True])
def test_active_install_rejects_old_vllm_before_mutation(monkeypatch, quantize):
    attention = _bare_attention()
    original_impl = attention.impl
    runner = _model_runner(
        nn.ModuleDict({"attn": attention}),
        sparse_metadata=_sparse_metadata(),
    )
    monkeypatch.setattr(vllm, "__version__", "0.14.0")

    install = (
        vllm_runtime.install_vllm_nvfp4_attention
        if quantize
        else vllm_runtime.install_vllm_sparse_attention_from_checkpoint
    )
    with pytest.raises(RuntimeError, match=r"vLLM >= 0\.15\.0"):
        install(runner)

    assert attention.impl is original_impl
    assert not hasattr(attention, "_query_quant_in_kernel")
    assert runner.cascade_attn_enabled is True


def _fake_quant_plugin(configured):
    def configure(module, *, device, dtype):
        configured.append(module)
        module.device = device
        module.dtype = dtype
        quantizer = SimpleNamespace(
            is_enabled=True,
            is_nvfp4_dynamic=True,
            block_sizes={-1: 16},
            _amax=torch.tensor(1.0),
        )
        module.q_bmm_quantizer = quantizer
        module.k_bmm_quantizer = quantizer
        module.p_bmm_quantizer = quantizer
        module.v_bmm_quantizer = quantizer
        return module

    return SimpleNamespace(
        _get_device_dtype=lambda module: (module.device, module.dtype),
        configure_vllm_nvfp4_attention_quantizers=configure,
    )


@pytest.mark.parametrize("with_sparse", [False, True])
@pytest.mark.parametrize("impl_cls", [FlashAttentionImpl, FlashInferImpl])
def test_nvfp4_install_composes_real_quantizers_with_optional_sparsity(
    monkeypatch,
    with_sparse,
    impl_cls,
):
    attention = _bare_attention(impl_cls)
    unrelated = nn.Linear(4, 4)
    model = nn.ModuleDict({"attn": attention, "linear": unrelated})
    runner = _model_runner(model, sparse_metadata=_sparse_metadata() if with_sparse else None)
    monkeypatch.setattr(
        quant_plugin,
        "create_parallel_state",
        lambda: quant_plugin.ParallelState(data_parallel_group=None),
    )
    monkeypatch.setattr(
        vllm_runtime.attention_plugin, "patch_flashinfer_metadata_builder", lambda: True
    )

    report = vllm_runtime.install_vllm_nvfp4_attention(runner)

    expected_impl_cls = (
        ModelOptSparseAttentionImpl
        if impl_cls is FlashAttentionImpl
        else get_flashinfer_sparse_impl_cls()
    )
    assert isinstance(attention, quant_plugin._QuantVLLMAttention)
    assert type(attention.impl) is expected_impl_cls
    for name in ("q", "k", "p", "v"):
        assert getattr(attention, f"{name}_bmm_quantizer").is_nvfp4_dynamic
    assert attention.k_bmm_quantizer._amax == 6.0 * 448.0
    assert attention.v_bmm_quantizer._amax == 6.0 * 448.0
    assert attention._query_quant_in_kernel is True
    assert attention._value_quant_in_kernel is True
    assert attention.impl.quant_kw == {
        "p_qdq": "nvfp4",
        "p_qdq_amax": 1.0,
        "v_qdq": "nvfp4",
        "v_qdq_amax": 6.0 * 448.0,
    }
    assert bool(attention.impl.sparse_kw) is with_sparse
    assert report.installed_layers == ("attn",)
    assert report.quantized_layers == ("attn",)
    assert bool(report.sparse_layers) is with_sparse
    assert report.cascade_disabled is True
    assert runner.cascade_attn_enabled is False
    assert not hasattr(unrelated, "q_bmm_quantizer")


def test_nvfp4_validation_of_all_layers_precedes_mutation(monkeypatch):
    valid = _bare_attention()
    invalid = _bare_attention()
    invalid.attn_type = "encoder"
    invalid.head_size_v = 32
    valid_impl = valid.impl
    invalid_impl = invalid.impl
    runner = _model_runner(nn.ModuleDict({"valid": valid, "invalid": invalid}))
    configured = []
    monkeypatch.setattr(vllm_runtime, "_load_quant_plugin", lambda: _fake_quant_plugin(configured))

    with pytest.raises(NotImplementedError) as exc:
        vllm_runtime.install_vllm_nvfp4_attention(runner)

    assert "invalid: attn_type" in str(exc.value)
    assert "head_size_v" in str(exc.value)
    assert configured == []
    assert valid.impl is valid_impl
    assert invalid.impl is invalid_impl
    assert not hasattr(valid, "_query_quant_in_kernel")
    assert runner.cascade_attn_enabled is True


def test_apply_failure_does_not_leave_configured_layer_on_native_impl(monkeypatch):
    first = _bare_attention()
    second = _bare_attention()
    first_impl = first.impl
    second_impl = second.impl
    runner = _model_runner(nn.ModuleDict({"first": first, "second": second}))
    configured = []
    quant_plugin = _fake_quant_plugin(configured)
    configure = quant_plugin.configure_vllm_nvfp4_attention_quantizers

    def fail_on_second(module, **kwargs):
        if module is second:
            raise RuntimeError("configuration failed")
        return configure(module, **kwargs)

    quant_plugin.configure_vllm_nvfp4_attention_quantizers = fail_on_second
    monkeypatch.setattr(vllm_runtime, "_load_quant_plugin", lambda: quant_plugin)

    with pytest.raises(RuntimeError, match="configuration failed"):
        vllm_runtime.install_vllm_nvfp4_attention(runner)

    assert type(first.impl) is ModelOptSparseAttentionImpl
    assert second.impl is second_impl
    assert first.impl is not first_impl
    assert first._query_quant_in_kernel is True
    assert first._value_quant_in_kernel is True
    assert not hasattr(second, "_query_quant_in_kernel")
    assert runner.cascade_attn_enabled is False


@pytest.mark.parametrize(
    ("mode", "rejected"),
    [(CUDAGraphMode.FULL, True), (CUDAGraphMode.FULL_AND_PIECEWISE, False)],
)
def test_nvfp4_mixed_cudagraph_policy(monkeypatch, mode, rejected):
    attention = _bare_attention()
    original_impl = attention.impl
    runner = _model_runner(nn.ModuleDict({"attn": attention}))
    runner.vllm_config.compilation_config.cudagraph_mode = mode
    configured = []
    monkeypatch.setattr(vllm_runtime, "_load_quant_plugin", lambda: _fake_quant_plugin(configured))

    if rejected:
        with pytest.raises(NotImplementedError, match="FULL mixed-batch"):
            vllm_runtime.install_vllm_nvfp4_attention(runner)
        assert configured == []
        assert attention.impl is original_impl
    else:
        report = vllm_runtime.install_vllm_nvfp4_attention(runner)
        assert report.installed_count == 1
        assert configured == [attention]


@pytest.mark.parametrize("impl_cls", [FlashAttentionImpl, FlashInferImpl])
def test_nvfp4_install_fp8_bmm2_uses_module_level_v(monkeypatch, impl_cls):
    """bmm2='fp8': P maps to the fp8 kernel mode; V is module-level (no kernel V)."""
    attention = _bare_attention(impl_cls)
    runner = _model_runner(nn.ModuleDict({"attn": attention}))
    monkeypatch.setattr(
        quant_plugin,
        "create_parallel_state",
        lambda: quant_plugin.ParallelState(data_parallel_group=None),
    )
    monkeypatch.setattr(
        vllm_runtime.attention_plugin, "patch_flashinfer_metadata_builder", lambda: True
    )

    report = vllm_runtime.install_vllm_nvfp4_attention(runner, p_format="fp8", v_format="fp8")

    assert report.installed_layers == ("attn",)
    # BMM2 quantizers configured FP8 per-tensor with fixed amax (F3)
    assert attention.p_bmm_quantizer.num_bits == (4, 3)
    assert float(attention.p_bmm_quantizer._amax) == 1.0
    assert attention.v_bmm_quantizer.num_bits == (4, 3)
    assert float(attention.v_bmm_quantizer._amax) == 448.0
    # BMM1 untouched (F1)
    assert attention.q_bmm_quantizer.is_nvfp4_dynamic
    assert attention.k_bmm_quantizer.is_nvfp4_dynamic
    # quant_kw: fp8 P reaches the kernel; V never does (module-level pre-cache-write)
    assert attention.impl.quant_kw["p_qdq"] == "fp8"
    assert attention.impl.quant_kw["p_qdq_amax"] == 1.0
    assert attention.impl.quant_kw["v_qdq"] is None
    assert attention.impl.quant_kw["v_qdq_amax"] is None
    # module forward owns V quant; Q stays in-kernel
    assert attention._query_quant_in_kernel is True
    assert attention._value_quant_in_kernel is False


def test_install_rejects_unknown_formats():
    with pytest.raises(ValueError, match="p_format must be"):
        vllm_runtime.install_vllm_nvfp4_attention(object(), p_format="int8")
    with pytest.raises(ValueError, match="v_format must be"):
        vllm_runtime.install_vllm_nvfp4_attention(object(), v_format="int8")
