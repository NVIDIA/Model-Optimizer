# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLLMModel's max_model_len forwarding.

Guards against the bug where runtime_params.engine_args values (e.g.
max_model_len) were unpacked into VLLMModel.__init__'s **kwargs but then
silently dropped because AsyncEngineArgs was constructed with a hardcoded
subset of kwargs. PR #1564 review caught this; the fix passes max_model_len
explicitly.
"""

from unittest.mock import MagicMock, patch


def _make_minimal_kwargs(**overrides):
    base = {
        "speculative_algorithm": "NONE",
        "tokenizer_path": "/tmp/model",
        "tensor_parallel_size": 1,
        "moe_expert_parallel_size": 1,
        "prefix_cache": False,
        "async_scheduling": True,
    }
    base.update(overrides)
    return base


def _patch_vllm():
    """Return a context-manager stack that stubs out vllm imports."""
    import sys
    import types

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = MagicMock(return_value=MagicMock())

    fake_engine = types.ModuleType("vllm.engine")
    fake_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    fake_arg_utils.AsyncEngineArgs = MagicMock()

    fake_inputs = types.ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = MagicMock()

    fake_v1 = types.ModuleType("vllm.v1")
    fake_v1_engine = types.ModuleType("vllm.v1.engine")
    fake_async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    fake_async_llm.AsyncLLM = MagicMock()
    fake_async_llm.AsyncLLM.from_engine_args = MagicMock(return_value=MagicMock())

    mods = {
        "vllm": fake_vllm,
        "vllm.engine": fake_engine,
        "vllm.engine.arg_utils": fake_arg_utils,
        "vllm.inputs": fake_inputs,
        "vllm.v1": fake_v1,
        "vllm.v1.engine": fake_v1_engine,
        "vllm.v1.engine.async_llm": fake_async_llm,
    }
    for name, mod in mods.items():
        sys.modules[name] = mod
    return fake_arg_utils.AsyncEngineArgs


def test_max_model_len_forwarded_to_engine_args():
    """max_model_len from runtime_params.engine_args reaches AsyncEngineArgs."""
    engine_args_cls = _patch_vllm()
    import sys
    for key in list(sys.modules):
        if "specdec_bench.models.vllm" in key or key == "specdec_bench.models.vllm":
            del sys.modules[key]

    from specdec_bench.models.vllm import VLLMModel

    kwargs = _make_minimal_kwargs(max_model_len=40960)
    VLLMModel.__init__(MagicMock(), "/model", 4, {}, **kwargs)

    call_kwargs = engine_args_cls.call_args[1]
    assert call_kwargs.get("max_model_len") == 40960, (
        "max_model_len was not forwarded to AsyncEngineArgs"
    )


def test_max_model_len_absent_passes_none():
    """When max_model_len is not in kwargs, None is passed — vLLM uses its default."""
    engine_args_cls = _patch_vllm()
    import sys
    for key in list(sys.modules):
        if "specdec_bench.models.vllm" in key:
            del sys.modules[key]

    from specdec_bench.models.vllm import VLLMModel

    kwargs = _make_minimal_kwargs()
    VLLMModel.__init__(MagicMock(), "/model", 4, {}, **kwargs)

    call_kwargs = engine_args_cls.call_args[1]
    assert "max_model_len" in call_kwargs
    assert call_kwargs["max_model_len"] is None


def test_no_duplicate_keyword_argument():
    """prefix_cache / moe_expert_parallel_size are remapped — passing them plus
    their vllm names (enable_prefix_caching / enable_expert_parallel) must NOT
    raise 'got multiple values for keyword argument'."""
    engine_args_cls = _patch_vllm()
    import sys
    for key in list(sys.modules):
        if "specdec_bench.models.vllm" in key:
            del sys.modules[key]

    from specdec_bench.models.vllm import VLLMModel

    kwargs = _make_minimal_kwargs(prefix_cache=True, moe_expert_parallel_size=2)
    # Should not raise
    VLLMModel.__init__(MagicMock(), "/model", 4, {}, **kwargs)
    call_kwargs = engine_args_cls.call_args[1]
    assert call_kwargs.get("enable_prefix_caching") is True
    assert call_kwargs.get("enable_expert_parallel") is True
