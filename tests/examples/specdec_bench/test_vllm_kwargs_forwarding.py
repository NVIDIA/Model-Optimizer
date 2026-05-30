# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLLMModel's runtime_params engine_args forwarding.

The bug being guarded against: prior to the fix, `runtime_params.engine_args`
keys (e.g. `max_model_len`) were unpacked into `VLLMModel.__init__`'s `**kwargs`
but then silently dropped because the explicit `AsyncEngineArgs(...)` call only
read a hardcoded subset of kwargs. PR #1564 review caught this.

These tests don't require vllm to be installed — they exercise the pure-Python
kwarg-filter logic against a fake `AsyncEngineArgs` dataclass.
"""

import dataclasses

import pytest


# Minimal stub mimicking vllm.engine.arg_utils.AsyncEngineArgs's relevant fields.
# Real AsyncEngineArgs has ~80 fields; we only need the ones referenced by the
# forwarding logic.
@dataclasses.dataclass
class _FakeAsyncEngineArgs:
    model: str = ""
    tokenizer: str = ""
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enable_prefix_caching: bool = False
    speculative_config: object = None
    max_num_seqs: int = 256
    skip_tokenizer_init: bool = False
    async_scheduling: bool = True
    enforce_eager: bool = False
    max_model_len: int = 0
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9


def _compute_forwarded_engine_kwargs(kwargs, consumed_kwargs):
    """Mirror of the dict-comprehension in vllm.py:VLLMModel.__init__."""
    engine_arg_fields = {f.name for f in dataclasses.fields(_FakeAsyncEngineArgs)}
    return {
        k: v
        for k, v in kwargs.items()
        if k in engine_arg_fields and k not in consumed_kwargs
    }


# Match the constant defined in specdec_bench/models/vllm.py
_VLLM_CONSUMED_KWARGS = frozenset({
    "tokenizer_path",
    "trust_remote_code",
    "tensor_parallel_size",
    "moe_expert_parallel_size",
    "prefix_cache",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_num_draft_tokens",
    "draft_model_dir",
    "parallel_draft_block_sizes",
    "max_matching_ngram_size",
    "async_scheduling",
})


def test_max_model_len_is_forwarded():
    """`max_model_len` from runtime_params.engine_args reaches AsyncEngineArgs."""
    kwargs = {
        "tokenizer_path": "/foo",
        "tensor_parallel_size": 2,
        "speculative_algorithm": "MTP",
        "max_model_len": 40960,
    }
    forwarded = _compute_forwarded_engine_kwargs(kwargs, _VLLM_CONSUMED_KWARGS)
    assert forwarded == {"max_model_len": 40960}


def test_multiple_engine_args_are_forwarded():
    """Other AsyncEngineArgs fields beyond max_model_len pass through."""
    kwargs = {
        "max_model_len": 40960,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
    }
    forwarded = _compute_forwarded_engine_kwargs(kwargs, _VLLM_CONSUMED_KWARGS)
    assert forwarded == {
        "max_model_len": 40960,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
    }


def test_consumed_kwargs_are_not_double_forwarded():
    """Kwargs VLLMModel reads itself must NOT also flow through `**`, or
    AsyncEngineArgs would raise `got multiple values for keyword argument`."""
    # `tensor_parallel_size` IS a real AsyncEngineArgs field AND is consumed
    # explicitly by VLLMModel — exactly the dangerous case.
    kwargs = {
        "tensor_parallel_size": 4,
        "trust_remote_code": True,
        "max_model_len": 32768,
    }
    forwarded = _compute_forwarded_engine_kwargs(kwargs, _VLLM_CONSUMED_KWARGS)
    # Only max_model_len passes through; the other two are caller-consumed.
    assert forwarded == {"max_model_len": 32768}
    assert "tensor_parallel_size" not in forwarded
    assert "trust_remote_code" not in forwarded


def test_unknown_kwargs_are_dropped():
    """A kwarg that's neither consumed nor an AsyncEngineArgs field is dropped
    silently — matches the original behaviour for typos / outdated configs."""
    kwargs = {
        "max_model_len": 1024,
        "completely_made_up_field": "ignored",
    }
    forwarded = _compute_forwarded_engine_kwargs(kwargs, _VLLM_CONSUMED_KWARGS)
    assert forwarded == {"max_model_len": 1024}


def test_module_constant_matches_test_expectations():
    """Pin the consumed-kwargs set defined in the module against this test's
    copy, so adding a new consumed kwarg without updating tests fails loudly.

    Skipped in environments without torch/vllm — the module's imports pull in
    `from .base import Model` which transitively requires torch.
    """
    try:
        from specdec_bench.models import vllm as vllm_module
    except ImportError as e:
        pytest.skip(f"specdec_bench.models.vllm not importable: {e}")

    assert vllm_module._VLLM_CONSUMED_KWARGS == _VLLM_CONSUMED_KWARGS, (
        "Update _VLLM_CONSUMED_KWARGS in tests/examples/specdec_bench/"
        "test_vllm_kwargs_forwarding.py to match the module's definition."
    )
