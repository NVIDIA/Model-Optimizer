# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR0917

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.block_config import AttentionConfig
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory import (
    calculate_additive_metrics,
    calculate_attention_memory,
)
from modelopt.torch.puzzletron.utils.misc import calculate_kv_dim


class DummyDescriptor:
    @staticmethod
    def get_language_model_config(config):
        return config


@pytest.mark.parametrize(
    ("name", "n_embd", "n_head", "num_kv_heads", "head_dim", "expected_kv_dim"),
    [
        # (a) Explicit head_dim smaller than inferred: Mistral Small 24B (128 < 160)
        ("Mistral-Small-24B", 5120, 32, 8, 128, 2048),
        # (b) Explicit head_dim larger than inferred: gpt-oss 20B (64 > 45)
        ("gpt-oss-20b", 2880, 64, 8, 64, 1024),
        # (c) Fallback when head_dim is None: inferred from n_embd // n_head
        ("Llama-fallback", 4096, 32, 8, None, 2048),
    ],
)
def test_calculate_kv_dim_arithmetic(name, n_embd, n_head, num_kv_heads, head_dim, expected_kv_dim):
    """Test calculate_kv_dim respects explicit head_dim and falls back to n_embd // n_head."""
    kv_dim = calculate_kv_dim(num_kv_heads, n_head, n_embd, head_dim=head_dim)
    assert kv_dim == expected_kv_dim


def test_calculate_attention_memory_with_model_head_dim():
    """Verify calculate_attention_memory uses model_config.head_dim over n_embd // n_head."""
    # Mistral Small 24B configuration: n_embd=5120, n_head=32, num_kv_heads=8, head_dim=128
    attention_config = AttentionConfig(num_query_heads=32, num_kv_heads=8)
    model_config = SimpleNamespace(head_dim=128)

    batch_size = 2
    prefill_seq_len = 10
    generation_seq_len = 0
    seq_len = prefill_seq_len + generation_seq_len
    dtype = torch.bfloat16
    dtype_bytes = 2

    # Expected: kv_dim = 2 * 8 * 128 = 2048. kv_cache_size = 2 * 10 * 2048 = 40960 elements.
    expected_kv_cache_bytes = batch_size * seq_len * (2 * 8 * 128) * dtype_bytes
    expected_kv_cache_mib = expected_kv_cache_bytes / 2**20

    mem = calculate_attention_memory(
        attention_config,
        model_config=model_config,
        descriptor=DummyDescriptor,
        batch_size=batch_size,
        prefill_seq_len=prefill_seq_len,
        generation_seq_len=generation_seq_len,
        n_embd=5120,
        n_head=32,
        weights_dtype=dtype,
        kv_cache_dtype=dtype,
        num_params=0,
    )

    assert mem["kv_cache_memory_mib"] == pytest.approx(expected_kv_cache_mib)


def test_calculate_additive_metrics_with_model_head_dim():
    """Verify calculate_additive_metrics uses model_config.head_dim for kv_cache_bytes_per_token."""
    # gpt-oss 20B configuration: n_embd=2880, n_head=64, num_kv_heads=8, head_dim=64
    attention_config = AttentionConfig(num_query_heads=64, num_kv_heads=8)
    model_config = SimpleNamespace(head_dim=64)

    metrics = calculate_additive_metrics(
        attention_config,
        model_config=model_config,
        descriptor=DummyDescriptor,
        batch_size=1,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=2880,
        n_head=64,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=0,
        active_params=0,
    )

    # Expected: kv_dim = 2 * 8 * 64 = 1024. Bytes per token = 1024 * 2 = 2048.
    expected_bytes_per_token = (2 * 8 * 64) * 2
    assert metrics["kv_cache_bytes_per_token"] == expected_bytes_per_token


def test_effective_head_dim_precedence():
    """Verify precedence: qk_head_dim > model_config.head_dim > (n_embd // n_head)."""
    # 1. When qk_head_dim is set, it overrides model_config.head_dim and fallback
    subblock_1 = AttentionConfig(num_query_heads=32, num_kv_heads=8, qk_head_dim=64)
    model_config_1 = SimpleNamespace(head_dim=128)
    m1 = calculate_additive_metrics(
        subblock_1,
        model_config=model_config_1,
        descriptor=DummyDescriptor,
        batch_size=1,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=5120,
        n_head=32,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=0,
        active_params=0,
    )
    assert m1["kv_cache_bytes_per_token"] == (2 * 8 * 64) * 2

    # 2. When qk_head_dim is None, model_config.head_dim overrides fallback
    subblock_2 = AttentionConfig(num_query_heads=32, num_kv_heads=8, qk_head_dim=None)
    model_config_2 = SimpleNamespace(head_dim=128)
    m2 = calculate_additive_metrics(
        subblock_2,
        model_config=model_config_2,
        descriptor=DummyDescriptor,
        batch_size=1,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=5120,
        n_head=32,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=0,
        active_params=0,
    )
    assert m2["kv_cache_bytes_per_token"] == (2 * 8 * 128) * 2

    # 3. When neither is set, falls back to n_embd // n_head = 5120 // 32 = 160
    subblock_3 = AttentionConfig(num_query_heads=32, num_kv_heads=8, qk_head_dim=None)
    model_config_3 = SimpleNamespace()
    m3 = calculate_additive_metrics(
        subblock_3,
        model_config=model_config_3,
        descriptor=DummyDescriptor,
        batch_size=1,
        prefill_seq_len=4,
        generation_seq_len=1,
        n_embd=5120,
        n_head=32,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        num_params=0,
        active_params=0,
    )
    assert m3["kv_cache_bytes_per_token"] == (2 * 8 * 160) * 2
