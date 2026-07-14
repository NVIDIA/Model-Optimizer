# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.block_config import AttentionConfig
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory import (
    calculate_attention_memory,
)


@pytest.mark.parametrize(
    ("window", "cached_tokens"),
    [(512, 512), ("full", 1100), (None, 1100)],
)
def test_attention_memory_caps_kv_cache_only_for_finite_windows(
    monkeypatch, window, cached_tokens
):
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory."
        "calculate_subblock_params",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory."
        "calculate_kv_dim",
        lambda *args, **kwargs: 8,
    )

    result = calculate_attention_memory(
        AttentionConfig(
            num_kv_heads=2,
            num_query_heads=8,
            sliding_window_size=window,
        ),
        SimpleNamespace(),
        object,
        batch_size=2,
        prefill_seq_len=1000,
        generation_seq_len=100,
        n_embd=32,
        n_head=8,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
    )

    expected_bytes = cached_tokens * 2 * 8 * 2
    assert result["kv_cache_memory_mib"] == pytest.approx(expected_bytes / 2**20)
