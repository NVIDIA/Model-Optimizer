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

from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory as calc
from modelopt.torch.puzzletron.block_config import AttentionConfig
from modelopt.torch.puzzletron.utils.misc import calculate_kv_dim


class _Descriptor:
    @staticmethod
    def get_language_model_config(config):
        return config


@pytest.mark.parametrize(
    ("n_embd", "n_head", "num_key_value_heads", "head_dim", "expected"),
    [
        (5120, 32, 8, 128, 2048),  # Mistral-Small-24B: head_dim < n_embd // n_head (160)
        (2880, 64, 8, 64, 1024),  # gpt-oss-20b: head_dim > n_embd // n_head (45)
        (4096, 32, 8, None, 2048),  # no head_dim: n_embd // n_head
    ],
)
def test_calculate_kv_dim(n_embd, n_head, num_key_value_heads, head_dim, expected):
    assert calculate_kv_dim(num_key_value_heads, n_head, n_embd, head_dim) == expected


@pytest.mark.parametrize(
    ("model_config", "head_dim"),
    [
        (SimpleNamespace(head_dim=128), 128),  # explicit head_dim (Mistral-Small-24B)
        (
            SimpleNamespace(hidden_size=5120, num_attention_heads=32),
            160,
        ),  # parent hidden_size // heads
    ],
)
@pytest.mark.parametrize("n_embd", [5120, 3840])  # unpruned, and hidden_size pruned to 3840
def test_attention_kv_cache_independent_of_pruned_hidden_size(
    monkeypatch, model_config, head_dim, n_embd
):
    # Pruning hidden_size leaves the head dimension unchanged, so the KV cache must not depend on n_embd.
    # The weight count is irrelevant here and would need a real model to compute.
    monkeypatch.setattr(calc, "calculate_subblock_params", lambda *args: 0)
    batch_size, seq_len, num_key_value_heads = 2, 16, 8
    mem = calc.calculate_attention_memory(
        AttentionConfig(num_key_value_heads=num_key_value_heads),
        model_config=model_config,
        descriptor=_Descriptor,
        batch_size=batch_size,
        prefill_seq_len=seq_len,
        generation_seq_len=0,
        prefill_queue_size=0,
        n_embd=n_embd,
        n_head=32,
        weights_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        allocate_prefill_query=False,
    )
    expected_bytes = batch_size * seq_len * 2 * num_key_value_heads * head_dim * 2
    assert mem["kv_cache_memory_mib"] == pytest.approx(expected_bytes / 2**20)
