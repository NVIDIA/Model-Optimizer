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

from modelopt.torch.quantization.plugins import vllm as vllm_plugin
from modelopt.torch.quantization.plugins.vllm import _QuantVLLMAttention


def _metadata(*, query_start_loc, seq_lens):
    return SimpleNamespace(
        query_start_loc=torch.tensor(query_start_loc),
        seq_lens=torch.tensor(seq_lens),
    )


def _apply_first_token_skip(metadata, num_tokens, first_n):
    attention = SimpleNamespace(kv_quant_skip_first_n=first_n)
    positions, _request_ids, valid = _QuantVLLMAttention._token_positions(metadata, num_tokens)
    values = torch.arange(num_tokens, dtype=torch.float32).view(-1, 1, 1)
    output = _QuantVLLMAttention._quantize_kv_after_first_tokens(
        attention, values, lambda tensor: tensor + 100, positions, valid
    )
    return positions, valid, output[:, 0, 0]


def test_token_positions_handle_multiple_requests_and_padding():
    metadata = _metadata(query_start_loc=[0, 2, 3], seq_lens=[5, 8])

    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 5)

    assert positions.tolist() == [3, 4, 7, 8, 9]
    assert request_ids.tolist() == [0, 0, 1, 1, 1]
    assert valid.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize(
    ("metadata", "num_tokens", "expected_positions", "expected_values"),
    [
        (_metadata(query_start_loc=[0, 4], seq_lens=[4]), 4, [0, 1, 2, 3], [0, 1, 102, 103]),
        (_metadata(query_start_loc=[0, 4], seq_lens=[8]), 4, [4, 5, 6, 7], [100, 101, 102, 103]),
        (_metadata(query_start_loc=[0, 1], seq_lens=[9]), 1, [8], [100]),
    ],
    ids=["prefill", "chunked-prefill", "decode"],
)
def test_first_token_skip_across_generation_phases(
    metadata, num_tokens, expected_positions, expected_values
):
    positions, _valid, values = _apply_first_token_skip(metadata, num_tokens, first_n=2)

    assert positions.tolist() == expected_positions
    assert values.tolist() == expected_values


def test_first_token_skip_handles_multi_request_batch():
    metadata = _metadata(query_start_loc=[0, 2, 3], seq_lens=[2, 5])

    positions, valid, values = _apply_first_token_skip(metadata, 3, first_n=1)

    assert positions.tolist() == [0, 1, 4]
    assert valid.tolist() == [True, True, True]
    assert values.tolist() == [0, 101, 102]


def test_first_token_skip_preserves_padding_rows():
    metadata = _metadata(query_start_loc=[0, 2], seq_lens=[2])

    positions, valid, values = _apply_first_token_skip(metadata, 4, first_n=1)

    assert positions.tolist() == [0, 1, 2, 3]
    assert valid.tolist() == [True, True, False, False]
    assert values.tolist() == [0, 101, 2, 3]


def test_first_token_skip_rejects_cuda_graph_runtime(monkeypatch):
    attention = SimpleNamespace(
        kv_cache_dtype="auto",
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        sliding_window=None,
        attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )
    forward_context = SimpleNamespace(
        attn_metadata={},
        cudagraph_runtime_mode=SimpleNamespace(name="FULL"),
    )
    monkeypatch.setattr(vllm_plugin, "_get_forward_context", lambda: forward_context)

    with pytest.raises(RuntimeError, match="requires eager"):
        _QuantVLLMAttention._get_boundary_skip_metadata(attention)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shared", "shared KV-cache"),
        ("sliding", "sliding-window"),
        ("cross", "decoder self-attention"),
        ("dbo", "vLLM DBO"),
        ("cascade", "cascade attention"),
        ("missing", "missing"),
    ],
)
def test_first_token_skip_rejects_unsupported_metadata(monkeypatch, case, message):
    attention = SimpleNamespace(
        kv_cache_dtype="auto",
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        sliding_window=None,
        attn_backend=SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
        layer_name="layer",
    )
    metadata = _metadata(query_start_loc=[0, 1], seq_lens=[1])
    metadata.use_cascade = False
    forward_context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        cudagraph_runtime_mode=SimpleNamespace(name="NONE"),
    )

    if case == "shared":
        attention.kv_sharing_target_layer_name = "layer.0"
    elif case == "sliding":
        attention.sliding_window = 4096
    elif case == "cross":
        attention.attn_type = "encoder_decoder"
    elif case == "dbo":
        forward_context.attn_metadata = []
    elif case == "cascade":
        metadata.use_cascade = True
    elif case == "missing":
        del metadata.seq_lens

    monkeypatch.setattr(vllm_plugin, "_get_forward_context", lambda: forward_context)

    with pytest.raises(RuntimeError, match=message):
        _QuantVLLMAttention._get_boundary_skip_metadata(attention)
