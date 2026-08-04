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


def _metadata(*, query_start_loc, seq_lens, block_table=None, slot_mapping=None):
    num_actual_tokens = query_start_loc[-1]
    return SimpleNamespace(
        query_start_loc=torch.tensor(query_start_loc),
        seq_lens=torch.tensor(seq_lens),
        block_table=(
            torch.tensor(block_table)
            if block_table is not None
            else torch.empty((len(seq_lens), 0), dtype=torch.int32)
        ),
        slot_mapping=(
            torch.tensor(slot_mapping)
            if slot_mapping is not None
            else torch.arange(num_actual_tokens)
        ),
        num_actual_tokens=num_actual_tokens,
    )


def _apply_first_token_skip(metadata, num_tokens, first_n):
    attention = SimpleNamespace(
        kv_quant_skip_first_n=first_n,
        kv_quant_skip_last_n=0,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, num_tokens)
    values = torch.arange(num_tokens, dtype=torch.float32).view(-1, 1, 1)
    output = _QuantVLLMAttention._quantize_new_kv(
        attention,
        values,
        lambda tensor: tensor + 100,
        positions,
        request_ids,
        valid,
        metadata.seq_lens,
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


def test_first_and_last_token_skip_uses_call_boundary_lengths():
    metadata = _metadata(query_start_loc=[0, 4, 6], seq_lens=[7, 5])
    attention = SimpleNamespace(
        kv_quant_skip_first_n=1,
        kv_quant_skip_last_n=2,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 8)
    values = torch.arange(8, dtype=torch.float32).view(-1, 1, 1)

    output = _QuantVLLMAttention._quantize_new_kv(
        attention,
        values,
        lambda tensor: tensor + 100,
        positions,
        request_ids,
        valid,
        metadata.seq_lens,
    )

    assert positions.tolist() == [3, 4, 5, 6, 3, 4, 5, 6]
    assert valid.tolist() == [True, True, True, True, True, True, False, False]
    assert output[:, 0, 0].tolist() == [100, 101, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("hnd_strides", [False, True])
def test_last_token_skip_quantizes_only_newly_aged_cache_rows(hnd_strides):
    metadata = _metadata(
        query_start_loc=[0, 2, 3],
        seq_lens=[7, 5],
        block_table=[[2, 0, 3], [1, 4, 5]],
        slot_mapping=[10, 11, 2],
    )
    attention = SimpleNamespace(
        kv_quant_skip_first_n=1,
        kv_quant_skip_last_n=2,
        k_bmm_quantizer=lambda tensor: tensor + 100,
        v_bmm_quantizer=lambda tensor: tensor + 200,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 3)
    if hnd_strides:
        storage = torch.arange(2 * 6 * 1 * 2 * 1, dtype=torch.bfloat16).reshape(6, 1, 2, 2, 1)
        kv_cache = storage.permute(2, 0, 3, 1, 4)
        assert not kv_cache.is_contiguous()
    else:
        kv_cache = torch.arange(2 * 6 * 2, dtype=torch.bfloat16).reshape(2, 6, 2, 1, 1)
    before = kv_cache.clone()

    _QuantVLLMAttention._quantize_aged_kv_cache(
        attention,
        kv_cache,
        metadata,
        positions,
        request_ids,
        valid,
    )

    # Request 0 ages positions 3 and 4 into physical slots (block 0, offset 1)
    # and (block 3, offset 0). Request 1 ages position 2 into (block 4, offset 0).
    torch.testing.assert_close(kv_cache[0, 0, 1], before[0, 0, 1] + 100)
    torch.testing.assert_close(kv_cache[0, 3, 0], before[0, 3, 0] + 100)
    torch.testing.assert_close(kv_cache[0, 4, 0], before[0, 4, 0] + 100)
    torch.testing.assert_close(kv_cache[1, 0, 1], before[1, 0, 1] + 200)
    torch.testing.assert_close(kv_cache[1, 3, 0], before[1, 3, 0] + 200)
    torch.testing.assert_close(kv_cache[1, 4, 0], before[1, 4, 0] + 200)
    unchanged = torch.ones((6, 2), dtype=torch.bool)
    unchanged[0, 1] = False
    unchanged[3, 0] = False
    unchanged[4, 0] = False
    torch.testing.assert_close(kv_cache[0][unchanged], before[0][unchanged])
    torch.testing.assert_close(kv_cache[1][unchanged], before[1][unchanged])


def test_last_token_skip_allows_unallocated_cache_during_profiling():
    metadata = _metadata(query_start_loc=[0, 2], seq_lens=[2], block_table=[[]])
    attention = SimpleNamespace(
        kv_quant_skip_first_n=0,
        kv_quant_skip_last_n=2,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 2)

    _QuantVLLMAttention._quantize_aged_kv_cache(
        attention,
        torch.tensor([]),
        metadata,
        positions,
        request_ids,
        valid,
    )


def test_last_token_skip_rejects_unallocated_cache_with_history():
    metadata = _metadata(query_start_loc=[0, 1], seq_lens=[3], block_table=[[]])
    attention = SimpleNamespace(
        kv_quant_skip_first_n=0,
        kv_quant_skip_last_n=2,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 1)

    with pytest.raises(RuntimeError, match="history before cache allocation"):
        _QuantVLLMAttention._quantize_aged_kv_cache(
            attention,
            torch.tensor([]),
            metadata,
            positions,
            request_ids,
            valid,
        )


def test_last_token_skip_masked_placeholder_does_not_touch_unaged_cache():
    metadata = _metadata(
        query_start_loc=[0, 2],
        seq_lens=[3],
        block_table=[[2, 0]],
        slot_mapping=[0, 1],
    )
    attention = SimpleNamespace(
        kv_quant_skip_first_n=0,
        kv_quant_skip_last_n=2,
        k_bmm_quantizer=lambda tensor: tensor + 100,
        v_bmm_quantizer=lambda tensor: tensor + 200,
    )
    positions, request_ids, valid = _QuantVLLMAttention._token_positions(metadata, 2)
    kv_cache = torch.arange(2 * 3 * 2, dtype=torch.bfloat16).reshape(2, 3, 2, 1, 1)
    before = kv_cache.clone()

    _QuantVLLMAttention._quantize_aged_kv_cache(
        attention,
        kv_cache,
        metadata,
        positions,
        request_ids,
        valid,
    )

    torch.testing.assert_close(kv_cache[0, 2, 0], before[0, 2, 0] + 100)
    torch.testing.assert_close(kv_cache[1, 2, 0], before[1, 2, 0] + 200)
    torch.testing.assert_close(kv_cache[:, 0], before[:, 0])
    torch.testing.assert_close(kv_cache[:, 1], before[:, 1])


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


def test_boundary_skip_allows_vllm_profiling_without_attention_metadata(monkeypatch):
    class RecordingAttention(torch.nn.Module):
        def forward(self, query, key, value, *args, **kwargs):
            return query, key, value

    class QuantRecordingAttention(_QuantVLLMAttention, RecordingAttention):
        pass

    attention = object.__new__(QuantRecordingAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_cache_dtype = "auto"
    attention.attn_type = "decoder"
    attention.kv_sharing_target_layer_name = None
    attention.sliding_window = None
    attention.attn_backend = SimpleNamespace(get_name=lambda: "FLASH_ATTN")
    attention.kv_quant_skip_first_n = 1
    attention.kv_quant_skip_last_n = 1
    attention.q_bmm_quantizer = lambda tensor: tensor + 10
    attention.k_bmm_quantizer = lambda tensor: tensor + 20
    attention.v_bmm_quantizer = lambda tensor: tensor + 30
    attention._token_positions = lambda *_args: pytest.fail("profiling used token metadata")
    attention._quantize_aged_kv_cache = lambda *_args: pytest.fail("profiling aged KV cache")
    forward_context = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(vllm_plugin, "_get_forward_context", lambda: forward_context)

    inputs = tuple(torch.zeros(1) for _ in range(3))
    query, key, value = attention(*inputs)

    torch.testing.assert_close(query, torch.full((1,), 10.0))
    torch.testing.assert_close(key, torch.full((1,), 20.0))
    torch.testing.assert_close(value, torch.full((1,), 30.0))


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
        kv_quant_skip_last_n=0,
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
