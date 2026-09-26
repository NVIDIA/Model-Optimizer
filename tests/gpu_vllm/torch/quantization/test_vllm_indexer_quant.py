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

"""Fake quantization of the DSA lightning-indexer K cache in the vLLM plugin.

Covers the FP8 cache read-back/re-quantization shared by the fused indexer layouts and the wiring
of each layout adapter on stand-in modules, without booting an ``LLM`` (see
``test_vllm_dynamic_modules.py`` for the end-to-end DeepSeek-V3.2 run).
"""

from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins import vllm_indexer
from modelopt.torch.utils.distributed import ParallelState

HEAD_DIM = 128
BLOCK_SIZE = 8
NUM_BLOCKS = 4


def _write_fp8_indexer_cache(k: torch.Tensor, kv_cache: torch.Tensor, slots: list[int]) -> None:
    """Torch reference of vLLM's ``indexer_k_quant_and_cache`` (E4M3 rows, ue8m0 fp32 scales)."""
    flat = kv_cache.view(NUM_BLOCKS, -1)
    scale_base = BLOCK_SIZE * HEAD_DIM
    for row, slot in zip(k, slots):
        if slot < 0:
            continue
        block, pos = divmod(slot, BLOCK_SIZE)
        scale = vllm_indexer._indexer_k_ue8m0_scale(row.float().abs().max())
        values = (row.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        flat[block, pos * HEAD_DIM : (pos + 1) * HEAD_DIM] = values.view(torch.uint8)
        flat[block, scale_base + pos * 4 : scale_base + (pos + 1) * 4] = scale.reshape(1).view(
            torch.uint8
        )


def _read_fp8_indexer_cache(kv_cache: torch.Tensor, slot: int) -> torch.Tensor:
    flat = kv_cache.view(NUM_BLOCKS, -1)
    block, pos = divmod(slot, BLOCK_SIZE)
    values = flat[block, pos * HEAD_DIM : (pos + 1) * HEAD_DIM].view(torch.float8_e4m3fn).float()
    scale_base = BLOCK_SIZE * HEAD_DIM
    scale = (
        flat[block, scale_base + pos * 4 : scale_base + (pos + 1) * 4].clone().view(torch.float32)
    )
    return values * scale


def _fp8_quantizer(amax: float | None = 2.0, enabled: bool = True) -> TensorQuantizer:
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), enable=enabled))
    if amax is not None:
        quantizer.amax = torch.tensor(amax)
    return quantizer


@pytest.fixture
def parallel_state_stub(monkeypatch):
    monkeypatch.setattr(
        vllm_indexer,
        "create_parallel_state",
        lambda: ParallelState(data_parallel_group=None),
    )


@pytest.fixture
def written_cache():
    torch.manual_seed(0)
    kv_cache = torch.randint(0, 255, (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4), dtype=torch.uint8)
    slots = [9, -1, 17, 18, -1, 30, 3, -1]
    k = torch.randn(len(slots), HEAD_DIM, dtype=torch.bfloat16) * 3
    _write_fp8_indexer_cache(k, kv_cache, slots)
    return kv_cache, torch.tensor(slots, dtype=torch.int64)


def _assert_requantized(kv_cache_before, kv_cache_after, slot_mapping, quantizer):
    """Valid slots hold ``write(QDQ(read))``; every other byte of the cache is untouched."""
    expected = kv_cache_before.clone()
    valid_slots = [int(s) for s in slot_mapping if s >= 0]
    for slot in valid_slots:
        k = quantizer(_read_fp8_indexer_cache(kv_cache_before, slot))
        _write_fp8_indexer_cache(k[None], expected, [slot])
    assert torch.equal(kv_cache_after, expected)


@pytest.mark.parametrize(
    "quantizer",
    [_fp8_quantizer(amax=2.0), _fp8_quantizer(amax=None, enabled=False)],
    ids=["fp8_static_clipping", "disabled"],
)
def test_requantize_fp8_indexer_k_cache(written_cache, quantizer):
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()

    vllm_indexer._requantize_fp8_indexer_k_cache(
        kv_cache, slot_mapping, slot_mapping >= 0, quantizer
    )

    _assert_requantized(before, kv_cache, slot_mapping, quantizer)
    if not quantizer.is_enabled:  # re-storing may renormalize the scale but never the value
        for slot in slot_mapping[slot_mapping >= 0].tolist():
            assert torch.equal(
                _read_fp8_indexer_cache(kv_cache, slot), _read_fp8_indexer_cache(before, slot)
            )


def test_requantize_fp8_indexer_k_cache_honors_valid_mask(written_cache):
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()
    valid = torch.zeros_like(slot_mapping, dtype=torch.bool)
    valid[0] = True  # only slot 9 was written in this step

    vllm_indexer._requantize_fp8_indexer_k_cache(
        kv_cache, slot_mapping.to(torch.int32), valid, _fp8_quantizer(amax=2.0)
    )

    _assert_requantized(before, kv_cache, torch.tensor([9]), _fp8_quantizer(amax=2.0))


class _IndexerOp(torch.nn.Module):
    def forward(self, hidden_states, q_quant, k, weights):
        return k


class _NativeIndexerOpIndexer(torch.nn.Module):
    """Stand-in for ``deepseek_v2.Indexer``: forward ends in ``self.indexer_op(..., k, ...)``."""

    def __init__(self):
        super().__init__()
        self.indexer_op = _IndexerOp()

    def forward(self, hidden_states, qr, positions, rotary_emb):
        return self.indexer_op(hidden_states, qr, positions, rotary_emb)


class _TestIndexerOpIndexer(vllm_indexer._QuantVLLMIndexerOpIndexer, _NativeIndexerOpIndexer):
    pass


def test_indexer_op_prehook_quantizes_k(parallel_state_stub):
    indexer = _TestIndexerOpIndexer.convert(_NativeIndexerOpIndexer())
    quantizer = indexer.indexer_k_quantizer = _fp8_quantizer(amax=2.0)

    assert set(dict(indexer.named_children())) == {"indexer_op", "indexer_k_quantizer"}
    k = torch.full((2, HEAD_DIM), 5.0, dtype=torch.bfloat16)
    assert torch.equal(indexer(None, None, k, None), quantizer(k))  # positional k
    assert torch.equal(indexer.indexer_op(None, None, k=k, weights=None), quantizer(k))
    assert indexer.indexer_op(None, None, None, None) is None  # DeepSeek-V4 passes k=None
    quantizer.disable()
    assert indexer(None, None, k, None) is k


def test_indexer_shared_by_two_parents_is_converted_once(parallel_state_stub):
    """vLLM's MLA wrapper holds the attention's indexer too, so conversion reaches it twice."""
    QuantModuleRegistry.register({_NativeIndexerOpIndexer: "test_shared_indexer"})(
        vllm_indexer._QuantVLLMIndexerOpIndexer
    )
    try:
        indexer = _NativeIndexerOpIndexer()
        attention = torch.nn.Module()
        attention.indexer = indexer
        attention.mla_attn = torch.nn.Module()
        attention.mla_attn.indexer = indexer
        mtq.replace_quant_module(attention)
    finally:
        QuantModuleRegistry.unregister(_NativeIndexerOpIndexer)
    assert attention.indexer is attention.mla_attn.indexer
    assert isinstance(attention.indexer, vllm_indexer._QuantVLLMIndexerOpIndexer)


class _NativeV32Attention(torch.nn.Module):
    """Stand-in for the fused ``DeepseekV32Attention`` (vLLM >= 0.28)."""

    def __init__(self, indexer):
        super().__init__()
        self.indexer = indexer
        self.skip_topk = False
        self.layer_name = "layer0"
        self.calls = []

    def _sparse_indexer_and_attn(
        self,
        positions,
        q_c,
        q_nope,
        q_pe,
        index_q_fp8,
        index_k,
        index_weights_out,
        kv_c,
        k_pe,
        ql_nope,
        mqa_q,
        output,
    ):
        self.calls.append(index_k)


class _NativeV32AttentionOptIn(_NativeV32Attention):
    """The vLLM 0.27 opt-in fused layout: no ``index_k`` argument, the kernel always writes the cache."""

    def _sparse_indexer_and_attn(self, q_c, index_q_fp8, index_weights_out, ql_nope, mqa_q, output):
        self.calls.append(None)


class _TestV32Attention(vllm_indexer._QuantVLLMDeepseekV32Attention, _NativeV32Attention):
    pass


class _TestV32AttentionOptIn(vllm_indexer._QuantVLLMDeepseekV32Attention, _NativeV32AttentionOptIn):
    pass


def _v32_args(index_k=None):
    """Positional arguments of the fused-attention call with ``index_k`` at its real position (5)."""
    args = [None] * 12
    args[5] = index_k
    return args


_V32_ARGS = _v32_args()


_V32_INDEXER_CACHE = "layer0.indexer.k_cache"


def _v32_indexer(kv_cache, quantizer):
    return SimpleNamespace(
        indexer_k_quantizer=quantizer,
        k_cache=SimpleNamespace(kv_cache=kv_cache, prefix=_V32_INDEXER_CACHE),
    )


def _v32_slot_mapping(indexer_slots):
    """The indexer cache has its own slot mapping; the MLA layer's differs so a mix-up shows."""
    mla_slots = torch.where(indexer_slots >= 0, indexer_slots + 1, indexer_slots)
    return {"layer0": mla_slots, _V32_INDEXER_CACHE: indexer_slots}


def test_deepseek_v32_attention_requantizes_written_keys(
    parallel_state_stub, written_cache, monkeypatch
):
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()
    quantizer = _fp8_quantizer(amax=2.0)
    attention = _TestV32Attention.convert(_NativeV32Attention(_v32_indexer(kv_cache, quantizer)))
    context = SimpleNamespace(attn_metadata={}, slot_mapping=_v32_slot_mapping(slot_mapping))
    monkeypatch.setattr(vllm_indexer, "get_forward_context", lambda: context)

    # Fused path: fused_norm_rope already wrote the FP8 entries at the indexer cache's slots.
    attention._sparse_indexer_and_attn(*_V32_ARGS)
    _assert_requantized(before, kv_cache, slot_mapping, quantizer)
    assert attention.calls == [None]

    # Profiling run (no attn_metadata) writes nothing and must not touch the cache.
    snapshot = kv_cache.clone()
    context.attn_metadata = None
    attention._sparse_indexer_and_attn(*_V32_ARGS)
    assert torch.equal(kv_cache, snapshot)

    # Prefill context parallelism hands the bf16 key over instead; it is fake-quantized directly.
    k = torch.full((2, HEAD_DIM), 5.0, dtype=torch.bfloat16)
    attention._sparse_indexer_and_attn(*_v32_args(k))
    assert torch.equal(attention.calls[-1], quantizer(k))

    # Disabled quantizer and index-share layers leave everything alone.
    quantizer.disable()
    context.attn_metadata = {}
    attention._sparse_indexer_and_attn(*_v32_args(k))
    assert attention.calls[-1] is k
    quantizer.enable()
    attention.skip_topk = True
    attention._sparse_indexer_and_attn(*_v32_args(k))
    assert attention.calls[-1] is k


def test_deepseek_v32_opt_in_layout_always_reads_back(
    parallel_state_stub, written_cache, monkeypatch
):
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()
    quantizer = _fp8_quantizer(amax=2.0)
    attention = _TestV32AttentionOptIn.convert(
        _NativeV32AttentionOptIn(_v32_indexer(kv_cache, quantizer))
    )
    context = SimpleNamespace(attn_metadata={}, slot_mapping=_v32_slot_mapping(slot_mapping))
    monkeypatch.setattr(vllm_indexer, "get_forward_context", lambda: context)

    attention._sparse_indexer_and_attn(*([None] * 6))

    _assert_requantized(before, kv_cache, slot_mapping, quantizer)
    assert attention.calls == [None]


class _NativeV4Indexer(torch.nn.Module):
    """Stand-in for ``DeepseekV4Indexer``: the compressor kernel wrote the cache in forward."""

    def __init__(self, kv_cache):
        super().__init__()
        self.compress_ratio = 4
        self.use_fp4_kv = False
        self.k_cache = SimpleNamespace(prefix="idx", kv_cache=kv_cache)
        self.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="state"))

    def forward(
        self, hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb
    ):
        return "out"


class _TestV4Indexer(vllm_indexer._QuantVLLMDeepseekV4Indexer, _NativeV4Indexer):
    pass


def test_deepseek_v4_indexer_requantizes_compressed_keys(
    parallel_state_stub, written_cache, monkeypatch
):
    kv_cache, k_slots = written_cache
    before = kv_cache.clone()
    indexer = _TestV4Indexer.convert(_NativeV4Indexer(kv_cache))
    indexer.indexer_k_quantizer = _fp8_quantizer(amax=2.0)
    # Only tokens closing a compression group with valid indexer and compressor slots were written.
    positions = torch.tensor([3, 7, 11, 12, 15, 19, 23, 27])
    state_slots = torch.tensor([0, 1, 2, 3, 4, 5, -1, 7])
    metadata = {
        "idx": SimpleNamespace(slot_mapping=k_slots),
        "state": SimpleNamespace(slot_mapping=state_slots),
    }
    monkeypatch.setattr(
        vllm_indexer, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )

    assert indexer(None, None, None, None, positions, None) == "out"

    written = [9, 17, 30]  # slot 18 is mid-group, slot 3 has no compressor-state slot
    _assert_requantized(before, kv_cache, torch.tensor(written), indexer.indexer_k_quantizer)


def test_deepseek_v4_indexer_rejects_mxfp4_cache(parallel_state_stub, monkeypatch):
    indexer = _TestV4Indexer.convert(
        _NativeV4Indexer(torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4, dtype=torch.uint8))
    )
    indexer.indexer_k_quantizer = _fp8_quantizer(amax=2.0)
    indexer.use_fp4_kv = True
    monkeypatch.setattr(
        vllm_indexer, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    with pytest.raises(NotImplementedError, match="FP8 indexer cache"):
        indexer(None, None, None, None, torch.zeros(2, dtype=torch.int64), None)


class _NativeGlm5NextIndexer(torch.nn.Module):
    def __init__(self, kv_cache):
        super().__init__()
        self.k_cache = SimpleNamespace(kv_cache=kv_cache)


class _TestGlm5NextIndexer(vllm_indexer._QuantVLLMGlm5NextIndexer, _NativeGlm5NextIndexer):
    pass


def test_glm5next_kpool_cache_writers_requantize_written_pools(
    parallel_state_stub, written_cache, monkeypatch
):
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()
    kernel_calls = []

    def kpool_compress_and_write_cache(
        kv_cache, slot_k, slot_score, ape, loc, pool_size, head_dim=128, write_mask=None, **kwargs
    ):
        kernel_calls.append("prefill")

    def kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        slot_mapping,
        positions,
        pool_size,
        head_dim=128,
        round_scale=True,
    ):
        kernel_calls.append("decode")

    kpool_ops = SimpleNamespace(
        kpool_compress_and_write_cache=kpool_compress_and_write_cache,
        kpool_decode_update_and_maybe_write_cache_batched=(
            kpool_decode_update_and_maybe_write_cache_batched
        ),
    )
    monkeypatch.setattr(_TestGlm5NextIndexer, "kpool_ops", kpool_ops)
    indexer = _TestGlm5NextIndexer.convert(_NativeGlm5NextIndexer(kv_cache))
    vllm_indexer._install_kpool_cache_hooks(kpool_ops)  # a second indexer must not double-wrap
    quantizer = indexer.indexer_k_quantizer = _fp8_quantizer(amax=2.0)

    # Prefill: pools at ``loc`` masked by ``write_mask``; an unrelated cache is left alone.
    other_cache = torch.zeros_like(kv_cache)
    kpool_ops.kpool_compress_and_write_cache(other_cache, None, None, None, slot_mapping, 4)
    assert torch.equal(other_cache, torch.zeros_like(kv_cache))
    write_mask = torch.tensor([True, True, False, True, True, True, True, True])
    kpool_ops.kpool_compress_and_write_cache(
        kv_cache, None, None, None, slot_mapping, 4, write_mask=write_mask
    )
    _assert_requantized(before, kv_cache, torch.tensor([9, 18, 30, 3]), quantizer)

    # Decode: ``[num_requests, next_n]`` slots, written only where a pool completes.
    before = kv_cache.clone()
    dec_slots = torch.tensor([[17, 9], [30, -1]], dtype=torch.int32)
    dec_pos = torch.tensor([[7, 8], [11, 15]], dtype=torch.int32)
    kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache, None, None, None, None, None, dec_slots, dec_pos, 4
    )
    _assert_requantized(before, kv_cache, torch.tensor([17, 30]), quantizer)
    assert kernel_calls == ["prefill", "prefill", "decode"]
    assert getattr(kpool_ops.kpool_compress_and_write_cache, "_modelopt_indexer_k_wrapped", False)
