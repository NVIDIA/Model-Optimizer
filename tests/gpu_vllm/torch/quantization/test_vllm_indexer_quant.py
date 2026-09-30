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

"""Fake quantization of the sparse-attention indexer query and K cache in the vLLM plugin.

Covers the FP8 cache read-back/re-quantization shared by the fused indexer layouts, the query
re-quantization and the wiring of each layout adapter on stand-in modules, without booting an
``LLM`` (see ``test_vllm_dynamic_modules.py`` for the end-to-end DeepSeek-V4 and GLM-5.3-Flash
runs).
"""

import sys
import weakref
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
        scale = vllm_indexer._indexer_ue8m0_scale(row.float().abs().max())
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


class _NativeSharedIndexer(torch.nn.Module):
    """Stand-in for an indexer that both the attention and vLLM's MLA wrapper hold."""

    def forward(self, x):
        return x


def test_indexer_shared_by_two_parents_is_converted_once(parallel_state_stub):
    """vLLM's MLA wrapper holds the attention's indexer too, so conversion reaches it twice."""
    QuantModuleRegistry.register({_NativeSharedIndexer: "test_shared_indexer"})(
        vllm_indexer._QuantVLLMIndexerBase
    )
    try:
        indexer = _NativeSharedIndexer()
        attention = torch.nn.Module()
        attention.indexer = indexer
        attention.mla_attn = torch.nn.Module()
        attention.mla_attn.indexer = indexer
        mtq.replace_quant_module(attention)
    finally:
        QuantModuleRegistry.unregister(_NativeSharedIndexer)
    assert attention.indexer is attention.mla_attn.indexer
    assert isinstance(attention.indexer, vllm_indexer._QuantVLLMIndexerBase)


class _NativeV4Indexer(torch.nn.Module):
    """Stand-in for ``DeepseekV4Indexer``: in forward, the compressor kernel wrote the cache and
    the query kernel returns ``outputs`` (FP8 query, no scale, weights with the scale folded in).
    """

    def __init__(self, kv_cache=None):
        super().__init__()
        self.n_head = 4
        self.softmax_scale = HEAD_DIM**-0.5
        self.compress_ratio = 4
        self.use_fp4_kv = False
        self.k_cache = SimpleNamespace(prefix="idx", kv_cache=kv_cache)
        self.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="state"))
        self.outputs = (None, None, None)  # the short-context shortcut computes no query

    def forward(
        self, hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb
    ):
        return self.outputs


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

    assert indexer(None, None, None, None, positions, None) == (None, None, None)

    written = [9, 17, 30]  # slot 18 is mid-group, slot 3 has no compressor-state slot
    _assert_requantized(before, kv_cache, torch.tensor(written), indexer.indexer_k_quantizer)


def _fused_indexer_q_quant(q, indexer_weights, softmax_scale, head_scale):
    """Torch reference of the FP8 path of vLLM's ``fused_indexer_q_rope_quant`` without RoPE."""
    scale = vllm_indexer._indexer_ue8m0_scale(q.float().abs().amax(dim=-1))
    q_fp8 = (q.float() / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q_fp8, indexer_weights.float() * scale * softmax_scale * head_scale


def test_deepseek_v4_indexer_requantizes_query(parallel_state_stub):
    torch.manual_seed(0)
    indexer = _TestV4Indexer.convert(_NativeV4Indexer())
    indexer.indexer_k_quantizer.disable()
    quantizer = indexer.indexer_q_quantizer = _fp8_quantizer(amax=2.0)
    n_head, head_scale = indexer.n_head, indexer.n_head**-0.5
    # Rows spanning several power-of-two scales; the query kernel folds each into the weights.
    q = torch.randn(3, n_head, HEAD_DIM) * torch.tensor([0.01, 1.0, 30.0])[:, None, None]
    indexer_weights = torch.randn(3, n_head, dtype=torch.bfloat16)
    indexer_weights[1, 2] = 0  # a head that does not score
    q_fp8, weights = _fused_indexer_q_quant(q, indexer_weights, indexer.softmax_scale, head_scale)
    indexer.outputs = (q_fp8, None, weights)

    new_q, q_scale, new_weights = indexer(None, None, None, indexer_weights, None, None)

    # The kernel's output for the fake-quantized dequantized query.
    old_scale = vllm_indexer._indexer_ue8m0_scale(q.abs().amax(dim=-1))
    expected_q, expected_weights = _fused_indexer_q_quant(
        quantizer(q_fp8.float() * old_scale[..., None]),
        indexer_weights,
        indexer.softmax_scale,
        head_scale,
    )
    live = indexer_weights != 0
    assert q_scale is None
    assert torch.equal(new_q.view(torch.uint8)[live], expected_q.view(torch.uint8)[live])
    assert torch.equal(new_weights[live], expected_weights[live])
    assert not new_q.float()[~live].any() and not new_weights[~live].any()

    indexer.outputs = (None, None, None)  # short context: no query to quantize
    assert indexer(None, None, None, indexer_weights, None, None) == (None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's query kernel needs a GPU")
def test_deepseek_v4_query_scale_matches_vllm_kernel(parallel_state_stub):
    """The query kernel folds its power-of-two scale into the weights as the plugin assumes."""
    fused_indexer_q = pytest.importorskip("vllm.models.deepseek_v4.common.ops.fused_indexer_q")
    torch.manual_seed(0)
    indexer = _TestV4Indexer.convert(_NativeV4Indexer())
    indexer.n_head = n_head = 64
    head_scale = n_head**-0.5
    q = torch.randn(5, n_head, HEAD_DIM, device="cuda")
    q = (q * torch.logspace(-2, 2, 5, device="cuda")[:, None, None]).to(torch.bfloat16)
    indexer_weights = torch.randn(5, n_head, device="cuda", dtype=torch.bfloat16)
    # cos 1 and sin 0: RoPE leaves the query unchanged, so the torch reference applies exactly.
    cos_sin_cache = torch.cat([torch.ones(8, 32), torch.zeros(8, 32)], dim=-1).cuda()
    q_fp8, weights = fused_indexer_q.fused_indexer_q_rope_quant(
        torch.arange(5, device="cuda"),
        q,
        cos_sin_cache,
        indexer_weights,
        indexer.softmax_scale,
        head_scale,
    )
    ref_q, ref_weights = _fused_indexer_q_quant(
        q, indexer_weights, indexer.softmax_scale, head_scale
    )
    assert torch.equal(q_fp8.view(torch.uint8), ref_q.view(torch.uint8))
    torch.testing.assert_close(weights, ref_weights, rtol=1e-6, atol=0)

    # With the quantizer disabled, re-quantizing keeps what the scores see: query times weight.
    indexer.indexer_q_quantizer.disable()
    new_q, new_weights = indexer._requantize_query(q_fp8, weights, indexer_weights)
    assert torch.equal(new_q.float() * new_weights[..., None], q_fp8.float() * weights[..., None])


@pytest.mark.parametrize("quantizer_name", ["indexer_k_quantizer", "indexer_q_quantizer"])
def test_deepseek_v4_indexer_rejects_mxfp4(parallel_state_stub, monkeypatch, quantizer_name):
    indexer = _TestV4Indexer.convert(
        _NativeV4Indexer(torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4, dtype=torch.uint8))
    )
    indexer.indexer_k_quantizer.disable()
    indexer.indexer_q_quantizer.disable()
    setattr(indexer, quantizer_name, _fp8_quantizer(amax=2.0))
    indexer.use_fp4_kv = True
    # MXFP4 query: packed E2M1 values and their ue8m0 block scales.
    indexer.outputs = (
        torch.zeros(2, 4, HEAD_DIM // 2, dtype=torch.uint8),
        torch.zeros(2, 4, dtype=torch.int32),
        torch.ones(2, 4),
    )
    monkeypatch.setattr(
        vllm_indexer, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    with pytest.raises(NotImplementedError, match="indexer_kv_dtype"):
        indexer(None, None, None, None, torch.zeros(2, dtype=torch.int64), None)


def fwht128_quant_fp8(q):
    """Stand-in for GLM-5.3-Flash's fused Hadamard + FP8 query kernel (rotation left out)."""
    scale = vllm_indexer._indexer_ue8m0_scale(q.float().abs().amax(dim=-1, keepdim=True))
    return (q.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


class _NativeGlm5NextIndexer(torch.nn.Module):
    def __init__(self, kv_cache=None):
        super().__init__()
        self.k_cache = SimpleNamespace(kv_cache=kv_cache)

    def forward(self, q):
        # Like vLLM's forward, look the query kernel up in the module globals on every call.
        return fwht128_quant_fp8(q)


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
    monkeypatch.setattr(vllm_indexer, "_glm5next_indexers", weakref.WeakSet())
    indexer = _TestGlm5NextIndexer.convert(_NativeGlm5NextIndexer(kv_cache))
    # A second layer's conversion must not wrap the writers again.
    other = _TestGlm5NextIndexer.convert(_NativeGlm5NextIndexer(torch.zeros_like(kv_cache)))
    other.indexer_k_quantizer.disable()
    assert kpool_ops.kpool_compress_and_write_cache.__wrapped__ is kpool_compress_and_write_cache

    # No enabled quantizer: the kernel runs and the wrapper returns before looking up the cache.
    indexer.indexer_k_quantizer.disable()
    with monkeypatch.context() as m:
        lookups = []
        m.setattr(vllm_indexer, "_glm5next_quantizer_for", lookups.append)
        kpool_ops.kpool_compress_and_write_cache(kv_cache, None, None, None, slot_mapping, 4)
    assert not lookups
    assert torch.equal(kv_cache, before)

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
    assert kernel_calls == ["prefill", "prefill", "prefill", "decode"]
    assert getattr(kpool_ops.kpool_compress_and_write_cache, "_modelopt_indexer_k_wrapped", False)


def test_glm5next_indexer_requantizes_query(parallel_state_stub, monkeypatch):
    kpool_ops = SimpleNamespace(
        kpool_compress_and_write_cache=lambda kv_cache, loc: None,
        kpool_decode_update_and_maybe_write_cache_batched=lambda kv_cache, slot_mapping: None,
    )
    monkeypatch.setattr(_TestGlm5NextIndexer, "kpool_ops", kpool_ops)
    monkeypatch.setattr(_TestGlm5NextIndexer, "indexer_module", sys.modules[__name__])
    monkeypatch.setattr(vllm_indexer, "_glm5next_indexers", weakref.WeakSet())
    indexer = _TestGlm5NextIndexer.convert(_NativeGlm5NextIndexer())
    indexer.indexer_k_quantizer.disable()
    torch.manual_seed(0)
    q = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16) * 3
    kernel = fwht128_quant_fp8
    ref_q, ref_scale = kernel(q)

    indexer.indexer_q_quantizer.disable()
    q_fp8, q_scale = indexer(q)
    assert torch.equal(q_fp8.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(q_scale, ref_scale)

    quantizer = indexer.indexer_q_quantizer = _fp8_quantizer(amax=2.0)
    q_fp8, q_scale = indexer(q)
    expected_q, expected_scale = kernel(quantizer(ref_q.float() * ref_scale))
    assert torch.equal(q_fp8.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(q_scale, expected_scale)
    assert fwht128_quant_fp8 is kernel  # the module global is restored

    monkeypatch.setattr(_TestGlm5NextIndexer, "indexer_module", SimpleNamespace())
    with pytest.raises(NotImplementedError, match="fwht128_quant_fp8"):
        indexer(q)
