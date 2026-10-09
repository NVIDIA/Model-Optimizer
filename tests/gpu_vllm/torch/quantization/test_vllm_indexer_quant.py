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

"""Fake quantization of the sparse-attention indexer query, K cache and scorer in the vLLM plugin.

Covers the FP8 cache read-back/re-quantization shared by the fused indexer layouts, the query
re-quantization, the scorer quantizers and the wiring of each layout adapter on stand-in modules,
without booting an ``LLM`` (see ``test_vllm_dynamic_modules.py`` for the end-to-end DeepSeek-V4 and
GLM-5.3-Flash runs).
"""

import functools
import inspect
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.nn import QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.plugins import vllm_indexer, vllm_layer_scope
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


# Stand-in for vLLM's ``static_forward_context``, where the indexer K cache registers its name.
_STATIC_FORWARD_CONTEXT: dict = {}


class _NativeGlm5NextIndexer(torch.nn.Module):
    def __init__(self, kv_cache=None, prefix="layers.0.indexer.k_cache"):
        super().__init__()
        self.k_cache = SimpleNamespace(kv_cache=kv_cache, prefix=prefix)
        _STATIC_FORWARD_CONTEXT[prefix] = self.k_cache

    def forward(self, q):
        # Like vLLM's forward, look the query kernel up in the module globals on every call.
        return fwht128_quant_fp8(q)


class _TestGlm5NextIndexer(vllm_indexer._QuantVLLMGlm5NextIndexer, _NativeGlm5NextIndexer):
    pass


def test_glm5next_kpool_cache_writers_requantize_written_pools(
    parallel_state_stub, written_cache, scorer_stand_ins, monkeypatch
):
    """In the indexer op of a layer with the quantizer, the writers re-quantize what they wrote."""
    kv_cache, slot_mapping = written_cache
    before = kv_cache.clone()
    kernel_calls, requantized = [], []
    requantize = vllm_indexer._requantize_fp8_indexer_k_cache
    monkeypatch.setattr(
        vllm_indexer,
        "_requantize_fp8_indexer_k_cache",
        lambda kv_cache, *args: requantized.append(kv_cache) or requantize(kv_cache, *args),
    )

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
    other_cache = torch.zeros_like(kv_cache)
    other = _TestGlm5NextIndexer.convert(
        _NativeGlm5NextIndexer(other_cache, prefix="layers.1.indexer.k_cache")
    )
    other.indexer_k_quantizer.disable()
    assert kpool_ops.kpool_compress_and_write_cache.__wrapped__ is kpool_compress_and_write_cache
    op = scorer_stand_ins.op_module.sparse_attn_indexer_kpool

    def write(layer, writer, *args, **kwargs):
        """``writer(*args, **kwargs)`` in the indexer op of ``layer``, like vLLM's op."""
        return op(None, f"layers.{layer}.indexer.k_cache", None, lambda: writer(*args, **kwargs))

    # No enabled quantizer: the kernel runs, also outside the indexer op, and nothing is
    # re-quantized.
    indexer.indexer_k_quantizer.disable()
    prefill = (kv_cache, None, None, None, slot_mapping, 4)
    write(0, kpool_ops.kpool_compress_and_write_cache, *prefill)
    kpool_ops.kpool_compress_and_write_cache(*prefill)
    assert not requantized

    quantizer = indexer.indexer_k_quantizer = _fp8_quantizer(amax=2.0)

    # A write outside the indexer op names no layer, and one to another layer's cache is misrouted:
    # both fail before the kernel runs. A layer without the quantizer re-quantizes nothing.
    with pytest.raises(RuntimeError, match="outside the indexer op"):
        kpool_ops.kpool_compress_and_write_cache(*prefill)
    prefill_other = (other_cache, None, None, None, slot_mapping, 4)
    with pytest.raises(RuntimeError, match="writes the K cache of another layer"):
        write(0, kpool_ops.kpool_compress_and_write_cache, *prefill_other)
    write(1, kpool_ops.kpool_compress_and_write_cache, *prefill_other)
    assert not requantized

    # Prefill: pools at ``loc`` masked by ``write_mask``.
    write_mask = torch.tensor([True, True, False, True, True, True, True, True])
    write(0, kpool_ops.kpool_compress_and_write_cache, *prefill, write_mask=write_mask)
    _assert_requantized(before, kv_cache, torch.tensor([9, 18, 30, 3]), quantizer)

    # Decode: ``[num_requests, next_n]`` slots, written only where a pool completes.
    before = kv_cache.clone()
    dec_slots = torch.tensor([[17, 9], [30, -1]], dtype=torch.int32)
    dec_pos = torch.tensor([[7, 8], [11, 15]], dtype=torch.int32)
    write(
        0,
        kpool_ops.kpool_decode_update_and_maybe_write_cache_batched,
        *(kv_cache, None, None, None, None, None, dec_slots, dec_pos, 4),
    )
    _assert_requantized(before, kv_cache, torch.tensor([17, 30]), quantizer)
    assert kernel_calls == ["prefill"] * 4 + ["decode"]
    assert len(requantized) == 2 and all(cache is kv_cache for cache in requantized)
    assert getattr(kpool_ops.kpool_compress_and_write_cache, "_modelopt_layer_callee", False)


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


# Indexer scorer quantizers (GLM-5.3-Flash). DeepGEMM and the indexer op are stand-ins: the
# scorer numerics need a DeepGEMM build that provides them, and the op a running engine.

NUM_HEADS = 32
NUMERICS = {
    "version": 1,
    "mma_accum": ("fp32", "fp16"),
    "post_relu": ("none", "fp32", "bf16", "fp16", "fp16_dynamic", "mxint8"),
    "reduce": ("fp32", "fp16"),
}
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the scorer quantizers run on the GPU"
)


class _FakeDeepGemm:
    """DeepGEMM's scorers and vLLM's wrappers of them, returning the preset ``scores``.

    Keys outside each row's ``valid`` range hold garbage, as DeepGEMM leaves them with
    ``clean_logits=False``. Every call records its weights, keyword arguments and logits.
    """

    def __init__(self):
        self.calls = []
        self.scores = self.valid = None
        self.module = SimpleNamespace(
            MQA_LOGITS_NUMERICS=NUMERICS,
            fp8_fp4_mqa_logits=self._mqa_logits,
            fp8_fp4_paged_mqa_logits=self._paged_mqa_logits,
        )
        self.utils = SimpleNamespace(
            fp8_fp4_mqa_logits=self._vllm_mqa_logits,
            fp8_fp4_paged_mqa_logits=self._vllm_paged_mqa_logits,
            _import_deep_gemm=lambda: self.module,
            _lazy_init=lambda: None,
        )

    def _logits(self, weights, block_table=None, **kwargs):
        rows, keys = self.scores.shape
        dtype = kwargs.get("logits_dtype", torch.float32)
        garbage = torch.where(torch.arange(keys, device="cuda") % 2 == 0, float("nan"), 1e30)
        # A view of a buffer with padded rows, like DeepGEMM's.
        logits = torch.empty(rows, keys + 9, dtype=dtype, device="cuda")[:, :keys]
        logits.copy_(torch.where(self.valid, self.scores, garbage))
        self.calls.append(
            SimpleNamespace(
                weights=weights, kwargs=kwargs, block_table=block_table, logits=logits.clone()
            )
        )
        return logits

    def _mqa_logits(self, q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, **kwargs):
        return self._logits(weights, **kwargs)

    def _paged_mqa_logits(
        self, q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, **kw
    ):
        return self._logits(weights, block_table=block_table, **kw)

    def _vllm_mqa_logits(self, q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
        return self._mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
        )

    def _vllm_paged_mqa_logits(
        self,
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits,
        indices=None,
    ):
        kwargs = {} if indices is None else {"indices": indices}
        return self._paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
            **kwargs,
        )


def _nan_to_minus_inf(logits):
    """The scorer maps NaN, which vLLM's prefill top-k mishandles, to -inf; +-inf stay."""
    return logits.nan_to_num(nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))


def _eager_break(fn):
    """Stand-in for vLLM's ``eager_break_during_capture``: graph replays call ``fn`` itself."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def _indexer_op(hidden_states, k_cache_prefix, kv_cache, run_scorer):
    """Stand-in for GLM-5.3-Flash's indexer op, which looks its DeepGEMM scorer up per call."""
    return run_scorer()


@pytest.fixture(autouse=True)
def scorer_stand_ins(monkeypatch):
    """Converted GLM-5.3-Flash indexers wrap these instead of the installed vLLM's modules.

    The indexer op finds its layer by name in a stand-in for vLLM's forward context.
    """
    _STATIC_FORWARD_CONTEXT.clear()
    monkeypatch.setattr(
        vllm_layer_scope,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers=_STATIC_FORWARD_CONTEXT),
    )
    fake = _FakeDeepGemm()
    op_module = SimpleNamespace(
        sparse_attn_indexer_kpool=_eager_break(_indexer_op),
        eager_break_during_capture=_eager_break,
    )
    monkeypatch.setattr(_TestGlm5NextIndexer, "op_module", op_module)
    monkeypatch.setattr(_TestGlm5NextIndexer, "deep_gemm_utils", fake.utils)
    return SimpleNamespace(fake=fake, op_module=op_module)


@pytest.fixture
def glm(parallel_state_stub, scorer_stand_ins, monkeypatch):
    """Converts GLM-5.3-Flash indexers (q and K quantizers off) and runs their indexer op."""
    kpool_ops = SimpleNamespace(
        kpool_compress_and_write_cache=lambda kv_cache, loc: None,
        kpool_decode_update_and_maybe_write_cache_batched=lambda kv_cache, slot_mapping: None,
    )
    monkeypatch.setattr(_TestGlm5NextIndexer, "kpool_ops", kpool_ops)
    monkeypatch.setattr(vllm_indexer, "_glm5next_indexers", weakref.WeakSet())

    def convert(layer=0):
        native = _NativeGlm5NextIndexer(prefix=f"layers.{layer}.indexer.k_cache")
        indexer = _TestGlm5NextIndexer.convert(native)
        indexer.indexer_q_quantizer.disable()
        indexer.indexer_k_quantizer.disable()
        return indexer

    def score(run_scorer, layer=0):
        op = scorer_stand_ins.op_module.sparse_attn_indexer_kpool
        return op(None, f"layers.{layer}.indexer.k_cache", None, run_scorer)

    return SimpleNamespace(fake=scorer_stand_ins.fake, convert=convert, score=score)


def _scorer_call(fake, paged, num_heads=NUM_HEADS, q_scale=None, block_tables=None, indices=None):
    """Valid keys and the scorer call of the indexer op, on prefill or decode rows."""
    if paged:  # rows score keys [0, context length) of their request's pages
        context_lens = torch.tensor([[33, 64], [1, 100]], dtype=torch.int32, device="cuda")
        start, end, num_keys = torch.zeros_like(context_lens).reshape(-1), context_lens, 128
    else:  # rows of several requests score their ranges [ks, ke) of one gathered key buffer
        start = torch.tensor([0, 0, 37, 37, 5], dtype=torch.int32, device="cuda")
        end, num_keys = torch.tensor([70, 31, 133, 37, 140], dtype=torch.int32, device="cuda"), 140
    end = end.reshape(-1)
    col = torch.arange(num_keys, device="cuda")
    fake.valid = (col >= start[:, None]) & (col < end[:, None])
    fake.scores = torch.randn(fake.valid.shape, device="cuda") * 3
    weights = torch.randn(len(start), num_heads, device="cuda")
    q = (None, q_scale)
    utils = fake.utils
    if paged:
        if block_tables is None:
            block_tables = torch.zeros(2, 4, dtype=torch.int32, device="cuda")

        def run_scorer():
            return utils.fp8_fp4_paged_mqa_logits(
                q, None, weights, context_lens, block_tables, None, num_keys, False, indices
            )
    else:

        def run_scorer():
            return utils.fp8_fp4_mqa_logits(q, None, weights, start, end, clean_logits=False)

    return SimpleNamespace(run=run_scorer, weights=weights, start=start, end=end)


def _kwargs_quantizer(**scorer_kwargs):
    """``indexer_scorer_kwargs_quantizer`` that passes ``scorer_kwargs`` to the scorer kernel."""
    return TensorQuantizer(
        QuantizerAttributeConfig(backend="scorer_kwargs", backend_extra_args=scorer_kwargs)
    )


def test_scorer_kwargs_quantizer_passes_its_backend_extra_args(glm):
    indexer = glm.convert()
    scorer_kwargs = {"mma_accum": "fp16", "post_relu": "bf16"}
    cfg = {"backend": "scorer_kwargs", "backend_extra_args": scorer_kwargs}
    set_quantizer_by_cfg(
        indexer, [{"quantizer_name": "*indexer_scorer_kwargs_quantizer", "cfg": cfg}]
    )
    assert indexer._scorer_kwargs() == scorer_kwargs

    indexer.indexer_scorer_kwargs_quantizer.disable_quant()  # calibration runs the stock kernel
    assert indexer._scorer_kwargs() == {}

    # Stock DeepGEMM's own logits_dtype, by its name in a config.
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(logits_dtype="bfloat16")
    assert indexer._scorer_kwargs() == {"logits_dtype": torch.bfloat16}


def test_registered_quantizers_turn_their_config_into_scorer_kwargs(glm, monkeypatch):
    """E.g. a plugin's quantizer for numerics options of a DeepGEMM build."""

    def post_relu(quantizer):
        if quantizer.num_bits != (8, 7):
            raise ValueError("BF16 only")
        return {"post_relu": "bf16"}

    registered = dict(vllm_indexer._SCORER_KWARGS_QUANTIZERS)
    monkeypatch.setattr(vllm_indexer, "_SCORER_KWARGS_QUANTIZERS", registered)
    vllm_indexer.register_indexer_scorer_kwargs_quantizer("indexer_test_quantizer", post_relu)
    indexer = glm.convert()
    assert not indexer.indexer_test_quantizer.is_enabled
    indexer.indexer_test_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(8, 7), backend="scorer_kwargs")
    )
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(mma_accum="fp16")
    assert indexer._scorer_kwargs() == {"mma_accum": "fp16", "post_relu": "bf16"}

    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(post_relu="fp16")
    with pytest.raises(ValueError, match=r"indexer_test_quantizer sets the scorer's post_relu to"):
        indexer.modelopt_post_restore()
    indexer.indexer_scorer_kwargs_quantizer.disable()
    indexer.indexer_test_quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(5, 10), backend="scorer_kwargs")
    )
    with pytest.raises(ValueError, match="BF16 only"):
        indexer.modelopt_post_restore()


@pytest.mark.parametrize(
    ("quantizer", "match"),
    [
        (
            lambda: TensorQuantizer(QuantizerAttributeConfig(num_bits=8)),
            "needs backend 'scorer_kwargs', got None",
        ),
        (lambda: _kwargs_quantizer(logits_dtype="float16"), "float32 or bfloat16"),
        (
            lambda: SequentialQuantizer(_kwargs_quantizer(), _kwargs_quantizer()),
            "a single format",
        ),
    ],
    ids=["backend", "logits_dtype", "list"],
)
def test_scorer_kwargs_quantizer_rejects_unsupported_config(glm, quantizer, match):
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = quantizer()
    with pytest.raises(ValueError, match=match):
        indexer.modelopt_post_restore()


def test_glm5next_forward_validates_scorer_quantizers(glm):
    """Bad configs and a DeepGEMM without the arguments fail in any forward, also in calibration.

    The scorer itself only runs for sequences longer than index_topk.
    """
    indexer = glm.convert()
    q = torch.randn(2, HEAD_DIM)
    quantizer = indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(post_relu="mxint8")
    quantizer.disable_quant()  # calibrating: the scorer would run the stock kernel
    indexer(q)
    del glm.fake.module.MQA_LOGITS_NUMERICS  # a DeepGEMM without the post_relu argument
    with pytest.raises(RuntimeError, match=r"\{'post_relu': 'mxint8'\} in MQA_LOGITS_NUMERICS"):
        indexer(q)
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(logits_dtype="bfloat16")
    indexer(q)  # stock DeepGEMM's own argument
    glm.fake.module.MQA_LOGITS_NUMERICS = {**NUMERICS, "post_relu": ("none",)}
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(post_relu="mxint8")
    with pytest.raises(RuntimeError, match=r"\{'post_relu': 'mxint8'\}"):
        indexer(q)
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(logits_dtype="float16")
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        indexer(q)


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["prefill", "decode"])
def test_scorer_passes_through_without_enabled_quantizer(glm, paged):
    indexer = glm.convert()
    call = _scorer_call(glm.fake, paged)
    logits = glm.score(call.run)
    assert glm.fake.calls[-1].weights is call.weights
    assert set(glm.fake.calls[-1].kwargs) <= {"clean_logits", "indices"}
    assert torch.equal(logits.view(torch.int32), glm.fake.calls[-1].logits.view(torch.int32))

    # Outside an indexer op there is no calling layer, whatever its quantizers.
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(mma_accum="fp16")
    logits = call.run()
    assert set(glm.fake.calls[-1].kwargs) <= {"clean_logits", "indices"}
    assert torch.equal(logits.view(torch.int32), glm.fake.calls[-1].logits.view(torch.int32))


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["prefill", "decode"])
def test_bf16_reduction_widens_scores_to_fp32(glm, paged):
    """DeepGEMM's BF16 head reduction takes BF16 weights and writes BF16 scores."""
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(logits_dtype="bfloat16")
    del glm.fake.module.MQA_LOGITS_NUMERICS  # stock DeepGEMM has the BF16 reduction
    call = _scorer_call(glm.fake, paged)
    logits = glm.score(call.run)

    kernel = glm.fake.calls[-1]
    assert kernel.kwargs["logits_dtype"] == torch.bfloat16
    assert torch.equal(kernel.weights, call.weights.to(torch.bfloat16))
    # vLLM's top-k reads FP32 scores with the strides of the kernel's output.
    assert logits.dtype == torch.float32
    assert logits.shape == kernel.logits.shape and logits.stride() == (logits.shape[1] + 9, 1)
    assert torch.equal(logits, _nan_to_minus_inf(kernel.logits.float()))


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["prefill", "decode"])
def test_scorer_maps_nan_scores_to_minus_inf(glm, paged):
    """vLLM's prefill top-k returns out-of-range indices for rows with NaN; it handles +-inf.

    E.g. FP16 accumulation in the scorer overflows to +-inf, and the head sum turns those into NaN.
    """
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(mma_accum="fp16")
    call = _scorer_call(glm.fake, paged)
    special = torch.tensor([float("nan"), float("inf"), float("-inf")], device="cuda")
    col = torch.arange(glm.fake.scores.shape[1], device="cuda")
    picked = glm.fake.valid & (col % 7 == 0)
    glm.fake.scores = torch.where(picked, special[col % 3], glm.fake.scores)
    logits = glm.score(call.run)

    assert glm.fake.calls[-1].kwargs["mma_accum"] == "fp16"
    assert not logits.isnan().any()
    assert (logits[picked & glm.fake.scores.isnan()] == float("-inf")).all()
    assert torch.equal(logits, _nan_to_minus_inf(glm.fake.calls[-1].logits))  # nothing else


@requires_cuda
def test_indexer_op_routes_scorer_to_calling_layer(glm, scorer_stand_ins):
    first, second = glm.convert(0), glm.convert(1)
    op = scorer_stand_ins.op_module.sparse_attn_indexer_kpool
    # Wrapped once, under the breakable-cudagraph decorator, whose replays call op.__wrapped__.
    assert op.__wrapped__.__wrapped__ is _indexer_op
    first.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(mma_accum="fp16")
    callers = []
    call = _scorer_call(glm.fake, paged=False)

    def run_scorer():
        callers.append(vllm_layer_scope.layer_owner())
        return call.run()

    for layer, quantized in ((0, True), (1, False)):
        op(None, f"layers.{layer}.indexer.k_cache", None, run_scorer)
        assert ("mma_accum" in glm.fake.calls[-1].kwargs) == quantized
    # A graph replay, with the layer name wrapped like vLLM's ``LayerName``.
    op.__wrapped__(None, SimpleNamespace(value="layers.0.indexer.k_cache"), None, run_scorer)
    # The op of a layer that no indexer owns, e.g. a draft model's: the scorer runs unchanged.
    op(None, "draft.layers.0.indexer.k_cache", None, run_scorer)
    assert "mma_accum" not in glm.fake.calls[-1].kwargs
    assert callers == [first, second, first, None]
    assert vllm_layer_scope.layer_owner() is None
    assert second._enabled_scorer_quantizers() == []


@requires_cuda
def test_scorer_kwargs_reach_deep_gemm(glm):
    indexer = glm.convert()
    numerics = {"mma_accum": "fp16", "post_relu": "mxint8"}
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(**numerics)

    glm.score(_scorer_call(glm.fake, paged=False).run)
    assert glm.fake.calls[-1].kwargs == {"clean_logits": False, **numerics}

    # Like vLLM's wrapper: pass the request indices, and a block table with a unit last stride.
    block_tables = torch.arange(8, dtype=torch.int32, device="cuda").reshape(4, 2).t()
    indices = torch.arange(4, dtype=torch.int32, device="cuda")
    glm.score(_scorer_call(glm.fake, True, block_tables=block_tables, indices=indices).run)
    kernel = glm.fake.calls[-1]
    assert kernel.kwargs == {"clean_logits": False, "indices": indices, **numerics}
    assert kernel.block_table.stride(-1) == 1 and torch.equal(kernel.block_table, block_tables)


@requires_cuda
def test_scorer_rejects_undeclared_kwargs(glm):
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(post_relu="mxint8")
    run = _scorer_call(glm.fake, paged=False).run
    del glm.fake.module.MQA_LOGITS_NUMERICS  # a DeepGEMM without the post_relu argument
    with pytest.raises(RuntimeError, match="does not declare the indexer scorer keyword arguments"):
        glm.score(run)
    glm.fake.module.MQA_LOGITS_NUMERICS = {**NUMERICS, "post_relu": ("none",)}
    with pytest.raises(RuntimeError, match=r"\{'post_relu': 'mxint8'\}"):
        glm.score(run)


@requires_cuda
def test_scorer_wraps_vllm_deep_gemm(glm, monkeypatch):
    """The wrappers bind vLLM's own scorer signatures and call the DeepGEMM it imports."""
    deep_gemm_utils = pytest.importorskip("vllm.utils.deep_gemm")
    paged_scorer = getattr(deep_gemm_utils, "fp8_fp4_paged_mqa_logits", None)
    if paged_scorer is None or "indices" not in inspect.signature(paged_scorer).parameters:
        pytest.skip("needs vLLM's fp8_fp4 scorer wrappers with indices (newer vLLM, e.g. 0.30)")
    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        monkeypatch.setattr(deep_gemm_utils, name, getattr(deep_gemm_utils, name))  # restore
    monkeypatch.setattr(deep_gemm_utils, "_import_deep_gemm", lambda: glm.fake.module)
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(_TestGlm5NextIndexer, "deep_gemm_utils", deep_gemm_utils)
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(mma_accum="fp16")
    glm.fake.utils = deep_gemm_utils  # the indexer op calls vLLM's wrappers

    indices = torch.arange(4, dtype=torch.int32, device="cuda")
    for paged in (False, True):
        glm.score(_scorer_call(glm.fake, paged, indices=indices if paged else None).run)
        expected = {"clean_logits": False, "mma_accum": "fp16"}
        assert glm.fake.calls[-1].kwargs == {**expected, **({"indices": indices} if paged else {})}


@requires_cuda
@pytest.mark.timeout(600)  # the JIT compiles the scorer
def test_in_kernel_numerics_in_the_real_scorer(glm, monkeypatch):
    """BF16 post-ReLU products in the scorer of the DeepGEMM that vLLM imports, if it has them.

    Runs with a DeepGEMM build with the scorer numerics first on the Python path.
    """
    deep_gemm_utils = pytest.importorskip("vllm.utils.deep_gemm")
    if not getattr(deep_gemm_utils._import_deep_gemm(), "MQA_LOGITS_NUMERICS", None):
        pytest.skip("vLLM's DeepGEMM has no scorer numerics (MQA_LOGITS_NUMERICS)")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the in-kernel scorer numerics need an SM100 GPU")
    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        monkeypatch.setattr(deep_gemm_utils, name, getattr(deep_gemm_utils, name))  # restore
    monkeypatch.setattr(_TestGlm5NextIndexer, "deep_gemm_utils", deep_gemm_utils)
    indexer = glm.convert()
    indexer.indexer_scorer_kwargs_quantizer = _kwargs_quantizer(post_relu="bf16")

    # Small integer codes keep the dots exact and up to 1152, which BF16 rounds to 8 significant
    # bits; signed powers of two as weights keep the head sum exact as well.
    torch.manual_seed(0)
    num_q, num_kv = 64, 256
    q = torch.randint(1, 4, (num_q, NUM_HEADS, HEAD_DIM), device="cuda").float()
    q[:, ::4] *= -1  # heads with negative dots, which the ReLU zeroes
    k = torch.randint(1, 4, (num_kv, HEAD_DIM), device="cuda").float()
    sign = torch.randint(0, 2, (num_q, NUM_HEADS), device="cuda") * 2 - 1
    weights = (sign * 2.0 ** torch.randint(-2, 2, (num_q, NUM_HEADS), device="cuda")).float()
    ks = torch.zeros(num_q, dtype=torch.int32, device="cuda")
    ke = torch.full_like(ks, num_kv)
    dots = torch.einsum("mhd,nd->mhn", q.double(), k.double()).relu()

    def scores(layer):
        def run_scorer():  # the indexer op calls vLLM's wrapper
            kv = (k.to(torch.float8_e4m3fn), torch.ones(num_kv, device="cuda"))
            q_fp8 = (q.to(torch.float8_e4m3fn), None)
            return deep_gemm_utils.fp8_fp4_mqa_logits(q_fp8, kv, weights, ks, ke, False)

        return glm.score(run_scorer, layer)[:, :num_kv].clone()

    def reference(post_relu):
        return torch.einsum("mh,mhn->mn", weights.double(), post_relu(dots)).float()

    stock = scores(layer=1)  # a layer without scorer quantizers
    assert torch.equal(stock, reference(lambda x: x))
    rounded = scores(layer=0)
    assert torch.equal(rounded, reference(lambda x: x.bfloat16().double()))
    assert not torch.equal(rounded, stock)


@pytest.mark.parametrize("quantizer_name", list(vllm_indexer._SCORER_KWARGS_QUANTIZERS))
def test_deepseek_v4_indexer_rejects_scorer_quantizers(parallel_state_stub, quantizer_name):
    indexer = _TestV4Indexer.convert(_NativeV4Indexer())
    indexer.indexer_k_quantizer.disable()
    indexer.indexer_q_quantizer.disable()
    setattr(indexer, quantizer_name, _fp8_quantizer(amax=2.0))
    with pytest.raises(NotImplementedError, match=quantizer_name):
        indexer(None, None, None, None, None, None)
