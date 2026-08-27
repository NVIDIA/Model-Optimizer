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

"""Tests for skip-softmax calibration through the vLLM adapters and installer."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flashinfer import FlashInferImpl

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_KERNEL_AVAILABLE
from modelopt.torch.quantization.plugins import vllm as quant_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm as attention_plugin
from modelopt.torch.sparsity.attention_sparsity.plugins import vllm_runtime
from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import (
    ModelOptSparseAttentionImpl,
    collect_calibration_counts,
    disable_calibration,
    enable_calibration,
    iter_sparse_impls,
)

TRIALS = [1e-3, 1e-1, 5e-1]


# ---------------------------------------------------------------------------
# Installer fakes (mirroring test_vllm_runtime.py)
# ---------------------------------------------------------------------------
def _bare_attention(impl_cls=FlashAttentionImpl):
    module = object.__new__(vllm_runtime._VLLM_ATTENTION)
    nn.Module.__init__(module)
    module.attn_type = "decoder"
    module.head_size = 64
    module.device = torch.device("cpu")
    module.dtype = torch.float16
    module.impl = object.__new__(impl_cls)
    module.impl.sinks = None
    return module


def _model_runner(model, *, sparse_metadata=None, cudagraph_mode=CUDAGraphMode.NONE):
    hf_config = SimpleNamespace(sparse_attention_config=sparse_metadata)
    model_config = SimpleNamespace(hf_config=hf_config, dtype=torch.float16)
    return SimpleNamespace(
        model=model,
        model_config=model_config,
        cascade_attn_enabled=True,
        vllm_config=SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                enable_dbo=False,
                use_ubatching=False,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
            kv_transfer_config=None,
            speculative_config=None,
        ),
    )


class TestCalibrationInstaller:
    def test_installs_adapters_without_enabling_measurement(self):
        first = _bare_attention()
        second = _bare_attention()
        runner = _model_runner(nn.ModuleDict({"a_attn": first, "b_attn": second}))

        report = vllm_runtime.install_vllm_skip_softmax_calibration(runner)

        assert report.installed_count == 2
        assert report.sparse_algorithm == "SKIP_SOFTMAX_CALIBRATION"
        assert report.cascade_disabled and runner.cascade_attn_enabled is False
        for module in (first, second):
            assert isinstance(module.impl, ModelOptSparseAttentionImpl)
            # Measurement starts only via enable_calibration, so warmup
            # launches after install are never recorded.
            assert not attention_plugin._calibration_active(module.impl)

    def test_rejects_non_eager_execution(self):
        runner = _model_runner(
            nn.ModuleDict({"attn": _bare_attention()}),
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
        )
        with pytest.raises(NotImplementedError, match="enforce_eager"):
            vllm_runtime.install_vllm_skip_softmax_calibration(runner)

    def test_rejects_active_attention_quantizers_atomically(self):
        quantized = _bare_attention()
        quantized.q_bmm_quantizer = SimpleNamespace(is_enabled=True)
        clean = _bare_attention()
        clean_impl = clean.impl
        runner = _model_runner(nn.ModuleDict({"q_attn": quantized, "c_attn": clean}))

        with pytest.raises(NotImplementedError, match="requires unquantized attention"):
            vllm_runtime.install_vllm_skip_softmax_calibration(runner)

        # Validation-before-mutation: the clean layer must not be touched either.
        assert clean.impl is clean_impl
        assert runner.cascade_attn_enabled is True

    def test_rejects_fp8_kv_cache(self):
        attention = _bare_attention()
        attention.kv_cache_dtype = "fp8"
        runner = _model_runner(nn.ModuleDict({"attn": attention}))
        with pytest.raises(NotImplementedError, match="FP8 KV cache"):
            vllm_runtime.install_vllm_skip_softmax_calibration(runner)

    def test_rejects_model_without_attention_layers(self):
        runner = _model_runner(nn.ModuleDict({}))
        with pytest.raises(NotImplementedError, match="no attention layers"):
            vllm_runtime.install_vllm_skip_softmax_calibration(runner)


class TestQuantSkipRejection:
    """Skip-softmax cannot be combined with attention quantization."""

    _CALIBRATED_META = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "threshold_scale_factor": {"prefill": {"a": 7.9, "b": 8.6}},
            }
        }
    }
    _NM_META = {
        "config_groups": {
            "group_0": {"algorithm": "sparse_softmax", "sparsity_n": 2, "sparsity_m": 4}
        }
    }

    def test_quantized_install_rejects_calibrated_skip(self):
        attention = _bare_attention()
        runner = _model_runner(
            nn.ModuleDict({"attn": attention}), sparse_metadata=self._CALIBRATED_META
        )
        with pytest.raises(
            NotImplementedError, match="cannot be combined with attention quantization"
        ):
            vllm_runtime.install_vllm_nvfp4_attention(runner)
        assert not isinstance(attention.impl, ModelOptSparseAttentionImpl)

    def test_sparse_only_install_rejects_skip_onto_quantized_layer(self):
        """Sparse-only installs must also refuse skip onto live quantizers."""
        attention = _bare_attention()
        attention.q_bmm_quantizer = SimpleNamespace(is_enabled=True)
        original_impl = attention.impl
        runner = _model_runner(
            nn.ModuleDict({"attn": attention}), sparse_metadata=self._CALIBRATED_META
        )
        with pytest.raises(
            NotImplementedError, match="cannot be combined with attention quantization"
        ):
            vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)
        assert attention.impl is original_impl

    def test_sparse_only_install_allows_skip_on_unquantized_layer(self):
        runner = _model_runner(
            nn.ModuleDict({"attn": _bare_attention()}), sparse_metadata=self._CALIBRATED_META
        )
        report = vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)
        assert report.installed_count == 1

    def test_quantized_plan_allows_nm_sparsity(self, monkeypatch):
        monkeypatch.setattr(
            quant_plugin,
            "_get_device_dtype",
            lambda module: (torch.device("cpu"), torch.float16),
        )
        runner = _model_runner(
            nn.ModuleDict({"attn": _bare_attention()}), sparse_metadata=self._NM_META
        )
        plan = vllm_runtime._plan_vllm_attention(runner, quantize=True, sparse_cfg="checkpoint")
        assert len(plan.layers) == 1
        assert plan.layers[0].sparse_kw.get("sparsity_n") == 2


class TestFlashInferLayout:
    def test_installer_accepts_flashinfer(self):
        attention = _bare_attention(FlashInferImpl)
        runner = _model_runner(nn.ModuleDict({"attn": attention}))
        report = vllm_runtime.install_vllm_skip_softmax_calibration(runner)
        assert report.installed_count == 1


class TestSparseOnlyGraphGuard:
    """Commit-contract: the calibrated-decode graph guard is not quantize-gated."""

    _CALIBRATED_META = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "threshold_scale_factor": {
                    "prefill": {"a": 7.9, "b": 8.6},
                    "decode": {"a": 0.12, "b": 9.8},
                },
            }
        }
    }

    def test_sparse_only_install_rejects_full_decode_graph(self):
        attention = _bare_attention()
        runner = _model_runner(
            nn.ModuleDict({"attn": attention}),
            sparse_metadata=self._CALIBRATED_META,
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        with pytest.raises(NotImplementedError, match="non-FULL CUDA graph"):
            vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)
        assert not isinstance(attention.impl, ModelOptSparseAttentionImpl)

    def test_sparse_only_install_allows_eager(self):
        attention = _bare_attention()
        runner = _model_runner(
            nn.ModuleDict({"attn": attention}),
            sparse_metadata=self._CALIBRATED_META,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        report = vllm_runtime.install_vllm_sparse_attention_from_checkpoint(runner)
        assert report.installed_count == 1


# ---------------------------------------------------------------------------
# Calibration forward through the FlashAttention adapter (GPU)
# ---------------------------------------------------------------------------
def _make_impl(num_heads, head_dim, num_kv_heads):
    return ModelOptSparseAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


def _paged_cache_for(seqs_kv, num_kv_heads, head_dim, page_size, device, dtype):
    """Scatter per-request contiguous K/V lists into a stacked paged cache."""
    blocks_per_seq = [(kv.shape[0] + page_size - 1) // page_size for kv, _ in seqs_kv]
    num_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)
    k_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(len(seqs_kv), max_blocks, device=device, dtype=torch.int32)
    g = 0
    for b, (k, v) in enumerate(seqs_kv):
        for blk in range(blocks_per_seq[b]):
            block_table[b, blk] = g
            ts, te = blk * page_size, min((blk + 1) * page_size, k.shape[0])
            k_cache[g, : te - ts] = k[ts:te]
            v_cache[g, : te - ts] = v[ts:te]
            g += 1
    return torch.stack([k_cache, v_cache], dim=0), block_table


def _sdpa_reference(q, k, v, is_causal):
    # [tokens, heads, dim] -> [1, heads, tokens, dim]
    qh, kh, vh = (t.transpose(0, 1).unsqueeze(0).float() for t in (q, k, v))
    kh = kh.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    vh = vh.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    if is_causal and q.shape[0] < k.shape[0]:
        # Suffix-causal mask for decode/chunked prefill shapes.
        mask = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool, device=q.device).tril(
            diagonal=k.shape[0] - q.shape[0]
        )
        out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=is_causal)
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


@pytest.mark.skipif(not TRITON_KERNEL_AVAILABLE, reason="Need CUDA + triton")
class TestCalibrationForward:
    def test_mixed_batch_records_phases_and_stays_dense(self):
        torch.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16
        num_heads, num_kv_heads, head_dim, page_size = 4, 2, 64, 16
        prefill_len, decode_ctx = 64, 48

        k0 = torch.randn(prefill_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v0 = torch.randn_like(k0)
        k1 = torch.randn(decode_ctx, num_kv_heads, head_dim, device=device, dtype=dtype)
        v1 = torch.randn_like(k1)
        q = torch.randn(prefill_len + 1, num_heads, head_dim, device=device, dtype=dtype)

        kv_cache, block_table = _paged_cache_for(
            [(k0, v0), (k1, v1)], num_kv_heads, head_dim, page_size, device, dtype
        )
        attn_metadata = SimpleNamespace(
            num_actual_tokens=prefill_len + 1,
            max_query_len=prefill_len,
            max_seq_len=max(prefill_len, decode_ctx),
            query_start_loc=torch.tensor(
                [0, prefill_len, prefill_len + 1], device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor([prefill_len, decode_ctx], device=device, dtype=torch.int32),
            block_table=block_table,
        )

        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        impl.sparse_kw = {}
        enable_calibration([impl], TRIALS)
        output = torch.empty_like(q)
        out = impl.forward(
            layer=None,
            query=q,
            key=q[:, :num_kv_heads],
            value=q[:, :num_kv_heads],
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )

        # Two records: one per request, phases decided per request.
        records = impl._calib_records
        assert [r["phase"] for r in records] == ["prefill", "decode"]
        assert [r["sample_length"] for r in records] == [prefill_len, decode_ctx]
        for record in records:
            assert len(record["total_tiles"]) == len(TRIALS)
            assert all(t > 0 for t in record["total_tiles"])
            assert all(0 <= s <= t for s, t in zip(record["skipped_tiles"], record["total_tiles"]))

        # Output is full dense attention (calibration never skips).
        ref_prefill = _sdpa_reference(q[:prefill_len], k0, v0, is_causal=True)
        ref_decode = _sdpa_reference(q[prefill_len:], k1, v1, is_causal=False)
        torch.testing.assert_close(out[:prefill_len], ref_prefill, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out[prefill_len:], ref_decode, rtol=2e-2, atol=2e-2)

    def test_collect_calibration_counts_sums_layers(self):
        class FakeModel(nn.Module):
            def __init__(self, impls):
                super().__init__()
                self._impls = impls
                self.layers = nn.ModuleList([nn.Identity() for _ in impls])
                for identity, impl in zip(self.layers, impls):
                    identity.impl = impl

        impls = [object.__new__(ModelOptSparseAttentionImpl) for _ in range(2)]
        enable_calibration(impls, TRIALS)
        for idx, impl in enumerate(impls):
            impl._calib_records = [
                {
                    "phase": "prefill",
                    "sample_length": 128,
                    "total_tiles": [4, 4, 4],
                    "skipped_tiles": [idx, idx + 1, idx + 2],
                }
            ]
        model = FakeModel(impls)
        assert len(list(iter_sparse_impls(model))) == 2
        disable_calibration(impls)

        counts = collect_calibration_counts(model)
        assert counts["prefill"] == [
            {"sample_length": 128, "total_tiles": [8, 8, 8], "skipped_tiles": [1, 3, 5]}
        ]

    def test_rejects_non_logical_cache_shape(self):
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        enable_calibration([impl], TRIALS)
        # A physical HND tensor must be exposed as a logical [blocks, page, heads, dim] view.
        kv_cache = torch.zeros(2, 1, num_kv_heads, 16, head_dim, dtype=torch.bfloat16)
        attn_metadata = SimpleNamespace(
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=8,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        q = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="logical KV-cache view"):
            impl.forward(
                layer=None,
                query=q,
                key=q[:, :num_kv_heads],
                value=q[:, :num_kv_heads],
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=torch.empty_like(q),
            )

    def test_rejects_non_16bit_cache(self):
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        impl = _make_impl(num_heads, head_dim, num_kv_heads)
        enable_calibration([impl], TRIALS)
        kv_cache = torch.zeros(2, 1, 16, num_kv_heads, head_dim, dtype=torch.uint8)
        attn_metadata = SimpleNamespace(
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=8,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        q = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="fp16/bf16 KV cache"):
            impl.forward(
                layer=None,
                query=q,
                key=q[:, :num_kv_heads],
                value=q[:, :num_kv_heads],
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=torch.empty_like(q),
            )


# ---------------------------------------------------------------------------
# FlashInfer adapter: cache write must precede the calibrate-kernel read
# ---------------------------------------------------------------------------
class TestFlashInferCalibrationOrdering:
    @pytest.mark.parametrize("layout", ["NHD", "HND"])
    def test_cache_write_happens_before_calibrate_read(self, monkeypatch, layout):
        calls = []
        monkeypatch.setattr(
            attention_plugin,
            "_maybe_update_flashinfer_cache",
            lambda *args, **kwargs: calls.append("cache_write"),
        )

        def fake_calibrate(q, *args, **kwargs):
            calls.append("calibrate")
            calls.append(kwargs["k_cache"].stride())
            counters = torch.zeros(len(TRIALS), 2, dtype=torch.int64)
            return torch.zeros_like(q), counters

        monkeypatch.setattr(attention_plugin, "attention_calibrate", fake_calibrate)

        num_heads, num_kv_heads, head_dim, page = 4, 2, 64, 16
        impl = SimpleNamespace(
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            _calibrate=True,
            _calib_threshold_trials=list(TRIALS),
            _calib_records=[],
        )
        shape = (3, 2, page, num_kv_heads, head_dim)
        kv_cache = torch.zeros(shape, dtype=torch.bfloat16)
        if layout == "HND":
            kv_cache = torch.zeros(
                shape[0], shape[1], shape[3], shape[2], shape[4], dtype=torch.bfloat16
            ).permute(0, 1, 3, 2, 4)
        attn_metadata = SimpleNamespace(
            _modelopt_block_table=torch.zeros(1, 1, dtype=torch.int32),
            _modelopt_seq_lens=torch.tensor([8], dtype=torch.int32),
            _modelopt_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            _modelopt_num_actual_tokens=1,
            _modelopt_max_query_len=1,
            _modelopt_max_seq_len=8,
            _modelopt_causal=False,
            slot_mapping=torch.zeros(1, dtype=torch.int64),
        )
        q = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16)

        out = attention_plugin._flashinfer_forward(
            impl,
            None,  # native_forward is unused on the calibration path
            None,  # layer
            q,
            q[:, :num_kv_heads],
            q[:, :num_kv_heads],
            kv_cache,
            attn_metadata,
            output=torch.empty_like(q),
        )

        assert calls == ["cache_write", "calibrate", kv_cache[:, 0].stride()]
        assert torch.isfinite(out).all()
        assert len(impl._calib_records) == 1
        assert impl._calib_records[0]["phase"] == "decode"
