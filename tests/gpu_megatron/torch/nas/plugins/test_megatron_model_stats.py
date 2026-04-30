# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for modelopt.torch.nas.plugins.megatron_model_stats.

Two test groups:
  - TestMcoreParamCountFormulas: pure arithmetic / no GPU needed.
  - Live model tests (_test_formula_matches_*): build a real MCore model on GPU,
    compare the analytical formula result against get_mcore_param_count().
"""

from types import SimpleNamespace

import pytest
from _test_utils.torch.megatron.models import (
    HAS_MAMBA,
    get_mcore_gpt_model,
    get_mcore_mamba_hybrid_model,
)

from modelopt.torch.nas.plugins.megatron_model_stats import (
    mcore_memory_footprint_mb,
    mcore_param_count,
)
from modelopt.torch.prune.plugins.mcore_minitron import get_mcore_param_count

# ---------------------------------------------------------------------------
# Small reference dimensions - easy to verify by hand
# ---------------------------------------------------------------------------

_H = 4  # hidden_size
_NH = 2  # num_attention_heads
_NKV = 1  # num_query_groups
_KV = 2  # kv_channels (== _H // _NH)
_FFN = 8  # ffn_hidden_size
_NE = 2  # num_moe_experts
_TOPK = 1  # moe_router_topk
_MOE_FFN = 8  # moe_ffn_hidden_size
_MNH = 2  # mamba_num_heads
_MDH = 2  # mamba_head_dim
_MNG = 2  # mamba_num_groups
_MDS = 2  # mamba_state_dim
_V = 10  # vocab_size

_BASE_CFG = SimpleNamespace(
    hidden_size=_H,
    num_layers=1,
    num_attention_heads=_NH,
    num_query_groups=_NKV,
    kv_channels=_KV,
    ffn_hidden_size=_FFN,
    num_moe_experts=_NE,
    moe_router_topk=_TOPK,
    moe_ffn_hidden_size=_MOE_FFN,
    moe_shared_expert_intermediate_size=None,
    moe_shared_expert_gate=False,
    mamba_num_heads=_MNH,
    mamba_head_dim=_MDH,
    mamba_num_groups=_MNG,
    mamba_state_dim=_MDS,
    gated_linear_unit=False,
    add_bias_linear=False,
    normalization="RMSNorm",
    qk_layernorm=False,
    attention_output_gate=False,
    moe_layer_freq=1,
)

# Pre-computed expected component sizes (all verified by hand):

_LN = _H  # single RMSNorm weight

# Embedding + final RMSNorm + output layer (untied)
_BASE_UNTIED = _V * _H + _LN + _H * _V  # 40 + 4 + 40 = 84

# Attention sublayer (*):
#   linear_qkv weight:  H * (NH + 2*NKV) * KV  +  input_layernorm: H
#   linear_proj weight: NH * KV * H
_QKV = _H * (_NH + 2 * _NKV) * _KV  # 4 * 4 * 2 = 32
_PROJ = _NH * _KV * _H  # 2 * 2 * 4 = 16
_ATTN = _QKV + _LN + _PROJ  # 32 + 4 + 16 = 52

# Dense MLP sublayer (-):
#   linear_fc1: H * FFN  +  pre_mlp_layernorm: H
#   linear_fc2: FFN * H
_DENSE_MLP = _H * _FFN + _LN + _FFN * _H  # 32 + 4 + 32 = 68

# MoE sublayer (E):
#   always-active: pre_mlp_layernorm + router weight (NE * H)  [not per-expert]
#   per routed expert: fc1 (H * MOE_FFN) + fc2 (MOE_FFN * H)
_MOE_ALWAYS = _LN + _NE * _H  # 4 + 8 = 12
_MOE_PER_EXP = _H * _MOE_FFN + _MOE_FFN * _H  # 32 + 32 = 64
_MOE_TOTAL = _MOE_ALWAYS + _NE * _MOE_PER_EXP  # 12 + 128 = 140
_MOE_ACTIVE = _MOE_ALWAYS + _TOPK * _MOE_PER_EXP  # 12 + 64 = 76

# Mamba sublayer (M):
#   in_proj:  H * (2*d_inner + 2*MNG*MDS + MNH)  +  input_layernorm: H
#   out_proj: d_inner * H
#   conv1d (depthwise):  conv_dim * d_conv  +  conv_dim  (weight + bias)
#   scalars: A_log + dt_bias + D  -> 3 * MNH
#   internal RMSNorm on d_inner: d_inner
_D_INNER = _MNH * _MDH  # 2 * 2 = 4
_IN_PROJ_OUT = 2 * _D_INNER + 2 * _MNG * _MDS + _MNH  # 8 + 8 + 2 = 18
_CONV_DIM = _D_INNER + 2 * _MNG * _MDS  # 4 + 8 = 12
_MAMBA = (
    _H * _IN_PROJ_OUT  # in_proj weight
    + _LN  # input_layernorm
    + _D_INNER * _H  # out_proj
    + _CONV_DIM * 4
    + _CONV_DIM  # conv weight + bias
    + 3 * _MNH  # scalars
    + _D_INNER  # internal RMSNorm
)  # 72 + 4 + 16 + 60 + 6 + 4 = 162


# ---------------------------------------------------------------------------
# Formula tests (no GPU required)
# ---------------------------------------------------------------------------


class TestMcoreParamCountFormulas:
    def test_single_attention_layer(self):
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="*")
        expected = _BASE_UNTIED + _ATTN
        assert total == expected
        assert active == expected  # attention has no MoE split

    def test_single_dense_mlp_layer(self):
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="-")
        expected = _BASE_UNTIED + _DENSE_MLP
        assert total == expected
        assert active == expected

    def test_single_moe_layer_total_and_active(self):
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="E")
        assert total == _BASE_UNTIED + _MOE_TOTAL
        assert active == _BASE_UNTIED + _MOE_ACTIVE
        assert active < total

    def test_single_mamba_layer(self):
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="M")
        expected = _BASE_UNTIED + _MAMBA
        assert total == expected
        assert active == expected

    def test_hybrid_pattern_is_sum_of_per_layer_costs(self):
        pattern = "MEM*E"
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern=pattern, num_layers=5)
        assert total == _BASE_UNTIED + 2 * _MAMBA + 2 * _MOE_TOTAL + _ATTN
        assert active == _BASE_UNTIED + 2 * _MAMBA + 2 * _MOE_ACTIVE + _ATTN

    def test_pipe_char_ignored(self):
        base = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="ME", num_layers=2)
        with_pipe = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="M|E", num_layers=2)
        assert base == with_pipe

    def test_mtp_separator_strips_suffix(self):
        # Everything from '/' onward is MTP and must be ignored.
        base = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="M")
        with_mtp = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="M/E*")
        assert base == with_mtp

    def test_tied_vocab_excludes_output_layer(self):
        untied, _ = mcore_param_count(
            _BASE_CFG, _V, share_embeddings_and_output_weights=False, hybrid_layer_pattern="M"
        )
        tied, _ = mcore_param_count(
            _BASE_CFG, _V, share_embeddings_and_output_weights=True, hybrid_layer_pattern="M"
        )
        assert untied - tied == _V * _H

    def test_moe_topk_equals_num_experts_gives_equal_total_active(self):
        total, active = mcore_param_count(
            _BASE_CFG, _V, hybrid_layer_pattern="E", moe_router_topk=_NE
        )
        assert total == active

    def test_empty_pattern_only_base_params(self):
        total, active = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="", num_layers=0)
        assert total == _BASE_UNTIED
        assert active == _BASE_UNTIED

    def test_layernorm_adds_bias_over_rmsnorm(self):
        # LayerNorm has weight + bias vs RMSNorm weight-only, so LayerNorm models are larger.
        rmsnorm, _ = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="*")
        layernorm, _ = mcore_param_count(
            _BASE_CFG, _V, hybrid_layer_pattern="*", normalization="LayerNorm"
        )
        assert layernorm > rmsnorm

    def test_pure_gpt_dense_scales_with_num_layers(self):
        # All-dense GPT (num_moe_experts=None): each layer = attn + dense MLP.
        total_1, _ = mcore_param_count(_BASE_CFG, _V, num_layers=1, num_moe_experts=None)
        total_2, _ = mcore_param_count(_BASE_CFG, _V, num_layers=2, num_moe_experts=None)
        assert total_2 - total_1 == _ATTN + _DENSE_MLP

    def test_pure_gpt_moe_layer_freq(self):
        # moe_layer_freq=2: layer indices 0, 2 are MoE; 1, 3 are dense.
        total, active = mcore_param_count(_BASE_CFG, _V, num_layers=4, moe_layer_freq=2)
        assert total == _BASE_UNTIED + 4 * _ATTN + 2 * _MOE_TOTAL + 2 * _DENSE_MLP
        assert active == _BASE_UNTIED + 4 * _ATTN + 2 * _MOE_ACTIVE + 2 * _DENSE_MLP


# ---------------------------------------------------------------------------
# Memory footprint tests (no GPU required)
# ---------------------------------------------------------------------------

_MB = 1024**2
_SEQ = 4  # sequence_length used in memory tests
_BSZ = 1  # batch_size used in memory tests

# Mamba state size (per layer, batch=1, dtype=2 bytes):
#   conv_state = batch * conv_dim * (d_conv - 1)  where d_conv=4
#   ssm_state  = batch * MNH * MDH * MDS
_MAMBA_CONV_DIM = _D_INNER + 2 * _MNG * _MDS  # same as _CONV_DIM = 12
_MAMBA_CONV_STATE = _BSZ * _MAMBA_CONV_DIM * (4 - 1)  # 1 * 12 * 3 = 36
_MAMBA_SSM_STATE = _BSZ * _MNH * _MDH * _MDS  # 1 * 2 * 2 * 2 = 8
_MAMBA_STATE_BYTES = (_MAMBA_CONV_STATE + _MAMBA_SSM_STATE) * 2  # * dtype_bytes=2 = 88


class TestMcoreMemoryFootprint:
    def _mem(self, pattern, num_layers=1, sequence_length=_SEQ, batch_size=_BSZ, **kw):
        return mcore_memory_footprint_mb(
            _BASE_CFG,
            _V,
            hybrid_layer_pattern=pattern,
            dtype_bytes=2,
            sequence_length=sequence_length,
            batch_size=batch_size,
            num_layers=num_layers,
            **kw,
        )

    def test_total_equals_sum_of_components(self):
        params_mb, kv_cache_mb, mamba_state_mb, total_mb = self._mem("M*", num_layers=2)
        assert total_mb == pytest.approx(params_mb + kv_cache_mb + mamba_state_mb)

    def test_params_mb_consistent_with_param_count(self):
        # params_mb must equal total_params * dtype_bytes / GB
        params_mb, _, _, _ = self._mem("*")
        total, _ = mcore_param_count(_BASE_CFG, _V, hybrid_layer_pattern="*")
        assert params_mb == pytest.approx(total * 2 / _MB)

    def test_dtype_bytes_scales_params_mb(self):
        params_mb2, _, _, _ = self._mem("*")
        params_mb4, _, _, _ = mcore_memory_footprint_mb(
            _BASE_CFG,
            _V,
            hybrid_layer_pattern="*",
            dtype_bytes=4,
            sequence_length=_SEQ,
            batch_size=_BSZ,
        )
        assert params_mb4 == pytest.approx(params_mb2 * 2)

    def test_kv_cache_mb_zero_for_pure_mamba(self):
        _, kv_cache_mb, _, _ = self._mem("M")
        assert kv_cache_mb == 0.0

    def test_mamba_state_mb_zero_for_pure_gpt(self):
        _, _, mamba_state_mb, _ = self._mem(None, num_layers=1)  # no hybrid pattern -> pure GPT
        assert mamba_state_mb == 0.0

    def test_kv_cache_mb_exact_for_single_attention_layer(self):
        # kv_per_layer = 2 * batch * seq * NKV * KV * dtype_bytes
        expected_bytes = 2 * _BSZ * _SEQ * _NKV * _KV * 2
        _, kv_cache_mb, _, _ = self._mem("*")
        assert kv_cache_mb == pytest.approx(expected_bytes / _MB)

    def test_kv_cache_scales_linearly_with_seq_and_batch(self):
        # KV cache is linear in both sequence_length and batch_size.
        _, base_kv, _, _ = self._mem("*")
        _, kv_seq4, _, _ = self._mem("*", sequence_length=_SEQ * 4)
        _, kv_bsz4, _, _ = self._mem("*", batch_size=4)
        assert kv_seq4 == pytest.approx(base_kv * 4)
        assert kv_bsz4 == pytest.approx(base_kv * 4)

    def test_kv_cache_dtype_bytes_independent_of_param_dtype(self):
        # kv_cache_dtype_bytes=4 doubles kv_cache_mb but not params_mb
        params_mb2, kv_cache_mb2, _, _ = self._mem("*")
        params_mb_kv4, kv_cache_mb_kv4, _, _ = mcore_memory_footprint_mb(
            _BASE_CFG,
            _V,
            hybrid_layer_pattern="*",
            dtype_bytes=2,
            kv_cache_dtype_bytes=4,
            sequence_length=_SEQ,
            batch_size=_BSZ,
        )
        assert kv_cache_mb_kv4 == pytest.approx(kv_cache_mb2 * 2)
        assert params_mb_kv4 == pytest.approx(params_mb2)

    def test_mamba_state_mb_exact(self):
        _, _, mamba_state_mb, _ = self._mem("M")
        assert mamba_state_mb == pytest.approx(_MAMBA_STATE_BYTES / _MB)

    def test_mamba_state_scales_with_num_mamba_layers(self):
        _, _, mamba_state1, _ = self._mem("M")
        _, _, mamba_state2, _ = self._mem("MM", num_layers=2)
        assert mamba_state2 == pytest.approx(mamba_state1 * 2)


# ---------------------------------------------------------------------------
# Live model tests: formula must match get_mcore_param_count() exactly
# ---------------------------------------------------------------------------


def _test_formula_matches_gpt_model(rank, size, parallelism):
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1 if parallelism != "tp" else size,
        pipeline_model_parallel_size=1 if parallelism != "pp" else size,
        initialize_megatron=True,
        num_layers=4,
        hidden_size=64,
        num_attention_heads=8,
        num_query_groups=4,
        ffn_hidden_size=128,
        vocab_size=128,
        normalization="RMSNorm",
        activation_func="swiglu",
        bf16=True,
    ).cuda()

    expected_total, expected_active = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
    )
    actual = get_mcore_param_count(model)

    assert expected_total == expected_active, "Non-MoE GPT: total must equal active"
    assert expected_total == actual, (
        f"Formula ({expected_total:,}) != live model ({actual:,}) for {parallelism}"
    )


@pytest.mark.parametrize("parallelism", ["tp", "pp"])
def test_formula_matches_gpt_model(dist_workers, parallelism, num_gpus):
    """Builds a real GPTModel and asserts the analytical formula matches the live count."""
    if num_gpus == 1 and parallelism != "tp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    dist_workers.run(_test_formula_matches_gpt_model, parallelism=parallelism)


def _test_formula_matches_gpt_moe_model(rank, size, parallelism):
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=1 if parallelism != "tp" else size,
        pipeline_model_parallel_size=1 if parallelism != "pp" else size,
        expert_model_parallel_size=1 if parallelism != "ep" else size,
        initialize_megatron=True,
        num_layers=4,
        hidden_size=64,
        num_attention_heads=8,
        ffn_hidden_size=128,
        moe_grouped_gemm=True,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=16,
        vocab_size=128,
        normalization="RMSNorm",
        activation_func="swiglu",
        moe_layer_freq=2,
        bf16=True,
    ).cuda()

    expected_total, expected_active = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
    )
    actual = get_mcore_param_count(model)

    assert expected_active < expected_total, "MoE model: active must be less than total"
    assert expected_total == actual, (
        f"Formula total ({expected_total:,}) != live model ({actual:,}) for {parallelism}"
    )


@pytest.mark.parametrize("parallelism", ["tp", "pp", "ep"])
def test_formula_total_matches_gpt_moe_model(dist_workers, parallelism, num_gpus):
    """Builds a GPTModel with MoE layers; formula total must match the live param count."""
    if num_gpus == 1 and parallelism != "tp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    dist_workers.run(_test_formula_matches_gpt_moe_model, parallelism=parallelism)


def _test_formula_matches_mamba_model(rank, size, parallelism):
    hidden_size = 64
    mamba_head_dim = 16
    mamba_num_heads = hidden_size // mamba_head_dim  # 4
    pattern = "ME*-"  # 4-layer hybrid

    model = get_mcore_mamba_hybrid_model(
        tensor_model_parallel_size=1 if parallelism != "tp" else size,
        pipeline_model_parallel_size=1 if parallelism != "pp" else size,
        expert_model_parallel_size=1 if parallelism != "ep" else size,
        initialize_megatron=True,
        num_layers=4,
        hidden_size=hidden_size,
        num_attention_heads=8,
        mamba_head_dim=mamba_head_dim,
        mamba_num_heads=mamba_num_heads,
        vocab_size=128,
        hybrid_override_pattern=pattern,
        moe_grouped_gemm=False,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=None,
        bf16=True,
    ).cuda()

    hybrid_layer_pattern = getattr(model, "hybrid_layer_pattern", pattern)
    expected_total, _ = mcore_param_count(
        model.config,
        model.vocab_size,
        model.share_embeddings_and_output_weights,
        hybrid_layer_pattern=hybrid_layer_pattern,
    )
    actual = get_mcore_param_count(model)

    assert expected_total == actual, (
        f"Formula ({expected_total:,}) != live model ({actual:,}) for {parallelism}"
    )


@pytest.mark.parametrize("parallelism", ["tp", "pp", "ep"])
def test_formula_matches_mamba_model(dist_workers, parallelism, num_gpus):
    """Builds a non-MoE hybrid MambaModel; formula must match the live param count."""
    if num_gpus == 1 and parallelism != "tp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    if not HAS_MAMBA:
        pytest.skip("Mamba not installed")
    dist_workers.run(_test_formula_matches_mamba_model, parallelism=parallelism)
