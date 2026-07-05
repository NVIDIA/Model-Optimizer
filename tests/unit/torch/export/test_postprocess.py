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

"""Tests for the single-process export postprocess logic (TP/PP config splitting, padding,
lm_head quantization update, weight shape validation and tensor postprocessing).

The multi-rank merge paths (``_merge_model_configs_to_first_tp`` and
``_model_model_configs_to_first_pp``) require multiple distributed processes exchanging
tensors through SharedMemory/NFS and are not covered here.
"""

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.model_config import (
    LINEAR_COLUMN,
    LINEAR_GROUP,
    LINEAR_ROW,
    AttentionConfig,
    ConvConfig,
    DecoderLayerConfig,
    EmbeddingConfig,
    ExpertConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
    ModelConfig,
    RelativeAttentionTableConfig,
)
from modelopt.torch.export.postprocess import (
    _same_tensor,
    _shallow_copy_with_field_instantiation,
    _split_model_config_for_pp,
    _split_model_config_for_tp,
    check_weight_shape_valid,
    pad_embedding_lm_head,
    postprocess_model_config,
    postprocess_tensors,
    update_lm_head_quantization,
    view_as_float8_e4m3fn_if_needed,
    view_as_uint8_if_needed,
)
from modelopt.torch.quantization.nn import SequentialQuantizer


def _column_linear(out_features=8, in_features=4, bias=False, **kwargs):
    return LinearConfig(
        linear_type=LINEAR_COLUMN,
        weight=torch.randn(out_features, in_features),
        bias=torch.randn(out_features) if bias else None,
        **kwargs,
    )


def _row_linear(out_features=4, in_features=8, bias=False, **kwargs):
    return LinearConfig(
        linear_type=LINEAR_ROW,
        weight=torch.randn(out_features, in_features),
        bias=torch.randn(out_features) if bias else None,
        **kwargs,
    )


def _make_model_config(num_layers=2, hidden=4, vocab=8):
    """Builds a minimal but structurally realistic ModelConfig."""
    torch.manual_seed(0)
    layers = []
    for _ in range(num_layers):
        attention = AttentionConfig(
            qkv=_column_linear(3 * hidden, hidden),
            dense=_row_linear(hidden, hidden),
        )
        mlp = MLPConfig(
            fc=_column_linear(2 * hidden, hidden),
            proj=_row_linear(hidden, 2 * hidden),
            hidden_act="silu",
        )
        layers.append(
            DecoderLayerConfig(
                attention=attention,
                mlp=mlp,
                input_layernorm=LayernormConfig(weight=torch.ones(hidden)),
                post_layernorm=LayernormConfig(weight=torch.ones(hidden)),
            )
        )
    return ModelConfig(
        vocab_size=vocab,
        vocab_embedding=EmbeddingConfig(weight=torch.randn(vocab, hidden)),
        position_embedding=EmbeddingConfig(weight=torch.randn(vocab, hidden)),
        ln_embed=LayernormConfig(weight=torch.ones(hidden)),
        layers=layers,
        ln_f=LayernormConfig(weight=torch.ones(hidden)),
        lm_head=_column_linear(vocab, hidden),
    )


INT4_BLOCK32_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": 4, "block_sizes": {-1: 32}},
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": 8, "axis": None},
            "enable": True,
        },
    ],
    "algorithm": "max",
}


def _make_quantized_lm_head(out_features=100, in_features=64, quant_cfg=INT4_BLOCK32_CFG):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
    mtq.quantize(model, quant_cfg, lambda m: m(torch.randn(2, 8, in_features)))
    return model[0]


class TestSameTensor:
    def test_empty_list_is_true(self):
        assert _same_tensor([])

    def test_all_none_is_true(self):
        assert _same_tensor([None, None, None])

    def test_single_tensor_is_true(self):
        assert _same_tensor([torch.ones(3)])

    def test_equal_tensors_are_true(self):
        assert _same_tensor([torch.ones(2, 2), torch.ones(2, 2)])

    def test_unequal_tensors_are_false(self):
        assert not _same_tensor([torch.ones(2), torch.zeros(2)])


class TestFp8Views:
    def test_fp8_viewed_as_uint8(self):
        t = torch.randn(4, 4).to(torch.float8_e4m3fn)
        viewed = view_as_uint8_if_needed(t)
        assert viewed.dtype == torch.uint8
        assert viewed.shape == t.shape

    def test_non_fp8_passthrough(self):
        t = torch.randn(4, 4)
        assert view_as_uint8_if_needed(t) is t

    def test_uint8_viewed_as_fp8(self):
        t = torch.randint(0, 255, (4, 4), dtype=torch.uint8)
        viewed = view_as_float8_e4m3fn_if_needed(t)
        assert viewed.dtype == torch.float8_e4m3fn
        assert viewed.shape == t.shape

    def test_non_uint8_passthrough(self):
        t = torch.randn(4, 4)
        assert view_as_float8_e4m3fn_if_needed(t) is t

    def test_roundtrip_preserves_bits(self):
        t = torch.randn(4, 4).to(torch.float8_e4m3fn)
        roundtrip = view_as_float8_e4m3fn_if_needed(view_as_uint8_if_needed(t))
        assert roundtrip.dtype == torch.float8_e4m3fn
        assert torch.equal(roundtrip.view(torch.uint8), t.view(torch.uint8))


class TestShallowCopyWithFieldInstantiation:
    def test_dataclass_fields_are_new_instances(self):
        expert = ExpertConfig(fc=_column_linear(), proj=_row_linear())
        clone = _shallow_copy_with_field_instantiation(expert)
        assert clone is not expert
        assert clone.fc is not expert.fc
        assert clone.proj is not expert.proj

    def test_tensors_within_fields_are_still_shared(self):
        expert = ExpertConfig(fc=_column_linear(), proj=_row_linear())
        clone = _shallow_copy_with_field_instantiation(expert)
        assert clone.fc.weight is expert.fc.weight

    def test_field_mutation_does_not_leak_to_original(self):
        expert = ExpertConfig(fc=_column_linear(), proj=_row_linear())
        clone = _shallow_copy_with_field_instantiation(expert)
        clone.fc.weight = torch.zeros(1, 1)
        assert expert.fc.weight.shape == (8, 4)


class TestSplitModelConfigForTp:
    def test_column_linear_splits_weight_and_bias_on_dim0(self):
        config = _column_linear(out_features=8, in_features=4, bias=True)
        weight, bias = config.weight, config.bias
        configs = _split_model_config_for_tp(config, 2)
        assert len(configs) == 2
        for i, cfg in enumerate(configs):
            assert cfg.weight.shape == (4, 4)
            assert torch.equal(cfg.weight, weight.chunk(2, dim=0)[i])
            assert torch.equal(cfg.bias, bias.chunk(2, dim=0)[i])

    def test_column_linear_pads_odd_rows_before_split(self):
        config = _column_linear(out_features=5, in_features=4)
        configs = _split_model_config_for_tp(config, 2)
        assert configs[0].weight.shape == (3, 4)
        assert configs[1].weight.shape == (3, 4)
        # The last row of the second shard is the zero padding.
        assert torch.equal(configs[1].weight[-1], torch.zeros(4))
        # NOTE: documents current behavior; arguably a bug because the input
        # (merged) config's weight is reassigned by _split_model_config_for_tp.
        assert config.weight.shape == (6, 4)

    def test_row_linear_splits_weight_on_dim1_and_keeps_bias(self):
        config = _row_linear(out_features=4, in_features=8, bias=True)
        weight, bias = config.weight, config.bias
        configs = _split_model_config_for_tp(config, 2)
        for i, cfg in enumerate(configs):
            assert cfg.weight.shape == (4, 4)
            assert torch.equal(cfg.weight, weight.chunk(2, dim=1)[i])
            # Row linear bias is not split; all shards share it.
            assert cfg.bias is bias

    def test_tp_disabled_linear_is_not_split(self):
        config = _column_linear(out_features=8, in_features=4, tp=False)
        configs = _split_model_config_for_tp(config, 2)
        assert len(configs) == 2
        for cfg in configs:
            assert cfg.weight is config.weight

    def test_scalar_weights_scaling_factor_is_shared(self):
        config = _column_linear(weights_scaling_factor=torch.tensor(0.5))
        configs = _split_model_config_for_tp(config, 2)
        for cfg in configs:
            assert cfg.weights_scaling_factor.numel() == 1
            assert cfg.weights_scaling_factor is config.weights_scaling_factor

    def test_column_per_channel_scaling_factor_split_on_dim0(self):
        wsf = torch.rand(8)
        config = _column_linear(out_features=8, weights_scaling_factor=wsf)
        configs = _split_model_config_for_tp(config, 2)
        for i, cfg in enumerate(configs):
            assert torch.equal(cfg.weights_scaling_factor, wsf.chunk(2, dim=0)[i])

    def test_column_fp8_scaling_factor_preserves_dtype(self):
        wsf = torch.rand(8, 2).to(torch.float8_e4m3fn)
        config = _column_linear(out_features=8, weights_scaling_factor=wsf)
        configs = _split_model_config_for_tp(config, 2)
        for cfg in configs:
            assert cfg.weights_scaling_factor.dtype == torch.float8_e4m3fn
            assert cfg.weights_scaling_factor.shape == (4, 2)

    def test_row_awq_scaling_factor_split_on_dim1(self):
        wsf = torch.rand(8, 4).to(torch.float8_e4m3fn)
        config = _row_linear(
            out_features=8, in_features=16, weights_scaling_factor=wsf, awq_block_size=4
        )
        configs = _split_model_config_for_tp(config, 2)
        for i, cfg in enumerate(configs):
            assert cfg.weights_scaling_factor.dtype == torch.float8_e4m3fn
            assert torch.equal(
                cfg.weights_scaling_factor.view(torch.uint8),
                wsf.view(torch.uint8).chunk(2, dim=1)[i],
            )

    def test_row_int8_sq_scaling_factor_not_split(self):
        # awq_block_size == 0 means INT8 SQ: per-channel scaling factor stays intact.
        wsf = torch.rand(8)
        config = _row_linear(out_features=8, in_features=16, weights_scaling_factor=wsf)
        configs = _split_model_config_for_tp(config, 2)
        for cfg in configs:
            assert cfg.weights_scaling_factor is wsf

    def test_row_prequant_scaling_factor_split_on_dim0(self):
        psf = torch.rand(16)
        config = _row_linear(out_features=8, in_features=16, prequant_scaling_factor=psf)
        configs = _split_model_config_for_tp(config, 2)
        for i, cfg in enumerate(configs):
            assert torch.equal(cfg.prequant_scaling_factor, psf.chunk(2, dim=0)[i])

    def test_group_linear_raises(self):
        config = LinearConfig(linear_type=LINEAR_GROUP, weight=torch.randn(8, 4))
        with pytest.raises(AssertionError, match="group linear"):
            _split_model_config_for_tp(config, 2)

    def test_conv_config_raises(self):
        with pytest.raises(NotImplementedError, match="ConvConfig"):
            _split_model_config_for_tp(ConvConfig(weight=torch.randn(4, 4, 3, 3)), 2)

    def test_embedding_split_on_dim0(self):
        weight = torch.randn(8, 4)
        config = EmbeddingConfig(weight=weight)
        configs = _split_model_config_for_tp(config, 2)
        for i, cfg in enumerate(configs):
            assert torch.equal(cfg.weight, weight.chunk(2, dim=0)[i])

    def test_embedding_split_pads_odd_vocab(self):
        config = EmbeddingConfig(weight=torch.randn(5, 4))
        configs = _split_model_config_for_tp(config, 2)
        assert configs[0].weight.shape == (3, 4)
        assert configs[1].weight.shape == (3, 4)
        assert torch.equal(configs[1].weight[-1], torch.zeros(4))

    def test_relative_attention_table_split_on_dim0(self):
        weight = torch.randn(6, 4)
        config = RelativeAttentionTableConfig(weight=weight)
        configs = _split_model_config_for_tp(config, 3)
        for i, cfg in enumerate(configs):
            assert torch.equal(cfg.weight, weight.chunk(3, dim=0)[i])

    def test_expert_config_split(self):
        hidden, moe_hidden, num_experts = 4, 8, 2
        config = ExpertConfig(
            fc=LinearConfig(
                linear_type=LINEAR_COLUMN,
                weight=torch.randn(num_experts, 2 * moe_hidden, hidden),
            ),
            proj=LinearConfig(
                linear_type=LINEAR_ROW,
                weight=torch.randn(num_experts, hidden, moe_hidden),
            ),
        )
        fc_weight, proj_weight = config.fc.weight, config.proj.weight
        configs = _split_model_config_for_tp(config, 2)

        # proj (row) is split on dim 2.
        for i, cfg in enumerate(configs):
            assert torch.equal(cfg.proj.weight, proj_weight.chunk(2, dim=2)[i])

        # fc holds concatenated [w3; w1]: each is split separately then re-concatenated
        # so each shard gets matching halves of w3 and w1.
        merged_w3, merged_w1 = torch.chunk(fc_weight, 2, dim=1)
        for i, cfg in enumerate(configs):
            expected = torch.cat(
                [merged_w3.chunk(2, dim=1)[i], merged_w1.chunk(2, dim=1)[i]], dim=1
            )
            assert torch.equal(cfg.fc.weight, expected)

    def test_expert_config_awq_scaling_factors_split(self):
        hidden, moe_hidden, num_experts, block = 8, 8, 2, 4
        fc_wsf = torch.rand(num_experts, 2 * moe_hidden, hidden // block)
        proj_wsf = torch.rand(num_experts, hidden, moe_hidden // block)
        proj_psf = torch.rand(num_experts, moe_hidden)
        config = ExpertConfig(
            fc=LinearConfig(
                linear_type=LINEAR_COLUMN,
                weight=torch.randn(num_experts, 2 * moe_hidden, hidden),
                weights_scaling_factor=fc_wsf,
                awq_block_size=block,
            ),
            proj=LinearConfig(
                linear_type=LINEAR_ROW,
                weight=torch.randn(num_experts, hidden, moe_hidden),
                weights_scaling_factor=proj_wsf,
                prequant_scaling_factor=proj_psf,
                awq_block_size=block,
            ),
        )
        configs = _split_model_config_for_tp(config, 2)

        merged_wsf_w3, merged_wsf_w1 = torch.chunk(fc_wsf, 2, dim=1)
        for i, cfg in enumerate(configs):
            expected_fc_wsf = torch.cat(
                [merged_wsf_w3.chunk(2, dim=1)[i], merged_wsf_w1.chunk(2, dim=1)[i]], dim=1
            )
            assert torch.equal(cfg.fc.weights_scaling_factor, expected_fc_wsf)
            assert torch.equal(cfg.proj.weights_scaling_factor, proj_wsf.chunk(2, dim=2)[i])
            assert torch.equal(cfg.proj.prequant_scaling_factor, proj_psf.chunk(2, dim=-1)[i])

    def test_expert_config_wrong_linear_types_raise(self):
        config = ExpertConfig(
            fc=LinearConfig(linear_type=LINEAR_ROW, weight=torch.randn(2, 8, 4)),
            proj=LinearConfig(linear_type=LINEAR_ROW, weight=torch.randn(2, 4, 8)),
        )
        with pytest.raises(AssertionError):
            _split_model_config_for_tp(config, 2)

    def test_recursion_through_dataclass_and_list(self):
        mlp = MLPConfig(
            fc=_column_linear(8, 4),
            proj=_row_linear(4, 8),
            hidden_act="gelu",
        )
        configs = _split_model_config_for_tp([mlp], 2)
        assert len(configs) == 2
        for i, cfg in enumerate(configs):
            assert isinstance(cfg, list)
            assert cfg[0].fc.weight.shape == (4, 4)
            assert cfg[0].proj.weight.shape == (4, 4)
            assert torch.equal(cfg[0].fc.weight, mlp.fc.weight.chunk(2, dim=0)[i])
            assert cfg[0].hidden_act == "gelu"


class TestSplitModelConfigForPp:
    def test_layers_partitioned_and_heads_assigned(self):
        model_config = _make_model_config(num_layers=4)
        all_layers = list(model_config.layers)
        configs = _split_model_config_for_pp(model_config, 2)
        assert len(configs) == 2

        first, last = configs
        assert first.layers == all_layers[:2]
        assert last.layers == all_layers[2:]

        # Only the first PP rank keeps the embeddings.
        assert first.vocab_embedding is not None
        assert first.position_embedding is not None
        assert first.ln_embed is not None
        assert last.vocab_embedding is None
        assert last.position_embedding is None
        assert last.ln_embed is None

        # Only the last PP rank keeps the final norm and lm_head.
        assert first.ln_f is None
        assert first.lm_head is None
        assert last.ln_f is not None
        assert last.lm_head is not None

    def test_split_factor_one_keeps_everything(self):
        model_config = _make_model_config(num_layers=2)
        configs = _split_model_config_for_pp(model_config, 1)
        assert len(configs) == 1
        assert configs[0].vocab_embedding is not None
        assert configs[0].lm_head is not None
        assert len(configs[0].layers) == 2

    def test_non_divisible_layers_raise(self):
        model_config = _make_model_config(num_layers=3)
        with pytest.raises(AssertionError):
            _split_model_config_for_pp(model_config, 2)


class TestPostprocessModelConfig:
    """Single-process cases: training TP == PP == 1, so only identity and split paths."""

    def test_identity_returns_same_config(self):
        model_config = _make_model_config()
        configs = postprocess_model_config(model_config)
        assert len(configs) == 1
        assert configs[0] is model_config
        assert model_config.rank == 0
        assert model_config.tensor_parallel == 1
        assert model_config.pipeline_parallel == 1

    def test_split_to_tp2(self):
        model_config = _make_model_config(hidden=4, vocab=8)
        qkv_weight = model_config.layers[0].attention.qkv.weight
        dense_weight = model_config.layers[0].attention.dense.weight
        embedding_weight = model_config.vocab_embedding.weight

        configs = postprocess_model_config(model_config, inference_tensor_parallel=2)
        assert len(configs) == 2
        assert [cfg.rank for cfg in configs] == [0, 1]
        assert all(cfg.tensor_parallel == 2 for cfg in configs)

        for i, cfg in enumerate(configs):
            # Column linear split on dim 0.
            assert torch.equal(cfg.layers[0].attention.qkv.weight, qkv_weight.chunk(2, dim=0)[i])
            # Row linear split on dim 1.
            assert torch.equal(
                cfg.layers[0].attention.dense.weight, dense_weight.chunk(2, dim=1)[i]
            )
            # Embedding split on dim 0.
            assert torch.equal(cfg.vocab_embedding.weight, embedding_weight.chunk(2, dim=0)[i])

        # Layernorm weights are replicated, not split.
        assert torch.equal(configs[0].ln_f.weight, configs[1].ln_f.weight)
        assert configs[0].ln_f.weight.shape == (4,)

    def test_split_to_tp2_pp2_rank_assignment(self):
        model_config = _make_model_config(num_layers=4)
        configs = postprocess_model_config(
            model_config, inference_tensor_parallel=2, inference_pipeline_parallel=2
        )
        assert len(configs) == 4
        # Ordered as [tp0/pp0, tp0/pp1, tp1/pp0, tp1/pp1] with rank = tp + pp * tp_size.
        assert [cfg.rank for cfg in configs] == [0, 2, 1, 3]
        assert all(cfg.tensor_parallel == 2 for cfg in configs)
        assert all(cfg.pipeline_parallel == 2 for cfg in configs)
        assert all(len(cfg.layers) == 2 for cfg in configs)

        # PP rank 0 keeps the embedding, PP rank 1 keeps the lm_head.
        pp0_configs = [configs[0], configs[2]]
        pp1_configs = [configs[1], configs[3]]
        assert all(cfg.vocab_embedding is not None for cfg in pp0_configs)
        assert all(cfg.lm_head is None for cfg in pp0_configs)
        assert all(cfg.vocab_embedding is None for cfg in pp1_configs)
        assert all(cfg.lm_head is not None for cfg in pp1_configs)


class TestPadEmbeddingLmHead:
    def test_noop_when_vocab_is_multiple_of_padding_factor(self):
        model_config = _make_model_config(vocab=128)
        embedding_weight = model_config.vocab_embedding.weight
        lm_head_weight = model_config.lm_head.weight
        pad_embedding_lm_head(model_config)
        assert model_config.vocab_size == 128
        assert model_config.vocab_embedding.weight is embedding_weight
        assert model_config.lm_head.weight is lm_head_weight

    def test_pads_vocab_embedding_and_lm_head(self):
        model_config = _make_model_config(vocab=100, hidden=4)
        model_config.lm_head.bias = torch.randn(100)
        original_embedding = model_config.vocab_embedding.weight.clone()
        original_lm_head = model_config.lm_head.weight.clone()
        original_bias = model_config.lm_head.bias.clone()

        pad_embedding_lm_head(model_config)

        assert model_config.vocab_size == 128
        assert model_config.vocab_embedding.weight.shape == (128, 4)
        assert model_config.lm_head.weight.shape == (128, 4)
        assert model_config.lm_head.bias.shape == (128,)

        assert torch.equal(model_config.vocab_embedding.weight[:100], original_embedding)
        assert torch.equal(model_config.vocab_embedding.weight[100:], torch.zeros(28, 4))
        assert torch.equal(model_config.lm_head.weight[:100], original_lm_head)
        assert torch.equal(model_config.lm_head.weight[100:], torch.zeros(28, 4))
        assert torch.equal(model_config.lm_head.bias[:100], original_bias)
        assert torch.equal(model_config.lm_head.bias[100:], torch.zeros(28))

    def test_pads_weights_scaling_factor_with_int4_maxbound(self):
        model_config = _make_model_config(vocab=100, hidden=4)
        original_wsf = torch.rand(100, 2)
        model_config.lm_head.weights_scaling_factor = original_wsf.clone()

        pad_embedding_lm_head(model_config)

        wsf = model_config.lm_head.weights_scaling_factor
        assert wsf.shape == (128, 2)
        assert torch.equal(wsf[:100], original_wsf)
        # Padded rows use 1 / 7.0 (int4 maxbound).
        assert torch.allclose(wsf[100:], torch.full((28, 2), 1.0 / 7.0))

    def test_custom_padding_factor(self):
        model_config = _make_model_config(vocab=10, hidden=4)
        pad_embedding_lm_head(model_config, padding_factor=16)
        assert model_config.vocab_size == 16
        assert model_config.vocab_embedding.weight.shape == (16, 4)
        assert model_config.lm_head.weight.shape == (16, 4)

    def test_non_2d_weights_scaling_factor_raises(self):
        model_config = _make_model_config(vocab=100, hidden=4)
        model_config.lm_head.weights_scaling_factor = torch.rand(100)
        with pytest.raises(AssertionError, match="2D"):
            pad_embedding_lm_head(model_config)


class TestUpdateLmHeadQuantization:
    def test_plain_linear_is_ignored(self):
        # No weight_quantizer/input_quantizer attributes: early return, no error.
        update_lm_head_quantization(ModelConfig(), nn.Linear(4, 4))

    def test_non_divisible_vocab_disables_quantization(self):
        # 100 % (32 * 1) != 0 -> quantization must be disabled.
        lm_head = _make_quantized_lm_head(out_features=100)
        assert lm_head.weight_quantizer.is_enabled
        update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)
        assert not lm_head.weight_quantizer.is_enabled
        assert not lm_head.input_quantizer.is_enabled

    def test_divisible_vocab_keeps_quantization_and_warns(self):
        # 128 % (32 * 1) == 0 -> quantization stays enabled with a warning.
        lm_head = _make_quantized_lm_head(out_features=128)
        with pytest.warns(UserWarning, match="lm_head quantization"):
            update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)
        assert lm_head.weight_quantizer.is_enabled
        assert lm_head.input_quantizer.is_enabled

    def test_inference_tp_affects_divisibility(self):
        # 96 % (32 * 1) == 0 but 96 % (32 * 2) != 0.
        lm_head = _make_quantized_lm_head(out_features=96)
        with pytest.warns(UserWarning, match="lm_head quantization"):
            update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)
        assert lm_head.weight_quantizer.is_enabled

        update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=2)
        assert not lm_head.weight_quantizer.is_enabled

    def test_awq_lite_pre_quant_scale_is_removed_on_disable(self):
        quant_cfg = {
            "quant_cfg": INT4_BLOCK32_CFG["quant_cfg"],
            "algorithm": "awq_lite",
        }
        lm_head = _make_quantized_lm_head(out_features=100, quant_cfg=quant_cfg)
        assert hasattr(lm_head.input_quantizer, "_pre_quant_scale")

        update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)
        assert not lm_head.weight_quantizer.is_enabled
        assert not hasattr(lm_head.input_quantizer, "_pre_quant_scale")

    def test_sequential_quantizer_uses_first_block_size(self):
        # W4A8-style config: the block size comes from the first quantizer in the sequence.
        quant_cfg = {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [
                        {"num_bits": 4, "block_sizes": {-1: 32}},
                        {"num_bits": (4, 3), "axis": None},
                    ],
                    "enable": True,
                },
                {
                    "quantizer_name": "*input_quantizer",
                    "cfg": {"num_bits": (4, 3), "axis": None},
                    "enable": True,
                },
            ],
            "algorithm": "max",
        }
        lm_head = _make_quantized_lm_head(out_features=100, quant_cfg=quant_cfg)
        assert isinstance(lm_head.weight_quantizer, SequentialQuantizer)

        update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)
        assert not lm_head.weight_quantizer.is_enabled
        assert not lm_head.input_quantizer.is_enabled

    def test_disabled_quantizer_still_warns_enable(self):
        # NOTE: documents current behavior; arguably a bug because the function
        # warns "Enable lm_head quantization" even when the quantizers are
        # already disabled and no quantization will actually happen.
        lm_head = _make_quantized_lm_head(out_features=100)
        lm_head.weight_quantizer.disable()
        with pytest.warns(UserWarning, match="Enable lm_head quantization"):
            update_lm_head_quantization(ModelConfig(), lm_head, inference_tensor_parallel=1)


class TestCheckWeightShapeValid:
    def test_column_valid(self):
        config = _column_linear(out_features=8, in_features=16)
        check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_column_non_divisible_raises(self):
        config = _column_linear(out_features=8, in_features=6)
        with pytest.raises(AssertionError, match="cannot be split"):
            check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_column_awq_block_size_violation_raises(self):
        # 16 / 4 = 4 per rank, not divisible by block size 8.
        config = _column_linear(out_features=8, in_features=16, awq_block_size=8)
        with pytest.raises(NotImplementedError, match="block size"):
            check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_column_awq_block_size_valid(self):
        config = _column_linear(out_features=8, in_features=32, awq_block_size=8)
        check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_row_non_divisible_k_raises(self):
        config = _row_linear(out_features=6, in_features=16)
        with pytest.raises(AssertionError, match="cannot be split"):
            check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_row_awq_block_size_violation_raises(self):
        # m=16, tp=4 -> 4 per rank, not divisible by block size 8.
        config = _row_linear(out_features=8, in_features=16, awq_block_size=8)
        with pytest.raises(NotImplementedError, match="block size"):
            check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_tp_disabled_config_ignores_inference_tp(self):
        config = _column_linear(out_features=8, in_features=6, tp=False)
        check_weight_shape_valid(config, inference_tensor_parallel=4)

    def test_training_tp_scales_merged_dim(self):
        # k=4 with training TP 4 -> merged k=16, divisible by inference TP 4.
        config = _column_linear(out_features=8, in_features=4)
        check_weight_shape_valid(config, inference_tensor_parallel=4, training_tensor_parallel=4)

    def test_expert_config_valid_and_invalid(self):
        valid = ExpertConfig(
            fc=LinearConfig(linear_type=LINEAR_COLUMN, weight=torch.randn(2, 8, 16)),
            proj=LinearConfig(linear_type=LINEAR_ROW, weight=torch.randn(2, 16, 8)),
        )
        check_weight_shape_valid(valid, inference_tensor_parallel=4)

        invalid = ExpertConfig(
            fc=LinearConfig(
                linear_type=LINEAR_COLUMN, weight=torch.randn(2, 8, 16), awq_block_size=8
            ),
            proj=LinearConfig(linear_type=LINEAR_ROW, weight=torch.randn(2, 16, 8)),
        )
        with pytest.raises(NotImplementedError, match="block size"):
            check_weight_shape_valid(invalid, inference_tensor_parallel=4)

    def test_recursion_through_model_config(self):
        model_config = _make_model_config(hidden=4)
        check_weight_shape_valid(model_config, inference_tensor_parallel=2)

        # An invalid nested linear is caught through the recursion.
        model_config.layers[1].attention.qkv.weight = torch.randn(12, 6)
        with pytest.raises(AssertionError, match="cannot be split"):
            check_weight_shape_valid(model_config, inference_tensor_parallel=4)

    def test_recursion_through_list(self):
        configs = [_column_linear(8, 16), _column_linear(8, 6)]
        with pytest.raises(AssertionError, match="cannot be split"):
            check_weight_shape_valid(configs, inference_tensor_parallel=4)


class TestPostprocessTensors:
    @pytest.mark.parametrize(
        "suffix",
        ["weight", "bias", "prequant_scaling_factor", "recurrent_param", "rel_attn_table"],
    )
    def test_float_tensors_with_matching_suffix_are_converted(self, suffix):
        weights = {f"model.layers.0.{suffix}": torch.randn(2, 2, dtype=torch.float32)}
        postprocess_tensors(weights, torch.float16)
        assert weights[f"model.layers.0.{suffix}"].dtype == torch.float16

    def test_bfloat16_source_is_converted(self):
        weights = {"a.weight": torch.randn(2, 2, dtype=torch.bfloat16)}
        postprocess_tensors(weights, torch.float16)
        assert weights["a.weight"].dtype == torch.float16

    def test_router_keys_are_not_converted(self):
        weights = {"mlp.router.weight": torch.randn(2, 2, dtype=torch.float32)}
        postprocess_tensors(weights, torch.float16)
        assert weights["mlp.router.weight"].dtype == torch.float32

    def test_non_matching_suffix_is_not_converted(self):
        weights = {"a.weights_scaling_factor_2": torch.randn(2, dtype=torch.float32)}
        # "weights_scaling_factor_2" does not end with any of the cast suffixes.
        postprocess_tensors(weights, torch.float16)
        assert weights["a.weights_scaling_factor_2"].dtype == torch.float32

    def test_non_float_dtypes_are_preserved(self):
        weights = {
            "a.weight": torch.randint(0, 255, (2, 2), dtype=torch.uint8),
            "b.weight": torch.randn(2, 2).to(torch.float8_e4m3fn),
            "c.weight": torch.randint(0, 10, (2, 2), dtype=torch.int32),
        }
        postprocess_tensors(weights, torch.float16)
        assert weights["a.weight"].dtype == torch.uint8
        assert weights["b.weight"].dtype == torch.float8_e4m3fn
        assert weights["c.weight"].dtype == torch.int32

    def test_views_are_cloned(self):
        base = torch.randn(4, 4)
        view = base[:2]
        assert view._is_view()
        weights = {"a.weight": view}
        postprocess_tensors(weights, torch.float32)
        assert not weights["a.weight"]._is_view()
        assert torch.equal(weights["a.weight"], base[:2])

    def test_non_contiguous_tensors_are_made_contiguous(self):
        transposed = torch.randn(2, 4).t()
        assert not transposed.is_contiguous()
        weights = {"a.scale": transposed}
        postprocess_tensors(weights, torch.float32)
        assert weights["a.scale"].is_contiguous()
        assert not weights["a.scale"]._is_view()

    def test_force_flags_disabled_keep_tensor_as_is(self):
        view = torch.randn(4, 4)[:2]
        weights = {"a.scale": view}
        postprocess_tensors(
            weights,
            torch.float32,
            force_cpu=False,
            force_contiguous=False,
            force_non_view=False,
        )
        assert weights["a.scale"] is view

    def test_values_are_updated_in_place_and_returns_none(self):
        weights = {"a.weight": torch.randn(2, 2, dtype=torch.float32)}
        result = postprocess_tensors(weights, torch.bfloat16)
        assert result is None
        assert set(weights.keys()) == {"a.weight"}
        assert weights["a.weight"].dtype == torch.bfloat16
