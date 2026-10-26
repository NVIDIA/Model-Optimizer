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

"""This module defines the TensorRT-LLM checkpoint model_config format.

This format can be converted from huggingface, megatron or modelopt-quantized model.
And we will build tensorrt_llm engine from the context saved with this format.
"""

import math
from dataclasses import dataclass, field

import torch

from modelopt.torch.quantization.qtensor import NVFP4QTensor

from ..quant_format import (
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_NVFP4_SVDQUANT,
    QUANTIZATION_W4A8_AWQ,
)

__all__ = [
    "LAYERNORM_DEFAULT",
    "LAYERNORM_RMS",
    "LINEAR_COLUMN",
    "LINEAR_GROUP",
    "LINEAR_ROW",
    "AttentionConfig",
    "ConvConfig",
    "DecoderLayerConfig",
    "EmbeddingConfig",
    "ExpertConfig",
    "LayernormConfig",
    "LinearActConfig",
    "LinearConfig",
    "MLPConfig",
    "MOEConfig",
    "MedusaHeadConfig",
    "ModelConfig",
    "QKVConfig",
    "RecurrentConfig",
    "RelativeAttentionTableConfig",
    "RgLruConfig",
]

LINEAR_COLUMN = "column"
LINEAR_ROW = "row"
LINEAR_GROUP = "group"

# These need to be synced with modelopt.torch.export.trtllm.tensorrt_llm_type
LAYERNORM_DEFAULT = "LayerNorm"
LAYERNORM_RMS = "RmsNorm"


@dataclass
class EmbeddingConfig:
    """The embedding layer config."""

    weight: torch.Tensor = None

    @property
    def local_vocab_size(self):
        """Infers the vocab_size from the embedding layer weights shape."""
        return self.weight.shape[0]

    @property
    def hidden_size(self):
        """Infers the hidden_size from the embedding layer weights shape."""
        return self.weight.shape[1]


@dataclass
class LayernormConfig:
    """The layernorm layer config."""

    weight: torch.Tensor = None
    bias: torch.Tensor = None
    layernorm_type: str = LAYERNORM_DEFAULT
    eps: float = 1e-5


@dataclass
class LinearConfig:
    """The linear layer config."""

    quantization: str | None = QUANTIZATION_NONE
    linear_type: str = LINEAR_COLUMN
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    activation_scaling_factor: torch.Tensor = None
    weights_scaling_factor: torch.Tensor = None

    # For methods like W4A8 AWQ, we have two quantizers for weights
    # For W4A8, the first quantizer is for INT4 quantization and the second quantizer is for FP8 quantization
    # `weight_scaling_factor_2` is the scaling factor for the second FP8 quantizer
    weights_scaling_factor_2: torch.Tensor = None

    prequant_scaling_factor: torch.Tensor = None

    # For FP8 per channel per token quantization, we have a per_channel_scale
    per_channel_scale: torch.Tensor = None

    awq_block_size: int = 0

    # If set to false, we do not split or merge this config during post tp processing.
    tp: bool = True

    def __del__(self):
        del self.weight
        del self.bias
        del self.activation_scaling_factor
        del self.weights_scaling_factor
        del self.weights_scaling_factor_2
        del self.prequant_scaling_factor


@dataclass
class LinearActConfig:
    """The linear + activation layer config."""

    linear: LinearConfig | None = None
    hidden_act: str = ""


@dataclass
class ConvConfig:
    """The Conv layer config."""

    quantization: str | None = QUANTIZATION_NONE
    weight: torch.Tensor = None
    bias: torch.Tensor = None


@dataclass
class QKVConfig:
    """The QKV layer config."""

    q: LinearConfig | None = None
    k: LinearConfig | None = None
    v: LinearConfig | None = None

    def _require_projections(self) -> tuple[LinearConfig, LinearConfig, LinearConfig]:
        assert self.q is not None and self.k is not None and self.v is not None, (
            "QKVConfig requires Q, K and V projections."
        )
        return self.q, self.k, self.v

    @property
    def weight(self):
        """The generated linear layer weight.

        The Q, K, V weights are concat together to fit the TensorRT-LLM QKV linear layer.
        """
        q, k, v = self._require_projections()
        return torch.cat((q.weight, k.weight, v.weight))

    @property
    def bias(self):
        """The generated linear layer bias.

        The Q, K, V bias are concat together to fit the TensorRT-LLM QKV linear layer.
        """
        q, k, v = self._require_projections()
        if q.bias is None:
            assert k.bias is None and v.bias is None, "K and V should have valid bias as Q"
            return None
        return torch.cat((q.bias, k.bias, v.bias))

    @property
    def activation_scaling_factor(self):
        """Returns the merged activation_scaling_factor across Q, K and V.

        The max of the Q, K, V activation scaling factors is returned.
        """
        q, k, v = self._require_projections()
        if (
            q.activation_scaling_factor is None
            or k.activation_scaling_factor is None
            or v.activation_scaling_factor is None
        ):
            return None

        return (
            torch.stack(
                [
                    q.activation_scaling_factor,
                    k.activation_scaling_factor,
                    v.activation_scaling_factor,
                ]
            )
            .max(dim=0)
            .values
        )

    @property
    def weights_scaling_factor(self):
        """Returns the merged weights_scaling_factor across Q, K and V.

        If the quantization is FP8, the max of the Q, K, V weight scaling factors is returned.
        If the quanitzation is INT8_SQ, the concat value is returned.
        """
        q, k, v = self._require_projections()
        if (
            q.weights_scaling_factor is None
            or k.weights_scaling_factor is None
            or v.weights_scaling_factor is None
        ):
            return None

        if q.weights_scaling_factor.numel() != 1:
            # for NVFP4, Int4 AWQ, Int8 SQ and Int8 WO case, we concatenate the
            # q_weight_scaling_factor, k_weight_scaling_factor, v_weight_scaling_factor
            if q.weights_scaling_factor.dtype == torch.float8_e4m3fn:
                # For NVFP4, we recompute the weights_scaling_factor using max weights_scaling_factor2
                qkv_weights_scaling_factor, _ = NVFP4QTensor.get_weights_scaling_factor(
                    self.weight, self.awq_block_size, self.weights_scaling_factor_2
                )
            else:
                qkv_weights_scaling_factor = torch.cat(
                    (
                        q.weights_scaling_factor,
                        k.weights_scaling_factor,
                        v.weights_scaling_factor,
                    )
                )
        else:
            # for FP8 set qkv_weight_scaling_factor to the max of
            # q_weight_scaling_factor, k_weight_scaling_factor, v_weight_scaling_factor
            qkv_weights_scaling_factor = (
                torch.stack(
                    [
                        q.weights_scaling_factor,
                        k.weights_scaling_factor,
                        v.weights_scaling_factor,
                    ],
                )
                .max(dim=0)
                .values
            )
        return qkv_weights_scaling_factor

    @property
    def weights_scaling_factor_2(self):
        """Returns the merged weights_scaling_factor_2 across Q, K and V.

        weight_scaling_factor_2 is needed for W4A8 AWQ.
        """
        q, k, v = self._require_projections()
        if (
            q.weights_scaling_factor_2 is None
            or k.weights_scaling_factor_2 is None
            or v.weights_scaling_factor_2 is None
        ):
            return None

        # For W4A8 AWQ, weight_scaling_factor_2 corresponds to the per-tensor FP8 quantization.
        # Hence weight_scaling_factor_2 should be a scalar.
        assert q.weights_scaling_factor_2.numel() == 1

        # set qkv_weight_scaling_factor_2 to the max of q,k,v weight_scaling_factor_2
        qkv_weights_scaling_factor_2 = (
            torch.stack(
                [
                    q.weights_scaling_factor_2,
                    k.weights_scaling_factor_2,
                    v.weights_scaling_factor_2,
                ]
            )
            .max(dim=0)
            .values
        )

        return qkv_weights_scaling_factor_2

    @property
    def prequant_scaling_factor(self):
        """Returns the merged prequant_scaling_factor across Q, K and V.

        Prequant scaling factors for Q, K, V should be the same. So just return one of them.
        """
        q, k, v = self._require_projections()
        if (
            q.prequant_scaling_factor is None
            or k.prequant_scaling_factor is None
            or v.prequant_scaling_factor is None
        ):
            return None

        assert torch.equal(q.prequant_scaling_factor, k.prequant_scaling_factor) and torch.equal(
            k.prequant_scaling_factor, v.prequant_scaling_factor
        ), "Prequant scaling factors of Q, K and V should be the same"
        return q.prequant_scaling_factor

    @property
    def awq_block_size(self):
        """Returns the awq_block_size of this QKV layer."""
        q, k, v = self._require_projections()
        assert q.awq_block_size == k.awq_block_size == v.awq_block_size, (
            "awq_block_size of QKV should be the same."
        )
        return q.awq_block_size


@dataclass
class RelativeAttentionTableConfig:
    """Relative attention table config. For splitting purpose."""

    weight: torch.Tensor = None


@dataclass
class AttentionConfig:
    """The attention layer config."""

    # QKV can either be stored as split (for easier postprocessing)
    # or merged (for TRT LLM export)
    qkv: QKVConfig | LinearConfig | None = None
    dense: LinearConfig | None = None
    # KV cache related attributes
    k_cache_scaling_factor: torch.Tensor = None
    v_cache_scaling_factor: torch.Tensor = None
    k_cache_bias: torch.Tensor = None
    v_cache_bias: torch.Tensor = None
    kv_cache_dtype: str | None = None

    rotary_dim: int = -math.inf
    # MPT variants
    clip_qkv: float | None = None
    # T5
    rel_attn_table: RelativeAttentionTableConfig | None = None

    # Mllama
    q_layernorm: LayernormConfig | None = None
    k_layernorm: LayernormConfig | None = None


@dataclass
class MLPConfig:
    """The MLP layer config."""

    fc: LinearConfig | None = None
    gate: LinearConfig | None = None
    proj: LinearConfig | None = None
    hidden_act: str = ""
    merge_gate_fc: bool = False


@dataclass
class ExpertConfig:
    """The Expert config."""

    # Aligning the naming conversion with TRT-LLM
    fc: LinearConfig | None = None  # stacked experts for concatenated w3 and w1
    proj: LinearConfig | None = None  # stacked experts for w2


@dataclass
class RgLruConfig:
    """The RG LRU from recurrentgemma."""

    recurrent_param: torch.Tensor = None
    input_gate: LinearConfig | None = None
    recurrent_gate: LinearConfig | None = None


@dataclass
class RecurrentConfig:
    """The RecurrentBlock from recurrentgemma."""

    linear_y: LinearConfig | None = None
    y_bias: torch.Tensor = None
    linear_x: LinearConfig | None = None
    linear_out: LinearConfig | None = None
    conv1d: ConvConfig | None = None
    rg_lru: RgLruConfig | None = None


@dataclass
class MOEConfig:
    """The Mixture of Expert layer config."""

    router: LinearConfig | None = None
    experts: ExpertConfig | None = None
    shared_expert: MLPConfig | None = None  # Deepseek MOE
    shared_expert_gate: LinearConfig | None = None  # Qwen MOE
    hidden_act: str = ""

    @property
    def fc(self):
        """Return the fc module from experts."""
        assert self.experts is not None, "MoE config requires experts."
        return self.experts.fc


@dataclass
class DecoderLayerConfig:
    """The decoder layer config."""

    decoder_type: str = ""
    input_layernorm: LayernormConfig | None = None
    mlp_layernorm: LayernormConfig | None = None
    attention: AttentionConfig | None = None
    recurrent: RecurrentConfig | None = None  # recurrent gemma
    post_layernorm: LayernormConfig | None = None
    pre_feedforward_layernorm: LayernormConfig | None = None  # gemma 2
    post_feedforward_layernorm: LayernormConfig | None = None  # gemma 2
    mlp: MLPConfig | MOEConfig | None = None

    num_attention_heads: int = 0
    # Supporting different attention_head_size per layer.
    attention_head_size: int | None = None

    num_kv_heads: int = 0
    max_position_embeddings: int = 0
    rotary_pct: float = 1.0

    # Falcon and Baichuan variants
    use_alibi: bool = False
    new_decoder_architecture: bool = False
    parallel_attention: bool = False

    # chatglm variants
    apply_residual_connection_post_layernorm: bool = False
    use_cache: bool = True
    chatglm_version: str = ""
    rope_ratio: float = 1.0

    # Qwen config
    seq_length: int = 0
    qwen_type: str = ""

    # Qwen and CodeLlama
    rotary_base: int = 0

    # Phi
    partial_rotary_factor: float = 0

    # Phi3
    original_max_position_embeddings: int = 0
    longrope_scaling_short_factors: list[float] | None = None
    longrope_scaling_long_factors: list[float] | None = None

    # Phi3 small
    mup_attn_multiplier: float = 0
    mup_embedding_multiplier: float = 0
    mup_use_scaling: float = 0
    mup_width_multiplier: float = 0
    blocksparse_block_size: int = 0
    blocksparse_homo_head_pattern: bool = False
    blocksparse_num_local_blocks: int = 0
    blocksparse_vertical_stride: int = 0
    dense_attention_every_n_layers: int = 0
    gegelu_limit: float = 0
    longrope_short_mscale: float = 0
    longrope_long_mscale: float = 0

    # Mixture of Experts
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_tp_mode: int = 0
    moe_renorm_mode: int = 0

    # MPT
    alibi_bias_max: int = 0

    # Arctic variants
    residual_layernorm: LayernormConfig | None = None
    residual_mlp: MLPConfig | None = None

    # Recurrent Gemma
    rnn_hidden_size: int = 0
    logits_soft_cap: float = 0
    emb_scale_by_sqrt_dim: bool = False
    layer_types: list[str] = field(default_factory=list)

    # Deci models
    attn_replacing_linear: LinearConfig | None = None
    mlp_replacing_linear: LinearConfig | None = None
    block_config: dict | None = None

    # Gemma2
    final_logit_softcapping: float = 0
    attn_logit_softcapping: float = 0
    query_pre_attn_scalar: float = 0

    # MPT
    clip_qkv: int = 0

    # T5, Mllama
    cross_attention: AttentionConfig | None = None
    cross_attention_layernorm: LayernormConfig | None = None
    self_attention: AttentionConfig | None = None
    self_attention_layernorm: LayernormConfig | None = None
    attention_layernorm: LayernormConfig | None = None
    rel_attn_max_distance: int = 0
    rel_attn_num_buckets: int = 0

    # Llama 3.1
    rope_scaling: dict | None = None

    # Mllama
    cross_attention_layers: dict | None = None
    vision_output_dim: int = 0
    gate_ffwd: torch.Tensor = None
    gate_attn: torch.Tensor = None

    # Phi3.5 MOE
    sparse_mixer_epsilon: float = 0

    # Mcore
    position_embedding_type: str | None = None

    @property
    def hidden_size(self):
        """Returns the hidden size of the transformer model."""
        assert self.mlp is not None and self.mlp.fc is not None, "MLP projection is required."
        fc = self.mlp.fc
        k = fc.weight.shape[-1] if isinstance(self.mlp, MOEConfig) else fc.weight.shape[1]
        if fc.quantization in [
            QUANTIZATION_NVFP4,
            QUANTIZATION_NVFP4_AWQ,
            QUANTIZATION_NVFP4_SVDQUANT,
        ]:
            return k * 2
        return k

    @property
    def ffn_hidden_size_local(self):
        """Returns the ffn hidden size of the transformer model."""
        assert self.mlp is not None and self.mlp.fc is not None, "MLP projection is required."
        fc = self.mlp.fc
        # fc in MoE merge fc and gate
        k = fc.weight.shape[1] // 2 if isinstance(self.mlp, MOEConfig) else fc.weight.shape[0]
        if fc.quantization not in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
            return k
        return k * 2


@dataclass
class MedusaHeadConfig:
    """The decoder layer config."""

    medusa_layers: list[LinearActConfig] | None = None
    lm_head: LinearConfig | None = None


@dataclass
class ModelConfig:
    """The full LLM model config that includes the full information needed for tensorrt_llm engine building.

    This class includes all the fields that tensorrt_llm supports, but not all of the fields are required.
    pipeline_parallel > 1 is only supported for TensorRT-LLM checkpoint.
    """

    # Global metadata
    architecture: str = ""
    quantization: str = QUANTIZATION_NONE
    dtype: str = "float16"
    vocab_size: int = 0

    # Parallel metadata
    rank: int = 0
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

    # Model structure and weights
    vocab_embedding: EmbeddingConfig | None = None
    position_embedding: EmbeddingConfig | None = None
    block_embedding: EmbeddingConfig | None = None
    ln_embed: LayernormConfig | None = None
    layers: list[DecoderLayerConfig] = field(default_factory=list)

    ln_f: LayernormConfig | None = None

    lm_head: LinearConfig | None = None
    share_embedding_table: bool = False

    medusa_heads: list[MedusaHeadConfig] | None = None
    num_medusa_heads: int = 0
    num_medusa_layers: int = 0

    # To differentiate encoder and decoder of Encoder-Decoder model
    enc_dec: str = ""

    # For decoder of Encoder-Decoder model that needs encoder information
    encoder_hidden_size: int = 0
    encoder_num_heads: int = 0
    encoder_head_size: int = 0
    decoder_start_token_id: int | None = None
    eos_token_id: int | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = None

    # For whisper encoder feature extractor
    conv1: ConvConfig | None = None
    conv2: ConvConfig | None = None

    @property
    def vocab_size_padded(self):
        """Returns the padded vocab_size of the model rounds to the tensor_parallel."""

        def _pad_vocab_size(vocab_size, tp_size):
            return int(math.ceil(vocab_size / tp_size) * tp_size)

        return _pad_vocab_size(self.vocab_size, self.tensor_parallel)

    @property
    def hidden_size(self):
        """Returns the hidden_size of the model."""
        return self.layers[0].hidden_size

    @property
    def max_position_embeddings(self):
        """Returns the max_position_embedding of the model."""
        return self.layers[0].max_position_embeddings

    @property
    def num_attention_heads(self):
        """Returns the num_attention_heads of the model."""
        return self.layers[0].num_attention_heads

    @property
    def num_kv_heads(self):
        """Returns the num_key_value_heads of the model."""
        return (
            self.layers[0].num_kv_heads
            if self.layers[0].num_kv_heads is not None and self.layers[0].num_kv_heads > 0
            else self.num_attention_heads
        )

    @property
    def hidden_act(self):
        """Returns the hidden_act of the model."""
        mlp = self.layers[0].mlp
        assert mlp is not None, "MLP config is required."
        return mlp.hidden_act


# Register all config classes as safe globals
torch.serialization.add_safe_globals(
    [
        EmbeddingConfig,
        LayernormConfig,
        LinearConfig,
        LinearActConfig,
        ConvConfig,
        QKVConfig,
        RelativeAttentionTableConfig,
        AttentionConfig,
        MLPConfig,
        ExpertConfig,
        RgLruConfig,
        RecurrentConfig,
        MOEConfig,
        DecoderLayerConfig,
        MedusaHeadConfig,
        ModelConfig,
    ]
)
