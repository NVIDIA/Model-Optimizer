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

"""Utils for TensorRT-LLM checkpoint export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

from .layer_utils import model_type_is_enc_dec
from .tensorrt_llm_type import LayerNormPositionType, LayerNormType, MLPType

if TYPE_CHECKING:
    from transformers import T5Config

from modelopt import __version__

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    QUANTIZATION_NONE,
    DecoderLayerConfig,
    MLPConfig,
    ModelConfig,
)

logger = logging.getLogger(__name__)

# For NEMO and ENC/DEC models where the TensorRT-LLM model architecture and HF config are not aligned.
MODEL_NAME_TO_HF_ARCH_MAP = {
    "llama": "LlamaForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma3": "Gemma3ForCausalLM",
    "gpt": "GPTForCausalLM",
    "qwen": "QWenForCausalLM",
    "enc": "EncoderModel",
    "dec": "DecoderModel",
    "mllama": "MLLaMAModel",
    "whisper_encoder": "WhisperEncoder",
}


def is_tensorrt_llm_0_8_or_9():
    """Returns true if tensorrt_llm version is 0.8 or 0.9."""
    try:
        import tensorrt_llm

        return tensorrt_llm.__version__.startswith(("0.8", "0.9"))
    except Exception:
        return False


def _find_layernorm_type(model_config: ModelConfig):
    if model_config.ln_f:
        return model_config.ln_f.layernorm_type
    for layer in model_config.layers:
        if layer.input_layernorm:
            return layer.input_layernorm.layernorm_type
        if layer.post_layernorm:
            return layer.post_layernorm.layernorm_type
    return LAYERNORM_DEFAULT


def convert_to_tensorrt_llm_config(
    model_config: ModelConfig,
    quant_config: dict[str, Any],
    hf_config=None,
):
    """Convert to TensorRT-LLM checkpoint config.

    Args:
        model_config: The model_config to convert.
        quant_config: The quantization config to convert. It will be updated with kv_cache_quant_algo.
        hf_config: The huggingface model config.
            If provided, we try to use the TensorRT-LLM's export method if available.
    """
    tp_size = model_config.tensor_parallel
    pp_size = model_config.pipeline_parallel

    first_attention_config = None
    first_attention_decoder_config = None
    first_mlp_decoder_config = None
    for decoder_layer in model_config.layers:
        first_attention_config = (
            decoder_layer.attention or decoder_layer.self_attention or decoder_layer.cross_attention
        )
        if first_attention_config and not first_attention_decoder_config:
            first_attention_decoder_config = decoder_layer

        first_mlp_decoder_config = decoder_layer if decoder_layer.mlp else None

        if first_attention_decoder_config and first_mlp_decoder_config:
            break

    assert (
        first_attention_config is not None
        and first_attention_decoder_config is not None
        and first_mlp_decoder_config is not None
    ), "Model must have at least one attention block and one MLP block"

    mapping = {
        "world_size": tp_size * pp_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
    }

    decoder_type = model_config.layers[0].decoder_type

    config_architecture = model_config.architecture
    if not config_architecture:
        config_architecture = MODEL_NAME_TO_HF_ARCH_MAP[decoder_type]
    # For Encoder-Decoder model
    is_enc_dec = model_type_is_enc_dec(decoder_type)
    if is_enc_dec:
        # For encoder
        if model_config.enc_dec == "enc":
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP[
                f"{decoder_type}_encoder" if decoder_type in ["whisper"] else "enc"
            ]
        # For decoder
        else:
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP["dec"]

    config = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "architecture": config_architecture,
        "dtype": model_config.dtype,
        "logits_dtype": "float16" if model_config.dtype == "bfloat16" else model_config.dtype,
        "num_hidden_layers": len(model_config.layers) * pp_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "hidden_size": first_mlp_decoder_config.hidden_size,
        "norm_epsilon": (
            first_attention_decoder_config.mlp_layernorm.eps
            if is_enc_dec
            else first_attention_decoder_config.input_layernorm.eps
        ),
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "hidden_act": first_mlp_decoder_config.mlp.hidden_act,
        "use_parallel_embedding": True,
        "embedding_sharding_dim": 0,
        "head_size": first_attention_decoder_config.attention_head_size,
        "intermediate_size": first_mlp_decoder_config.ffn_hidden_size_local * tp_size,
        "position_embedding_type": (
            "alibi"
            if first_attention_decoder_config.use_alibi
            else (
                first_attention_decoder_config.position_embedding_type
                if first_attention_decoder_config.position_embedding_type
                else "rope_gpt_neox"
            )
        ),
        "share_embedding_table": bool(model_config.lm_head is None and pp_size == 1),
        "residual_mlp": first_attention_decoder_config.residual_mlp is not None,
        # Model Optimizer customized fields
        "bias": first_attention_config.dense.bias is not None,
        "rotary_pct": first_attention_decoder_config.rotary_pct,
        "rank": model_config.rank,
        "decoder": first_attention_decoder_config.decoder_type,
        "rmsnorm": _find_layernorm_type(model_config) == LAYERNORM_RMS,
        "lm_head_bias": model_config.lm_head is not None and model_config.lm_head.bias is not None,
    }

    # Update quantization config
    if model_config.quantization != QUANTIZATION_NONE:
        # In TRT LLM, the embedding table is shared for the following models, so lm_head quantization format
        # won't be automatically detected in the excluded_modules. We need to manually add it to the exclusions.
        share_embedding_table = model_config.lm_head is None and pp_size == 1
        if share_embedding_table:
            quant_config.setdefault("exclude_modules", []).append("lm_head")

    quant_config["kv_cache_quant_algo"] = first_attention_config.kv_cache_dtype

    if hf_config is not None:
        try:
            from tensorrt_llm.models import MODEL_MAP
            from tensorrt_llm.models.modeling_utils import Mapping, QuantConfig

            model_cls = MODEL_MAP[config_architecture]
            config_cls = getattr(model_cls, "config_class", None)

            if config_cls is not None:
                tensorrt_llm_config = config_cls.from_hugging_face(
                    hf_config_or_dir=hf_config,
                    dtype=model_config.dtype,
                    mapping=Mapping.from_dict(mapping),
                    quant_config=QuantConfig.from_dict(
                        {k: v for k, v in quant_config.items() if k != "quantized_layers"}
                    ),
                ).to_dict()

                config.update(tensorrt_llm_config)
                # Modelopt uses parallel_embedding by default.
                config.update(
                    {
                        "use_parallel_embedding": True,
                        "embedding_sharding_dim": 0,
                        "logits_dtype": "float16"
                        if model_config.dtype == "bfloat16"
                        else model_config.dtype,
                    }
                )

                return config

        except Exception as e:
            warn(
                "Cannot export tensorrt_llm checkpoint config due to the following error. "
                f"Trying the manual approach: \n{e}"
            )

    if first_attention_decoder_config.rotary_base:
        config["rotary_base"] = first_attention_decoder_config.rotary_base

    if first_attention_decoder_config.rope_scaling:
        config["rotary_scaling"] = first_attention_decoder_config.rope_scaling

    config["mapping"] = mapping
    config["quantization"] = quant_config

    layernorm_type_map = {i.name: i.value for i in LayerNormType}
    layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}

    if decoder_type in ["gpt", "gemma", "llama", "qwen"]:
        pass
    elif decoder_type == "mpt":
        config.update(
            {
                "clip_qkv": first_attention_config.clip_qkv,
                "alibi_bias_max": model_config.layers[0].alibi_bias_max,
            }
        )
    elif decoder_type == "recurrentgemma":
        config["conv_kernel"] = 4
        config["state_size"] = 1
        config["state_dtype"] = "float32"
        config["rnn_hidden_size"] = model_config.layers[0].rnn_hidden_size
        config["rnn_conv_dim_size"] = model_config.layers[0].rnn_hidden_size
        config["logits_soft_cap"] = model_config.layers[0].logits_soft_cap
        config["emb_scale_by_sqrt_dim"] = model_config.layers[0].emb_scale_by_sqrt_dim
        config["layer_types"] = model_config.layers[0].layer_types
    elif is_enc_dec:
        if decoder_type in ["whisper"]:
            config["n_mels"] = hf_config.num_mel_bins
            model_is_multilingual = hf_config.vocab_size >= 51865
            num_languages = hf_config.vocab_size - 51765 - int(model_is_multilingual)
            config["num_languages"] = num_languages

        # T5 models use relative position embedding
        # Bart models use learned_absolute position embedding
        config["position_embedding_type"] = (
            "relative" if decoder_type == "t5" else "learned_absolute"
        )
        config["share_embedding_table"] = getattr(model_config, "share_embedding_table")
        config["has_position_embedding"] = bool(getattr(model_config, "position_embedding"))
        # fallback to RmsNorm if not specified as HF might change class naming for layernorm
        layernorm_type = model_config.layers[0].mlp_layernorm.layernorm_type
        if not layernorm_type:
            layernorm_type = "RmsNorm"
        config["layernorm_type"] = layernorm_type_map[layernorm_type]
        config["has_attention_qkvo_bias"] = bool(
            model_config.layers[0].attention.qkv.bias is not None
            if model_config.enc_dec == "enc"
            else model_config.layers[0].self_attention.qkv.bias is not None
        )
        config["has_mlp_bias"] = model_config.layers[0].mlp.fc.bias is not None
        config["has_model_final_layernorm"] = bool(model_config.ln_f)

        mlp_type_map = {i.name: i.value for i in MLPType}

        config["mlp_type"] = mlp_type_map[
            (
                "GatedMLP"
                if isinstance(model_config.layers[0].mlp, MLPConfig)
                and model_config.layers[0].mlp.gate
                else "MLP"
            )
        ]
        config["use_prompt_tuning"] = False
        config["has_position_embedding"] = bool(model_config.position_embedding)
        config["has_embedding_layernorm"] = bool(model_config.ln_embed)
        config["has_embedding_scale"] = False
        config["ffn_hidden_size"] = model_config.layers[0].mlp.fc.weight.shape[0]
        # T5 uses q_scaling to offset attention scaling effect. Bart does not.
        config["q_scaling"] = 1 / config["head_size"] ** 0.5 if decoder_type == "t5" else 1.0
        config["layernorm_position"] = (
            layernorm_position_map["post_layernorm"]
            if decoder_type == "bart"
            else layernorm_position_map["pre_layernorm"]
        )
        config["relative_attention"] = config["position_embedding_type"] == "relative"
        config["max_distance"] = model_config.layers[0].rel_attn_max_distance
        config["num_buckets"] = model_config.layers[0].rel_attn_num_buckets
        config["model_type"] = decoder_type
        config["use_parallel_embedding"] = True
        config["use_implicit_relative_attention"] = False
        if model_config.enc_dec == "dec":
            config["rescale_before_lm_head"] = False
            config["encoder_hidden_size"] = model_config.encoder_hidden_size
            config["encoder_num_heads"] = model_config.encoder_num_heads
            config["encoder_head_size"] = model_config.encoder_head_size
            config["decoder_start_token_id"] = model_config.decoder_start_token_id
            config["eos_token_id"] = model_config.eos_token_id
            config["bos_token_id"] = model_config.bos_token_id
            config["pad_token_id"] = model_config.pad_token_id
    elif decoder_type == "dbrx":
        config["clip_qkv"] = first_attention_decoder_config.clip_qkv

    elif decoder_type == "mllama":
        num_layers = config["num_hidden_layers"]
        num_kv_heads = model_config.num_kv_heads
        cross_attention_layers = set(model_config.layers[0].cross_attention_layers)

        config["num_kv_heads_per_layer"] = [
            0 if i in cross_attention_layers else num_kv_heads for i in range(num_layers)
        ]
        config["num_kv_heads_per_cross_attn_layer"] = [
            num_kv_heads if i in cross_attention_layers else 0 for i in range(num_layers)
        ]

        config["cross_attention"] = True
        config["cross_attention_layers"] = model_config.layers[0].cross_attention_layers
        config["embed_vocab_size"] = model_config.vocab_size + 8
        vision_output_dim = model_config.layers[0].vision_output_dim
        config["vision_output_dim"] = vision_output_dim if vision_output_dim != 0 else 7680
    else:
        raise NotImplementedError(
            f"Cannot export tensorrt_llm checkpoint for model {decoder_type}: {config_architecture}. "
            "It's not supported by TensorRT-LLM yet. Please try exporting the model to unified HF "
            "checkpoint with ModelOpt and deploy the checkpoint with TensorRT-LLM pytorch backend."
        )

    # For Mixtral and Arctic
    if first_attention_decoder_config.moe_num_experts:
        config["moe"] = {
            "num_experts": first_attention_decoder_config.moe_num_experts,
            "top_k": first_attention_decoder_config.moe_top_k,
            "normalization_mode": 1,  # ExpertScaleNormalizationMode.RENORMALIZE
        }

        if decoder_type == "phi3":
            config["moe"]["normalization_mode"] = 2  # ExpertScaleNormalizationMode.SPARSE_MIXER,
            config["moe"]["sparse_mixer_epsilon"] = (
                first_attention_decoder_config.sparse_mixer_epsilon
            )

        config["mapping"]["moe_tp_size"] = config["mapping"]["tp_size"]
        config["mapping"]["moe_ep_size"] = 1

    # Handle Medusa decoding
    # TODO (chenhany): when inference pp > 1; only last pp has medusa heads
    if model_config.medusa_heads is not None:
        config["base_architecture"] = config["architecture"]
        config["architecture"] = "MedusaForCausalLM"
        # NOTE: max_draft_len is related to the medusa tree len. Currently it is hardcoded to 63.
        config["max_draft_len"] = 63
        config["num_medusa_heads"] = len(model_config.medusa_heads)
        config["num_medusa_layers"] = len(model_config.medusa_heads[0].medusa_layers)

    return config


def prepare_enc_dec_export_dir(tensorrt_llm_config: dict[str, Any], export_root: Path):
    """Prepare the export directory for encoder-decoder model."""
    # For encoder
    if tensorrt_llm_config["architecture"] in ["EncoderModel", "WhisperEncoder"]:
        export_dir = export_root.joinpath("encoder")
    # For decoder
    else:
        export_dir = export_root.joinpath("decoder")
    return export_dir


def prepare_t5_decoder_layer(
    layer_config: DecoderLayerConfig,
    model_config: "T5Config",
    enc_dec: str,
    layers: list[DecoderLayerConfig],
):
    """Prepare the config for each decoder layer of encoder-decoder model."""
    layer_config.rel_attn_max_distance = model_config.relative_attention_max_distance
    layer_config.rel_attn_num_buckets = model_config.relative_attention_num_buckets
    if enc_dec == "enc" and layer_config.attention.rel_attn_table is None:
        layer_config.attention.rel_attn_table = layers[0].attention.rel_attn_table
    elif enc_dec == "dec" and layer_config.self_attention.rel_attn_table is None:
        layer_config.self_attention.rel_attn_table = layers[0].self_attention.rel_attn_table
