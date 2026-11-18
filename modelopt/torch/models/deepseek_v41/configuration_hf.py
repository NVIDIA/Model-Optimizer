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


"""Configuration for the differentiable DeepSeek V4.1 text backbone."""

from transformers import PreTrainedConfig

__all__ = ["DeepseekV41Config"]


class DeepseekV41Config(PreTrainedConfig):
    """Accept the published root/text configuration and expose native HF text dimensions.

    Source checkpoint quantization metadata is retained as
    ``checkpoint_quantization_config``. Packed source tensors must be converted
    explicitly with ``DeepseekV41ForCausalLM.from_checkpoint`` before normal HF use.
    """

    model_type = "deepseek_v41"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts", "num_experts": "n_routed_experts"}
    base_model_tp_plan = {
        "layers.*.ffn.experts.gate_up_proj": "packed_colwise",
        "layers.*.ffn.experts.down_proj": "rowwise",
        "layers.*.ffn.experts": "moe_tp_experts",
        "layers.*.ffn.shared_experts.w1": "colwise",
        "layers.*.ffn.shared_experts.w3": "colwise",
        "layers.*.ffn.shared_experts.w2": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.ffn.experts.gate_up_proj": "grouped_gemm",
        "layers.*.ffn.experts.down_proj": "grouped_gemm",
        "layers.*.ffn.experts": "ep_dispatch_experts",
    }
    base_model_fsdp_plan = {
        "embed": "free_full_weight",
        "layers.*.attn.wq_a": "free_full_weight",
        "layers.*.attn.q_norm": "free_full_weight",
        "layers.*.attn.wq_b": "free_full_weight",
        "layers.*.attn.wkv": "free_full_weight",
        "layers.*.attn.kv_norm": "free_full_weight",
        "layers.*.attn.wo_a": "free_full_weight",
        "layers.*.attn.wo_b": "free_full_weight",
        "layers.*.attn": "free_full_weight",
        "layers.*.ffn": "free_full_weight",
        "layers.*.attn_norm": "free_full_weight",
        "layers.*.ffn_norm": "free_full_weight",
        "layers.*": "free_full_weight",
        "norm": "keep_full_weight",
    }

    def __init__(self, text_config=None, **kwargs):
        """Read published nested fields or explicitly supplied text dimensions."""
        values = dict(text_config or {})
        values.pop("model_type", None)
        values.update(kwargs)
        defaults = {
            "vocab_size": 129280,
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-20,
            "initializer_range": 0.02,
            "use_cache": True,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            },
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus",
            "gate_temp": 1.0,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 128,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092,
            "engram_token_map": None,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [37, 38, 39],
            "attention_quantization": True,
            "checkpoint_quantized_projections": [],
            "vision_routing_bias": False,
        }
        for name, default in defaults.items():
            setattr(self, name, values.pop(name, default))
        self.vision_config = values.pop("vision_config", None)
        self.vision_routing_bias = self.vision_routing_bias or bool(self.vision_config)
        self.checkpoint_quantization_config = values.pop(
            "quantization_config", values.pop("checkpoint_quantization_config", None)
        )
        values.setdefault("tie_word_embeddings", False)
        values.setdefault("bos_token_id", 0)
        values.setdefault("eos_token_id", 1)
        values.setdefault("pad_token_id", 2)
        values.setdefault("experts_implementation", "eager")
        super().__init__(**values)
        self._validate_text_architecture()
        self.base_model_fsdp_plan = dict(type(self).base_model_fsdp_plan)
        for name, layer_ids in (
            ("engram", self.engram_layer_ids),
            ("attn.compressor", self.kv_source_layer_ids),
            ("attn.compressor.wkv", self.kv_source_layer_ids),
            (
                "attn.compressor.wgate",
                [
                    layer
                    for layer in self.kv_source_layer_ids
                    if 0 <= layer < self.num_hidden_layers and self.compress_ratios[layer] > 1
                ],
            ),
            ("attn.indexer", self.index_source_layer_ids),
        ):
            if any(0 <= layer < self.num_hidden_layers for layer in layer_ids):
                self.base_model_fsdp_plan[f"layers.*.{name}"] = "free_full_weight"

    def _validate_text_architecture(self):
        if (
            self.num_key_value_heads != 1
            or getattr(self, "attention_bias", False)
            or getattr(self, "attention_dropout", 0.0)
        ):
            raise ValueError("Published V4.1 attention uses one KV head, no bias, and no dropout.")
        if self.hidden_act != "silu" or self.n_shared_experts != 1:
            raise ValueError("V4.1 requires SiLU and exactly one shared expert.")
        if not 0 < self.qk_rope_head_dim <= min(self.head_dim, self.index_head_dim):
            raise ValueError("V4.1 rotary width must fit the attention and index heads.")
        if self.qk_rope_head_dim % 2 or self.num_attention_heads % self.o_groups:
            raise ValueError("Invalid V4.1 rotary pairs or output projection groups.")
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError("V4.1 compress_ratios must cover all backbone layers.")
        kv_owner = index_owner = None
        for layer in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer]
            if ratio < 0:
                raise ValueError("V4.1 compression ratios must be nonnegative.")
            if layer in self.kv_source_layer_ids:
                if not ratio or layer not in self.index_source_layer_ids:
                    raise ValueError("A compressed KV owner must also own an indexer.")
                kv_owner = layer
            if layer in self.index_source_layer_ids:
                index_owner = layer
            if ratio and (
                kv_owner is None
                or index_owner is None
                or self.compress_ratios[kv_owner] != ratio
                or self.compress_ratios[index_owner] != ratio
            ):
                raise ValueError(
                    "V4.1 compressed attention requires preceding owners with the same ratio."
                )
        if self.attention_quantization and (
            self.head_dim % 32
            or self.index_head_dim % 32
            or (self.num_attention_heads * self.head_dim // self.o_groups) % 32
        ):
            raise ValueError("Published attention rounding requires widths divisible by 32.")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Each Engram layer requires its own embedding row count.")
        if self.engram_layer_ids and (self.engram_max_ngram_size < 2 or self.engram_n_heads < 1):
            raise ValueError(
                "Engram requires n-grams of size two or greater and at least one head."
            )
