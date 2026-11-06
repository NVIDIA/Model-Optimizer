# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text configuration for the published Kimi-K3.1 architecture."""

import copy
from collections.abc import Mapping

from transformers.configuration_utils import PreTrainedConfig

__all__ = ["KimiK31Config"]


class KimiK31Config(PreTrainedConfig):
    """K3.1 text backbone, accepting either text fields or the published nested config."""

    model_type = "kimi_k31"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    attribute_map = {
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_token",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "num_shared_experts",
        "norm_topk_prob": "moe_renormalize",
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        """Load the complete K3.1 text architecture and retain optional vision metadata."""
        values = dict(
            text_config.to_dict() if hasattr(text_config, "to_dict") else text_config or {}
        )
        values.update(kwargs)
        values.pop("model_type", None)
        defaults = {
            "vocab_size": 163840,
            "hidden_size": 7168,
            "intermediate_size": 33792,
            "moe_intermediate_size": 3072,
            "routed_expert_hidden_size": 3584,
            "num_hidden_layers": 93,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "num_expert_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "latent_moe_use_norm": True,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "hidden_act": "situ",
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "attn_res_block_size": 12,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "max_position_embeddings": 1048576,
            "attention_dropout": 0.0,
            "use_cache": True,
            "kda_implementation": "auto",
        }
        for name, default in defaults.items():
            setattr(self, name, values.pop(name, default))
        self.vision_config = copy.deepcopy(vision_config)
        self.checkpoint_quantization_config = copy.deepcopy(
            values.pop("quantization_config", values.pop("checkpoint_quantization_config", None))
        )
        linear = values.pop("linear_attn_config", None)
        if linear is None:
            full = [
                i
                for i in range(1, self.num_hidden_layers + 1)
                if i % 4 == 0 or i == self.num_hidden_layers
            ]
            linear = {
                "kda_layers": [i for i in range(1, self.num_hidden_layers + 1) if i not in full],
                "full_attn_layers": full,
                "num_heads": self.num_attention_heads,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            }
        self.linear_attn_config = copy.deepcopy(linear)
        self.dsa_config = copy.deepcopy(
            values.pop("dsa_config", None)
            or {
                "dsa_layers": linear["full_attn_layers"],
                "index_topk": 2048,
                "index_num_heads": 32,
                "index_head_dim": 128,
                "index_rope_dim": 0,
                "index_k_norm_type": "layernorm",
                "index_topk_freq": 1,
                "index_skip_topk_offset": 1,
            }
        )
        self.layer_types = [
            "linear_attention" if i + 1 in linear["kda_layers"] else "indexed_attention"
            for i in range(self.num_hidden_layers)
        ]
        self.number_of_conv_states = 3
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        for name, default in [
            ("pad_token_id", 163839),
            ("bos_token_id", 163584),
            ("eos_token_id", 163586),
        ]:
            values.setdefault(name, default)
        values.setdefault("tie_word_embeddings", False)
        super().__init__(**values)
        self._validate_architecture()

    def _validate_architecture(self):
        kda = set(self.linear_attn_config["kda_layers"])
        full = set(self.linear_attn_config["full_attn_layers"])
        if kda & full or kda | full != set(range(1, self.num_hidden_layers + 1)):
            raise ValueError("KDA and DSA layers must partition the 1-indexed text layers.")
        if full != set(self.dsa_config["dsa_layers"]):
            raise ValueError("K3.1 requires DSA at every non-KDA layer.")
        dsa = self.dsa_config
        if (
            dsa.get("index_rope_dim", 0) != 0
            or dsa.get("index_k_norm_type", "layernorm") != "layernorm"
        ):
            raise ValueError("K3.1 requires a NoPE, LayerNorm DSA indexer.")
        if dsa.get("index_topk_freq", 1) != 1:
            raise ValueError("K3.1 requires an independent indexer at every DSA layer.")
        if not self.mla_use_nope or not self.mla_use_output_gate or self.hidden_act != "situ":
            raise ValueError("K3.1 requires NoPE gated MLA and SiTU MLPs.")
        if not 1 <= self.num_experts_per_token <= self.num_experts:
            raise ValueError("Invalid routed expert top-k.")
        if self.moe_router_activation_func not in ("sigmoid", "softmax"):
            raise ValueError("K3.1 router activation must be sigmoid or softmax.")
        if (
            self.num_experts % self.num_expert_group
            or not 1 <= self.topk_group <= self.num_expert_group
        ):
            raise ValueError("Invalid grouped routing dimensions.")
        if self.activation_situ_beta <= 0 or self.activation_situ_linear_beta <= 0:
            raise ValueError("SiTU scales must be positive.")
        if self.attn_res_block_size is not None and self.attn_res_block_size <= 0:
            raise ValueError("Attention residual block size must be positive.")
        if self.kda_implementation not in ("auto", "fla", "torch"):
            raise ValueError("KDA implementation must be auto, fla or torch.")

    @classmethod
    def from_hf_config(cls, config, **overrides):
        """Extract the text architecture from an HF root config, with explicit overrides."""
        raw = copy.deepcopy(dict(config) if isinstance(config, Mapping) else config.to_dict())
        text = raw.pop("text_config", raw)
        text = text.to_dict() if hasattr(text, "to_dict") else dict(text)
        text.update(overrides)
        vision = text.pop("vision_config", raw.get("vision_config"))
        return cls(vision_config=vision, **text)
