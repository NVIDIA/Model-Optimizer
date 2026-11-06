# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the published Step-5 Preview text architecture."""

import copy

from transformers import PretrainedConfig

__all__ = ["Step5Config", "Step5SparseConfig"]


class Step5SparseConfig(PretrainedConfig):
    """CSA compression, provider grouping, and historical region budget."""

    model_type = "step5_sparse"

    def __init__(
        self,
        enabled=True,
        proxy_dim=256,
        sparse_indexer_rope_dim=32,
        sparse_indexer_use_rope=True,
        sparse_indexer_num_heads=16,
        sparse_indexer_num_k_heads=1,
        sparse_indexer_q_norm_type="rmsnorm",
        sparse_indexer_k_norm_type="layernorm",
        sparse_indexer_csa_z_norm_type="none",
        sparse_indexer_softmax_variant="ssmax",
        sparse_indexer_ssmax_s_granularity="q_head",
        num_provider_groups=4,
        topk=512,
        region_block_size=8,
        compression_method="csa_block_compress",
        attention_impl="sparse_gqa",
        apply_to_layer_types=None,
        **kwargs,
    ):
        """Validate the released CSA semantics and selection geometry."""
        super().__init__(**kwargs)
        values = locals().copy()
        for key in ("self", "kwargs", "__class__"):
            values.pop(key, None)
        for key, value in values.items():
            setattr(self, key, value)
        self.apply_to_layer_types = (
            ["full_attention"] if apply_to_layer_types is None else list(apply_to_layer_types)
        )
        semantics = (
            sparse_indexer_num_k_heads,
            sparse_indexer_q_norm_type,
            sparse_indexer_k_norm_type,
            sparse_indexer_csa_z_norm_type,
            sparse_indexer_softmax_variant,
            sparse_indexer_ssmax_s_granularity,
            compression_method,
            attention_impl,
        )
        if enabled and semantics != (
            1,
            "rmsnorm",
            "layernorm",
            "none",
            "ssmax",
            "q_head",
            "csa_block_compress",
            "sparse_gqa",
        ):
            raise ValueError(f"Unsupported Step-5 sparse semantics: {semantics}")
        if (
            min(proxy_dim, sparse_indexer_num_heads, num_provider_groups, topk, region_block_size)
            < 1
        ):
            raise ValueError("Step-5 sparse dimensions and budget must be positive")
        if sparse_indexer_num_heads % num_provider_groups:
            raise ValueError("Indexer heads must divide evenly into provider groups")
        if sparse_indexer_rope_dim % 2 or not 0 <= sparse_indexer_rope_dim <= proxy_dim:
            raise ValueError("Indexer rotary dimension must be even and within proxy_dim")


class Step5Config(PretrainedConfig):
    """Step-5 Preview language configuration, including checkpoint-compatible field aliases."""

    model_type = "step5"
    base_config_key = "text_config"
    sub_configs = {"sparse_config": Step5SparseConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_gather_output",
        "layers.*.self_attn.g_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.moe.experts.gate_up_proj": "packed_colwise",
        "layers.*.moe.experts.down_proj": "rowwise",
        "layers.*.moe.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.share_expert.gate_proj": "colwise",
        "layers.*.share_expert.up_proj": "colwise",
        "layers.*.share_expert.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.moe.experts.gate_up_proj": "grouped_gemm",
        "layers.*.moe.experts.down_proj": "grouped_gemm",
        "layers.*.moe.experts": "ep_dispatch_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=128896,
        hidden_size=4096,
        intermediate_size=13824,
        num_hidden_layers=92,
        num_attention_heads=64,
        num_attention_groups=4,
        head_dim=192,
        layer_types=None,
        rope_theta=None,
        partial_rotary_factors=None,
        rope_scaling=None,
        yarn_only_types=None,
        sliding_window=512,
        rms_norm_eps=1e-5,
        fp32_residual_connection=True,
        use_head_wise_attn_gate=True,
        use_moe=True,
        moe_layer_list=None,
        moe_num_experts=352,
        moe_top_k=8,
        moe_intermediate_size=1536,
        share_expert_dim=1536,
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
        swiglu_limits=None,
        swiglu_limits_shared=None,
        sparse_config=None,
        max_position_embeddings=1048576,
        initializer_range=0.02,
        use_cache=False,
        **kwargs,
    ):
        """Build a text configuration using the released per-layer defaults."""
        if rope_scaling is None:
            rope_scaling = kwargs.pop("rope_parameters", None)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("architectures", ["Step5ForCausalLM"])
        self.checkpoint_quantization_config = copy.deepcopy(
            kwargs.pop("quantization_config", kwargs.pop("checkpoint_quantization_config", None))
        )
        super().__init__(**kwargs)
        values = locals().copy()
        for key in ("self", "kwargs", "__class__"):
            values.pop(key, None)
        for key, value in values.items():
            setattr(self, "rope_parameters" if key == "rope_scaling" else key, value)
        n = num_hidden_layers
        serialized_layer_types = layer_types or [
            "full_attention" if i % 4 == 3 else "sliding_attention" for i in range(n)
        ]
        self.layer_types = serialized_layer_types[:n]
        self.mtp_layer_types = serialized_layer_types[n:] or kwargs.get("mtp_layer_types", [])
        self.rope_theta = rope_theta or [
            1e7 if kind == "full_attention" else 1e4 for kind in self.layer_types
        ]
        self.partial_rotary_factors = partial_rotary_factors or [
            1 / 3 if kind == "full_attention" else 1.0 for kind in self.layer_types
        ]
        self.yarn_only_types = ["full_attention"] if yarn_only_types is None else yarn_only_types
        nested = rope_scaling and any(kind in rope_scaling for kind in self.layer_types)
        self.rope_parameters = {}
        for i, kind in enumerate(self.layer_types):
            theta = self.rope_theta[i] if isinstance(self.rope_theta, list) else self.rope_theta
            if nested:
                scaling = rope_scaling.get(kind, {})
            else:
                scaling = (rope_scaling or {}) if kind in self.yarn_only_types else {}
            self.rope_parameters.setdefault(
                kind, {"rope_type": "default", **scaling, "rope_theta": theta}
            )
        self.swiglu_limits = [7.0] * n if swiglu_limits is None else swiglu_limits
        self.swiglu_limits_shared = (
            [7.0] * n if swiglu_limits_shared is None else swiglu_limits_shared
        )
        enum = kwargs.get("moe_layers_enum", list(range(3, min(n - 1, 91))))
        if isinstance(enum, str):
            enum = [int(part) for part in enum.split(",") if part]
        self.moe_layer_list = list(enum if moe_layer_list is None else moe_layer_list)
        self.moe_layers_enum = self.moe_layer_list
        self.num_key_value_heads = num_attention_groups
        self.num_experts = self.num_local_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.hidden_act = "silu"
        self.sparse_config = (
            sparse_config
            if isinstance(sparse_config, Step5SparseConfig)
            else Step5SparseConfig(**(sparse_config or {}))
        )
        if not fp32_residual_connection or kwargs.get("norm_dtype", "float32") != "float32":
            raise ValueError("Step-5 requires FP32 residual connections and normalization")
        if kwargs.get("att_impl_type", "GQA") != "GQA" or kwargs.get("attention_dropout", 0) != 0:
            raise ValueError("Step-5 requires deterministic GQA attention")
        if kwargs.get("moe_router_activation", "sigmoid") != "sigmoid":
            raise ValueError("Step-5 requires sigmoid routing")
        if num_attention_heads % num_attention_groups:
            raise ValueError("Attention query heads must divide evenly into KV groups")
        if not 1 <= moe_top_k <= moe_num_experts:
            raise ValueError("Step-5 MoE top-k is outside the expert count")
        if (
            self.sparse_config.enabled
            and self.sparse_config.num_provider_groups != num_attention_groups
        ):
            raise ValueError("CSA provider groups must equal attention KV groups")
        for name in (
            "layer_types",
            "partial_rotary_factors",
            "swiglu_limits",
            "swiglu_limits_shared",
        ):
            if len(getattr(self, name)) < n:
                raise ValueError(f"Step-5 {name} must cover num_hidden_layers entries")
        if isinstance(self.rope_theta, list) and len(self.rope_theta) < n:
            raise ValueError("Step-5 rope_theta must cover num_hidden_layers entries")
        if any(kind not in ("sliding_attention", "full_attention") for kind in self.layer_types):
            raise ValueError("Unsupported Step-5 attention layer type")
        for scaling in self.rope_parameters.values():
            if scaling["rope_type"] not in ("default", "llama3"):
                raise ValueError("Step-5 supports only default or llama3 rotary scaling")
