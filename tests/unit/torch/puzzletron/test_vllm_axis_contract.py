from types import SimpleNamespace

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    MambaConfig,
    MLAConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.candidates import build_candidate_library
from modelopt.torch.puzzletron.utils.vllm_adapter import convert_block_configs_to_per_layer_config


def test_mla_search_axes_create_cartesian_typed_candidates() -> None:
    teacher = BlockConfig(
        subblock_configs=(MLAConfig(num_heads=16, q_lora_rank=768, kv_lora_rank=512),)
    )

    candidates = build_candidate_library(
        [teacher],
        search_space={
            "axes": {
                "mla_heads": {"enabled": True, "values": [8]},
                "mla_q_lora_rank": {"enabled": True, "values": [384]},
                "mla_kv_lora_rank": {"enabled": True, "values": [256]},
            }
        },
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=False,
    )

    configs = {
        (
            candidate.block_config.require_subblock("mla").num_heads,
            candidate.block_config.require_subblock("mla").q_lora_rank,
            candidate.block_config.require_subblock("mla").kv_lora_rank,
        )
        for candidate in candidates
    }
    assert configs == {
        (16, 768, 512),
        (8, 768, 512),
        (16, 384, 512),
        (8, 384, 512),
        (16, 768, 256),
        (8, 768, 256),
        (16, 384, 256),
        (8, 384, 256),
    }


def test_vllm_adapter_exports_per_layer_mla_ranks() -> None:
    config = SimpleNamespace(
        base_architecture="DeepseekV3ForCausalLM",
        num_hidden_layers=1,
        q_lora_rank=768,
        kv_lora_rank=512,
        block_configs=[
            BlockConfig(
                subblock_configs=(MLAConfig(num_heads=8, q_lora_rank=384, kv_lora_rank=256),)
            )
        ],
    )

    assert convert_block_configs_to_per_layer_config(config)
    assert config.per_layer_config == {
        "0": {"num_attention_heads": 8, "q_lora_rank": 384, "kv_lora_rank": 256}
    }


def test_gpt_oss_window_axis_respects_each_parent_layers_attention_scope() -> None:
    teachers = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(
                    num_query_heads=4,
                    num_kv_heads=2,
                    qk_head_dim=16,
                    sliding_window_size=128,
                ),
            )
        ),
        BlockConfig(
            subblock_configs=(
                AttentionConfig(
                    num_query_heads=4,
                    num_kv_heads=2,
                    qk_head_dim=16,
                    sliding_window_size="full",
                ),
            )
        ),
    ]

    candidates = build_candidate_library(
        teachers,
        search_space={
            "axes": {
                "sliding_window_size": {
                    "enabled": True,
                    "values": [64, 128, "full"],
                }
            }
        },
        parent_checkpoint_identity="teacher",
        include_self=True,
        include_noops=False,
    )

    by_layer = {
        layer: {
            candidate.block_config.require_subblock("attention").sliding_window_size
            for candidate in candidates
            if candidate.layer_idx == layer
        }
        for layer in range(2)
    }
    assert by_layer == {0: {64, 128}, 1: {64, 128, "full"}}


def test_vllm_adapter_exports_gpt_oss_window_type_and_size_per_layer() -> None:
    config = SimpleNamespace(
        base_architecture="GptOssForCausalLM",
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
        block_configs=[
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(
                        num_query_heads=4,
                        num_kv_heads=2,
                        sliding_window_size="full",
                    ),
                )
            ),
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(
                        num_query_heads=4,
                        num_kv_heads=2,
                        sliding_window_size=64,
                    ),
                )
            ),
        ],
    )

    assert convert_block_configs_to_per_layer_config(config)
    assert config.per_layer_config == {
        "0": {
            "layer_types": ["full_attention", "full_attention"],
            "sliding_window": None,
        },
        "1": {
            "layer_types": ["sliding_attention", "sliding_attention"],
            "sliding_window": 64,
        },
    }


def test_vllm_adapter_exports_full_window_without_layer_types() -> None:
    config = SimpleNamespace(
        base_architecture="LlamaForCausalLM",
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        sliding_window=512,
        block_configs=[
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(
                        num_query_heads=4,
                        num_kv_heads=2,
                        sliding_window_size="full",
                    ),
                )
            )
        ],
    )

    assert convert_block_configs_to_per_layer_config(config)
    assert config.per_layer_config == {"0": {"sliding_window": None}}


def test_vllm_adapter_exports_moe_noop_as_mlp_skip() -> None:
    config = SimpleNamespace(
        base_architecture="GptOssForCausalLM",
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        block_configs=[
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=4, num_kv_heads=2),
                    MoEConfig(
                        no_op=True,
                        num_experts=4,
                        expert_intermediate_size=32,
                        top_k=2,
                    ),
                )
            )
        ],
    )

    assert convert_block_configs_to_per_layer_config(config)
    assert config.per_layer_config == {"0": {"skip": ["mlp"]}}


def test_gpt_oss_descriptor_exposes_moe_runtime_benchmark_contract() -> None:
    from modelopt.torch.puzzletron.anymodel.models.gpt_oss.gpt_oss_model_descriptor import (
        GptOssModelDescriptor,
    )

    runtime = SimpleNamespace(
        num_attention_heads=64,
        num_key_value_heads=8,
        model_config_value=lambda key, default=None: {
            "head_dim": 64,
            "sliding_window": 128,
            "num_local_experts": 32,
            "intermediate_size": 2880,
            "num_experts_per_tok": 4,
        }.get(key, default),
    )

    assert GptOssModelDescriptor.runtime_benchmark_supported()
    base = GptOssModelDescriptor.runtime_benchmark_base_block_config(runtime)
    assert base.require_subblock("attention").sliding_window_size == 128
    assert base.require_subblock("moe").num_experts == 32
    assert base.require_subblock("moe").expert_intermediate_size == 2880
    assert base.require_subblock("moe").top_k == 4
    proxy = GptOssModelDescriptor._runtime_proxy_block_config(runtime, base)
    assert proxy.require_subblock("moe").num_experts == 16
    assert proxy.require_subblock("moe").expert_intermediate_size == 256
    assert GptOssModelDescriptor.anymodel_arch_info() == {
        "decoder_layer_module": ".gpt_oss",
        "decoder_layer_class": "TransformerBlock",
    }


def test_qwen35_moe_descriptor_exposes_bounded_runtime_benchmark_contract() -> None:
    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
        Qwen3P5MoeTextModelDescriptor,
        Qwen3P5MoeVLModelDescriptor,
    )

    values = {
        "head_dim": 256,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "runtime_proxy_max_experts": 16,
        "runtime_proxy_max_intermediate": 256,
        "runtime_proxy_max_shared_intermediate": 256,
    }
    runtime = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=2,
        model_config_value=lambda key, default=None: values.get(key, default),
    )

    assert Qwen3P5MoeTextModelDescriptor.runtime_benchmark_supported()
    assert (
        Qwen3P5MoeVLModelDescriptor.runtime_benchmark_export_descriptor()
        is Qwen3P5MoeTextModelDescriptor
    )
    flat_text_config = SimpleNamespace(hidden_size=2048)
    nested_vlm_config = SimpleNamespace(text_config=flat_text_config)
    assert (
        Qwen3P5MoeVLModelDescriptor.get_language_model_config(flat_text_config)
        is flat_text_config
    )
    assert (
        Qwen3P5MoeVLModelDescriptor.get_language_model_config(nested_vlm_config)
        is flat_text_config
    )
    base = Qwen3P5MoeTextModelDescriptor.runtime_benchmark_base_block_config(runtime)
    assert base.require_subblock("attention").num_query_heads == 16
    assert base.require_subblock("attention").num_kv_heads == 2
    assert base.require_subblock("moe").num_experts == 256
    assert base.require_subblock("moe").expert_intermediate_size == 512
    assert base.require_subblock("moe").shared_expert_intermediate_size == 512
    assert base.require_subblock("moe").top_k == 8

    proxy = Qwen3P5MoeTextModelDescriptor._runtime_proxy_block_config(runtime, base)
    assert proxy.require_subblock("moe").num_experts == 16
    assert proxy.require_subblock("moe").expert_intermediate_size == 256
    assert proxy.require_subblock("moe").shared_expert_intermediate_size == 256
    assert proxy.require_subblock("moe").top_k == 8


def test_vllm_adapter_exports_qwen35_moe_and_gdn_axes_per_layer() -> None:
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=256,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
    )
    config = SimpleNamespace(
        base_architecture="Qwen3_5MoeForCausalLM",
        text_config=text_config,
        block_configs=[
            BlockConfig(
                subblock_configs=(
                    MambaConfig(
                        num_groups=8,
                        num_heads=16,
                        state_dim=64,
                        head_dim=64,
                        conv_kernel_size=2,
                    ),
                    MoEConfig(
                        num_experts=128,
                        expert_intermediate_size=256,
                        shared_expert_intermediate_size=256,
                        top_k=4,
                    ),
                )
            )
        ],
    )

    assert convert_block_configs_to_per_layer_config(config)
    assert text_config.per_layer_config == {
        "0": {
            "num_experts": 128,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "num_experts_per_tok": 4,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 64,
            "linear_key_head_dim": 64,
            "linear_num_key_heads": 8,
            "linear_conv_kernel_dim": 2,
        }
    }
