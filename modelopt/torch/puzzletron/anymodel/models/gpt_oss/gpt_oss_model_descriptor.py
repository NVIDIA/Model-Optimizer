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
# mypy: ignore-errors

"""GPT-OSS model descriptor for AnyModel compression."""

import re
import types
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch.nn as nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssRotaryEmbedding,
)

from ....block_config import AttentionConfig, BlockConfig, MoEConfig
from ....pruning.embedding_pruning import TensorAxisRule
from ....pruning.expert_removal_pruning_mixin import (
    ExpertRemovalLayerDescriptor,
    ExpertRemovalPruningMixIn,
)
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor, KVHeadsPruningMixIn

# Expert removal is supported for unquantized models (test models).
# Production models use MXFP4 quantized MoE with combined tensors
# (gate_up_proj_blocks, down_proj_blocks), which is not yet supported.
from ....pruning.pruning_mixin import PruningMixIn
from ....utils.dummy_modules import DummyBlock
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer import deci_x_patcher
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size
from ..generic_decoder import (
    DecoderLayout,
    GenericDecoderContract,
    RoutedMoEContract,
    StandardGQAAttentionContract,
)

__all__ = [
    "GptOssExpertRemovalLayerDescriptor",
    "GptOssKVHeadsLayerDescriptor",
    "GptOssModelDescriptor",
]


@ModelDescriptorFactory.register_decorator("gpt_oss")
class GptOssModelDescriptor(ModelDescriptor):
    """Model descriptor for GPT-OSS (pure MoE model)."""

    _DECODER_LAYER_CLS: Type[nn.Module] = None

    @classmethod
    def create_dummy_block(cls, original_layer: GptOssDecoderLayer, block_index: int) -> nn.Module:
        dummy_block = DummyBlock(block_index=block_index)
        # Required by `GptOssModel.forward` in transformers<5.4
        if hasattr(original_layer, "attention_type"):
            dummy_block.attention_type = original_layer.attention_type
        return dummy_block

    @staticmethod
    def decoder_layer_cls():
        """Get the decoder layer class for GPT-OSS models.

        GPT-OSS is a standard transformers model in recent versions.
        Import directly from transformers.models.gpt_oss.modeling_gpt_oss.
        """
        return GptOssDecoderLayer

    @classmethod
    def generic_decoder_contract(cls, config):
        return GenericDecoderContract(
            descriptor_name="gpt_oss",
            model_family="gpt_oss",
            layout=DecoderLayout(
                language_config_path=(),
                language_prefix="model",
                layer_template="model.layers.{layer_idx}",
                input_embedding="model.embed_tokens",
                output_embedding="lm_head",
                final_norm="model.norm",
                layer_norm_names=("input_layernorm", "post_attention_layernorm"),
            ),
            attention=StandardGQAAttentionContract(),
            routed_moe=RoutedMoEContract(
                module_name="mlp",
                experts_name="experts",
                router_name="router",
                shared_expert_name=None,
                num_experts_field="num_local_experts",
                intermediate_field="intermediate_size",
            ),
            additional_tensor_rules=(
                TensorAxisRule(
                    r"^model\.layers\.\d+\.mlp\.experts\.gate_up_proj$",
                    (1,),
                    "GPT-OSS dequantized fused expert residual input",
                ),
                TensorAxisRule(
                    r"^model\.layers\.\d+\.mlp\.experts\.down_proj$",
                    (2,),
                    "GPT-OSS dequantized fused expert residual output",
                ),
                TensorAxisRule(
                    r"^model\.layers\.\d+\.mlp\.experts\.gate_up_proj_(?:blocks|scales)$",
                    (),
                    "GPT-OSS MXFP4 expert residual input blocks",
                    grouped_axes=((2, 32),),
                ),
                TensorAxisRule(
                    r"^model\.layers\.\d+\.mlp\.experts\.down_proj_(?:blocks|scales)$",
                    (1,),
                    "GPT-OSS fused expert residual output",
                ),
                TensorAxisRule(
                    r"^model\.layers\.\d+\.mlp\.experts\.down_proj_bias$",
                    (1,),
                    "GPT-OSS expert residual output bias",
                ),
            ),
            native_automodel_supported=True,
            ep_supported=True,
            explicit_full_attention_window=True,
            # MXFP4 input-channel scales cover 32 channels.  Keep those blocks
            # intact while globally ranking and permuting residual channels.
            hidden_permutation_group_size=32,
        )

    @classmethod
    def embedding_pruning_spec(cls, config, *, widths, alignment: int):
        return cls.generic_decoder_contract(config).embedding_pruning_spec(
            config, widths=widths, alignment=alignment
        )

    @staticmethod
    def sorted_teacher_layout_kwargs(_lm_config) -> dict[str, object]:
        """Declare query-head state and fused expert storage owned by GPT-OSS."""
        return {
            "attention_q_head_subnames": ("sinks",),
            "moe_router_subname": "router",
            "moe_router_aux_subnames": ("router.bias",),
            "moe_fused_expert_subnames": (
                "experts.gate_up_proj_blocks",
                "experts.gate_up_proj_scales",
                "experts.gate_up_proj_bias",
                "experts.down_proj_blocks",
                "experts.down_proj_scales",
                "experts.down_proj_bias",
            ),
            "moe_fused_gate_up_subnames": (
                "experts.gate_up_proj_blocks",
                "experts.gate_up_proj_scales",
                "experts.gate_up_proj_bias",
            ),
            "moe_fused_down_subnames": (
                "experts.down_proj_blocks",
                "experts.down_proj_scales",
            ),
            "moe_expert_intermediate_group_size": 32,
            "moe_expert_order_mode": "metadata_only",
            "moe_fused_gate_layout": "interleaved",
        }

    @classmethod
    def puzzletron_capabilities(cls, config):
        return cls.generic_decoder_contract(config).capabilities(config)

    @classmethod
    def anymodel_arch_info(cls) -> dict[str, object]:
        """Map HF GPT-OSS layers to the vLLM implementation used by AnyModel."""
        return {
            "decoder_layer_module": ".gpt_oss",
            "decoder_layer_class": "TransformerBlock",
        }

    @classmethod
    def automodel_model_kwargs(cls, config, *, distributed=None):
        """Use GPT-OSS's supported non-TE attention backend.

        Passing a partial ``backend`` mapping for native TP disables the
        AutoModel class's ``backend=None`` default, so the descriptor must keep
        its required Flex attention choice explicit.
        """
        del config, distributed
        return {"backend": {"attn": "flex"}}

    @classmethod
    def checkpoint_equivalence_tolerances(cls) -> dict[str, float]:
        """Allow measured batch-sensitive MXFP4 drift from a basis permutation.

        The LM-loss delta reaches 0.01944 on the eight-sample diagnostic batch
        even when KL, top-1 agreement, cosine, and normalized hidden MSE all
        satisfy their independent equivalence gates.
        """
        return {
            "max_abs_lm_loss_delta": 2.5e-2,
            "max_kl_div": 5.0e-2,
            "min_top_1_logit_agreement": 0.9,
        }

    @classmethod
    def patch_pipeline_model_part(cls, model_part: nn.Module) -> bool:
        """Restore GPT-OSS's stage-aware native inner forward after PP split.

        AutoModel's generic CausalLM PP forward precomputes HF-style rotary
        embeddings. GPT-OSS instead computes ``freqs_cis`` inside its native
        inner forward, which already tolerates stage-local missing embeddings
        and norms. Keep the generic outer LM wrapper but restore the inner
        class method on each split chunk.
        """
        inner = getattr(model_part, "model", None)
        native_forward = getattr(type(inner), "forward", None) if inner is not None else None
        if inner is None or not callable(native_forward):
            return False
        inner.forward = types.MethodType(native_forward, inner)
        return True

    @classmethod
    def block_config_to_layer_overrides(cls, block_config: BlockConfig):
        """Map BlockConfig through the shared structural contract."""
        contract = cls.generic_decoder_contract(None)
        override_kwargs = {}
        attention = block_config.require_subblock("attention")
        moe = block_config.get_subblock("moe")

        override_kwargs.update(contract.attention.layer_override_fields(attention))

        if moe is not None:
            moe_contract = contract.routed_moe
            if moe.expert_intermediate_size is not None:
                override_kwargs[moe_contract.intermediate_field] = (
                    moe.expert_intermediate_size
                )
            if moe.num_experts is not None:
                override_kwargs[moe_contract.num_experts_field] = moe.num_experts
            if moe.top_k is not None:
                override_kwargs[moe_contract.top_k_field] = moe.top_k

        return override_kwargs

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, object]:
        """Fields needed to build bounded, structurally faithful GPT-OSS proxies."""
        return {
            "head_dim": getattr(lm_config, "head_dim", 64),
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "intermediate_size": getattr(lm_config, "intermediate_size", 2880),
            "num_local_experts": getattr(lm_config, "num_local_experts", 32),
            "num_experts_per_tok": getattr(lm_config, "num_experts_per_tok", 4),
            "sliding_window": getattr(lm_config, "sliding_window", 128),
            "layer_types": list(getattr(lm_config, "layer_types", ()) or ()),
            "rope_parameters": getattr(lm_config, "rope_parameters", None),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-5),
            "attention_bias": getattr(lm_config, "attention_bias", True),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
            # Runtime-only bounds preserve candidate ratios while avoiding a
            # multi-gigabyte synthetic MoE and vocabulary on every worker.
            "runtime_proxy_max_experts": 16,
            "runtime_proxy_max_intermediate": 256,
            "runtime_proxy_max_vocab": 32768,
        }

    @staticmethod
    def _scale_runtime_proxy_value(value: int, base_value: int, cap: int) -> int:
        if base_value <= cap:
            return int(value)
        return max(1, min(int(cap), int(round(int(value) * int(cap) / int(base_value)))))

    @classmethod
    def _runtime_proxy_moe(cls, runtime_config, moe: MoEConfig) -> MoEConfig:
        if moe.no_op:
            return moe
        base_experts = int(runtime_config.model_config_value("num_local_experts", 32))
        base_intermediate = int(runtime_config.model_config_value("intermediate_size", 2880))
        num_experts = cls._scale_runtime_proxy_value(
            int(moe.num_experts or base_experts),
            base_experts,
            int(runtime_config.model_config_value("runtime_proxy_max_experts", 16)),
        )
        intermediate = cls._scale_runtime_proxy_value(
            int(moe.expert_intermediate_size or base_intermediate),
            base_intermediate,
            int(runtime_config.model_config_value("runtime_proxy_max_intermediate", 256)),
        )
        top_k = min(
            int(moe.top_k or runtime_config.model_config_value("num_experts_per_tok", 4)),
            num_experts,
        )
        return MoEConfig(
            no_op=False,
            num_experts=num_experts,
            expert_intermediate_size=intermediate,
            top_k=top_k,
        )

    @classmethod
    def _runtime_proxy_block_config(cls, runtime_config, block_config: BlockConfig) -> BlockConfig:
        return BlockConfig(
            subblock_configs=tuple(
                cls._runtime_proxy_moe(runtime_config, subblock)
                if isinstance(subblock, MoEConfig)
                else subblock
                for subblock in block_config.subblock_configs
            )
        )

    @classmethod
    def runtime_benchmark_base_block_config(cls, runtime_config) -> BlockConfig:
        base_experts = int(runtime_config.model_config_value("num_local_experts", 32))
        base_intermediate = int(runtime_config.model_config_value("intermediate_size", 2880))
        # Keep benchmark scaffolding canonical. Proxy scaling is applied once,
        # together with candidate blocks, inside create_runtime_benchmark_model.
        return BlockConfig(
            subblock_configs=(
                AttentionConfig(
                    num_query_heads=runtime_config.num_attention_heads,
                    num_kv_heads=runtime_config.num_key_value_heads,
                    qk_head_dim=runtime_config.model_config_value("head_dim", 64),
                    sliding_window_size=runtime_config.model_config_value(
                        "sliding_window", 128
                    ),
                ),
                MoEConfig(
                    num_experts=base_experts,
                    expert_intermediate_size=base_intermediate,
                    top_k=int(runtime_config.model_config_value("num_experts_per_tok", 4)),
                ),
            )
        )

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        block_configs = [
            cls._runtime_proxy_block_config(runtime_config, block_config)
            for block_config in block_configs
        ]
        base_experts = int(runtime_config.model_config_value("num_local_experts", 32))
        base_intermediate = int(runtime_config.model_config_value("intermediate_size", 2880))
        proxy_experts = cls._scale_runtime_proxy_value(
            base_experts,
            base_experts,
            int(runtime_config.model_config_value("runtime_proxy_max_experts", 16)),
        )
        proxy_intermediate = cls._scale_runtime_proxy_value(
            base_intermediate,
            base_intermediate,
            int(runtime_config.model_config_value("runtime_proxy_max_intermediate", 256)),
        )
        model_config = GptOssConfig(
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            vocab_size=min(
                runtime_config.vocab_size,
                int(runtime_config.model_config_value("runtime_proxy_max_vocab", 32768)),
            ),
            hidden_size=runtime_config.hidden_size,
            intermediate_size=proxy_intermediate,
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim", 64),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            num_local_experts=proxy_experts,
            num_experts_per_tok=min(
                int(runtime_config.model_config_value("num_experts_per_tok", 4)),
                proxy_experts,
            ),
            sliding_window=runtime_config.model_config_value("sliding_window", 128),
            rope_parameters=runtime_config.model_config_value("rope_parameters"),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-5),
            attention_bias=runtime_config.model_config_value("attention_bias", True),
            tie_word_embeddings=runtime_config.model_config_value(
                "tie_word_embeddings", False
            ),
        )
        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = GptOssForCausalLM(model_config)
        model.config.block_configs = [block.to_dict() for block in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "GptOssForCausalLM"
        return model

    @classmethod
    def patch_layer_config(cls, layer_config, block_config: BlockConfig, layer_idx: int) -> None:
        attention = block_config.get_subblock("attention")
        if attention is None or attention.sliding_window_size is None:
            return
        window = attention.sliding_window_size
        desired_type = "full_attention" if window == "full" else "sliding_attention"
        layer_types = list(getattr(layer_config, "layer_types", ()) or ())
        if len(layer_types) <= layer_idx:
            layer_types.extend(
                ["full_attention"] * (layer_idx + 1 - len(layer_types))
            )
        layer_types[layer_idx] = desired_type
        layer_config.layer_types = layer_types
        if window != "full":
            layer_config.sliding_window = int(window)

    @staticmethod
    def attn_no_op_post_init(decoder_layer):
        """Replace attention sublayers with no-op modules."""
        decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer):
        """Replace MLP sublayers with no-op modules.

        Note: GPT-OSS MoE layers return (hidden_states, router_scores), so we need
        to return a tuple of 2 values.
        """
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """Initialize rotary embeddings on the correct device."""
        # GPT-OSS uses RoPE with YARN scaling

        model.model.rotary_emb = GptOssRotaryEmbedding(
            config=model.config,
            device=runtime.device,
        )

    @staticmethod
    def input_embedding_name():
        return "model.embed_tokens"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "model.norm"

    @staticmethod
    def layer_block_name(index: int):
        return f"model.layers.{index}"

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        """Define regex patterns for grouping weights into subblocks."""
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            """FFN is MoE in GPT-OSS with MXFP4 quantization."""
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^model\.layers\.{layer_idx}\."
                    r"(post_attention_layernorm\.weight"
                    r"|mlp\.(router|gate)\.(weight|bias)"
                    r"|mlp\.experts\.((gate_up_proj|down_proj)"
                    r"(_(bias|blocks|scales))?|gate_and_up_projs|down_projs))$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^model\.layers\.{layer_idx}\."
                    r"(input_layernorm\.weight"
                    r"|self_attn\.q_proj\.weight"
                    r"|self_attn\.q_proj\.bias"
                    r"|self_attn\.k_proj\.weight"
                    r"|self_attn\.k_proj\.bias"
                    r"|self_attn\.v_proj\.weight"
                    r"|self_attn\.v_proj\.bias"
                    r"|self_attn\.o_proj\.weight"
                    r"|self_attn\.o_proj\.bias"
                    r"|self_attn\.sinks)$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(
            **build_ffn_predicates(),
            **build_attention_predicates(),
        )

        return layer_name_patterns

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        """Return available pruning mixins for GPT-OSS.

        Note: Expert removal works for unquantized models (test models).
        Production models use MXFP4 quantization which is not yet supported.
        """
        # Single instance shared between the canonical key and the legacy alias
        # so resolve_pruning_mixin returns the same object regardless of which
        # name a caller uses.
        expert_mixin = ExpertRemovalPruningMixIn(GptOssExpertRemovalLayerDescriptor())
        return {
            "experts_removal": expert_mixin,
            # Backward-compat alias: this key was "expert_removal" before the
            # pruning configuration standardised on "experts_removal" (matching the
            # NemotronH descriptor). Kept so external scripts that still call
            # `resolve_pruning_mixin("expert_removal", GptOssModelDescriptor)`
            # continue to work. Remove after a deprecation cycle.
            "expert_removal": expert_mixin,
            "kv_heads": KVHeadsPruningMixIn(GptOssKVHeadsLayerDescriptor()),
        }


@dataclass
class GptOssKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class GptOssExpertRemovalLayerDescriptor(ExpertRemovalLayerDescriptor):
    """
    GPT-OSS MoE layer descriptor for expert removal.

    Note: This only works for unquantized models (e.g., test models).
    Production GPT-OSS models use MXFP4 quantization with fused experts
    (_blocks, _scales, _bias), which requires a different approach.

    Structure:
    - Router: mlp.router with .weight and .bias
    - Experts: mlp.experts.{idx}.{gate_up_proj,down_proj} with .weight and .bias
    """

    target_name: str = "mlp"
    moe_prefix_name: str = "model.layers.{layer_idx}.mlp"
    expert_prefix_name: str = "experts"

    # Router has both weight and bias
    router_weights: List[str] = field(default_factory=lambda: ["router.weight"])
    router_biases: List[str] = field(default_factory=lambda: ["router.bias"])

    # Fused format: experts stored as single tensors
    is_fused_experts: bool = True

    # Fused format: single tensors containing all experts (test models)
    fused_expert_weights: List[str] = field(
        default_factory=lambda: [
            "experts.gate_up_proj",
            "experts.gate_up_proj_bias",
            "experts.down_proj",
            "experts.down_proj_bias",
        ]
    )

    # Not used for fused format, but kept for compatibility
    expert_weights: List[str] = field(default_factory=lambda: ["gate_up_proj", "down_proj"])
    expert_biases: List[str] = field(
        default_factory=lambda: ["gate_up_proj_bias", "down_proj_bias"]
    )

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        target_class_name = "GptOssTopKRouter"

        module_names_to_hook = []
        for module_name, module in model.named_modules():
            if (
                module_name.endswith(self.target_name)
                and module.__class__.__name__ == target_class_name
            ):
                module_names_to_hook.append(
                    (self.block_idx_from_module_name(module_name), module_name)
                )
        return module_names_to_hook
