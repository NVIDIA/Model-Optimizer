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

import importlib
import inspect
import json
import pkgutil
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Type

import torch
import torch.nn as nn

from ....block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
    maybe_cast_block_configs,
)
from ....pruning.expert_removal_pruning_mixin import (
    ExpertRemovalLayerDescriptor,
    ExpertRemovalPruningMixIn,
)
from ....pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor, KVHeadsPruningMixIn
from ....pruning.moe_mamba_pruning_mixin import (
    MambaLayerDescriptor,
    MambaPruningMixIn,
    MoELayerDescriptor,
    MoEPruningMixIn,
)
from ....pruning.pruning_mixin import PruningMixIn
from ....pruning.embedding_pruning import EmbeddingPruningSpec, TensorAxisRule
from ...capabilities import (
    AxisCapabilities,
    ExportCapabilities,
    ParallelCapabilities,
    default_capabilities,
)
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer import deci_x_patcher
from ...puzzformer.no_op import MatchingZeros, Same

__all__ = [
    "NemotronHExpertRemovalLayerDescriptor",
    "NemotronHFFNIntermediateLayerDescriptor",
    "NemotronHKVHeadsLayerDescriptor",
    "NemotronHMoELayerDescriptor",
    "NemotronHMambaLayerDescriptor",
    "NemotronHModelDescriptor",
]


def get_dynamic_modules(module_cls_str: str) -> List[Type[nn.Module]]:
    try:
        import transformers_modules
    except ModuleNotFoundError:
        return []

    matches = []
    for finder, modname, ispkg in pkgutil.walk_packages(
        transformers_modules.__path__, transformers_modules.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__ == module_cls_str:
                matches.append(obj)

    return matches


@dataclass
class NemotronHFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mixer.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mixer"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "up_proj"]
    )


@dataclass
class NemotronHKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "mixer.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.mixer"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class NemotronHExpertRemovalLayerDescriptor(ExpertRemovalLayerDescriptor):
    target_name: str = "mixer.gate"
    moe_prefix_name: str = "model.layers.{layer_idx}.mixer"
    expert_prefix_name: str = "experts.{expert_idx}"
    router_weights: List[str] = field(default_factory=lambda: ["gate.weight"])
    router_biases: List[str] = field(default_factory=lambda: ["gate.e_score_correction_bias"])
    expert_weights: List[str] = field(
        default_factory=lambda: ["up_proj.weight", "down_proj.weight"]
    )

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        if self.target_name != "mixer":
            return super().get_modules_names_to_hook(model)

        # when target is `mixer` we'll target moe layers of class type: `NemotronHMOE`, as NemotronH models use auto-map we'll check for class name instead of class type.
        target_class_name = "NemotronHMOE"

        module_names_to_hook = []
        for module_name, module in model.named_modules():
            # restrict to attributes called "mixer" and with the desired class name
            if (
                module_name.endswith(self.target_name)
                and module.__class__.__name__ == target_class_name
            ):
                module_names_to_hook.append(
                    (self.block_idx_from_module_name(module_name), module_name)
                )
        return module_names_to_hook


@dataclass
class NemotronHMoELayerDescriptor(MoELayerDescriptor):
    moe_prefix_name: str = "model.layers.{layer_idx}.mixer"


@dataclass
class NemotronHMambaLayerDescriptor(MambaLayerDescriptor):
    mamba_prefix_name: str = "model.layers.{layer_idx}.mixer"


@ModelDescriptorFactory.register_decorator("nemotron_h")
class NemotronHModelDescriptor(ModelDescriptor):
    _DECODER_LAYER_CLS: Type[nn.Module] = None
    _FUSED_EXPERT_MODEL_KEY_RE = re.compile(
        r"^(model|backbone)\.layers\.(\d+)\.mixer\.experts\.(up_proj|down_proj)(?:\.weight)?$"
    )
    _SPLIT_EXPERT_CHECKPOINT_KEY_RE = re.compile(
        r"^(model|backbone)\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
    )

    @staticmethod
    def decoder_layer_cls():
        decoder_cls_list = get_dynamic_modules("NemotronHBlock")
        if not decoder_cls_list:
            try:
                from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock

                return [NemotronHBlock]
            except Exception as exc:
                raise AssertionError(
                    "NemotronH contains dynamic modules that should be cached beforehand, "
                    "make sure to load your config using `load_model_config` or manually "
                    "call `force_cache_dynamic_modules(config, checkpoint_dir)`"
                ) from exc
        return decoder_cls_list

    @classmethod
    def anymodel_arch_info(cls) -> dict:
        return {
            "decoder_layer_module": ".nemotron_h",
            "decoder_layer_class": "NemotronHAttentionDecoderLayer",
            "decoder_layer_class_map": {
                "*": "NemotronHAttentionDecoderLayer",
                "-": "NemotronHMLPDecoderLayer",
                "E": "NemotronHMoEDecoderLayer",
                "M": "NemotronHMambaDecoderLayer",
            },
            "hybrid_pattern_field": "hybrid_override_pattern",
            "attn_module": "mixer",
            "attn_norm_module": "norm",
            "ffn_module": "mixer",
            "ffn_norm_module": "norm",
        }

    @staticmethod
    def requires_trust_remote_code() -> bool:
        return True

    @staticmethod
    def _pattern_from_layer_types(layer_types: Iterable[str]) -> str:
        return "".join(
            {
                "mamba": "M",
                "attention": "*",
                "mlp": "-",
                "moe": "E",
            }[layer_type]
            for layer_type in layer_types
        )

    @staticmethod
    def _active_layer_type(block_config: BlockConfig) -> str:
        mamba = block_config.get_subblock("mamba")
        if mamba is not None and not mamba.no_op:
            return "mamba"
        attention = block_config.get_subblock("attention")
        if attention is not None and not attention.no_op:
            return "attention"
        moe = block_config.get_subblock("moe")
        if moe is not None and not moe.no_op:
            return "moe"
        ffn = block_config.get_subblock("ffn")
        if ffn is not None and not ffn.no_op:
            return "mlp"
        # Fully skipped blocks still need a concrete class for model init.
        return "mamba" if mamba is not None else "attention"

    @classmethod
    def _set_layer_types(cls, config, layer_types: list[str]) -> None:
        declared = inspect.getattr_static(type(config), "layers_block_type", None)
        if isinstance(declared, property) and declared.fset is None:
            # Older remote-code Nemotron configs derive layers_block_type from
            # hybrid_override_pattern and intentionally expose no setter.
            config.hybrid_override_pattern = cls._pattern_from_layer_types(layer_types)
        else:
            config.layers_block_type = layer_types

    @classmethod
    def set_block_configs(cls, model_config, block_configs: list[BlockConfig | dict]) -> None:
        block_configs = maybe_cast_block_configs(block_configs)
        super().set_block_configs(model_config, block_configs)
        lm_config = cls.get_language_model_config(model_config)
        layer_types = [cls._active_layer_type(block_config) for block_config in block_configs]
        cls._set_layer_types(lm_config, layer_types)
        if lm_config is not model_config:
            cls._set_layer_types(model_config, layer_types)

    @staticmethod
    def truncate_pattern_for_subblock(lm_config, parent_layer_index: int | None = None) -> None:
        layer_types = getattr(lm_config, "layers_block_type", None)
        if layer_types:
            if parent_layer_index is not None and 0 <= parent_layer_index < len(layer_types):
                NemotronHModelDescriptor._set_layer_types(
                    lm_config, [layer_types[parent_layer_index]]
                )
            else:
                NemotronHModelDescriptor._set_layer_types(lm_config, [layer_types[0]])
            return
        ModelDescriptor.truncate_pattern_for_subblock(lm_config, parent_layer_index)

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {}
        ffn = block_config.get_subblock("ffn")
        attention = block_config.get_subblock("attention")
        moe = block_config.get_subblock("moe")
        mamba = block_config.get_subblock("mamba")

        if ffn is not None and ffn.intermediate_size is not None:
            override_kwargs["intermediate_size"] = ffn.intermediate_size

        if attention is not None and attention.num_kv_heads is not None:
            override_kwargs["num_key_value_heads"] = attention.num_kv_heads
        if attention is not None and attention.num_query_heads is not None:
            override_kwargs["num_attention_heads"] = attention.num_query_heads

        if moe is not None:
            if moe.expert_intermediate_size is not None:
                override_kwargs["moe_intermediate_size"] = moe.expert_intermediate_size
            if moe.num_experts is not None:
                override_kwargs["n_routed_experts"] = moe.num_experts
            if moe.top_k is not None:
                override_kwargs["num_experts_per_tok"] = moe.top_k
            if moe.shared_expert_intermediate_size is not None:
                override_kwargs["moe_shared_expert_intermediate_size"] = (
                    moe.shared_expert_intermediate_size
                )
            if moe.latent_dim is not None:
                override_kwargs["moe_latent_size"] = moe.latent_dim

        if mamba is not None:
            if mamba.num_heads is not None:
                override_kwargs["mamba_num_heads"] = mamba.num_heads
            if mamba.head_dim is not None:
                override_kwargs["mamba_head_dim"] = mamba.head_dim
            if mamba.state_dim is not None:
                override_kwargs["ssm_state_size"] = mamba.state_dim

        return override_kwargs

    @staticmethod
    def _runtime_config_cls():
        config_classes = get_dynamic_modules("NemotronHConfig")
        if config_classes:
            return config_classes[0]
        from transformers import NemotronHConfig

        return NemotronHConfig

    @staticmethod
    def _runtime_model_cls():
        model_classes = get_dynamic_modules("NemotronHForCausalLM")
        if model_classes:
            return model_classes[0]
        from transformers import NemotronHForCausalLM

        return NemotronHForCausalLM

    @classmethod
    def _runtime_layer_config_kwargs(cls, config_cls, layer_types: list[str]) -> dict[str, Any]:
        parameters = inspect.signature(config_cls.__init__).parameters
        if "layers_block_type" in parameters:
            return {"layers_block_type": layer_types}
        if "hybrid_override_pattern" in parameters:
            return {"hybrid_override_pattern": cls._pattern_from_layer_types(layer_types)}
        return {"layers_block_type": layer_types}

    @staticmethod
    def _normalize_runtime_tied_weight_metadata(model: nn.Module) -> None:
        for module in model.modules():
            tied = getattr(module, "_tied_weights_keys", None)
            if isinstance(tied, list | tuple | set):
                module._tied_weights_keys = {str(key): str(key) for key in tied}

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        return {
            "head_dim": getattr(lm_config, "head_dim", 128),
            "intermediate_size": getattr(lm_config, "intermediate_size", 2688),
            "mlp_hidden_act": getattr(lm_config, "mlp_hidden_act", "relu2"),
            "attention_bias": getattr(lm_config, "attention_bias", False),
            "mlp_bias": getattr(lm_config, "mlp_bias", False),
            "use_bias": getattr(lm_config, "use_bias", False),
            "initializer_range": getattr(lm_config, "initializer_range", 0.02),
            "layer_norm_epsilon": getattr(lm_config, "layer_norm_epsilon", 1e-5),
            "residual_in_fp32": getattr(lm_config, "residual_in_fp32", False),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
            "sliding_window": getattr(lm_config, "sliding_window", None),
            "attention_dropout": getattr(lm_config, "attention_dropout", 0.0),
            "hidden_dropout": getattr(lm_config, "hidden_dropout", 0.0),
            "use_mamba_kernels": getattr(lm_config, "use_mamba_kernels", True),
            "ssm_state_size": getattr(lm_config, "ssm_state_size", 128),
            "mamba_num_heads": getattr(lm_config, "mamba_num_heads", 128),
            "mamba_head_dim": getattr(lm_config, "mamba_head_dim", 64),
            "n_groups": getattr(lm_config, "n_groups", 8),
            "conv_kernel": getattr(lm_config, "conv_kernel", 4),
            "expand": getattr(lm_config, "expand", 2),
            "mamba_hidden_act": getattr(lm_config, "mamba_hidden_act", "silu"),
            "time_step_min": getattr(lm_config, "time_step_min", 0.001),
            "time_step_max": getattr(lm_config, "time_step_max", 0.1),
            "time_step_limit": getattr(lm_config, "time_step_limit", (0.0, float("inf"))),
            "time_step_floor": getattr(lm_config, "time_step_floor", 1e-4),
            "use_conv_bias": getattr(lm_config, "use_conv_bias", True),
            "mamba_proj_bias": getattr(lm_config, "mamba_proj_bias", False),
            "chunk_size": getattr(lm_config, "chunk_size", 128),
            "mamba_ssm_cache_dtype": getattr(lm_config, "mamba_ssm_cache_dtype", "float32"),
            "rescale_prenorm_residual": getattr(lm_config, "rescale_prenorm_residual", True),
            "n_routed_experts": getattr(lm_config, "n_routed_experts", 512),
            "n_shared_experts": getattr(lm_config, "n_shared_experts", 1),
            "moe_intermediate_size": getattr(lm_config, "moe_intermediate_size", 2688),
            "moe_shared_expert_intermediate_size": getattr(
                lm_config, "moe_shared_expert_intermediate_size", 5376
            ),
            "moe_latent_size": getattr(lm_config, "moe_latent_size", None),
            "moe_shared_expert_overlap": getattr(lm_config, "moe_shared_expert_overlap", True),
            "num_experts_per_tok": getattr(lm_config, "num_experts_per_tok", 22),
            "routed_scaling_factor": getattr(lm_config, "routed_scaling_factor", 5.0),
            "n_group": getattr(lm_config, "n_group", 1),
            "topk_group": getattr(lm_config, "topk_group", 1),
            "norm_topk_prob": getattr(lm_config, "norm_topk_prob", True),
            "pad_token_id": getattr(lm_config, "pad_token_id", 0),
            "bos_token_id": getattr(lm_config, "bos_token_id", 1),
            "eos_token_id": getattr(lm_config, "eos_token_id", 2),
            # Runtime-only proxy caps. Production measurements use physical
            # dimensions; smoke configs may explicitly opt into bounded proxies.
            "runtime_proxy_enabled": getattr(lm_config, "runtime_proxy_enabled", False),
            "runtime_proxy_max_experts": 16,
            "runtime_proxy_max_expert_intermediate": 512,
            "runtime_proxy_max_shared_expert_intermediate": 512,
            "runtime_proxy_max_latent": 256,
            "runtime_proxy_max_top_k": 4,
        }

    @classmethod
    def _scale_runtime_proxy_value(cls, value: int | None, base_value: int | None, cap: int) -> int | None:
        if value is None:
            return None
        if base_value is None or base_value <= cap:
            return value
        return max(1, min(cap, int(round(int(value) * cap / int(base_value)))))

    @classmethod
    def _runtime_proxy_block_config(cls, runtime_config, block_config: BlockConfig) -> BlockConfig:
        if not runtime_config.model_config_value("runtime_proxy_enabled", False):
            return block_config

        moe = block_config.get_subblock("moe")
        if moe is not None and not moe.no_op:
            bounded_moe = replace(
                moe,
                num_experts=cls._scale_runtime_proxy_value(
                    moe.num_experts,
                    runtime_config.model_config_value("n_routed_experts"),
                    runtime_config.model_config_value("runtime_proxy_max_experts", 16),
                ),
                expert_intermediate_size=cls._scale_runtime_proxy_value(
                    moe.expert_intermediate_size,
                    runtime_config.model_config_value("moe_intermediate_size"),
                    runtime_config.model_config_value(
                        "runtime_proxy_max_expert_intermediate", 512
                    ),
                ),
                shared_expert_intermediate_size=cls._scale_runtime_proxy_value(
                    moe.shared_expert_intermediate_size,
                    runtime_config.model_config_value(
                        "moe_shared_expert_intermediate_size"
                    ),
                    runtime_config.model_config_value(
                        "runtime_proxy_max_shared_expert_intermediate", 512
                    ),
                ),
                latent_dim=cls._scale_runtime_proxy_value(
                    moe.latent_dim,
                    runtime_config.model_config_value("moe_latent_size"),
                    runtime_config.model_config_value("runtime_proxy_max_latent", 256),
                ),
                top_k=cls._scale_runtime_proxy_value(
                    moe.top_k,
                    runtime_config.model_config_value("num_experts_per_tok"),
                    runtime_config.model_config_value("runtime_proxy_max_top_k", 4),
                ),
            )
            if (
                bounded_moe.num_experts is not None
                and bounded_moe.top_k is not None
                and bounded_moe.top_k > bounded_moe.num_experts
            ):
                bounded_moe = replace(bounded_moe, top_k=bounded_moe.num_experts)
            return BlockConfig(
                subblock_configs=tuple(
                    bounded_moe if subblock is moe else subblock
                    for subblock in block_config.subblock_configs
                )
            )
        return block_config

    @classmethod
    def _runtime_proxy_block_configs(cls, runtime_config, block_configs: list[BlockConfig]) -> list[BlockConfig]:
        return [cls._runtime_proxy_block_config(runtime_config, block_config) for block_config in block_configs]

    @classmethod
    def runtime_benchmark_base_block_config(cls, runtime_config) -> BlockConfig:
        return BlockConfig(
            subblock_configs=(
                AttentionConfig(no_op=False, num_kv_heads=runtime_config.num_key_value_heads),
                FFNConfig(no_op=True),
            )
        )

    @classmethod
    def runtime_benchmark_sublayers_are_exclusive(cls) -> bool:
        """Nemotron-H hybrid layers select one of attention, Mamba, MLP, or MoE."""
        return True

    @classmethod
    def runtime_benchmark_scaffold_policy(cls, block_config: BlockConfig) -> str:
        """Keep one attention cache anchor per PP stage for cacheless hybrid candidates."""

        attention = block_config.get_subblock("attention")
        if attention is None or attention.no_op:
            return "attention_scaffold_per_pp_stage"
        return "none"

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        config_cls = cls._runtime_config_cls()
        model_cls = cls._runtime_model_cls()
        block_configs = cls._runtime_proxy_block_configs(runtime_config, block_configs)
        layer_types = [cls._active_layer_type(block_config) for block_config in block_configs]
        active_mambas = [
            mamba
            for block_config in block_configs
            if (mamba := block_config.get_subblock("mamba")) is not None and not mamba.no_op
        ]
        runtime_mamba = active_mambas[0] if active_mambas else None
        global_num_experts = runtime_config.model_config_value("n_routed_experts", 512)
        global_moe_intermediate = runtime_config.model_config_value(
            "moe_intermediate_size", 2688
        )
        global_shared_intermediate = runtime_config.model_config_value(
            "moe_shared_expert_intermediate_size", 5376
        )
        global_latent_size = runtime_config.model_config_value("moe_latent_size")
        global_top_k = runtime_config.model_config_value("num_experts_per_tok", 22)
        if runtime_config.model_config_value("runtime_proxy_enabled", False):
            global_num_experts = cls._scale_runtime_proxy_value(
                global_num_experts,
                global_num_experts,
                runtime_config.model_config_value("runtime_proxy_max_experts", 16),
            )
            global_moe_intermediate = cls._scale_runtime_proxy_value(
                global_moe_intermediate,
                global_moe_intermediate,
                runtime_config.model_config_value("runtime_proxy_max_expert_intermediate", 512),
            )
            global_shared_intermediate = cls._scale_runtime_proxy_value(
                global_shared_intermediate,
                global_shared_intermediate,
                runtime_config.model_config_value(
                    "runtime_proxy_max_shared_expert_intermediate", 512
                ),
            )
            global_latent_size = cls._scale_runtime_proxy_value(
                global_latent_size,
                global_latent_size,
                runtime_config.model_config_value("runtime_proxy_max_latent", 256),
            )
            global_top_k = cls._scale_runtime_proxy_value(
                global_top_k,
                global_top_k,
                runtime_config.model_config_value("runtime_proxy_max_top_k", 4),
            )
        if global_top_k is not None and global_num_experts is not None:
            global_top_k = min(global_top_k, global_num_experts)

        model_config = config_cls(
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            **cls._runtime_layer_config_kwargs(config_cls, layer_types),
            num_hidden_layers=len(block_configs),
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            head_dim=runtime_config.model_config_value("head_dim", 128),
            intermediate_size=runtime_config.model_config_value("intermediate_size", 2688),
            mlp_hidden_act=runtime_config.model_config_value("mlp_hidden_act", "relu2"),
            attention_bias=runtime_config.model_config_value("attention_bias", False),
            mlp_bias=runtime_config.model_config_value("mlp_bias", False),
            use_bias=runtime_config.model_config_value("use_bias", False),
            initializer_range=runtime_config.model_config_value("initializer_range", 0.02),
            layer_norm_epsilon=runtime_config.model_config_value("layer_norm_epsilon", 1e-5),
            residual_in_fp32=runtime_config.model_config_value("residual_in_fp32", False),
            sliding_window=runtime_config.model_config_value("sliding_window", None),
            attention_dropout=runtime_config.model_config_value("attention_dropout", 0.0),
            hidden_dropout=runtime_config.model_config_value("hidden_dropout", 0.0),
            use_mamba_kernels=runtime_config.model_config_value("use_mamba_kernels", True),
            ssm_state_size=(
                runtime_mamba.state_dim
                if runtime_mamba is not None and runtime_mamba.state_dim is not None
                else runtime_config.model_config_value("ssm_state_size", 128)
            ),
            mamba_num_heads=(
                runtime_mamba.num_heads
                if runtime_mamba is not None and runtime_mamba.num_heads is not None
                else runtime_config.model_config_value("mamba_num_heads", 128)
            ),
            mamba_head_dim=(
                runtime_mamba.head_dim
                if runtime_mamba is not None and runtime_mamba.head_dim is not None
                else runtime_config.model_config_value("mamba_head_dim", 64)
            ),
            n_groups=runtime_config.model_config_value("n_groups", 8),
            conv_kernel=runtime_config.model_config_value("conv_kernel", 4),
            expand=runtime_config.model_config_value("expand", 2),
            mamba_hidden_act=runtime_config.model_config_value("mamba_hidden_act", "silu"),
            time_step_min=runtime_config.model_config_value("time_step_min", 0.001),
            time_step_max=runtime_config.model_config_value("time_step_max", 0.1),
            time_step_limit=runtime_config.model_config_value("time_step_limit", (0.0, float("inf"))),
            time_step_floor=runtime_config.model_config_value("time_step_floor", 1e-4),
            use_conv_bias=runtime_config.model_config_value("use_conv_bias", True),
            mamba_proj_bias=runtime_config.model_config_value("mamba_proj_bias", False),
            chunk_size=runtime_config.model_config_value("chunk_size", 128),
            mamba_ssm_cache_dtype=runtime_config.model_config_value("mamba_ssm_cache_dtype", "float32"),
            rescale_prenorm_residual=runtime_config.model_config_value("rescale_prenorm_residual", True),
            n_routed_experts=global_num_experts,
            n_shared_experts=runtime_config.model_config_value("n_shared_experts", 1),
            moe_intermediate_size=global_moe_intermediate,
            moe_shared_expert_intermediate_size=global_shared_intermediate,
            moe_latent_size=global_latent_size,
            moe_shared_expert_overlap=runtime_config.model_config_value("moe_shared_expert_overlap", True),
            num_experts_per_tok=global_top_k,
            routed_scaling_factor=runtime_config.model_config_value("routed_scaling_factor", 5.0),
            n_group=runtime_config.model_config_value("n_group", 1),
            topk_group=runtime_config.model_config_value("topk_group", 1),
            norm_topk_prob=runtime_config.model_config_value("norm_topk_prob", True),
            num_nextn_predict_layers=0,
            mtp_layers_block_type=[],
            pad_token_id=runtime_config.model_config_value("pad_token_id", 0),
            bos_token_id=runtime_config.model_config_value("bos_token_id", 1),
            eos_token_id=runtime_config.model_config_value("eos_token_id", 2),
        )

        cls.set_block_configs(model_config, block_configs)
        from transformers.initialization import no_init_weights

        with no_init_weights(), deci_x_patcher(cls, block_configs):
            model = model_cls(model_config)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            for buffer in model.buffers():
                if buffer.is_floating_point() or buffer.is_complex():
                    buffer.zero_()

        model.config.block_configs = [block_config.to_dict() for block_config in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "NemotronHForCausalLM"
        model.config.anymodel_arch_info = cls.anymodel_arch_info()
        model.config.num_nextn_predict_layers = 0
        model.config.mtp_layers_block_type = []
        cls._normalize_runtime_tied_weight_metadata(model)
        return model

    @classmethod
    def update_runtime_benchmark_config(cls, config_data: dict[str, Any]) -> None:
        block_configs = config_data.get("block_configs") or []
        if block_configs:
            typed_blocks = maybe_cast_block_configs(block_configs)
            layer_types = [cls._active_layer_type(block_config) for block_config in typed_blocks]
            config_data["layers_block_type"] = layer_types
            config_data["num_hidden_layers"] = len(layer_types)
            config_data["hybrid_override_pattern"] = cls._pattern_from_layer_types(layer_types)
        config_data["architectures"] = ["AnyModel"]
        config_data["base_architecture"] = "NemotronHForCausalLM"
        config_data["model_type"] = "nemotron_h"
        config_data["num_nextn_predict_layers"] = 0
        config_data["mtp_layers_block_type"] = []
        config_data["mtp_hybrid_override_pattern"] = ""
        config_data.pop("auto_map", None)

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: Any) -> list[str]:
        return ["--model-loader-extra-config", '{"enable_weights_track": false}']

    @classmethod
    def postprocess_runtime_benchmark_checkpoint(cls, output_dir: Any) -> None:
        from safetensors.torch import load_file, save_file

        output_dir = Path(output_dir)

        def _runtime_key(name: str) -> str:
            for prefix in ("model.", "backbone."):
                if name == f"{prefix}embedding.weight":
                    return f"{prefix}embeddings.weight"
                if name == f"{prefix}norm.weight":
                    return f"{prefix}norm_f.weight"
            return name

        index_path = output_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = dict(index.get("weight_map") or {})
            shards = sorted(set(weight_map.values()))
            changed = False
            new_weight_map: dict[str, str] = {}
            for shard in shards:
                shard_path = output_dir / shard
                tensors = load_file(shard_path)
                rewritten = {}
                shard_changed = False
                for key, tensor in tensors.items():
                    new_key = _runtime_key(key)
                    rewritten[new_key] = tensor
                    shard_changed = shard_changed or new_key != key
                    new_weight_map[new_key] = shard
                if shard_changed:
                    save_file(rewritten, shard_path)
                    changed = True
            if changed:
                index["weight_map"] = new_weight_map
                index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            return

        single_path = output_dir / "model.safetensors"
        if single_path.exists():
            tensors = load_file(single_path)
            rewritten = {_runtime_key(key): tensor for key, tensor in tensors.items()}
            if set(rewritten) != set(tensors):
                save_file(rewritten, single_path)

    @staticmethod
    def _block_no_op_post_init(decoder_layer):
        """
        Due to the subblock structure of NemotronH always one of the subblock is set to no-op, for a real no-op both attention & ffn no-op should be set to True.
        """
        block_config = decoder_layer.config.block_configs[decoder_layer.layer_idx]
        ffn = block_config.get_subblock("ffn")
        attention = block_config.get_subblock("attention")
        ffn_no_op = ffn is not None and ffn.no_op
        attn_no_op = attention is not None and attention.no_op
        if ffn_no_op and attn_no_op:
            decoder_layer.norm = Same()
            decoder_layer.mixer = MatchingZeros()

    @staticmethod
    def attn_no_op_post_init(decoder_layer):
        NemotronHModelDescriptor._block_no_op_post_init(decoder_layer)

    @staticmethod
    def mlp_no_op_post_init(decoder_layer):
        NemotronHModelDescriptor._block_no_op_post_init(decoder_layer)

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        dummy_block = super().create_dummy_block(original_layer, block_index)
        # Required by `NemotronHModel.forward`.
        dummy_block.block_type = original_layer.block_type
        # Preserve layer_idx if it exists (used by _block_no_op_post_init)
        if hasattr(original_layer, "layer_idx"):
            dummy_block.layer_idx = original_layer.layer_idx
        # Preserve config if it exists (used by _block_no_op_post_init to access block_configs)
        if hasattr(original_layer, "config"):
            dummy_block.config = original_layer.config
        return dummy_block

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """
        NemotronH has no positional embeddings
        """

    @staticmethod
    def input_embedding_name():
        return "backbone.embeddings"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "backbone.norm_f"

    @staticmethod
    def layer_block_name(index: int):
        return f"backbone.layers.{index}"

    @classmethod
    def local_kd_subblock_module_paths(
        cls, block_config: BlockConfig, *, layer_idx: int
    ) -> dict[tuple[str, str], str]:
        """Map each hybrid sublayer to Nemotron-H's exclusive mixer module."""

        del cls, layer_idx
        return {
            (subblock.kind, subblock.name): "mixer"
            for subblock in block_config.subblock_configs
        }

    @classmethod
    def adapt_module_name_for_model(cls, module_name: str, model: nn.Module) -> str:
        if module_name.startswith("backbone."):
            candidate = f"model.{module_name[len('backbone.'):]}"
            try:
                model.get_submodule(candidate)
                return candidate
            except AttributeError:
                pass
        return module_name

    @classmethod
    def _num_routed_experts_for_layer(cls, config: Any, layer_idx: int) -> int | None:
        block_configs = getattr(config, "block_configs", None)
        if block_configs:
            typed_blocks = maybe_cast_block_configs(block_configs)
            if 0 <= layer_idx < len(typed_blocks):
                moe = typed_blocks[layer_idx].get_subblock("moe")
                if moe is not None and moe.num_experts is not None:
                    return int(moe.num_experts)
        lm_config = cls.get_language_model_config(config)
        num_experts = getattr(lm_config, "n_routed_experts", None)
        return None if num_experts is None else int(num_experts)

    @classmethod
    def checkpoint_key_candidates_for_model_key(
        cls,
        model_key: str,
        *,
        model: nn.Module,
        config: Any,
    ) -> tuple[str, ...]:
        match = cls._FUSED_EXPERT_MODEL_KEY_RE.match(model_key)
        if match:
            _, layer_idx_s, proj = match.groups()
            layer_idx = int(layer_idx_s)
            num_experts = cls._num_routed_experts_for_layer(config, layer_idx)
            if num_experts is not None:
                candidates = []
                for prefix in ("backbone", "model"):
                    candidates.extend(
                        f"{prefix}.layers.{layer_idx}.mixer.experts.{expert_idx}.{proj}.weight"
                        for expert_idx in range(num_experts)
                    )
                candidates.append(model_key)
                return tuple(candidates)
        if model_key.startswith("model."):
            return (model_key, f"backbone.{model_key[len('model.'):]}")
        if model_key.startswith("backbone."):
            return (model_key, f"model.{model_key[len('backbone.'):]}")
        return (model_key,)

    @classmethod
    def adapt_loaded_state_dict_for_model(
        cls,
        state_dict: dict[str, Any],
        *,
        model: nn.Module,
        config: Any,
        checkpoint_to_model_key: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        model_state = model.state_dict()
        fused_experts: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
        rewritten: dict[str, Any] = {}

        for checkpoint_key, tensor in state_dict.items():
            model_key = (
                checkpoint_to_model_key.get(checkpoint_key, checkpoint_key)
                if checkpoint_to_model_key
                else checkpoint_key
            )
            split_match = cls._SPLIT_EXPERT_CHECKPOINT_KEY_RE.match(checkpoint_key)
            if split_match and cls._FUSED_EXPERT_MODEL_KEY_RE.match(model_key):
                _, _, expert_idx_s, _ = split_match.groups()
                fused_experts[model_key].append((int(expert_idx_s), tensor))
                continue
            rewritten[model_key] = tensor

        for model_key, expert_tensors in fused_experts.items():
            expected = model_state.get(model_key)
            expert_tensors.sort(key=lambda item: item[0])
            if expected is not None and len(expert_tensors) != expected.shape[0]:
                raise ValueError(
                    f"NemotronH fused expert load for {model_key} expected "
                    f"{expected.shape[0]} experts, got {len(expert_tensors)}"
                )
            expert_ids = [idx for idx, _ in expert_tensors]
            if expert_ids != list(range(len(expert_ids))):
                raise ValueError(
                    f"NemotronH fused expert load for {model_key} has non-contiguous "
                    f"expert ids: {expert_ids[:8]}...{expert_ids[-8:]}"
                )
            fused = torch.stack([tensor for _, tensor in expert_tensors], dim=0)
            if expected is not None and tuple(fused.shape) != tuple(expected.shape):
                raise ValueError(
                    f"NemotronH fused expert load for {model_key} produced shape "
                    f"{tuple(fused.shape)}, expected {tuple(expected.shape)}"
                )
            rewritten[model_key] = fused

        return cls.adapt_materialized_state_dict_for_model(
            rewritten,
            model=model,
            config=config,
        )

    @classmethod
    def adapt_materialized_state_dict_for_model(
        cls,
        state_dict: dict[str, Any],
        *,
        model: nn.Module,
        config: Any,
    ) -> dict[str, Any]:
        model_keys = set(model.state_dict().keys())
        wants_model = any(key.startswith("model.") for key in model_keys)
        wants_backbone = any(key.startswith("backbone.") for key in model_keys)
        has_backbone = any(key.startswith("backbone.") for key in state_dict)
        has_model = any(key.startswith("model.") for key in state_dict)
        if not (wants_model and not wants_backbone and has_backbone and not has_model):
            return state_dict

        rewritten = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                key = f"model.{key[len('backbone.'):]}"
            rewritten[key] = value
        return rewritten

    @classmethod
    def pipeline_module_fqns_per_model_part(
        cls,
        config: Any,
        *,
        pp_size: int,
        pipeline_config: dict[str, Any] | None = None,
    ) -> list[list[str]] | None:
        """Use NemotronH's real HF remote-code names for AutoModel PP splitting.

        NeMo's built-in HF splitter assumes ``model.embed_tokens`` and ``model.norm``.
        NemotronH exposes the token embedding and final norm as ``model.embeddings`` and
        ``model.norm_f``.  If we let the generic split run, stage 0 drops the embedding and
        the first PP forward receives integer token ids with no embedding module.  Keeping
        these names in the descriptor makes the PP layout explicit for NemotronH and gives
        future remote-code families one place to declare their own split names.
        """
        pp_size = int(pp_size or 1)
        if pp_size <= 1:
            return None

        pipeline_config = dict(pipeline_config or {})
        if pipeline_config.get("module_fqns_per_model_part"):
            return None
        layers_per_stage = pipeline_config.get("layers_per_stage")
        if layers_per_stage not in (None, "none", "None"):
            raise ValueError(
                "NemotronH descriptor PP splitting currently expects NeMo's even "
                "one-stage-per-rank layout (layers_per_stage unset)."
            )
        force_hf = bool(pipeline_config.get("_puzzletron_force_hf", True))
        pp_schedule = str(pipeline_config.get("pp_schedule") or "1f1b").lower()
        single_stage_schedules = {"1f1b", "gpipe"}
        num_stages = pp_size if pp_schedule in single_stage_schedules else pp_size * 2

        lm_config = cls.get_language_model_config(config)
        num_layers = getattr(lm_config, "num_hidden_layers", None)
        if num_layers is None:
            layer_types = getattr(lm_config, "layers_block_type", None)
            num_layers = len(layer_types) if layer_types is not None else None
        if num_layers is None:
            raise ValueError("NemotronH PP split needs num_hidden_layers in the model config")

        if force_hf:
            first_stage_fqns = ("model.embeddings",)
            last_stage_fqns = ("model.norm_f", "lm_head")
        else:
            first_stage_fqns = ("model.embed_tokens",)
            last_stage_fqns = ("model.norm", "lm_head")

        return cls.build_sequential_pipeline_module_fqns(
            num_stages=num_stages,
            num_layers=int(num_layers),
            first_stage_fqns=first_stage_fqns,
            layer_fqn_template="model.layers.{layer_idx}",
            last_stage_fqns=last_stage_fqns,
        )

    @classmethod
    def patch_pipeline_model_part(cls, model_part: nn.Module) -> bool:
        """Install transient aliases expected by NeMo's generic HF PP forward.

        The aliases live only on the stage-local model chunk:
        ``embed_tokens`` points at NemotronH's ``embeddings``, ``norm`` points at
        ``norm_f``, and ``rotary_emb`` is explicitly absent.  This lets NeMo's patched
        pipeline forward drive NemotronH without changing the checkpoint or the NeMo fork.
        """
        inner = getattr(model_part, "model", model_part)
        changed = False

        embeddings = getattr(inner, "embeddings", None)
        if embeddings is not None and getattr(inner, "embed_tokens", None) is None:
            object.__setattr__(inner, "embed_tokens", embeddings)
            changed = True

        norm_f = getattr(inner, "norm_f", None)
        if norm_f is not None and getattr(inner, "norm", None) is None:
            object.__setattr__(inner, "norm", norm_f)
            changed = True

        if not hasattr(inner, "rotary_emb"):
            object.__setattr__(inner, "rotary_emb", None)
            changed = True

        return changed

    @classmethod
    def get_weight_groups(
        cls, layer_names: Iterable[str], num_hidden_layers: int
    ) -> Dict[str, List[str]]:
        """
        Problem with NemotronH is that `norm.weight` can be in both block_{i}_ffn and block_{i}_attention. duplicate groups with `norm.weight` should be removed.
        """
        weight_groups = defaultdict(list)
        for name in layer_names:
            is_matched = False
            for group, pattern in cls.layer_name_predicates(num_hidden_layers).items():
                if pattern.match(name):
                    weight_groups[group].append(name)
                    is_matched = True
            if not is_matched:
                raise ValueError(f"Couldn't find a match for {name}")

        valid_weight_groups = {}
        for group, names in weight_groups.items():
            if len(names) == 1:
                only_name = names[0]
                if only_name.endswith("norm.weight") and "layers" in only_name:
                    # Skip and don't append this group to valid_weight_groups
                    continue
            valid_weight_groups[group] = names

        return valid_weight_groups

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        layer_name_patterns = {
            "embeddings": re.compile(
                r"^(model\.(embed_tokens|embeddings)\.weight|backbone\.embeddings?\.weight)$"
            ),
            "lm_head": re.compile(
                r"^(lm_head\.weight|backbone\.norm_f\.weight|model\.(norm_f|norm)\.weight)$"
            ),
        }

        def build_ffn_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_ffn": re.compile(
                    rf"^(backbone|model)\.layers\.{layer_idx}\."
                    r"(norm\.weight|"  # ← INCLUDED IN FFN
                    r"mixer\.(gate\.e_score_correction_bias"
                    r"|gate\.weight"
                    r"|experts\.(up_proj|down_proj|gate_up_proj)(\.weight)?"
                    r"|experts\.(gate_and_up_projs|down_projs)"
                    r"|experts\.\d+\.up_proj\.weight"
                    r"|experts\.\d+\.down_proj\.weight"
                    r"|fc1_latent_proj\.weight"
                    r"|fc1_latent_proj\.bias"
                    r"|fc2_latent_proj\.weight"
                    r"|fc2_latent_proj\.bias"
                    r"|shared_experts\.up_proj\.weight"
                    r"|shared_experts\.down_proj\.weight))$"
                )
                for layer_idx in range(num_layers)
            }

        def build_attention_predicates() -> Dict[str, re.Pattern]:
            return {
                f"block_{layer_idx}_attention": re.compile(
                    rf"^(backbone|model)\.layers\.{layer_idx}\."
                    r"(norm\.weight|"  # ← INCLUDED IN ATTENTION
                    r"mixer\.(norm\.weight"
                    r"|A_log"
                    r"|D"
                    r"|conv1d\.weight"
                    r"|conv1d\.bias"
                    r"|dt_bias"
                    r"|in_proj\.weight"
                    r"|out_proj\.weight"
                    r"|q_proj\.weight"
                    r"|k_proj\.weight"
                    r"|v_proj\.weight"
                    r"|o_proj\.weight))$"
                )
                for layer_idx in range(num_layers)
            }

        layer_name_patterns.update(
            **build_ffn_predicates(),
            **build_attention_predicates(),
        )

        return layer_name_patterns

    @staticmethod
    def sorted_teacher_layout_kwargs(lm_config) -> dict:
        return {
            "mlp_module": "mixer",
            "attn_module": "mixer",
            "moe_module": "mixer",
            "mamba_module": "mixer",
            # Nemotron dense FFN is non-gated; using up_proj for both gate/up keeps
            # existing FFN surgery paths working without introducing a fake tensor key.
            "ffn_subnames": ("up_proj", "up_proj", "down_proj"),
            "attn_subnames": ("q_proj", "k_proj", "v_proj", "o_proj"),
        }

    @classmethod
    def embedding_pruning_spec(cls, config, *, widths, alignment: int):
        """Describe Nemotron-H's shared residual width across hybrid layers.

        The consolidated checkpoint uses ``backbone`` plus split experts while
        native AutoModel uses ``model`` plus fused expert tensors.  Rules cover
        both representations so ranking, runtime prefix views, and physical HF
        materialization share one descriptor-owned contract.
        """

        hidden_size = int(config.hidden_size)
        layer = r"(?:(?:backbone|model)\.layers|mtp\.layers)\.\d+"
        rules = [
            TensorAxisRule(
                r"^(?:backbone\.embeddings|model\.embed_tokens)\.weight$",
                (1,),
                "Nemotron token embedding channels",
            ),
            TensorAxisRule(
                r"^lm_head\.weight$",
                (1,),
                "Nemotron language-head input channels",
            ),
            TensorAxisRule(
                r"^(?:backbone\.norm_f|model\.norm)\.weight$",
                (0,),
                "Nemotron final residual normalization channels",
            ),
            TensorAxisRule(
                r"^mtp\.layers\.\d+\.(?:enorm|hnorm)\.weight$",
                (0,),
                "Nemotron MTP fusion normalization channels",
            ),
            TensorAxisRule(
                r"^mtp\.layers\.\d+\.eh_proj\.weight$",
                (0,),
                "Nemotron MTP residual fusion input/output channels",
                chunked_axes=((1, 2),),
            ),
            TensorAxisRule(
                rf"^{layer}\.(?:norm|final_layernorm)\.weight$",
                (0,),
                "Nemotron decoder residual normalization channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.(?:q_proj|k_proj|v_proj|in_proj|up_proj)\.weight$",
                (1,),
                "Nemotron attention, Mamba, or dense residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.(?:o_proj|out_proj|down_proj)\.(?:weight|bias)$",
                (0,),
                "Nemotron attention, Mamba, or dense residual output channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.gate\.weight$",
                (1,),
                "Nemotron MoE router residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.shared_experts\.up_proj\.weight$",
                (1,),
                "Nemotron shared expert residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.shared_experts\.down_proj\.weight$",
                (0,),
                "Nemotron shared expert residual output channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.fc1_latent_proj\.weight$",
                (1,),
                "Nemotron routed-expert residual-to-latent input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mixer\.fc2_latent_proj\.weight$",
                (0,),
                "Nemotron routed-expert latent-to-residual output channels",
            ),
        ]
        if getattr(config, "moe_latent_size", None) is None:
            rules.extend(
                (
                    TensorAxisRule(
                        rf"^{layer}\.mixer\.experts\.\d+\.up_proj\.weight$",
                        (1,),
                        "Nemotron non-latent routed-expert residual input channels",
                    ),
                    TensorAxisRule(
                        rf"^{layer}\.mixer\.experts\.\d+\.down_proj\.weight$",
                        (0,),
                        "Nemotron non-latent routed-expert residual output channels",
                    ),
                    TensorAxisRule(
                        rf"^{layer}\.mixer\.experts\.gate_and_up_projs$",
                        (1,),
                        "Nemotron native non-latent routed-expert residual input channels",
                    ),
                    TensorAxisRule(
                        rf"^{layer}\.mixer\.experts\.down_projs$",
                        (2,),
                        "Nemotron native non-latent routed-expert residual output channels",
                    ),
                )
            )
        ties = (
            (("backbone.embeddings.weight", "lm_head.weight"),)
            if bool(getattr(config, "tie_word_embeddings", False))
            else ()
        )
        return EmbeddingPruningSpec(
            hidden_size=hidden_size,
            legal_widths=tuple(int(width) for width in widths),
            alignment=int(alignment),
            tensor_rules=tuple(rules),
            exempt_patterns=(
                rf"^{layer}\.mixer\.(?:A_log|D|dt_bias)$",
                rf"^{layer}\.mixer\.(?:conv1d|norm)\.",
                rf"^{layer}\.mixer\.gate\.e_score_correction_bias$",
            ),
            tie_groups=ties,
            config_paths=(("hidden_size",),),
            residual_norm_patterns=(
                r"^(?:backbone|model)\.layers\.\d+\.norm$",
            ),
        )

    @classmethod
    def checkpoint_equivalence_tolerances(cls) -> dict[str, float]:
        """Account for measured BF16 reduction-order drift after a residual permutation.

        Structural-only sorting moved LM loss by 3.89e-4.  Adding the exact global
        hidden-basis permutation moved it by 2.714e-3 while KL remained 3.088e-3
        and top-1 agreement 0.9829.  The transform is tensor-exact; the additional
        output drift comes from GEMM accumulation order across 52 hybrid layers.
        """

        return {
            "max_abs_lm_loss_delta": 5.0e-3,
            "max_kl_div": 1.0e-2,
            "min_top_1_logit_agreement": 0.9,
        }

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                NemotronHFFNIntermediateLayerDescriptor()
            ),
            "experts_removal": ExpertRemovalPruningMixIn(NemotronHExpertRemovalLayerDescriptor()),
            "kv_heads": KVHeadsPruningMixIn(NemotronHKVHeadsLayerDescriptor()),
            "moe_experts": MoEPruningMixIn(
                NemotronHMoELayerDescriptor(target_name="mixer", require_attrs=("gate", "experts"))
            ),
            "moe_expert_removal": MoEPruningMixIn(
                NemotronHMoELayerDescriptor(target_name="mixer", require_attrs=("gate", "experts"))
            ),
            "moe_expert_intermediate": MoEPruningMixIn(
                NemotronHMoELayerDescriptor(
                    target_name=r"regex:(^|\.)(backbone|model)?\.?layers\.\d+\.mixer\.experts(\.\d+\.down_proj)?$"
                )
            ),
            "moe_shared_expert_intermediate": MoEPruningMixIn(
                NemotronHMoELayerDescriptor(target_name="mixer.shared_experts.down_proj")
            ),
            "moe_latent_dim": MoEPruningMixIn(
                NemotronHMoELayerDescriptor(target_name="mixer", require_attrs=("gate", "experts"))
            ),
            "mamba_heads": MambaPruningMixIn(
                NemotronHMambaLayerDescriptor(target_name="mixer.in_proj")
            ),
            "mamba_head_dim": MambaPruningMixIn(
                NemotronHMambaLayerDescriptor(target_name="mixer.in_proj")
            ),
        }

    @staticmethod
    def puzzletron_capabilities(config):
        caps = default_capabilities(
            descriptor_name="nemotron_h",
            model_family="nemotron_h",
            native_automodel_supported=True,
        )
        axes = dict(caps.axes)
        axes["hidden_width"] = AxisCapabilities(
            axis_id="hidden_width",
            subblock_kind="model",
            field="hidden_size",
            score_hooks=("minitron_hidden_width",),
            sort_impl="sorted_teacher.embedding",
            materialize_impl="materialize.hidden_width",
            runtime_slice_impl="runtime_hidden_width",
            vllm_export=True,
            native_automodel_required=True,
            constraints=("global_residual_width", "hybrid_mamba_moe"),
        )
        # Nemotron-H alternates attention/Mamba mixers with routed MoE mixers;
        # it has no dense FFN sublayer to rank or replace.
        axes.pop("ffn_intermediate", None)
        axes.pop("v_head_dim", None)
        axes.pop("mamba_state_dim", None)
        axes["moe_experts"] = replace(
            axes["moe_experts"],
            score_hooks=("removed_expert_diff",),
            sort_impl="sorted_teacher.moe_experts",
            materialize_impl="materialize.moe_experts",
            runtime_slice_impl="solution_recipe.moe_expert_reroute",
            native_automodel_required=True,
            vllm_export=True,
        )
        axes["moe_expert_intermediate"] = replace(
            axes["moe_expert_intermediate"],
            score_hooks=("moe_channel",),
            native_automodel_required=True,
        )
        axes["moe_shared_expert_intermediate"] = replace(
            axes["moe_shared_expert_intermediate"],
            score_hooks=("shared_expert_intermediate_contribution",),
            native_automodel_required=True,
        )
        if getattr(config, "moe_latent_size", None) is None:
            axes.pop("moe_latent_dim", None)
        else:
            axes["moe_latent_dim"] = replace(
                axes["moe_latent_dim"],
                score_hooks=("moe_latent",),
                native_automodel_required=False,
                constraints=(
                    "latent_projection_present",
                    "activation_covariance",
                    "rotation_metadata",
                ),
            )
        axes["moe_top_k"] = replace(
            axes["moe_top_k"],
            score_hooks=(),
            materialize_impl="materialize.config_only_moe_top_k",
            native_automodel_required=False,
            vllm_export=True,
        )
        axes["mamba_heads"] = replace(
            axes["mamba_heads"],
            score_hooks=("mamba_head_and_dim",),
            native_automodel_required=False,
            vllm_export=True,
        )
        axes["mamba_head_dim"] = replace(
            axes["mamba_head_dim"],
            score_hooks=("mamba_head_and_dim",),
            native_automodel_required=False,
            vllm_export=True,
        )
        return replace(
            caps,
            axes=axes,
            parallelism=ParallelCapabilities(
                tp=True,
                pp=True,
                cp=True,
                fsdp=True,
                ep=True,
                sequence_parallel=False,
                invalid_combinations=("force_hf+ep", "sequence_parallel"),
            ),
            export=replace(ExportCapabilities(), hf=True, vllm=True, per_layer_config=True, mamba_cache=True),
            notes=("Nemotron3 MoE/Mamba axes are descriptor-bound; force_hf=False is required for EP.",),
        )
