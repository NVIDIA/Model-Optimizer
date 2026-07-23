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

# mypy: ignore-errors

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionRotaryEmbedding,
)

from ....block_config import AttentionConfig, BlockConfig, MoEConfig, maybe_cast_block_configs
from ....pruning.embedding_pruning import EmbeddingPruningSpec, TensorAxisRule
from ....pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from ....pruning.gated_delta_net_pruning_mixin import (
    GatedDeltaNetLayerDescriptor,
    GatedDeltaNetPruningMixIn,
)
from ....pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor, KVHeadsPruningMixIn
from ....pruning.pruning_mixin import PruningMixIn
from ....pruning.pruning_utils import (
    GQAInitMode,
    _init_attention_biases,
    _init_attention_weights,
    _lm_head_dim,
)
from ....utils.dummy_modules import DummyBlock
from ...capabilities import AxisCapabilities, default_capabilities
from ...model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ...puzzformer import deci_x_patcher
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size
from ..generic_decoder import (
    DecoderLayout,
    GenericContractModelDescriptor,
    GenericDecoderContract,
    MTPContract,
    RoutedMoEContract,
    StandardGQAAttentionContract,
    VisionLanguageContract,
)

__all__ = [
    "Qwen3P5TextModelDescriptor",
    "Qwen3P5VLModelDescriptor",
    "Qwen3P5TextFFNIntermediateLayerDescriptor",
    "Qwen3P5VLFFNIntermediateLayerDescriptor",
    "Qwen3P5TextKVHeadsLayerDescriptor",
    "Qwen3P5VLKVHeadsLayerDescriptor",
    "Qwen3P5KVHeadsPruningMixIn",
    "Qwen3P5TextGatedDeltaNetLayerDescriptor",
    "Qwen3P5VLGatedDeltaNetLayerDescriptor",
    "Qwen3P5MoeTextModelDescriptor",
    "Qwen3P5MoeVLModelDescriptor",
]


_QWEN3P5_ANYMODEL_ARCH_INFO = {
    "decoder_layer_module": ".qwen3_5",
    "decoder_layer_class": "Qwen3_5DecoderLayer",
    "base_model_module": ".qwen3_5",
    "layers_path": "model.layers",
    "layer_hf_config": "text_config",
}

_QWEN3P5_TEXT_RUNTIME_CONFIG_KEYS = {
    "architectures",
    "anymodel_arch_info",
    "base_architecture",
    "block_configs",
}


def _qwen3p5_embedding_pruning_spec(
    config,
    *,
    widths,
    alignment: int,
    language_prefix: str,
    is_vlm: bool,
) -> EmbeddingPruningSpec:
    lm_config = config.text_config if hasattr(config, "text_config") else config
    hidden_size = int(lm_config.hidden_size)
    escaped = re.escape(language_prefix)
    layer_root = rf"(?:{escaped}\.layers\.\d+|mtp\.layers\.\d+)"
    rules = [
        TensorAxisRule(
            rf"^{escaped}\.embed_tokens\.weight$",
            (1,),
            "language token embedding channels",
        ),
        TensorAxisRule(r"^lm_head\.weight$", (1,), "language head input channels"),
        TensorAxisRule(
            rf"^(?:{escaped}\.norm|mtp\.norm|mtp\.pre_fc_norm_embedding|mtp\.pre_fc_norm_hidden)\.weight$",
            (0,),
            "residual normalization channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.(?:input_layernorm|post_attention_layernorm)\.weight$",
            (0,),
            "decoder residual normalization channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.mlp\.(?:up_proj|gate_proj)\.weight$",
            (1,),
            "FFN residual input channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.mlp\.down_proj\.(?:weight|bias)$",
            (0,),
            "FFN residual output channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$",
            (1,),
            "attention residual input channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.self_attn\.o_proj\.(?:weight|bias)$",
            (0,),
            "attention residual output channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.linear_attn\.in_proj_(?:qkv|z|b|a)\.weight$",
            (1,),
            "GDN residual input channels",
        ),
        TensorAxisRule(
            rf"^{layer_root}\.linear_attn\.out_proj\.(?:weight|bias)$",
            (0,),
            "GDN residual output channels",
        ),
        TensorAxisRule(
            r"^mtp\.fc\.weight$",
            (0,),
            "MTP residual fusion input/output channels",
            chunked_axes=((1, 2),),
        ),
        TensorAxisRule(r"^mtp\.fc\.bias$", (0,), "MTP residual fusion bias"),
    ]
    exempt = [r"(?:^|\.)rotary_emb\.", r"(?:^|\.)inv_freq$"]
    config_paths = (
        (("text_config", "hidden_size"), ("vision_config", "out_hidden_size"))
        if is_vlm
        else (("hidden_size",),)
    )
    if is_vlm:
        rules.extend(
            [
                TensorAxisRule(
                    r"^model\.visual\.merger\.linear_fc2\.weight$",
                    (0,),
                    "VLM projector language-width output",
                ),
                TensorAxisRule(
                    r"^model\.visual\.merger\.linear_fc2\.bias$",
                    (0,),
                    "VLM projector language-width bias",
                ),
            ]
        )
        exempt.append(r"^model\.visual\.")
    ties = ()
    if bool(getattr(lm_config, "tie_word_embeddings", False)):
        ties = ((f"{language_prefix}.embed_tokens.weight", "lm_head.weight"),)
    return EmbeddingPruningSpec(
        hidden_size=hidden_size,
        legal_widths=tuple(int(width) for width in widths),
        alignment=int(alignment),
        tensor_rules=tuple(rules),
        exempt_patterns=tuple(exempt),
        tie_groups=ties,
        config_paths=config_paths,
        residual_norm_patterns=(
            rf"^{escaped}\.layers\.\d+\.input_layernorm$",
            rf"^{escaped}\.layers\.\d+\.post_attention_layernorm$",
        ),
    )


def _wrap_qwen3p5_text_runtime_config(config_data: dict[str, Any]) -> None:
    """Wrap a flat Qwen3.5 text config for vLLM AnyModel.

    vLLM has Qwen3.5 support through the top-level qwen3_5 config, while the
    temporary runtime benchmark model is saved from the text-only HF class.
    """
    text_config = {
        key: value
        for key, value in config_data.items()
        if key not in _QWEN3P5_TEXT_RUNTIME_CONFIG_KEYS
    }
    block_configs = config_data.get("block_configs")
    architectures = config_data.get("architectures", ["AnyModel"])
    base_architecture = config_data.get("base_architecture", "Qwen3_5ForCausalLM")

    config_data.clear()
    config_data.update(
        {
            "architectures": architectures,
            "anymodel_arch_info": dict(_QWEN3P5_ANYMODEL_ARCH_INFO),
            "base_architecture": base_architecture,
            "model_type": "qwen3_5",
            "text_config": text_config,
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        }
    )
    if block_configs is not None:
        config_data["block_configs"] = block_configs


def _wrap_qwen3p5_moe_text_runtime_config(config_data: dict[str, Any]) -> None:
    """Wrap a flat Qwen3.5-MoE text config for vLLM AnyModel."""
    text_config = {
        key: value
        for key, value in config_data.items()
        if key not in _QWEN3P5_TEXT_RUNTIME_CONFIG_KEYS
    }
    block_configs = config_data.get("block_configs")
    architectures = config_data.get("architectures", ["AnyModel"])
    base_architecture = config_data.get("base_architecture", "Qwen3_5MoeForCausalLM")

    config_data.clear()
    config_data.update(
        {
            "architectures": architectures,
            "anymodel_arch_info": dict(_QWEN3P5_ANYMODEL_ARCH_INFO),
            "base_architecture": base_architecture,
            "model_type": "qwen3_5_moe",
            "text_config": text_config,
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        }
    )
    if block_configs is not None:
        config_data["block_configs"] = block_configs


class _Qwen3P5BaseModelDescriptor(ModelDescriptor):
    @staticmethod
    def position_id_axes(config) -> int:
        del config
        return 3

    @classmethod
    def local_kd_subblock_module_paths(
        cls, block_config: BlockConfig, *, layer_idx: int
    ) -> dict[tuple[str, str], str]:
        del layer_idx
        module_by_kind = {
            "attention": "self_attn",
            "mamba": "linear_attn",
            "ffn": "mlp",
            "moe": "mlp",
        }
        return {
            (subblock.kind, subblock.name): module_by_kind[subblock.kind]
            for subblock in block_config.subblock_configs
        }

    @classmethod
    def puzzletron_capabilities(cls, config):
        model_type = getattr(config, "model_type", None)
        text_config = getattr(config, "text_config", None)
        text_model_type = getattr(text_config, "model_type", None)
        family = (
            "qwen3_6"
            if model_type in {"qwen3_6", "qwen3_6_text"} or text_model_type == "qwen3_6_text"
            else "qwen3_5"
        )
        caps = default_capabilities(
            descriptor_name=family,
            model_family=family,
            native_automodel_supported=True,
        )
        axes = dict(caps.axes)
        # Qwen3.5/Qwen3.6 full attention keeps head_dim tied to Q/K RoPE, q/k norms, and
        # the gated q_proj layout.  Keep only whole-KV-group and query-head axes by default.
        axes.pop("qk_head_dim", None)
        axes.pop("v_head_dim", None)
        # This base descriptor owns the dense Qwen3.5/3.6 variants.  The MoE variants
        # have a separate descriptor below and opt into their expert axes there.  Keeping
        # the broad defaults here used to make dense checkpoints advertise nonexistent
        # expert tensors during generated campaigns.
        for axis_id in tuple(axes):
            if axis_id.startswith("moe_"):
                axes.pop(axis_id)
        axes.update(
            {
                "hidden_width": AxisCapabilities(
                    axis_id="hidden_width",
                    subblock_kind="model",
                    field="hidden_size",
                    score_hooks=("minitron_hidden_width",),
                    sort_impl="embedding_pruning.global_residual_permutation",
                    materialize_impl="materialize.hidden_width",
                    runtime_slice_impl="embedding_pruning.prefix_view",
                    vllm_export=True,
                    constraints=("global_width", "tp_aligned", "pp_envelope"),
                ),
                "gdn_key_groups": AxisCapabilities(
                    axis_id="gdn_key_groups",
                    subblock_kind="mamba",
                    field="num_groups",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                    constraints=("gdn_nested_groups",),
                ),
                "gdn_value_heads_per_group": AxisCapabilities(
                    axis_id="gdn_value_heads_per_group",
                    subblock_kind="mamba",
                    field="num_heads",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                    constraints=("gdn_nested_groups",),
                ),
                "gdn_key_head_dim": AxisCapabilities(
                    axis_id="gdn_key_head_dim",
                    subblock_kind="mamba",
                    field="state_dim",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                ),
                "gdn_value_head_dim": AxisCapabilities(
                    axis_id="gdn_value_head_dim",
                    subblock_kind="mamba",
                    field="head_dim",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                ),
            }
        )
        return replace(
            caps,
            axes=axes,
            notes=(
                "Qwen3.5/Qwen3.6 native AutoModel supported; full-attention pruning "
                "supports whole KV groups and query heads per group, not head-dim axes.",
            ),
        )

    @staticmethod
    def decoder_layer_cls():
        return Qwen3_5DecoderLayer

    @classmethod
    def anymodel_arch_info(cls) -> dict[str, Any]:
        return dict(_QWEN3P5_ANYMODEL_ARCH_INFO)

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5TextFFNIntermediateLayerDescriptor()
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5TextKVHeadsLayerDescriptor()),
            "gated_delta_net": GatedDeltaNetPruningMixIn(Qwen3P5TextGatedDeltaNetLayerDescriptor()),
        }

    @staticmethod
    def passthrough_weight_name_predicates() -> Dict[str, re.Pattern]:
        return {"mtp": re.compile(r"^mtp.*")}

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        dummy = DummyBlock(block_index=block_index)
        if hasattr(original_layer, "layer_type"):
            dummy.layer_type = original_layer.layer_type
        return dummy

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {}
        ffn = block_config.get_subblock("ffn")
        attention = block_config.get_subblock("attention")
        if ffn is not None:
            override_kwargs["intermediate_size"] = ffn.intermediate_size
        if attention is not None and attention.num_kv_heads is not None:
            override_kwargs["num_key_value_heads"] = attention.num_kv_heads
        # Per-layer query-head count (KV-group removal / within-group query pruning). Only emitted
        # when set; None keeps the model-global num_attention_heads. The HF attention module must
        # size q_proj/o_proj from config.num_attention_heads at __init__ for this to take effect.
        if attention is not None and attention.num_query_heads is not None:
            override_kwargs["num_attention_heads"] = attention.num_query_heads
        mamba = block_config.get_subblock("mamba")
        if mamba is not None:
            override_kwargs.update(
                linear_num_key_heads=mamba.num_groups,
                linear_num_value_heads=mamba.num_heads,
                linear_key_head_dim=mamba.state_dim,
                linear_value_head_dim=mamba.head_dim,
            )
        return override_kwargs

    @staticmethod
    def sorted_teacher_layout_kwargs(lm_config) -> dict[str, Any]:
        # Qwen3.5 full-attention q_proj stores [query, output_gate] rows per head.
        # The default Qwen3.5 search space only prunes whole KV groups and query heads per
        # group.  If an experiment explicitly enables V/head-dim sorting, the gate half must
        # follow the post-attention/o_proj channel order while query rows stay fixed.
        return {
            "q_gate_row_group": 1,
            "mamba_module": "linear_attn",
            "gated_delta_net": True,
        }

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        decoder_layer.input_layernorm = Same()
        if decoder_layer.layer_type == "linear_attention":
            decoder_layer.linear_attn = MatchingZeros()
        else:
            decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3_5DecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @classmethod
    def set_block_configs(cls, model_config, block_configs: list[BlockConfig | dict]) -> None:
        block_configs = maybe_cast_block_configs(block_configs)
        super().set_block_configs(model_config, block_configs)
        lm_config = cls.get_language_model_config(model_config)
        lm_config.layer_types = [
            "linear_attention"
            if block_config.get_subblock("mamba") is not None
            else "full_attention"
            for block_config in block_configs
        ]

    @classmethod
    def truncate_pattern_for_subblock(
        cls, lm_config, parent_layer_index: int | None = None
    ) -> None:
        layer_types = getattr(lm_config, "layer_types", None)
        if not layer_types:
            return super().truncate_pattern_for_subblock(lm_config, parent_layer_index)

        if parent_layer_index is not None and 0 <= parent_layer_index < len(layer_types):
            lm_config.layer_types = [layer_types[parent_layer_index]]
        else:
            lm_config.layer_types = [layer_types[0]]

    @staticmethod
    def _text_attention_pattern(prefix: str, layer_idx: int) -> re.Pattern:
        return re.compile(
            rf"^{prefix}\.{layer_idx}\.(input_layernorm\.weight"
            r"|self_attn\.q_proj\.weight"
            r"|self_attn\.k_proj\.weight"
            r"|self_attn\.v_proj\.weight"
            r"|self_attn\.o_proj\.weight"
            r"|self_attn\.q_norm\.weight"
            r"|self_attn\.k_norm\.weight"
            r"|linear_attn\.conv1d\.weight"
            r"|linear_attn\.dt_bias"
            r"|linear_attn\.A_log"
            r"|linear_attn\.norm\.weight"
            r"|linear_attn\.out_proj\.weight"
            r"|linear_attn\.in_proj_qkv\.weight"
            r"|linear_attn\.in_proj_z\.weight"
            r"|linear_attn\.in_proj_b\.weight"
            r"|linear_attn\.in_proj_a\.weight)$"
        )

    @staticmethod
    def _text_ffn_pattern(prefix: str, layer_idx: int) -> re.Pattern:
        return re.compile(
            rf"^{prefix}\.{layer_idx}\.(post_attention_layernorm\.weight"
            r"|mlp\.up_proj\.weight"
            r"|mlp\.gate_proj\.weight"
            r"|mlp\.down_proj\.weight)$"
        )

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        head_dim = (
            getattr(lm_config, "head_dim", None)
            or lm_config.hidden_size // lm_config.num_attention_heads
        )
        return {
            "head_dim": head_dim,
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "intermediate_size": 256,
            "linear_conv_kernel_dim": getattr(lm_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(lm_config, "linear_key_head_dim", head_dim),
            "linear_num_key_heads": getattr(
                lm_config, "linear_num_key_heads", lm_config.num_key_value_heads
            ),
            "linear_num_value_heads": getattr(
                lm_config, "linear_num_value_heads", lm_config.num_attention_heads
            ),
            "linear_value_head_dim": getattr(lm_config, "linear_value_head_dim", head_dim),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-6),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
        }

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        model_config = Qwen3_5TextConfig(
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            intermediate_size=runtime_config.model_config_value("intermediate_size", 256),
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim"),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            linear_conv_kernel_dim=runtime_config.model_config_value("linear_conv_kernel_dim", 4),
            linear_key_head_dim=runtime_config.model_config_value("linear_key_head_dim"),
            linear_num_key_heads=runtime_config.model_config_value("linear_num_key_heads"),
            linear_num_value_heads=runtime_config.model_config_value("linear_num_value_heads"),
            linear_value_head_dim=runtime_config.model_config_value("linear_value_head_dim"),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-6),
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
        )

        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = Qwen3_5ForCausalLM(model_config)

        model.config.block_configs = [block_config.to_dict() for block_config in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "Qwen3_5ForCausalLM"
        return model

    @classmethod
    def update_runtime_benchmark_config(cls, config_data: dict[str, Any]) -> None:
        if config_data.get("model_type") in {"qwen3_5_text", "qwen3_6_text"}:
            _wrap_qwen3p5_text_runtime_config(config_data)

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: Any) -> list[str]:
        # ``config`` may be a dict or a SimpleNamespace (built from config.json).
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # The runtime benchmark config is wrapped as a multimodal ``qwen3_5``
        # config (with a nested ``text_config``) so vLLM's AnyModel can resolve
        # it. Restrict initialization to the language model so the benchmark
        # loads a pure text model (no vision tower). The generative runner is
        # selected automatically by vLLM's ``AnyModelConfig`` from the base
        # architecture (``Qwen3_5ForCausalLM``); do NOT pass ``--runner
        # generate`` here, as that both disables that auto-detection (which only
        # runs when ``runner == "auto"``) and is rejected by ModelConfig
        # validation before the AnyModel hook runs.
        args = [
            "--language-model-only",
            "--model-loader-extra-config",
            '{"enable_weights_track": false}',
        ]
        text_config = _get(config, "text_config", config)
        layer_types = _get(text_config, "layer_types", [])
        if "linear_attention" in layer_types:
            # Qwen3.5's linear-attention (mamba) layers only support
            # ``--mamba-cache-mode=align`` (the base model rejects ``all``), and
            # that mode's ``preprocess_mamba`` asserts prefix caching is enabled.
            # ``vllm bench latency`` defaults prefix caching OFF, so enable it
            # explicitly or the benchmark aborts during the warmup step.
            args += ["--mamba-cache-mode", "align", "--enable-prefix-caching"]
        return args


@ModelDescriptorFactory.register_decorator("qwen3_6_text")
@ModelDescriptorFactory.register_decorator("qwen3_5_text")
class Qwen3P5TextModelDescriptor(_Qwen3P5BaseModelDescriptor):
    @classmethod
    def embedding_pruning_spec(cls, config, *, widths, alignment: int):
        return _qwen3p5_embedding_pruning_spec(
            config,
            widths=widths,
            alignment=alignment,
            language_prefix="model",
            is_vlm=False,
        )

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5TextFFNIntermediateLayerDescriptor()
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5TextKVHeadsLayerDescriptor()),
            "gated_delta_net": GatedDeltaNetPruningMixIn(Qwen3P5TextGatedDeltaNetLayerDescriptor()),
        }

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        head_dim = (
            getattr(lm_config, "head_dim", None)
            or lm_config.hidden_size // lm_config.num_attention_heads
        )
        return {
            "head_dim": head_dim,
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "intermediate_size": 256,
            "linear_conv_kernel_dim": getattr(lm_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(lm_config, "linear_key_head_dim", head_dim),
            "linear_num_key_heads": getattr(
                lm_config, "linear_num_key_heads", lm_config.num_key_value_heads
            ),
            "linear_num_value_heads": getattr(
                lm_config, "linear_num_value_heads", lm_config.num_attention_heads
            ),
            "linear_value_head_dim": getattr(lm_config, "linear_value_head_dim", head_dim),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-6),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
        }

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        model_config = Qwen3_5TextConfig(
            max_position_embeddings=runtime_config.prefill_seq_len
            + runtime_config.generation_seq_len,
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            intermediate_size=runtime_config.model_config_value("intermediate_size", 256),
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim"),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            linear_conv_kernel_dim=runtime_config.model_config_value("linear_conv_kernel_dim", 4),
            linear_key_head_dim=runtime_config.model_config_value("linear_key_head_dim"),
            linear_num_key_heads=runtime_config.model_config_value("linear_num_key_heads"),
            linear_num_value_heads=runtime_config.model_config_value("linear_num_value_heads"),
            linear_value_head_dim=runtime_config.model_config_value("linear_value_head_dim"),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-6),
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
        )

        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = Qwen3_5ForCausalLM(model_config)

        model.config.block_configs = [block_config.to_dict() for block_config in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "Qwen3_5ForCausalLM"
        return model

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: Any) -> list[str]:
        # ``config`` may be a dict or a SimpleNamespace (built from config.json).
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # The runtime benchmark config is wrapped as a multimodal ``qwen3_5``
        # config (with a nested ``text_config``) so vLLM's AnyModel can resolve
        # it. Restrict initialization to the language model so the benchmark
        # loads a pure text model (no vision tower). The generative runner is
        # selected automatically by vLLM's ``AnyModelConfig`` from the base
        # architecture (``Qwen3_5ForCausalLM``); do NOT pass ``--runner
        # generate`` here, as that both disables that auto-detection (which only
        # runs when ``runner == "auto"``) and is rejected by ModelConfig
        # validation before the AnyModel hook runs.
        args = [
            "--language-model-only",
            "--model-loader-extra-config",
            '{"enable_weights_track": false}',
        ]
        text_config = _get(config, "text_config", config)
        layer_types = _get(text_config, "layer_types", [])
        if "linear_attention" in layer_types:
            # Qwen3.5's linear-attention (mamba) layers only support
            # ``--mamba-cache-mode=align`` (the base model rejects ``all``), and
            # that mode's ``preprocess_mamba`` asserts prefix caching is enabled.
            # ``vllm bench latency`` defaults prefix caching OFF, so enable it
            # explicitly or the benchmark aborts during the warmup step.
            args += ["--mamba-cache-mode", "align", "--enable-prefix-caching"]
        return args

    @staticmethod
    def init_rotary_embedding(model, runtime):
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=model.config).to(
            device=runtime.device, dtype=runtime.dtype
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
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
        }
        layer_name_patterns.update(
            **{
                f"block_{layer_idx}_ffn": _Qwen3P5BaseModelDescriptor._text_ffn_pattern(
                    "model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
            **{
                f"block_{layer_idx}_attention": _Qwen3P5BaseModelDescriptor._text_attention_pattern(
                    "model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
        )
        return layer_name_patterns


@ModelDescriptorFactory.register_decorator("qwen3_6")
@ModelDescriptorFactory.register_decorator("qwen3_5")
class Qwen3P5VLModelDescriptor(_Qwen3P5BaseModelDescriptor):
    @classmethod
    def vision_module_names(cls) -> tuple[str, ...]:
        return ("model.visual",)

    @classmethod
    def embedding_pruning_spec(cls, config, *, widths, alignment: int):
        return _qwen3p5_embedding_pruning_spec(
            config,
            widths=widths,
            alignment=alignment,
            language_prefix="model.language_model",
            is_vlm=True,
        )

    @staticmethod
    def get_language_model_config(config):
        return config.text_config if hasattr(config, "text_config") else config

    @classmethod
    def runtime_benchmark_export_descriptor(cls) -> Type[ModelDescriptor]:
        return Qwen3P5TextModelDescriptor

    @classmethod
    def anymodel_arch_info(cls) -> dict[str, Any]:
        return {
            "decoder_layer_module": ".qwen3_5",
            "decoder_layer_class": "Qwen3_5DecoderLayer",
            "base_model_module": ".qwen3_5",
            "layers_path": "language_model.model.layers",
            "init_prefix": "model",
            "layer_hf_config": "text_config",
        }

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                Qwen3P5VLFFNIntermediateLayerDescriptor()
            ),
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(Qwen3P5VLKVHeadsLayerDescriptor()),
            "gated_delta_net": GatedDeltaNetPruningMixIn(Qwen3P5VLGatedDeltaNetLayerDescriptor()),
        }

    @staticmethod
    def init_rotary_embedding(model, runtime):
        text_config = Qwen3P5VLModelDescriptor.get_language_model_config(model.config)
        model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_config).to(
            device=runtime.device, dtype=runtime.dtype
        )
        vision_config = (
            model.config.vision_config if hasattr(model.config, "vision_config") else None
        )
        if vision_config is not None:
            head_dim = vision_config.hidden_size // vision_config.num_heads
            model.model.visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2).to(
                device=runtime.device, dtype=runtime.dtype
            )

    @staticmethod
    def input_embedding_name():
        return "model.language_model.embed_tokens"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "model.language_model.norm"

    @staticmethod
    def layer_block_name(index: int):
        return f"model.language_model.layers.{index}"

    @staticmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        layer_name_patterns = {
            "embeddings": re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
            "lm_head": re.compile(r"^(model\.language_model\.norm\.weight|lm_head\.weight)$"),
            "vision_encoding": re.compile(r"^model\.visual\..*"),
        }
        layer_name_patterns.update(
            **{
                f"block_{layer_idx}_ffn": _Qwen3P5BaseModelDescriptor._text_ffn_pattern(
                    "model\\.language_model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
            **{
                f"block_{layer_idx}_attention": _Qwen3P5BaseModelDescriptor._text_attention_pattern(
                    "model\\.language_model\\.layers", layer_idx
                )
                for layer_idx in range(num_layers)
            },
        )
        return layer_name_patterns


@dataclass
class Qwen3P5TextFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3P5VLFFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mlp.down_proj"
    ffn_prefix_name: str = "model.language_model.layers.{layer_idx}.mlp"
    linear_weight_names: List[str] = field(
        default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
    )


@dataclass
class Qwen3P5TextGatedDeltaNetLayerDescriptor(GatedDeltaNetLayerDescriptor):
    target_name: str = "linear_attn"
    gdn_prefix_name: str = "model.layers.{layer_idx}.linear_attn"


@dataclass
class Qwen3P5VLGatedDeltaNetLayerDescriptor(GatedDeltaNetLayerDescriptor):
    target_name: str = "linear_attn"
    gdn_prefix_name: str = "model.language_model.layers.{layer_idx}.linear_attn"


@dataclass
class Qwen3P5TextKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class Qwen3P5VLKVHeadsLayerDescriptor(KVHeadsLayerDescriptor):
    o_proj_name: str = "self_attn.o_proj"
    attn_prefix_name: str = "model.language_model.layers.{layer_idx}.self_attn"
    qkvo_weight_names: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def _split_qwen3p5_q_proj(
    tensor: torch.Tensor, num_q_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    q_proj = tensor.reshape(num_q_heads, 2 * head_size, *tensor.shape[1:])
    query = q_proj[:, :head_size].reshape(num_q_heads * head_size, *tensor.shape[1:])
    gate = q_proj[:, head_size:].reshape(num_q_heads * head_size, *tensor.shape[1:])
    return query, gate


def _merge_qwen3p5_q_proj(
    query: torch.Tensor, gate: torch.Tensor, num_q_heads: int, head_size: int
) -> torch.Tensor:
    trailing_shape = query.shape[1:]
    query = query.reshape(num_q_heads, head_size, *trailing_shape)
    gate = gate.reshape(num_q_heads, head_size, *trailing_shape)
    return torch.cat([query, gate], dim=1).reshape(num_q_heads * 2 * head_size, *trailing_shape)


def _state_dict_with_tensor(
    state_dict: dict[str, torch.Tensor], key: str, value: torch.Tensor
) -> dict[str, torch.Tensor]:
    patched_state_dict = dict(state_dict)
    patched_state_dict[key] = value
    return patched_state_dict


class Qwen3P5KVHeadsPruningMixIn(KVHeadsPruningMixIn):
    """KV-head pruning for Qwen3.5 gated full-attention layers."""

    def __init__(self, layer_descriptor: KVHeadsLayerDescriptor):
        assert isinstance(layer_descriptor, KVHeadsLayerDescriptor)
        super().__init__(layer_descriptor)

    def _init_gated_attention_weights(
        self,
        *,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor: Type[_Qwen3P5BaseModelDescriptor],
        q_key: str,
        k_key: str,
        v_key: str,
        o_key: str,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_q_heads = descriptor.get_language_model_config(new_config).num_attention_heads
        orig_query, orig_gate = _split_qwen3p5_q_proj(
            parent_state_dict[q_key], num_q_heads, head_size
        )
        new_query, new_gate = _split_qwen3p5_q_proj(new_state_dict[q_key], num_q_heads, head_size)

        query_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_query)
        query_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_query)
        wq, wk, wv, wo = _init_attention_weights(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=query_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=query_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )

        gate_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_gate)
        gate_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_gate)
        wg, _, _, _ = _init_attention_weights(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=gate_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=gate_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )
        wq = _merge_qwen3p5_q_proj(wq, wg, num_q_heads, head_size)
        return wq, wk, wv, wo

    def _init_gated_attention_biases(
        self,
        *,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor: Type[_Qwen3P5BaseModelDescriptor],
        q_key: str,
        k_key: str,
        v_key: str,
        o_key: str,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        head_size: int,
    ) -> dict[str, torch.Tensor]:
        num_q_heads = descriptor.get_language_model_config(new_config).num_attention_heads
        orig_query, orig_gate = _split_qwen3p5_q_proj(
            parent_state_dict[q_key], num_q_heads, head_size
        )
        new_query, new_gate = _split_qwen3p5_q_proj(new_state_dict[q_key], num_q_heads, head_size)

        query_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_query)
        query_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_query)
        bias_sd = _init_attention_biases(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=query_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=query_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )

        gate_state_dict = _state_dict_with_tensor(parent_state_dict, q_key, orig_gate)
        gate_new_state_dict = _state_dict_with_tensor(new_state_dict, q_key, new_gate)
        gate_bias_sd = _init_attention_biases(
            gqa_init_mode=gqa_init_mode,
            layer_idx=layer_idx,
            new_state_dict=gate_new_state_dict,
            new_config=new_config,
            descriptor=descriptor,
            original_state_dict=gate_state_dict,
            original_config=original_config,
            q_key=q_key,
            k_key=k_key,
            v_key=v_key,
            o_key=o_key,
            is_original_mha=is_original_mha,
            head_size=head_size,
            mlp_init_config=mlp_init_config,
        )
        if "q" in bias_sd:
            bias_sd["q"] = _merge_qwen3p5_q_proj(
                bias_sd["q"], gate_bias_sd["q"], num_q_heads, head_size
            )
        return bias_sd

    def prune_single_layer(
        self,
        layer_idx: int,
        parent_state_dict: dict,
        new_state_dict: dict,
        original_config: PretrainedConfig,
        new_config: PretrainedConfig,
        descriptor,
        gqa_init_mode: GQAInitMode,
        mlp_init_config: Optional[dict[str, Any]],
        is_original_mha: bool,
        keys: dict,
        keys_to_remove: dict,
        **kwargs,
    ):
        layer_out_state_dict = {}
        attn_prefix = self.layer_descriptor.attn_prefix(layer_idx)
        q_name, k_name, v_name, o_name = [
            f"{attn_prefix}.{proj_name}" for proj_name in self.layer_descriptor.qkvo_weight_names
        ]

        head_size = _lm_head_dim(new_config, descriptor)
        for part in ["weight", "bias"]:
            attn_keys = [f"{name}.{part}" for name in [q_name, k_name, v_name, o_name]]
            q_key, k_key, v_key, o_key = attn_keys
            attn_keys = [key for key in attn_keys if key in new_state_dict]
            if not attn_keys or not all(key in keys for key in attn_keys):
                continue

            for key in attn_keys:
                keys_to_remove[key] = keys[key]

            if not all(key in new_state_dict for key in attn_keys):
                continue

            if q_key not in new_state_dict:
                continue

            if part == "weight":
                wq, wk, wv, wo = self._init_gated_attention_weights(
                    layer_idx=layer_idx,
                    parent_state_dict=parent_state_dict,
                    new_state_dict=new_state_dict,
                    original_config=original_config,
                    new_config=new_config,
                    descriptor=descriptor,
                    q_key=q_key,
                    k_key=k_key,
                    v_key=v_key,
                    o_key=o_key,
                    gqa_init_mode=gqa_init_mode,
                    mlp_init_config=mlp_init_config,
                    is_original_mha=is_original_mha,
                    head_size=head_size,
                )
                layer_out_state_dict[q_key], layer_out_state_dict[k_key] = wq, wk
                layer_out_state_dict[v_key], layer_out_state_dict[o_key] = wv, wo
            else:
                bias_sd = self._init_gated_attention_biases(
                    layer_idx=layer_idx,
                    parent_state_dict=parent_state_dict,
                    new_state_dict=new_state_dict,
                    original_config=original_config,
                    new_config=new_config,
                    descriptor=descriptor,
                    q_key=q_key,
                    k_key=k_key,
                    v_key=v_key,
                    o_key=o_key,
                    gqa_init_mode=gqa_init_mode,
                    mlp_init_config=mlp_init_config,
                    is_original_mha=is_original_mha,
                    head_size=head_size,
                )
                for bias_key, sd_key in zip("qkvo", [q_key, k_key, v_key, o_key]):
                    if bias_key in bias_sd:
                        layer_out_state_dict[sd_key] = bias_sd[bias_key]

        return layer_out_state_dict


def _qwen3p5_moe_contract(*, is_vlm: bool) -> GenericDecoderContract:
    language_prefix = "model.language_model" if is_vlm else "model"
    escaped = re.escape(language_prefix)
    layer = rf"{escaped}\.layers\.\d+"
    vision = None
    if is_vlm:
        vision = VisionLanguageContract(
            module_names=("model.visual",),
            projector_output_config_paths=(("vision_config", "out_hidden_size"),),
            projector_rules=(
                TensorAxisRule(
                    r"^model\.visual\.merger\.linear_fc2\.(?:weight|bias)$",
                    (0,),
                    "Qwen MoE VLM projector output",
                ),
            ),
        )
    return GenericDecoderContract(
        descriptor_name="qwen3_5_moe" if is_vlm else "qwen3_5_moe_text",
        model_family="qwen3_5_moe",
        layout=DecoderLayout(
            language_config_path=("text_config",) if is_vlm else (),
            language_prefix=language_prefix,
            layer_template=f"{language_prefix}.layers.{{layer_idx}}",
            input_embedding=f"{language_prefix}.embed_tokens",
            output_embedding="lm_head",
            final_norm=f"{language_prefix}.norm",
            layer_norm_names=("input_layernorm", "post_attention_layernorm"),
        ),
        attention=StandardGQAAttentionContract(),
        routed_moe=RoutedMoEContract(
            module_name="mlp",
            experts_name="experts",
            router_name="gate",
            shared_expert_name="shared_expert",
            num_experts_field="num_experts",
            intermediate_field="moe_intermediate_size",
            shared_intermediate_field="shared_expert_intermediate_size",
        ),
        vision=vision,
        mtp=MTPContract(
            tensor_rules=(
                TensorAxisRule(
                    r"^mtp\.fc\.weight$",
                    (0,),
                    "MTP residual fusion",
                    chunked_axes=((1, 2),),
                ),
                TensorAxisRule(r"^mtp\.fc\.bias$", (0,), "MTP residual fusion bias"),
                TensorAxisRule(
                    r"^mtp\.(?:norm|pre_fc_norm_embedding|pre_fc_norm_hidden)\.weight$",
                    (0,),
                    "MTP residual normalization",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight$",
                    (0,),
                    "MTP decoder normalization",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$",
                    (1,),
                    "MTP attention residual input",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.self_attn\.o_proj\.(?:weight|bias)$",
                    (0,),
                    "MTP attention residual output",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.linear_attn\.in_proj_(?:qkv|z|b|a)\.weight$",
                    (1,),
                    "MTP GDN residual input",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.linear_attn\.out_proj\.(?:weight|bias)$",
                    (0,),
                    "MTP GDN residual output",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.gate\.weight$",
                    (1,),
                    "MTP MoE router residual input",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.experts\.gate_up_proj$",
                    (2,),
                    "MTP fused expert residual input",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.experts\.down_proj$",
                    (1,),
                    "MTP fused expert residual output",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.shared_expert\.(?:gate_proj|up_proj)\.weight$",
                    (1,),
                    "MTP shared expert residual input",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.shared_expert\.down_proj\.(?:weight|bias)$",
                    (0,),
                    "MTP shared expert residual output",
                ),
                TensorAxisRule(
                    r"^mtp\.layers\.\d+\.mlp\.shared_expert_gate\.weight$",
                    (1,),
                    "MTP shared expert gate residual input",
                ),
            ),
        ),
        additional_tensor_rules=(
            TensorAxisRule(
                rf"^{layer}\.linear_attn\.in_proj_(?:qkv|z|b|a)\.weight$",
                (1,),
                "GDN residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.linear_attn\.out_proj\.(?:weight|bias)$",
                (0,),
                "GDN residual output channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.mlp\.experts\.gate_up_proj$",
                (2,),
                "Qwen fused expert residual input",
            ),
            TensorAxisRule(
                rf"^{layer}\.mlp\.experts\.down_proj$",
                (1,),
                "Qwen fused expert residual output",
            ),
            TensorAxisRule(
                rf"^{layer}\.mlp\.shared_expert_gate\.weight$",
                (1,),
                "Qwen shared-expert gate residual input",
            ),
        ),
        native_automodel_supported=True,
        ep_supported=True,
    )


class _Qwen3P5MoeModelDescriptor(GenericContractModelDescriptor):
    DECODER_LAYER_CLS = Qwen3_5MoeDecoderLayer
    _IS_VLM = False

    @classmethod
    def local_kd_subblock_module_paths(
        cls, block_config: BlockConfig, *, layer_idx: int
    ) -> dict[tuple[str, str], str]:
        """Map hybrid Qwen MoE subblocks to their exclusive residual branches."""

        del cls, layer_idx
        module_by_kind = {
            "attention": "self_attn",
            "mamba": "linear_attn",
            "ffn": "mlp",
            "moe": "mlp",
        }
        return {
            (subblock.kind, subblock.name): module_by_kind[subblock.kind]
            for subblock in block_config.subblock_configs
        }

    @classmethod
    def get_language_model_config(cls, config):
        """Accept the native nested VLM config and synthetic flat text config."""
        return config.text_config if hasattr(config, "text_config") else config

    @classmethod
    def checkpoint_equivalence_tolerances(cls) -> dict[str, float]:
        """Gate exact fused-expert permutations in output space.

        Reordering every intermediate channel in every routed expert is exactly
        function preserving in real arithmetic, and the structural verifier
        checks the composed expert/channel/hidden permutation tensor-for-tensor.
        The native grouped BF16 kernels nevertheless accumulate the down
        projection in the new channel order.  Across the 20-layer Qwen3.6 MoE
        this produces a measured LM-loss delta of 0.01396 on the eight-sample
        parent sweep while retaining 97.09% top-1 agreement and KL=0.00489.
        Keep the output gate tight enough to
        catch layout errors (which produced multi-point loss deltas), but account
        for that unavoidable reduction-order sensitivity.
        """
        return {
            "max_abs_lm_loss_delta": 1.5e-2,
            "max_kl_div": 1.0e-2,
            "min_top_1_logit_agreement": 0.95,
        }

    @classmethod
    def generic_decoder_contract(cls, config):
        return _qwen3p5_moe_contract(is_vlm=cls._IS_VLM)

    @staticmethod
    def sorted_teacher_layout_kwargs(lm_config) -> dict[str, Any]:
        del lm_config
        # MoE Qwen3.5/3.6 uses the same gated full-attention and GDN tensor
        # layouts as the dense family.  Declaring this here keeps sorting
        # descriptor-driven instead of relying on the dense subclass.
        return {
            "q_gate_row_group": 1,
            "mamba_module": "linear_attn",
            "gated_delta_net": True,
            "moe_fused_expert_subnames": (
                "experts.gate_up_proj",
                "experts.down_proj",
            ),
            "moe_fused_gate_up_subnames": ("experts.gate_up_proj",),
            "moe_fused_down_subnames": ("experts.down_proj",),
            # HF Qwen MoE uses singular ``shared_expert`` with unfused gate/up.
            "moe_shared_expert_subname": "shared_expert",
            "moe_shared_gate_subname": "gate_proj",
        }

    @classmethod
    def pruning_mixins(cls) -> Dict[str, PruningMixIn]:
        """Selectors shared by dense and MoE Qwen3.5/3.6 token mixers.

        Expert selectors are derived from the generic routed-MoE contract by the
        campaign compiler.  Full attention and GDN retain Qwen's established,
        diagnosed target descriptors because their module layouts are identical in
        the dense and MoE decoder variants.
        """
        kv_descriptor = (
            Qwen3P5VLKVHeadsLayerDescriptor()
            if cls._IS_VLM
            else Qwen3P5TextKVHeadsLayerDescriptor()
        )
        gdn_descriptor = (
            Qwen3P5VLGatedDeltaNetLayerDescriptor()
            if cls._IS_VLM
            else Qwen3P5TextGatedDeltaNetLayerDescriptor()
        )
        return {
            "kv_heads": Qwen3P5KVHeadsPruningMixIn(kv_descriptor),
            "gated_delta_net": GatedDeltaNetPruningMixIn(gdn_descriptor),
        }

    @classmethod
    def puzzletron_capabilities(cls, config):
        caps = super().puzzletron_capabilities(config)
        axes = dict(caps.axes)
        axes.update(
            {
                "gdn_key_groups": AxisCapabilities(
                    axis_id="gdn_key_groups",
                    subblock_kind="mamba",
                    field="num_groups",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                    constraints=("gdn_nested_groups",),
                ),
                "gdn_value_heads_per_group": AxisCapabilities(
                    axis_id="gdn_value_heads_per_group",
                    subblock_kind="mamba",
                    field="num_heads",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                    constraints=("gdn_nested_groups",),
                ),
                "gdn_key_head_dim": AxisCapabilities(
                    axis_id="gdn_key_head_dim",
                    subblock_kind="mamba",
                    field="state_dim",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                ),
                "gdn_value_head_dim": AxisCapabilities(
                    axis_id="gdn_value_head_dim",
                    subblock_kind="mamba",
                    field="head_dim",
                    score_hooks=("gdn_activation_contribution",),
                    sort_impl="sorted_teacher.gated_delta_net",
                    materialize_impl="materialize.gated_delta_net",
                    runtime_slice_impl="solution_recipe.gated_delta_net",
                    vllm_export=True,
                ),
            }
        )
        return replace(caps, axes=axes)

    @classmethod
    def block_config_to_layer_overrides(cls, block_config: BlockConfig):
        overrides = super().block_config_to_layer_overrides(block_config)
        mamba = block_config.get_subblock("mamba")
        if mamba is not None:
            overrides.update(
                linear_num_key_heads=mamba.num_groups,
                linear_num_value_heads=mamba.num_heads,
                linear_key_head_dim=mamba.state_dim,
                linear_value_head_dim=mamba.head_dim,
            )
        return overrides

    @staticmethod
    def attn_no_op_post_init(decoder_layer: Qwen3_5MoeDecoderLayer):
        decoder_layer.input_layernorm = Same()
        if decoder_layer.layer_type == "linear_attention":
            decoder_layer.linear_attn = MatchingZeros()
        else:
            decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: Qwen3_5MoeDecoderLayer):
        decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @classmethod
    def set_block_configs(cls, model_config, block_configs: list[BlockConfig | dict]) -> None:
        block_configs = maybe_cast_block_configs(block_configs)
        super().set_block_configs(model_config, block_configs)
        lm_config = cls.get_language_model_config(model_config)
        lm_config.layer_types = [
            "linear_attention"
            if block_config.get_subblock("mamba") is not None
            else "full_attention"
            for block_config in block_configs
        ]

    @classmethod
    def runtime_benchmark_config_fields(cls, lm_config) -> dict[str, Any]:
        head_dim = (
            getattr(lm_config, "head_dim", None)
            or lm_config.hidden_size // lm_config.num_attention_heads
        )
        return {
            "head_dim": head_dim,
            "hidden_act": getattr(lm_config, "hidden_act", "silu"),
            "moe_intermediate_size": getattr(lm_config, "moe_intermediate_size", 512),
            "shared_expert_intermediate_size": getattr(
                lm_config, "shared_expert_intermediate_size", 512
            ),
            "num_experts": getattr(lm_config, "num_experts", 256),
            "num_experts_per_tok": getattr(lm_config, "num_experts_per_tok", 8),
            "linear_conv_kernel_dim": getattr(lm_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(lm_config, "linear_key_head_dim", head_dim),
            "linear_num_key_heads": getattr(
                lm_config, "linear_num_key_heads", lm_config.num_key_value_heads
            ),
            "linear_num_value_heads": getattr(
                lm_config, "linear_num_value_heads", lm_config.num_attention_heads
            ),
            "linear_value_head_dim": getattr(lm_config, "linear_value_head_dim", head_dim),
            "rms_norm_eps": getattr(lm_config, "rms_norm_eps", 1e-6),
            "attention_bias": getattr(lm_config, "attention_bias", False),
            "tie_word_embeddings": getattr(lm_config, "tie_word_embeddings", False),
            # Keep the synthetic checkpoints small while preserving every
            # candidate's ratio to the canonical teacher geometry.
            "runtime_proxy_max_experts": 16,
            "runtime_proxy_max_intermediate": 256,
            "runtime_proxy_max_shared_intermediate": 256,
            "runtime_proxy_max_vocab": 32768,
        }

    @staticmethod
    def _scale_runtime_proxy_value(value: int, base_value: int, cap: int) -> int:
        if base_value <= cap:
            return int(value)
        return max(
            1,
            min(int(cap), int(round(int(value) * int(cap) / int(base_value)))),
        )

    @classmethod
    def _runtime_proxy_moe(cls, runtime_config, moe: MoEConfig) -> MoEConfig:
        if moe.no_op:
            return moe
        base_experts = int(runtime_config.model_config_value("num_experts", 256))
        base_intermediate = int(runtime_config.model_config_value("moe_intermediate_size", 512))
        base_shared = int(runtime_config.model_config_value("shared_expert_intermediate_size", 512))
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
        shared_intermediate = cls._scale_runtime_proxy_value(
            int(moe.shared_expert_intermediate_size or base_shared),
            base_shared,
            int(runtime_config.model_config_value("runtime_proxy_max_shared_intermediate", 256)),
        )
        top_k = min(
            int(moe.top_k or runtime_config.model_config_value("num_experts_per_tok", 8)),
            num_experts,
        )
        return MoEConfig(
            no_op=False,
            num_experts=num_experts,
            expert_intermediate_size=intermediate,
            shared_expert_intermediate_size=shared_intermediate,
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
        return BlockConfig(
            subblock_configs=(
                AttentionConfig(
                    num_query_heads=runtime_config.num_attention_heads,
                    num_kv_heads=runtime_config.num_key_value_heads,
                    qk_head_dim=runtime_config.model_config_value("head_dim", 256),
                    v_head_dim=runtime_config.model_config_value("head_dim", 256),
                ),
                MoEConfig(
                    num_experts=int(runtime_config.model_config_value("num_experts", 256)),
                    expert_intermediate_size=int(
                        runtime_config.model_config_value("moe_intermediate_size", 512)
                    ),
                    shared_expert_intermediate_size=int(
                        runtime_config.model_config_value("shared_expert_intermediate_size", 512)
                    ),
                    top_k=int(runtime_config.model_config_value("num_experts_per_tok", 8)),
                ),
            )
        )

    @classmethod
    def create_runtime_benchmark_model(cls, runtime_config, block_configs: list[BlockConfig]):
        block_configs = [
            cls._runtime_proxy_block_config(runtime_config, block_config)
            for block_config in block_configs
        ]
        base_experts = int(runtime_config.model_config_value("num_experts", 256))
        base_intermediate = int(runtime_config.model_config_value("moe_intermediate_size", 512))
        base_shared = int(runtime_config.model_config_value("shared_expert_intermediate_size", 512))
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
        proxy_shared = cls._scale_runtime_proxy_value(
            base_shared,
            base_shared,
            int(runtime_config.model_config_value("runtime_proxy_max_shared_intermediate", 256)),
        )
        model_config = Qwen3_5MoeTextConfig(
            max_position_embeddings=(
                runtime_config.prefill_seq_len + runtime_config.generation_seq_len
            ),
            vocab_size=min(
                runtime_config.vocab_size,
                int(runtime_config.model_config_value("runtime_proxy_max_vocab", 32768)),
            ),
            hidden_size=runtime_config.hidden_size,
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            num_hidden_layers=len(block_configs),
            head_dim=runtime_config.model_config_value("head_dim", 256),
            hidden_act=runtime_config.model_config_value("hidden_act", "silu"),
            linear_conv_kernel_dim=runtime_config.model_config_value("linear_conv_kernel_dim", 4),
            linear_key_head_dim=runtime_config.model_config_value("linear_key_head_dim", 128),
            linear_num_key_heads=runtime_config.model_config_value("linear_num_key_heads", 16),
            linear_num_value_heads=runtime_config.model_config_value("linear_num_value_heads", 32),
            linear_value_head_dim=runtime_config.model_config_value("linear_value_head_dim", 128),
            moe_intermediate_size=proxy_intermediate,
            shared_expert_intermediate_size=proxy_shared,
            num_experts=proxy_experts,
            num_experts_per_tok=min(
                int(runtime_config.model_config_value("num_experts_per_tok", 8)),
                proxy_experts,
            ),
            rms_norm_eps=runtime_config.model_config_value("rms_norm_eps", 1e-6),
            attention_bias=runtime_config.model_config_value("attention_bias", False),
            tie_word_embeddings=runtime_config.model_config_value("tie_word_embeddings", False),
        )
        cls.set_block_configs(model_config, block_configs)
        with deci_x_patcher(cls, block_configs):
            model = Qwen3_5MoeForCausalLM(model_config)
        model.config.block_configs = [block.to_dict() for block in block_configs]
        model.config.architectures = ["AnyModel"]
        model.config.base_architecture = "Qwen3_5MoeForCausalLM"
        model.config.anymodel_arch_info = dict(_QWEN3P5_ANYMODEL_ARCH_INFO)
        return model

    @classmethod
    def update_runtime_benchmark_config(cls, config_data: dict[str, Any]) -> None:
        if config_data.get("model_type") == "qwen3_5_moe_text":
            _wrap_qwen3p5_moe_text_runtime_config(config_data)

    @classmethod
    def runtime_vllm_benchmark_args(cls, config: Any) -> list[str]:
        return _Qwen3P5BaseModelDescriptor.runtime_vllm_benchmark_args(config)


@ModelDescriptorFactory.register_decorator("qwen3_5_moe_text")
class Qwen3P5MoeTextModelDescriptor(_Qwen3P5MoeModelDescriptor):
    _IS_VLM = False

    @classmethod
    def anymodel_arch_info(cls) -> dict[str, Any]:
        return dict(_QWEN3P5_ANYMODEL_ARCH_INFO)

    @staticmethod
    def init_rotary_embedding(model, runtime):
        model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=model.config).to(
            device=runtime.device, dtype=runtime.dtype
        )


@ModelDescriptorFactory.register_decorator("qwen3_5_moe")
class Qwen3P5MoeVLModelDescriptor(_Qwen3P5MoeModelDescriptor):
    _IS_VLM = True

    @classmethod
    def runtime_benchmark_export_descriptor(cls) -> Type[ModelDescriptor]:
        return Qwen3P5MoeTextModelDescriptor

    @staticmethod
    def init_rotary_embedding(model, runtime):
        text_config = model.config.text_config
        model.model.language_model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(
            config=text_config
        ).to(device=runtime.device, dtype=runtime.dtype)
        vision_config = getattr(model.config, "vision_config", None)
        if vision_config is not None:
            head_dim = vision_config.hidden_size // vision_config.num_heads
            model.model.visual.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2).to(
                device=runtime.device, dtype=runtime.dtype
            )
