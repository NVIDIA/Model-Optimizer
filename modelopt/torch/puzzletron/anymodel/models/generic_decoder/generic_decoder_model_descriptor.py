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

"""Structural descriptor components shared by dense, MoE, text, and VLM decoders."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ....pruning.embedding_pruning import EmbeddingPruningSpec, TensorAxisRule
from ...capabilities import AxisCapabilities, default_capabilities
from ...model_descriptor import ModelDescriptor
from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

if TYPE_CHECKING:
    from torch import nn

    from ....block_config import BlockConfig


def _read_path(value: Any, path: tuple[str, ...]) -> Any:
    node = value
    for key in path:
        node = node[key] if isinstance(node, dict) else getattr(node, key)
    return node


def _aligned_half(value: int, alignment: int) -> int:
    value = int(value)
    alignment = int(alignment)
    if value < 2:
        raise ValueError(f"cannot derive a reduced value from {value}")
    reduced = value // 2
    if alignment > 1:
        reduced = (reduced // alignment) * alignment
    if reduced < max(1, alignment):
        raise ValueError(f"half of {value} cannot satisfy alignment={alignment}")
    return reduced


def _layer_regex(template: str) -> str:
    marker = re.escape("{layer_idx}")
    escaped = re.escape(template)
    if marker not in escaped:
        raise ValueError("layer_template must contain {layer_idx}")
    return escaped.replace(marker, r"\d+")


@dataclass(frozen=True)
class DecoderLayout:
    language_config_path: tuple[str, ...]
    language_prefix: str
    layer_template: str
    input_embedding: str
    output_embedding: str
    final_norm: str
    layer_norm_names: tuple[str, ...]
    hidden_size_field: str = "hidden_size"

    def language_config(self, config: Any) -> Any:
        return _read_path(config, self.language_config_path)

    def layer_path(self, layer_idx: int) -> str:
        return self.layer_template.format(layer_idx=int(layer_idx))

    @property
    def layer_pattern(self) -> str:
        return _layer_regex(self.layer_template)


@dataclass(frozen=True)
class StandardGQAAttentionContract:
    module_name: str = "self_attn"
    q_proj_name: str = "q_proj"
    k_proj_name: str = "k_proj"
    v_proj_name: str = "v_proj"
    o_proj_name: str = "o_proj"
    query_heads_field: str = "num_attention_heads"
    kv_heads_field: str = "num_key_value_heads"
    head_dim_field: str = "head_dim"
    global_head_dim_field: str | None = None
    global_kv_heads_field: str | None = None
    k_eq_v_field: str | None = None
    shared_kv_layers_field: str | None = None

    def layer_override_fields(self, attention: Any) -> dict[str, int]:
        """Map one typed attention config to constructor fields.

        Families with heterogeneous local/global geometry describe the alternate
        fields above.  The generic descriptor can therefore target the correct
        constructor fields without naming the family or duplicating its rules.
        """

        overrides: dict[str, int] = {}
        query_heads = getattr(attention, "num_query_heads", None)
        if query_heads is not None:
            overrides[self.query_heads_field] = int(query_heads)

        k_eq_v = bool(getattr(attention, "k_eq_v", False))
        kv_field = (
            self.global_kv_heads_field
            if k_eq_v and self.global_kv_heads_field is not None
            else self.kv_heads_field
        )
        kv_heads = getattr(attention, "num_kv_heads", None)
        if kv_heads is not None:
            overrides[kv_field] = int(kv_heads)

        head_dim = getattr(attention, "qk_head_dim", None)
        if head_dim is not None:
            is_full = getattr(attention, "sliding_window_size", None) == "full"
            head_dim_field = (
                self.global_head_dim_field
                if (is_full or k_eq_v) and self.global_head_dim_field is not None
                else self.head_dim_field
            )
            overrides[head_dim_field] = int(head_dim)
        return overrides

    def validate(self, config: Any) -> tuple[int, int, int]:
        query_heads = int(getattr(config, self.query_heads_field))
        kv_heads = int(getattr(config, self.kv_heads_field))
        hidden_size = int(getattr(config, "hidden_size"))
        head_dim = int(getattr(config, self.head_dim_field, 0) or hidden_size // query_heads)
        if min(query_heads, kv_heads, head_dim) < 1:
            raise ValueError("attention head counts and dimensions must be positive")
        if query_heads % kv_heads:
            raise ValueError(
                f"query heads ({query_heads}) must be divisible by KV groups ({kv_heads})"
            )
        return query_heads, kv_heads, head_dim

    def layer_geometry(
        self,
        config: Any,
        *,
        layer_idx: int,
        layer_types: tuple[str, ...],
    ) -> tuple[int, int, int, bool, int | None]:
        """Resolve one layer's true Q/K/V geometry and KV owner.

        Standard models simply return their global GQA fields. Families with
        full/sliding heterogeneity can declare alternate full-attention fields,
        K=V coupling, and trailing KV-sharing layers without forking conversion.
        """

        query_heads, kv_heads, head_dim = self.validate(config)
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
        is_full = layer_type == "full_attention"
        if is_full and self.global_head_dim_field:
            global_head_dim = getattr(config, self.global_head_dim_field, None)
            if global_head_dim is not None:
                head_dim = int(global_head_dim)

        k_eq_v = bool(is_full and self.k_eq_v_field and getattr(config, self.k_eq_v_field, False))
        if k_eq_v and self.global_kv_heads_field:
            global_kv_heads = getattr(config, self.global_kv_heads_field, None)
            if global_kv_heads is not None:
                kv_heads = int(global_kv_heads)

        kv_source_layer = None
        if self.shared_kv_layers_field:
            shared_layers = int(getattr(config, self.shared_kv_layers_field, 0) or 0)
            first_shared = int(getattr(config, "num_hidden_layers")) - shared_layers
            if shared_layers and layer_idx >= first_shared:
                if first_shared <= 0:
                    raise ValueError("KV sharing requires at least one non-shared source layer")
                prior_types = layer_types[:first_shared]
                try:
                    reverse_offset = prior_types[::-1].index(layer_type)
                except ValueError as exc:
                    raise ValueError(
                        f"layer {layer_idx} shares {layer_type!r} KV but has no source layer"
                    ) from exc
                kv_source_layer = first_shared - 1 - reverse_offset

        if query_heads % kv_heads:
            raise ValueError(
                f"layer {layer_idx} query heads ({query_heads}) must be divisible by "
                f"KV groups ({kv_heads})"
            )
        return query_heads, kv_heads, head_dim, k_eq_v, kv_source_layer


@dataclass(frozen=True)
class LatentAttentionContract:
    """Descriptor-owned MLA rank fields shared by conversion and runtime export."""

    module_name: str = "self_attn"
    q_lora_rank_field: str = "q_lora_rank"
    kv_lora_rank_field: str = "kv_lora_rank"


@dataclass(frozen=True)
class GatedDenseFFNContract:
    module_name: str = "mlp"
    gate_proj_name: str = "gate_proj"
    up_proj_name: str = "up_proj"
    down_proj_name: str = "down_proj"
    intermediate_field: str = "intermediate_size"
    double_wide_field: str | None = None
    shared_kv_layers_field: str | None = None

    def layer_intermediate_size(self, config: Any, *, layer_idx: int) -> int:
        intermediate = int(getattr(config, self.intermediate_field))
        if self.double_wide_field and self.shared_kv_layers_field:
            shared_layers = int(getattr(config, self.shared_kv_layers_field, 0) or 0)
            first_shared = int(getattr(config, "num_hidden_layers")) - shared_layers
            if (
                bool(getattr(config, self.double_wide_field, False))
                and first_shared > 0
                and layer_idx >= first_shared
            ):
                intermediate *= 2
        return intermediate

    def constructor_intermediate_size(
        self,
        config: Any,
        *,
        layer_idx: int,
        actual_size: int,
    ) -> int:
        """Invert constructor-only expansion for a requested runtime width."""

        actual_size = int(actual_size)
        if self.double_wide_field and self.shared_kv_layers_field:
            shared_layers = int(getattr(config, self.shared_kv_layers_field, 0) or 0)
            first_shared = int(getattr(config, "num_hidden_layers")) - shared_layers
            if (
                bool(getattr(config, self.double_wide_field, False))
                and first_shared > 0
                and layer_idx >= first_shared
            ):
                if actual_size % 2:
                    raise ValueError(
                        f"layer {layer_idx} runtime FFN width {actual_size} cannot be "
                        "represented by a double-wide constructor"
                    )
                return actual_size // 2
        return actual_size


@dataclass(frozen=True)
class RoutedMoEContract:
    module_name: str = "mlp"
    experts_name: str = "experts"
    router_name: str = "gate"
    shared_expert_name: str | None = "shared_expert"
    gate_proj_name: str = "gate_proj"
    up_proj_name: str = "up_proj"
    down_proj_name: str = "down_proj"
    num_experts_field: str = "num_experts"
    intermediate_field: str = "moe_intermediate_size"
    shared_intermediate_field: str = "shared_expert_intermediate_size"
    top_k_field: str = "num_experts_per_tok"
    replaces_dense_ffn: bool = True


@dataclass(frozen=True)
class VisionLanguageContract:
    module_names: tuple[str, ...]
    projector_rules: tuple[TensorAxisRule, ...]
    projector_output_config_paths: tuple[tuple[str, ...], ...] = ()
    exempt_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class MTPContract:
    tensor_rules: tuple[TensorAxisRule, ...]
    exempt_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PLEContract:
    """Per-layer embedding channel geometry shared by Gemma-style decoders."""

    width_field: str = "hidden_size_per_layer_input"
    layer_gate_name: str = "per_layer_input_gate"
    layer_projection_name: str = "per_layer_projection"
    model_embedding_name: str = "embed_tokens_per_layer"
    model_projection_name: str = "per_layer_model_projection"
    model_norm_name: str = "per_layer_projection_norm"


@dataclass(frozen=True)
class GenericDecoderContract:
    """A model-name-independent contract assembled from decoder components."""

    descriptor_name: str
    model_family: str
    layout: DecoderLayout
    attention: StandardGQAAttentionContract | None = None
    latent_attention: LatentAttentionContract | None = None
    dense_ffn: GatedDenseFFNContract | None = None
    routed_moe: RoutedMoEContract | None = None
    vision: VisionLanguageContract | None = None
    mtp: MTPContract | None = None
    ple: PLEContract | None = None
    additional_tensor_rules: tuple[TensorAxisRule, ...] = ()
    additional_exempt_patterns: tuple[str, ...] = ()
    # Some checkpoints were serialized by an older Transformers layout even
    # though their current config resolves to a newer runtime module tree.
    # Conversion owns this one-time migration; all later Puzzletron stages use
    # only the canonical runtime names declared by ``layout``.
    checkpoint_key_rewrites: tuple[tuple[str, str], ...] = ()
    native_automodel_supported: bool = True
    ep_supported: bool = False
    sequence_parallel_supported: bool = False
    hidden_permutation_group_size: int = 1
    explicit_full_attention_window: bool = False

    def __post_init__(self) -> None:
        if self.dense_ffn is None and self.routed_moe is None:
            raise ValueError("a generic decoder contract requires a dense FFN or routed MoE")
        if self.ep_supported and self.routed_moe is None:
            raise ValueError("EP support requires a routed MoE contract")

    def language_config(self, config: Any) -> Any:
        return self.layout.language_config(config)

    def _base_hidden_rules(self) -> list[TensorAxisRule]:
        layout = self.layout
        layer = layout.layer_pattern
        rules = [
            TensorAxisRule(
                rf"^{re.escape(layout.input_embedding)}\.weight$",
                (1,),
                "token embedding channels",
            ),
            TensorAxisRule(
                rf"^{re.escape(layout.output_embedding)}\.weight$",
                (1,),
                "language head input channels",
            ),
            TensorAxisRule(
                rf"^{re.escape(layout.final_norm)}\.weight$",
                (0,),
                "final residual normalization channels",
            ),
        ]
        if layout.layer_norm_names:
            norm_names = "|".join(re.escape(name) for name in layout.layer_norm_names)
            rules.append(
                TensorAxisRule(
                    rf"^{layer}\.(?:{norm_names})\.weight$",
                    (0,),
                    "decoder residual normalization channels",
                )
            )
        return rules

    def _attention_hidden_rules(self) -> list[TensorAxisRule]:
        if self.attention is None:
            return []
        layer = self.layout.layer_pattern
        attention = self.attention
        module = re.escape(attention.module_name)
        inputs = "|".join(
            re.escape(name)
            for name in (attention.q_proj_name, attention.k_proj_name, attention.v_proj_name)
        )
        return [
            TensorAxisRule(
                rf"^{layer}\.{module}\.(?:{inputs})\.weight$",
                (1,),
                "attention residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.{module}\.{re.escape(attention.o_proj_name)}\.(?:weight|bias)$",
                (0,),
                "attention residual output channels",
            ),
        ]

    def _dense_hidden_rules(self) -> list[TensorAxisRule]:
        if self.dense_ffn is None:
            return []
        layer = self.layout.layer_pattern
        ffn = self.dense_ffn
        module = re.escape(ffn.module_name)
        inputs = "|".join(re.escape(name) for name in (ffn.gate_proj_name, ffn.up_proj_name))
        return [
            TensorAxisRule(
                rf"^{layer}\.{module}\.(?:{inputs})\.weight$",
                (1,),
                "FFN residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.{module}\.{re.escape(ffn.down_proj_name)}\.(?:weight|bias)$",
                (0,),
                "FFN residual output channels",
            ),
        ]

    def _moe_hidden_rules(self) -> list[TensorAxisRule]:
        if self.routed_moe is None:
            return []
        layer = self.layout.layer_pattern
        moe = self.routed_moe
        module = re.escape(moe.module_name)
        module_prefix = rf"{module}\." if module else ""
        inputs = "|".join(re.escape(name) for name in (moe.gate_proj_name, moe.up_proj_name))
        expert_root = rf"{module_prefix}{re.escape(moe.experts_name)}\.\d+"
        roots = [expert_root]
        if moe.shared_expert_name:
            roots.append(rf"{module_prefix}{re.escape(moe.shared_expert_name)}")
        root = "|".join(roots)
        return [
            TensorAxisRule(
                rf"^{layer}\.{module_prefix}{re.escape(moe.router_name)}\.weight$",
                (1,),
                "MoE router residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.(?:{root})\.(?:{inputs})\.weight$",
                (1,),
                "MoE expert residual input channels",
            ),
            TensorAxisRule(
                rf"^{layer}\.(?:{root})\.{re.escape(moe.down_proj_name)}\.(?:weight|bias)$",
                (0,),
                "MoE expert residual output channels",
            ),
        ]

    def embedding_pruning_spec(
        self,
        config: Any,
        *,
        widths: tuple[int, ...] | list[int],
        alignment: int,
    ) -> EmbeddingPruningSpec:
        lm_config = self.language_config(config)
        hidden_size = int(getattr(lm_config, self.layout.hidden_size_field))
        rules = (
            self._base_hidden_rules()
            + self._attention_hidden_rules()
            + self._dense_hidden_rules()
            + self._moe_hidden_rules()
        )
        exempt = [r"(?:^|\.)rotary_emb\.", r"(?:^|\.)inv_freq$"]
        if self.vision is not None:
            rules.extend(self.vision.projector_rules)
            exempt.extend(rf"^{re.escape(name)}\." for name in self.vision.module_names)
            exempt.extend(self.vision.exempt_patterns)
        if self.mtp is not None:
            rules.extend(self.mtp.tensor_rules)
            exempt.extend(self.mtp.exempt_patterns)
        rules.extend(self.additional_tensor_rules)
        exempt.extend(self.additional_exempt_patterns)

        ties: tuple[tuple[str, ...], ...] = ()
        if bool(getattr(lm_config, "tie_word_embeddings", False)):
            ties = (
                (f"{self.layout.input_embedding}.weight", f"{self.layout.output_embedding}.weight"),
            )
        hidden_path = (*self.layout.language_config_path, self.layout.hidden_size_field)
        config_paths = (hidden_path,)
        if self.vision is not None:
            config_paths += self.vision.projector_output_config_paths
        norm_names = "|".join(re.escape(name) for name in self.layout.layer_norm_names)
        residual_norms = (
            (rf"^{self.layout.layer_pattern}\.(?:{norm_names})$",) if norm_names else ()
        )
        return EmbeddingPruningSpec(
            hidden_size=hidden_size,
            legal_widths=tuple(int(width) for width in widths),
            alignment=int(alignment),
            tensor_rules=tuple(rules),
            exempt_patterns=tuple(exempt),
            tie_groups=ties,
            config_paths=config_paths,
            residual_norm_patterns=residual_norms,
            permutation_group_size=int(self.hidden_permutation_group_size),
        )

    def reduced_axis_values(self, config: Any, *, alignment: int) -> dict[str, int]:
        lm_config = self.language_config(config)
        hidden_size = int(getattr(lm_config, self.layout.hidden_size_field))
        values = {"hidden_width": _aligned_half(hidden_size, alignment)}
        if self.attention is not None:
            query_heads, kv_heads, _ = self.attention.validate(lm_config)
            values["kv_groups"] = _aligned_half(kv_heads, 1)
            values["q_heads_per_group"] = _aligned_half(query_heads // kv_heads, 1)
        if self.dense_ffn is not None:
            intermediate = int(getattr(lm_config, self.dense_ffn.intermediate_field))
            values["ffn_intermediate"] = _aligned_half(intermediate, alignment)
        if self.routed_moe is not None:
            moe = self.routed_moe
            values["moe_experts"] = _aligned_half(int(getattr(lm_config, moe.num_experts_field)), 1)
            values["moe_expert_intermediate"] = _aligned_half(
                int(getattr(lm_config, moe.intermediate_field)), alignment
            )
            shared = getattr(lm_config, moe.shared_intermediate_field, None)
            if moe.shared_expert_name and shared is not None:
                values["moe_shared_expert_intermediate"] = _aligned_half(int(shared), alignment)
        if self.ple is not None:
            ple_width = int(getattr(lm_config, self.ple.width_field, 0) or 0)
            if ple_width:
                values["ple_width"] = _aligned_half(ple_width, alignment)
        return values

    def discover_prunable_modules(self, model: nn.Module) -> dict[str, nn.Module]:
        layer_pattern = re.compile(rf"^{self.layout.layer_pattern}$")
        targets: dict[str, nn.Module] = {}
        for layer_path, layer in model.named_modules():
            if not layer_pattern.match(layer_path):
                continue
            if self.attention is not None:
                targets[f"{layer_path}.attention"] = layer.get_submodule(self.attention.module_name)
            if self.dense_ffn is not None:
                targets[f"{layer_path}.ffn"] = layer.get_submodule(self.dense_ffn.module_name)
            if self.routed_moe is not None:
                targets[f"{layer_path}.moe"] = (
                    layer.get_submodule(self.routed_moe.module_name)
                    if self.routed_moe.module_name
                    else layer
                )
        return targets

    def capabilities(self, config: Any | None = None):
        base = default_capabilities(
            descriptor_name=self.descriptor_name,
            model_family=self.model_family,
            native_automodel_supported=self.native_automodel_supported,
        )
        axes: dict[str, AxisCapabilities] = {
            "hidden_width": AxisCapabilities(
                axis_id="hidden_width",
                subblock_kind="model",
                field="hidden_size",
                score_hooks=("minitron_hidden_width",),
                sort_impl="sorted_teacher.embedding",
                materialize_impl="materialize.hidden_width",
                runtime_slice_impl="runtime_hidden_width",
                vllm_export=True,
            )
        }
        lm_config = None
        if config is not None:
            try:
                lm_config = self.language_config(config)
            except (AttributeError, KeyError, TypeError):
                # Registry inference can operate on architecture-only metadata.
                # Config-dependent axes are added once the nested language
                # config is available; structural base capabilities remain valid.
                lm_config = None
        if self.attention is not None:
            axes["kv_groups"] = base.axes["kv_groups"]
            axes["q_heads_per_group"] = base.axes["q_heads_per_group"]
            if lm_config is not None:
                if hasattr(lm_config, "sliding_window"):
                    window = int(getattr(lm_config, "sliding_window", 0) or 0)
                    values = tuple(
                        value
                        for value in (
                            window // 2 if window > 1 else None,
                            window or None,
                            "full",
                        )
                        if value is not None
                    )
                    axes["sliding_window_size"] = AxisCapabilities(
                        axis_id="sliding_window_size",
                        subblock_kind="attention",
                        field="sliding_window_size",
                        sortable=False,
                        variant_only=True,
                        vllm_export=True,
                        values=values,
                        constraints=("per_layer_attention_window",),
                    )
        if self.dense_ffn is not None:
            axes["ffn_intermediate"] = base.axes["ffn_intermediate"]
        if self.routed_moe is not None:
            for name in (
                "moe_experts",
                "moe_expert_intermediate",
                "moe_shared_expert_intermediate",
                "moe_top_k",
            ):
                axes[name] = base.axes[name]
        if self.ple is not None and lm_config is not None:
            ple_width = int(getattr(lm_config, self.ple.width_field, 0) or 0)
            if ple_width:
                axes["ple_width"] = AxisCapabilities(
                    axis_id="ple_width",
                    subblock_kind="model",
                    field=self.ple.width_field,
                    score_hooks=("ple_channel_contribution",),
                    sort_impl="sorted_teacher.ple",
                    materialize_impl="materialize.ple",
                    runtime_slice_impl="runtime_ple",
                    vllm_export=True,
                    native_automodel_required=True,
                    constraints=("global_equal_width", "packed_layer_chunks"),
                )
        subblock_names = {axis.subblock_kind for axis in axes.values()}
        subblocks = {
            name: value for name, value in base.subblocks.items() if name in subblock_names
        }
        return dataclasses.replace(
            base,
            axes=axes,
            subblocks=subblocks,
            parallelism=dataclasses.replace(
                base.parallelism,
                ep=self.ep_supported,
                sequence_parallel=self.sequence_parallel_supported,
            ),
        )


class GenericContractModelDescriptor(ModelDescriptor):
    """ModelDescriptor adapter whose behavior is derived from a structural contract."""

    DECODER_LAYER_CLS: type[nn.Module] | tuple[type[nn.Module], ...] | None = None

    @classmethod
    def generic_decoder_contract(cls, config) -> GenericDecoderContract:
        raise NotImplementedError

    @classmethod
    def _layout(cls) -> DecoderLayout:
        return cls.generic_decoder_contract(None).layout

    @classmethod
    def decoder_layer_cls(cls):
        if cls.DECODER_LAYER_CLS is None:
            raise RuntimeError(f"{cls.__name__} did not declare DECODER_LAYER_CLS")
        if isinstance(cls.DECODER_LAYER_CLS, tuple):
            return list(cls.DECODER_LAYER_CLS)
        return cls.DECODER_LAYER_CLS

    @classmethod
    def get_language_model_config(cls, config):
        return cls.generic_decoder_contract(config).language_config(config)

    @classmethod
    def puzzletron_capabilities(cls, config):
        return cls.generic_decoder_contract(config).capabilities(config)

    @classmethod
    def embedding_pruning_spec(cls, config, *, widths, alignment: int):
        return cls.generic_decoder_contract(config).embedding_pruning_spec(
            config, widths=widths, alignment=alignment
        )

    @classmethod
    def ple_pruning_spec(cls, config):
        from ....pruning.ple_pruning import PLEPruningSpec

        contract = cls.generic_decoder_contract(config)
        if contract.ple is None:
            return None
        lm_config = contract.language_config(config)
        width = int(getattr(lm_config, contract.ple.width_field, 0) or 0)
        if width <= 0:
            return None
        return PLEPruningSpec(
            language_prefix=contract.layout.language_prefix,
            layer_template=contract.layout.layer_template,
            num_layers=int(getattr(lm_config, "num_hidden_layers")),
            width=width,
            layer_gate_name=contract.ple.layer_gate_name,
            layer_projection_name=contract.ple.layer_projection_name,
            model_embedding_name=contract.ple.model_embedding_name,
            model_projection_name=contract.ple.model_projection_name,
            model_norm_name=contract.ple.model_norm_name,
        )

    @classmethod
    def input_embedding_name(cls):
        return cls._layout().input_embedding

    @classmethod
    def output_embedding_name(cls):
        return cls._layout().output_embedding

    @classmethod
    def final_norm_name(cls):
        return cls._layout().final_norm

    @classmethod
    def layer_block_name(cls, index: int):
        return cls._layout().layer_path(index)

    @classmethod
    def local_kd_subblock_module_paths(
        cls, block_config: BlockConfig, *, layer_idx: int
    ) -> dict[tuple[str, str], str]:
        del layer_idx
        contract = cls.generic_decoder_contract(None)
        module_by_kind = {}
        if contract.attention is not None:
            module_by_kind["attention"] = contract.attention.module_name
        if contract.dense_ffn is not None:
            module_by_kind["ffn"] = contract.dense_ffn.module_name
        if contract.routed_moe is not None:
            module_by_kind["moe"] = contract.routed_moe.module_name
        paths = {}
        for subblock in block_config.subblock_configs:
            module_path = module_by_kind.get(subblock.kind)
            if not module_path:
                raise NotImplementedError(
                    f"{cls.__name__} has no local-KD module boundary for "
                    f"subblock kind {subblock.kind!r}"
                )
            paths[(subblock.kind, subblock.name)] = module_path
        return paths

    @classmethod
    def vision_module_names(cls) -> tuple[str, ...]:
        vision = cls.generic_decoder_contract(None).vision
        return vision.module_names if vision is not None else ()

    @classmethod
    def block_config_to_layer_overrides(cls, block_config: BlockConfig):
        contract = cls.generic_decoder_contract(None)
        overrides: dict[str, Any] = {}
        attention = block_config.get_subblock("attention")
        if attention is not None and contract.attention is not None:
            overrides.update(contract.attention.layer_override_fields(attention))
        ffn = block_config.get_subblock("ffn")
        if ffn is not None and ffn.intermediate_size is not None and contract.dense_ffn is not None:
            overrides[contract.dense_ffn.intermediate_field] = ffn.intermediate_size
        moe = block_config.get_subblock("moe")
        if moe is not None and contract.routed_moe is not None:
            moe_contract = contract.routed_moe
            if moe.num_experts is not None:
                overrides[moe_contract.num_experts_field] = moe.num_experts
            if moe.expert_intermediate_size is not None:
                overrides[moe_contract.intermediate_field] = moe.expert_intermediate_size
            if (
                moe.shared_expert_intermediate_size is not None
                and moe_contract.shared_expert_name is not None
            ):
                overrides[moe_contract.shared_intermediate_field] = (
                    moe.shared_expert_intermediate_size
                )
            if moe.top_k is not None:
                overrides[moe_contract.top_k_field] = moe.top_k
        return overrides

    @classmethod
    def patch_layer_config(
        cls,
        layer_config: Any,
        block_config: BlockConfig,
        layer_idx: int,
    ) -> None:
        super().patch_layer_config(layer_config, block_config, layer_idx)
        contract = cls.generic_decoder_contract(None)
        ffn = block_config.get_subblock("ffn")
        if ffn is None or ffn.intermediate_size is None or contract.dense_ffn is None:
            return
        field = contract.dense_ffn.intermediate_field
        setattr(
            layer_config,
            field,
            contract.dense_ffn.constructor_intermediate_size(
                layer_config,
                layer_idx=layer_idx,
                actual_size=int(ffn.intermediate_size),
            ),
        )

    @staticmethod
    def attn_no_op_post_init(decoder_layer: nn.Module):
        if hasattr(decoder_layer, "input_layernorm"):
            decoder_layer.input_layernorm = Same()
        decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: nn.Module):
        if hasattr(decoder_layer, "post_attention_layernorm"):
            decoder_layer.post_attention_layernorm = Same()
        decoder_layer.mlp = MatchingZeros()

    @staticmethod
    def init_rotary_embedding(model, runtime):
        # Most current Transformers/AutoModel families register RoPE buffers during
        # model construction. Families that require explicit meta-device repair override this.
        return None

    @classmethod
    def layer_name_predicates(cls, num_layers: int):
        layout = cls._layout()
        vision = cls.generic_decoder_contract(None).vision
        embedding_parts = [rf"{re.escape(layout.input_embedding)}\.weight"]
        if vision is not None:
            embedding_parts.extend(rf"{re.escape(name)}\..+" for name in vision.module_names)
        patterns: dict[str, re.Pattern] = {
            "embeddings": re.compile(rf"^(?:{'|'.join(embedding_parts)})$"),
            "lm_head": re.compile(
                rf"^(?:{re.escape(layout.final_norm)}\.weight|"
                rf"{re.escape(layout.output_embedding)}\.weight)$"
            ),
        }
        for layer_idx in range(num_layers):
            layer = re.escape(layout.layer_path(layer_idx))
            patterns[f"block_{layer_idx}_ffn"] = re.compile(
                rf"^{layer}\.(?:mlp|experts|router|post_attention_layernorm|"
                r"pre_feedforward_layernorm|post_feedforward_layernorm).*$"
            )
            patterns[f"block_{layer_idx}_attention"] = re.compile(rf"^{layer}\..+$")
        return patterns

    @staticmethod
    def passthrough_weight_name_predicates():
        return {"mtp": re.compile(r"^mtp(?:\.|$).*")}
