# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative, dependency-free model capabilities for Puzzletron setup."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from . import SetupError

__all__ = [
    "AxisInventory",
    "AxisSpec",
    "CandidateCounts",
    "ModelInventory",
    "ModelProfile",
    "UnsupportedModelError",
    "count_candidate_options",
    "resolve_profile",
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(config: Mapping[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _nested(config, path)
        if value is not None:
            return value
    return None


def _language_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("text_config")) or config


def _model_types(config: Mapping[str, Any]) -> tuple[str, ...]:
    result = []
    for candidate in (config, _language_config(config)):
        model_type = candidate.get("model_type")
        if model_type and str(model_type) not in result:
            result.append(str(model_type))
    return tuple(result)


def _architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    result = []
    for candidate in (config, _language_config(config)):
        architectures = candidate.get("architectures") or ()
        if isinstance(architectures, str):
            architectures = (architectures,)
        for architecture in architectures:
            if str(architecture) not in result:
                result.append(str(architecture))
    return tuple(result)


@dataclass(frozen=True)
class AxisSpec:
    """One supported pruning axis and its config-field aliases."""

    axis_id: str
    label: str
    fields: tuple[str, ...]
    alignment: int = 1
    minimum: int = 1

    def teacher_value(self, config: Mapping[str, Any]) -> int | None:
        """Resolve the teacher value, including derived grouped-head axes."""
        language = _language_config(config)
        if self.axis_id == "q_heads_per_group":
            query_heads = _first(language, ("num_attention_heads", "n_head"))
            kv_groups = _first(language, ("num_key_value_heads", "n_head_kv"))
            if query_heads is None or kv_groups in (None, 0):
                return None
            return int(query_heads) // int(kv_groups)
        if self.axis_id == "gdn_value_heads_per_group":
            value_heads = _first(
                language,
                ("linear_num_value_heads", "linear_attention.num_value_heads"),
            )
            key_groups = _first(
                language,
                ("linear_num_key_heads", "linear_attention.num_key_heads"),
            )
            if value_heads is None or key_groups in (None, 0):
                return None
            return int(value_heads) // int(key_groups)
        value = _first(language, self.fields)
        return int(value) if value is not None else None

    def options(self, teacher: int, limit: int = 16) -> tuple[int, ...]:
        """Return a compact descending domain containing teacher and half size."""
        if teacher < self.minimum:
            return (teacher,)
        legal = list(range(teacher, self.minimum - 1, -self.alignment))
        if not legal:
            return (teacher,)
        if len(legal) <= limit:
            return tuple(legal)
        half = min(legal, key=lambda value: (abs(value - teacher / 2), -value))
        required = {teacher, half, legal[-1]}
        sampled = {legal[round(index * (len(legal) - 1) / (limit - 1))] for index in range(limit)}
        values = sorted(required | sampled, reverse=True)
        while len(values) > limit:
            removable = [value for value in values if value not in required]
            values.remove(removable[len(removable) // 2])
        return tuple(values)


@dataclass(frozen=True)
class AxisInventory:
    """Resolved teacher geometry and legal wizard choices for one axis."""

    axis_id: str
    label: str
    teacher_value: int
    values: tuple[int, ...]
    alignment: int


@dataclass(frozen=True)
class ModelInventory:
    """Normalized model facts consumed by the setup questions and renderer."""

    family: str
    descriptor: str
    family_config: str
    model_type: str
    architectures: tuple[str, ...]
    multimodal: bool
    moe: bool
    num_layers: int
    num_sublayers: int
    layer_counts: Mapping[str, int]
    facts: Mapping[str, Any]
    axes: tuple[AxisInventory, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the inventory to YAML-safe built-in values."""
        return asdict(self)


@dataclass(frozen=True)
class CandidateCounts:
    """Exact vLLM configuration and replace-one solution counts."""

    vllm_subblock: int
    vllm_block: int
    replacement_subblock_per_width: int
    replacement_block_per_width: int
    width_count: int

    @property
    def replacement_subblock_total(self) -> int:
        """Return subblock replacement solutions across all selected widths."""
        return self.replacement_subblock_per_width * self.width_count

    @property
    def replacement_block_total(self) -> int:
        """Return block replacement solutions across all selected widths."""
        return self.replacement_block_per_width * self.width_count


class UnsupportedModelError(SetupError):
    """The inspected config does not match an intentionally supported profile."""

    def __init__(self, config: Mapping[str, Any]):
        """Build an onboarding message from detected config identity fields."""
        model_types = _model_types(config) or ("unknown",)
        architectures = _architectures(config) or ("unknown",)
        message = (
            "Unsupported model family. Detected model types "
            f"{list(model_types)} and architectures {list(architectures)}. "
            "Use .agents/skills/running-puzzletron/SKILL.md to add the runtime "
            "descriptor and a lightweight puzzletron_setup profile for this family."
        )
        super().__init__(message)
        self.model_types = model_types
        self.architectures = architectures


@dataclass(frozen=True)
class ModelProfile:
    """Recognition and normalization rules for one supported model family."""

    family: str
    descriptor: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    family_config: str
    axes: tuple[AxisSpec, ...]
    moe: bool = False
    sublayers_per_layer: int = 1
    descriptor_by_model_type: Mapping[str, str] | None = None
    descriptor_by_architecture: Mapping[str, str] | None = None

    def matches(self, config: Mapping[str, Any]) -> bool:
        """Return whether exact model-type or architecture metadata matches."""
        return bool(
            set(_model_types(config)) & set(self.model_types)
            or set(_architectures(config)) & set(self.architectures)
        )

    def _descriptor(self, config: Mapping[str, Any]) -> str:
        aliases = self.descriptor_by_model_type or {}
        for model_type in _model_types(config):
            if model_type in aliases:
                return aliases[model_type]
        aliases = self.descriptor_by_architecture or {}
        for architecture in _architectures(config):
            if architecture in aliases:
                return aliases[architecture]
        return self.descriptor

    def inventory(self, config: Mapping[str, Any]) -> ModelInventory:
        """Normalize one supported Hugging Face config dictionary."""
        language = _language_config(config)
        num_layers = int(_first(language, ("num_hidden_layers", "n_layer", "num_layers")) or 0)
        if num_layers <= 0:
            raise SetupError("The model config does not declare a positive layer count.")
        layer_types = _first(
            language,
            ("layers_block_type", "layer_types", "hybrid_override_pattern"),
        )
        if isinstance(layer_types, str):
            layer_counts = Counter(
                {
                    "attention": layer_types.count("*"),
                    "mamba": layer_types.count("M"),
                    "moe": layer_types.count("E"),
                    "ffn": layer_types.count("-"),
                }
            )
            layer_counts = Counter({key: value for key, value in layer_counts.items() if value})
        elif isinstance(layer_types, list):
            layer_counts = Counter(str(item) for item in layer_types)
        else:
            layer_counts = Counter({"decoder": num_layers})

        multimodal = bool(
            config.get("vision_config")
            or config.get("visual")
            or (
                any("ConditionalGeneration" in item for item in _architectures(config))
                and not all(item.endswith("_text") for item in _model_types(config))
            )
        )
        axes = []
        for spec in self.axes:
            teacher = spec.teacher_value(config)
            if teacher is None or teacher <= 0:
                continue
            axes.append(
                AxisInventory(
                    axis_id=spec.axis_id,
                    label=spec.label,
                    teacher_value=teacher,
                    values=spec.options(teacher),
                    alignment=spec.alignment,
                )
            )

        fact_fields = {
            "hidden_size": ("hidden_size", "d_model"),
            "intermediate_size": ("intermediate_size", "ffn_hidden_size"),
            "num_attention_heads": ("num_attention_heads", "n_head"),
            "num_key_value_heads": ("num_key_value_heads", "n_head_kv"),
            "head_dim": ("head_dim", "attention_head_dim"),
            "vocab_size": ("vocab_size",),
            "num_experts": ("n_routed_experts", "num_experts", "num_local_experts"),
            "num_experts_per_tok": ("num_experts_per_tok", "num_selected_experts"),
        }
        facts = {
            name: value
            for name, paths in fact_fields.items()
            if (value := _first(language, paths)) is not None
        }
        facts["tie_word_embeddings"] = bool(language.get("tie_word_embeddings", False))
        model_types = _model_types(config)
        return ModelInventory(
            family=self.family,
            descriptor=self._descriptor(config),
            family_config=self.family_config,
            model_type=model_types[0] if model_types else "unknown",
            architectures=_architectures(config),
            multimodal=multimodal,
            moe=self.moe,
            num_layers=num_layers,
            num_sublayers=num_layers * self.sublayers_per_layer,
            layer_counts=dict(layer_counts),
            facts=facts,
            axes=tuple(axes),
        )


_ATTENTION_AXES = (
    AxisSpec("hidden_width", "Residual/embedding width", ("hidden_size",), 256, 256),
    AxisSpec("kv_groups", "KV groups", ("num_key_value_heads", "n_head_kv")),
    AxisSpec("q_heads_per_group", "Query heads per KV group", ()),
)

_QWEN_AXES = (
    *_ATTENTION_AXES,
    AxisSpec("ffn_intermediate", "FFN intermediate width", ("intermediate_size",), 256, 256),
    AxisSpec(
        "gdn_key_groups",
        "Gated-delta key groups",
        ("linear_num_key_heads", "linear_attention.num_key_heads"),
        1,
        1,
    ),
    AxisSpec("gdn_value_heads_per_group", "Gated-delta value heads per group", ()),
    AxisSpec(
        "gdn_key_head_dim",
        "Gated-delta key head dimension",
        ("linear_key_head_dim", "linear_attention.key_head_dim"),
        32,
        32,
    ),
    AxisSpec(
        "gdn_value_head_dim",
        "Gated-delta value head dimension",
        ("linear_value_head_dim", "linear_attention.value_head_dim"),
        32,
        32,
    ),
)

_MOE_AXES = (
    AxisSpec(
        "moe_experts",
        "Routed experts",
        ("n_routed_experts", "num_experts", "num_local_experts"),
        16,
        16,
    ),
    AxisSpec(
        "moe_expert_intermediate",
        "Expert intermediate width",
        ("moe_intermediate_size", "expert_intermediate_size"),
        256,
        256,
    ),
    AxisSpec(
        "moe_shared_expert_intermediate",
        "Shared expert intermediate width",
        ("moe_shared_expert_intermediate_size", "shared_expert_intermediate_size"),
        256,
        256,
    ),
    AxisSpec("moe_top_k", "Active experts per token", ("num_experts_per_tok",), 1, 1),
    AxisSpec("moe_latent_dim", "MoE latent width", ("moe_latent_size",), 128, 128),
)

_NEMOTRON_AXES = (
    AxisSpec("hidden_width", "Residual/embedding width", ("hidden_size",), 128, 128),
    AxisSpec("kv_groups", "KV groups", ("num_key_value_heads", "n_head_kv")),
    AxisSpec("q_heads_per_group", "Query heads per KV group", ()),
    *_MOE_AXES,
    AxisSpec("mamba_heads", "Mamba heads", ("mamba_num_heads",), 8, 8),
    AxisSpec("mamba_head_dim", "Mamba head dimension", ("mamba_head_dim",), 8, 8),
)

_SUBBLOCK_AXES = {
    "attention": ("kv_groups", "q_heads_per_group"),
    "gdn": (
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ),
    "ffn": ("ffn_intermediate",),
    "moe": (
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
        "moe_top_k",
        "moe_latent_dim",
    ),
    "mamba": ("mamba_heads", "mamba_head_dim"),
}

SUPPORTED_PROFILES = (
    ModelProfile(
        family="nemotron3",
        descriptor="nemotron_h",
        model_types=("nemotron_h", "nemotron_h_v2"),
        architectures=("NemotronHForCausalLM", "NemotronHV2ForCausalLM"),
        family_config="examples/puzzletron/configs/families/nemotron3/family.yaml",
        axes=_NEMOTRON_AXES,
        moe=True,
        descriptor_by_model_type={"nemotron_h_v2": "nemotron_h_v2"},
        descriptor_by_architecture={
            "NemotronHForCausalLM": "nemotron_h",
            "NemotronHV2ForCausalLM": "nemotron_h_v2",
        },
    ),
    ModelProfile(
        family="qwen3_5",
        descriptor="qwen3_5_moe",
        model_types=(
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_6_moe",
            "qwen3_6_moe_text",
        ),
        architectures=(
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_6MoeForConditionalGeneration",
            "Qwen3_6MoeForCausalLM",
        ),
        family_config="examples/puzzletron/configs/families/qwen3_5/family.yaml",
        axes=(*_QWEN_AXES, *_MOE_AXES),
        moe=True,
        sublayers_per_layer=2,
        descriptor_by_model_type={
            "qwen3_5_moe": "qwen3_5_moe",
            "qwen3_5_moe_text": "qwen3_5_moe_text",
            "qwen3_6_moe": "qwen3_5_moe",
            "qwen3_6_moe_text": "qwen3_5_moe_text",
        },
        descriptor_by_architecture={
            "Qwen3_5MoeForConditionalGeneration": "qwen3_5_moe",
            "Qwen3_5MoeForCausalLM": "qwen3_5_moe_text",
            "Qwen3_6MoeForConditionalGeneration": "qwen3_5_moe",
            "Qwen3_6MoeForCausalLM": "qwen3_5_moe_text",
        },
    ),
    ModelProfile(
        family="qwen3_5",
        descriptor="qwen3_5",
        model_types=("qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_6_text"),
        architectures=(
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5ForCausalLM",
            "Qwen3_6ForConditionalGeneration",
            "Qwen3_6ForCausalLM",
        ),
        family_config="examples/puzzletron/configs/families/qwen3_5/family.yaml",
        axes=_QWEN_AXES,
        sublayers_per_layer=2,
        descriptor_by_model_type={
            "qwen3_5": "qwen3_5",
            "qwen3_5_text": "qwen3_5_text",
            "qwen3_6": "qwen3_6",
            "qwen3_6_text": "qwen3_6_text",
        },
        descriptor_by_architecture={
            "Qwen3_5ForConditionalGeneration": "qwen3_5",
            "Qwen3_5ForCausalLM": "qwen3_5_text",
            "Qwen3_6ForConditionalGeneration": "qwen3_6",
            "Qwen3_6ForCausalLM": "qwen3_6_text",
        },
    ),
)


def _axis_domain_size(axes: Mapping[str, Any], axis_id: str) -> int:
    axis = _mapping(axes.get(axis_id))
    if not axis or not bool(axis.get("enabled", True)):
        return 1
    values = tuple(dict.fromkeys(int(value) for value in axis.get("values") or ()))
    return len(values) or 1


def _qwen_block_families(
    config: Mapping[str, Any], *, moe: bool
) -> tuple[tuple[str, ...], ...]:
    language = _language_config(config)
    layer_types = tuple(str(value) for value in language.get("layer_types") or ())
    num_layers = int(language.get("num_hidden_layers", 0))
    if not layer_types:
        interval = int(language.get("full_attention_interval", 0))
        if interval <= 0 or num_layers <= 0:
            raise SetupError(
                "Qwen candidate counting requires layer_types or full_attention_interval."
            )
        layer_types = tuple(
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(num_layers)
        )
    if len(layer_types) != num_layers:
        raise SetupError("Qwen layer_types does not match num_hidden_layers.")
    feed_forward = "moe" if moe else "ffn"
    mapping = {
        "full_attention": ("attention", feed_forward),
        "linear_attention": ("gdn", feed_forward),
    }
    try:
        return tuple(mapping[layer_type] for layer_type in layer_types)
    except KeyError as error:
        raise SetupError(f"Unsupported Qwen layer type: {error.args[0]}") from error


def _nemotron_block_families(config: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    language = _language_config(config)
    pattern = str(language.get("hybrid_override_pattern") or "")
    mapping = {
        "*": ("attention",),
        "M": ("mamba",),
        "E": ("moe",),
        "-": ("ffn",),
    }
    if len(pattern) != int(language.get("num_hidden_layers", 0)):
        raise SetupError("Nemotron hybrid pattern does not match num_hidden_layers.")
    try:
        return tuple(mapping[character] for character in pattern)
    except KeyError as error:
        raise SetupError(f"Unsupported Nemotron hybrid marker: {error.args[0]}") from error


def count_candidate_options(
    config: Mapping[str, Any],
    inventory: ModelInventory,
    axes: Mapping[str, Any],
) -> CandidateCounts:
    """Count exact configuration-only vLLM and replace-one candidates."""
    if inventory.family == "qwen3_5":
        layer_families = _qwen_block_families(config, moe=inventory.moe)
    elif inventory.family == "nemotron3":
        layer_families = _nemotron_block_families(config)
    else:
        raise SetupError(f"Candidate counting is not implemented for {inventory.family}")

    active_subblocks = {subblock for family in layer_families for subblock in family}
    domains = {
        subblock: math.prod(
            _axis_domain_size(axes, axis_id) for axis_id in _SUBBLOCK_AXES[subblock]
        )
        for subblock in active_subblocks
    }
    unique_block_families = tuple(dict.fromkeys(layer_families))
    vllm_subblock = sum(domains[subblock] for subblock in active_subblocks)
    vllm_block = sum(
        math.prod(domains[subblock] for subblock in family) for family in unique_block_families
    )
    replacement_subblock = sum(
        sum(domains[subblock] - 1 for subblock in family) for family in layer_families
    )
    replacement_block = sum(
        math.prod(domains[subblock] for subblock in family) - 1 for family in layer_families
    )
    hidden = _mapping(axes.get("hidden_width"))
    widths = tuple(dict.fromkeys(int(value) for value in hidden.get("values") or ()))
    return CandidateCounts(
        vllm_subblock=vllm_subblock,
        vllm_block=vllm_block,
        replacement_subblock_per_width=replacement_subblock,
        replacement_block_per_width=replacement_block,
        width_count=len(widths) or 1,
    )


def resolve_profile(config: Mapping[str, Any]) -> ModelProfile:
    """Resolve an exact supported profile or raise an actionable handoff."""
    for profile in SUPPORTED_PROFILES:
        if profile.matches(config):
            return profile
    raise UnsupportedModelError(config)
