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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, locally customizable Puzzletron setup-v2 wizard."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from puzzletron_orchestrator.mesh import vllm_topology_to_mesh
from puzzletron_setup import (
    WORKER_REPOSITORY_PLACEHOLDER,
    WORKER_VENV_PLACEHOLDER,
    SetupError,
    validate_worker_path,
)
from puzzletron_setup.inspection import (
    infer_dataset_modality,
    inspect_model,
    normalize_dataset_source,
    normalize_model_source,
)
from puzzletron_setup.profiles import CandidateCounts, count_candidate_options

from .bundle import build_bundles_v2
from .defaults import DefaultsResolver, load_defaults
from .hf_datasets import (
    HfSubsetCatalog,
    discover_hf_subset_catalog,
    format_subset_choice,
    proportional_subset_weights,
)
from .parallel_validation import validate_automodel_parallelism, validate_vllm_parallelism
from .post_mip import FlowDraft, NodeDraft, PostMIPFlowEditor, recommended_flow
from .presets import QUICK_SETUP_PRESETS, SetupPreset, get_setup_preset
from .prompts import BACK, InteractiveBackend, PromptBackend, PromptChoice
from .resources import (
    ParallelProfile,
    ResourceProfileRegistry,
    StageResources,
    allocation_summary,
    resolve_batch,
)
from .session import WizardSession
from .state import WizardState

__all__ = ["SECTION_BUILDERS", "run_wizard_v2"]

BUILTINS = {
    "data": {"layout": "fixed", "sequence_length": 4096},
    "infrastructure": {
        "gpus_per_node": 8,
        "execution_contract": {
            "repository": WORKER_REPOSITORY_PLACEHOLDER,
            "venv": WORKER_VENV_PLACEHOLDER,
            "container": None,
            "container_mounts": None,
            "prerun_commands": [],
            "postrun_commands": [],
        },
        "runner": {
            "kind": "slurm",
            "slurm": {
                "account": "",
                "partition_interactive": "interactive",
                "partition_batch": "batch",
                "partition_cpu": None,
                "time_limit": "4:00:00",
                "qos": None,
                "max_nodes": 64,
            },
        },
    },
    "pruning": {
        "depth_granularity": "subblock",
        "depth_remove": 4,
        "replacement_granularity": "subblock",
        "width_importance_samples": 32768,
        "sort_sanity": False,
        "sort_sanity_samples": 128,
        "width_sanity": False,
        "width_sanity_samples": 128,
        "width_sanity_layer_count": 3,
        "width_sanity_targets_per_axis": 2,
        "slicing_sanity": False,
        "replacement_samples": 128,
        "bypass": {
            "enabled": True,
            "granularity": "subblock",
            "samples": 4096,
            "sequence_length": 4096,
            "batch_size": 8,
            "grad_accumulation_steps": 1,
        },
    },
    "vllm": {
        "enabled": False,
        "granularity": "subblock",
        "prefill_seq_len": 4096,
        "generation_seq_len": 1024,
        "batch_size": 1,
        "max_num_seqs": 1,
        "topology": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": False,
            "distributed_executor_backend": "mp",
        },
    },
    "mip": {
        "goal_metric": "params",
        "goal_value": "75%",
        "objective": "metrics.cosine_embedding_loss_hidden_states",
        "num_solutions": 8,
    },
}

STATIC_MODEL_STAGES = (
    "depth_importance",
    "width_importance",
    "bypass",
    "replacement_scoring",
)
STATIC_MODEL_BATCH_PATHS = {
    "depth_importance": "depth_importance.micro_batch_size",
    "width_importance": "pruning.micro_batch_size",
    "sort_sanity": "sort_sanity.micro_batch_size",
    "width_sanity": "width_sanity.micro_batch_size",
    "bypass": "bypass.training.micro_batch_size",
    "replacement_scoring": "replacement_scoring.micro_batch_size",
}
CANONICAL_STAGE_STRATEGIES = {
    "depth_importance": "persistent_pool",
    "width_importance": "single",
    "sort_sanity": "single",
    "width_sanity": "single",
    "bypass": "single",
    "replacement_scoring": "persistent_pool",
}

_CUSTOM_MODEL_SOURCE = "__custom_model_source__"
_DEFAULT_MODEL_SOURCE = "__default_model_source__"
_CUSTOM_DATA_SOURCE = "__custom_data_source__"
_DEFAULT_DATA_SOURCE = "__default_data_source__"
_PUZZLE_KD_DATA_SOURCE = "nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2"
_NEMOTRON_VLM_DATA_SOURCE = "nvidia/Nemotron-VLM-Dataset-v2"
_PUZZLE_KD_ADAPTER = "puzzle_kd_v2"
_CREATE_SERVING_WORKLOAD = "/create-new-serving-workload"
_NEMOTRON_VLM_ADAPTER = "nemotron_vlm_v2"
_NEMOTRON_VLM_DEFAULT_SUBSETS = ("sparsetables", "plotqa_cot", "wiki_en")

SUPPORTED_MODEL_GROUPS = (
    (
        "Nemotron 3",
        (
            (
                "Ultra 550B-A55B",
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            ),
            (
                "Super 120B-A12B",
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            ),
            (
                "Nano 30B-A3B",
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            ),
        ),
    ),
    (
        "Qwen 3.5/3.6 Dense",
        (
            ("Qwen 3.5 0.8B", "https://huggingface.co/Qwen/Qwen3.5-0.8B"),
            ("Qwen 3.5 2B", "https://huggingface.co/Qwen/Qwen3.5-2B"),
            ("Qwen 3.5 4B", "https://huggingface.co/Qwen/Qwen3.5-4B"),
            ("Qwen 3.5 9B", "https://huggingface.co/Qwen/Qwen3.5-9B"),
            ("Qwen 3.6 27B", "https://huggingface.co/Qwen/Qwen3.6-27B"),
        ),
    ),
    (
        "Qwen 3.5/3.6 MoE",
        (
            ("Qwen 3.6 35B-A3B", "https://huggingface.co/Qwen/Qwen3.6-35B-A3B"),
            ("Qwen 3.5 122B-A10B", "https://huggingface.co/Qwen/Qwen3.5-122B-A10B"),
            ("Qwen 3.5 397B-A17B", "https://huggingface.co/Qwen/Qwen3.5-397B-A17B"),
        ),
    ),
)


def _model_family_choices(resolver: DefaultsResolver) -> list[PromptChoice]:
    choices = []
    explicit = resolver.file_default("model.source")
    if explicit is not None and explicit.value:
        choices.append(PromptChoice(f"Default — {explicit.value}", _DEFAULT_MODEL_SOURCE))
    choices.extend(
        [
            PromptChoice("Custom", _CUSTOM_MODEL_SOURCE),
            *[PromptChoice(group, group) for group, _ in SUPPORTED_MODEL_GROUPS],
        ]
    )
    return choices


def _model_choices_for_family(family: str) -> list[PromptChoice]:
    for group, models in SUPPORTED_MODEL_GROUPS:
        if group == family:
            return [PromptChoice(short_name, url) for short_name, url in models]
    raise SetupError(f"Unknown supported model family: {family}")


def _data_source_choices(
    resolver: DefaultsResolver,
    *,
    multimodal_model: bool = True,
) -> list[PromptChoice]:
    choices = []
    explicit = resolver.file_default("data.source")
    explicit_source = None
    if explicit is not None and explicit.value:
        explicit_source = normalize_dataset_source(str(explicit.value))
        if multimodal_model or explicit_source != _NEMOTRON_VLM_DATA_SOURCE:
            choices.append(PromptChoice(f"Default — {explicit.value}", _DEFAULT_DATA_SOURCE))
    first_class = [
        (
            "NVIDIA Puzzle-KD v2: recommended text pruning dataset",
            _PUZZLE_KD_DATA_SOURCE,
        )
    ]
    if multimodal_model:
        first_class.append(
            (
                "NVIDIA Nemotron-VLM v2: recommended image-text dataset",
                _NEMOTRON_VLM_DATA_SOURCE,
            )
        )
    choices.extend(
        PromptChoice(title, source) for title, source in first_class if source != explicit_source
    )
    choices.append(
        PromptChoice(
            "Custom dataset: choose a local path or Hugging Face dataset",
            _CUSTOM_DATA_SOURCE,
        )
    )
    return choices


def _first_class_data_adapter(source: str) -> str | None:
    normalized = normalize_dataset_source(source)
    if normalized == _PUZZLE_KD_DATA_SOURCE:
        return _PUZZLE_KD_ADAPTER
    if normalized == _NEMOTRON_VLM_DATA_SOURCE:
        return _NEMOTRON_VLM_ADAPTER
    return None


def _cached_hf_subset_catalog(
    state: WizardState,
    source: str,
) -> HfSubsetCatalog | None:
    cache = state.collection("hf_dataset_catalogs")
    if not isinstance(cache, Mapping):
        return None
    selection = state.collection("data_subset_selection")
    revision = (
        str(selection.get("revision", ""))
        if isinstance(selection, Mapping) and selection.get("source") == source
        else ""
    )
    if revision:
        payload = cache.get(f"{source}@{revision}")
        if isinstance(payload, Mapping):
            return HfSubsetCatalog.from_dict(payload)
    for payload in reversed(tuple(cache.values())):
        if isinstance(payload, Mapping) and payload.get("source") == source:
            return HfSubsetCatalog.from_dict(payload)
    return None


def _select_hf_subsets(
    session: WizardSession,
    resolver: DefaultsResolver,
    source: str,
    *,
    require_hosted_media: bool,
    catalog_loader: Callable[..., HfSubsetCatalog],
) -> Any:
    catalog = _cached_hf_subset_catalog(session.state, source)
    if catalog is None:
        print(f"  Discovering Hugging Face subsets for {source}...")
        catalog = catalog_loader(
            source,
            require_hosted_media=require_hosted_media,
        )
        cache = session.state.collection("hf_dataset_catalogs")
        updated_cache = dict(cache) if isinstance(cache, Mapping) else {}
        updated_cache[f"{catalog.source}@{catalog.revision}"] = catalog.to_dict()
        session.state.set_collection("hf_dataset_catalogs", updated_cache)

    selectable = [item for item in catalog.subsets if item.selectable]
    if not selectable:
        raise SetupError(
            f"Hugging Face dataset {source} has no selectable subsets with "
            "known positive row counts and sizes."
        )
    configured = resolver.resolve("data.subsets", None)
    if configured.value is None:
        configured = resolver.resolve("data.acquisition.subsets", None)
    configured_defaults = configured.value
    if isinstance(configured_defaults, str):
        configured_defaults = [
            item.strip() for item in configured_defaults.split(",") if item.strip()
        ]
    if configured_defaults:
        preferred = [str(item) for item in configured_defaults]
    elif source == _NEMOTRON_VLM_DATA_SOURCE:
        preferred = list(_NEMOTRON_VLM_DEFAULT_SUBSETS)
    elif catalog.default_subset:
        preferred = [catalog.default_subset]
    else:
        preferred = [selectable[0].name]
    selectable_names = {item.name for item in selectable}
    unavailable = [name for name in preferred if name not in selectable_names]
    if configured_defaults and unavailable:
        raise SetupError(
            "Configured dataset subsets are missing or unavailable for "
            f"{source}: {', '.join(unavailable)}. Choose from: "
            f"{', '.join(sorted(selectable_names))}."
        )
    defaults = [name for name in preferred if name in selectable_names]
    if not defaults:
        defaults = [selectable[0].name]

    def validate(selected_names: list[str]) -> bool | str:
        try:
            proportional_subset_weights(catalog, selected_names)
        except SetupError as error:
            return str(error)
        return True

    if session.guided:
        selected_names = list(defaults)
        verdict = validate(selected_names)
        if verdict is not True:
            raise SetupError(str(verdict))
        print(
            "  Dataset subsets: "
            f"{', '.join(selected_names)} (recommended defaults; use --full to customize)"
        )
    else:
        selected = session.checkbox(
            "data.subsets",
            "Dataset subsets:",
            [
                PromptChoice(
                    format_subset_choice(item),
                    item.name,
                    disabled=item.disabled_reason,
                )
                for item in catalog.subsets
            ],
            defaults=defaults,
            validate=validate,
        )
        if selected is BACK:
            return BACK
        selected_names = [str(name) for name in selected]
    weights = proportional_subset_weights(catalog, selected_names)
    return catalog, selected_names, weights


def _nested_records(state: WizardState) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, record in state.records().items():
        current = nested
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = deepcopy(record.effective)
    return nested


def _resolver(
    state: WizardState,
    defaults_path: Path | None,
    preset: SetupPreset | None = None,
    family_config: str | Path | None = None,
    model_inventory: Any | None = None,
) -> DefaultsResolver:
    preset_defaults = {}
    model_profile_defaults = {}
    if preset is not None and family_config is not None:
        preset_defaults, model_profile_defaults = preset.resolved_default_layers(
            family_config,
            model_inventory,
        )
    return DefaultsResolver(
        builtins=BUILTINS,
        model_derived={},
        preset_defaults=preset_defaults,
        model_profile_defaults=model_profile_defaults,
        file_defaults=load_defaults(defaults_path),
        preserved=_nested_records(state),
    )


def _resolved(
    state: WizardState,
    resolver: DefaultsResolver,
    path: str,
    fallback: Any = None,
) -> Any:
    value = resolver.resolve(path, fallback)
    print(f"  {path}: {value.value!r} ({value.source})")
    return value


def _record_default(
    state: WizardState,
    resolver: DefaultsResolver,
    path: str,
    fallback: Any = None,
    *,
    dependencies: tuple[str, ...] = (),
) -> Any:
    resolved = resolver.resolve_default(path, fallback)
    state.set_field(
        path,
        resolved.value,
        source=resolved.source,
        dependencies=dependencies,
    )
    return resolved.value


def _section_action(
    session: WizardSession,
    section: str,
    summary: str,
    defaults: Mapping[str, Any],
    *,
    prompt_in_guided: bool = False,
) -> Any:
    session.begin(section)
    if session.guided and not prompt_in_guided:
        return "defaults"
    print(f"\n[{section}] {summary}")
    _print_default_decisions(defaults)
    return session.select(
        f"{section}.action",
        f"{section.replace('_', ' ').title()}:",
        [
            ("Use defaults shown above", "defaults"),
            ("Customize", "customize"),
        ],
        default="defaults",
    )


def _print_default_decisions(defaults: Mapping[str, Any]) -> None:
    print("  Resolved defaults:")
    rendered = yaml.safe_dump(
        _plain_review_value(defaults),
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    for line in rendered.splitlines():
        print(f"    {line}")


def _plain_review_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_review_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_review_value(item) for item in value]
    return value


def _vllm_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]:
    width_count = counts.effective_vllm_width_count
    if width_count == 1:
        return [
            (
                f"Sublayer — {counts.vllm_subblock_total} unique configurations",
                "subblock",
            ),
            (
                f"Whole block — {counts.vllm_block_total} unique configurations",
                "block",
            ),
        ]
    return [
        (
            f"Sublayer — {counts.vllm_subblock} configurations/width, "
            f"{counts.vllm_subblock_total} total across {width_count} widths",
            "subblock",
        ),
        (
            f"Whole block — {counts.vllm_block} configurations/width, "
            f"{counts.vllm_block_total} total across {width_count} widths",
            "block",
        ),
    ]


def _replacement_granularity_choices(
    counts: CandidateCounts,
) -> list[tuple[str, str]]:
    width_count = counts.width_count

    def label(name: str, per_width: int, total: int) -> str:
        if width_count == 1:
            return f"{name} — {total} options"
        return f"{name} — {per_width} options/width, {total} total across {width_count} widths"

    return [
        (
            label(
                "One sublayer at a time",
                counts.replacement_subblock_per_width,
                counts.replacement_subblock_total,
            ),
            "subblock",
        ),
        (
            label(
                "One layer at a time",
                counts.replacement_block_per_width,
                counts.replacement_block_total,
            ),
            "block",
        ),
    ]


def _depth_granularity_choices(inventory: Any) -> list[tuple[str, str]]:
    return [
        (f"Sublayer — {inventory.num_sublayers} available", "subblock"),
        (f"Whole layer — {inventory.num_layers} available", "block"),
    ]


def _default_axis_values(axis: Any) -> list[int]:
    legal_values = tuple(int(value) for value in axis.values)
    teacher = int(axis.teacher_value)
    half = min(
        legal_values,
        key=lambda value: (abs(value - teacher // 2), -value),
    )
    return list(dict.fromkeys((teacher, half)))


def _text_field(
    session: WizardSession,
    resolver: DefaultsResolver,
    path: str,
    label: str,
    fallback: str = "",
    *,
    validate: Callable[[Any], bool | str] | None = None,
) -> Any:
    resolved = _resolved(session.state, resolver, path, fallback)
    value = session.text(
        path,
        label,
        default=str(resolved.value or ""),
        validate=validate,
    )
    if value is not BACK:
        session.state.set_field(path, value, source="user")
    return value


def _integer_field(
    session: WizardSession,
    resolver: DefaultsResolver,
    path: str,
    label: str,
    fallback: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> Any:
    resolved = _resolved(session.state, resolver, path, fallback)
    value = session.integer(
        path,
        label,
        default=int(resolved.value),
        minimum=minimum,
        maximum=maximum,
    )
    if value is not BACK:
        session.state.set_field(path, value, source="user")
    return value


def _select_model_source(
    session: WizardSession,
    resolver: DefaultsResolver,
) -> Any:
    while True:
        family = session.select(
            "model.source_family",
            "Model:",
            _model_family_choices(resolver),
            default=_CUSTOM_MODEL_SOURCE,
        )
        if family is BACK:
            return BACK
        if family == _DEFAULT_MODEL_SOURCE:
            explicit = resolver.file_default("model.source")
            if explicit is None:
                raise SetupError("The selected model default is no longer available.")
            return explicit.value, "defaults_file"
        if family == _CUSTOM_MODEL_SOURCE:
            source = _text_field(
                session,
                resolver,
                "model.source",
                "Local model path or Hugging Face URL:",
            )
            if source is BACK:
                continue
            return source, "user"
        source = session.select(
            "model.source_model",
            f"{family} model:",
            _model_choices_for_family(str(family)),
        )
        if source is not BACK:
            return source, "user"


def model_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    session.begin("model")
    while True:
        selected = _select_model_source(session, resolver)
        if selected is BACK:
            return False
        source, source_kind = selected
        try:
            source = normalize_model_source(str(source))
        except SetupError as error:
            print(f"  {error}")
            continue
        session.state.set_field("model.source", source, source=source_kind)
        break
    model = inspect_model(str(source))
    session.state.set_model(model.to_dict(), model.inventory.to_dict())
    context["model"] = model
    print(
        f"  Detected {model.inventory.family}; {model.inventory.num_layers} layers, "
        f"{model.inventory.num_sublayers} sublayers, MoE={model.inventory.moe}."
    )
    return True


def data_section(
    session: WizardSession,
    resolver: DefaultsResolver,
    context: dict,
    *,
    catalog_loader: Callable[..., HfSubsetCatalog] = discover_hf_subset_catalog,
) -> bool:
    session.begin("data")
    previous_acquisition = _mapping_copy(session.state.collection("data_acquisition"))
    explicit = resolver.file_default("data.source")
    if explicit is not None and explicit.value:
        _print_default_decisions({"source": explicit.value})
    choices = _data_source_choices(
        resolver,
        multimodal_model=bool(context["model"].inventory.multimodal),
    )
    default_choice_available = any(choice.value == _DEFAULT_DATA_SOURCE for choice in choices)
    mode = session.select(
        "data.source_mode",
        "Dataset:",
        choices,
        default=(_DEFAULT_DATA_SOURCE if default_choice_available else _CUSTOM_DATA_SOURCE),
    )
    if mode is BACK:
        return False
    if mode == _CUSTOM_DATA_SOURCE:
        source = _text_field(
            session,
            resolver,
            "data.source",
            "Local dataset path or Hugging Face URL:",
        )
        if source is BACK:
            return False
        source = normalize_dataset_source(str(source))
        source_kind = "user"
    elif mode == _DEFAULT_DATA_SOURCE:
        if explicit is None or not explicit.value:
            raise SetupError("The selected dataset default is no longer available.")
        source = normalize_dataset_source(str(explicit.value))
        source_kind = explicit.source
    else:
        source = normalize_dataset_source(str(mode))
        source_kind = "builtin"

    adapter = _first_class_data_adapter(source)
    fixed_modality = (
        "text"
        if adapter == _PUZZLE_KD_ADAPTER
        else "multimodal"
        if adapter == _NEMOTRON_VLM_ADAPTER
        else None
    )
    finding = infer_dataset_modality(source) if fixed_modality is None else None
    finding_modality = finding.modality if finding is not None else fixed_modality
    finding_evidence = finding.evidence if finding is not None else f"{adapter} first-class source"
    modality_choices = [("Text", "text")]
    if context["model"].inventory.multimodal:
        modality_choices.append(("Multimodal", "multimodal"))
    modality_default = resolver.resolve(
        "data.modality",
        finding_modality if finding_modality != "unknown" else "text",
    )
    suggested_modality = str(modality_default.value)
    modality_source = (
        "inferred" if modality_default.source == "fallback" else modality_default.source
    )
    if fixed_modality == "multimodal" and not context["model"].inventory.multimodal:
        raise SetupError(
            "NVIDIA Nemotron-VLM v2 requires a multimodal model. "
            "Choose a multimodal checkpoint or a text dataset."
        )
    valid_modalities = {value for _, value in modality_choices}
    if suggested_modality not in valid_modalities:
        if modality_source != "inferred" or session.guided:
            raise SetupError(
                f"Resolved data modality {suggested_modality!r} ({modality_source}) "
                "is incompatible with the selected model."
            )
        suggested_modality = "text"
        modality_source = "inferred"
    if (
        fixed_modality is not None
        and modality_source != "inferred"
        and suggested_modality != fixed_modality
    ):
        raise SetupError(
            f"Configured data modality {suggested_modality!r} ({modality_source}) "
            f"conflicts with {source}, which requires {fixed_modality!r}."
        )
    if fixed_modality is None and session.guided:
        modality = suggested_modality
        print(
            f"  Data modality: {modality} ({modality_source}; {finding_evidence}; "
            "use --full to override)"
        )
    elif fixed_modality is None:
        modality = session.select(
            "data.modality",
            f"Data modality ({finding_evidence}):",
            modality_choices,
            default=suggested_modality,
        )
        if modality is BACK:
            return False
        modality_source = "user"
    else:
        modality = fixed_modality
        print(f"  Data modality: {modality} ({finding_evidence})")

    acquisition: dict[str, Any] | None = None
    subset_selection: dict[str, Any] | None = None
    selected_subsets: list[str] | None = None
    runtime_source = source
    if adapter is not None:
        default_root = (
            session.state.campaign_dir.parent
            / f"{session.state.campaign_dir.name}_datasets"
            / adapter
        )
        if session.guided:
            output = _record_default(
                session.state,
                resolver,
                "data.acquisition.output",
                str(default_root),
            )
            print(f"  Local materialization directory: {output}")
        else:
            output = _text_field(
                session,
                resolver,
                "data.acquisition.output",
                "Local materialization directory:",
                str(default_root),
            )
            if output is BACK:
                return False
        runtime_source = str(Path(str(output)).expanduser().resolve())
        seed_default = 408 if adapter == _PUZZLE_KD_ADAPTER else 42
        if session.guided:
            seed = _record_default(
                session.state,
                resolver,
                "data.acquisition.seed",
                seed_default,
            )
            print(f"  Deterministic dataset selection seed: {seed}")
        else:
            seed = _integer_field(
                session,
                resolver,
                "data.acquisition.seed",
                "Deterministic dataset selection seed:",
                seed_default,
                minimum=0,
            )
            if seed is BACK:
                return False
        acquisition = {
            "adapter": adapter,
            "source": source,
            "output": runtime_source,
            "seed": int(seed),
        }

    if not Path(source).expanduser().exists():
        selection = _select_hf_subsets(
            session,
            resolver,
            source,
            require_hosted_media=adapter == _NEMOTRON_VLM_ADAPTER,
            catalog_loader=catalog_loader,
        )
        if selection is BACK:
            return False
        catalog, selected_subsets, weights = selection
        by_name = {item.name: item for item in catalog.subsets}
        subset_selection = {
            "source": catalog.source,
            "revision": catalog.revision,
            "subsets": [
                {
                    "name": name,
                    "num_rows": by_name[name].num_rows,
                    "num_bytes_original_files": by_name[name].num_bytes_original_files,
                    "num_media_shards": by_name[name].num_media_shards,
                    "weight": weights[name],
                }
                for name in selected_subsets
            ],
        }

    if adapter is not None:
        if adapter == _PUZZLE_KD_ADAPTER:
            print(
                "  Puzzle-KD train and validation row counts will be inferred "
                "from the completed stage configuration."
            )
        else:
            if not selected_subsets:
                raise SetupError(
                    "Nemotron-VLM v2 requires at least one selectable hosted-media subset."
                )
            subset_media_shards = {
                record["name"]: record["num_media_shards"]
                for record in subset_selection["subsets"]
                if record["num_media_shards"] is not None
            }
            acquisition.update(
                subsets=selected_subsets,
                subset_rows={
                    record["name"]: record["num_rows"] for record in subset_selection["subsets"]
                },
                subset_weights={
                    record["name"]: record["weight"] for record in subset_selection["subsets"]
                },
                subset_media_shards=subset_media_shards,
                revision=subset_selection["revision"],
            )
            if len(subset_media_shards) != len(selected_subsets):
                previous_shard_cap = int(previous_acquisition.get("max_shards_per_subset", 0))
                if previous_shard_cap > 0:
                    acquisition["max_shards_per_subset"] = previous_shard_cap
            print(
                "  Nemotron-VLM row and media-shard counts will be inferred "
                "from the completed stage configuration and selected subsets."
            )
    default_layout = str(resolver.resolve("data.layout", "fixed").value)
    if default_layout == "padded":
        default_layout = "padded_varlen"
    if session.guided:
        layout_default = resolver.resolve_default("data.layout", default_layout)
        layout = str(layout_default.value)
        if layout == "padded":
            layout = "padded_varlen"
        sequence_default = resolver.resolve_default("data.sequence_length", 4096)
        sequence = int(sequence_default.value)
        print(
            f"  Data shape: {layout}, sequence length {sequence} "
            "(resolved defaults; use --full to customize)"
        )
    else:
        layout = session.select(
            "data.layout",
            "Dataset layout:",
            [
                ("Fixed-length", "fixed"),
                ("Packed variable-length", "packed_varlen"),
                ("Padded variable-length", "padded_varlen"),
            ],
            default=default_layout,
        )
        if layout is BACK:
            return False
        sequence = _integer_field(
            session,
            resolver,
            "data.sequence_length",
            "Sequence length used by width, depth, bypass, evaluation, and global KD:",
            4096,
        )
        if sequence is BACK:
            return False
    session.state.set_field("data.source", runtime_source, source=source_kind)
    session.state.set_field("data.selected_source", source, source=source_kind)
    session.state.set_field("data.adapter", adapter or "custom", source=source_kind)
    session.state.set_collection("data_acquisition", acquisition or {})
    session.state.set_collection("data_subset_selection", subset_selection or {})
    session.state.set_field("data.modality", modality, source=modality_source)
    session.state.set_field(
        "data.layout",
        layout,
        source=layout_default.source if session.guided else "user",
    )
    session.state.set_field(
        "data.sequence_length",
        int(sequence),
        source=sequence_default.source if session.guided else "user",
    )
    return True


def infrastructure_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    del context
    paths = (
        ("infrastructure.execution_contract.repository", WORKER_REPOSITORY_PLACEHOLDER),
        ("infrastructure.execution_contract.venv", WORKER_VENV_PLACEHOLDER),
        ("infrastructure.execution_contract.container", None),
        ("infrastructure.execution_contract.container_mounts", None),
        ("infrastructure.runner.kind", "slurm"),
        ("infrastructure.runner.slurm.account", ""),
        ("infrastructure.runner.slurm.partition_interactive", "interactive"),
        ("infrastructure.runner.slurm.partition_batch", "batch"),
        ("infrastructure.runner.slurm.partition_cpu", None),
        ("infrastructure.runner.slurm.time_limit", "4:00:00"),
        ("infrastructure.runner.slurm.qos", None),
        ("infrastructure.runner.slurm.max_nodes", 64),
        ("infrastructure.gpus_per_node", 8),
    )
    preview = {
        path.removeprefix("infrastructure."): resolver.resolve_default(path, fallback).value
        for path, fallback in paths
    }
    preview["execution_contract.prerun_commands"] = resolver.resolve_default(
        "infrastructure.execution_contract.prerun_commands", []
    ).value
    preview["execution_contract.postrun_commands"] = resolver.resolve_default(
        "infrastructure.execution_contract.postrun_commands", []
    ).value
    action = _section_action(
        session,
        "infrastructure",
        "Configure the worker contract and cluster facts before stage allocations.",
        preview,
        prompt_in_guided=True,
    )
    if action is BACK:
        return False
    if action == "defaults":
        for path, fallback in paths:
            _record_default(session.state, resolver, path, fallback)
        commands = resolver.resolve_default("infrastructure.execution_contract.prerun_commands", [])
        session.state.set_field(
            "infrastructure.execution_contract.prerun_commands",
            list(commands.value or ()),
            source=commands.source,
        )
        postrun = resolver.resolve_default("infrastructure.execution_contract.postrun_commands", [])
        session.state.set_field(
            "infrastructure.execution_contract.postrun_commands",
            list(postrun.value or ()),
            source=postrun.source,
        )
        return True

    worker_path_fields = {
        "infrastructure.execution_contract.repository",
        "infrastructure.execution_contract.venv",
    }
    for path, label, fallback in (
        (
            "infrastructure.execution_contract.repository",
            "Repository path on workers:",
            WORKER_REPOSITORY_PLACEHOLDER,
        ),
        (
            "infrastructure.execution_contract.venv",
            "Python environment:",
            WORKER_VENV_PLACEHOLDER,
        ),
        ("infrastructure.execution_contract.container", "Container image (blank for none):", ""),
        (
            "infrastructure.execution_contract.container_mounts",
            "Container mounts (blank for none):",
            "",
        ),
        ("infrastructure.runner.slurm.account", "Slurm account:", ""),
        (
            "infrastructure.runner.slurm.partition_interactive",
            "Interactive partition:",
            "interactive",
        ),
        ("infrastructure.runner.slurm.partition_batch", "Batch partition:", "batch"),
        ("infrastructure.runner.slurm.partition_cpu", "CPU partition:", ""),
        ("infrastructure.runner.slurm.time_limit", "Default time limit:", "4:00:00"),
    ):
        value = _text_field(
            session,
            resolver,
            path,
            label,
            fallback,
            validate=validate_worker_path if path in worker_path_fields else None,
        )
        if value is BACK:
            return False
        if value == "":
            session.state.set_field(path, None, source="user")
    gpus = _integer_field(
        session,
        resolver,
        "infrastructure.gpus_per_node",
        "GPUs per node:",
        8,
    )
    if gpus is BACK:
        return False
    prerun_default = resolver.resolve("infrastructure.execution_contract.prerun_commands", []).value
    prerun = session.text(
        "infrastructure.execution_contract.prerun_commands",
        "Pre-run commands separated by ';;':",
        default=";;".join(prerun_default or ()),
    )
    if prerun is BACK:
        return False
    session.state.set_field(
        "infrastructure.execution_contract.prerun_commands",
        [item.strip() for item in str(prerun).split(";;") if item.strip()],
        source="user",
    )
    session.state.set_field(
        "infrastructure.execution_contract.postrun_commands", [], source="builtin"
    )
    session.state.set_field("infrastructure.runner.kind", "slurm", source="user")
    session.state.set_field("infrastructure.runner.slurm.qos", None, source="builtin")
    session.state.set_field("infrastructure.runner.slurm.max_nodes", 64, source="builtin")
    return True


def pruning_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    model = context["model"]
    inventory = model.inventory
    default_granularity = str(
        resolver.resolve_default("pruning.depth_granularity", "subblock").value
    )
    default_count = (
        inventory.num_sublayers if default_granularity == "subblock" else inventory.num_layers
    )
    default_remove = min(
        int(resolver.resolve_default("pruning.depth_remove", min(4, default_count - 1)).value),
        default_count - 1,
    )
    default_axes = {
        axis.axis_id: resolver.resolve_default(
            f"pruning.axes.{axis.axis_id}.values",
            _default_axis_values(axis),
        ).value
        for axis in inventory.axes
    }
    action = _section_action(
        session,
        "pruning",
        "Choose depth and every descriptor-supported pruning-axis domain.",
        {
            "depth_granularity": default_granularity,
            "maximum_depth_removed": default_remove,
            "axes": default_axes,
            "width_importance_samples": resolver.resolve_default(
                "pruning.width_importance_samples", 32768
            ).value,
            "replacement_scoring_samples": resolver.resolve_default(
                "pruning.replacement_samples", 128
            ).value,
            "sort_sanity": resolver.resolve_default("pruning.sort_sanity", False).value,
            "width_sanity": resolver.resolve_default("pruning.width_sanity", False).value,
            "slicing_sanity": resolver.resolve_default("pruning.slicing_sanity", False).value,
            "replacement_granularity": resolver.resolve_default(
                "pruning.replacement_granularity", default_granularity
            ).value,
            "bypass": {
                "enabled": resolver.resolve_default("pruning.bypass.enabled", True).value,
                "granularity": resolver.resolve_default(
                    "pruning.bypass.granularity", "subblock"
                ).value,
                "samples": resolver.resolve_default("pruning.bypass.samples", 4096).value,
                "sequence_length": session.state.get_field("data.sequence_length", 4096),
                "batch_size": resolver.resolve_default("pruning.bypass.batch_size", 8).value,
                "gradient_accumulation_steps": resolver.resolve_default(
                    "pruning.bypass.grad_accumulation_steps", 1
                ).value,
            },
        },
    )
    if action is BACK:
        return False

    defaults = deepcopy(BUILTINS["pruning"])
    defaults["width_importance_samples"] = int(
        resolver.resolve_default(
            "pruning.width_importance_samples",
            defaults["width_importance_samples"],
        ).value
    )
    defaults["replacement_samples"] = int(
        resolver.resolve_default(
            "pruning.replacement_samples",
            defaults["replacement_samples"],
        ).value
    )
    defaults["bypass"]["enabled"] = bool(
        resolver.resolve_default(
            "pruning.bypass.enabled",
            defaults["bypass"]["enabled"],
        ).value
    )
    for key, fallback in (
        ("granularity", defaults["bypass"]["granularity"]),
        ("samples", defaults["bypass"]["samples"]),
        ("batch_size", defaults["bypass"]["batch_size"]),
        (
            "grad_accumulation_steps",
            defaults["bypass"]["grad_accumulation_steps"],
        ),
    ):
        defaults["bypass"][key] = resolver.resolve_default(f"pruning.bypass.{key}", fallback).value
    if action == "customize":
        granularity = session.select(
            "pruning.depth_granularity",
            "Depth pruning granularity:",
            _depth_granularity_choices(inventory),
            default="subblock",
        )
        if granularity is BACK:
            return False
        count = inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        remove = session.integer(
            "pruning.depth_remove",
            (
                "Maximum number to remove "
                f"(network has {count} "
                f"{'sublayers' if granularity == 'subblock' else 'layers'}):"
            ),
            default=min(4, count - 1),
            minimum=0,
            maximum=count - 1,
        )
        if remove is BACK:
            return False
    else:
        granularity = default_granularity
        count = inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        remove = default_remove
    axes = {}
    for axis in inventory.axes:
        axis_values = list(axis.values)
        default_values = list(
            resolver.resolve_default(
                f"pruning.axes.{axis.axis_id}.values",
                _default_axis_values(axis),
            ).value
        )
        if action == "customize":
            selected = session.checkbox(
                f"pruning.axes.{axis.axis_id}",
                f"Values for {axis.label}:",
                [(str(value), value) for value in axis_values],
                defaults=default_values,
                validate=lambda values: bool(values) or "Select at least one value.",
            )
            if selected is BACK:
                return False
        else:
            selected = default_values
        axes[axis.axis_id] = {
            "enabled": True,
            "teacher_value": axis.teacher_value,
            "values": sorted({int(value) for value in selected}, reverse=True),
            "alignment": axis.alignment,
        }
    defaults.update(
        {
            "depth_granularity": granularity,
            "depth_remove": int(remove),
            "axes": axes,
            "sort_sanity": bool(resolver.resolve_default("pruning.sort_sanity", False).value),
            "width_sanity": bool(resolver.resolve_default("pruning.width_sanity", False).value),
            "slicing_sanity": bool(resolver.resolve_default("pruning.slicing_sanity", False).value),
            "replacement_granularity": str(
                resolver.resolve_default("pruning.replacement_granularity", granularity).value
            ),
        }
    )
    defaults["bypass"]["sequence_length"] = int(
        session.state.get_field("data.sequence_length", 4096)
    )
    if action == "customize":
        defaults["width_importance_samples"] = session.integer(
            "pruning.width_importance_samples",
            "Width-importance samples:",
            default=int(defaults["width_importance_samples"]),
            minimum=1,
        )
        defaults["replacement_samples"] = session.integer(
            "pruning.replacement_samples",
            "Replacement-scoring samples:",
            default=int(defaults["replacement_samples"]),
            minimum=1,
        )
        bypass_enabled = session.confirm(
            "pruning.bypass.enabled", "Run local bypass distillation?", default=True
        )
        if BACK in (
            defaults["width_importance_samples"],
            defaults["replacement_samples"],
            bypass_enabled,
        ):
            return False
        defaults["bypass"]["enabled"] = bool(bypass_enabled)
    session.state.set_collection("pruning", defaults)
    for path, value in (
        ("pruning.depth_granularity", granularity),
        ("pruning.depth_remove", int(remove)),
    ):
        session.state.set_field(path, value, source="user" if action == "customize" else "builtin")
    return True


def _profile_summary(profile: ParallelProfile) -> str:
    return (
        f"TP={profile.tp} CP={profile.cp} PP={profile.pp} "
        f"DP-shard={profile.dp_shard} DP-replicate={profile.dp_replicate} "
        f"EP={profile.ep} — {profile.gpu_count} GPUs per task"
    )


def _profile_compatibility_issues(
    session: WizardSession,
    profile: ParallelProfile,
    stage_id: str,
    model: Any,
    *,
    node_type: str | None = None,
):
    return validate_automodel_parallelism(
        profile,
        model.inventory,
        _mapping_copy(session.state.collection("pruning")),
        stage_id=stage_id,
        node_type=node_type,
        sequence_length=int(session.state.get_field("data.sequence_length", 4096)),
    )


def _print_parallel_issues(issues) -> None:
    for issue in issues:
        print(f"  {issue.path}: {issue.message}")
    print("  Choose a different parallel setting.")


def _compatible_default_profile(
    session: WizardSession,
    registry: ResourceProfileRegistry,
    stage_id: str,
    model: Any,
    *,
    node_type: str | None = None,
) -> tuple[ParallelProfile | None, tuple[Any, ...]]:
    """Reuse the first compatible profile without opening advanced prompts."""
    last_issues: tuple[Any, ...] = ()
    for name in registry.names():
        profile = registry.get(name)
        issues = tuple(
            _profile_compatibility_issues(
                session,
                profile,
                stage_id,
                model,
                node_type=node_type,
            )
        )
        if not issues:
            return registry.reuse(name, consumer=stage_id), ()
        last_issues = issues
    if not registry.names():
        profile = ParallelProfile(stage_id)
        issues = tuple(
            _profile_compatibility_issues(
                session,
                profile,
                stage_id,
                model,
                node_type=node_type,
            )
        )
        if not issues:
            return registry.create(profile, consumer=stage_id), ()
        last_issues = issues
    return None, last_issues


def _profile_prompt(
    session: WizardSession,
    registry: ResourceProfileRegistry,
    stage_id: str,
    model: Any,
    *,
    node_type: str | None = None,
) -> Any:
    while True:
        names = registry.names()
        if names:
            choices = [
                (
                    f"Reuse {name} — {_profile_summary(registry.get(name))}",
                    f"reuse:{name}",
                )
                for name in names
            ]
            choices.append(("Create a new configuration", "new"))
            action = session.select(
                f"stages.{stage_id}.profile_action",
                f"Parallel setting for {stage_id}:",
                choices,
                default=f"reuse:{names[0]}",
            )
            if action is BACK:
                return BACK
            if str(action).startswith("reuse:"):
                name = str(action).split(":", 1)[1]
                profile = registry.get(name)
                issues = _profile_compatibility_issues(
                    session,
                    profile,
                    stage_id,
                    model,
                    node_type=node_type,
                )
                if issues:
                    _print_parallel_issues(issues)
                    continue
                return registry.reuse(name, consumer=stage_id)

        base = ParallelProfile(stage_id)
        name = session.text(
            f"stages.{stage_id}.profile_name",
            "Parallel configuration name:",
            default=stage_id,
        )
        if name is BACK:
            return BACK
        values = {}
        for field_name, label, default in (
            ("tp", "Tensor parallel (TP):", base.tp),
            ("cp", "Context parallel (CP):", base.cp),
            ("pp", "Pipeline parallel (PP):", base.pp),
            ("dp_shard", "FSDP shard degree:", base.dp_shard),
            ("dp_replicate", "Data-parallel replicas:", base.dp_replicate),
            ("ep", "Expert parallel (EP):", base.ep if model.inventory.moe else 1),
        ):
            if field_name == "ep" and not model.inventory.moe:
                values[field_name] = 1
                continue
            value = session.integer(
                f"stages.{stage_id}.parallel.{field_name}",
                label,
                default=int(default),
                minimum=1,
            )
            if value is BACK:
                return BACK
            values[field_name] = int(value)
        sequence_parallel = session.confirm(
            f"stages.{stage_id}.parallel.sequence_parallel",
            "Enable sequence parallelism?",
            default=base.sequence_parallel,
        )
        if sequence_parallel is BACK:
            return BACK
        profile = ParallelProfile(
            name=str(name), sequence_parallel=bool(sequence_parallel), **values
        )
        issues = _profile_compatibility_issues(
            session,
            profile,
            stage_id,
            model,
            node_type=node_type,
        )
        if issues:
            _print_parallel_issues(issues)
            continue
        confirmed = session.confirm(
            f"stages.{stage_id}.parallel.confirm",
            (f"Are you sure you want to use {profile.name} — {_profile_summary(profile)}?"),
            default=True,
        )
        if confirmed is BACK:
            return BACK
        if confirmed:
            break
    if str(name) in registry.names():
        registry.update(profile)
        registry.reuse(str(name), consumer=stage_id)
    else:
        registry.create(profile, consumer=stage_id)
    return profile


def _pruning_payload(state: WizardState) -> dict[str, Any]:
    payload = deepcopy(BUILTINS["pruning"])
    current = state.collection("pruning")
    if not isinstance(current, Mapping):
        return payload
    for key, value in current.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), Mapping):
            merged = dict(payload[key])
            merged.update(deepcopy(dict(value)))
            payload[key] = merged
        else:
            payload[key] = deepcopy(value)
    return payload


def _resource_registry(
    session: WizardSession,
    resolver: DefaultsResolver,
) -> ResourceProfileRegistry:
    registry = ResourceProfileRegistry.from_dict(
        session.state.collection("parallel_profiles") or {}
    )
    if not registry.names():
        registry = ResourceProfileRegistry.from_dict(
            resolver.resolve_default("profiles", {}).value or {}
        )
    return registry


def _stage_resource_defaults(
    session: WizardSession,
    resolver: DefaultsResolver,
    stage_id: str,
    *,
    strategy: str | None = None,
    batch: int,
) -> dict[str, Any]:
    registry = _resource_registry(session, resolver)
    profile = registry.get(registry.names()[0]) if registry.names() else ParallelProfile(stage_id)
    strategy = strategy or CANONICAL_STAGE_STRATEGIES[stage_id]
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    resolved_instances = resolver.resolve_default(
        f"stages.{stage_id}.instances",
        gpus_per_node,
    )
    instances = 1 if strategy == "single" else int(resolved_instances.value)
    resolved_batch = resolver.resolve_default(f"stages.{stage_id}.batch", batch)
    requested_batch = int(resolved_batch.value)
    resolution = resolve_batch(requested_batch, profile)
    return {
        "instances": instances,
        "instances_source": resolved_instances.source,
        "parallel_profile": profile.name,
        "parallel": {
            "tp": profile.tp,
            "cp": profile.cp,
            "pp": profile.pp,
            "dp_shard": profile.dp_shard,
            "dp_replicate": profile.dp_replicate,
            "ep": profile.ep,
            "sequence_parallel": profile.sequence_parallel,
        },
        "requested_batch": resolution.requested,
        "effective_batch": resolution.effective,
        "batch_source": resolved_batch.source,
    }


def _remove_stage_resource(session: WizardSession, stage_id: str) -> None:
    resources = _mapping_copy(session.state.collection("stage_resources"))
    resources.pop(stage_id, None)
    session.state.set_collection("stage_resources", resources)
    batches = _mapping_copy(session.state.collection("stage_batches"))
    batch_path = STATIC_MODEL_BATCH_PATHS.get(stage_id)
    if batch_path is not None:
        batches.pop(batch_path, None)
    session.state.set_collection("stage_batches", batches)


def _configure_stage_resource(
    session: WizardSession,
    resolver: DefaultsResolver,
    model: Any,
    stage_id: str,
    *,
    action: str,
    batch_default: int,
) -> Any:
    defaults = _stage_resource_defaults(
        session,
        resolver,
        stage_id,
        batch=batch_default,
    )
    registry = _resource_registry(session, resolver)
    strategy = CANONICAL_STAGE_STRATEGIES[stage_id]
    if action == "customize":
        profile = _profile_prompt(session, registry, stage_id, model)
        if profile is BACK:
            return BACK
        if strategy == "single":
            instances = 1
        else:
            instances = session.integer(
                f"stages.{stage_id}.instances",
                "Independent model instances/workers:",
                default=int(defaults["instances"]),
                minimum=1,
            )
            if instances is BACK:
                return BACK
        batch_unit = profile.batch_unit
        requested_batch = session.integer(
            f"stages.{stage_id}.batch",
            f"Local/micro batch size (minimum and scheduling unit: {batch_unit}):",
            default=int(defaults["effective_batch"]),
            minimum=batch_unit,
        )
        if requested_batch is BACK:
            return BACK
    else:
        profile, issues = _compatible_default_profile(
            session,
            registry,
            stage_id,
            model,
        )
        if profile is None:
            _print_parallel_issues(issues)
            if session.guided:
                raise SetupError(
                    f"No configured parallel profile is compatible with {stage_id}. "
                    "Supply a compatible profile in --defaults or resume with --full."
                )
            profile = _profile_prompt(session, registry, stage_id, model)
            if profile is BACK:
                return BACK
        instances = defaults["instances"]
        requested_batch = defaults["requested_batch"]

    resolution = resolve_batch(int(requested_batch), profile)
    if resolution.adjusted:
        print(
            f"  {stage_id} batch {resolution.requested} rounds up to "
            f"{resolution.effective} "
            f"(unit PP x DP-shard x DP-replicate={resolution.unit})."
        )
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    resource = StageResources(
        stage_id=stage_id,
        strategy=str(strategy),
        instances=int(instances),
        profile=profile,
        gpus_per_node=gpus_per_node,
    )
    summary = allocation_summary(resource, gpus_per_node=gpus_per_node)
    print(
        f"  {stage_id}: {summary.instances} instance(s), "
        f"{summary.gpus_per_instance} GPU/instance, {summary.task_count} task(s), "
        f"{summary.nodes} node(s)."
    )
    resources = _mapping_copy(session.state.collection("stage_resources"))
    resources[stage_id] = {
        "strategy": resource.strategy,
        "instances": resource.instances,
        "resource": resource.resource,
        "gpus_per_node": gpus_per_node,
        "profile_name": profile.name,
    }
    batches = _mapping_copy(session.state.collection("stage_batches"))
    batches[STATIC_MODEL_BATCH_PATHS[stage_id]] = resolution.effective
    session.state.set_collection("parallel_profiles", registry.to_dict())
    session.state.set_collection("stage_resources", resources)
    session.state.set_collection("stage_batches", batches)
    session.state.set_field(
        f"stages.{stage_id}.batch",
        resolution.effective,
        source="user" if action == "customize" else str(defaults["batch_source"]),
        requested=resolution.requested,
        effective=resolution.effective,
        dependencies=(f"profiles.{profile.name}",),
    )
    return resource


def depth_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    inventory = context["model"].inventory
    pruning = _pruning_payload(session.state)
    granularity = str(
        resolver.resolve_default(
            "pruning.depth_granularity",
            pruning.get("depth_granularity", "subblock"),
        ).value
    )
    count = inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
    remove = min(
        int(
            resolver.resolve_default(
                "pruning.depth_remove",
                pruning.get("depth_remove", min(4, count - 1)),
            ).value
        ),
        count - 1,
    )
    samples = int(
        resolver.resolve_default(
            "pruning.depth_importance_samples",
            pruning.get("depth_importance_samples", 128),
        ).value
    )
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "depth_importance",
        batch=8,
    )
    action = _section_action(
        session,
        "depth",
        "Configure depth pruning and its importance-evaluation resources.",
        {
            "granularity": granularity,
            "maximum_removed": remove,
            "eval_samples": samples,
            "sequence_length": session.state.get_field("data.sequence_length", 4096),
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        granularity = session.select(
            "pruning.depth_granularity",
            "Depth pruning granularity:",
            _depth_granularity_choices(inventory),
            default=granularity,
        )
        if granularity is BACK:
            return False
        count = inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        remove = session.integer(
            "pruning.depth_remove",
            (
                "Maximum number to remove "
                f"(network has {count} "
                f"{'sublayers' if granularity == 'subblock' else 'layers'}):"
            ),
            default=min(remove, count - 1),
            minimum=0,
            maximum=count - 1,
        )
        if remove is BACK:
            return False
        if int(remove) > 0:
            samples = session.integer(
                "pruning.depth_importance_samples",
                "Depth-importance evaluation samples:",
                default=samples,
                minimum=1,
            )
            if samples is BACK:
                return False
    pruning["depth_granularity"] = str(granularity)
    pruning["depth_remove"] = int(remove)
    pruning["depth_importance_samples"] = int(samples)
    session.state.set_collection("pruning", pruning)
    if int(remove) == 0:
        _remove_stage_resource(session, "depth_importance")
        return True
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "depth_importance",
        action=str(action),
        batch_default=8,
    )
    return configured is not BACK


def _normalized_axis_values(axis: Any, raw_values: Any) -> list[int]:
    values = sorted({int(value) for value in (raw_values or ())}, reverse=True)
    legal_values = {int(value) for value in axis.values}
    invalid = sorted(set(values) - legal_values, reverse=True)
    if invalid:
        raise SetupError(
            f"{axis.label} contains unsupported sizes {invalid}; "
            f"choose from {sorted(legal_values, reverse=True)}."
        )
    return values


def _axis_selection_validation(
    axis: Any,
    raw_values: Any,
    *,
    require_reduced: bool,
) -> bool | str:
    values = _normalized_axis_values(axis, raw_values)
    if not values:
        return "Select at least one size."
    if require_reduced and not any(value < int(axis.teacher_value) for value in values):
        return (
            "Select at least one size smaller than the teacher size. "
            "At least one pruning axis must actually reduce a dimension."
        )
    return True


def width_axes_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    pruning = _pruning_payload(session.state)
    inventory = context["model"].inventory
    current_axes = _mapping_copy(pruning.get("axes"))
    defaults = {
        axis.axis_id: list(
            resolver.resolve_default(
                f"pruning.axes.{axis.axis_id}.values",
                _mapping_copy(current_axes.get(axis.axis_id)).get(
                    "values",
                    _default_axis_values(axis),
                ),
            ).value
        )
        for axis in inventory.axes
    }
    action = _section_action(
        session,
        "width_axes",
        "Choose the legal values searched for every model width axis.",
        defaults,
    )
    if action is BACK:
        return False
    defaults_are_nonempty = all(
        bool(_normalized_axis_values(axis, defaults[axis.axis_id])) for axis in inventory.axes
    )
    defaults_reduce_a_dimension = any(
        any(
            value < int(axis.teacher_value)
            for value in _normalized_axis_values(axis, defaults[axis.axis_id])
        )
        for axis in inventory.axes
    )
    if action == "defaults" and (not defaults_are_nonempty or not defaults_reduce_a_dimension):
        print(
            "  Resolved width-axis defaults are not a pruning search: every axis "
            "needs a selected size and at least one axis needs a reduced size. "
            "Customize the selections."
        )
        action = "customize"
    axes = {}
    has_reduced_axis = False
    for index, axis in enumerate(inventory.axes):
        selected = defaults[axis.axis_id]
        if action == "customize":
            require_reduced = index == len(inventory.axes) - 1 and not has_reduced_axis
            selected = session.checkbox(
                f"pruning.axes.{axis.axis_id}",
                f"Values for {axis.label}:",
                [(str(value), value) for value in axis.values],
                defaults=selected,
                validate=lambda values, axis=axis, require_reduced=require_reduced: (
                    _axis_selection_validation(
                        axis,
                        values,
                        require_reduced=require_reduced,
                    )
                ),
            )
            if selected is BACK:
                return False
        selected_values = _normalized_axis_values(axis, selected)
        enabled = any(value < int(axis.teacher_value) for value in selected_values)
        has_reduced_axis = has_reduced_axis or enabled
        axes[axis.axis_id] = {
            "enabled": enabled,
            "teacher_value": axis.teacher_value,
            "values": selected_values,
            "alignment": axis.alignment,
        }
        if not enabled:
            print(
                f"  {axis.label}: teacher size only; activation hook and "
                "width/slicing sanity checks disabled."
            )
    if not has_reduced_axis:
        raise SetupError(
            "Width-axis selections do not prune any dimension. "
            "Select at least one size smaller than its teacher size."
        )
    pruning["axes"] = axes
    session.state.set_collection("pruning", pruning)
    return True


def width_importance_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    pruning = _pruning_payload(session.state)
    samples = int(
        resolver.resolve_default(
            "pruning.width_importance_samples",
            pruning.get("width_importance_samples", 32768),
        ).value
    )
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "width_importance",
        batch=8,
    )
    action = _section_action(
        session,
        "width_importance",
        "Configure width-importance evaluation and its resources.",
        {
            "eval_samples": samples,
            "sequence_length": session.state.get_field("data.sequence_length", 4096),
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        samples = session.integer(
            "pruning.width_importance_samples",
            "Width-importance evaluation samples:",
            default=samples,
            minimum=1,
        )
        if samples is BACK:
            return False
    pruning["width_importance_samples"] = int(samples)
    session.state.set_collection("pruning", pruning)
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "width_importance",
        action=str(action),
        batch_default=8,
    )
    return configured is not BACK


def sort_sanity_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    pruning = _pruning_payload(session.state)
    enabled = bool(
        resolver.resolve_default("pruning.sort_sanity", pruning.get("sort_sanity", False)).value
    )
    samples = int(
        resolver.resolve_default(
            "pruning.sort_sanity_samples",
            pruning.get("sort_sanity_samples", 128),
        ).value
    )
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "sort_sanity",
        batch=8,
    )
    action = _section_action(
        session,
        "sort_sanity",
        "Optionally validate that sorting preserves model quality.",
        {
            "enabled": enabled,
            "eval_samples": samples,
            "sequence_length": session.state.get_field("data.sequence_length", 4096),
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        enabled = session.confirm(
            "pruning.sort_sanity",
            "Run sorting sanity evaluation?",
            default=enabled,
        )
        if enabled is BACK:
            return False
        if enabled:
            samples = session.integer(
                "pruning.sort_sanity_samples",
                "Sorting-sanity evaluation samples:",
                default=samples,
                minimum=1,
            )
            if samples is BACK:
                return False
    pruning["sort_sanity"] = bool(enabled)
    pruning["sort_sanity_samples"] = int(samples)
    session.state.set_collection("pruning", pruning)
    if not enabled:
        _remove_stage_resource(session, "sort_sanity")
        return True
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "sort_sanity",
        action=str(action),
        batch_default=8,
    )
    return configured is not BACK


def width_sanity_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    pruning = _pruning_payload(session.state)
    if not bool(pruning.get("sort_sanity", False)):
        pruning["width_sanity"] = False
        pruning["slicing_sanity"] = False
        session.state.set_collection("pruning", pruning)
        _remove_stage_resource(session, "width_sanity")
        return True

    enabled = bool(
        resolver.resolve_default("pruning.width_sanity", pruning.get("width_sanity", False)).value
    )
    samples = int(
        resolver.resolve_default(
            "pruning.width_sanity_samples",
            pruning.get("width_sanity_samples", 128),
        ).value
    )
    layer_count = int(
        resolver.resolve_default(
            "pruning.width_sanity_layer_count",
            pruning.get("width_sanity_layer_count", 3),
        ).value
    )
    targets_per_axis = int(
        resolver.resolve_default(
            "pruning.width_sanity_targets_per_axis",
            pruning.get("width_sanity_targets_per_axis", 2),
        ).value
    )
    reduced_target_counts = {}
    for axis_id, raw_axis in _mapping_copy(pruning.get("axes")).items():
        axis = _mapping_copy(raw_axis)
        if not bool(axis.get("enabled", False)):
            continue
        teacher = int(axis.get("teacher_value", 0))
        reduced_target_counts[axis_id] = len(
            {int(value) for value in axis.get("values", ()) if int(value) < teacher}
        )
    max_targets_per_axis = max(reduced_target_counts.values(), default=1)
    targets_per_axis = min(targets_per_axis, max_targets_per_axis)
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "width_sanity",
        batch=8,
    )
    action = _section_action(
        session,
        "width_sanity",
        "Optionally compare sorted, random, and reverse reduced-width models.",
        {
            "enabled": enabled,
            "eval_samples": samples,
            "representative_layers": layer_count,
            "targets_per_axis": targets_per_axis,
            "selected_reduced_targets": reduced_target_counts,
            "physical_realization": False,
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        enabled = session.confirm(
            "pruning.width_sanity",
            "Run width sanity evaluation?",
            default=enabled,
        )
        if enabled is BACK:
            return False
        if enabled:
            samples = session.integer(
                "pruning.width_sanity_samples",
                "Width-sanity evaluation samples:",
                default=samples,
                minimum=1,
            )
            layer_count = session.integer(
                "pruning.width_sanity_layer_count",
                "Representative layers to check:",
                default=layer_count,
                minimum=1,
            )
            targets_per_axis = session.integer(
                "pruning.width_sanity_targets_per_axis",
                (
                    "Reduced-width targets per pruned axis "
                    "(teacher excluded; each axis uses at most its selected values):"
                ),
                default=targets_per_axis,
                minimum=1,
                maximum=max_targets_per_axis,
            )
            if BACK in (samples, layer_count, targets_per_axis):
                return False
    pruning["width_sanity"] = bool(enabled)
    pruning["width_sanity_samples"] = int(samples)
    pruning["width_sanity_layer_count"] = int(layer_count)
    pruning["width_sanity_targets_per_axis"] = int(targets_per_axis)
    if not enabled:
        pruning["slicing_sanity"] = False
    session.state.set_collection("pruning", pruning)
    if not enabled:
        _remove_stage_resource(session, "width_sanity")
        return True
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "width_sanity",
        action=str(action),
        batch_default=8,
    )
    return configured is not BACK


def slicing_sanity_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    del context
    pruning = _pruning_payload(session.state)
    if not bool(pruning.get("sort_sanity", False)) or not bool(pruning.get("width_sanity", False)):
        pruning["slicing_sanity"] = False
        session.state.set_collection("pruning", pruning)
        return True

    enabled = bool(
        resolver.resolve_default(
            "pruning.slicing_sanity", pruning.get("slicing_sanity", False)
        ).value
    )
    action = _section_action(
        session,
        "slicing_sanity",
        "Optionally verify dynamic slicing against physical materialization.",
        {
            "enabled": enabled,
            "physical_realization_for_width_sanity": enabled,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        enabled = session.confirm(
            "pruning.slicing_sanity",
            "Run slicing sanity evaluation?",
            default=enabled,
        )
        if enabled is BACK:
            return False
    pruning["slicing_sanity"] = bool(enabled)
    session.state.set_collection("pruning", pruning)
    return True


def bypass_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    pruning = _pruning_payload(session.state)
    bypass = dict(pruning.get("bypass") or {})
    enabled = bool(
        resolver.resolve_default("pruning.bypass.enabled", bypass.get("enabled", True)).value
    )
    granularity = str(
        resolver.resolve_default(
            "pruning.bypass.granularity",
            bypass.get("granularity", "subblock"),
        ).value
    )
    samples = int(
        resolver.resolve_default("pruning.bypass.samples", bypass.get("samples", 4096)).value
    )
    grad_accumulation_steps = int(
        resolver.resolve_default(
            "pruning.bypass.grad_accumulation_steps",
            bypass.get("grad_accumulation_steps", 1),
        ).value
    )
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "bypass",
        batch=int(bypass.get("batch_size", 8)),
    )
    action = _section_action(
        session,
        "bypass",
        "Configure optional local bypass distillation and its resources.",
        {
            "enabled": enabled,
            "granularity": granularity,
            "samples": samples,
            "gradient_accumulation_steps": grad_accumulation_steps,
            "sequence_length": session.state.get_field("data.sequence_length", 4096),
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        enabled = session.confirm(
            "pruning.bypass.enabled",
            "Run local bypass distillation?",
            default=enabled,
        )
        if enabled is BACK:
            return False
        if enabled:
            granularity = session.select(
                "pruning.bypass.granularity",
                "Bypass granularity:",
                [("Sublayer", "subblock"), ("Whole block", "block")],
                default=granularity,
            )
            samples = session.integer(
                "pruning.bypass.samples",
                "Bypass samples:",
                default=samples,
                minimum=1,
            )
            grad_accumulation_steps = session.integer(
                "pruning.bypass.grad_accumulation_steps",
                "Gradient accumulation steps:",
                default=grad_accumulation_steps,
                minimum=1,
            )
            if BACK in (granularity, samples, grad_accumulation_steps):
                return False
    bypass.update(
        {
            "enabled": bool(enabled),
            "granularity": str(granularity),
            "samples": int(samples),
            "grad_accumulation_steps": int(grad_accumulation_steps),
            "sequence_length": int(session.state.get_field("data.sequence_length", 4096)),
        }
    )
    pruning["bypass"] = bypass
    session.state.set_collection("pruning", pruning)
    if not enabled:
        _remove_stage_resource(session, "bypass")
        return True
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "bypass",
        action=str(action),
        batch_default=int(bypass.get("batch_size", 8)),
    )
    return configured is not BACK


def replacement_scoring_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    pruning = _pruning_payload(session.state)
    inventory = context["model"].inventory
    counts = count_candidate_options(
        context["model"].config,
        inventory,
        _mapping_copy(pruning.get("axes")),
    )
    granularity = str(
        resolver.resolve_default(
            "pruning.replacement_granularity",
            pruning.get(
                "replacement_granularity",
                pruning.get("depth_granularity", "subblock"),
            ),
        ).value
    )
    samples = int(
        resolver.resolve_default(
            "pruning.replacement_samples",
            pruning.get("replacement_samples", 128),
        ).value
    )
    resource_defaults = _stage_resource_defaults(
        session,
        resolver,
        "replacement_scoring",
        batch=8,
    )
    action = _section_action(
        session,
        "replacement_scoring",
        "Configure replace-one scoring and its evaluation resources.",
        {
            "granularity": granularity,
            "eval_samples": samples,
            "sequence_length": session.state.get_field("data.sequence_length", 4096),
            "resources": resource_defaults,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        granularity = session.select(
            "pruning.replacement_granularity",
            "Replace and score:",
            _replacement_granularity_choices(counts),
            default=granularity,
        )
        samples = session.integer(
            "pruning.replacement_samples",
            "Replacement-scoring evaluation samples:",
            default=samples,
            minimum=1,
        )
        if BACK in (granularity, samples):
            return False
    pruning["replacement_granularity"] = str(granularity)
    pruning["replacement_samples"] = int(samples)
    session.state.set_collection("pruning", pruning)
    configured = _configure_stage_resource(
        session,
        resolver,
        context["model"],
        "replacement_scoring",
        action=str(action),
        batch_default=8,
    )
    return configured is not BACK


def pre_mip_stages_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    current_registry = ResourceProfileRegistry.from_dict(
        session.state.collection("parallel_profiles") or {}
    )
    default_registry = ResourceProfileRegistry.from_dict(
        resolver.resolve_default("profiles", {}).value or {}
    )
    resources = _mapping_copy(session.state.collection("stage_resources"))
    batches = _mapping_copy(session.state.collection("stage_batches"))
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    preview_profile = (
        default_registry.get(default_registry.names()[0])
        if default_registry.names()
        else ParallelProfile(STATIC_MODEL_STAGES[0])
    )
    preview = {}
    for stage_id in STATIC_MODEL_STAGES:
        if stage_id == "bypass" and not (session.state.collection("pruning") or {}).get(
            "bypass", {}
        ).get("enabled", False):
            continue
        strategy = CANONICAL_STAGE_STRATEGIES[stage_id]
        instances = (
            1
            if strategy == "single"
            else int(
                resolver.resolve_default(
                    f"stages.{stage_id}.instances",
                    gpus_per_node,
                ).value
            )
        )
        requested_batch = int(resolver.resolve_default(f"stages.{stage_id}.batch", 8).value)
        batch = resolve_batch(requested_batch, preview_profile)
        preview[stage_id] = {
            "instances": instances,
            "parallel": {
                "tp": preview_profile.tp,
                "cp": preview_profile.cp,
                "pp": preview_profile.pp,
                "dp_shard": preview_profile.dp_shard,
                "dp_replicate": preview_profile.dp_replicate,
                "ep": preview_profile.ep,
            },
            "requested_batch": batch.requested,
            "effective_batch": batch.effective,
        }
    action = _section_action(
        session,
        "pre_mip_stages",
        "Each stage has canonical execution behavior and independent parallelism and batch settings.",
        preview,
    )
    if action is BACK:
        return False
    registry = default_registry if action == "defaults" else current_registry
    if action == "defaults":
        for stage_id in STATIC_MODEL_STAGES:
            resources.pop(stage_id, None)
            batches.pop(STATIC_MODEL_BATCH_PATHS[stage_id], None)
    customize_all = action == "customize"
    for stage_id in STATIC_MODEL_STAGES:
        if stage_id == "bypass" and not (session.state.collection("pruning") or {}).get(
            "bypass", {}
        ).get("enabled", False):
            continue
        customize = customize_all
        if customize_all:
            customize = session.confirm(
                f"stages.{stage_id}.customize",
                f"Customize {stage_id}? (No accepts the shown defaults)",
                default=False,
            )
            if customize is BACK:
                return False
        if customize:
            profile = _profile_prompt(session, registry, stage_id, context["model"])
            if profile is BACK:
                return False
            strategy = CANONICAL_STAGE_STRATEGIES[stage_id]
            if strategy == "single":
                instances = 1
            else:
                instances = session.integer(
                    f"stages.{stage_id}.instances",
                    "Independent model instances/workers:",
                    default=gpus_per_node,
                    minimum=1,
                )
                if instances is BACK:
                    return False
            requested_batch = session.integer(
                f"stages.{stage_id}.batch",
                (f"Local/micro batch size (minimum and scheduling unit: {profile.batch_unit}):"),
                default=resolve_batch(8, profile).effective,
                minimum=profile.batch_unit,
            )
            if requested_batch is BACK:
                return False
        else:
            profile = (
                registry.reuse(registry.names()[0], consumer=stage_id)
                if registry.names()
                else registry.create(ParallelProfile(stage_id), consumer=stage_id)
            )
            strategy = CANONICAL_STAGE_STRATEGIES[stage_id]
            instances = (
                1
                if strategy == "single"
                else int(
                    resolver.resolve_default(
                        f"stages.{stage_id}.instances",
                        gpus_per_node,
                    ).value
                )
            )
            requested_batch = int(resolver.resolve_default(f"stages.{stage_id}.batch", 8).value)
        resolution = resolve_batch(int(requested_batch), profile)
        if resolution.adjusted:
            print(
                f"  {stage_id} batch {resolution.requested} rounds up to "
                f"{resolution.effective} "
                f"(unit PP x DP-shard x DP-replicate={resolution.unit})."
            )
        resource = StageResources(
            stage_id=stage_id,
            strategy=str(strategy),
            instances=int(instances),
            profile=profile,
            gpus_per_node=gpus_per_node,
        )
        summary = allocation_summary(resource, gpus_per_node=gpus_per_node)
        print(
            f"  {stage_id}: {summary.instances} instance(s), "
            f"{summary.gpus_per_instance} GPU/instance, {summary.task_count} task(s), "
            f"{summary.nodes} node(s)."
        )
        resources[stage_id] = {
            "strategy": resource.strategy,
            "instances": resource.instances,
            "resource": resource.resource,
            "gpus_per_node": gpus_per_node,
            "profile_name": profile.name,
        }
        batches[STATIC_MODEL_BATCH_PATHS[stage_id]] = resolution.effective
        session.state.set_field(
            f"stages.{stage_id}.batch",
            resolution.effective,
            source="user" if customize else "builtin",
            requested=resolution.requested,
            effective=resolution.effective,
            dependencies=(f"profiles.{profile.name}",),
        )
    session.state.set_collection("parallel_profiles", registry.to_dict())
    session.state.set_collection("stage_resources", resources)
    session.state.set_collection("stage_batches", batches)
    return True


def _mapping_copy(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _identifier_validation(value: str) -> bool | str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not value or any(character not in allowed for character in value):
        return "Use only letters, digits, '_' and '-'."
    return True


def _default_vllm_topology(resolver: DefaultsResolver) -> dict[str, Any]:
    topology = {
        name: int(resolver.resolve_default(f"vllm.topology.{name}", 1).value)
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    }
    topology["enable_expert_parallel"] = bool(
        resolver.resolve_default("vllm.topology.enable_expert_parallel", False).value
    )
    topology["gpu_group_size"] = (
        topology["tensor_parallel_size"]
        * topology["pipeline_parallel_size"]
        * topology["data_parallel_size"]
        * topology["prefill_context_parallel_size"]
    )
    topology["distributed_executor_backend"] = str(
        resolver.resolve_default("vllm.topology.distributed_executor_backend", "mp").value
    )
    return topology


def _default_vllm_measurement(
    resolver: DefaultsResolver,
    *,
    sequence_length: int,
) -> dict[str, Any]:
    concurrency = int(
        resolver.resolve_default(
            "vllm.max_num_seqs",
            resolver.resolve_default("vllm.batch_size", 1).value,
        ).value
    )
    granularity = str(resolver.resolve_default("vllm.granularity", "subblock").value)
    topology = _default_vllm_topology(resolver)
    return {
        "prefill_seq_len": int(
            resolver.resolve_default("vllm.prefill_seq_len", sequence_length).value
        ),
        "generation_seq_len": int(resolver.resolve_default("vllm.generation_seq_len", 1024).value),
        "batch_size": concurrency,
        "max_num_seqs": concurrency,
        "granularity": granularity,
        "runtime_stats": {
            "granularity": granularity,
            "max_num_seqs": concurrency,
            "topology": topology,
        },
    }


def _default_serving_workload(
    resolver: DefaultsResolver,
    *,
    sequence_length: int,
) -> dict[str, int]:
    measurement = _default_vllm_measurement(
        resolver,
        sequence_length=sequence_length,
    )
    return {
        "prefill_seq_len": int(measurement["prefill_seq_len"]),
        "generation_seq_len": int(measurement["generation_seq_len"]),
        "batch_size": int(measurement["batch_size"]),
        "max_num_seqs": int(measurement["max_num_seqs"]),
    }


def _serving_workload_label(name: str, setting: Mapping[str, Any]) -> str:
    return (
        f"{name} — ISL {setting['prefill_seq_len']}, "
        f"OSL {setting['generation_seq_len']}, "
        f"concurrency {setting['max_num_seqs']}"
    )


def _serving_workload_prompt(
    session: WizardSession,
    prompt_id: str,
    *,
    default_name: str,
    default_workload: Mapping[str, Any],
    existing_names: set[str],
) -> Any:
    while True:
        name = session.text(
            f"{prompt_id}.id",
            "Serving workload ID:",
            default=default_name,
            validate=_identifier_validation,
        )
        if name is BACK:
            return BACK
        name = str(name)
        if name in existing_names:
            print(f"  Serving workload {name!r} already exists.")
            continue
        break
    values: dict[str, int] = {}
    for key, label in (
        ("prefill_seq_len", "Input sequence length (ISL):"),
        ("generation_seq_len", "Output sequence length (OSL):"),
        ("max_num_seqs", "Concurrency:"),
    ):
        value = session.integer(
            f"{prompt_id}.{name}.{key}",
            label,
            default=int(default_workload[key]),
            minimum=1,
        )
        if value is BACK:
            return BACK
        values[key] = int(value)
    values["batch_size"] = values["max_num_seqs"]
    return name, values


def serving_workloads_section(
    session: WizardSession,
    resolver: DefaultsResolver,
    context: dict,
) -> bool:
    default_workload = _default_serving_workload(
        resolver,
        sequence_length=int(session.state.get_field("data.sequence_length", 4096)),
    )
    action = _section_action(
        session,
        "serving_workloads",
        (
            "Define the serving workloads used for analytical memory budgets. "
            "vLLM statistics can measure any subset of these workloads later."
        ),
        {
            "serving-default": {
                "input_sequence_length": default_workload["prefill_seq_len"],
                "output_sequence_length": default_workload["generation_seq_len"],
                "concurrency": default_workload["max_num_seqs"],
            }
        },
    )
    if action is BACK:
        return False
    if action == "defaults":
        session.state.set_collection(
            "serving_workloads",
            {"serving-default": default_workload},
        )
        return True

    workloads: OrderedDict[str, Any] = OrderedDict()
    while True:
        result = _serving_workload_prompt(
            session,
            "serving_workloads",
            default_name=("serving-default" if not workloads else f"serving-{len(workloads) + 1}"),
            default_workload=default_workload,
            existing_names=set(workloads),
        )
        if result is BACK:
            return False
        name, workload = result
        workloads[name] = workload
        add_more = session.confirm(
            "serving_workloads.add",
            "Add another serving workload?",
            default=False,
        )
        if add_more is BACK:
            return False
        if not add_more:
            break
    session.state.set_collection("serving_workloads", workloads)
    return True


def _set_vllm_stage_resource(
    session: WizardSession,
    *,
    instances: int,
    topology: Mapping[str, Any],
    source: str,
) -> None:
    allocation_mesh = vllm_topology_to_mesh(topology)
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    resources = _mapping_copy(session.state.collection("stage_resources"))
    resources["vllm_stats"] = {
        "strategy": "sharded",
        "instances": int(instances),
        "resource": "gpu",
        "gpus_per_node": gpus_per_node,
        "parallel": {
            **allocation_mesh.as_dict(),
            "sequence_parallel": False,
        },
    }
    session.state.set_collection("stage_resources", resources)
    session.state.set_field(
        "stages.vllm_stats.instances",
        int(instances),
        source=source,
    )


def vllm_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    enabled_default = bool(resolver.resolve_default("vllm.enabled", False).value)
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    resolved_instances = resolver.resolve_default(
        "stages.vllm_stats.instances",
        gpus_per_node,
    )
    default_instances = int(resolved_instances.value)
    default_measurement_settings = _default_vllm_measurement(
        resolver,
        sequence_length=int(session.state.get_field("data.sequence_length", 4096)),
    )
    default_topology = default_measurement_settings["runtime_stats"]["topology"]
    workloads = OrderedDict(_mapping_copy(session.state.collection("serving_workloads")).items())
    if not workloads:
        workloads["serving-default"] = _default_serving_workload(
            resolver,
            sequence_length=int(session.state.get_field("data.sequence_length", 4096)),
        )
    pruning = _pruning_payload(session.state)
    counts = count_candidate_options(
        context["model"].config,
        context["model"].inventory,
        _mapping_copy(pruning.get("axes")),
    )
    granularity_choices = _vllm_granularity_choices(counts)
    granularity_labels = {value: label for label, value in granularity_choices}
    preview = {
        "enabled": enabled_default,
        "available_serving_workloads": [
            _serving_workload_label(name, setting) for name, setting in workloads.items()
        ],
    }
    if enabled_default:
        default_name, default_workload = next(iter(workloads.items()))
        preview["measurement"] = {
            "workload": default_name,
            "input_sequence_length": default_workload["prefill_seq_len"],
            "output_sequence_length": default_workload["generation_seq_len"],
            "concurrency": default_workload["max_num_seqs"],
            "granularity": granularity_labels.get(
                default_measurement_settings["granularity"],
                default_measurement_settings["granularity"],
            ),
            "gpus_per_task": default_topology["gpu_group_size"],
            "instances": default_instances,
        }
    action = _section_action(
        session,
        "vllm",
        "Define zero or more exact workload/topology measurement points.",
        preview,
    )
    if action is BACK:
        return False
    if action == "defaults":
        if not enabled_default:
            session.state.set_collection("vllm_measurements", {})
            _remove_stage_resource(session, "vllm_stats")
            return True
        default_name, default_workload = next(iter(workloads.items()))
        issues = validate_vllm_parallelism(
            default_topology,
            context["model"].inventory,
            pruning,
            stage_id=f"vllm_stats.{default_name}",
        )
        if issues:
            _print_parallel_issues(issues)
            default_topology = _vllm_topology_prompt(
                session,
                f"vllm.measurements.{default_name}.topology",
                default_topology,
                inventory=context["model"].inventory,
                pruning=pruning,
                stage_id=f"vllm_stats.{default_name}",
            )
            if default_topology is BACK:
                return False
        granularity = default_measurement_settings["granularity"]
        session.state.set_collection(
            "vllm_measurements",
            {
                default_name: {
                    **default_workload,
                    "granularity": granularity,
                    "runtime_stats": {
                        "granularity": granularity,
                        "max_num_seqs": default_workload["max_num_seqs"],
                        "topology": default_topology,
                    },
                }
            },
        )
        _set_vllm_stage_resource(
            session,
            instances=default_instances,
            topology=default_topology,
            source=resolved_instances.source,
        )
        return True

    enabled = session.confirm(
        "vllm.enabled",
        "Collect vLLM runtime statistics?",
        default=enabled_default,
    )
    if enabled is BACK:
        return False
    if not enabled:
        session.state.set_collection("vllm_measurements", {})
        _remove_stage_resource(session, "vllm_stats")
        return True

    measurements: OrderedDict[str, Any] = OrderedDict()
    instances: int | None = None
    while True:
        unused_workloads = OrderedDict(
            (name, setting) for name, setting in workloads.items() if name not in measurements
        )
        if unused_workloads:
            workload_choice = session.select(
                "vllm.measurement.workload",
                "Serving workload for this vLLM measurement:",
                [
                    *(
                        PromptChoice(
                            _serving_workload_label(name, setting),
                            name,
                            (
                                "Already selected for vLLM statistics"
                                if name in measurements
                                else None
                            ),
                        )
                        for name, setting in workloads.items()
                    ),
                    ("Create a new serving workload", _CREATE_SERVING_WORKLOAD),
                ],
                default=next(iter(unused_workloads)),
            )
        else:
            workload_choice = _CREATE_SERVING_WORKLOAD
        if workload_choice is BACK:
            return False
        if workload_choice == _CREATE_SERVING_WORKLOAD:
            previous_workload = workloads[next(reversed(workloads))]
            result = _serving_workload_prompt(
                session,
                "vllm.measurement.new_workload",
                default_name=f"serving-{len(workloads) + 1}",
                default_workload=previous_workload,
                existing_names=set(workloads),
            )
            if result is BACK:
                return False
            name, values = result
            workloads[name] = values
        else:
            name = str(workload_choice)
            values = _mapping_copy(workloads[name])
        granularity = session.select(
            f"vllm.measurements.{name}.granularity",
            "Measurement granularity:",
            granularity_choices,
            default=str(resolver.resolve_default("vllm.granularity", "subblock").value),
        )
        if granularity is BACK:
            return False
        topology = _vllm_topology_prompt(
            session,
            f"vllm.measurements.{name}.topology",
            default_topology,
            inventory=context["model"].inventory,
            pruning=pruning,
            stage_id=f"vllm_stats.{name}",
        )
        if topology is BACK:
            return False
        if instances is None:
            instances = session.integer(
                "stages.vllm_stats.instances",
                "Independent vLLM statistics tasks/instances:",
                default=default_instances,
                minimum=1,
            )
            if instances is BACK:
                return False
        measurements[str(name)] = {
            **values,
            "granularity": granularity,
            "runtime_stats": {
                "granularity": granularity,
                "max_num_seqs": values["max_num_seqs"],
                "topology": topology,
            },
        }
        add_more = session.confirm(
            "vllm.measurements.add", "Add another vLLM measurement?", default=False
        )
        if add_more is BACK:
            return False
        if not add_more:
            break
    session.state.set_collection("vllm_measurements", measurements)
    session.state.set_collection("serving_workloads", workloads)
    first_measurement = _mapping_copy(next(iter(measurements.values())))
    first_topology = _mapping_copy(
        _mapping_copy(first_measurement.get("runtime_stats")).get("topology")
    )
    _set_vllm_stage_resource(
        session,
        instances=int(instances),
        topology=first_topology,
        source="user",
    )
    return True


_MIP_RANKING_CHOICES = (
    ("Cosine embedding distance", "metrics.cosine_embedding_loss_hidden_states"),
    ("Language-model loss", "metrics.lm_loss"),
)

_MIP_AXIS_ALIASES = {
    "ffn_intermediate": "ffn.intermediate_size",
    "kv_heads": "num_key_value_heads",
    "q_heads_per_group": "q_per_group",
    "moe_experts": "n_routed_experts",
    "moe_expert_intermediate": "moe_intermediate_size",
    "moe_shared_expert_intermediate": "moe_shared_expert_intermediate_size",
    "moe_latent_dim": "moe.latent_dim",
    "moe_top_k": "num_experts_per_tok",
    "mamba_heads": "mamba_num_heads",
    "mamba_head_dim": "mamba_head_dim",
}


_mip_identifier_validation = _identifier_validation


def _mip_default_search_id(metric: str, value: Any) -> str:
    suffix = str(value).replace("%", "")
    slug = "".join(
        character if character.isalnum() or character in "_-" else "-" for character in suffix
    ).strip("-")
    return f"{metric}-{slug or 'target'}"


def _mip_scalar_validation(value: str) -> bool | str:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return "Enter a number, percentage, or value with a supported unit."
    if parsed is None or isinstance(parsed, (bool, list, tuple, Mapping)):
        return "Enter one scalar value, such as 70%, 22.5B, or 120GiB."
    return True


def _mip_parse_scalar(value: str) -> Any:
    return yaml.safe_load(str(value))


def _mip_range_lower_default(value: Any) -> str:
    text = str(value)
    if text.endswith("%"):
        try:
            return f"{max(1.0, float(text[:-1]) - 2.0):g}%"
        except ValueError:
            pass
    return text


def _mip_bound_prompt(
    session: WizardSession,
    prompt_id: str,
    label: str,
    *,
    default: Any,
) -> Any:
    mode = session.select(
        f"{prompt_id}.mode",
        f"{label} bound:",
        [
            ("Maximum", "max"),
            ("Range", "range"),
            ("Exact", "eq"),
        ],
        default="max",
    )
    if mode is BACK:
        return BACK
    if mode == "range":
        lower = session.text(
            f"{prompt_id}.minimum",
            f"{label} minimum:",
            default=_mip_range_lower_default(default),
            validate=_mip_scalar_validation,
        )
        upper = session.text(
            f"{prompt_id}.maximum",
            f"{label} maximum:",
            default=str(default),
            validate=_mip_scalar_validation,
        )
        if BACK in (lower, upper):
            return BACK
        return {"range": [_mip_parse_scalar(lower), _mip_parse_scalar(upper)]}
    value = session.text(
        f"{prompt_id}.value",
        f"{label} value:",
        default=str(default),
        validate=_mip_scalar_validation,
    )
    if value is BACK:
        return BACK
    return {str(mode): _mip_parse_scalar(value)}


def _mip_maximum_prompt(
    session: WizardSession,
    prompt_id: str,
    label: str,
    *,
    default: Any,
) -> Any:
    value = session.text(
        f"{prompt_id}.maximum",
        f"{label} maximum:",
        default=str(default),
        validate=_mip_scalar_validation,
    )
    if value is BACK:
        return BACK
    return {"max": _mip_parse_scalar(value)}


def _mip_workload_choices(
    workloads: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    return [
        (
            (
                f"{name} — ISL {setting['isl']}, OSL {setting['osl']}, "
                f"concurrency {setting['concurrency']}"
            ),
            str(name),
        )
        for name, setting in workloads.items()
    ]


def _mip_constraint_choices(
    *,
    moe: bool,
    workloads: Mapping[str, Any],
    runtime_workloads: Mapping[str, Any],
    include_axis_aggregates: bool,
    axis_ids: set[str],
) -> list[tuple[str, str]]:
    choices = [("Total parameters", "params")]
    if moe:
        choices.append(("Active parameters", "active_params"))
    if workloads:
        choices.append(("Memory at selected workloads", "memory"))
    if runtime_workloads:
        choices.append(("Runtime at selected measured workloads", "runtime"))
    if include_axis_aggregates:
        if "moe_experts" in axis_ids:
            choices.append(("Total expert slots across all layers", "experts"))
        if "kv_heads" in axis_ids:
            choices.append(("Total KV heads across all layers", "kv_heads"))
    return choices


def _mip_constraint_default(metric: str, configured_goal: Any) -> Any:
    return {
        "params": configured_goal,
        "active_params": "80%",
        "memory": "70%",
        "runtime": "75%",
        "experts": "75%",
        "kv_heads": "75%",
    }[metric]


def _mip_constraints_prompt(
    session: WizardSession,
    prompt_id: str,
    *,
    choices: list[tuple[str, str]],
    workloads: Mapping[str, Mapping[str, Any]],
    runtime_workloads: Mapping[str, Mapping[str, Any]],
    configured_goal: Any,
    defaults: list[str],
    require_one: bool,
) -> Any:
    selected = session.checkbox(
        f"{prompt_id}.metrics",
        "Constraints:",
        choices,
        defaults=defaults,
        validate=(
            (lambda values: bool(values) or "Select at least one constraint.")
            if require_one
            else None
        ),
    )
    if selected is BACK:
        return BACK
    labels = {value: title for title, value in choices}
    constraints: OrderedDict[str, Any] = OrderedDict()
    for metric in selected:
        metric = str(metric)
        default = _mip_constraint_default(metric, configured_goal)
        if metric not in {"memory", "runtime"}:
            prompt = (
                _mip_maximum_prompt if metric in {"params", "active_params"} else _mip_bound_prompt
            )
            bound = prompt(
                session,
                f"{prompt_id}.{metric}",
                labels[metric],
                default=default,
            )
            if bound is BACK:
                return BACK
            constraints[metric] = bound
            continue
        available_workloads = runtime_workloads if metric == "runtime" else workloads
        selected_workloads = session.checkbox(
            f"{prompt_id}.{metric}.workloads",
            f"{labels[metric]}:",
            _mip_workload_choices(available_workloads),
            defaults=list(available_workloads),
            validate=lambda values: bool(values) or "Select at least one workload.",
        )
        if selected_workloads is BACK:
            return BACK
        workload_bounds: OrderedDict[str, Any] = OrderedDict()
        for workload in selected_workloads:
            bound = _mip_maximum_prompt(
                session,
                f"{prompt_id}.{metric}.{workload}",
                f"{labels[metric]} at {workload}",
                default=default,
            )
            if bound is BACK:
                return BACK
            workload_bounds[str(workload)] = bound
        constraints[metric] = {"at": workload_bounds}
    return constraints


def _mip_axis_specs(inventory: Any, pruning: Mapping[str, Any]) -> list[dict[str, Any]]:
    configured = _mapping_copy(pruning.get("axes"))
    specs = []
    for axis in inventory.axes:
        if axis.axis_id == "hidden_width" or axis.axis_id not in _MIP_AXIS_ALIASES:
            continue
        setting = _mapping_copy(configured.get(axis.axis_id))
        values = list(dict.fromkeys(int(value) for value in setting.get("values") or axis.values))
        if not values or not bool(setting.get("enabled", False)):
            continue
        specs.append(
            {
                "axis_id": str(axis.axis_id),
                "mip_axis": _MIP_AXIS_ALIASES[axis.axis_id],
                "label": str(axis.label),
                "values": values,
                "teacher_value": int(axis.teacher_value),
            }
        )
    return specs


def _mip_scenario_domains(
    inventory: Any,
    pruning: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    maximum_depth = int(pruning.get("depth_remove", 0))
    depths = list(range(maximum_depth + 1))
    hidden = _mapping_copy(_mapping_copy(pruning.get("axes")).get("hidden_width"))
    embeddings = list(dict.fromkeys(int(value) for value in hidden.get("values") or ()))
    if not embeddings:
        teacher_width = next(
            (int(axis.teacher_value) for axis in inventory.axes if axis.axis_id == "hidden_width"),
            None,
        )
        if teacher_width is None:
            raise SetupError("The model inventory has no embedding-width domain for MIP.")
        embeddings = [teacher_width]
    return depths, embeddings


def _mip_variant_prompt(
    session: WizardSession,
    search_id: str,
    *,
    existing_ids: set[str],
    constraint_choices: list[tuple[str, str]],
    workloads: Mapping[str, Mapping[str, Any]],
    runtime_workloads: Mapping[str, Mapping[str, Any]],
    configured_goal: Any,
    axis_specs: list[dict[str, Any]],
) -> Any:
    while True:
        variant_id = session.text(
            f"mip.search.{search_id}.variant.id",
            "Variant ID:",
            default="restricted",
            validate=_mip_identifier_validation,
        )
        if variant_id is BACK:
            return BACK
        variant_id = str(variant_id)
        if variant_id == "baseline":
            print("  'baseline' is reserved for the main search without extra restrictions.")
            continue
        if variant_id in existing_ids:
            print(f"  Variant {variant_id!r} already exists.")
            continue
        break
    restriction_choices = [("Additional model constraints", "constraints")]
    if axis_specs:
        restriction_choices.append(("Pruning-axis restrictions", "axes"))
    restrictions = session.checkbox(
        f"mip.search.{search_id}.variant.{variant_id}.restrictions",
        "Variant restrictions:",
        restriction_choices,
        defaults=[restriction_choices[0][1]],
        validate=lambda values: bool(values) or "Select at least one restriction.",
    )
    if restrictions is BACK:
        return BACK
    variant: dict[str, Any] = {}
    if "constraints" in restrictions:
        constraints = _mip_constraints_prompt(
            session,
            f"mip.search.{search_id}.variant.{variant_id}.constraints",
            choices=constraint_choices,
            workloads=workloads,
            runtime_workloads=runtime_workloads,
            configured_goal=configured_goal,
            defaults=[constraint_choices[0][1]],
            require_one=True,
        )
        if constraints is BACK:
            return BACK
        variant["constraints"] = constraints
    if "axes" in restrictions:
        mode = session.select(
            f"mip.search.{search_id}.variant.{variant_id}.axes.mode",
            "Pruning-axis policy:",
            [
                ("Restrict values for selected axes; inherit all other axes", "restrict"),
                (
                    "Only selected width axes may change; keep other width axes at teacher",
                    "only",
                ),
            ],
            default="restrict",
        )
        if mode is BACK:
            return BACK
        axes = session.checkbox(
            f"mip.search.{search_id}.variant.{variant_id}.axes.selected",
            "Pruning axes:",
            [(spec["label"], spec["axis_id"]) for spec in axis_specs],
            defaults=[axis_specs[0]["axis_id"]],
            validate=lambda values: bool(values) or "Select at least one pruning axis.",
        )
        if axes is BACK:
            return BACK
        selected_axes = {str(axis_id) for axis_id in axes}
        axis_options: OrderedDict[str, Any] = OrderedDict()
        for spec in axis_specs:
            if spec["axis_id"] not in selected_axes:
                continue
            values = session.checkbox(
                (f"mip.search.{search_id}.variant.{variant_id}.axes.{spec['axis_id']}.values"),
                f"Allowed values for {spec['label']}:",
                [
                    (
                        f"{value}" + (" (teacher)" if value == spec["teacher_value"] else ""),
                        value,
                    )
                    for value in spec["values"]
                ],
                defaults=spec["values"],
                validate=lambda selected: bool(selected) or "Select at least one value.",
            )
            if values is BACK:
                return BACK
            axis_options[spec["mip_axis"]] = [int(value) for value in values]
        variant["search_space"] = {
            "axes_default": "teacher" if mode == "only" else "all",
            "axes": axis_options,
        }
    return variant_id, variant


def _mip_solution_estimate(
    *,
    variant_count: int,
    metric_count: int,
    embedding_count: int,
    depth_count: int,
    heterogeneous_per_solve: int,
    homogeneous_keep: int | str,
    axis_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    solve_count = variant_count * metric_count * embedding_count * depth_count
    if homogeneous_keep == "all":
        homogeneous_per_solve = 1
        for spec in axis_specs:
            homogeneous_per_solve *= len(spec["values"])
        homogeneous_label = f"at most about {homogeneous_per_solve} per solve"
    else:
        homogeneous_per_solve = int(homogeneous_keep)
        homogeneous_label = str(homogeneous_per_solve)
    candidate_upper_bound = solve_count * (heterogeneous_per_solve + homogeneous_per_solve)
    return {
        "concrete_solves": solve_count,
        "heterogeneous_per_solve": heterogeneous_per_solve,
        "homogeneous_per_solve": homogeneous_label,
        "candidate_origin_upper_bound": candidate_upper_bound,
    }


def _print_mip_search_review(
    search_id: str,
    *,
    constraints: Mapping[str, Any],
    depths: list[int],
    embeddings: list[int],
    objectives: list[str],
    variants: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> None:
    print(f"\n  Search {search_id!r} review:")
    print(
        "    Main budgets: "
        f"{yaml.safe_dump(_plain_review_value(constraints), sort_keys=False).strip()}"
    )
    print(f"    Depth selections: {depths}")
    print(f"    Embedding widths: {embeddings}")
    print(f"    Ranking metrics: {objectives}")
    print(f"    Variants: {list(variants) if variants else ['baseline']}")
    print(f"    Concrete solves: {estimate['concrete_solves']}")
    print(
        "    Requested candidates per solve: "
        f"{estimate['heterogeneous_per_solve']} heterogeneous + "
        f"{estimate['homogeneous_per_solve']} homogeneous"
    )
    print(
        "    Candidate upper bound before deduplication: "
        f"{estimate['candidate_origin_upper_bound']}"
    )
    print(
        "    MIP scores are comparable only within the same ranking metric, "
        "embedding width, and depth selection."
    )


def mip_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    inventory = context["model"].inventory
    pruning = _mapping_copy(session.state.collection("pruning"))
    serving_workloads = _mapping_copy(session.state.collection("serving_workloads"))
    measurements = _mapping_copy(session.state.collection("vllm_measurements"))
    workloads = OrderedDict(
        (
            str(name),
            {
                "isl": int(raw["prefill_seq_len"]),
                "osl": int(raw["generation_seq_len"]),
                "batch_size": int(raw["batch_size"]),
                "concurrency": int(raw["max_num_seqs"]),
            },
        )
        for name, raw in serving_workloads.items()
    )
    runtime_workloads = OrderedDict(
        (name, workloads[name]) for name in measurements if name in workloads
    )
    default_goal_metric = str(resolver.resolve_default("mip.goal_metric", "params").value)
    default_goal_value = resolver.resolve_default("mip.goal_value", "75%").value
    default_objective = str(
        resolver.resolve_default(
            "mip.objective", "metrics.cosine_embedding_loss_hidden_states"
        ).value
    )
    default_num_solutions = int(resolver.resolve_default("mip.num_solutions", 8).value)
    available_depths, available_embeddings = _mip_scenario_domains(inventory, pruning)
    teacher_embedding = next(
        (int(axis.teacher_value) for axis in inventory.axes if axis.axis_id == "hidden_width"),
        max(available_embeddings),
    )
    axis_specs = _mip_axis_specs(inventory, pruning)
    axis_ids = {str(axis.axis_id) for axis in inventory.axes}
    main_constraint_choices = _mip_constraint_choices(
        moe=bool(inventory.moe),
        workloads=workloads,
        runtime_workloads=runtime_workloads,
        include_axis_aggregates=False,
        axis_ids=axis_ids,
    )
    variant_constraint_choices = _mip_constraint_choices(
        moe=bool(inventory.moe),
        workloads=workloads,
        runtime_workloads=runtime_workloads,
        include_axis_aggregates=True,
        axis_ids=axis_ids,
    )
    available_main_metrics = {value for _, value in main_constraint_choices}
    if default_goal_metric not in available_main_metrics:
        default_goal_metric = "params"
    if default_objective not in {value for _, value in _MIP_RANKING_CHOICES}:
        default_objective = _MIP_RANKING_CHOICES[0][1]
    default_search_id = _mip_default_search_id(default_goal_metric, default_goal_value)
    default_estimate = _mip_solution_estimate(
        variant_count=1,
        metric_count=1,
        embedding_count=len(available_embeddings),
        depth_count=len(available_depths),
        heterogeneous_per_solve=default_num_solutions,
        homogeneous_keep=5,
        axis_specs=axis_specs,
    )
    action = _section_action(
        session,
        "mip",
        (
            "Create one or more searches. Each search shares main budgets, "
            "width/depth scenarios, ranking metrics, solution policy, and named variants."
        ),
        {
            "search": default_search_id,
            "main_budgets": {default_goal_metric: {"max": default_goal_value}},
            "depths": available_depths,
            "embedding_widths": available_embeddings,
            "ranking_metrics": [default_objective],
            "variants": ["baseline"],
            "heterogeneous_per_solve": default_num_solutions,
            "homogeneous_per_solve": 5,
            "candidate_origin_upper_bound": default_estimate["candidate_origin_upper_bound"],
        },
    )
    if action is BACK:
        return False

    if action == "defaults":
        constraint: Any = {"max": _mip_parse_scalar(str(default_goal_value))}
        if default_goal_metric in {"memory", "runtime"}:
            target_workloads = runtime_workloads if default_goal_metric == "runtime" else workloads
            if not target_workloads:
                raise SetupError(
                    f"Default MIP constraint {default_goal_metric!r} requires "
                    + (
                        "an enabled vLLM measurement."
                        if default_goal_metric == "runtime"
                        else "a serving workload."
                    )
                )
            constraint = {"at": dict.fromkeys(target_workloads, constraint)}
        session.state.set_collection(
            "mip_config",
            {
                "defaults": {},
                "workloads": workloads,
                "runs": {
                    default_search_id: {
                        "constraints": {default_goal_metric: constraint},
                        "objectives": [{"metric": default_objective, "direction": "minimize"}],
                        "search_space": {
                            "depth": available_depths,
                            "embedding": available_embeddings,
                        },
                        "solver": {
                            "backend": "auto",
                            "num_solutions": default_num_solutions,
                            "min_hamming_distance": 2,
                            "max_seconds_per_solution": 60,
                        },
                        "homogeneous": {
                            "enabled": True,
                            "keep": 5,
                            "rank_by": "objective",
                        },
                    }
                },
            },
        )
        session.state.set_collection("mip_search_estimates", {default_search_id: default_estimate})
        return True

    runs: OrderedDict[str, Any] = OrderedDict()
    estimates: OrderedDict[str, Any] = OrderedDict()
    while True:
        print("\n  MIP search step 1/8 — Search identity")
        while True:
            search_id = session.text(
                "mip.search.id",
                "Search ID:",
                default=default_search_id if not runs else f"search-{len(runs) + 1}",
                validate=_mip_identifier_validation,
            )
            if search_id is BACK:
                return False
            search_id = str(search_id)
            if search_id in runs:
                print(f"  Search {search_id!r} already exists.")
                continue
            break

        print("\n  MIP search step 2/8 — Main budgets")
        constraints = _mip_constraints_prompt(
            session,
            f"mip.search.{search_id}.budgets",
            choices=main_constraint_choices,
            workloads=workloads,
            runtime_workloads=runtime_workloads,
            configured_goal=default_goal_value,
            defaults=[default_goal_metric],
            require_one=True,
        )
        if constraints is BACK:
            return False

        print("\n  MIP search step 3/8 — Width and depth scenarios")
        depths = session.checkbox(
            f"mip.search.{search_id}.depths",
            "Depth removals:",
            [(str(depth), depth) for depth in available_depths],
            defaults=available_depths,
            validate=lambda values: bool(values) or "Select at least one depth.",
        )
        embeddings = session.checkbox(
            f"mip.search.{search_id}.embeddings",
            "Embedding widths:",
            [
                (
                    f"{width}" + (" (teacher)" if width == teacher_embedding else ""),
                    width,
                )
                for width in available_embeddings
            ],
            defaults=available_embeddings,
            validate=lambda values: bool(values) or "Select at least one embedding width.",
        )
        if BACK in (depths, embeddings):
            return False
        depths = [int(value) for value in depths]
        embeddings = [int(value) for value in embeddings]

        print("\n  MIP search step 4/8 — Ranking metrics")
        objectives = session.checkbox(
            f"mip.search.{search_id}.objectives",
            "Ranking metrics (each creates an independent solution pool):",
            list(_MIP_RANKING_CHOICES),
            defaults=[default_objective],
            validate=lambda values: bool(values) or "Select at least one ranking metric.",
        )
        if objectives is BACK:
            return False
        objectives = [str(value) for value in objectives]

        print("\n  MIP search step 5/8 — Solution types")
        solution_types = session.select(
            f"mip.search.{search_id}.solution_types",
            "Solution types:",
            [
                ("Heterogeneous only", "heterogeneous"),
                ("Heterogeneous and homogeneous", "both"),
            ],
            default="both",
        )
        if solution_types is BACK:
            return False

        print("\n  MIP search step 6/8 — Named variants")
        variants: OrderedDict[str, Any] = OrderedDict()
        add_variant = session.confirm(
            f"mip.search.{search_id}.variant.add",
            "Add a named variant in addition to baseline?",
            default=False,
        )
        if add_variant is BACK:
            return False
        while add_variant:
            variant_result = _mip_variant_prompt(
                session,
                search_id,
                existing_ids=set(variants),
                constraint_choices=variant_constraint_choices,
                workloads=workloads,
                runtime_workloads=runtime_workloads,
                configured_goal=default_goal_value,
                axis_specs=axis_specs,
            )
            if variant_result is BACK:
                return False
            variant_id, variant = variant_result
            variants[str(variant_id)] = variant
            add_variant = session.confirm(
                f"mip.search.{search_id}.variant.add_more",
                "Add another named variant?",
                default=False,
            )
            if add_variant is BACK:
                return False

        print("\n  MIP search step 7/8 — Candidate pool sizes")
        heterogeneous_per_solve = session.integer(
            f"mip.search.{search_id}.heterogeneous_per_solve",
            "Heterogeneous solutions requested per solve:",
            default=default_num_solutions,
            minimum=1,
        )
        if heterogeneous_per_solve is BACK:
            return False
        homogeneous_keep: int | str = 0
        if solution_types == "both":
            homogeneous_mode = session.select(
                f"mip.search.{search_id}.homogeneous.mode",
                "Homogeneous solutions retained per solve:",
                [
                    ("Top N", "top_n"),
                    ("All feasible homogeneous solutions", "all"),
                ],
                default="top_n",
            )
            if homogeneous_mode is BACK:
                return False
            if homogeneous_mode == "all":
                homogeneous_keep = "all"
            else:
                homogeneous_keep = session.integer(
                    f"mip.search.{search_id}.homogeneous.keep",
                    "Homogeneous solutions retained per solve:",
                    default=5,
                    minimum=1,
                )
                if homogeneous_keep is BACK:
                    return False

        explicit_variants: OrderedDict[str, Any] = OrderedDict()
        if variants:
            explicit_variants["baseline"] = {}
            explicit_variants.update(variants)
        estimate = _mip_solution_estimate(
            variant_count=max(1, len(explicit_variants)),
            metric_count=len(objectives),
            embedding_count=len(embeddings),
            depth_count=len(depths),
            heterogeneous_per_solve=int(heterogeneous_per_solve),
            homogeneous_keep=homogeneous_keep,
            axis_specs=axis_specs,
        )

        print("\n  MIP search step 8/8 — Review")
        _print_mip_search_review(
            search_id,
            constraints=constraints,
            depths=depths,
            embeddings=embeddings,
            objectives=objectives,
            variants=explicit_variants,
            estimate=estimate,
        )
        accept = session.confirm(
            f"mip.search.{search_id}.accept",
            "Add this search?",
            default=True,
        )
        if accept is BACK:
            return False
        if not accept:
            print("  Search discarded; starting it again.")
            continue

        run = {
            "constraints": constraints,
            "objectives": [{"metric": metric, "direction": "minimize"} for metric in objectives],
            "search_space": {
                "depth": depths,
                "embedding": embeddings,
            },
            "solver": {
                "backend": "auto",
                "num_solutions": int(heterogeneous_per_solve),
                "min_hamming_distance": 2,
                "max_seconds_per_solution": 60,
            },
            "homogeneous": {
                "enabled": solution_types == "both",
                "keep": homogeneous_keep if solution_types == "both" else "all",
                "rank_by": "objective",
            },
        }
        if explicit_variants:
            run["variants"] = explicit_variants
        runs[search_id] = run
        estimates[search_id] = estimate
        more = session.confirm(
            "mip.search.add",
            "Define another MIP search?",
            default=False,
        )
        if more is BACK:
            return False
        if not more:
            break

    session.state.set_collection(
        "mip_config",
        {
            "defaults": {},
            "workloads": workloads,
            "runs": runs,
        },
    )
    session.state.set_collection("mip_search_estimates", estimates)
    return True


def _post_mip_strategy(node: NodeDraft) -> str:
    return (
        "persistent_pool"
        if node.node_type == "evaluation" and node.input_id == "source"
        else "sharded"
    )


def post_mip_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    mip = _mapping_copy(session.state.collection("mip_config"))
    runs = _mapping_copy(mip.get("runs"))
    sequence = int(session.state.get_field("data.sequence_length", 4096))
    workloads = _mapping_copy(session.state.collection("serving_workloads"))
    first_workload = next(iter(workloads.values()), {})
    serving = {
        "input_tokens": int(first_workload.get("prefill_seq_len", sequence)),
        "output_tokens": int(first_workload.get("generation_seq_len", 1024)),
        "concurrency": [int(first_workload.get("max_num_seqs", 1))],
        "request_count": max(
            32,
            4 * int(first_workload.get("max_num_seqs", 1)),
        ),
        "best_selection_mode": "individual_best",
    }
    preview = {}
    for run_id, run in runs.items():
        objectives = [
            str(item.get("metric"))
            for item in run.get("objectives", ())
            if isinstance(item, Mapping)
        ]
        flow = recommended_flow(
            str(run_id),
            objectives,
            {"sequence_length": sequence},
            serving,
            node_prefix=(f"{run_id}_" if len(runs) > 1 else ""),
        )
        node_previews = []
        for node_id, node in flow.nodes.items():
            node_preview = {
                "id": node_id,
                "type": node.node_type,
                **({"config": dict(node.config)} if node.config else {}),
            }
            if node.node_type in {"evaluation", "global_kd"}:
                node_preview["resources"] = _stage_resource_defaults(
                    session,
                    resolver,
                    f"post.{run_id}.{node_id}",
                    strategy=_post_mip_strategy(node),
                    batch=1,
                )
            elif node.node_type == "aiperf":
                node_preview["resources"] = {
                    "instances": int(session.state.get_field("infrastructure.gpus_per_node", 8)),
                    "topology": node.config.get("topology", {}),
                }
            node_previews.append(node_preview)
        preview[str(run_id)] = {
            "mode": "recommended",
            "nodes": node_previews,
            "serving": serving,
        }
    action = _section_action(
        session,
        "post_mip",
        "For each MIP run, accept the recommended flow or add typed nodes one by one.",
        preview,
    )
    if action is BACK:
        return False
    editor = PostMIPFlowEditor(runs)
    for run_id, run in runs.items():
        flow_mode = "recommended"
        if action == "customize":
            flow_mode = session.select(
                f"post_mip.{run_id}.mode",
                f"Post-MIP flow for {run_id}:",
                [
                    ("Use recommended eight-node flow", "recommended"),
                    ("Add nodes one by one", "custom"),
                    ("No post-MIP flow", "none"),
                ],
                default="recommended",
            )
            if flow_mode is BACK:
                return False
        if flow_mode == "none":
            continue
        objectives = [
            str(item.get("metric"))
            for item in run.get("objectives", ())
            if isinstance(item, Mapping)
        ]
        if flow_mode == "recommended":
            run_serving = deepcopy(serving)
            if action == "customize":
                configured = _serving_setting_prompt(
                    session,
                    f"post_mip.{run_id}.serving",
                    run_serving,
                    inventory=context["model"].inventory,
                    pruning=_mapping_copy(_pruning_payload(session.state)),
                    stage_id=f"post.{run_id}.serving",
                )
                if configured is BACK:
                    return False
                run_serving = configured
            flow = recommended_flow(
                str(run_id),
                objectives,
                {"sequence_length": sequence},
                run_serving,
                node_prefix=(f"{run_id}_" if len(runs) > 1 else ""),
            )
            editor.add_flow(flow)
            if action == "customize":
                configured = _configure_post_mip_algorithms(
                    session,
                    editor,
                    str(run_id),
                    sequence_length=sequence,
                )
                if configured is BACK:
                    return False
            configured = _configure_dynamic_resources(
                session,
                editor,
                str(run_id),
                context["model"],
                ask=action == "customize",
            )
            if configured is BACK:
                return False
            continue
        flow = FlowDraft(str(run_id), str(run_id))
        editor.add_flow(flow)
        while True:
            node_type = session.select(
                f"post_mip.{run_id}.node.type",
                "Node type:",
                [
                    "filter",
                    "manual_filter",
                    "materialize",
                    "evaluation",
                    "aiperf",
                    "global_kd",
                    ("PTQ — unavailable", "unavailable"),
                    ("Downstream evaluation — unavailable", "unavailable"),
                ],
                default="evaluation",
            )
            if node_type is BACK:
                return False
            if node_type == "unavailable":
                print("  That node type is reserved but not implemented.")
                continue
            node_id = session.text(
                f"post_mip.{run_id}.node.id",
                "Node ID:",
                default=str(node_type),
            )
            input_choices = ["source", *editor.flow(str(run_id)).nodes]
            input_id = session.select(
                f"post_mip.{run_id}.node.input",
                "Candidate input:",
                input_choices,
                default=input_choices[-1],
            )
            if BACK in (node_id, input_id):
                return False
            selector = {}
            config = {}
            if node_type == "filter":
                metric = session.text(
                    f"post_mip.{run_id}.node.metric",
                    "Metric reference:",
                    default="mip.score",
                )
                top_k = session.integer(
                    f"post_mip.{run_id}.node.top_k",
                    "Top K:",
                    default=32,
                    minimum=1,
                )
                if BACK in (metric, top_k):
                    return False
                selector = {
                    "mode": "top_k",
                    "metric": str(metric),
                    "direction": "minimize",
                    "top_k": int(top_k),
                }
            elif node_type == "evaluation":
                eval_samples = session.integer(
                    f"post_mip.{run_id}.{node_id}.eval_samples",
                    "Evaluation samples:",
                    default=128,
                    minimum=1,
                )
                if eval_samples is BACK:
                    return False
                config = {
                    "eval_samples": int(eval_samples),
                    "block_size": sequence,
                }
            elif node_type == "aiperf":
                configured = _serving_setting_prompt(
                    session,
                    f"post_mip.{run_id}.{node_id}",
                    serving,
                    inventory=context["model"].inventory,
                    pruning=_mapping_copy(_pruning_payload(session.state)),
                    stage_id=f"post.{run_id}.{node_id}",
                )
                if configured is BACK:
                    return False
                config = {
                    **configured,
                    "concurrency": list(configured["concurrency"]),
                    "benchmark_timeout": 900,
                }
            elif node_type == "global_kd":
                max_steps = session.integer(
                    f"post_mip.{run_id}.{node_id}.max_steps",
                    "Global-KD training steps:",
                    default=128,
                    minimum=1,
                )
                global_batch = session.integer(
                    f"post_mip.{run_id}.{node_id}.global_batch_size",
                    "Global-KD sample/global batch size:",
                    default=128,
                    minimum=1,
                )
                if BACK in (max_steps, global_batch):
                    return False
                config = {
                    "max_steps": int(max_steps),
                    "global_batch_size": int(global_batch),
                }
            editor.add_node(
                str(run_id),
                NodeDraft(
                    str(node_id),
                    str(node_type),
                    input_id=str(input_id),
                    selector=selector,
                    config=config,
                ),
            )
            more = session.confirm(f"post_mip.{run_id}.node.add", "Add another node?", default=True)
            if more is BACK:
                return False
            if not more:
                break
        if action == "customize":
            configured = _configure_dynamic_resources(
                session,
                editor,
                str(run_id),
                context["model"],
                ask=True,
            )
            if configured is BACK:
                return False
    session.state.set_collection("post_mip_flows", editor.to_config())
    return True


def _configure_post_mip_algorithms(
    session: WizardSession,
    editor: PostMIPFlowEditor,
    flow_id: str,
    *,
    sequence_length: int,
) -> Any:
    for node_id, node in tuple(editor.flow(flow_id).nodes.items()):
        config = dict(node.config)
        if node.node_type == "evaluation":
            samples = session.integer(
                f"post_mip.{flow_id}.{node_id}.eval_samples",
                f"Evaluation samples for {node_id}:",
                default=int(config.get("eval_samples", 128)),
                minimum=1,
            )
            if samples is BACK:
                return BACK
            config.update(
                eval_samples=int(samples),
                block_size=int(sequence_length),
            )
        elif node.node_type == "global_kd":
            max_steps = session.integer(
                f"post_mip.{flow_id}.{node_id}.max_steps",
                f"Global-KD training steps for {node_id}:",
                default=int(config.get("max_steps", 128)),
                minimum=1,
            )
            global_batch = session.integer(
                f"post_mip.{flow_id}.{node_id}.global_batch_size",
                f"Global-KD sample/global batch size for {node_id}:",
                default=int(config.get("global_batch_size", 128)),
                minimum=1,
            )
            if BACK in (max_steps, global_batch):
                return BACK
            config.update(
                max_steps=int(max_steps),
                global_batch_size=int(global_batch),
            )
        else:
            continue
        editor.edit_node(flow_id, node_id, config=config)
    return True


def _parse_positive_int_list(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("Enter one or more comma-separated positive integers.")
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Enter one or more comma-separated positive integers.") from exc
    if any(item <= 0 for item in values):
        raise ValueError("Enter one or more comma-separated positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("Concurrency values must be unique.")
    return values


def _vllm_topology_prompt(
    session: WizardSession,
    prefix: str,
    defaults: Mapping[str, Any],
    *,
    inventory: Any,
    pruning: Mapping[str, Any],
    stage_id: str,
    label_prefix: str = "vLLM runtime",
) -> Any:
    """Ask for and validate one complete vLLM parallel topology."""
    topology_defaults = _mapping_copy(defaults)
    while True:
        topology = {}
        for name, label in (
            ("tensor_parallel_size", f"{label_prefix} tensor parallel (TP):"),
            ("pipeline_parallel_size", f"{label_prefix} pipeline parallel (PP):"),
            ("data_parallel_size", f"{label_prefix} data parallel (DP):"),
            (
                "prefill_context_parallel_size",
                f"{label_prefix} prefill context parallel (CP):",
            ),
            (
                "decode_context_parallel_size",
                f"{label_prefix} decode context parallel (CP):",
            ),
        ):
            value = session.integer(
                f"{prefix}.{name}",
                label,
                default=int(topology_defaults.get(name, 1)),
                minimum=1,
            )
            if value is BACK:
                return BACK
            topology[name] = int(value)
        enable_expert_parallel = False
        inventory_moe = (
            bool(inventory.get("moe", False))
            if isinstance(inventory, Mapping)
            else bool(getattr(inventory, "moe", False))
        )
        if inventory_moe:
            enable_expert_parallel = session.confirm(
                f"{prefix}.enable_expert_parallel",
                (
                    "Enable vLLM expert parallelism? vLLM has no separate EP size; "
                    "for this MoE model, effective EP is TP * DP."
                ),
                default=bool(topology_defaults.get("enable_expert_parallel", False)),
            )
            if enable_expert_parallel is BACK:
                return BACK
        topology.update(
            {
                "enable_expert_parallel": bool(enable_expert_parallel),
                "gpu_group_size": (
                    topology["tensor_parallel_size"]
                    * topology["pipeline_parallel_size"]
                    * topology["data_parallel_size"]
                    * topology["prefill_context_parallel_size"]
                ),
                "distributed_executor_backend": str(
                    topology_defaults.get("distributed_executor_backend", "mp")
                ),
            }
        )
        issues = validate_vllm_parallelism(
            topology,
            inventory,
            pruning,
            stage_id=stage_id,
        )
        if issues:
            _print_parallel_issues(issues)
            topology_defaults = topology
            continue
        return topology


def _serving_setting_prompt(
    session: WizardSession,
    prefix: str,
    defaults: Mapping[str, Any],
    *,
    inventory: Any,
    pruning: Mapping[str, Any],
    stage_id: str,
) -> Any:
    """Ask the complete AIPerf workload and serving-only parallel setting."""
    values = {}
    for name, label, default in (
        ("input_tokens", "Serving input sequence length (ISL):", defaults["input_tokens"]),
        ("output_tokens", "Serving output sequence length (OSL):", defaults["output_tokens"]),
    ):
        value = session.integer(f"{prefix}.{name}", label, default=int(default), minimum=1)
        if value is BACK:
            return BACK
        values[name] = int(value)
    raw_concurrency_default = defaults.get("concurrency", [1])
    if isinstance(raw_concurrency_default, (int, str)):
        concurrency_default = [int(raw_concurrency_default)]
    else:
        concurrency_default = [int(item) for item in raw_concurrency_default]

    def validate_concurrency(value: str) -> bool | str:
        try:
            _parse_positive_int_list(value)
        except ValueError as exc:
            return str(exc)
        return True

    concurrency_text = session.text(
        f"{prefix}.concurrency",
        "Serving concurrency sweep (comma-separated; one value is allowed):",
        default=", ".join(str(item) for item in concurrency_default),
        validate=validate_concurrency,
    )
    if concurrency_text is BACK:
        return BACK
    values["concurrency"] = _parse_positive_int_list(concurrency_text)
    request_count = session.integer(
        f"{prefix}.request_count",
        "AIPerf request count:",
        default=int(defaults.get("request_count", 32)),
        minimum=1,
    )
    if request_count is BACK:
        return BACK
    values["request_count"] = int(request_count)
    best_selection_mode = session.select(
        f"{prefix}.best_selection_mode",
        "How should the best models be selected?",
        [
            ("Best models at their individual best concurrency", "individual_best"),
            ("Top K models per concurrency, unioned", "best_per_concurrency"),
        ],
        default=str(defaults.get("best_selection_mode", "individual_best")),
    )
    if best_selection_mode is BACK:
        return BACK
    values["best_selection_mode"] = str(best_selection_mode)
    topology = _vllm_topology_prompt(
        session,
        f"{prefix}.topology",
        _mapping_copy(defaults.get("topology")),
        inventory=inventory,
        pruning=pruning,
        stage_id=stage_id,
        label_prefix="Serving",
    )
    if topology is BACK:
        return BACK
    values["topology"] = topology
    return values


def _configure_dynamic_resources(
    session: WizardSession,
    editor: PostMIPFlowEditor,
    flow_id: str,
    model: Any,
    *,
    ask: bool,
) -> Any:
    """Attach an independent resource/batch card to every node in one flow."""
    registry = ResourceProfileRegistry.from_dict(
        session.state.collection("parallel_profiles") or {}
    )
    resources = _mapping_copy(session.state.collection("stage_resources"))
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
    cpu_partition = session.state.get_field("infrastructure.runner.slurm.partition_cpu", None)
    for node_id, node in tuple(editor.flow(flow_id).nodes.items()):
        stage_id = f"post.{flow_id}.{node_id}"
        if node.node_type in {"filter", "manual_filter", "materialize"}:
            resources[stage_id] = {
                "strategy": "single",
                "instances": 1,
                "resource": "cpu" if cpu_partition else "gpu",
                "partition": cpu_partition,
                "gpus_per_node": gpus_per_node,
            }
            continue
        customize = False
        if ask:
            customize = session.confirm(
                f"{stage_id}.resources.customize",
                f"Customize resources and batch for {node_id}?",
                default=False,
            )
            if customize is BACK:
                return BACK
        strategy = _post_mip_strategy(node)
        instances = gpus_per_node
        if customize:
            instances = session.integer(
                f"{stage_id}.instances",
                "Independent model instances/workers:",
                default=gpus_per_node,
                minimum=1,
            )
            if instances is BACK:
                return BACK
        entry = {
            "strategy": str(strategy),
            "instances": int(instances),
            "resource": "gpu",
            "gpus_per_node": gpus_per_node,
        }
        if node.node_type == "aiperf":
            topology = _mapping_copy(node.config.get("topology"))
            allocation_mesh = vllm_topology_to_mesh(topology)
            entry["parallel"] = {
                **allocation_mesh.as_dict(),
                "sequence_parallel": False,
            }
        else:
            if customize:
                profile = _profile_prompt(
                    session,
                    registry,
                    stage_id,
                    model,
                    node_type=node.node_type,
                )
            else:
                profile, issues = _compatible_default_profile(
                    session,
                    registry,
                    stage_id,
                    model,
                    node_type=node.node_type,
                )
                if profile is None:
                    _print_parallel_issues(issues)
                    if session.guided:
                        raise SetupError(
                            "No configured parallel profile is compatible with "
                            f"{stage_id}. Supply a compatible profile in --defaults "
                            "or resume with --full."
                        )
                    profile = _profile_prompt(
                        session,
                        registry,
                        stage_id,
                        model,
                        node_type=node.node_type,
                    )
            if profile is BACK:
                return BACK
            issues = _profile_compatibility_issues(
                session,
                profile,
                stage_id,
                model,
                node_type=node.node_type,
            )
            if issues:
                _print_parallel_issues(issues)
                profile = _profile_prompt(
                    session,
                    registry,
                    stage_id,
                    model,
                    node_type=node.node_type,
                )
                if profile is BACK:
                    return BACK
            entry["profile_name"] = profile.name
            requested = 1
            if customize:
                requested = session.integer(
                    f"{stage_id}.batch",
                    (
                        "Local/micro batch size "
                        f"(minimum and scheduling unit: {profile.batch_unit}):"
                    ),
                    default=profile.batch_unit,
                    minimum=profile.batch_unit,
                )
                if requested is BACK:
                    return BACK
            resolved = resolve_batch(int(requested), profile)
            config = dict(node.config)
            automodel = _mapping_copy(config.get("automodel"))
            automodel["parallel"] = {
                "tp": profile.tp,
                "cp": profile.cp,
                "pp": profile.pp,
                "ep": profile.ep,
                "dp_shard": profile.dp_shard,
                "dp_replicate": profile.dp_replicate,
                "sequence_parallel": profile.sequence_parallel,
            }
            config["automodel"] = automodel
            key = "local_batch_size" if node.node_type == "global_kd" else "micro_batch_size"
            config[key] = resolved.effective
            editor.edit_node(flow_id, node_id, config=config)
            session.state.set_field(
                f"{stage_id}.batch",
                resolved.effective,
                source="user" if customize else "builtin",
                requested=resolved.requested,
                effective=resolved.effective,
                dependencies=(f"profiles.{profile.name}",),
            )
        resources[stage_id] = entry
    session.state.set_collection("parallel_profiles", registry.to_dict())
    session.state.set_collection("stage_resources", resources)
    return True


def _acquisition_sample_requirements(state: WizardState) -> tuple[int, int]:
    pruning = _mapping_copy(state.collection("pruning"))
    train_requirements = [int(pruning.get("width_importance_samples", 1))]
    bypass = _mapping_copy(pruning.get("bypass"))
    if bool(bypass.get("enabled", False)):
        train_requirements.append(int(bypass.get("samples", 1)))

    validation_requirements = [int(pruning.get("replacement_samples", 1))]
    if int(pruning.get("depth_remove", 0)) > 0:
        validation_requirements.append(int(pruning.get("depth_importance_samples", 1)))
    if bool(pruning.get("sort_sanity", False)):
        validation_requirements.append(int(pruning.get("sort_sanity_samples", 1)))
    if bool(pruning.get("sort_sanity", False)) and bool(pruning.get("width_sanity", False)):
        validation_requirements.append(int(pruning.get("width_sanity_samples", 1)))

    for raw_flow in _mapping_copy(state.collection("post_mip_flows")).values():
        for raw_node in _mapping_copy(_mapping_copy(raw_flow).get("nodes")).values():
            node = _mapping_copy(raw_node)
            config = _mapping_copy(node.get("config"))
            if node.get("type") == "global_kd":
                train_requirements.append(
                    int(config.get("max_steps", 1))
                    * int(
                        config.get(
                            "global_batch_size",
                            config.get("local_batch_size", 1),
                        )
                    )
                )
            elif node.get("type") == "evaluation":
                validation_requirements.append(int(config.get("eval_samples", 1)))

    return max(1, *train_requirements), max(1, *validation_requirements)


def _apportion_vlm_samples(
    subset_rows: Mapping[str, Any],
    total: int,
) -> dict[str, int]:
    rows = {str(name): int(value) for name, value in subset_rows.items()}
    if not rows or any(value <= 0 for value in rows.values()):
        raise SetupError(
            "Cannot infer Nemotron-VLM acquisition without positive selected-subset row counts."
        )
    source_total = sum(rows.values())
    quotas = {name: total * value // source_total for name, value in rows.items()}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        enumerate(rows.items()),
        key=lambda item: (
            -(total * item[1][1] % source_total),
            item[0],
        ),
    )
    for _, (name, _) in ranked[:remaining]:
        quotas[name] += 1
    return quotas


def _infer_vlm_shard_cap(
    acquisition: Mapping[str, Any],
    num_samples: int,
) -> int:
    subset_rows = _mapping_copy(acquisition.get("subset_rows"))
    subset_media_shards = _mapping_copy(acquisition.get("subset_media_shards"))
    missing = [name for name in subset_rows if int(subset_media_shards.get(name, 0)) <= 0]
    if missing:
        previous_shard_cap = int(acquisition.get("max_shards_per_subset", 0))
        if previous_shard_cap > 0:
            return previous_shard_cap
        raise SetupError(
            "Cannot infer Nemotron-VLM media-shard count because the revision-locked "
            f"catalog lacks tar metadata for: {', '.join(missing)}. "
            "Return to the data section to refresh the dataset catalog."
        )

    quotas = _apportion_vlm_samples(subset_rows, num_samples)
    estimates = []
    for name, source_rows in subset_rows.items():
        available_shards = int(subset_media_shards[name])
        quota = quotas[name]
        estimated = (quota * available_shards + int(source_rows) - 1) // int(source_rows)
        estimates.append(min(available_shards, max(1, estimated)))
    return max(estimates)


def _infer_acquisition_samples(state: WizardState) -> None:
    acquisition = _mapping_copy(state.collection("data_acquisition"))
    adapter = acquisition.get("adapter")
    if adapter not in {_PUZZLE_KD_ADAPTER, _NEMOTRON_VLM_ADAPTER}:
        return

    train_samples, validation_samples = _acquisition_sample_requirements(state)
    if adapter == _PUZZLE_KD_ADAPTER:
        inferred = {
            "train_samples": train_samples,
            "validation_samples": validation_samples,
        }
    else:
        num_samples = max(train_samples, validation_samples)
        inferred = {
            "num_samples": num_samples,
            "max_shards_per_subset": _infer_vlm_shard_cap(
                acquisition,
                num_samples,
            ),
        }
    acquisition.update(inferred)
    state.set_collection("data_acquisition", acquisition)
    for key, value in inferred.items():
        state.set_field(
            f"data.acquisition.{key}",
            value,
            source="inferred",
        )


def output_review_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    del context
    _infer_acquisition_samples(session.state)
    default_root = str(
        resolver.resolve_default(
            "output.result_root",
            str(session.state.campaign_dir / "results"),
        ).value
    )
    action = _section_action(
        session,
        "output",
        "Review effective values, then validate and generate both bundles.",
        {
            "result_root": default_root,
            "generate_smoke_bundle": True,
            "generate_production_bundle": True,
        },
    )
    if action is BACK:
        return False
    if action == "customize":
        root = _text_field(
            session,
            resolver,
            "output.result_root",
            "Campaign results location:",
            default_root,
        )
        if root is BACK:
            return False
    else:
        _record_default(session.state, resolver, "output.result_root", default_root)
    print("\nEffective setup:")
    if session.guided:
        pruning = _mapping_copy(session.state.collection("pruning"))
        mip = _mapping_copy(session.state.collection("mip_config"))
        runs = _mapping_copy(mip.get("runs"))
        profiles = _mapping_copy(session.state.collection("parallel_profiles"))
        axes = {
            axis_id: list(_mapping_copy(axis).get("values") or ())
            for axis_id, axis in _mapping_copy(pruning.get("axes")).items()
        }
        mip_review = {
            run_id: {
                "constraints": _mapping_copy(_mapping_copy(run).get("constraints")),
                "num_solutions": _mapping_copy(_mapping_copy(run).get("solver")).get(
                    "num_solutions"
                ),
            }
            for run_id, run in runs.items()
        }
        profile_review = {
            name: {
                key: _mapping_copy(profile).get(key)
                for key in (
                    "tp",
                    "cp",
                    "pp",
                    "dp_shard",
                    "dp_replicate",
                    "ep",
                    "sequence_parallel",
                )
            }
            for name, profile in profiles.items()
        }
        summary = {
            "preset": session.state.preset,
            "model": session.state.get_field("model.source"),
            "dataset": session.state.get_field("data.selected_source"),
            "sequence_length": session.state.get_field("data.sequence_length"),
            "pruning": {
                "maximum_depth_removed": pruning.get("depth_remove"),
                "depth_importance_samples": pruning.get("depth_importance_samples"),
                "width_importance_samples": pruning.get("width_importance_samples"),
                "width_axes": axes,
                "sort_sanity": pruning.get("sort_sanity"),
                "width_sanity": pruning.get("width_sanity"),
                "slicing_sanity": pruning.get("slicing_sanity"),
                "bypass": _mapping_copy(pruning.get("bypass")),
                "replacement_samples": pruning.get("replacement_samples"),
            },
            "mip_searches": mip_review,
            "parallel_profiles": profile_review,
            "execution": {
                "repository": session.state.get_field(
                    "infrastructure.execution_contract.repository"
                ),
                "venv": session.state.get_field("infrastructure.execution_contract.venv"),
                "container": session.state.get_field("infrastructure.execution_contract.container"),
                "slurm_account": session.state.get_field("infrastructure.runner.slurm.account"),
                "interactive_partition": session.state.get_field(
                    "infrastructure.runner.slurm.partition_interactive"
                ),
                "batch_partition": session.state.get_field(
                    "infrastructure.runner.slurm.partition_batch"
                ),
                "gpus_per_node": session.state.get_field("infrastructure.gpus_per_node"),
            },
            "results": session.state.get_field("output.result_root"),
        }
        print(yaml.safe_dump(summary, sort_keys=False))
        print(
            "  Nested values and provenance are saved in answers_v2.yaml. "
            "Use --full for per-section customization."
        )
        print(
            "  Execution values target the machine or cluster running setup. "
            "Use --defaults or --full when that environment needs different values."
        )
    else:
        print(
            yaml.safe_dump(
                _plain_review_value(
                    {
                        "fields": {
                            path: {
                                "effective": record.effective,
                                "requested": record.requested,
                                "source": record.source,
                            }
                            for path, record in session.state.records().items()
                        },
                        "profiles": session.state.collection("parallel_profiles"),
                        "serving_workloads": session.state.collection("serving_workloads"),
                        "vllm_measurements": session.state.collection("vllm_measurements"),
                        "mip": session.state.collection("mip_config"),
                        "post_mip": session.state.collection("post_mip_flows"),
                    }
                ),
                sort_keys=False,
            )
        )
    generate = session.confirm(
        "output.generate",
        "Validate and generate smoke and production bundles?",
        default=True,
    )
    if generate is BACK:
        return False
    if not generate:
        raise SetupError(f"Answers saved at {session.state.path}; no bundle was generated.")
    return True


SECTION_BUILDERS: tuple[Callable[..., bool], ...] = (
    model_section,
    data_section,
    infrastructure_section,
    depth_section,
    width_axes_section,
    width_importance_section,
    sort_sanity_section,
    width_sanity_section,
    slicing_sanity_section,
    bypass_section,
    replacement_scoring_section,
    serving_workloads_section,
    vllm_section,
    mip_section,
    post_mip_section,
    output_review_section,
)
SECTION_NAMES = (
    "model",
    "data",
    "infrastructure",
    "depth",
    "width_axes",
    "width_importance",
    "sort_sanity",
    "width_sanity",
    "slicing_sanity",
    "bypass",
    "replacement_scoring",
    "serving_workloads",
    "vllm",
    "mip",
    "post_mip",
    "output",
)


def _fresh_state(
    backend: PromptBackend,
    defaults_path: Path | None,
    *,
    full: bool,
) -> WizardState:
    if full:
        while True:
            value = backend.text("Campaign directory:", "")
            if value is BACK:
                continue
            path = Path(str(value)).expanduser()
            if str(path):
                return WizardState.start(
                    path,
                    defaults_path=defaults_path,
                    setup_mode="full",
                )

    while True:
        preset = _select_setup_preset(backend)
        if preset is BACK:
            continue
        value = backend.text("Campaign directory:", "")
        if value is BACK:
            continue
        path = Path(str(value)).expanduser()
        if str(path):
            return WizardState.start(
                path,
                defaults_path=defaults_path,
                setup_mode="quick",
                preset=str(preset),
            )


def _select_setup_preset(
    backend: PromptBackend,
    *,
    default: str = "balanced",
) -> Any:
    return backend.select(
        "Setup profile:",
        [PromptChoice(item.choice_title, item.name) for item in QUICK_SETUP_PRESETS],
        default,
    )


def _refresh_legacy_state(state: WizardState) -> None:
    pruning = deepcopy(state.collection("pruning") or {})
    subset_selection = _mapping_copy(state.collection("data_subset_selection"))
    subset_records = [_mapping_copy(item) for item in subset_selection.get("subsets") or ()]
    serving_workloads = _mapping_copy(state.collection("serving_workloads"))
    measurements = _mapping_copy(state.collection("vllm_measurements"))
    first_workload = next(
        iter(serving_workloads.items()),
        ("serving-default", {}),
    )
    workload_id, workload = first_workload
    measurement = _mapping_copy(measurements.get(workload_id))
    if not measurement and measurements:
        measurement = _mapping_copy(next(iter(measurements.values())))
    runtime = {
        "vllm_enabled": bool(measurements),
        "granularity": measurement.get("granularity", "subblock"),
        "workload_id": workload_id,
        "isl": int(workload.get("prefill_seq_len", state.get_field("data.sequence_length", 4096))),
        "osl": int(workload.get("generation_seq_len", 1024)),
        "concurrency": int(workload.get("max_num_seqs", 1)),
    }
    infrastructure = {
        "runner": {
            "kind": state.get_field("infrastructure.runner.kind", "slurm"),
            "slurm": {
                "account": state.get_field("infrastructure.runner.slurm.account", ""),
                "partition_interactive": state.get_field(
                    "infrastructure.runner.slurm.partition_interactive", "interactive"
                ),
                "partition_batch": state.get_field(
                    "infrastructure.runner.slurm.partition_batch", "batch"
                ),
                "partition_cpu": state.get_field("infrastructure.runner.slurm.partition_cpu", None),
                "time_limit": state.get_field("infrastructure.runner.slurm.time_limit", "4:00:00"),
                "qos": state.get_field("infrastructure.runner.slurm.qos", None),
                "max_nodes": state.get_field("infrastructure.runner.slurm.max_nodes", 64),
            },
        },
        "execution_contract": {
            "repository": state.get_field(
                "infrastructure.execution_contract.repository", WORKER_REPOSITORY_PLACEHOLDER
            ),
            "venv": state.get_field(
                "infrastructure.execution_contract.venv", WORKER_VENV_PLACEHOLDER
            ),
            "container": state.get_field("infrastructure.execution_contract.container", None),
            "container_mounts": state.get_field(
                "infrastructure.execution_contract.container_mounts", None
            ),
            "prerun_commands": state.get_field(
                "infrastructure.execution_contract.prerun_commands", []
            ),
            "postrun_commands": state.get_field(
                "infrastructure.execution_contract.postrun_commands", []
            ),
        },
        "gpus_per_node": state.get_field("infrastructure.gpus_per_node", 8),
        "meshes": {
            "common": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
            "bypass": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
            "global_kd": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
        },
        "workers": {
            "pool": state.get_field("infrastructure.gpus_per_node", 8),
            "sharded": state.get_field("infrastructure.gpus_per_node", 8),
        },
    }
    profiles = _mapping_copy(state.collection("parallel_profiles"))
    if profiles:
        first = next(iter(profiles.values()))
        mesh = {
            key: first.get(key, 1) for key in ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")
        }
        infrastructure["meshes"] = {
            "common": deepcopy(mesh),
            "bypass": deepcopy(mesh),
            "global_kd": deepcopy(mesh),
        }
    legacy = {
        "schema_version": 1,
        "wizard_version": "1",
        "detailed": True,
        "model": deepcopy(state.payload.get("model") or {}),
        "inventory": deepcopy(state.payload.get("inventory") or {}),
        "answers": {
            "data": {
                "source": state.get_field("data.source"),
                "selected_source": state.get_field(
                    "data.selected_source", state.get_field("data.source")
                ),
                "adapter": state.get_field("data.adapter", "custom"),
                "modality": state.get_field("data.modality", "text"),
                "layout": state.get_field("data.layout", "fixed"),
                "sequence_length": state.get_field("data.sequence_length", 4096),
                "subsets": [record["name"] for record in subset_records],
                "subset_revision": subset_selection.get("revision"),
                "subset_weights": {record["name"]: record["weight"] for record in subset_records},
                "acquisition": deepcopy(state.collection("data_acquisition") or {}),
            },
            "pruning": pruning,
            "runtime": runtime,
            "mip": deepcopy(state.collection("mip_config") or {"runs": {}}),
            "post_mip": {"flows": deepcopy(state.collection("post_mip_flows") or {})},
            "infrastructure": infrastructure,
            "output": {"result_root": state.get_field("output.result_root")},
        },
    }
    state.set_collection("legacy_state", legacy)


def run_wizard_v2(
    *,
    resume: Path | None,
    defaults_path: Path | None,
    backend: PromptBackend | None = None,
    full: bool = False,
) -> Path:
    """Run setup v2, save every answer, validate bundles, and never launch jobs."""
    backend = backend or InteractiveBackend()
    print("Welcome to Puzzletron setup v2.")
    if resume is None:
        if full:
            print("  Full setup enabled: every advanced section is customizable.")
        else:
            print(
                "  Guided setup asks only for essential choices and applies nested "
                "defaults from a profile."
            )
            print("  Use --full only when you need every advanced control.")
        state = _fresh_state(backend, defaults_path, full=full)
    else:
        state = WizardState.resume(resume)
        if full and state.setup_mode != "full":
            print(
                "  Promoting this guided campaign to full setup. Existing model "
                "and dataset answers are preserved."
            )
            state.set_setup_mode("full")
    if state.setup_mode not in {"quick", "full"}:
        raise SetupError(f"Unsupported setup mode in {state.path}: {state.setup_mode!r}.")
    preset = None
    if state.preset:
        preset = get_setup_preset(state.preset)
    if state.setup_mode == "quick":
        if preset is None:
            raise SetupError(f"Guided setup state {state.path} does not record a setup preset.")
        print(f"  Profile: {preset.choice_title}")
    elif preset is not None:
        print(f"  Full setup baseline: {preset.choice_title}")
    else:
        print("  Full setup enabled: every advanced section is customizable.")
    selected_defaults = defaults_path or state.defaults_path
    if resume is not None and defaults_path is not None:
        state.set_defaults_path(defaults_path)
        print(f"  Persisted replacement defaults file: {state.defaults_path}")
    session = WizardSession(
        state,
        backend,
        guided=state.setup_mode == "quick",
    )
    context: dict[str, Any] = {}
    if state.payload.get("model", {}).get("source"):
        saved = state.payload["model"]
        context["model"] = inspect_model(str(saved["source"]))
    family_config = context["model"].inventory.family_config if "model" in context else None
    model_inventory = context["model"].inventory if "model" in context else None
    resolver = _resolver(state, selected_defaults, preset, family_config, model_inventory)

    index = 0
    while index < len(SECTION_BUILDERS):
        if session.guided and index == 2:
            print(
                "\nApplying the selected profile to advanced pruning, runtime, "
                "MIP, and post-MIP settings..."
            )
        builder = SECTION_BUILDERS[index]
        completed = builder(session, resolver, context)
        if completed:
            if SECTION_NAMES[index] == "model" and "model" in context:
                resolver = _resolver(
                    state,
                    selected_defaults,
                    preset,
                    context["model"].inventory.family_config,
                    context["model"].inventory,
                )
            index += 1
        else:
            target = session.consume_back_target()
            if target is None and index == 0 and session.guided:
                replacement = _select_setup_preset(
                    backend,
                    default=state.preset or "balanced",
                )
                if replacement is not BACK:
                    state.set_preset(str(replacement))
                    preset = get_setup_preset(str(replacement))
                    family_config = (
                        context["model"].inventory.family_config if "model" in context else None
                    )
                    model_inventory = context["model"].inventory if "model" in context else None
                    resolver = _resolver(
                        state,
                        selected_defaults,
                        preset,
                        family_config,
                        model_inventory,
                    )
                    print(f"  Profile changed to: {preset.choice_title}")
                continue
            index = SECTION_NAMES.index(target.section) if target is not None else index
    state.set_collection(
        "default_resolutions",
        {
            path: {
                "value": resolved.value,
                "source": resolved.source,
            }
            for path, resolved in resolver.resolutions().items()
        },
    )
    _refresh_legacy_state(state)
    build_bundles_v2(state.campaign_dir, state)
    return state.campaign_dir
