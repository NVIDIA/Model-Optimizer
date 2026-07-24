# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, locally customizable Puzzletron setup-v2 wizard."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from puzzletron_setup import SetupError
from puzzletron_setup.inspection import (
    infer_dataset_modality,
    inspect_model,
    normalize_dataset_source,
    normalize_model_source,
)

from .bundle import build_bundles_v2
from .defaults import DefaultsResolver, load_defaults
from .post_mip import (
    FlowDraft,
    NodeDraft,
    PostMIPFlowEditor,
    recommended_flow,
)
from .prompts import BACK, InteractiveBackend, PromptBackend, PromptChoice
from .resources import (
    ParallelProfile,
    ResourceProfileRegistry,
    StageResources,
    allocation_summary,
    resolve_batch,
    validate_parallel_profile,
)
from .session import WizardSession
from .state import WizardState

__all__ = ["SECTION_BUILDERS", "run_wizard_v2"]

BUILTINS = {
    "data": {"modality": "text", "layout": "fixed", "sequence_length": 4096},
    "infrastructure": {
        "gpus_per_node": 8,
        "execution_contract": {
            "repository": str(Path.cwd()),
            "venv": ".venv",
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
        "replacement_samples": 128,
        "bypass": {
            "enabled": True,
            "granularity": "subblock",
            "samples": 4096,
            "sequence_length": 4096,
            "batch_size": 8,
        },
    },
    "vllm": {
        "enabled": False,
        "granularity": "subblock",
        "prefill_seq_len": 4096,
        "generation_seq_len": 1024,
        "batch_size": 1,
        "max_num_seqs": 1,
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
    "bypass": "bypass.training.micro_batch_size",
    "replacement_scoring": "replacement_scoring.micro_batch_size",
}
CANONICAL_STAGE_STRATEGIES = {
    "depth_importance": "persistent_pool",
    "width_importance": "single",
    "sort_sanity": "single",
    "bypass": "single",
    "replacement_scoring": "persistent_pool",
}

_CUSTOM_MODEL_SOURCE = "__custom_model_source__"
_DEFAULT_MODEL_SOURCE = "__default_model_source__"
_CUSTOM_DATA_SOURCE = "__custom_data_source__"
_DEFAULT_DATA_SOURCE = "__default_data_source__"

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


def _data_source_choices(resolver: DefaultsResolver) -> list[PromptChoice]:
    choices = []
    explicit = resolver.file_default("data.source")
    if explicit is not None and explicit.value:
        choices.append(PromptChoice(f"Default — {explicit.value}", _DEFAULT_DATA_SOURCE))
    choices.append(PromptChoice("Custom", _CUSTOM_DATA_SOURCE))
    return choices


def _nested_records(state: WizardState) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, record in state.records().items():
        current = nested
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = deepcopy(record.effective)
    return nested


def _resolver(state: WizardState, defaults_path: Optional[Path]) -> DefaultsResolver:
    return DefaultsResolver(
        builtins=BUILTINS,
        model_derived={},
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
) -> Any:
    session.begin(section)
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
        deepcopy(dict(defaults)),
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    for line in rendered.splitlines():
        print(f"    {line}")


def _text_field(
    session: WizardSession,
    resolver: DefaultsResolver,
    path: str,
    label: str,
    fallback: str = "",
) -> Any:
    resolved = _resolved(session.state, resolver, path, fallback)
    value = session.text(path, label, default=str(resolved.value or ""))
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
    maximum: Optional[int] = None,
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


def data_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    session.begin("data")
    explicit = resolver.file_default("data.source")
    if explicit is not None and explicit.value:
        _print_default_decisions({"source": explicit.value})
    choices = _data_source_choices(resolver)
    mode = session.select(
        "data.source_mode",
        "Dataset:",
        choices,
        default=(
            _DEFAULT_DATA_SOURCE
            if explicit is not None and explicit.value
            else _CUSTOM_DATA_SOURCE
        ),
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
    else:
        if explicit is None or not explicit.value:
            raise SetupError("The selected dataset default is no longer available.")
        source = normalize_dataset_source(str(explicit.value))
        source_kind = explicit.source

    finding = infer_dataset_modality(source)
    modality_choices = [("Text", "text")]
    if context["model"].inventory.multimodal:
        modality_choices.append(("Multimodal", "multimodal"))
    suggested_modality = str(
        resolver.resolve(
            "data.modality",
            finding.modality if finding.modality != "unknown" else "text",
        ).value
    )
    valid_modalities = {value for _, value in modality_choices}
    if suggested_modality not in valid_modalities:
        suggested_modality = "text"
    modality = session.select(
        "data.modality",
        f"Data modality ({finding.evidence}):",
        modality_choices,
        default=suggested_modality,
    )
    if modality is BACK:
        return False
    layout = session.select(
        "data.layout",
        "Dataset layout:",
        [
            ("Fixed-length", "fixed"),
            ("Packed variable-length", "packed_varlen"),
            ("Padded", "padded"),
        ],
        default=str(resolver.resolve("data.layout", "fixed").value),
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
    session.state.set_field("data.source", source, source=source_kind)
    session.state.set_field("data.modality", modality, source="user")
    session.state.set_field("data.layout", layout, source="user")
    session.state.set_field(
        "data.sequence_length",
        int(sequence),
        source="user",
    )
    return True


def infrastructure_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    del context
    paths = (
        ("infrastructure.execution_contract.repository", str(Path.cwd())),
        ("infrastructure.execution_contract.venv", ".venv"),
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
        path.removeprefix("infrastructure."): resolver.resolve_default(
            path, fallback
        ).value
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
    )
    if action is BACK:
        return False
    if action == "defaults":
        for path, fallback in paths:
            _record_default(session.state, resolver, path, fallback)
        commands = resolver.resolve_default(
            "infrastructure.execution_contract.prerun_commands", []
        )
        session.state.set_field(
            "infrastructure.execution_contract.prerun_commands",
            list(commands.value or ()),
            source=commands.source,
        )
        postrun = resolver.resolve_default(
            "infrastructure.execution_contract.postrun_commands", []
        )
        session.state.set_field(
            "infrastructure.execution_contract.postrun_commands",
            list(postrun.value or ()),
            source=postrun.source,
        )
        return True

    for path, label, fallback in (
        (
            "infrastructure.execution_contract.repository",
            "Repository path on workers:",
            str(Path.cwd()),
        ),
        ("infrastructure.execution_contract.venv", "Python environment:", ".venv"),
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
        value = _text_field(session, resolver, path, label, fallback)
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
    prerun_default = resolver.resolve(
        "infrastructure.execution_contract.prerun_commands", []
    ).value
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
        inventory.num_sublayers
        if default_granularity == "subblock"
        else inventory.num_layers
    )
    default_remove = min(
        int(
            resolver.resolve_default(
                "pruning.depth_remove", min(4, default_count - 1)
            ).value
        ),
        default_count - 1,
    )
    default_axes = {
        axis.axis_id: resolver.resolve_default(
            f"pruning.axes.{axis.axis_id}.values",
            list(axis.values)[: min(2, len(axis.values))],
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
            "sort_sanity": resolver.resolve_default(
                "pruning.sort_sanity", False
            ).value,
            "replacement_granularity": resolver.resolve_default(
                "pruning.replacement_granularity", default_granularity
            ).value,
            "bypass": {
                "enabled": resolver.resolve_default(
                    "pruning.bypass.enabled", True
                ).value,
                "granularity": resolver.resolve_default(
                    "pruning.bypass.granularity", "subblock"
                ).value,
                "samples": resolver.resolve_default(
                    "pruning.bypass.samples", 4096
                ).value,
                "sequence_length": session.state.get_field(
                    "data.sequence_length", 4096
                ),
                "batch_size": resolver.resolve_default(
                    "pruning.bypass.batch_size", 8
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
    ):
        defaults["bypass"][key] = resolver.resolve_default(
            f"pruning.bypass.{key}", fallback
        ).value
    if action == "customize":
        granularity = session.select(
            "pruning.depth_granularity",
            "Depth pruning granularity:",
            [("Sublayer", "subblock"), ("Whole block", "block")],
            default="subblock",
        )
        if granularity is BACK:
            return False
        count = (
            inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        )
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
        count = (
            inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        )
        remove = default_remove
    axes = {}
    for axis in inventory.axes:
        axis_values = list(axis.values)
        default_values = list(
            resolver.resolve_default(
                f"pruning.axes.{axis.axis_id}.values",
                axis_values[: min(2, len(axis_values))],
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
            "sort_sanity": bool(
                resolver.resolve_default("pruning.sort_sanity", False).value
            ),
            "replacement_granularity": str(
                resolver.resolve_default(
                    "pruning.replacement_granularity", granularity
                ).value
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


def _profile_prompt(
    session: WizardSession,
    registry: ResourceProfileRegistry,
    stage_id: str,
    model: Any,
) -> Any:
    names = registry.names()
    if names:
        choices = [
            (
                f"Reuse {name} — TP={registry.get(name).tp} CP={registry.get(name).cp} "
                f"PP={registry.get(name).pp} DP-shard={registry.get(name).dp_shard} "
                f"DP-replicate={registry.get(name).dp_replicate} EP={registry.get(name).ep}",
                f"reuse:{name}",
            )
            for name in names
        ]
        choices.extend(
            [
                ("Copy and modify an existing configuration", "copy"),
                ("Create a new configuration", "new"),
            ]
        )
        action = session.select(
            f"stages.{stage_id}.profile_action",
            f"Parallel setting for {stage_id}:",
            choices,
            default=f"reuse:{names[0]}",
        )
        if action is BACK:
            return BACK
        if str(action).startswith("reuse:"):
            return registry.reuse(str(action).split(":", 1)[1], consumer=stage_id)
        base = registry.get(names[0]) if action == "copy" else ParallelProfile(stage_id)
    else:
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
    validate_parallel_profile(profile, model.inventory)
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
    strategy: Optional[str] = None,
    batch: int,
) -> dict[str, Any]:
    registry = _resource_registry(session, resolver)
    profile = (
        registry.get(registry.names()[0])
        if registry.names()
        else ParallelProfile(stage_id)
    )
    strategy = strategy or CANONICAL_STAGE_STRATEGIES[stage_id]
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
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
    requested_batch = int(
        resolver.resolve_default(f"stages.{stage_id}.batch", batch).value
    )
    resolution = resolve_batch(requested_batch, profile)
    return {
        "strategy": strategy,
        "instances": instances,
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
        profile = (
            registry.reuse(registry.names()[0], consumer=stage_id)
            if registry.names()
            else registry.create(ParallelProfile(stage_id), consumer=stage_id)
        )
        strategy = defaults["strategy"]
        instances = defaults["instances"]
        requested_batch = defaults["requested_batch"]

    resolution = resolve_batch(int(requested_batch), profile)
    if resolution.adjusted:
        print(
            f"  {stage_id} batch {resolution.requested} rounds up to "
            f"{resolution.effective} "
            f"(unit PP×DP-shard×DP-replicate={resolution.unit})."
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
        source="user" if action == "customize" else "builtin",
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
            [("Sublayer", "subblock"), ("Whole block", "block")],
            default=granularity,
        )
        if granularity is BACK:
            return False
        count = (
            inventory.num_sublayers
            if granularity == "subblock"
            else inventory.num_layers
        )
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
        samples = session.integer(
            "pruning.depth_importance_samples",
            "Depth-importance evaluation samples:",
            default=samples,
            minimum=1,
        )
        if BACK in (remove, samples):
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


def width_axes_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    pruning = _pruning_payload(session.state)
    inventory = context["model"].inventory
    current_axes = _mapping_copy(pruning.get("axes"))
    defaults = {
        axis.axis_id: list(
            resolver.resolve_default(
                f"pruning.axes.{axis.axis_id}.values",
                _mapping_copy(current_axes.get(axis.axis_id)).get(
                    "values",
                    list(axis.values)[: min(2, len(axis.values))],
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
    axes = {}
    for axis in inventory.axes:
        selected = defaults[axis.axis_id]
        if action == "customize":
            selected = session.checkbox(
                f"pruning.axes.{axis.axis_id}",
                f"Values for {axis.label}:",
                [(str(value), value) for value in axis.values],
                defaults=selected,
                validate=lambda values: bool(values) or "Select at least one value.",
            )
            if selected is BACK:
                return False
        axes[axis.axis_id] = {
            "enabled": True,
            "teacher_value": axis.teacher_value,
            "values": sorted({int(value) for value in selected}, reverse=True),
            "alignment": axis.alignment,
        }
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


def sort_sanity_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    pruning = _pruning_payload(session.state)
    enabled = bool(
        resolver.resolve_default(
            "pruning.sort_sanity", pruning.get("sort_sanity", False)
        ).value
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
        resolver.resolve_default(
            "pruning.bypass.samples", bypass.get("samples", 4096)
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
            if BACK in (granularity, samples):
                return False
    bypass.update(
        {
            "enabled": bool(enabled),
            "granularity": str(granularity),
            "samples": int(samples),
            "sequence_length": int(
                session.state.get_field("data.sequence_length", 4096)
            ),
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
            [
                (
                    f"One sublayer at a time ({inventory.num_sublayers} sublayers)",
                    "subblock",
                ),
                (
                    f"One block at a time ({inventory.num_layers} layers)",
                    "block",
                ),
            ],
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
        if stage_id == "bypass" and not (
            session.state.collection("pruning") or {}
        ).get("bypass", {}).get("enabled", False):
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
        requested_batch = int(
            resolver.resolve_default(f"stages.{stage_id}.batch", 8).value
        )
        batch = resolve_batch(requested_batch, preview_profile)
        preview[stage_id] = {
            "strategy": strategy,
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
        "Each stage has independent instances, strategy, parallelism, and batch behavior.",
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
        if stage_id == "bypass" and not (
            session.state.collection("pruning") or {}
        ).get("bypass", {}).get("enabled", False):
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
                (
                    "Local/micro batch size "
                    f"(minimum and scheduling unit: {profile.batch_unit}):"
                ),
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
            requested_batch = int(
                resolver.resolve_default(f"stages.{stage_id}.batch", 8).value
            )
        resolution = resolve_batch(int(requested_batch), profile)
        if resolution.adjusted:
            print(
                f"  {stage_id} batch {resolution.requested} rounds up to "
                f"{resolution.effective} (unit PP×DP-shard×DP-replicate={resolution.unit})."
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


def _default_vllm_topology(resolver: DefaultsResolver) -> dict[str, Any]:
    topology = {
        key: int(
            resolver.resolve_default(f"vllm.topology.{key}", 1).value
        )
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    }
    topology["gpu_group_size"] = (
        topology["tensor_parallel_size"]
        * topology["pipeline_parallel_size"]
        * topology["prefill_context_parallel_size"]
    )
    topology["distributed_executor_backend"] = "mp"
    return topology


def _default_vllm_measurement(
    resolver: DefaultsResolver,
    *,
    sequence_length: int,
) -> dict[str, Any]:
    batch = int(resolver.resolve_default("vllm.batch_size", 1).value)
    max_num_seqs = max(
        batch,
        int(resolver.resolve_default("vllm.max_num_seqs", 1).value),
    )
    granularity = str(
        resolver.resolve_default("vllm.granularity", "subblock").value
    )
    topology = _default_vllm_topology(resolver)
    return {
        "prefill_seq_len": int(
            resolver.resolve_default("vllm.prefill_seq_len", sequence_length).value
        ),
        "generation_seq_len": int(
            resolver.resolve_default("vllm.generation_seq_len", 1024).value
        ),
        "batch_size": batch,
        "max_num_seqs": max_num_seqs,
        "granularity": granularity,
        "runtime_stats": {
            "granularity": granularity,
            "max_num_seqs": max_num_seqs,
            "topology": topology,
        },
    }


def vllm_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    del context
    enabled_default = bool(resolver.resolve_default("vllm.enabled", False).value)
    default_measurement = _default_vllm_measurement(
        resolver,
        sequence_length=int(session.state.get_field("data.sequence_length", 4096)),
    )
    default_topology = default_measurement["runtime_stats"]["topology"]
    preview = {"enabled": enabled_default}
    if enabled_default:
        preview["measurement"] = default_measurement
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
            return True
        session.state.set_collection(
            "vllm_measurements",
            {"serving-default": default_measurement},
        )
        return True
    measurements: OrderedDict[str, Any] = OrderedDict()
    while True:
        name = session.text(
            "vllm.measurement.id",
            "Measurement ID:",
            default="serving-default" if not measurements else f"serving-{len(measurements) + 1}",
        )
        if name is BACK:
            return False
        defaults = {
            key: int(resolver.resolve_default(f"vllm.{key}", fallback).value)
            for key, fallback in (
                ("prefill_seq_len", session.state.get_field("data.sequence_length", 4096)),
                ("generation_seq_len", 1024),
                ("batch_size", 1),
                ("max_num_seqs", 1),
            )
        }
        values = {}
        for key, label in (
            ("prefill_seq_len", "Input sequence length (ISL):"),
            ("generation_seq_len", "Output sequence length (OSL):"),
            ("batch_size", "Measurement batch size:"),
            ("max_num_seqs", "Maximum concurrent sequences:"),
        ):
            value = session.integer(
                f"vllm.measurements.{name}.{key}",
                label,
                default=defaults[key],
                minimum=1,
            )
            if value is BACK:
                return False
            values[key] = int(value)
        values["max_num_seqs"] = max(values["max_num_seqs"], values["batch_size"])
        granularity = session.select(
            f"vllm.measurements.{name}.granularity",
            "Measurement granularity:",
            ["subblock", "block"],
            default=str(resolver.resolve_default("vllm.granularity", "subblock").value),
        )
        if granularity is BACK:
            return False
        topology = {}
        for key, label in (
            ("tensor_parallel_size", "vLLM tensor parallel (TP):"),
            ("pipeline_parallel_size", "vLLM pipeline parallel (PP):"),
            ("prefill_context_parallel_size", "vLLM prefill context parallel:"),
            ("decode_context_parallel_size", "vLLM decode context parallel:"),
        ):
            value = session.integer(
                f"vllm.measurements.{name}.topology.{key}",
                label,
                default=int(default_topology[key]),
                minimum=1,
            )
            if value is BACK:
                return False
            topology[key] = int(value)
        topology["gpu_group_size"] = (
            topology["tensor_parallel_size"]
            * topology["pipeline_parallel_size"]
            * topology["prefill_context_parallel_size"]
        )
        topology["distributed_executor_backend"] = "mp"
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
    return True


def mip_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    del context
    measurements = _mapping_copy(session.state.collection("vllm_measurements"))
    workloads = OrderedDict(
        (
            name,
            {
                "isl": int(raw["prefill_seq_len"]),
                "osl": int(raw["generation_seq_len"]),
                "batch_size": int(raw["batch_size"]),
                "concurrency": int(raw["max_num_seqs"]),
            },
        )
        for name, raw in measurements.items()
    )
    default_goal = resolver.resolve_default("mip.goal_metric", "params")
    default_goal_metric = str(default_goal.value)
    default_goal_value = resolver.resolve_default("mip.goal_value", "75%").value
    default_objective = str(
        resolver.resolve_default(
            "mip.objective", "metrics.cosine_embedding_loss_hidden_states"
        ).value
    )
    default_num_solutions = int(
        resolver.resolve_default("mip.num_solutions", 8).value
    )
    workload_metrics = {"memory", "runtime"}
    goal_choices = [
        ("Parameters", "params"),
        ("Active parameters", "active_params"),
    ]
    if workloads:
        goal_choices.extend(
            [
                ("Memory at a workload", "memory"),
                ("Runtime at a workload", "runtime"),
            ]
        )
    available_goal_metrics = {value for _, value in goal_choices}
    if default_goal_metric not in available_goal_metrics:
        if default_goal_metric == "throughput":
            raise SetupError(
                "MIP throughput goals are not supported by setup v2; "
                "use runtime, memory, parameters, or active parameters."
            )
        if default_goal_metric in workload_metrics and not workloads:
            raise SetupError(
                f"Default MIP goal {default_goal_metric!r} requires an enabled "
                "vLLM measurement."
            )
        raise SetupError(f"Unsupported default MIP goal: {default_goal_metric!r}")
    action = _section_action(
        session,
        "mip",
        "Create independent goals, then add internal constraints, variants, and matrices.",
        {
            "goal": {
                "metric": default_goal_metric,
                "bound": default_goal_value,
                "workload": (
                    next(iter(workloads))
                    if default_goal_metric
                    in workload_metrics
                    and workloads
                    else None
                ),
            },
            "objective": default_objective,
            "internal_constraints": [],
            "variants": [],
            "search_space": {"depth": "all", "embedding": "all"},
            "solver": {
                "backend": "auto",
                "num_solutions": default_num_solutions,
                "min_hamming_distance": 2,
                "max_seconds_per_solution": 60,
            },
            "homogeneous": {"enabled": True, "keep": "all"},
        },
    )
    if action is BACK:
        return False
    if action == "defaults":
        default_workload = None
        if default_goal_metric in workload_metrics:
            default_workload = next(iter(workloads))
        goal_bound = yaml.safe_load(str(default_goal_value))
        goal_config = (
            {"at": {str(default_workload): goal_bound}}
            if default_workload is not None
            else goal_bound
        )
        run_id = f"{default_goal_metric}-{str(default_goal_value).replace('%', '')}"
        session.state.set_collection(
            "mip_config",
            {
                "defaults": {},
                "workloads": workloads,
                "runs": {
                    run_id: {
                        "constraints": {default_goal_metric: goal_config},
                        "objectives": [
                            {
                                "metric": default_objective,
                                "direction": "minimize",
                            }
                        ],
                        "search_space": {"depth": "all", "embedding": "all"},
                        "solver": {
                            "backend": "auto",
                            "num_solutions": default_num_solutions,
                            "min_hamming_distance": 2,
                            "max_seconds_per_solution": 60,
                        },
                        "homogeneous": {
                            "enabled": True,
                            "keep": "all",
                            "rank_by": "objective",
                        },
                    }
                },
            },
        )
        return True
    runs: OrderedDict[str, Any] = OrderedDict()
    while True:
        goal_metric = session.select(
            "mip.run.goal.metric",
            "Main MIP goal:",
            goal_choices,
            default=default_goal_metric,
        )
        if goal_metric is BACK:
            return False
        goal_value = session.text(
            "mip.run.goal.value",
            "Goal bound (for example 75%, 22.5B, or 5000):",
            default=str(default_goal_value),
        )
        if goal_value is BACK:
            return False
        workload = None
        if goal_metric in workload_metrics:
            workload = session.select(
                "mip.run.goal.workload",
                "Goal workload:",
                list(workloads),
                default=next(iter(workloads)),
            )
            if workload is BACK:
                return False
        run_id = session.text(
            "mip.run.id",
            "MIP run ID:",
            default=f"{goal_metric}-{str(goal_value).replace('%', '')}",
        )
        if run_id is BACK:
            return False
        objective = session.select(
            "mip.run.objective",
            "Objective:",
            [
                ("Cosine embedding distance", "metrics.cosine_embedding_loss_hidden_states"),
                ("Language-model loss", "metrics.lm_loss"),
            ],
            default=str(
                resolver.resolve(
                    "mip.objective", "metrics.cosine_embedding_loss_hidden_states"
                ).value
            ),
        )
        if objective is BACK:
            return False
        goal_bound: Any = yaml.safe_load(str(goal_value))
        goal_config = (
            {"at": {str(workload): goal_bound}} if workload is not None else goal_bound
        )
        constraints = {str(goal_metric): goal_config}
        variants: OrderedDict[str, Any] = OrderedDict()
        if action == "customize":
            while True:
                extra = session.confirm(
                    "mip.run.constraint.add", "Add an internal AND constraint?", default=False
                )
                if extra is BACK:
                    return False
                if not extra:
                    break
                metric = session.text(
                    "mip.run.constraint.metric",
                    "Constraint metric (friendly name or stats.*):",
                    default="experts",
                )
                mode = session.select(
                    "mip.run.constraint.mode",
                    "Bound type:",
                    ["min", "max", "eq", "range"],
                    default="max",
                )
                raw = session.text(
                    "mip.run.constraint.value",
                    "Bound value (YAML scalar/list):",
                    default="[64, 96]" if mode == "range" else "75%",
                )
                if BACK in (metric, mode, raw):
                    return False
                constraints[str(metric)] = {str(mode): yaml.safe_load(str(raw))}
            add_variant = session.confirm(
                "mip.run.variant.add", "Add a variant or matrix sweep?", default=False
            )
            if add_variant is BACK:
                return False
            while add_variant:
                variant_id = session.text(
                    "mip.run.variant.id", "Variant ID:", default="sweep"
                )
                matrix_path = session.text(
                    "mip.run.variant.matrix.path",
                    "Matrix path (embedding, depth, constraints.*, solver.*, homogeneous.*):",
                    default="depth",
                )
                matrix_values = session.text(
                    "mip.run.variant.matrix.values",
                    "Matrix values as a YAML list:",
                    default="[0, 2, 4]",
                )
                if BACK in (variant_id, matrix_path, matrix_values):
                    return False
                variants[str(variant_id)] = {
                    "matrix": {str(matrix_path): yaml.safe_load(str(matrix_values))}
                }
                add_variant = session.confirm(
                    "mip.run.variant.add_more", "Add another variant?", default=False
                )
                if add_variant is BACK:
                    return False
        run = {
            "constraints": constraints,
            "objectives": [{"metric": str(objective), "direction": "minimize"}],
            "search_space": {"depth": "all", "embedding": "all"},
            "solver": {
                "backend": "auto",
                "num_solutions": default_num_solutions,
                "min_hamming_distance": 2,
                "max_seconds_per_solution": 60,
            },
            "homogeneous": {"enabled": True, "keep": "all", "rank_by": "objective"},
        }
        if variants:
            run["variants"] = variants
        runs[str(run_id)] = run
        more = session.confirm("mip.run.add", "Add another independent MIP run?", default=False)
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
    return True


def _post_mip_strategy(node: NodeDraft) -> str:
    return (
        "persistent_pool"
        if node.node_type == "evaluation" and node.input_id == "source"
        else "sharded"
    )


def post_mip_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    mip = _mapping_copy(session.state.collection("mip_config"))
    runs = _mapping_copy(mip.get("runs"))
    sequence = int(session.state.get_field("data.sequence_length", 4096))
    measurements = _mapping_copy(session.state.collection("vllm_measurements"))
    first_measurement = next(iter(measurements.values()), {})
    serving = {
        "input_tokens": int(first_measurement.get("prefill_seq_len", sequence)),
        "output_tokens": int(first_measurement.get("generation_seq_len", 1024)),
        "concurrency": int(first_measurement.get("max_num_seqs", 1)),
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
                    "strategy": "sharded",
                    "instances": int(
                        session.state.get_field(
                            "infrastructure.gpus_per_node", 8
                        )
                    ),
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
                    moe=context["model"].inventory.moe,
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
                    moe=context["model"].inventory.moe,
                )
                if configured is BACK:
                    return False
                config = {
                    **configured,
                    "concurrency": [configured["concurrency"]],
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
            more = session.confirm(
                f"post_mip.{run_id}.node.add", "Add another node?", default=True
            )
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


def _serving_setting_prompt(
    session: WizardSession,
    prefix: str,
    defaults: Mapping[str, Any],
    *,
    moe: bool,
) -> Any:
    """Ask the complete AIPerf workload and serving-only parallel setting."""

    values = {}
    for name, label, default in (
        ("input_tokens", "Serving input sequence length (ISL):", defaults["input_tokens"]),
        ("output_tokens", "Serving output sequence length (OSL):", defaults["output_tokens"]),
        ("concurrency", "Serving concurrency:", defaults["concurrency"]),
    ):
        value = session.integer(
            f"{prefix}.{name}", label, default=int(default), minimum=1
        )
        if value is BACK:
            return BACK
        values[name] = int(value)
    topology = {}
    for name, label in (
        ("tensor_parallel_size", "Serving tensor parallel (TP):"),
        ("pipeline_parallel_size", "Serving pipeline parallel (PP):"),
        ("prefill_context_parallel_size", "Serving prefill context parallel (CP):"),
        ("decode_context_parallel_size", "Serving decode context parallel (CP):"),
    ):
        value = session.integer(
            f"{prefix}.topology.{name}", label, default=1, minimum=1
        )
        if value is BACK:
            return BACK
        topology[name] = int(value)
    ep = 1
    if moe:
        ep = session.integer(
            f"{prefix}.topology.expert_parallel_size",
            "Serving expert parallel (EP):",
            default=1,
            minimum=1,
        )
        if ep is BACK:
            return BACK
    topology.update(
        {
            "data_parallel_size": int(ep),
            "expert_parallel_size": int(ep),
            "gpu_group_size": (
                topology["tensor_parallel_size"]
                * topology["pipeline_parallel_size"]
                * topology["prefill_context_parallel_size"]
                * int(ep)
            ),
            "distributed_executor_backend": "mp",
        }
    )
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
    cpu_partition = session.state.get_field(
        "infrastructure.runner.slurm.partition_cpu", None
    )
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
            tp = int(topology.get("tensor_parallel_size", 1))
            pp = int(topology.get("pipeline_parallel_size", 1))
            cp = int(topology.get("prefill_context_parallel_size", 1))
            ep = int(topology.get("expert_parallel_size", 1))
            dp = int(topology.get("data_parallel_size", ep))
            entry["parallel"] = {
                "tp": tp,
                "cp": cp,
                "pp": pp,
                "ep": ep,
                "dp_shard": ep if ep > 1 else 1,
                "dp_replicate": max(1, dp // ep),
                "sequence_parallel": False,
            }
        else:
            profile = (
                _profile_prompt(session, registry, stage_id, model)
                if customize
                else registry.reuse(registry.names()[0], consumer=stage_id)
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
            key = (
                "local_batch_size"
                if node.node_type == "global_kd"
                else "micro_batch_size"
            )
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


def output_review_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    del context
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
    print(
        yaml.safe_dump(
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
                "vllm_measurements": session.state.collection("vllm_measurements"),
                "mip": session.state.collection("mip_config"),
                "post_mip": session.state.collection("post_mip_flows"),
            },
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
    bypass_section,
    replacement_scoring_section,
    vllm_section,
    mip_section,
    post_mip_section,
    output_review_section,
)


def _fresh_state(
    backend: PromptBackend,
    defaults_path: Optional[Path],
) -> WizardState:
    while True:
        value = backend.text("Campaign directory:", "")
        if value is BACK:
            continue
        path = Path(str(value)).expanduser()
        if str(path):
            return WizardState.start(path, defaults_path=defaults_path)


def _refresh_legacy_state(state: WizardState) -> None:
    pruning = deepcopy(state.collection("pruning") or {})
    first_measurement = next(
        iter(_mapping_copy(state.collection("vllm_measurements")).items()),
        ("serving-default", {}),
    )
    measurement_id, measurement = first_measurement
    runtime = {
        "vllm_enabled": bool(state.collection("vllm_measurements")),
        "granularity": measurement.get("granularity", "subblock"),
        "workload_id": measurement_id,
        "isl": int(measurement.get("prefill_seq_len", state.get_field("data.sequence_length", 4096))),
        "osl": int(measurement.get("generation_seq_len", 1024)),
        "concurrency": int(measurement.get("max_num_seqs", 1)),
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
                "partition_cpu": state.get_field(
                    "infrastructure.runner.slurm.partition_cpu", None
                ),
                "time_limit": state.get_field(
                    "infrastructure.runner.slurm.time_limit", "4:00:00"
                ),
                "qos": state.get_field("infrastructure.runner.slurm.qos", None),
                "max_nodes": state.get_field(
                    "infrastructure.runner.slurm.max_nodes", 64
                ),
            },
        },
        "execution_contract": {
            "repository": state.get_field(
                "infrastructure.execution_contract.repository", str(Path.cwd())
            ),
            "venv": state.get_field("infrastructure.execution_contract.venv", ".venv"),
            "container": state.get_field(
                "infrastructure.execution_contract.container", None
            ),
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
            key: first.get(key, 1)
            for key in ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")
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
                "modality": state.get_field("data.modality", "text"),
                "layout": state.get_field("data.layout", "fixed"),
                "sequence_length": state.get_field("data.sequence_length", 4096),
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
    resume: Optional[Path],
    defaults_path: Optional[Path],
    backend: Optional[PromptBackend] = None,
) -> Path:
    """Run setup v2, save every answer, validate bundles, and never launch jobs."""

    backend = backend or InteractiveBackend()
    print("Welcome to Puzzletron setup v2 — defaults are local, control is per stage.")
    if resume is None:
        state = _fresh_state(backend, defaults_path)
    else:
        state = WizardState.resume(resume)
    selected_defaults = defaults_path or state.defaults_path
    resolver = _resolver(state, selected_defaults)
    session = WizardSession(state, backend)
    context: dict[str, Any] = {}
    if state.payload.get("model", {}).get("source"):
        saved = state.payload["model"]
        context["model"] = inspect_model(str(saved["source"]))

    index = 0
    while index < len(SECTION_BUILDERS):
        builder = SECTION_BUILDERS[index]
        completed = builder(session, resolver, context)
        if completed:
            index += 1
        else:
            index = max(0, index - 1)
    _refresh_legacy_state(state)
    build_bundles_v2(state.campaign_dir, state)
    return state.campaign_dir
