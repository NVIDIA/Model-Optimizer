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
        "num_solutions": 1000,
    },
}

STATIC_MODEL_STAGES = (
    "depth_importance",
    "width_importance",
    "bypass",
    "replacement_scoring",
)

_CUSTOM_MODEL_SOURCE = "__custom_model_source__"

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


def _model_family_choices() -> list[PromptChoice]:
    return [
        PromptChoice("Custom", _CUSTOM_MODEL_SOURCE),
        *[PromptChoice(group, group) for group, _ in SUPPORTED_MODEL_GROUPS],
    ]


def _model_choices_for_family(family: str) -> list[PromptChoice]:
    for group, models in SUPPORTED_MODEL_GROUPS:
        if group == family:
            return [PromptChoice(short_name, url) for short_name, url in models]
    raise SetupError(f"Unknown supported model family: {family}")


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
    resolved = resolver.resolve(path, fallback)
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
) -> Any:
    session.begin(section)
    print(f"\n[{section}] {summary}")
    return session.select(
        f"{section}.action",
        f"{section.replace('_', ' ').title()}:",
        [
            ("Use defaults", "defaults"),
            ("Customize", "customize"),
            ("Review current values", "review"),
        ],
        default="defaults",
    )


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
            _model_family_choices(),
            default=_CUSTOM_MODEL_SOURCE,
        )
        if family is BACK:
            return BACK
        if family == _CUSTOM_MODEL_SOURCE:
            source = _text_field(
                session,
                resolver,
                "model.source",
                "Local model path or Hugging Face URL:",
            )
            if source is BACK:
                continue
            return source
        source = session.select(
            "model.source_model",
            f"{family} model:",
            _model_choices_for_family(str(family)),
        )
        if source is not BACK:
            return source


def model_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    session.begin("model")
    while True:
        source = _select_model_source(session, resolver)
        if source is BACK:
            return False
        try:
            source = normalize_model_source(str(source))
        except SetupError as error:
            print(f"  {error}")
            continue
        session.state.set_field("model.source", source, source="user")
        revision = None
        if not Path(source).exists():
            revision = _text_field(
                session, resolver, "model.revision", "Hugging Face revision:", "main"
            )
            if revision is BACK:
                continue
        break
    model = inspect_model(str(source), str(revision) if revision else None)
    session.state.set_model(model.to_dict(), model.inventory.to_dict())
    context["model"] = model
    print(
        f"  Detected {model.inventory.family}; {model.inventory.num_layers} layers, "
        f"{model.inventory.num_sublayers} sublayers, MoE={model.inventory.moe}."
    )
    return True


def data_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    action = _section_action(
        session,
        "data",
        "Choose a local dataset path or a Hugging Face dataset and its sequence layout.",
    )
    if action is BACK:
        return False
    if action == "review":
        print(
            yaml.safe_dump(
                {
                    key: record.effective
                    for key, record in session.state.records().items()
                    if key.startswith("data.")
                }
            )
        )
        return data_section(session, resolver, context)
    source_default = resolver.resolve("data.source", session.state.get_field("data.source", ""))
    if action == "defaults" and not source_default.value:
        action = "customize"
    if action == "customize":
        source = _text_field(
            session,
            resolver,
            "data.source",
            "Local dataset path or Hugging Face URL:",
        )
        if source is BACK:
            return False
        source = normalize_dataset_source(str(source))
        finding = infer_dataset_modality(source)
        modality = session.select(
            "data.modality",
            f"Data modality ({finding.evidence}):",
            [("Text", "text"), ("Multimodal", "multimodal")],
            default=finding.modality if finding.modality != "unknown" else "text",
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
            default="fixed",
        )
        if layout is BACK:
            return False
        sequence = _integer_field(
            session,
            resolver,
            "data.sequence_length",
            "Calibration sequence length:",
            4096,
        )
        if sequence is BACK:
            return False
        session.state.set_field("data.source", source, source="user")
        session.state.set_field("data.modality", modality, source="user")
        session.state.set_field("data.layout", layout, source="user")
    else:
        for path, fallback in (
            ("data.source", ""),
            ("data.modality", "text"),
            ("data.layout", "fixed"),
            ("data.sequence_length", 4096),
        ):
            _record_default(session.state, resolver, path, fallback)
    return True


def infrastructure_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    action = _section_action(
        session,
        "infrastructure",
        "Configure the worker contract and cluster facts before stage allocations.",
    )
    if action is BACK:
        return False
    if action == "review":
        print(
            yaml.safe_dump(
                {
                    key: record.effective
                    for key, record in session.state.records().items()
                    if key.startswith("infrastructure.")
                }
            )
        )
        return infrastructure_section(session, resolver, context)
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
    if action == "defaults":
        for path, fallback in paths:
            _record_default(session.state, resolver, path, fallback)
        commands = resolver.resolve(
            "infrastructure.execution_contract.prerun_commands", []
        )
        session.state.set_field(
            "infrastructure.execution_contract.prerun_commands",
            list(commands.value or ()),
            source=commands.source,
        )
        session.state.set_field(
            "infrastructure.execution_contract.postrun_commands", [], source="builtin"
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
    action = _section_action(
        session,
        "pruning",
        "Choose depth and every descriptor-supported pruning-axis domain.",
    )
    if action is BACK:
        return False
    if action == "review" and session.state.collection("pruning"):
        print(yaml.safe_dump(session.state.collection("pruning")))
        return pruning_section(session, resolver, context)

    inventory = model.inventory
    defaults = deepcopy(BUILTINS["pruning"])
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
            "Maximum number to remove:",
            default=min(4, count - 1),
            minimum=0,
            maximum=count - 1,
        )
        if remove is BACK:
            return False
    else:
        granularity = str(resolver.resolve("pruning.depth_granularity", "subblock").value)
        count = (
            inventory.num_sublayers if granularity == "subblock" else inventory.num_layers
        )
        remove = min(
            int(resolver.resolve("pruning.depth_remove", min(4, count - 1)).value),
            count - 1,
        )
    axes = {}
    for axis in inventory.axes:
        axis_values = list(axis.values)
        default_values = axis_values[: min(2, len(axis_values))]
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
            "sort_sanity": False,
            "replacement_granularity": granularity,
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


def pre_mip_stages_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    action = _section_action(
        session,
        "pre_mip_stages",
        "Each stage has independent instances, strategy, parallelism, and batch behavior.",
    )
    if action is BACK:
        return False
    if action == "review" and session.state.collection("stage_resources"):
        print(
            yaml.safe_dump(
                {
                    "profiles": session.state.collection("parallel_profiles"),
                    "stages": session.state.collection("stage_resources"),
                    "batches": session.state.collection("stage_batches"),
                }
            )
        )
        return pre_mip_stages_section(session, resolver, context)
    registry = ResourceProfileRegistry.from_dict(
        session.state.collection("parallel_profiles") or {}
    )
    resources = _mapping_copy(session.state.collection("stage_resources"))
    batches = _mapping_copy(session.state.collection("stage_batches"))
    gpus_per_node = int(session.state.get_field("infrastructure.gpus_per_node", 8))
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
            strategy = session.select(
                f"stages.{stage_id}.strategy",
                "Execution strategy:",
                ["single", "persistent_pool", "sharded"],
                default=(
                    "persistent_pool"
                    if stage_id in {"depth_importance", "replacement_scoring"}
                    else "single"
                ),
            )
            if strategy is BACK:
                return False
            instances = session.integer(
                f"stages.{stage_id}.instances",
                "Independent model instances/workers:",
                default=(gpus_per_node if strategy != "single" else 1),
                minimum=1,
            )
            if instances is BACK:
                return False
            requested_batch = session.integer(
                f"stages.{stage_id}.batch",
                "Local/micro batch size:",
                default=8,
                minimum=1,
            )
            if requested_batch is BACK:
                return False
        else:
            profile = (
                registry.reuse(registry.names()[0], consumer=stage_id)
                if registry.names()
                else registry.create(ParallelProfile(stage_id), consumer=stage_id)
            )
            strategy = (
                "persistent_pool"
                if stage_id in {"depth_importance", "replacement_scoring"}
                else "single"
            )
            instances = gpus_per_node if strategy != "single" else 1
            requested_batch = 8
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
        batch_paths = {
            "depth_importance": "depth_importance.micro_batch_size",
            "width_importance": "pruning.micro_batch_size",
            "bypass": "bypass.training.micro_batch_size",
            "replacement_scoring": "replacement_scoring.micro_batch_size",
        }
        batches[batch_paths[stage_id]] = resolution.effective
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


def vllm_section(session: WizardSession, resolver: DefaultsResolver, context: dict) -> bool:
    action = _section_action(
        session,
        "vllm",
        "Define zero or more exact workload/topology measurement points.",
    )
    if action is BACK:
        return False
    if action == "review":
        print(yaml.safe_dump(session.state.collection("vllm_measurements") or {}))
        return vllm_section(session, resolver, context)
    enabled_default = bool(resolver.resolve("vllm.enabled", False).value)
    if action == "defaults" and not enabled_default:
        session.state.set_collection("vllm_measurements", {})
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
            key: int(resolver.resolve(f"vllm.{key}", fallback).value)
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
            default=str(resolver.resolve("vllm.granularity", "subblock").value),
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
                default=1,
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
    action = _section_action(
        session,
        "mip",
        "Create independent goals, then add internal constraints, variants, and matrices.",
    )
    if action is BACK:
        return False
    if action == "review" and session.state.collection("mip_config"):
        print(yaml.safe_dump(session.state.collection("mip_config")))
        return mip_section(session, resolver, context)
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
    runs: OrderedDict[str, Any] = OrderedDict()
    while True:
        goal_metric = session.select(
            "mip.run.goal.metric",
            "Main MIP goal:",
            [
                ("Parameters", "params"),
                ("Active parameters", "active_params"),
                ("Memory at a workload", "memory"),
                ("Runtime at a workload", "runtime"),
                ("Throughput at a workload", "throughput"),
            ],
            default=str(resolver.resolve("mip.goal_metric", "params").value),
        )
        if goal_metric is BACK:
            return False
        if goal_metric in {"memory", "runtime", "throughput"} and not workloads:
            print("  This goal requires a named vLLM measurement. Choose another goal or go Back.")
            continue
        goal_value = session.text(
            "mip.run.goal.value",
            "Goal bound (for example 75%, 22.5B, or 5000):",
            default=str(resolver.resolve("mip.goal_value", "75%").value),
        )
        if goal_value is BACK:
            return False
        workload = None
        if goal_metric in {"memory", "runtime", "throughput"}:
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
                "num_solutions": int(resolver.resolve("mip.num_solutions", 1000).value),
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


def post_mip_section(
    session: WizardSession, resolver: DefaultsResolver, context: dict
) -> bool:
    action = _section_action(
        session,
        "post_mip",
        "For each MIP run, accept the recommended flow or add typed nodes one by one.",
    )
    if action is BACK:
        return False
    if action == "review" and session.state.collection("post_mip_flows"):
        print(yaml.safe_dump(session.state.collection("post_mip_flows")))
        return post_mip_section(session, resolver, context)
    mip = _mapping_copy(session.state.collection("mip_config"))
    runs = _mapping_copy(mip.get("runs"))
    editor = PostMIPFlowEditor(runs)
    sequence = int(session.state.get_field("data.sequence_length", 4096))
    measurements = _mapping_copy(session.state.collection("vllm_measurements"))
    first_measurement = next(iter(measurements.values()), {})
    serving = {
        "input_tokens": int(first_measurement.get("prefill_seq_len", sequence)),
        "output_tokens": int(first_measurement.get("generation_seq_len", 1024)),
        "concurrency": int(first_measurement.get("max_num_seqs", 1)),
    }
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
                config = {"eval_samples": 128, "block_size": sequence}
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
                config = {"max_steps": 128}
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
        strategy_default = (
            "persistent_pool"
            if node.node_type == "evaluation" and node.input_id == "source"
            else "sharded"
        )
        strategy = strategy_default
        instances = gpus_per_node
        if customize:
            strategy = session.select(
                f"{stage_id}.strategy",
                "Execution strategy:",
                ["single", "persistent_pool", "sharded"],
                default=strategy_default,
            )
            if strategy is BACK:
                return BACK
            instances = session.integer(
                f"{stage_id}.instances",
                "Independent model instances/workers:",
                default=1 if strategy == "single" else gpus_per_node,
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
                    "Local/micro batch size:",
                    default=profile.batch_unit,
                    minimum=1,
                )
                if requested is BACK:
                    return BACK
            resolved = resolve_batch(int(requested), profile)
            config = dict(node.config)
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
    action = _section_action(
        session,
        "output",
        "Review effective values, then validate and generate both bundles.",
    )
    if action is BACK:
        return False
    default_root = str(session.state.campaign_dir / "results")
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
    pruning_section,
    pre_mip_stages_section,
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
        context["model"] = inspect_model(
            str(saved["source"]), saved.get("requested_revision")
        )

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
