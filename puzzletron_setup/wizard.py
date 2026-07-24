# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive normal and detailed Puzzletron campaign question flows."""

from __future__ import annotations

import math
import re
import shlex
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from . import SetupError
from .inspection import (
    InspectedModel,
    infer_dataset_modality,
    inspect_model,
    normalize_dataset_source,
    normalize_model_source,
)
from .profiles import CandidateCounts, UnsupportedModelError, count_candidate_options
from .prompts import PromptSession
from .state import AnswerState

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["run_wizard"]

_MESH_KEYS = ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")
_DEFAULT_MIP_SOLUTION_COUNT = 3
_DEFAULT_HOMOGENEOUS_SOLUTIONS_PER_SCENARIO = 8


def _default(state: AnswerState, section: str, key: str, fallback: Any) -> Any:
    return state.section(section).get(key, fallback)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def _parse_yaml_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        raise ValueError("Enter one scalar value.")
    return parsed


def _parse_yaml_value(value: str) -> Any:
    return yaml.safe_load(value)


def _print_inventory(model: InspectedModel) -> None:
    inventory = model.inventory
    kind = "MoE" if inventory.moe else "dense"
    modality = "multimodal" if inventory.multimodal else "text"
    print(f"\nDetected {inventory.family} / {inventory.descriptor} ({kind}, {modality})")
    print(
        f"Layers: {inventory.num_layers}; sublayers: {inventory.num_sublayers}; "
        f"layer types: {dict(inventory.layer_counts)}"
    )
    if inventory.facts:
        print("Geometry: " + ", ".join(f"{key}={value}" for key, value in inventory.facts.items()))
    print("Supported axes:")
    for axis in inventory.axes:
        print(
            f"  - {axis.axis_id}: teacher={axis.teacher_value}, "
            f"alignment={axis.alignment}, choices={list(axis.values)}"
        )


def _inspect_fresh_model(prompts: PromptSession, state: AnswerState) -> InspectedModel:
    prompts.begin(state, "model")
    while True:
        source = prompts.text(
            "Local model path or Hugging Face URL:",
            description=(
                "Examples: ../models/model, /checkpoints/model, or "
                "https://huggingface.co/Qwen/Qwen3.5-0.8B. A trailing slash is optional."
            ),
            validate=lambda value: bool(str(value).strip()) or "Enter a model path or URL.",
        ).strip()
        try:
            source = normalize_model_source(source)
        except SetupError as error:
            print(f"Could not inspect that model: {error}")
            prompts.reset()
            continue
        is_local = Path(source).exists()
        revision = None
        if not is_local:
            revision = (
                prompts.text(
                    "Hugging Face revision:",
                    default="main",
                    description="The resolved immutable commit is stored in the campaign.",
                ).strip()
                or None
            )
        try:
            model = inspect_model(source, revision)
        except UnsupportedModelError as error:
            state.payload["model"] = {
                "source": source,
                "requested_revision": revision,
                "detected_model_types": list(error.model_types),
                "detected_architectures": list(error.architectures),
                "supported": False,
            }
            state.save()
            raise
        except SetupError as error:
            print(f"Could not inspect that model: {error}")
            prompts.reset()
            continue
        state.set_model(model.to_dict(), model.inventory.to_dict())
        return model


def _resume_model(state: AnswerState) -> InspectedModel:
    saved = state.payload.get("model") or {}
    source = saved.get("source")
    if not source:
        raise SetupError("Resume state does not contain a model source.")
    model = inspect_model(str(source), saved.get("requested_revision"))
    normalized = yaml.safe_load(yaml.safe_dump(model.to_dict(), sort_keys=False))
    if normalized != saved:
        print("Model configuration or resolved revision changed; later answers were invalidated.")
        state.invalidate_after("model")
        state.set_model(model.to_dict(), model.inventory.to_dict())
    return model


def _ask_data(prompts: PromptSession, state: AnswerState, model: InspectedModel) -> None:
    saved = state.section("data")
    if saved:
        try:
            normalized = normalize_dataset_source(str(saved.get("source", "")))
        except SetupError as error:
            print(f"Saved dataset source is no longer valid: {error}")
            state.invalidate_after("model")
        else:
            if normalized != saved.get("source"):
                state.record("data", "source", normalized)
            return
    prompts.begin(state, "data")
    while True:
        source = prompts.text(
            "Local dataset path or Hugging Face URL:",
            description=(
                "Examples: ../datasets/data, /datasets/data, or "
                "https://huggingface.co/datasets/owner/repository. "
                "A trailing slash is optional."
            ),
            validate=lambda value: bool(str(value).strip()) or "Enter a dataset path or URL.",
        ).strip()
        try:
            source = normalize_dataset_source(source)
        except SetupError as error:
            print(f"Could not inspect that dataset: {error}")
            prompts.reset()
            continue
        break
    finding = infer_dataset_modality(source)
    print(f"Dataset modality finding: {finding.modality} — {finding.evidence}.")
    default_modality = (
        finding.modality
        if finding.modality in {"text", "multimodal"}
        else "multimodal"
        if model.inventory.multimodal
        else "text"
    )
    modality_choices = [("Text", "text")]
    if model.inventory.multimodal:
        modality_choices.append(("Multimodal", "multimodal"))
    modality = prompts.select(
        "Data modality:",
        modality_choices,
        default=default_modality,
        description="Confirm or correct the inferred modality.",
    )
    layout = prompts.select(
        "Dataset layout:",
        [
            ("Fixed-length", "fixed"),
            ("Packed variable-length", "packed_varlen"),
            ("Padded", "padded"),
        ],
        default="packed_varlen",
    )
    sequence_length = prompts.integer("Calibration sequence length:", default=4096)
    state.record_many(
        "data",
        {
            "source": source,
            "modality": modality,
            "modality_finding": finding.modality,
            "modality_evidence": finding.evidence,
            "layout": layout,
            "sequence_length": sequence_length,
        },
    )


def _axis_defaults(values: tuple[int, ...], teacher: int) -> list[int]:
    half = min(values, key=lambda value: (abs(value - teacher / 2), -value))
    defaults = list(values)
    if half not in defaults:
        defaults.append(half)
    return defaults


def _vllm_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]:
    width_count = counts.effective_vllm_width_count
    if width_count == 1:
        subblock_label = f"Sublayer — {counts.vllm_subblock_total} unique configurations"
        block_label = f"Whole block — {counts.vllm_block_total} unique configurations"
    else:
        subblock_label = (
            f"Sublayer — {counts.vllm_subblock} configurations/width, "
            f"{counts.vllm_subblock_total} total across {width_count} widths"
        )
        block_label = (
            f"Whole block — {counts.vllm_block} configurations/width, "
            f"{counts.vllm_block_total} total across {width_count} widths"
        )
    return [
        (subblock_label, "subblock"),
        (block_label, "block"),
    ]


def _replacement_count_label(
    label: str,
    per_width: int,
    total: int,
    width_count: int,
) -> str:
    if width_count == 1:
        return f"{label} — {total} solutions"
    return f"{label} — {per_width} solutions/width, {total} total across {width_count} widths"


def _replacement_granularity_choices(
    counts: CandidateCounts,
) -> list[tuple[str, str]]:
    return [
        (
            _replacement_count_label(
                "Subblock",
                counts.replacement_subblock_per_width,
                counts.replacement_subblock_total,
                counts.width_count,
            ),
            "subblock",
        ),
        (
            _replacement_count_label(
                "Whole block",
                counts.replacement_block_per_width,
                counts.replacement_block_total,
                counts.width_count,
            ),
            "block",
        ),
    ]


def _ask_pruning(prompts: PromptSession, state: AnswerState, model: InspectedModel) -> None:
    if state.section("pruning"):
        return
    prompts.begin(state, "pruning")
    inventory = model.inventory
    depth_granularity = prompts.select(
        "Depth pruning granularity:",
        [("Sublayer", "subblock"), ("Whole layer", "block")],
        default="subblock",
        description=(
            "Sublayer search is more accurate but costlier and needs heterogeneous deployment; "
            "support outside vLLM may be limited."
        ),
    )
    depth_count = (
        inventory.num_sublayers if depth_granularity == "subblock" else inventory.num_layers
    )
    depth_remove = prompts.integer(
        "Maximum number to remove:",
        default=max(1, depth_count // 4),
        minimum=0,
        maximum=depth_count - 1,
        description=f"There are {depth_count} selectable {depth_granularity}s.",
    )

    axes = {}
    for axis in inventory.axes:
        choices = [(str(value), value) for value in axis.values]
        selected = prompts.checkbox(
            f"Values to sort for {axis.label}:",
            choices,
            defaults=_axis_defaults(axis.values, axis.teacher_value),
            description=(
                f"Teacher={axis.teacher_value}; alignment={axis.alignment}; "
                "at most 16 valid values are shown."
            ),
            validate=lambda values: bool(values) or "Select at least one value.",
        )
        axes[axis.axis_id] = {
            "enabled": True,
            "teacher_value": axis.teacher_value,
            "values": sorted({int(value) for value in selected}, reverse=True),
            "alignment": axis.alignment,
        }
    counts = count_candidate_options(model.config, inventory, axes)

    sort_sanity = prompts.confirm("Run sorting sanity checks?", default=False)
    bypass_sanity = prompts.confirm("Run bypass sanity checks?", default=False)
    bypass_enabled = prompts.confirm(
        "Run local distillation (bypass)?",
        default=True,
        description=(
            "Local distillation improves candidate-ranking accuracy, but adds a training pass "
            "and additional GPU cost."
        ),
    )
    bypass = {"enabled": bypass_enabled, "sanity": bypass_sanity}
    data = state.section("data")
    if bypass_enabled:
        bypass.update(
            {
                "granularity": prompts.select(
                    "Bypass granularity:",
                    [("Sublayer", "subblock"), ("Whole layer", "block")],
                    default="subblock",
                ),
                "samples": prompts.integer("Bypass samples:", default=4096),
                "sequence_length": prompts.integer(
                    "Bypass sequence length:", default=int(data["sequence_length"])
                ),
                "batch_size": prompts.integer("Bypass batch size:", default=8),
            }
        )

    print(f"The model has {inventory.num_layers} blocks and {inventory.num_sublayers} subblocks.")
    replacement_granularity = prompts.select(
        "Replace and score one block or subblock at a time?",
        _replacement_granularity_choices(counts),
        default="subblock",
    )
    width_samples = prompts.integer("Width-importance samples:", default=32 * 1024)
    replacement_samples = prompts.integer("Replacement-scoring samples:", default=128)
    state.record_many(
        "pruning",
        {
            "depth_granularity": depth_granularity,
            "depth_remove": depth_remove,
            "sort_all_axes": True,
            "axes": axes,
            "sort_sanity": sort_sanity,
            "bypass": bypass,
            "width_importance_samples": width_samples,
            "replacement_granularity": replacement_granularity,
            "replacement_samples": replacement_samples,
        },
    )


def _ask_runtime(
    prompts: PromptSession,
    state: AnswerState,
    model: InspectedModel,
) -> None:
    if state.section("runtime"):
        return
    prompts.begin(state, "runtime")
    data = state.section("data")
    enabled = prompts.confirm(
        "Collect vLLM runtime statistics?",
        default=False,
        description=(
            "These measurements are approximate; parameter or memory objectives are often "
            "nearly as useful."
        ),
    )
    sequence_length = int(data["sequence_length"])
    isl = sequence_length
    osl = 1024
    concurrency = 1
    granularity = "subblock"
    if enabled:
        axes = state.section("pruning")["axes"]
        counts = count_candidate_options(model.config, model.inventory, axes)
        granularity = prompts.select(
            "vLLM measurement granularity:",
            _vllm_granularity_choices(counts),
            default="subblock",
            description=(
                "Sublayer measurements isolate attention/GDN and FFN costs; whole-block "
                "measurements capture their runtime interaction."
            ),
        )
        isl = prompts.integer("Serving input sequence length (ISL):", default=sequence_length)
        osl = prompts.integer("Serving output sequence length (OSL):", default=1024)
        concurrency = prompts.integer("Serving concurrency:", default=1)
    state.record_many(
        "runtime",
        {
            "vllm_enabled": enabled,
            "granularity": granularity,
            "workload_id": "serving-default",
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
        },
    )


def _infer_objective_direction(metric: str) -> tuple[str, str]:
    """Infer whether a metric is naturally minimized or maximized."""
    normalized = metric.lower()
    minimize_tokens = (
        "loss",
        "distance",
        "divergence",
        "latency",
        "memory",
        "param",
        "cost",
        "error",
        "perplexity",
    )
    maximize_tokens = ("throughput", "accuracy", "reward", "precision", "recall", "f1")
    if token := next((token for token in minimize_tokens if token in normalized), None):
        return "minimize", f"detected {token!r} metric"
    if token := next((token for token in maximize_tokens if token in normalized), None):
        return "maximize", f"detected {token!r} metric"
    return "minimize", "unknown metric; using the safe default"


def _ask_objectives(prompts: PromptSession, detailed: bool) -> list[dict[str, str]]:
    choices = [
        ("Cosine embedding distance", "metrics.cosine_embedding_loss_hidden_states"),
        ("Language-model loss", "metrics.lm_loss"),
        ("Custom metric", "custom"),
    ]
    objectives = []
    while True:
        objective_checkpoint = prompts.checkpoint()
        metric = prompts.select("MIP objective metric:", choices, default=choices[0][1])
        if metric == "custom":
            metric = prompts.text("Objective metric path:", default="metrics.lm_loss").strip()
        direction, reason = _infer_objective_direction(metric)
        print(f"Objective direction: {direction} ({reason}).")
        if any(objective["metric"] == metric for objective in objectives):
            print(f"Objective metric {metric!r} is already included; choose a different metric.")
            prompts.rewind(objective_checkpoint)
            continue
        objectives.append({"metric": metric, "direction": direction})
        if not detailed or not prompts.confirm("Add another objective?", default=False):
            return objectives


def _constraint_value(prompts: PromptSession, label: str, default: str) -> Any:
    while True:
        raw = prompts.text(label, default=default)
        try:
            return _parse_yaml_scalar(raw)
        except (ValueError, yaml.YAMLError) as error:
            print(f"Invalid value: {error}")


def _add_constraint(constraints: dict[str, Any], metric: str, value: Any, workload_id: str) -> None:
    key = "runtime" if metric == "latency" else metric
    if key in {"memory", "runtime", "throughput"}:
        constraints[key] = {"at": {workload_id: value}}
    else:
        constraints[key] = value


def _ask_mip(prompts: PromptSession, state: AnswerState) -> None:
    if state.section("mip"):
        return
    prompts.begin(state, "mip")
    runtime = state.section("runtime")
    detailed = state.detailed
    runs: OrderedDict[str, Any] = OrderedDict()
    while True:
        basis_choices = [("Parameters", "params"), ("Memory", "memory")]
        if runtime["vllm_enabled"]:
            basis_choices.append(("Latency", "latency"))
        basis = prompts.select(
            "Primary MIP constraint:",
            basis_choices,
            default="params",
        )
        percentage = prompts.integer(
            "Target percentage of the teacher:", default=75, minimum=1, maximum=100
        )
        objectives = _ask_objectives(prompts, detailed)
        homogeneous = prompts.confirm(
            "Include homogeneous candidates?",
            default=True,
            description="Homogeneous architectures are generally easier to deploy.",
        )
        constraints: dict[str, Any] = {}
        _add_constraint(
            constraints,
            basis,
            f"{percentage}%",
            str(runtime["workload_id"]),
        )
        if detailed:
            while prompts.confirm("Add an extra constraint?", default=False):
                metric_choices = [
                    ("Parameters", "params"),
                    ("Memory", "memory"),
                    ("Experts", "experts"),
                ]
                if runtime["vllm_enabled"]:
                    metric_choices.extend([("Latency", "latency"), ("Throughput", "throughput")])
                metric = prompts.select(
                    "Constraint metric:",
                    metric_choices,
                    default="params",
                )
                mode = prompts.select(
                    "Constraint form:",
                    [("Maximum", "max"), ("Minimum", "min"), ("Range", "range")],
                    default="max",
                )
                if mode == "range":
                    low = _constraint_value(prompts, "Range minimum:", "128")
                    high = _constraint_value(prompts, "Range maximum:", "144")
                    value = {"range": [low, high]}
                else:
                    bound = _constraint_value(prompts, f"Constraint {mode}:", "75%")
                    value = {mode: bound}
                _add_constraint(constraints, metric, value, str(runtime["workload_id"]))

        search_space: dict[str, Any] = {"depth": "all", "embedding": "all"}
        if detailed:
            while True:
                raw = prompts.text(
                    "Depth loop (all, integer, or YAML range mapping):", default="all"
                )
                try:
                    search_space["depth"] = _parse_yaml_value(raw)
                    break
                except yaml.YAMLError as error:
                    print(f"Invalid YAML value: {error}")
        while True:
            raw = prompts.text(
                "Embedding widths for this MIP run (all or YAML list):",
                default="all",
            )
            try:
                search_space["embedding"] = _parse_yaml_value(raw)
                break
            except yaml.YAMLError as error:
                print(f"Invalid YAML value: {error}")
        solver = {
            "backend": "auto",
            "num_solutions": _DEFAULT_MIP_SOLUTION_COUNT,
            "min_hamming_distance": 3,
            "max_seconds_per_solution": 120,
        }
        if detailed:
            solver.update(
                {
                    "backend": prompts.select(
                        "MIP solver backend:",
                        ["auto", "pulp", "cuopt"],
                        default="auto",
                    ),
                    "num_solutions": prompts.integer(
                        "MIP solution count:", default=_DEFAULT_MIP_SOLUTION_COUNT
                    ),
                    "min_hamming_distance": prompts.integer(
                        "Minimum Hamming distance:", default=3, minimum=1
                    ),
                    "max_seconds_per_solution": prompts.integer(
                        "Seconds per solution:", default=120
                    ),
                }
            )
        homogeneous_config: dict[str, Any] = {
            "enabled": homogeneous,
            "keep": _DEFAULT_HOMOGENEOUS_SOLUTIONS_PER_SCENARIO,
            "rank_by": "objective",
        }
        if detailed and homogeneous:
            homogeneous_config["keep"] = prompts.integer(
                "Homogeneous candidates to keep per width/depth scenario:",
                default=_DEFAULT_HOMOGENEOUS_SOLUTIONS_PER_SCENARIO,
            )
            rank_by = prompts.select(
                "Rank homogeneous candidates by:",
                [("MIP objective", "objective"), ("Constraint closeness", "closeness")],
                default="objective",
            )
            homogeneous_config["rank_by"] = (
                "objective"
                if rank_by == "objective"
                else {"constraint_closeness": {"weights": {basis: 1}}}
            )
        base_id = _slug(f"{basis}-{percentage:03d}")
        run_id = base_id
        suffix = 2
        while run_id in runs:
            run_id = f"{base_id}-{suffix}"
            suffix += 1
        runs[run_id] = {
            "constraints": constraints,
            "objectives": objectives,
            "search_space": search_space,
            "homogeneous": homogeneous_config,
            "solver": solver,
        }
        print(f"Added MIP run {run_id}.")
        if not prompts.confirm("Add another independent MIP run?", default=False):
            break
    state.record_many("mip", {"runs": dict(runs)})


def _default_flow(
    run_id: str,
    run: Mapping[str, Any],
    runtime: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    prefix: str,
    objective: Mapping[str, Any] | None = None,
    include_initial_filter: bool = True,
) -> dict[str, Any]:
    def node_id(name: str) -> str:
        return f"{prefix}{name}"

    initial_top_k: Any = 128
    if run.get("homogeneous", {}).get("enabled"):
        initial_top_k = {"homogeneous": 100, "heterogeneous": 28}
    initial = node_id("initial_filter")
    online = node_id("online_eval")
    best_kl = node_id("best_kl")
    materialized = node_id("materialized")
    serving = node_id("serving")
    fastest = node_id("fastest")
    short_kd = node_id("short_kd")
    final_eval = node_id("final_eval")
    best = node_id("best")
    if objective is None:
        objective = next(iter(run.get("objectives") or ()), {})
    objective_metric = str(objective.get("metric", "metrics.lm_loss"))
    objective_direction = str(objective.get("direction", "minimize"))
    nodes: OrderedDict[str, Any] = OrderedDict()
    if include_initial_filter:
        nodes[initial] = {
            "type": "filter",
            "mode": "top_k",
            "metric": "mip.score",
            "direction": objective_direction,
            "top_k": initial_top_k,
        }
    online_node = {
        "type": "evaluation",
        "config": {
            "eval_samples": 128,
            "block_size": int(data["sequence_length"]),
        },
    }
    if include_initial_filter:
        online_node["input"] = initial
    nodes[online] = online_node
    nodes.update(
        [
            (
                best_kl,
                {
                    "type": "filter",
                    "input": online,
                    "mode": "top_k",
                    "metric": f"{online}.kl_div",
                    "direction": "minimize",
                    "top_k": 32,
                },
            ),
            (materialized, {"type": "materialize", "input": best_kl}),
            (
                serving,
                {
                    "type": "aiperf",
                    "input": materialized,
                    "config": {
                        "input_tokens": int(runtime["isl"]),
                        "output_tokens": int(runtime["osl"]),
                        "concurrency": [int(runtime["concurrency"])],
                        "use_server_token_count": True,
                        "benchmark_timeout": 600,
                    },
                },
            ),
            (
                fastest,
                {
                    "type": "filter",
                    "input": serving,
                    "mode": "top_k",
                    "metric": f"{serving}.request_throughput",
                    "direction": "maximize",
                    "top_k": 4,
                },
            ),
            (
                short_kd,
                {
                    "type": "global_kd",
                    "input": fastest,
                    "config": {"max_steps": 128},
                },
            ),
            (
                final_eval,
                {
                    "type": "evaluation",
                    "input": short_kd,
                    "config": {
                        "eval_samples": 128,
                        "block_size": int(data["sequence_length"]),
                    },
                },
            ),
            (
                best,
                {
                    "type": "filter",
                    "input": final_eval,
                    "mode": "top_k",
                    "metric": f"{final_eval}.kl_div",
                    "direction": "minimize",
                    "top_k": 1,
                },
            ),
        ]
    )
    return {
        "source": {"run": run_id, "variants": "all", "objectives": [objective_metric]},
        "nodes": nodes,
    }


def _ask_filter_config(prompts: PromptSession, available_metrics: list[str]) -> dict[str, Any]:
    mode = prompts.select(
        "Filter mode:",
        ["top_k", "threshold", "pareto", "aggregate_rank"],
        default="top_k",
    )
    config: dict[str, Any] = {"mode": mode}
    if mode in {"top_k", "threshold"}:
        config["metric"] = prompts.text(
            "Metric path:", default=available_metrics[-1] if available_metrics else "mip.score"
        )
    if mode == "top_k":
        config["direction"] = prompts.select(
            "Direction:", ["minimize", "maximize"], default="minimize"
        )
        config["top_k"] = prompts.integer("Keep how many?", default=32)
    elif mode == "threshold":
        bound = prompts.select("Threshold bound:", ["min", "max"], default="min")
        config[bound] = _constraint_value(prompts, f"Threshold {bound}:", "0")
    else:
        metrics = prompts.text(
            "Comma-separated metric paths:",
            default=",".join(available_metrics[-2:] or ["mip.score"]),
        )
        config["metrics"] = [
            {"metric": metric.strip(), "direction": "minimize"}
            for metric in metrics.split(",")
            if metric.strip()
        ]
        if mode == "aggregate_rank":
            config["top_k"] = prompts.integer("Keep how many?", default=1)
    return config


def _custom_flow(
    prompts: PromptSession,
    run_id: str,
    runtime: Mapping[str, Any],
    data: Mapping[str, Any],
    used_ids: set[str],
) -> dict[str, Any]:
    nodes: OrderedDict[str, Any] = OrderedDict()
    available_metrics = ["mip.score"]
    transformer_nodes = []
    while True:
        node_type = prompts.select(
            "Post-MIP node type:",
            [
                "filter",
                "evaluation",
                "materialize",
                "aiperf",
                "global_kd",
                "manual_filter",
                ("PTQ (reserved; not executable yet)", "ptq"),
                ("Downstream evaluation (reserved; not executable yet)", "downstream_evaluation"),
            ],
            default="filter",
        )
        default_id = _slug(f"{run_id}-{node_type}-{len(nodes) + 1}")
        node_id = _slug(prompts.text("Node ID:", default=default_id))
        if node_id in used_ids:
            print(f"Node ID {node_id!r} is already used; choose another.")
            continue
        input_id = prompts.select(
            "Candidate input:",
            ["source", *nodes.keys()],
            default=next(reversed(nodes)) if nodes else "source",
        )
        node: dict[str, Any] = {"type": node_type}
        if input_id != "source":
            node["input"] = input_id
        if node_type not in {"filter", "manual_filter"}:
            model_source = prompts.select(
                "Model artifact source:",
                ["latest", "origin", *transformer_nodes],
                default="latest",
            )
            if model_source != "latest":
                node["model_source"] = model_source
        if node_type == "filter":
            node.update(_ask_filter_config(prompts, available_metrics))
        elif node_type == "manual_filter":
            node["prompt"] = prompts.text(
                "Reviewer prompt:", default="Select candidates to continue"
            )
        elif node_type == "evaluation":
            node["config"] = {
                "eval_samples": prompts.integer("Evaluation samples:", default=128),
                "block_size": prompts.integer(
                    "Evaluation sequence length:", default=int(data["sequence_length"])
                ),
            }
            available_metrics.append(f"{node_id}.kl_div")
        elif node_type == "aiperf":
            node["config"] = {
                "input_tokens": prompts.integer("AIPerf ISL:", default=int(runtime["isl"])),
                "output_tokens": prompts.integer("AIPerf OSL:", default=int(runtime["osl"])),
                "concurrency": [
                    prompts.integer("AIPerf concurrency:", default=int(runtime["concurrency"]))
                ],
                "use_server_token_count": True,
                "benchmark_timeout": prompts.integer(
                    "Per-candidate AIPerf timeout (seconds):", default=600
                ),
            }
            available_metrics.append(f"{node_id}.request_throughput")
        elif node_type == "global_kd":
            node["config"] = {"max_steps": prompts.integer("Global KD steps:", default=128)}
        elif node_type in {"ptq", "downstream_evaluation"}:
            print(
                f"{node_type} records the reserved interface, but current orchestration "
                "validation will report it as unimplemented."
            )
            node["config"] = {}
        if node_type in {"materialize", "global_kd", "ptq"}:
            transformer_nodes.append(node_id)
        nodes[node_id] = node
        used_ids.add(node_id)
        if not prompts.confirm("Add another post-MIP node?", default=True):
            break
    return {"source": {"run": run_id, "variants": "all", "objectives": "all"}, "nodes": nodes}


def _ask_post_mip(prompts: PromptSession, state: AnswerState) -> None:
    if state.section("post_mip"):
        return
    prompts.begin(state, "post_mip")
    mip_runs = state.section("mip")["runs"]
    runtime = state.section("runtime")
    data = state.section("data")
    flows = OrderedDict()
    used_ids: set[str] = set()
    for run_id, run in mip_runs.items():
        if state.detailed and prompts.confirm(
            f"Build a custom post-MIP flow for {run_id}?", default=False
        ):
            flow = _custom_flow(prompts, run_id, runtime, data, used_ids)
            flows[run_id] = flow
        else:
            objectives = list(run.get("objectives") or ())
            for objective_index, objective in enumerate(objectives):
                flow_id = run_id
                if len(objectives) > 1:
                    flow_id = f"{run_id}-objective-{objective_index + 1}"
                prefix = "" if not used_ids else f"{flow_id}_"
                flow = _default_flow(
                    run_id,
                    run,
                    runtime,
                    data,
                    prefix=prefix,
                    objective=objective,
                    include_initial_filter=state.detailed,
                )
                used_ids.update(flow["nodes"])
                flows[flow_id] = flow
    state.record_many("post_mip", {"flows": dict(flows)})


def _mesh_product(mesh: Mapping[str, int]) -> int:
    return math.prod(int(mesh[key]) for key in _MESH_KEYS if key != "ep")


def _ask_mesh(
    prompts: PromptSession,
    name: str,
    *,
    moe: bool,
    defaults: Mapping[str, int] | None = None,
) -> dict[str, int]:
    defaults = defaults or {}
    print(f"{name} parallel mesh:")
    labels = {
        "tp": "Tensor parallel (TP)",
        "cp": "Context parallel (CP)",
        "pp": "Pipeline parallel (PP)",
        "dp_shard": "FSDP shard degree (DP shard)",
        "dp_replicate": "Data-parallel replicas (DP replicate)",
        "ep": "Expert parallel (EP; must divide DP shard)",
    }
    mesh = {
        key: prompts.integer(f"  {labels[key]}:", default=int(defaults.get(key, 1)))
        for key in _MESH_KEYS
        if key != "ep"
    }
    mesh["ep"] = (
        prompts.integer(f"  {labels['ep']}:", default=int(defaults.get("ep", 1))) if moe else 1
    )
    if not moe:
        print("  Expert parallel (EP): 1 (not applicable to dense models).")
    print(
        "  Allocation: "
        f"{mesh['tp']} TP * {mesh['cp']} CP * {mesh['pp']} PP * "
        f"{mesh['dp_shard']} DP shard * {mesh['dp_replicate']} DP replicate "
        f"= {_mesh_product(mesh)} GPU(s) per model instance."
    )
    return mesh


def _print_mesh_guidance(model: InspectedModel) -> None:
    print(
        "Parallel allocation uses TP * CP * PP * DP shard * DP replicate GPUs per model "
        "instance. EP overlays DP shard and is not multiplied into the GPU count."
    )
    if model.inventory.moe:
        print("For this MoE model, DP shard % EP must equal 0 and EP must divide the experts.")
    print(
        "Choose batch, microbatch, and gradient-accumulation counts that schedule evenly "
        "across PP stages and the resulting data-parallel lanes; for example, the number "
        "of pipeline microbatches should be divisible by PP."
    )


def _validate_mesh(mesh: Mapping[str, int], model: InspectedModel, name: str) -> None:
    if any(int(value) < 1 for value in mesh.values()):
        raise SetupError(f"{name} mesh dimensions must all be positive.")
    if int(mesh["dp_shard"]) % int(mesh["ep"]):
        raise SetupError(
            f"{name} DP shard={mesh['dp_shard']} must be divisible by EP={mesh['ep']}."
        )
    experts = model.inventory.facts.get("num_experts")
    if experts and int(experts) % int(mesh["ep"]):
        raise SetupError(f"{name} EP={mesh['ep']} does not divide the model's {experts} experts.")


def _resource_rows(
    state: AnswerState,
    common: Mapping[str, int],
    bypass: Mapping[str, int],
    global_kd: Mapping[str, int],
    gpus_per_node: int,
    workers: Mapping[str, int],
) -> list[dict[str, Any]]:
    from .bundle import _post_mip_candidate_limits

    rows = []
    single_gpu = dict.fromkeys(_MESH_KEYS, 1)
    flows = state.section("post_mip").get("flows") or {}
    candidate_limits = _post_mip_candidate_limits({"post_mip": {"flows": flows}})

    def post_mip_instances(
        node_type: str,
        worker_limit: int,
        default: int,
        *,
        source_worker_limit: int | None = None,
    ) -> int:
        instances = []
        for flow_id, flow in flows.items():
            for node_id, node in (flow.get("nodes") or {}).items():
                if node.get("type") != node_type:
                    continue
                node_worker_limit = (
                    source_worker_limit
                    if source_worker_limit is not None
                    and str(node.get("input", "source")) == "source"
                    else worker_limit
                )
                candidate_limit = candidate_limits[f"post.{flow_id}.{node_id}"]
                instances.append(
                    node_worker_limit
                    if candidate_limit is None
                    else min(node_worker_limit, candidate_limit)
                )
        return max(instances, default=default)

    pool_workers = int(workers["pool"])
    sharded_workers = int(workers["sharded"])
    aiperf_workers = int(workers.get("aiperf", 2 * gpus_per_node))
    stages = [
        ("importance/scoring", common, int(workers["pool"])),
        ("vLLM stats", single_gpu, int(workers["sharded"])),
        (
            "AIPerf",
            single_gpu,
            post_mip_instances("aiperf", aiperf_workers, aiperf_workers),
        ),
        (
            "evaluation",
            common,
            post_mip_instances(
                "evaluation",
                sharded_workers,
                sharded_workers,
                source_worker_limit=pool_workers if not state.detailed else None,
            ),
        ),
        ("bypass", bypass, 1),
        ("global KD", global_kd, post_mip_instances("global_kd", sharded_workers, 1)),
    ]
    for name, mesh, instances in stages:
        gpus = _mesh_product(mesh)
        rows.append(
            {
                "stage": name,
                "instances": instances,
                "gpus_per_instance": gpus,
                "nodes": math.ceil(gpus * instances / gpus_per_node),
            }
        )
    return rows


def _print_resource_rows(rows: list[Mapping[str, Any]]) -> None:
    print("\nDerived resource plan:")
    print(f"{'Stage':30} {'Instances':>10} {'GPU/instance':>14} {'Nodes':>8}")
    for row in rows:
        print(
            f"{row['stage']!s:30} {int(row['instances']):10d} "
            f"{int(row['gpus_per_instance']):14d} {int(row['nodes']):8d}"
        )


def _ask_infrastructure(
    prompts: PromptSession,
    state: AnswerState,
    model: InspectedModel,
) -> None:
    if state.section("infrastructure"):
        return
    prompts.begin(state, "infrastructure")
    runner_kind = prompts.select(
        "Cluster type:",
        [("Slurm", "slurm"), ("SSH bare metal", "baremetal")],
        default="slurm",
    )
    repository = prompts.text("Repository path on workers:", default=str(Path.cwd()))
    venv = prompts.text("Python virtual environment on workers:", default=".venv")
    container = prompts.text("Container image/path (blank for none):", default="").strip()
    mounts = prompts.text(
        "Container mounts (blank for none):",
        default="",
        description=(
            "Use /host/path:/container/path; separate multiple mounts with commas, for "
            "example /lustre:/lustre,/datasets:/datasets."
        ),
    ).strip()
    prerun = prompts.text("Pre-run commands separated by ';;' (blank for none):", default="")
    gpus_per_node = prompts.integer("GPUs per node:", default=8)
    _print_mesh_guidance(model)
    mesh_checkpoint = prompts.checkpoint()
    while True:
        common_mesh = _ask_mesh(prompts, "Common", moe=model.inventory.moe)
        if not state.detailed or prompts.confirm("Reuse the common mesh for bypass?", default=True):
            bypass_mesh = dict(common_mesh)
        else:
            bypass_mesh = _ask_mesh(
                prompts, "Bypass", moe=model.inventory.moe, defaults=common_mesh
            )
        if not state.detailed or prompts.confirm(
            "Reuse the common mesh for global KD?", default=True
        ):
            global_kd_mesh = dict(common_mesh)
        else:
            global_kd_mesh = _ask_mesh(
                prompts, "Global KD", moe=model.inventory.moe, defaults=common_mesh
            )
        try:
            _validate_mesh(common_mesh, model, "Common")
            _validate_mesh(bypass_mesh, model, "Bypass")
            _validate_mesh(global_kd_mesh, model, "Global KD")
        except SetupError as error:
            print(f"Invalid parallel mesh: {error}")
            prompts.rewind(mesh_checkpoint)
            continue
        break
    workers = {
        "pool": prompts.integer("Workers for persistent-pool stages:", default=gpus_per_node),
        "sharded": prompts.integer("Workers for sharded stages:", default=gpus_per_node),
        "aiperf": 2 * gpus_per_node,
    }
    runner: dict[str, Any] = {"kind": runner_kind}
    if runner_kind == "slurm":
        cpu_partition = prompts.text(
            "CPU partition (blank to use one GPU node for CPU/IO stages):",
            default="",
            description=(
                "Used for conversion, tokenization, sorting, block-library construction, "
                "MIP, filters, and materialization. Leave blank when the cluster has no "
                "CPU-only partition."
            ),
        ).strip()
        runner["slurm"] = {
            "account": prompts.text("Slurm account:", default=""),
            "partition_interactive": prompts.text("Interactive partition:", default="interactive"),
            "partition_batch": prompts.text("Batch partition:", default="batch"),
            "partition_cpu": cpu_partition or None,
            "time_limit": prompts.text("Default time limit:", default="4:00:00"),
            "qos": prompts.text("QoS (blank for none):", default="").strip() or None,
            "max_nodes": prompts.integer("Maximum simultaneous nodes:", default=64),
        }
    else:
        hosts = []
        while True:
            hostname = prompts.text(
                "SSH hostname:", default="localhost" if not hosts else ""
            ).strip()
            hosts.append(
                {
                    "hostname": hostname,
                    "gpus": prompts.integer("GPUs on this host:", default=gpus_per_node),
                }
            )
            if not prompts.confirm("Add another SSH host?", default=False):
                break
        runner["inventory"] = {
            "hosts": hosts,
            "rendezvous_host": hosts[0]["hostname"],
            "rendezvous_port_base": 29500,
        }
    rows = _resource_rows(
        state,
        common_mesh,
        bypass_mesh,
        global_kd_mesh,
        gpus_per_node,
        workers,
    )
    _print_resource_rows(rows)
    state.record_many(
        "infrastructure",
        {
            "runner": runner,
            "execution_contract": {
                "repository": repository,
                "venv": venv,
                "container": container or None,
                "container_mounts": mounts or None,
                "prerun_commands": [item.strip() for item in prerun.split(";;") if item.strip()],
                "postrun_commands": [],
            },
            "gpus_per_node": gpus_per_node,
            "meshes": {
                "common": common_mesh,
                "bypass": bypass_mesh,
                "global_kd": global_kd_mesh,
            },
            "workers": workers,
            "resource_rows": rows,
        },
    )


def _ask_output(prompts: PromptSession, state: AnswerState) -> None:
    if state.section("output"):
        return
    prompts.begin(state, "output")
    default_root = state.path.parent / "results"
    result_root = prompts.text("Campaign results location:", default=str(default_root))
    state.record_many("output", {"result_root": result_root})


def _print_orchestration_commands(campaign_dir: Path, state: AnswerState) -> None:
    """Print exact dry-run and launch commands for both generated bundles."""
    infrastructure = state.section("infrastructure")
    contract = infrastructure.get("execution_contract") or {}
    orchestrator = Path(str(contract.get("repository", "."))) / "examples/puzzletron/orchestrate.py"
    print("\nRun orchestration with these exact commands:")
    for budget in ("smoke", "production"):
        bundle = campaign_dir / budget
        base = [
            "python",
            str(orchestrator),
            "--experiment",
            str(bundle / "experiment.yaml"),
            "--runner",
            str(bundle / "runner.yaml"),
            "--execution",
            str(bundle / "execution.yaml"),
            "--stage",
            "full",
        ]
        print(f"\n{budget.title()} dry-run:\n{shlex.join([*base, '--dry-run'])}")
        print(f"{budget.title()} launch:\n{shlex.join(base)}")


def run_wizard(*, detailed: bool, resume: Path | None) -> Path:
    """Run the complete question flow and generate both campaign bundles."""
    prompts = PromptSession()
    print("Welcome to Puzzletron — build a model-aware pruning campaign.")
    if resume is None:
        campaign_dir = Path(
            prompts.text(
                "Campaign directory:",
                validate=lambda value: bool(str(value).strip()) or "Enter a campaign directory.",
            )
        ).expanduser()
        state = AnswerState.start(campaign_dir, detailed=detailed)
        model = _inspect_fresh_model(prompts, state)
    else:
        state = AnswerState.resume(resume)
        if detailed and not state.detailed:
            print("Resume uses the mode stored in answers.yaml; continuing in normal mode.")
        model = _resume_model(state)
    _print_inventory(model)
    _ask_data(prompts, state, model)
    _ask_pruning(prompts, state, model)
    _ask_runtime(prompts, state, model)
    _ask_mip(prompts, state)
    _ask_post_mip(prompts, state)
    _ask_infrastructure(prompts, state, model)
    _ask_output(prompts, state)
    prompts.begin(state, "output")
    generate_checkpoint = prompts.checkpoint()
    if not prompts.confirm("Generate smoke and production bundles?", default=True):
        prompts.rewind(generate_checkpoint)
        raise SetupError(f"Answers saved at {state.path}; no bundle was generated.")

    from .bundle import build_bundles

    build_bundles(state.path.parent, state.payload)
    _print_orchestration_commands(state.path.parent, state)
    return state.path.parent
