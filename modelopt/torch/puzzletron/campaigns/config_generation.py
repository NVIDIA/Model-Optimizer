"""Generate one canonical, stage-parallel Puzzletron config per campaign model."""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..anymodel.capabilities import resolve_score_method
from ..anymodel.registry import resolve_descriptor
from .activation_passes import compile_activation_passes
from .preflight import CampaignPreflight, ModelPreflight
from .schema import CampaignModel, CrossModelCampaign


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BASE_CONFIG = _REPO_ROOT / "examples/puzzletron/configs/base.yaml"
_TEXT_DATASET = Path(os.environ.get("PUZZLETRON_TEXT_DATASET", "data/puzzle_kd_text"))
_VLM_DATASET = Path(
    os.environ.get("PUZZLETRON_MULTIMODAL_DATASET", "data/pinned_intersyn")
)

ConfigLoader = Callable[[CampaignModel, ModelPreflight], Any]


def _equivalence_tolerances(descriptor: Any) -> dict[str, float]:
    """Resolve model-format-aware functional gates from the descriptor."""
    return dict(descriptor.checkpoint_equivalence_tolerances())


def _first(config: Any, *names: str) -> Any:
    for name in names:
        value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if value is not None:
            return value
    return None


def _teacher_value(axis_id: str, config: Any) -> int | None:
    query_heads = _first(config, "num_attention_heads")
    kv_heads = _first(config, "num_key_value_heads", "num_global_key_value_heads")
    mapping = {
        "hidden_width": _first(config, "hidden_size"),
        "ple_width": _first(config, "hidden_size_per_layer_input"),
        "ffn_intermediate": _first(config, "intermediate_size"),
        "kv_groups": kv_heads,
        "q_heads_per_group": (
            int(query_heads) // int(kv_heads) if query_heads and kv_heads else None
        ),
        "gdn_key_groups": _first(config, "linear_num_key_heads"),
        "gdn_value_heads_per_group": (
            int(_first(config, "linear_num_value_heads"))
            // int(_first(config, "linear_num_key_heads"))
            if _first(config, "linear_num_value_heads")
            and _first(config, "linear_num_key_heads")
            else None
        ),
        "gdn_key_head_dim": _first(config, "linear_key_head_dim"),
        "gdn_value_head_dim": _first(config, "linear_value_head_dim"),
        "moe_experts": _first(config, "num_experts", "num_local_experts", "n_routed_experts"),
        "moe_expert_intermediate": _first(
            config, "moe_intermediate_size", "expert_intermediate_size", "intermediate_size"
        ),
        "moe_shared_expert_intermediate": _first(
            config, "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"
        ),
        "moe_latent_dim": _first(config, "moe_latent_size"),
        "moe_top_k": _first(config, "experts_per_token", "num_experts_per_tok"),
        "mamba_heads": _first(config, "mamba_num_heads"),
        "mamba_head_dim": _first(config, "mamba_head_dim"),
        "mla_q_lora_rank": _first(config, "q_lora_rank"),
        "mla_kv_lora_rank": _first(config, "kv_lora_rank"),
        "mla_heads": _first(config, "num_attention_heads"),
    }
    value = mapping.get(axis_id)
    return int(value) if value is not None else None


def _reduced_value(axis_id: str, teacher: int) -> int | None:
    if teacher < 2:
        return None
    reduced = teacher // 2
    alignment = 1
    if axis_id in {
        "hidden_width",
        "ple_width",
        "ffn_intermediate",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
    }:
        alignment = 128
    elif axis_id.endswith("head_dim") or "lora_rank" in axis_id:
        alignment = 16
    if reduced < alignment or reduced % alignment:
        alignment = 1
    reduced = (reduced // alignment) * alignment
    return reduced if 0 < reduced < teacher else None


def _embedding_alignment(hidden_axis: dict[str, Any] | None) -> int:
    """Choose a conservative alignment legal for every configured width.

    Cross-family hidden sizes are not uniformly multiples of 128 (GPT-OSS is
    2880, for example).  Use the greatest common divisor with the preferred
    128-channel tile so generated widths and the descriptor validation contract
    cannot disagree.  This retains alignment 128 for Qwen/Llama and naturally
    selects 32 for 2880 -> 1440.
    """
    if hidden_axis is None:
        return 1
    widths = [int(hidden_axis["teacher_value"]), *map(int, hidden_axis.get("values", ()))]
    return max(1, math.gcd(128, *widths))


def _block_axis_value(axis_id: str, axis: Any, block: Any) -> int | None:
    subblock = block.get_subblock(axis.subblock_kind)
    if subblock is None or getattr(subblock, "no_op", False):
        return None
    if axis_id == "q_heads_per_group":
        query_heads = getattr(subblock, "num_query_heads", None)
        kv_heads = getattr(subblock, "num_kv_heads", None)
        if query_heads is None or kv_heads is None:
            return None
        return int(query_heads) // int(kv_heads)
    value = getattr(subblock, axis.field, None)
    return int(value) if isinstance(value, int) else None


def _heterogeneous_block_axis_values(
    descriptor: Any,
    config: Any,
    capabilities: Any,
) -> dict[str, tuple[int, ...]]:
    """Discover true per-layer values from a composable decoder contract."""

    contract_factory = getattr(descriptor, "generic_decoder_contract", None)
    if not callable(contract_factory) or contract_factory(config) is None:
        return {}
    lm_config = descriptor.get_language_model_config(config)
    if _first(lm_config, "num_hidden_layers") is None:
        # Capability-only fixture/config metadata cannot prove heterogeneity;
        # retain the ordinary global option instead of inventing layer values.
        return {}
    from ..anymodel.converter.generic_decoder import GenericDecoderConverter

    blocks = GenericDecoderConverter.create_block_configs(descriptor, config)
    result: dict[str, tuple[int, ...]] = {}
    for axis_id, axis in capabilities.axes.items():
        if axis.subblock_kind == "model" or axis.variant_only:
            continue
        values = tuple(
            value
            for block in blocks
            if (value := _block_axis_value(axis_id, axis, block)) is not None
        )
        unique = tuple(sorted(set(values)))
        if len(unique) > 1:
            result[axis_id] = unique
    return result


def _axis_inventory(descriptor, config: Any) -> tuple[dict[str, dict], list[dict]]:
    capabilities = descriptor.puzzletron_capabilities(config)
    lm_config = descriptor.get_language_model_config(config)
    heterogeneous_values = _heterogeneous_block_axis_values(
        descriptor,
        config,
        capabilities,
    )
    allowed = {
        "hidden_width",
        "ple_width",
        "ffn_intermediate",
        "kv_groups",
        "q_heads_per_group",
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
        "moe_experts",
        "moe_expert_intermediate",
        "moe_shared_expert_intermediate",
        "moe_latent_dim",
        "moe_top_k",
        "mamba_heads",
        "mamba_head_dim",
        "mla_q_lora_rank",
        "mla_kv_lora_rank",
        "mla_heads",
        "sliding_window_size",
    }
    search_axes: dict[str, dict] = {}
    activation_axes: list[dict] = []
    for axis_id, axis in capabilities.axes.items():
        if axis_id not in allowed:
            continue
        if not axis.vllm_export:
            raise ValueError(
                f"Campaign search axis {axis_id!r} from {capabilities.descriptor_name} "
                "has no verified vLLM control path"
            )
        if axis.variant_only:
            if axis.values:
                search_axes[axis_id] = {
                    "enabled": True,
                    "values": list(axis.values),
                }
            else:
                teacher = _teacher_value(axis_id, lm_config)
                reduced = _reduced_value(axis_id, teacher) if teacher is not None else None
                if teacher is None or reduced is None:
                    continue
                search_axes[axis_id] = {
                    "enabled": True,
                    "teacher_value": teacher,
                    "values": [reduced],
                }
            continue
        if not axis.sortable:
            continue
        teacher = _teacher_value(axis_id, lm_config)
        if teacher is None:
            continue
        reduced = _reduced_value(axis_id, teacher)
        if reduced is None:
            continue
        if axis_id in heterogeneous_values:
            teacher_values = list(heterogeneous_values[axis_id])
            if not any(_reduced_value(axis_id, value) is not None for value in teacher_values):
                continue
            search_axes[axis_id] = {
                "enabled": True,
                "teacher_values": teacher_values,
                "ratios": [0.5],
            }
        else:
            search_axes[axis_id] = {
                "enabled": True,
                "teacher_value": teacher,
                "values": [reduced],
            }
        method = resolve_score_method(axis)
        if axis_id == "ffn_intermediate" and "iterative" in axis.score_hooks:
            method = "iterative"
        entry = {"axis_id": axis_id, "method": method}
        if axis.magnitude_fallback is not None:
            entry["magnitude_fallback"] = dataclasses.asdict(axis.magnitude_fallback)
        activation_axes.append(entry)
    return search_axes, activation_axes


def _stage_parallel(model: CampaignModel, *, capabilities) -> dict[str, Any]:
    """Map the campaign's independent EP/FSDP axes to AutoModel's EP overlay."""

    topology = model.topology
    return {
        "tp": topology.tp,
        "cp": topology.cp,
        "pp": topology.pp,
        "ep": topology.ep,
        "dp_shard": topology.ep,
        "dp_replicate": topology.fsdp,
        "sequence_parallel": (
            topology.tp > 1 and capabilities.parallelism.sequence_parallel
        ),
        "pipeline_schedule": "1f1b",
    }


def _deferred_sort_axes(capabilities, activation_axes: list[dict]) -> list[str]:
    """Defer sortable structural axes that did not produce a scoring pass."""
    active = {str(entry["axis_id"]) for entry in activation_axes}
    return sorted(
        axis_id
        for axis_id, axis in capabilities.axes.items()
        if axis.sortable and not axis.variant_only and axis_id not in active
    )


def _model_config(
    campaign: CrossModelCampaign,
    model: CampaignModel,
    record: ModelPreflight,
    source_config: Any,
    *,
    output_root: Path,
) -> dict[str, Any]:
    base = yaml.safe_load(_BASE_CONFIG.read_text())
    base.pop("defaults", None)
    # Descriptor names are a derived runtime property of the converted checkpoint.  Keep
    # campaign configs family-agnostic so they cannot silently force a stale adapter when
    # checkpoint metadata or the registered native implementation changes.
    base.pop("descriptor", None)
    base.get("pruning", {}).pop("descriptor", None)
    descriptor = resolve_descriptor(
        source_config,
        descriptor_override=record.descriptor_name,
    ).descriptor
    search_axes, activation_axes = _axis_inventory(descriptor, source_config)
    equivalence_tolerances = _equivalence_tolerances(descriptor)
    activation_passes = compile_activation_passes(
        descriptor,
        source_config,
        activation_axes,
    )
    deferred_sort_axes = _deferred_sort_axes(
        descriptor.puzzletron_capabilities(source_config), activation_axes
    )
    hidden_axis = search_axes.get("hidden_width")
    puzzle_dir = output_root / "models" / model.model_id
    stage_parallel = _stage_parallel(
        model,
        capabilities=descriptor.puzzletron_capabilities(source_config),
    )
    data_path = _VLM_DATASET if model.is_multimodal else _TEXT_DATASET
    # Packed rows can be split by their cu_seqlens metadata. Fixed/padded rows
    # must instead form a global batch divisible across the data-parallel ranks.
    model_stage_microbatch_size = (
        1
        if campaign.data_layout == "packed_varlen"
        else max(1, model.topology.fsdp)
    )
    base.update(
        {
            "input_hf_model_path": model.hf_id,
            "puzzle_dir": str(puzzle_dir),
            "model": {
                "source": model.hf_id,
                "revision": record.immutable_revision,
                "trust_remote_code": True,
                "force_hf": False,
                "selected_model_class": record.selected_model_class,
                "native_automodel": record.native_automodel,
            },
            "capability_validation": {
                "require_complete_pipeline": True,
            },
            "execution": {
                key: list(value)
                for key, value in descriptor.stage_execution_policy().items()
            },
            "data": {
                "modality": "multimodal" if model.is_multimodal else "text",
                "layout": campaign.data_layout,
                # A tiny campaign may reduce the pack below the production
                # per-sample cap.  Keep the canonical packing invariant at the
                # generator boundary instead of emitting an invalid config.
                "max_sample_length": min(1536, campaign.sequence_length),
                "path": str(data_path),
                "revision": "pinned-intersyn-2026-07-06" if model.is_multimodal else "local-puzzle-kd",
                "processor_identity": model.hf_id,
                "packing": {
                    "pack_size": campaign.sequence_length,
                    "packing_ratio": 0.9,
                    "drop_long_samples": True,
                },
                "calibration": {
                    "path": str(data_path),
                    "num_samples": campaign.activation_samples,
                    "micro_batch_size": model_stage_microbatch_size,
                    "seq_len": campaign.sequence_length,
                },
                "scoring": {"num_samples": 8, "micro_batch_size": 1},
            },
            "embedding_pruning": {
                "enabled": hidden_axis is not None,
                "widths": (
                    [hidden_axis["teacher_value"], hidden_axis["values"][0]]
                    if hidden_axis is not None
                    else []
                ),
                "alignment": _embedding_alignment(hidden_axis),
                "cycle_widths": False,
                "sampling_policy": "inverse_width",
            },
            "search_space": {
                "no_op": {
                    "subblocks": list(model.elastic_no_op_subblocks),
                    "whole_block": False,
                    "cartesian": bool(model.elastic_no_op_subblocks),
                },
                "axes": search_axes,
            },
            "sort": {"deferred_axes": deferred_sort_axes},
            "sort_sanity": {
                "eval_samples": min(4, campaign.activation_samples),
                "micro_batch_size": model_stage_microbatch_size,
                "block_size": campaign.sequence_length,
                "include_reverse": True,
                "reverse_checkpoint_dir": str(
                    puzzle_dir / "ckpts" / "reverse_sorted_teacher"
                ),
                "reverse_activation_logs_dir": str(
                    puzzle_dir
                    / "pruning"
                    / "pruning_scores"
                    / "automodel"
                    / "reverse_all_axes"
                ),
                "max_abs_lm_loss_delta": equivalence_tolerances[
                    "max_abs_lm_loss_delta"
                ],
            },
            "width_sanity": {
                "enabled": True,
                "single_load_parent_sweep": True,
                "methods": ["activation", "random", "reverse"],
                "ratios": [0.5],
                "target_values": {
                    axis_id: axis["values"][0]
                    for axis_id, axis in search_axes.items()
                    if axis.get("values")
                },
                "layer_count": 3,
                "layer_selection": "random",
                "layer_seed": 1234,
                "eval_samples": min(8, campaign.activation_samples),
                "micro_batch_size": model_stage_microbatch_size,
                "block_size": campaign.sequence_length,
                "deferred_axes": deferred_sort_axes,
                "overwrite": True,
                "reverse_checkpoint_dir": str(
                    puzzle_dir / "ckpts" / "reverse_sorted_teacher"
                ),
                "reverse_activation_logs_dir": str(
                    puzzle_dir
                    / "pruning"
                    / "pruning_scores"
                    / "automodel"
                    / "reverse_all_axes"
                ),
                "cleanup_reverse_on_success": False,
                # The unsorted checkpoint prefix is a deterministic control,
                # not a sampled random ordering.  It is reported separately;
                # reverse ranking remains the campaign's hard ordering gate.
                "require_beats_random": False,
                "parent_equivalence_tolerances": equivalence_tolerances,
            },
        }
    )
    base["pruning"].update(
        {
            "experiment_id": "cross_model_stage_matrix",
            "eval_samples": campaign.activation_samples,
            "micro_batch_size": model_stage_microbatch_size,
            "block_size": campaign.sequence_length,
            "activation_axes": activation_axes,
            "activation_passes": activation_passes,
            "activations_log_dir": str(puzzle_dir / "pruning/pruning_scores/automodel/all_axes"),
        }
    )
    base["pruning"].setdefault("automodel", {})["parallel"] = dict(stage_parallel)
    for section_name in (
        "sort_sanity",
        "width_sanity",
        "realize_model",
    ):
        base[section_name].setdefault("automodel", {})["parallel"] = dict(stage_parallel)
    if campaign.data_layout != "packed_varlen":
        base["data"].pop("packing", None)
    global_microbatch_size = max(1, model.topology.fsdp)
    base["bypass"].update(
        {
            "enabled": True,
            "backend": "automodel",
            "elastic": True,
            "use_nested_bypassed_checkpoint_for_scoring": True,
            "overfit": {
                "enabled": True,
                "repetitions": 32,
                "modes": ["diverse_resampled", "smallest_fixed"],
                "diverse_trend_window": 8,
                "fixed_trend_window": 4,
            },
            "elastic_include_no_op": bool(model.elastic_no_op_subblocks),
            "elastic_cycle_all_targets": False,
            "elastic_coverage_mode": "coverage_then_uniform",
        }
    )
    base["bypass"].setdefault("automodel", {})["parallel"] = dict(stage_parallel)
    base["bypass"]["data"].update(
        {
            "block_size": campaign.sequence_length,
            "max_eval_samples": campaign.activation_samples,
        }
    )
    base["bypass"]["training"].update(
        {
            "optimizer": "sgd" if model.model_kind == "moe" else "adamw",
            "training_tokens": (
                campaign.kd_steps * global_microbatch_size * campaign.sequence_length
            ),
            "micro_batch_size": global_microbatch_size,
            "val_micro_batch_size": global_microbatch_size,
            "grad_accumulation_steps": 1,
            "learning_rate": 1.0e-5,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "log_interval": 1,
            "eval_interval": 1_000_000,
        }
    )
    base["bypass"]["model_factory"]["keys_to_learn"] = "entire_block"
    base["build_library"]["include_noops"] = bool(
        model.elastic_no_op_subblocks
    )
    base["vllm_stats"]["runtime_stats"].update(
        {
            "execution": "sharded",
            "num_iters": 2,
            "num_warmup_iters": 1,
            "granularity": "subblock",
        }
    )
    base["replacement_scoring"].update(
        {
            "eval_samples": campaign.activation_samples,
            "block_size": campaign.sequence_length,
        }
    )
    base["replacement_scoring"].setdefault("automodel", {})["parallel"] = dict(
        stage_parallel
    )
    base["depth_importance"] = {
        "max_subblocks_to_remove": 3,
        "ranking_metric": "lm_loss",
        "eval_samples": campaign.activation_samples,
        "micro_batch_size": 1,
        "block_size": campaign.sequence_length,
        "automodel": {"parallel": dict(stage_parallel)},
    }
    base["replacement_scoring"].update(
        {
            "reference": "scoring_parent",
            "samples": campaign.activation_samples,
            "max_candidates_per_width": 50,
        }
    )
    base["zero_shot_evaluation"].update(
        {
            "enabled": True,
            "eval_samples": campaign.activation_samples,
            "block_size": campaign.sequence_length,
        }
    )
    base["aiperf"].update(
        {
            "enabled": True,
            "checkpoint_source": "global_kd",
            "input_tokens": 256,
            "output_tokens": 32,
            "endpoint_type": "completions",
            "concurrency": [1, 2],
            "use_server_token_count": True,
            "topology": {
                "gpu_group_size": 1,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "extra_vllm_args": ["-cc.cudagraph_mode=NONE"],
            },
            "expected_solution_count": 8,
        }
    )
    base["global_distillation"].update(
        {
            "backend": "automodel",
            "force_hf": False,
            "teacher_force_hf": False,
            "student_force_hf": False,
            "domain": "vlm" if model.is_multimodal else "llm",
            "max_steps": campaign.kd_steps,
            "validation_enabled": False,
            "mtp_enabled": bool(record.mtp_fields),
            "scenario_grid": True,
            "freeze_policy": "train_all",
            "local_batch_size": int(model.topology.pp),
            "global_batch_size": int(model.topology.pp * model.topology.fsdp * 2),
            "objective": {
                "main_ce": {"weight": 1.0},
                "main_kd": {"weight": 1.0},
                "mtp_ce": {"weight": 1.0 if record.mtp_fields else 0.0},
                "mtp_kd": {
                    "weight": 1.0 if record.mtp_fields else 0.0,
                    # MTP projection/loss is vocabulary dominated.  Use one
                    # efficient GEMM for the complete CP-local sequence while
                    # keeping vocabulary logits TP sharded end to end.
                    "chunk_size": int(
                        campaign.sequence_length // max(1, model.topology.cp)
                    ),
                },
            },
        }
    )
    base["global_distillation"].setdefault("automodel", {})["parallel"] = dict(
        stage_parallel
    )
    base["global_distillation"].pop("teacher_descriptor", None)
    base["global_distillation"].pop("student_descriptor", None)
    base["global_distillation"].pop("descriptor", None)
    return base


def generate_campaign_configs(
    campaign: CrossModelCampaign,
    preflight: CampaignPreflight,
    *,
    output_root: str | Path,
    config_loader: ConfigLoader,
) -> tuple[Path, ...]:
    """Generate configs only after a complete matching preflight succeeds."""

    campaign.validate()
    if not preflight.success:
        raise ValueError("cannot generate campaign configs from a failed preflight")
    if preflight.campaign_fingerprint != campaign.fingerprint:
        raise ValueError("preflight fingerprint does not match the campaign")
    records = {record.model_id: record for record in preflight.models}
    if set(records) != {model.model_id for model in campaign.models}:
        raise ValueError("preflight model inventory does not match the campaign")

    root = Path(output_root)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for model in campaign.models:
        record = records[model.model_id]
        source_config = config_loader(model, record)
        config_path = config_dir / f"{model.model_id}.yaml"
        config = _model_config(
            campaign,
            model,
            record,
            source_config,
            output_root=root,
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        outputs.append(config_path)
    return tuple(outputs)
