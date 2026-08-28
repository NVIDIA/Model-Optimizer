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

"""AutoModel-native block-local distillation with stage-local PP losses.

The teacher alone runs a forward-only AutoModel pipeline schedule. Hooks retain
detached inputs and targets for decoder layers owned by the local PP stage. The
matching student layers are then replayed locally under grad. Student losses and
gradients never cross PP stages.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from ...anymodel.model_descriptor import ModelDescriptorFactory
from ...block_config import maybe_cast_block_configs
from ...bypass_distillation.bypass_utils import normalize_keys_to_learn
from ...bypass_distillation.checkpointing import (
    quarantine_incomplete_checkpoint,
    require_distributed_path_consensus,
    validate_automodel_bypass_checkpoint,
)
from ...bypass_distillation.elastic_supernet import (
    CanonicalCandidateMasker,
    build_canonical_block_elastics,
    logical_data_lane_from_peer_sets,
    validate_lane_architecture_assignments,
)
from ...bypass_distillation.losses import resolve_local_kd_loss
from ...bypass_distillation.observations import (
    CandidateCatalog,
    ObservationWriter,
    merge_rank_observations,
)
from ...bypass_distillation.parameter_selection import set_keys_to_learn
from ...bypass_distillation.schedule import get_learning_rate
from ...bypass_distillation.subblock_boundaries import (
    install_teacher_subblock_capture_hooks,
    replay_subblock,
    resolve_subblock_boundaries,
    selected_subblock_kinds,
)
from ...dataset import DataLayout
from ...granularity import resolve_granularity
from ...pruning.elastic_sampling import inverse_width_probs
from ...pruning.runtime_hidden_width import hidden_width_layer_context, retained_hidden_prefix
from ...pruning.runtime_ple import ple_layer_context
from ...tools.bypassed_training.child_init import update_model_config
from ...tools.checkpoint_utils_hf import load_model_config
from ...tools.logger import aprint, mprint
from ...utils.data.dataloaders import (
    create_train_dataloader,
    create_validation_dataloader,
    load_from_disk_fn,
    load_streaming_fn,
)
from .config import _int_or_default
from .scoring_recipe import ActivationScoringRecipe


def _consolidated_export_enabled(value) -> bool:
    from nemo_automodel.components.checkpoint.config import SaveConsolidatedMode

    return value != SaveConsolidatedMode.FALSE


from .solution_recipe import ReplaceBlockScoringRecipe

logger = logging.getLogger(__name__)

__all__ = ["AutoModelLocalDistillationRecipe"]


_HF_AUXILIARY_FILENAMES = {
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "vocab.txt",
}
_HF_AUXILIARY_FILE_SIZE_LIMIT = 64 * 1024 * 1024
_HF_AUXILIARY_TOTAL_SIZE_LIMIT = 256 * 1024 * 1024


def _mask_local_kd_tensors(
    student: torch.Tensor,
    teacher: torch.Tensor,
    hidden_mask: torch.Tensor | None,
    *,
    record_index: int,
    record_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select real tokens from one block/subblock replay tensor."""

    if hidden_mask is None or bool(hidden_mask.all()):
        return student, teacher
    mask = hidden_mask.bool()
    if student.ndim < 2:
        raise RuntimeError("padded/packed local KD requires replay tensors with token dimensions")
    if tuple(student.shape[:2]) == tuple(mask.shape):
        return student[mask], teacher[mask]
    if tuple(student.shape[:2]) == tuple(reversed(mask.shape)):
        transposed = mask.transpose(0, 1)
        return student[transposed], teacher[transposed]

    if record_count > 0 and mask.shape[0] % record_count == 0:
        rows = mask.shape[0] // record_count
        local_mask = mask[record_index * rows : (record_index + 1) * rows]
        if tuple(student.shape[:2]) == tuple(local_mask.shape):
            return student[local_mask], teacher[local_mask]
        if tuple(student.shape[:2]) == tuple(reversed(local_mask.shape)):
            transposed = local_mask.transpose(0, 1)
            return student[transposed], teacher[transposed]
    if student.ndim == 2 and student.shape[0] == mask.numel():
        flat = mask.reshape(-1)
        return student[flat], teacher[flat]
    raise RuntimeError(
        "cannot align canonical hidden_mask with local KD replay tensor: "
        f"mask={tuple(mask.shape)} tensor={tuple(student.shape)} "
        f"record={record_index + 1}/{record_count}"
    )


def _local_kd_loss_or_zero(loss_fn, student, teacher) -> tuple[torch.Tensor, bool]:
    """Keep empty DP/CP shards in autograd without treating them as observations."""

    if student.numel() == 0:
        # Every rank must participate in FSDP gradient synchronization.  A
        # differentiable zero preserves that collective schedule, while
        # excluding the empty shard from metrics avoids NaN (and zero bias).
        return student.sum() * 0.0, False
    return loss_fn(student, teacher), True


def _cached_subblock_cost(
    cache: dict[tuple[int, str], int],
    *,
    width: int,
    subblock,
    calculate,
) -> int:
    """Calculate each semantic subblock cost once per hidden width."""

    key = (
        int(width),
        json.dumps(subblock.to_dict(), sort_keys=True, separators=(",", ":")),
    )
    if key not in cache:
        cache[key] = int(calculate(subblock))
    return cache[key]


def _load_optimizer_with_lazy_state(
    checkpointer,
    optimizer,
    model,
    weights_path: str,
    scheduler=None,
) -> None:
    """Restore Adam's lazily-created state while rejecting corrupt saved entries."""

    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    optimizer_state = OptimizerState(
        model,
        optimizer,
        scheduler,
        is_peft=checkpointer.config.is_peft,
    )
    state_dict = optimizer_state.state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=os.path.join(weights_path, "optim"),
        planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
    )
    optimizer_state.load_state_dict(state_dict)


def _nested_hidden_widths(
    teacher_width: int,
    configured_widths: tuple[int, ...],
) -> tuple[int, ...]:
    """Include the full-width identity candidate exactly once."""

    return tuple(dict.fromkeys((int(teacher_width), *map(int, configured_widths))))


def _merge_lane_axis_counts(
    gathered: list[dict[str, object]],
    *,
    count_key: str,
) -> dict[int, int]:
    """Merge one architecture-axis count map per logical data lane.

    Model-parallel ranks within a lane must sample the same architecture, so
    their identical count maps are replicas rather than independent samples.
    Distinct logical data lanes are independent and their counts are summed.
    """

    counts_by_lane: dict[int, dict[int, int]] = {}
    for payload in gathered:
        lane = int(payload["dp_lane"])
        counts = {
            int(value): int(count)
            for value, count in dict(payload.get(count_key) or {}).items()
            if int(count) > 0
        }
        previous = counts_by_lane.setdefault(lane, counts)
        if previous != counts:
            raise RuntimeError(
                f"inconsistent {count_key} within logical data lane {lane}: "
                f"expected={previous} observed={counts}"
            )

    merged: dict[int, int] = defaultdict(int)
    for counts in counts_by_lane.values():
        for value, count in counts.items():
            merged[value] += count
    return dict(sorted(merged.items()))


def _select_hidden_width(
    widths: tuple[int, ...],
    *,
    step: int,
    cycle: bool,
    policy: str | None,
    generator: torch.Generator,
) -> int:
    """Select one global hidden width before per-layer elastic candidates."""

    if not widths:
        raise ValueError("hidden-width selection requires at least one width")
    if policy == "inverse_width":
        index = int(
            torch.multinomial(
                inverse_width_probs(widths),
                1,
                generator=generator,
            ).item()
        )
        return int(widths[index])
    if policy not in (None, "uniform", "cycle"):
        raise ValueError(f"unsupported hidden-width sampling policy: {policy!r}")
    if policy == "cycle" or (policy is None and cycle):
        return int(widths[(int(step) - 1) % len(widths)])
    index = int(torch.randint(len(widths), (), generator=generator).item())
    return int(widths[index])


def _elastic_selection_record(
    *,
    step: int,
    micro_step: int | None = None,
    hidden_width: int,
    targets: dict[int, object],
    ple_width: int | None = None,
    parameter_counts: dict[int, dict[str, int]] | None = None,
) -> dict[str, object]:
    """Stable per-step candidate provenance written with local-KD metrics."""

    record = {
        "step": int(step),
        "hidden_width": int(hidden_width),
        "ple_width": int(ple_width) if ple_width is not None else None,
        "layers": [
            {
                "layer_idx": int(layer_idx),
                "candidate_id": str(candidate.identity.value),
                "parameter_count": (
                    (parameter_counts or {})
                    .get(int(layer_idx), {})
                    .get(str(candidate.identity.value))
                ),
                "changed_axes": dict(candidate.metadata.get("slice_axes") or {}),
            }
            for layer_idx, candidate in sorted(targets.items())
        ],
    }
    if micro_step is not None:
        record["micro_step"] = int(micro_step)
    return record


def _recipe_parallel_size(recipe_cfg, key: str) -> int:
    """Read a topology axis from the canonical AutoModel recipe config."""
    distributed = recipe_cfg.get("distributed", {}) or {}
    return _int_or_default(distributed.get(key), 1)


def _backward_disjoint_loss(
    loss: torch.Tensor,
    *,
    grad_scaler,
    grad_accum: int,
) -> torch.Tensor:
    """Backpropagate one disjoint local loss and retain only its scalar value."""

    grad_scaler.scale(loss / grad_accum).backward()
    return loss.detach()


def _iter_metric_loggers(loggers) -> tuple:
    if loggers is None:
        return ()
    values = getattr(loggers, "values", None)
    return tuple(values()) if callable(values) else (loggers,)


def _overfit_loss_records_are_comparable(
    *,
    single_batch_overfit: bool,
    resample_structure: bool,
) -> bool:
    """Return whether an overfit probe repeats one architecture across steps."""

    return bool(single_batch_overfit and not resample_structure)


def _loss_trend_summary(
    records: list[dict[str, Any]],
    *,
    comparable: bool,
    window_size: int = 4,
    minimum_relative_decrease: float = 0.0,
) -> dict[str, Any]:
    finite = [
        record for record in records if math.isfinite(float(record.get("loss", float("nan"))))
    ]
    requested_window = max(1, int(window_size))
    if len(finite) < 2 * requested_window:
        return {
            "sufficient_evidence": False,
            "required_records": 2 * requested_window,
            "observed_records": len(finite),
            "comparable": bool(comparable),
            "hard_gate_passed": False if comparable else None,
        }

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        window = min(requested_window, len(rows) // 2)
        if window == 0:
            return {}
        first = median(float(row["loss"]) for row in rows[:window])
        last = median(float(row["loss"]) for row in rows[-window:])
        return {
            "count": len(rows),
            "window": window,
            "first_median": first,
            "last_median": last,
            "decreased": last < first,
        }

    aggregate = summarize(finite)
    per_hidden_width: dict[str, Any] = {}
    for width in sorted({record.get("hidden_width") for record in finite}, key=str):
        rows = [record for record in finite if record.get("hidden_width") == width]
        summary = summarize(rows)
        if summary:
            per_hidden_width[str(width)] = summary
    relative_decrease = (
        (aggregate["first_median"] - aggregate["last_median"]) / abs(aggregate["first_median"])
        if aggregate["first_median"] != 0
        else None
    )
    material_decrease = (
        aggregate["decreased"]
        and relative_decrease is not None
        and relative_decrease >= float(minimum_relative_decrease)
    )
    return {
        "sufficient_evidence": True,
        "window": aggregate["window"],
        "first_window_median": aggregate["first_median"],
        "last_window_median": aggregate["last_median"],
        "last_to_first_ratio": (
            aggregate["last_median"] / aggregate["first_median"]
            if aggregate["first_median"] != 0
            else None
        ),
        "first_four_median": aggregate["first_median"],
        "last_four_median": aggregate["last_median"],
        "decreased": aggregate["decreased"],
        "relative_decrease": relative_decrease,
        "minimum_relative_decrease": float(minimum_relative_decrease),
        "comparable": bool(comparable),
        "hard_gate_passed": material_decrease if comparable else None,
        "per_hidden_width": per_hidden_width,
        "incomparability_reason": (
            None
            if comparable
            else "elastic candidate and/or global-width schedule changes between steps"
        ),
    }


def _copy_hf_auxiliary_assets(source_dir: Path, consolidated_dir: Path) -> None:
    """Fill a consolidated checkpoint with non-weight HF processor assets.

    NeMo AutoModel's consolidated saver persists the tokenizer but does not
    currently preserve every multimodal processor file.  Copying only absent,
    non-weight files keeps the consolidated model/config authoritative while
    retaining bounded tokenizer and image/video processor assets needed by
    ``AutoProcessor.from_pretrained``.
    """
    if not source_dir.is_dir() or not consolidated_dir.is_dir():
        return
    if source_dir.is_symlink() or consolidated_dir.is_symlink():
        raise ValueError("checkpoint asset directories must not be symbolic links")

    candidates = [source_dir / name for name in sorted(_HF_AUXILIARY_FILENAMES)]
    chat_templates = source_dir / "chat_templates"
    if chat_templates.is_symlink():
        raise ValueError("checkpoint chat_templates directory must not be a symbolic link")
    if chat_templates.is_dir():
        candidates.extend(sorted(chat_templates.glob("*.jinja")))

    validated = []
    total_size = 0
    for source in candidates:
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"checkpoint asset must be a regular file: {source.name}")
        relative = source.relative_to(source_dir)
        destination = consolidated_dir / relative
        if destination.is_symlink():
            raise ValueError(f"checkpoint destination must not be a symbolic link: {relative}")
        if destination.parent.is_symlink():
            raise ValueError(
                f"checkpoint destination directory must not be a symbolic link: {destination.parent.name}"
            )
        if destination.exists():
            continue
        size = source.stat().st_size
        if size > _HF_AUXILIARY_FILE_SIZE_LIMIT:
            raise ValueError(f"checkpoint asset exceeds the per-file size limit: {relative}")
        total_size += size
        if total_size > _HF_AUXILIARY_TOTAL_SIZE_LIMIT:
            raise ValueError("checkpoint assets exceed the aggregate size limit")
        validated.append((source, destination))

    for source, destination in validated:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _output_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _loss_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


class _DistributedValidationCoordinator:
    """Minimal accelerator interface used by the shared validation loader."""

    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process


def _decoder_layers(parts, descriptor, num_hidden_layers: int) -> dict[int, torch.nn.Module]:
    """Find owned main-decoder layers without matching auxiliary layer trees."""
    found: dict[int, torch.nn.Module] = {}
    for layer_idx in range(int(num_hidden_layers)):
        canonical_name = descriptor.layer_block_name(layer_idx)
        for part in parts:
            adapted_name = descriptor.adapt_module_name_for_model(canonical_name, part)
            for module_name in dict.fromkeys((adapted_name, canonical_name)):
                try:
                    layer = part.get_submodule(module_name)
                except AttributeError:
                    continue
                existing = found.get(layer_idx)
                if existing is not None and existing is not layer:
                    raise RuntimeError(f"duplicate decoder layer {layer_idx} on one PP rank")
                found[layer_idx] = layer
                break
            if layer_idx in found:
                break
    return found


def _block_context_kwargs(teacher_block, child_block, *, head_dim: int) -> dict[str, Any]:
    teacher_ffn = teacher_block.get_subblock("ffn")
    child_ffn = child_block.get_subblock("ffn")
    teacher_attn = teacher_block.get_subblock("attention")
    child_attn = child_block.get_subblock("attention")
    return {
        "teacher_block_config": teacher_block,
        "child_block_config": child_block,
        "orig_intermediate": getattr(teacher_ffn, "intermediate_size", None),
        "target_intermediate": (
            0
            if getattr(child_ffn, "no_op", False)
            else getattr(child_ffn, "intermediate_size", None)
        ),
        "orig_num_q": getattr(teacher_attn, "num_query_heads", None),
        "orig_num_kv": getattr(teacher_attn, "num_kv_heads", None),
        "target_num_q": (
            0
            if getattr(child_attn, "no_op", False)
            else getattr(child_attn, "num_query_heads", None)
        ),
        "target_num_kv": (
            0 if getattr(child_attn, "no_op", False) else getattr(child_attn, "num_kv_heads", None)
        ),
        "head_dim": head_dim,
    }


class AutoModelLocalDistillationRecipe(ReplaceBlockScoringRecipe):
    """Train a full-width prefix-sliceable student from local teacher targets."""

    def __init__(self, cfg, *, hydra_cfg, resume_path: str | None = None):
        super().__init__(
            cfg,
            pruning_cfg=hydra_cfg.bypass,
            eval_iters=None,
            use_puzzletron_dataloader=False,
            data_cfg=OmegaConf.to_container(hydra_cfg.get("data", {}), resolve=True),
        )
        self._hydra_cfg = hydra_cfg
        self._resume_path = resume_path
        self._capture_enabled = False
        self._teacher_records: dict[Any, list[Any]] = defaultdict(list)
        self._capture_handles: list[Any] = []
        self._student_target_outputs: dict[int, Any] = {}
        self._student_capture_handles: list[Any] = []
        self._fixed_child_blocks = None
        self._elastic_masker = None
        self._last_elastic_targets: dict[int, object] = {}
        self._elastic_selection_history: list[dict[str, object]] = []
        self._elastic_parameter_counts_by_width: dict[int, dict[int, dict[str, int]]] = {}
        self._elastic_candidate_metadata_by_width: dict[
            int, dict[int, dict[str, dict[str, Any]]]
        ] = {}
        self._layouts_by_idx = {}
        self._last_child_blocks = None
        self._embedding_spec = None
        self._hidden_widths: tuple[int, ...] = ()
        self._hidden_width_counts: dict[int, int] = defaultdict(int)
        self._teacher_hidden_width: int | None = None
        self._ple_spec = None
        self._ple_widths: tuple[int, ...] = ()
        self._ple_width_counts: dict[int, int] = defaultdict(int)
        self._local_kd_granularity = resolve_granularity("bypass", hydra_cfg.bypass)
        self._teacher_subblock_boundaries = {}
        self._student_subblock_boundaries = {}
        self._current_subblock_metrics: dict[str, float] = {}
        self._logical_dp_lane = 0
        self._logical_dp_size = 1
        self._lane_assignment_validated = False

    def setup(self):
        """Build the teacher through the proven forward-only PP path, then the student."""
        # ReplaceBlockScoringRecipe.setup additionally gathers the full LM-head
        # weight and installs final-output capture for candidate scoring. Local
        # KD needs only its runtime pruning context, so use the forward-only
        # grandparent setup and avoid that last-stage memory spike.
        ActivationScoringRecipe.setup(self)
        self._initialize_logical_data_lane()
        object.__setattr__(self, "_teacher_parts", list(self.model_parts))
        object.__setattr__(self, "_teacher_pp", self.pp)
        if self._teacher_pp is not None:
            from torch.distributed.pipelining.schedules import _ScheduleForwardOnly

            schedule = self._teacher_pp.info.schedule
            if not isinstance(schedule, _ScheduleForwardOnly):
                raise RuntimeError(
                    "AutoModel local KD requires a forward-only-compatible PP schedule. "
                    "Use distributed.pipeline.pp_schedule=1f1b without virtual/interleaved "
                    f"stages; got {type(schedule).__name__}."
                )
        self.checkpointer.config.enabled = True
        self.checkpointer.config.checkpoint_dir = str(self.cfg.local_kd_checkpoint_dir)

        from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline
        from nemo_automodel.recipes._dist_utils import shard_optimizers_for_megatron_fsdp

        if self._use_vlm_recipe:
            from nemo_automodel.recipes.vlm.finetune import build_model
        else:
            from nemo_automodel.recipes.llm.train_ft import build_model

        student_cfg = self.cfg.get("student_model", None)
        if student_cfg is None:
            raise ValueError("AutoModel local KD recipe is missing student_model")
        if self._use_vlm_recipe:
            student = build_model(
                student_cfg,
                cfg_freeze=None,
                cfg_peft=None,
                seed=self.cfg.get("seed", 42),
                cfg_fp8=None,
                cfg_compile=None,
                distributed_setup=self.distributed_setup,
                cfg_quantization=None,
            )
        else:
            student = build_model(
                student_cfg,
                cfg_peft=None,
                has_packed_sequence=self.cfg.get("packed_sequence.packed_sequence_size", 0) > 0,
                seed=self.cfg.get("seed", 42),
                cfg_fp8=None,
                cfg_compile=None,
                cfg_quantization=None,
                distributed_setup=self.distributed_setup,
                cfg_qat=None,
                sdpa_method=self.cfg.get("sdpa_method", None),
            )
        student_parts = list(student.parts) if isinstance(student, AutoPipeline) else [student]
        student_pp = student if isinstance(student, AutoPipeline) else None

        descriptor = ModelDescriptorFactory.get(str(self._hydra_cfg.descriptor))
        if descriptor is None:
            raise ValueError(f"unknown Puzzletron descriptor {self._hydra_cfg.descriptor!r}")
        object.__setattr__(self, "_descriptor", descriptor)
        for part in student_parts:
            if hasattr(descriptor, "patch_pipeline_model_part"):
                descriptor.patch_pipeline_model_part(part)

        # Construct the optimizer while all parameters are visible, then freeze to
        # the existing keys_to_learn contract. Ranks with embedding/head-only PP
        # parts retain a valid no-gradient optimizer instead of failing on an empty list.
        optimizer = self.cfg.optimizer.build(
            student,
            device_mesh=self.device_mesh,
            is_peft=False,
        )
        optimizer = shard_optimizers_for_megatron_fsdp(
            student,
            optimizer,
            self.distributed_config,
            allow=getattr(self.cfg.optimizer, "supports_megatron_fsdp_sharding", True),
        )
        for part in student_parts:
            part.train()
            part.requires_grad_(False)
            set_keys_to_learn(
                part,
                descriptor,
                self._hydra_cfg.bypass.model_factory.keys_to_learn,
            )
        # Make the student the only checkpoint-tracked model. Teacher references use
        # object.__setattr__ and therefore cannot be mistaken for trainable state.
        self.untrack_state("model_parts", "optimizer", "lr_scheduler")
        self.model_parts = student_parts
        self.optimizer = optimizer
        self.lr_scheduler = None
        object.__setattr__(self, "_student_pp", student_pp)
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(self._hydra_cfg.bypass.training.use_grad_scaling),
        )

        lm_config = descriptor.get_language_model_config(student_parts[0].config)
        num_hidden_layers = int(lm_config.num_hidden_layers)
        self._expected_checkpoint_layer_prefixes = tuple(
            descriptor.layer_block_name(layer_idx) for layer_idx in range(num_hidden_layers)
        )
        self._teacher_layers = _decoder_layers(self._teacher_parts, descriptor, num_hidden_layers)
        self._student_layers = _decoder_layers(self.model_parts, descriptor, num_hidden_layers)
        if set(self._teacher_layers) != set(self._student_layers):
            raise RuntimeError(
                "teacher/student PP layer ownership differs on this rank: "
                f"teacher={sorted(self._teacher_layers)} student={sorted(self._student_layers)}"
            )

        self._prepare_block_configs_and_masks()
        self._ensure_trainable_targets()
        local_trainable = sum(
            parameter.numel()
            for part in self.model_parts
            for parameter in part.parameters()
            if parameter.requires_grad
        )
        trainable_tensor = torch.tensor(local_trainable, device=self.dist_env.device)
        torch.distributed.all_reduce(trainable_tensor)
        if int(trainable_tensor.item()) == 0:
            raise ValueError(
                "AutoModel local KD keys_to_learn matched no student parameters: "
                f"{self._hydra_cfg.bypass.model_factory.keys_to_learn!r}"
            )
        self._install_local_capture_hooks()
        native_dataloader = self.dataloader
        self.untrack_state("dataloader")
        self.dataloader = (
            native_dataloader
            if (
                self._use_vlm_recipe
                or (self._data_spec is not None and self._data_spec.layout is not DataLayout.FIXED)
            )
            else self._build_puzzletron_train_dataloader()
        )
        object.__setattr__(
            self,
            "_local_val_dataloader",
            None if self._use_vlm_recipe else self._build_validation_dataloader(),
        )
        if self._resume_path:
            self._validate_resumable_checkpoint(
                Path(self._resume_path),
                require_completed=True,
            )
            strict_loader = self.checkpointer.load_optimizer
            self.checkpointer.load_optimizer = lambda optimizer, model, path, scheduler=None: (
                _load_optimizer_with_lazy_state(
                    self.checkpointer,
                    optimizer,
                    model,
                    path,
                    scheduler,
                )
            )
            try:
                self.load_checkpoint(self._resume_path)
            finally:
                self.checkpointer.load_optimizer = strict_loader
        _, self._autocast_dtype = self._resolve_forward_autocast()
        self._use_autocast = bool(self._descriptor.uses_autocast())
        aprint(
            "[bypass/automodel] local stage owns layers="
            f"{sorted(self._student_layers)} trainable_params={local_trainable:,}"
        )

    def _initialize_logical_data_lane(self) -> None:
        """Derive data lanes as connected components of model-parallel peers."""

        local_peer_sets: list[tuple[int, ...]] = []
        for mesh, axes in (
            (self.device_mesh, ("pp", "cp", "tp")),
            (self.moe_mesh, ("ep",)),
        ):
            if mesh is None:
                continue
            for axis in axes:
                if axis not in mesh.mesh_dim_names or mesh[axis].size() <= 1:
                    continue
                group = mesh[axis].get_group()
                local_peer_sets.append(tuple(torch.distributed.get_process_group_ranks(group)))
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, tuple(local_peer_sets))
        lane, lane_count = logical_data_lane_from_peer_sets(
            torch.distributed.get_rank(),
            gathered,
        )
        self._logical_dp_lane = lane
        self._logical_dp_size = lane_count
        aprint(
            "[bypass/automodel] architecture lane="
            f"{lane}/{lane_count} model_parallel_peers={local_peer_sets}"
        )

    def _architecture_sample_index(self, step: int, micro_step: int = 0) -> int:
        hydra_cfg = getattr(self, "_hydra_cfg", None)
        grad_accum = (
            int(hydra_cfg.bypass.training.grad_accumulation_steps) if hydra_cfg is not None else 1
        )
        iteration = (int(step) - 1) * grad_accum + int(micro_step)
        return iteration * self._logical_dp_size + self._logical_dp_lane

    def _build_puzzletron_train_dataloader(self):
        cfg = self._hydra_cfg
        data_cfg = cfg.bypass.data
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            str(self.cfg.model.pretrained_model_name_or_path),
            trust_remote_code=bool(self.cfg.model.get("trust_remote_code", True)),
            token=True,
        )
        load_dataset_fn = load_from_disk_fn if data_cfg.load_from_disk else load_streaming_fn
        return create_train_dataloader(
            seed=int(cfg.bypass.seed),
            tokenizer=tokenizer,
            block_size=int(data_cfg.block_size),
            dataset_path=cfg.dataset_path,
            content_field=data_cfg.data_column,
            fim_rate=float(data_cfg.fim_rate),
            fim_spm_rate=float(data_cfg.fim_spm_rate),
            micro_batch_size=int(cfg.bypass.training.micro_batch_size),
            load_dataset_fn=load_dataset_fn,
            keep_in_memory=bool(data_cfg.keep_in_memory),
            source_datasets_to_discard=data_cfg.get("source_datasets_to_discard", tuple()),
            bos_rate=float(data_cfg.bos_rate),
            shuffle_seed=int(data_cfg.shuffle_train_data_seed),
            packed_token_cache_path=data_cfg.get("packed_token_cache_path", None),
        )

    def _build_validation_dataloader(self):
        if bool(self._hydra_cfg.bypass.get("disable_validation", True)):
            return None
        cfg = self._hydra_cfg
        data_cfg = cfg.bypass.data
        if data_cfg.eval_samples_per_process is not None:
            max_eval_samples = int(data_cfg.eval_samples_per_process) * int(
                self.dist_env.world_size
            )
        else:
            configured_max = data_cfg.get("max_eval_samples", None)
            max_eval_samples = None if configured_max is None else int(configured_max)
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
            str(self.cfg.model.pretrained_model_name_or_path),
            trust_remote_code=bool(self.cfg.model.get("trust_remote_code", True)),
            token=True,
        )
        if self._data_spec is not None and self._data_spec.layout is not DataLayout.FIXED:
            from ...utils.data.dataloaders import prepare_validation_dataloader

            validation_args = OmegaConf.to_container(data_cfg, resolve=True)
            validation_args.update(
                {
                    "dataset_path": str(cfg.dataset_path),
                    "eval_samples": max_eval_samples,
                    "micro_batch_size": int(
                        cfg.bypass.training.val_micro_batch_size
                        or cfg.bypass.training.micro_batch_size
                    ),
                    "seed": int(cfg.bypass.seed),
                }
            )
            return prepare_validation_dataloader(
                validation_args,
                tokenizer,
                data_layout=self._data_spec.layout.value,
            )
        load_dataset_fn = load_from_disk_fn if data_cfg.load_from_disk else load_streaming_fn
        return create_validation_dataloader(
            accelerator=_DistributedValidationCoordinator(self.dist_env.is_main),
            seed=int(cfg.bypass.seed),
            tokenizer=tokenizer,
            block_size=int(data_cfg.block_size),
            dataset=cfg.dataset_path,
            content_field=data_cfg.data_column,
            fim_rate=float(data_cfg.fim_rate),
            fim_spm_rate=float(data_cfg.fim_spm_rate),
            micro_batch_size=int(
                cfg.bypass.training.val_micro_batch_size or cfg.bypass.training.micro_batch_size
            ),
            eval_samples=max_eval_samples,
            load_dataset_fn=load_dataset_fn,
            dataset_name=data_cfg.val_dataset_name,
            keep_in_memory=bool(data_cfg.keep_in_memory),
            source_datasets_to_discard=data_cfg.get("source_datasets_to_discard", tuple()),
            bos_rate=float(data_cfg.bos_rate),
            realized_cache_dir=data_cfg.get("realized_cache_dir", None),
            packed_token_cache_path=data_cfg.get(
                "validation_packed_token_cache_path",
                data_cfg.get("packed_token_cache_path", None),
            ),
        )

    def _find_decoder_layer(self, layer_idx: int):
        """Resolve global layer indices after AutoModel PP has compacted ModuleLists."""
        return self._student_layers.get(int(layer_idx))

    def _ensure_trainable_targets(self) -> None:
        """Fallback from checkpoint-key selection to semantic live-module selection.

        Native AutoModel model parts can expose different parameter prefixes than
        their canonical HF checkpoint descriptor. The descriptor selector remains
        authoritative; this fallback only handles a local decoder layer for which
        it selected no parameter at all.
        """
        requested = set(
            normalize_keys_to_learn(self._hydra_cfg.bypass.model_factory.keys_to_learn)["subblocks"]
        )
        for layer_idx, layer in self._student_layers.items():
            if any(parameter.requires_grad for parameter in layer.parameters()):
                continue
            block = self._teacher_blocks[layer_idx]
            modules = []
            if "entire_block" in requested:
                modules = [layer]
            else:
                if "subblock_ffn" in requested and (
                    block.get_subblock("ffn") is not None or block.get_subblock("moe") is not None
                ):
                    modules.extend(
                        module
                        for module in (getattr(layer, "mlp", None), getattr(layer, "mixer", None))
                        if module is not None
                    )
                if (
                    "subblock_attention" in requested
                    and block.get_subblock("attention") is not None
                ):
                    modules.extend(
                        module
                        for module in (
                            getattr(layer, "self_attn", None),
                            getattr(layer, "mixer", None),
                        )
                        if module is not None
                    )
                if "subblock_mamba" in requested and block.get_subblock("mamba") is not None:
                    modules.extend(
                        module
                        for module in (
                            getattr(layer, "linear_attn", None),
                            getattr(layer, "mixer", None),
                        )
                        if module is not None
                    )
            for module in {id(module): module for module in modules}.values():
                for parameter in module.parameters():
                    if torch.is_floating_point(parameter):
                        parameter.requires_grad_(True)

    def _prepare_block_configs_and_masks(self):
        cfg = self._hydra_cfg
        teacher_config = load_model_config(
            str(self.cfg.model.pretrained_model_name_or_path),
            trust_remote_code=bool(self.cfg.model.get("trust_remote_code", True)),
        )
        teacher_blocks = maybe_cast_block_configs(teacher_config.block_configs)
        if not teacher_blocks:
            raise ValueError("AutoModel local KD requires block_configs on the teacher checkpoint")
        self._teacher_blocks = teacher_blocks
        lm = self._descriptor.get_language_model_config(teacher_config)
        self._teacher_hidden_width = int(lm.hidden_size)
        embedding_cfg = self._hydra_cfg.get("embedding_pruning", {})
        if bool(embedding_cfg.get("enabled", False)):
            configured_widths = tuple(int(width) for width in embedding_cfg.get("widths", ()))
            if not configured_widths:
                raise ValueError("embedding_pruning.enabled requires at least one width")
            configured_widths = _nested_hidden_widths(
                self._teacher_hidden_width,
                configured_widths,
            )
            self._embedding_spec = self._descriptor.embedding_pruning_spec(
                teacher_config,
                widths=configured_widths,
                alignment=int(embedding_cfg.get("alignment", 1)),
            )
            tp_size = _recipe_parallel_size(self.cfg, "tp_size")
            self._hidden_widths = tuple(
                self._embedding_spec.validate_width(width, tp_size=tp_size)
                for width in configured_widths
            )
            if int(lm.hidden_size) not in self._hidden_widths:
                raise ValueError(
                    "nested bypass widths must include the full teacher width "
                    f"{lm.hidden_size}; got {self._hidden_widths}"
                )
        ple_axis = (
            (self._hydra_cfg.get("search_space", {}) or {}).get("axes", {}).get("ple_width", {})
        )
        if bool(ple_axis.get("enabled", False)):
            self._ple_spec = self._descriptor.ple_pruning_spec(teacher_config)
            if self._ple_spec is None:
                raise ValueError(
                    "ple_width is enabled but the descriptor/config has no PLE contract"
                )
            teacher_ple_width = int(self._ple_spec.width)
            options = [teacher_ple_width]
            options.extend(int(value) for value in ple_axis.get("values", ()))
            options.extend(
                max(1, int(round(teacher_ple_width * float(ratio))))
                for ratio in ple_axis.get("ratios", ())
            )
            self._ple_widths = tuple(dict.fromkeys(options))
            invalid = [width for width in self._ple_widths if not 0 < width <= teacher_ple_width]
            if invalid:
                raise ValueError(
                    f"invalid PLE widths {invalid}; teacher width is {teacher_ple_width}"
                )
        num_q = int(lm.num_attention_heads)
        self._head_dim = int(getattr(lm, "head_dim", None) or (lm.hidden_size // num_q))

        from ...pruning.sorted_teacher import build_layer_layouts

        layout_kwargs = {}
        if hasattr(self._descriptor, "sorted_teacher_layout_kwargs"):
            layout_kwargs.update(self._descriptor.sorted_teacher_layout_kwargs(lm))
        layer_prefix_tmpl = self._descriptor.layer_block_name(0).rsplit(".", 1)[0] + ".{i}"
        layouts = build_layer_layouts(
            teacher_blocks,
            layer_prefix_tmpl=layer_prefix_tmpl,
            num_attention_heads=num_q,
            head_dim=self._head_dim,
            **layout_kwargs,
        )
        self._layouts_by_idx = {layout.layer_idx: layout for layout in layouts}

        if not bool(cfg.bypass.get("elastic", False)):
            overrides = OmegaConf.to_container(
                cfg.bypass.model.model_config_overrides,
                resolve=True,
            )
            child_config = update_model_config(
                model_config=copy.deepcopy(teacher_config),
                model_config_overrides=overrides,
            )
            self._fixed_child_blocks = maybe_cast_block_configs(child_config.block_configs)
            if len(self._fixed_child_blocks) != len(self._teacher_blocks):
                raise ValueError(
                    "AutoModel local KD requires teacher and child to retain the same "
                    "decoder-layer count; got "
                    f"{len(self._teacher_blocks)} and {len(self._fixed_child_blocks)}"
                )
            return

        elastics = build_canonical_block_elastics(
            teacher_blocks,
            search_space=cfg.get("search_space", {}),
            model_config=teacher_config,
            descriptor=self._descriptor,
            include_no_op=bool(cfg.bypass.get("elastic_include_no_op", True)),
        )
        if not elastics:
            raise ValueError("elastic AutoModel local KD found no configured elastic subblocks")
        self._elastic_masker = CanonicalCandidateMasker(
            elastics,
            layouts_by_idx=self._layouts_by_idx,
            head_dim=self._head_dim,
            seed=int(cfg.bypass.get("elastic_seed", 42)),
        )
        from ...subblock_stats.calc_subblock_params_and_memory import calc_subblock_active_params

        candidate_cost_cache: dict[tuple[int, str], int] = {}
        subblock_cost_cache: dict[tuple[int, str], int] = {}
        for width in self._hidden_widths or (self._teacher_hidden_width,):
            width_config = copy.deepcopy(teacher_config)
            width_lm = self._descriptor.get_language_model_config(width_config)
            width_lm.hidden_size = int(width)
            if width_lm is not width_config and hasattr(width_config, "hidden_size"):
                width_config.hidden_size = int(width)
            by_layer: dict[int, dict[str, int]] = {}
            metadata_by_layer: dict[int, dict[str, dict[str, Any]]] = {}

            def subblock_cost(subblock):
                return _cached_subblock_cost(
                    subblock_cost_cache,
                    width=int(width),
                    subblock=subblock,
                    calculate=lambda value: calc_subblock_active_params(
                        value,
                        width_config,
                        self._descriptor,
                        int(width),
                    ),
                )

            for elastic in elastics:
                layer_costs: dict[str, int] = {}
                layer_metadata: dict[str, dict[str, Any]] = {}
                teacher_subblocks = {
                    (subblock.kind, subblock.name): subblock
                    for subblock in elastic.parent_block_config.subblock_configs
                }
                teacher_counts = {
                    key: subblock_cost(subblock) for key, subblock in teacher_subblocks.items()
                }
                for candidate in elastic.sampler.sizes:
                    candidate_id = str(candidate.identity.value)
                    cache_key = (
                        int(width),
                        json.dumps(candidate.block_config.to_dict(), sort_keys=True),
                    )
                    if cache_key not in candidate_cost_cache:
                        candidate_cost_cache[cache_key] = sum(
                            subblock_cost(subblock)
                            for subblock in candidate.block_config.subblock_configs
                        )
                    layer_costs[candidate_id] = candidate_cost_cache[cache_key]
                    subblocks = []
                    for subblock in candidate.block_config.subblock_configs:
                        key = (subblock.kind, subblock.name)
                        if key not in teacher_counts:
                            raise RuntimeError(
                                "elastic candidate contains a subblock absent from its teacher: "
                                f"layer={elastic.layer_idx} subblock={key}"
                            )
                        subblocks.append(
                            {
                                "kind": subblock.kind,
                                "name": subblock.name,
                                "config": subblock.to_dict(),
                                "parameter_count": subblock_cost(subblock),
                                "teacher_parameter_count": teacher_counts[key],
                            }
                        )
                    layer_metadata[candidate_id] = {
                        "layer_idx": int(elastic.layer_idx),
                        "candidate_id": candidate_id,
                        "block_config": candidate.block_config.to_dict(),
                        "parameter_count": layer_costs[candidate_id],
                        "teacher_parameter_count": sum(teacher_counts.values()),
                        "subblocks": subblocks,
                    }
                by_layer[int(elastic.layer_idx)] = layer_costs
                metadata_by_layer[int(elastic.layer_idx)] = layer_metadata
            self._elastic_parameter_counts_by_width[int(width)] = by_layer
            self._elastic_candidate_metadata_by_width[int(width)] = metadata_by_layer
        # Preserve the per-rank coverage counter for compatibility. Candidate
        # identity itself is stateless from optimizer step and logical DP lane.
        self._elastic_masker.sample_step = max(
            0,
            (int(cfg.bypass.get("step_num", 1) or 1) - 1)
            * int(cfg.bypass.training.grad_accumulation_steps),
        )

    def _child_blocks_for_step(self, step: int, micro_step: int = 0):
        if self._elastic_masker is None:
            self._last_elastic_targets = {}
            return self._fixed_child_blocks
        targets = self._elastic_masker.sample_targets(
            sample_index=self._architecture_sample_index(step, micro_step),
            cycle_all=bool(self._hydra_cfg.bypass.get("elastic_cycle_all_targets", False)),
            coverage_mode=self._hydra_cfg.bypass.get("elastic_coverage_mode", None),
            selection=self._hydra_cfg.bypass.get("elastic_fixed_selection", None),
        )
        self._last_elastic_targets = dict(targets)
        children = list(self._teacher_blocks)
        for layer_idx, candidate in targets.items():
            children[layer_idx] = candidate.block_config
        return children

    def _hidden_width_for_step(self, step: int, micro_step: int = 0) -> int:
        if self._embedding_spec is None:
            if self._teacher_hidden_width is None:
                raise RuntimeError("teacher hidden width was not initialized")
            return self._teacher_hidden_width
        sample_index = self._architecture_sample_index(step, micro_step)
        generator = torch.Generator().manual_seed(
            int(self._hydra_cfg.bypass.get("elastic_seed", 42)) + 7_919 + 104_729 * sample_index
        )
        width = _select_hidden_width(
            self._hidden_widths,
            step=sample_index + 1,
            cycle=bool(self._hydra_cfg.embedding_pruning.get("cycle_widths", True)),
            policy=self._hydra_cfg.embedding_pruning.get("sampling_policy", None),
            generator=generator,
        )
        self._embedding_spec.validate_width(width)
        self._hidden_width_counts[width] += 1
        return width

    def _ple_width_for_step(self, step: int, micro_step: int = 0) -> int | None:
        if self._ple_spec is None:
            return None
        # Hidden width cycles fastest; PLE advances after one complete hidden
        # cycle so the combined global axes cover their Cartesian product.
        hidden_period = max(1, len(self._hidden_widths))
        sample_index = self._architecture_sample_index(step, micro_step)
        width = int(self._ple_widths[(sample_index // hidden_period) % len(self._ple_widths)])
        self._ple_width_counts[width] += 1
        return width

    def _validate_lane_assignment(self, selection_record: dict[str, object]) -> None:
        if self._lane_assignment_validated:
            return
        digest = json.dumps(selection_record, sort_keys=True, separators=(",", ":"))
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            gathered,
            (self._logical_dp_lane, digest),
        )
        validate_lane_architecture_assignments(gathered)
        self._lane_assignment_validated = True

    def _install_local_capture_hooks(self):
        if self._local_kd_granularity == "subblock":
            teacher_boundaries = resolve_subblock_boundaries(
                self._teacher_layers,
                self._descriptor,
                self._teacher_blocks,
            )
            student_boundaries = resolve_subblock_boundaries(
                self._student_layers,
                self._descriptor,
                self._teacher_blocks,
            )
            selected_kinds = selected_subblock_kinds(
                self._hydra_cfg.bypass.model_factory.keys_to_learn
            )
            if selected_kinds is not None:
                teacher_boundaries = {
                    key: boundary
                    for key, boundary in teacher_boundaries.items()
                    if boundary.kind in selected_kinds
                }
                student_boundaries = {
                    key: boundary
                    for key, boundary in student_boundaries.items()
                    if boundary.kind in selected_kinds
                }
            if not teacher_boundaries:
                raise RuntimeError("subblock bypass selected no local teacher boundaries")
            if set(teacher_boundaries) != set(student_boundaries):
                raise RuntimeError(
                    "teacher/student subblock boundary ownership differs: "
                    f"teacher={sorted(teacher_boundaries)}, student={sorted(student_boundaries)}"
                )
            self._teacher_subblock_boundaries = teacher_boundaries
            self._student_subblock_boundaries = student_boundaries
            self._capture_handles.extend(
                install_teacher_subblock_capture_hooks(
                    teacher_boundaries,
                    self._teacher_records,
                    capture_enabled=lambda: self._capture_enabled,
                )
            )
            return

        target_path = self._hydra_cfg.bypass.model_factory.get(
            "submodule_for_loss_calculation", None
        )
        for layer_idx, teacher_layer in self._teacher_layers.items():
            target_module = (
                teacher_layer.get_submodule(str(target_path)) if target_path else teacher_layer
            )

            def _teacher_pre(module, args, kwargs, *, idx=layer_idx):
                if self._capture_enabled:
                    self._teacher_records[idx].append(
                        {"args": _detach_tree(args), "kwargs": _detach_tree(kwargs)}
                    )

            def _teacher_target(module, args, output, *, idx=layer_idx):
                if not self._capture_enabled:
                    return
                records = self._teacher_records[idx]
                if not records:
                    raise RuntimeError(f"teacher target for layer {idx} fired without an input")
                records[-1]["target"] = _detach_tree(_output_tensor(output))

            self._capture_handles.append(
                teacher_layer.register_forward_pre_hook(_teacher_pre, with_kwargs=True)
            )
            self._capture_handles.append(target_module.register_forward_hook(_teacher_target))

        for layer_idx, student_layer in self._student_layers.items():
            target_module = (
                student_layer.get_submodule(str(target_path)) if target_path else student_layer
            )

            def _student_target(module, args, output, *, idx=layer_idx):
                self._student_target_outputs[idx] = _output_tensor(output)

            self._student_capture_handles.append(
                target_module.register_forward_hook(_student_target)
            )

    def _teacher_forward(self, batch, *, step: int):
        saved_parts = self.model_parts
        saved_pp = self.pp
        object.__setattr__(self, "model_parts", self._teacher_parts)
        object.__setattr__(self, "pp", self._teacher_pp)
        try:
            super()._forward_batch(batch, step=step)
        finally:
            object.__setattr__(self, "model_parts", saved_parts)
            object.__setattr__(self, "pp", saved_pp)

    def _local_replay_loss(
        self,
        child_blocks,
        hidden_width: int | None = None,
        ple_width: int | None = None,
        *,
        backward: bool = False,
        grad_accum: int = 1,
    ) -> tuple[torch.Tensor | None, dict[int, float], bool]:
        block_loss_name = str(self._hydra_cfg.bypass.model_factory.block_loss_func)
        loss_fn = resolve_local_kd_loss(block_loss_name)

        if self._local_kd_granularity == "subblock":
            return self._local_subblock_replay_loss(
                child_blocks,
                hidden_width,
                ple_width,
                loss_fn=loss_fn,
                backward=backward,
                grad_accum=grad_accum,
            )

        partials = []
        did_backward = False
        values: dict[int, list[torch.Tensor]] = defaultdict(list)
        for layer_idx in sorted(self._student_layers):
            records = self._teacher_records.get(layer_idx, [])
            if not records:
                continue
            student_layer = self._student_layers[layer_idx]
            with ExitStack() as layer_stack:
                layer_stack.enter_context(
                    self.prune_block_context(
                        layer_idx,
                        **_block_context_kwargs(
                            self._teacher_blocks[layer_idx],
                            child_blocks[layer_idx],
                            head_dim=self._head_dim,
                        ),
                    )
                )
                if self._embedding_spec is not None and hidden_width is not None:
                    layer_stack.enter_context(
                        hidden_width_layer_context(
                            student_layer,
                            canonical_layer_name=self._descriptor.layer_block_name(layer_idx),
                            spec=self._embedding_spec,
                            width=hidden_width,
                        )
                    )
                if self._ple_spec is not None and ple_width is not None:
                    layer_stack.enter_context(
                        ple_layer_context(
                            student_layer,
                            spec=self._ple_spec,
                            width=ple_width,
                        )
                    )
                for record_index, record in enumerate(records):
                    if "target" not in record:
                        raise RuntimeError(
                            f"teacher layer {layer_idx} capture has no target output"
                        )
                    self._student_target_outputs.pop(layer_idx, None)
                    student_layer(*record["args"], **record["kwargs"])
                    student_output = self._student_target_outputs.pop(layer_idx, None)
                    if student_output is None:
                        raise RuntimeError(f"student layer {layer_idx} produced no local KD target")
                    student_tensor = _loss_tensor(student_output)
                    teacher_tensor = _loss_tensor(record["target"])
                    if self._embedding_spec is not None and hidden_width is not None:
                        student_tensor = retained_hidden_prefix(student_tensor, hidden_width)
                        teacher_tensor = retained_hidden_prefix(teacher_tensor, hidden_width)
                    student_tensor, teacher_tensor = _mask_local_kd_tensors(
                        student_tensor,
                        teacher_tensor,
                        self._last_canonical_hidden_mask,
                        record_index=record_index,
                        record_count=len(records),
                    )
                    if tuple(student_tensor.shape) != tuple(teacher_tensor.shape):
                        raise RuntimeError(
                            f"local KD shape mismatch at layer {layer_idx}: "
                            f"student={tuple(student_tensor.shape)} teacher={tuple(teacher_tensor.shape)}"
                        )
                    loss, contributes_metric = _local_kd_loss_or_zero(
                        loss_fn,
                        student_tensor,
                        teacher_tensor,
                    )
                    if backward and loss.requires_grad:
                        loss = _backward_disjoint_loss(
                            loss,
                            grad_scaler=self.grad_scaler,
                            grad_accum=grad_accum,
                        )
                        did_backward = True
                    partials.append(loss)
                    if contributes_metric:
                        values[layer_idx].append(loss.detach())

        if not partials:
            return None, {}, False
        # Each block has disjoint trainable parameters, so summing retains the
        # same per-block gradient as the legacy one-optimizer-per-block loop.
        total = torch.stack([loss.reshape(()) for loss in partials]).sum()
        metrics = {
            layer_idx: float(torch.stack(layer_losses).mean().detach().cpu())
            for layer_idx, layer_losses in values.items()
        }
        return total, metrics, did_backward

    def _local_subblock_replay_loss(
        self,
        child_blocks,
        hidden_width: int | None,
        ple_width: int | None,
        *,
        loss_fn,
        backward: bool,
        grad_accum: int,
    ) -> tuple[torch.Tensor | None, dict[int, float], bool]:
        partials = []
        did_backward = False
        layer_values: dict[int, list[torch.Tensor]] = defaultdict(list)
        subblock_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        for key in sorted(self._student_subblock_boundaries):
            records = self._teacher_records.get(key, [])
            if not records:
                continue
            layer_idx, kind, name = key
            student_layer = self._student_layers[layer_idx]
            boundary = self._student_subblock_boundaries[key]
            with ExitStack() as layer_stack:
                layer_stack.enter_context(
                    self.prune_block_context(
                        layer_idx,
                        **_block_context_kwargs(
                            self._teacher_blocks[layer_idx],
                            child_blocks[layer_idx],
                            head_dim=self._head_dim,
                        ),
                    )
                )
                if self._embedding_spec is not None and hidden_width is not None:
                    layer_stack.enter_context(
                        hidden_width_layer_context(
                            student_layer,
                            canonical_layer_name=self._descriptor.layer_block_name(layer_idx),
                            spec=self._embedding_spec,
                            width=hidden_width,
                        )
                    )
                if self._ple_spec is not None and ple_width is not None:
                    layer_stack.enter_context(
                        ple_layer_context(
                            student_layer,
                            spec=self._ple_spec,
                            width=ple_width,
                        )
                    )
                for record_index, record in enumerate(records):
                    student_tensor = _loss_tensor(replay_subblock(boundary, record))
                    teacher_tensor = _loss_tensor(record.target)
                    if self._embedding_spec is not None and hidden_width is not None:
                        student_tensor = retained_hidden_prefix(student_tensor, hidden_width)
                        teacher_tensor = retained_hidden_prefix(teacher_tensor, hidden_width)
                    student_tensor, teacher_tensor = _mask_local_kd_tensors(
                        student_tensor,
                        teacher_tensor,
                        self._last_canonical_hidden_mask,
                        record_index=record_index,
                        record_count=len(records),
                    )
                    if tuple(student_tensor.shape) != tuple(teacher_tensor.shape):
                        raise RuntimeError(
                            f"local subblock KD shape mismatch at {key}: "
                            f"student={tuple(student_tensor.shape)} "
                            f"teacher={tuple(teacher_tensor.shape)}"
                        )
                    loss, contributes_metric = _local_kd_loss_or_zero(
                        loss_fn,
                        student_tensor,
                        teacher_tensor,
                    )
                    if backward and loss.requires_grad:
                        loss = _backward_disjoint_loss(
                            loss,
                            grad_scaler=self.grad_scaler,
                            grad_accum=grad_accum,
                        )
                        did_backward = True
                    partials.append(loss)
                    if contributes_metric:
                        detached = loss.detach()
                        layer_values[layer_idx].append(detached)
                        subblock_values[f"{layer_idx}:{kind}:{name}"].append(detached)

        if not partials:
            self._current_subblock_metrics = {}
            return None, {}, False
        self._current_subblock_metrics = {
            key: float(torch.stack(values).mean().cpu()) for key, values in subblock_values.items()
        }
        total = torch.stack([loss.reshape(()) for loss in partials]).sum()
        metrics = {
            layer_idx: float(torch.stack(values).mean().cpu())
            for layer_idx, values in layer_values.items()
        }
        return total, metrics, did_backward

    def _prime_pp_shapes(self, batch):
        if self._teacher_pp is None or self._pp_metadata_ready():
            return
        mprint("[bypass/automodel] priming PP shape metadata with captures disabled")
        self._capture_enabled = False
        with torch.no_grad():
            self._teacher_forward(batch, step=-1)

    def _set_learning_rate(self, step: int):
        training = self._hydra_cfg.bypass.training
        lr = (
            get_learning_rate(self._hydra_cfg, step)
            if bool(training.decay_lr)
            else float(training.learning_rate)
        )
        optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr

    def _record_periodic_checkpoint(self, checkpoint_path: Path, step: int) -> None:
        """Expose an AutoModel checkpoint through the legacy resume manifest."""
        from ...bypass_distillation.bypass_utils import update_bypass_checkpoint_state

        self._validate_resumable_checkpoint(checkpoint_path)

        alias = Path(self._hydra_cfg.bypass.experiment_dir) / f"step-{step:06d}-ckpt"
        if self.dist_env.is_main:
            (checkpoint_path / "args.json").write_text(
                json.dumps(
                    OmegaConf.to_container(self._hydra_cfg.bypass, resolve=True),
                    indent=2,
                    default=str,
                )
                + "\n"
            )
            update_bypass_checkpoint_state(self._hydra_cfg, checkpoint_path, "resume")
            (checkpoint_path / "saving_completed").touch()
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            alias.symlink_to(checkpoint_path.resolve(), target_is_directory=True)
            latest = Path(self._hydra_cfg.bypass.experiment_dir) / "latest"
            temporary = latest.with_name(f".latest_automodel_{os.getpid()}")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(alias.name)
            temporary.replace(latest)
        torch.distributed.barrier()

    def _validate_resumable_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        require_completed: bool = False,
    ) -> dict[str, object]:
        """Validate one shared AutoModel checkpoint and broadcast the result."""

        require_distributed_path_consensus(checkpoint_path, "checkpoint load")
        validation_result: list[dict[str, object] | None] = [None]
        if self.dist_env.is_main:
            try:
                if require_completed and not (checkpoint_path / "saving_completed").is_file():
                    raise RuntimeError(
                        f"checkpoint has no saving_completed marker: {checkpoint_path}"
                    )
                validation_result[0] = validate_automodel_bypass_checkpoint(
                    checkpoint_path,
                    expected_rng_ranks=range(self._get_dp_group_size(include_cp=True)),
                )
            except BaseException as error:
                validation_result[0] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
        torch.distributed.broadcast_object_list(validation_result, src=0)
        result = validation_result[0] or {
            "status": "error",
            "error": "rank 0 returned no checkpoint validation result",
        }
        if result.get("status") != "complete":
            raise RuntimeError(
                f"refusing to use incomplete checkpoint {checkpoint_path}: "
                f"{result.get('error', result)}"
            )
        return result

    def _prepare_checkpoint_save(self, checkpoint_path: Path) -> None:
        """Ensure every rank saves to one fresh checkpoint transaction path."""

        require_distributed_path_consensus(checkpoint_path, "checkpoint save")
        quarantine_error = [None]
        if self.dist_env.is_main:
            try:
                quarantined = quarantine_incomplete_checkpoint(checkpoint_path)
                if quarantined is not None:
                    mprint(
                        "[bypass/automodel] quarantined incomplete checkpoint "
                        f"{checkpoint_path} -> {quarantined}"
                    )
            except BaseException as error:
                quarantine_error[0] = f"{type(error).__name__}: {error}"
        torch.distributed.broadcast_object_list(quarantine_error, src=0)
        if quarantine_error[0] is not None:
            raise RuntimeError(
                f"cannot prepare checkpoint transaction {checkpoint_path}: {quarantine_error[0]}"
            )
        torch.distributed.barrier()

    def _run_local_validation(
        self,
        child_blocks,
        hidden_width: int | None = None,
        ple_width: int | None = None,
    ) -> float | None:
        if self._local_val_dataloader is None:
            return None
        local_values: dict[int, list[float]] = defaultdict(list)
        for part in self.model_parts:
            part.eval()
        try:
            with torch.no_grad():
                for batch in self._local_val_dataloader:
                    self._teacher_records.clear()
                    self._capture_enabled = True
                    self._teacher_forward(batch, step=-2)
                    self._capture_enabled = False
                    with self._forward_autocast_context():
                        _, layer_metrics, _ = self._local_replay_loss(
                            child_blocks,
                            hidden_width,
                            ple_width,
                        )
                    for layer_idx, value in layer_metrics.items():
                        local_values[layer_idx].append(value)
                    self._teacher_records.clear()
        finally:
            self._capture_enabled = False
            for part in self.model_parts:
                part.train()

        local_means = {
            layer_idx: sum(values) / len(values) for layer_idx, values in local_values.items()
        }
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_means)
        merged: dict[int, list[float]] = defaultdict(list)
        for rank_metrics in gathered:
            for layer_idx, value in (rank_metrics or {}).items():
                merged[int(layer_idx)].append(float(value))
        per_layer = [sum(values) / len(values) for values in merged.values()]
        return sum(per_layer) / len(per_layer) if per_layer else None

    def run_local_distillation(self) -> tuple[Path, dict[str, Any]]:
        """Execute the local-KD loop and write a consolidated AutoModel checkpoint."""
        from nemo_automodel.components.distributed.utils import get_sync_ctx
        from nemo_automodel.components.training.utils import (
            prepare_after_first_microbatch,
            prepare_for_final_backward,
            prepare_for_grad_accumulation,
            scale_grads_and_clip_grad_norm,
        )

        training = self._hydra_cfg.bypass.training
        max_steps = int(training.max_steps)
        grad_accum = int(training.grad_accumulation_steps)
        single_batch_overfit = bool(self._hydra_cfg.bypass.get("single_batch_overfit", False))
        overfit_mode = str(self._hydra_cfg.bypass.get("overfit_probe_mode", "smallest_fixed"))
        resample_overfit_structure = bool(
            self._hydra_cfg.bypass.get("single_batch_overfit_resample_structure", False)
        )
        overfit_trend_window = int(self._hydra_cfg.bypass.get("overfit_trend_window", 4) or 4)
        overfit_minimum_relative_decrease = float(
            self._hydra_cfg.bypass.get("overfit_minimum_relative_decrease", 0.0) or 0.0
        )
        overfit_source_identity = self._hydra_cfg.bypass.get(
            "overfit_source_checkpoint_identity", None
        )
        data_iter = iter(self.dataloader)

        def next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                return next(data_iter)

        skip_batches = int(training.get("skip_first_batches", 0) or 0)
        if self._resume_path:
            # The Puzzletron loader is deterministic but not stateful. Recreate
            # its position from the persisted token count so resumed training
            # does not silently replay the beginning of the dataset.
            skip_batches += int(self._hydra_cfg.bypass.token_count) // int(training.tokens_per_iter)
        for _ in range(skip_batches):
            next_batch()

        first_batch = next_batch()
        self._prime_pp_shapes(first_batch)
        pending_first_batch = first_batch
        overfit_batch = first_batch if single_batch_overfit else None
        overfit_structure: tuple[int, Any, int | None, dict[int, object]] | None = None
        last_metrics: dict[str, Any] = {}
        start = time.monotonic()

        start_step = int(self._hydra_cfg.bypass.get("step_num", 1) or 1)
        history_root = Path(self._hydra_cfg.puzzle_dir) / "artifacts" / "bypass"
        if single_batch_overfit:
            history_root = history_root / "overfit_probe" / overfit_mode
        history_path = history_root / "local_kd_loss_history.json"
        observation_path = history_root / "dp_observations.jsonl"
        candidate_catalog_path = history_root / "candidate_catalog.json"
        loss_history: list[dict[str, Any]] = []
        probe_summary: dict[str, Any] = {}
        if self._resume_path and history_path.is_file():
            prior = json.loads(history_path.read_text())
            loss_history = [
                row
                for row in list(prior.get("records") or [])
                if int(row.get("step", 0)) < start_step
            ]
        observation_writer = None
        candidate_catalog = None
        if self.dist_env.is_main and self._elastic_masker is not None:
            catalog_entries = (
                json.loads(candidate_catalog_path.read_text())
                if self._resume_path and candidate_catalog_path.is_file()
                else None
            )
            candidate_catalog = CandidateCatalog(catalog_entries)
            observation_writer = ObservationWriter(observation_path)
            observation_writer.truncate_after_step(start_step - 1)

        def write_loss_history() -> None:
            if not self.dist_env.is_main:
                return
            history_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = history_path.with_name(f".{history_path.name}.{os.getpid()}.tmp")
            temporary_path.write_text(
                json.dumps(
                    {
                        "max_steps": max_steps,
                        "start_step": 1,
                        "granularity": self._local_kd_granularity,
                        "loss_name": str(self._hydra_cfg.bypass.model_factory.block_loss_func),
                        "mode": overfit_mode if single_batch_overfit else None,
                        "source_checkpoint_identity": overfit_source_identity,
                        "resample_structure": resample_overfit_structure,
                        "trend_window": overfit_trend_window,
                        "minimum_relative_decrease": overfit_minimum_relative_decrease,
                        "dp_observation_path": str(observation_path),
                        "candidate_catalog_path": str(candidate_catalog_path),
                        "summary": probe_summary,
                        "records": loss_history,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, history_path)

        last_checkpoint_time = start
        for step in range(start_step, max_steps + 1):
            self._hydra_cfg.bypass.step_num = step
            lr = self._set_learning_rate(step)
            accumulated: dict[int, list[float]] = defaultdict(list)
            accumulated_subblocks: dict[str, list[float]] = defaultdict(list)
            selection_records: list[dict[str, object]] = []
            micro_observation_payloads: list[dict[str, object]] = []
            did_backward = False
            prepare_for_grad_accumulation(self.model_parts, pp_enabled=False)
            for micro_step in range(grad_accum):
                if overfit_structure is None or resample_overfit_structure:
                    hidden_width = self._hidden_width_for_step(step, micro_step)
                    child_blocks = self._child_blocks_for_step(step, micro_step)
                    ple_width = self._ple_width_for_step(step, micro_step)
                    if single_batch_overfit and not resample_overfit_structure:
                        overfit_structure = (
                            hidden_width,
                            child_blocks,
                            ple_width,
                            dict(self._last_elastic_targets),
                        )
                else:
                    hidden_width, child_blocks, ple_width, fixed_targets = overfit_structure
                    self._last_elastic_targets = dict(fixed_targets)
                selection_record = _elastic_selection_record(
                    step=step,
                    micro_step=micro_step,
                    hidden_width=hidden_width,
                    ple_width=ple_width,
                    targets=(
                        self._last_elastic_targets if self._elastic_masker is not None else {}
                    ),
                    parameter_counts=self._elastic_parameter_counts_by_width.get(
                        int(hidden_width), {}
                    ),
                )
                selection_records.append(selection_record)
                if self._elastic_masker is not None:
                    self._elastic_selection_history.append(selection_record)
                self._validate_lane_assignment(selection_record)
                self._last_child_blocks = child_blocks
                if single_batch_overfit:
                    batch = overfit_batch
                    pending_first_batch = None
                elif pending_first_batch is not None:
                    batch = pending_first_batch
                    pending_first_batch = None
                else:
                    batch = next_batch()
                self._hydra_cfg.bypass.iter_num = (
                    int(self._hydra_cfg.bypass.get("iter_num", 0) or 0) + 1
                )
                self._hydra_cfg.bypass.token_count = int(
                    self._hydra_cfg.bypass.get("token_count", 0) or 0
                ) + int(training.tokens_per_iter)
                self._teacher_records.clear()
                self._capture_enabled = True
                with torch.no_grad():
                    self._teacher_forward(batch, step=step)
                self._capture_enabled = False

                with ExitStack() as stack:
                    final_microbatch = micro_step == grad_accum - 1
                    if final_microbatch:
                        prepare_for_final_backward(self.model_parts, pp_enabled=False)
                    for part in self.model_parts:
                        stack.enter_context(
                            get_sync_ctx(
                                part,
                                final_microbatch,
                                defer_fsdp_grad_sync=getattr(
                                    self.distributed_config, "defer_fsdp_grad_sync", True
                                ),
                            )
                        )
                    with self._forward_autocast_context():
                        local_loss, layer_metrics, micro_did_backward = self._local_replay_loss(
                            child_blocks,
                            hidden_width,
                            ple_width,
                            backward=True,
                            grad_accum=grad_accum,
                        )
                    if micro_did_backward:
                        print(
                            f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                            f"step={step} backward_done",
                            flush=True,
                        )
                        did_backward = True
                    elif local_loss is not None:
                        # A fully bypassed block has no student parameter on its
                        # output path: replay returns the detached teacher input
                        # unchanged.  Its loss is still a valid endpoint metric,
                        # but autograd correctly has nothing to optimize.
                        print(
                            f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                            f"step={step} backward_skipped_no_trainable_path",
                            flush=True,
                        )
                    for layer_idx, value in layer_metrics.items():
                        accumulated[layer_idx].append(value)
                    for subblock_key, value in self._current_subblock_metrics.items():
                        accumulated_subblocks[subblock_key].append(value)
                    if self._elastic_masker is not None:
                        micro_observation_payloads.append(
                            {
                                "dp_lane": self._logical_dp_lane,
                                "selection": selection_record,
                                "per_layer_loss": dict(layer_metrics),
                                "per_subblock_loss": dict(self._current_subblock_metrics),
                            }
                        )
                self._teacher_records.clear()
                if micro_step == 0:
                    prepare_after_first_microbatch()

            optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
            if did_backward:
                print(
                    f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                    f"step={step} grad_sync_start",
                    flush=True,
                )
                for optimizer in optimizers:
                    self.grad_scaler.unscale_(optimizer)
                grad_norm = scale_grads_and_clip_grad_norm(
                    float(training.grad_clip) if training.grad_clip is not None else None,
                    self.model_parts,
                    norm_type=2.0,
                    pp_enabled=False,
                    device_mesh=self.device_mesh,
                    moe_mesh=self.moe_mesh,
                    ep_axis_name=(
                        "ep"
                        if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names
                        else None
                    ),
                    pp_axis_name=None,
                    foreach=True,
                    num_label_tokens=None,
                    dp_group_size=self._get_dp_group_size(include_cp=True),
                )
                print(
                    f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                    f"step={step} grad_sync_done",
                    flush=True,
                )
                for optimizer in optimizers:
                    self.grad_scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.update()
            else:
                grad_norm = 0.0
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

            local_layer_metrics = {
                layer_idx: sum(values) / len(values) for layer_idx, values in accumulated.items()
            }
            local_subblock_metrics = {
                key: sum(values) / len(values) for key, values in accumulated_subblocks.items()
            }
            print(
                f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                f"step={step} metrics_gather_start",
                flush=True,
            )
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local_layer_metrics)
            gathered_subblocks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_subblocks, local_subblock_metrics)
            gathered_observation_payloads = None
            if self._elastic_masker is not None:
                gathered_observation_payloads = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(
                    gathered_observation_payloads,
                    micro_observation_payloads,
                )
            print(
                f"[bypass/automodel] rank={torch.distributed.get_rank()} "
                f"step={step} metrics_gather_done",
                flush=True,
            )
            merged: dict[int, list[float]] = defaultdict(list)
            for rank_metrics in gathered:
                for layer_idx, value in (rank_metrics or {}).items():
                    merged[int(layer_idx)].append(float(value))
            per_layer = {
                str(layer_idx): sum(values) / len(values)
                for layer_idx, values in sorted(merged.items())
            }
            if not per_layer:
                raise RuntimeError(
                    "local KD batch contains no valid tokens after applying the hidden mask"
                )
            merged_subblocks: dict[str, list[float]] = defaultdict(list)
            for rank_metrics in gathered_subblocks:
                for key, value in (rank_metrics or {}).items():
                    merged_subblocks[str(key)].append(float(value))
            per_subblock = {
                key: sum(values) / len(values) for key, values in sorted(merged_subblocks.items())
            }
            mean_loss = sum(per_layer.values()) / max(len(per_layer), 1)
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.detach().float().cpu().item()
            dp_observations = []
            if gathered_observation_payloads is not None:
                step_catalog = CandidateCatalog()
                for micro_step in range(grad_accum):
                    micro_observations, micro_catalog = merge_rank_observations(
                        [
                            rank_payloads[micro_step]
                            for rank_payloads in gathered_observation_payloads
                        ],
                        step=step,
                        micro_step=micro_step,
                        granularity=self._local_kd_granularity,
                        learning_rate=lr,
                        grad_norm=float(grad_norm) if grad_norm is not None else None,
                        elapsed_seconds=time.monotonic() - start,
                        candidate_metadata=self._elastic_candidate_metadata_by_width,
                    )
                    dp_observations.extend(micro_observations)
                    step_catalog.merge(micro_catalog)
                if self.dist_env.is_main:
                    candidate_catalog.merge(step_catalog)
                    candidate_catalog.write(candidate_catalog_path)
                    observation_writer.append_step(step, dp_observations)
            last_metrics = {
                "step": step,
                "loss": mean_loss,
                "per_layer_loss": per_layer,
                "per_subblock_loss": per_subblock,
                "subblock_coverage": sorted(per_subblock),
                "lr": lr,
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "elapsed_seconds": time.monotonic() - start,
                "hidden_width": hidden_width,
                "ple_width": ple_width,
                "elastic_selection": selection_record,
                "elastic_selections": selection_records,
                "dp_observations": [point.to_dict() for point in dp_observations],
            }
            history_record = {
                "step": step,
                "loss": mean_loss,
                "per_layer_loss": per_layer,
                "per_subblock_loss": per_subblock,
                "subblock_coverage": sorted(per_subblock),
                "lr": lr,
                "grad_norm": last_metrics["grad_norm"],
                "elapsed_seconds": last_metrics["elapsed_seconds"],
                "hidden_width": hidden_width,
                "ple_width": ple_width,
                "elastic_selection": selection_record,
                "elastic_selections": selection_records,
                "dp_observation_count": len(dp_observations),
                "dp_observation_path": str(observation_path),
            }
            eval_interval = int(training.get("eval_interval", 0) or 0)
            if eval_interval > 0 and step % eval_interval == 0:
                val_loss = self._run_local_validation(
                    child_blocks,
                    hidden_width,
                    ple_width,
                )
                if val_loss is not None:
                    last_metrics["val_loss"] = val_loss
                    history_record["val_loss"] = val_loss
            loss_history.append(history_record)
            if self.dist_env.is_main and (
                step in (1, max_steps) or step % int(training.log_interval) == 0
            ):
                val_suffix = (
                    f" val_loss={last_metrics['val_loss']:.6g}"
                    if "val_loss" in last_metrics
                    else ""
                )
                mprint(
                    "[bypass/automodel] "
                    f"step={step}/{max_steps} loss={mean_loss:.6g} "
                    f"layers={len(per_layer)} lr={lr:.3g} "
                    f"grad_norm={last_metrics['grad_norm']:.4g} "
                    f"hidden_width={hidden_width} ple_width={ple_width}{val_suffix}"
                )

            save_interval = int(
                self._hydra_cfg.bypass.model.model_overrides.get("save_interval", 0) or 0
            )
            save_at_steps = {
                int(value)
                for value in (
                    self._hydra_cfg.bypass.model.model_overrides.get("save_at_steps", []) or []
                )
            }
            save_interval_seconds = float(
                self._hydra_cfg.bypass.model.model_overrides.get("save_interval_seconds", 0) or 0
            )
            step_due = step in save_at_steps or (save_interval > 0 and step % save_interval == 0)
            time_due = (
                save_interval_seconds > 0
                and time.monotonic() - last_checkpoint_time >= save_interval_seconds
            )
            if step < max_steps and (step_due or time_due):
                write_loss_history()
                periodic_path = (
                    Path(self.checkpointer.config.checkpoint_dir) / f"epoch_0_step_{step}"
                )
                self._prepare_checkpoint_save(periodic_path)
                # Periodic checkpoints exist for interruption recovery, not HF
                # publication.  Avoid consolidating the full model on every
                # timed save: that is both slow and leaves most PP ranks idle.
                # The final checkpoint still uses the configured consolidation
                # mode and remains publishable through the normal path below.
                from nemo_automodel.components.checkpoint.config import SaveConsolidatedMode

                configured_consolidation = self.checkpointer.config.save_consolidated
                self.checkpointer.config.save_consolidated = SaveConsolidatedMode.FALSE
                try:
                    self.save_checkpoint(
                        epoch=0,
                        step=step,
                        train_loss=float(last_metrics.get("loss", 0.0)),
                        val_loss=None,
                    )
                finally:
                    self.checkpointer.config.save_consolidated = configured_consolidation
                self._finalize_pending_checkpoint()
                self._record_periodic_checkpoint(periodic_path, step)
                last_checkpoint_time = time.monotonic()

        if self._elastic_masker is not None:
            last_metrics["elastic_coverage"] = {
                "steps": self._elastic_masker.sample_step,
                "coverage_mode": self._hydra_cfg.bypass.get("elastic_coverage_mode", None),
                "coverage_schedule": self._elastic_masker.coverage_schedule_manifest(),
                "configured_options": {
                    f"layer_{elastic.layer_idx}": [
                        {
                            "candidate_id": str(candidate.identity.value),
                            "block_config": candidate.block_config.to_dict(),
                            "changed_axes": dict(candidate.metadata.get("slice_axes") or {}),
                        }
                        for candidate in elastic.sampler.sizes
                    ]
                    for elastic in self._elastic_masker.elastics
                },
                "observed_options": sorted(
                    self._elastic_masker.coverage.values(),
                    key=lambda item: (int(item["layer_idx"]), str(item["candidate_id"])),
                ),
            }
            last_metrics["elastic_selection_history"] = list(self._elastic_selection_history)

        global_hidden_width_counts = dict(self._hidden_width_counts)
        global_ple_width_counts = dict(self._ple_width_counts)
        if (
            self._embedding_spec is not None or self._ple_spec is not None
        ) and not single_batch_overfit:
            payload = {
                "dp_lane": self._logical_dp_lane,
                "hidden_width_counts": dict(self._hidden_width_counts),
                "ple_width_counts": dict(self._ple_width_counts),
            }
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered_axis_counts = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered_axis_counts, payload)
            else:
                gathered_axis_counts = [payload]
            global_hidden_width_counts = _merge_lane_axis_counts(
                gathered_axis_counts,
                count_key="hidden_width_counts",
            )
            global_ple_width_counts = _merge_lane_axis_counts(
                gathered_axis_counts,
                count_key="ple_width_counts",
            )

        if self._embedding_spec is not None and not single_batch_overfit:
            missing_widths = sorted(set(self._hidden_widths) - set(global_hidden_width_counts))
            if missing_widths:
                raise RuntimeError(
                    "nested bypass did not sample every configured hidden width: "
                    f"missing={missing_widths} counts={global_hidden_width_counts}"
                )
            last_metrics["hidden_width_coverage"] = {
                "configured_widths": list(self._hidden_widths),
                "sampling_policy": self._hydra_cfg.embedding_pruning.get("sampling_policy", None),
                "configured_probabilities": {
                    str(width): float(probability)
                    for width, probability in zip(
                        self._hidden_widths,
                        inverse_width_probs(self._hidden_widths),
                    )
                }
                if self._hydra_cfg.embedding_pruning.get("sampling_policy", None) == "inverse_width"
                else None,
                "selection_counts": {
                    str(width): int(global_hidden_width_counts.get(width, 0))
                    for width in self._hidden_widths
                },
            }

        if self._ple_spec is not None and not single_batch_overfit:
            missing_ple_widths = sorted(set(self._ple_widths) - set(global_ple_width_counts))
            if missing_ple_widths:
                raise RuntimeError(
                    "nested bypass did not sample every configured PLE width: "
                    f"missing={missing_ple_widths} "
                    f"counts={global_ple_width_counts}"
                )
            last_metrics["ple_width_coverage"] = {
                "configured_widths": list(self._ple_widths),
                "selection_counts": {
                    str(width): int(global_ple_width_counts.get(width, 0))
                    for width in self._ple_widths
                },
            }

        last_metrics["observability"] = self.observability_metadata()
        comparable_trend = _overfit_loss_records_are_comparable(
            single_batch_overfit=single_batch_overfit,
            resample_structure=resample_overfit_structure,
        ) or (
            not single_batch_overfit
            and self._elastic_masker is None
            and len(self._hidden_widths) <= 1
            and len(self._ple_widths) <= 1
        )
        loss_trend = _loss_trend_summary(
            loss_history,
            comparable=comparable_trend,
            window_size=overfit_trend_window if single_batch_overfit else 4,
            minimum_relative_decrease=(
                overfit_minimum_relative_decrease if single_batch_overfit else 0.0
            ),
        )
        if loss_trend:
            last_metrics["loss_trend"] = loss_trend

        if single_batch_overfit:
            fixed_targets = overfit_structure[3] if overfit_structure is not None else {}
            structure_identities = set()
            for record in loss_history:
                selections = record.get("elastic_selections") or (record.get("elastic_selection"),)
                for raw_selection in selections:
                    selection = dict(raw_selection or {})
                    selection.pop("step", None)
                    selection.pop("micro_step", None)
                    structure_identities.add(
                        json.dumps(selection, sort_keys=True, separators=(",", ":"))
                    )
            distinct_structure_count = len(structure_identities)
            multiple_legal_structures = (
                len(self._hidden_widths) > 1
                or len(self._ple_widths) > 1
                or (
                    self._elastic_masker is not None
                    and any(
                        len(elastic.sampler.sizes) > 1 for elastic in self._elastic_masker.elastics
                    )
                )
            )
            diversity_passed = (
                overfit_mode != "diverse_resampled"
                or not multiple_legal_structures
                or distinct_structure_count > 1
            )
            probe_summary.update(
                {
                    "mode": overfit_mode,
                    "source_checkpoint_identity": overfit_source_identity,
                    "steps": len(loss_history),
                    "distinct_structure_count": distinct_structure_count,
                    "multiple_legal_structures": multiple_legal_structures,
                    "diversity_passed": diversity_passed,
                    "loss_trend": loss_trend,
                }
            )
            last_metrics["overfit_probe"] = {
                "single_batch": True,
                "mode": overfit_mode,
                "resample_structure": resample_overfit_structure,
                "source_checkpoint_identity": overfit_source_identity,
                "repetitions": max_steps,
                "distinct_structure_count": distinct_structure_count,
                "multiple_legal_structures": multiple_legal_structures,
                "diversity_passed": diversity_passed,
                "hidden_width": (overfit_structure[0] if overfit_structure is not None else None),
                "ple_width": (overfit_structure[2] if overfit_structure is not None else None),
                "layers": [
                    {
                        "layer_idx": int(layer_idx),
                        "candidate_id": str(candidate.identity.value),
                        "changed_axes": dict(candidate.metadata.get("slice_axes") or {}),
                    }
                    for layer_idx, candidate in sorted(fixed_targets.items())
                ],
            }

        last_metrics["loss_history"] = loss_history
        probe_findings: list[dict[str, Any]] = []
        if loss_trend and loss_trend["hard_gate_passed"] is False:
            probe_findings.append(
                {
                    "stage": "bypass_sanity",
                    "message": (
                        "local KD acceptance loss did not decrease: "
                        f"mode={overfit_mode if single_batch_overfit else 'training'} "
                        f"trend={loss_trend}"
                    ),
                    "evidence": {"trend": loss_trend, "history_path": str(history_path)},
                    "severity": "warning",
                }
            )
        if single_batch_overfit and not probe_summary.get("diversity_passed", True):
            probe_findings.append(
                {
                    "stage": "bypass_sanity",
                    "message": (
                        "diverse fixed-batch overfit probe did not sample multiple structures: "
                        f"mode={overfit_mode} summary={probe_summary}"
                    ),
                    "evidence": {"summary": probe_summary, "history_path": str(history_path)},
                    "severity": "warning",
                }
            )
        probe_summary["findings"] = probe_findings
        probe_summary["passed"] = not probe_findings
        write_loss_history()

        self._hydra_cfg.bypass.step_num = max_steps
        checkpoint_path = (
            Path(self.checkpointer.config.checkpoint_dir) / f"epoch_0_step_{max_steps}"
        )
        self._prepare_checkpoint_save(checkpoint_path)
        self.save_checkpoint(
            epoch=0,
            step=max_steps,
            train_loss=float(last_metrics.get("loss", 0.0)),
            val_loss=None,
        )
        self._finalize_pending_checkpoint()
        resume_validation = self._validate_resumable_checkpoint(checkpoint_path)
        from ...bypass_distillation.checkpointing import validate_consolidated_hf_checkpoint

        export_consolidated = _consolidated_export_enabled(
            self.checkpointer.config.save_consolidated
        )
        validation_result: list[dict[str, Any] | None] = [None]
        if export_consolidated:
            if self.dist_env.is_main:
                try:
                    validation_result[0] = validate_consolidated_hf_checkpoint(
                        checkpoint_path / "model" / "consolidated",
                        expected_layer_prefixes=self._expected_checkpoint_layer_prefixes,
                    )
                except BaseException as exc:
                    validation_result[0] = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            torch.distributed.broadcast_object_list(validation_result, src=0)
            checkpoint_validation = validation_result[0] or {
                "status": "error",
                "error": "rank 0 returned no checkpoint validation result",
            }
            if checkpoint_validation.get("status") != "complete":
                raise RuntimeError(
                    "AutoModel local-KD checkpoint failed publication validation: "
                    f"{checkpoint_validation.get('error', checkpoint_validation)}"
                )
        else:
            checkpoint_validation = {
                "status": "skipped",
                "reason": "consolidated HF export disabled",
            }
        last_metrics["checkpoint_validation"] = {
            "resume": resume_validation,
            "consolidated": checkpoint_validation,
        }
        if export_consolidated and self.dist_env.is_main:
            _copy_hf_auxiliary_assets(
                Path(self.cfg.model.pretrained_model_name_or_path),
                checkpoint_path / "model" / "consolidated",
            )
        torch.distributed.barrier()
        return checkpoint_path, last_metrics

    def close(self):
        self.close_observability()
        inherited_handle = getattr(self, "_handle", None)
        if inherited_handle is not None:
            inherited_handle.remove()
            self._handle = None
        for handle in reversed(self._student_capture_handles):
            handle.remove()
        for handle in reversed(self._capture_handles):
            handle.remove()
        self._student_capture_handles.clear()
        self._capture_handles.clear()
        metric_logger = getattr(self, "metric_logger_train", None)
        if metric_logger is not None:
            metric_logger.close()
        for logger_ in _iter_metric_loggers(getattr(self, "metric_logger_valid", None)):
            logger_.close()
        if getattr(self, "checkpointer", None) is not None:
            self.checkpointer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
