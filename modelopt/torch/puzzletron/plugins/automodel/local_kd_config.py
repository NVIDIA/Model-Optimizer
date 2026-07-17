# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a NeMo AutoModel recipe for Puzzletron block-local distillation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config import (
    _as_dict,
    _inject_canonical_data,
    _int_or_default,
    build_stage_recipe_config,
    inject_descriptor_model_kwargs,
    inject_descriptor_pipeline_config,
)

__all__ = ["build_local_kd_recipe_config", "prepare_local_kd_training_budget"]


def prepare_local_kd_training_budget(hydra_cfg) -> None:
    """Resolve the legacy token budget into optimizer and microbatch step counts."""
    training = hydra_cfg.bypass.training
    micro_batch_size = int(training.micro_batch_size)
    block_size = int(hydra_cfg.bypass.data.block_size)
    grad_accumulation_steps = int(training.grad_accumulation_steps)
    if micro_batch_size <= 0 or block_size <= 0 or grad_accumulation_steps <= 0:
        raise ValueError(
            "bypass micro_batch_size, block_size, and grad_accumulation_steps must be positive"
        )
    if str(training.get("grad_clip_type", "norm")).lower() != "norm":
        raise ValueError(
            "AutoModel local KD supports distributed norm clipping only; "
            f"got grad_clip_type={training.get('grad_clip_type')!r}"
        )

    training.tokens_per_iter = micro_batch_size * block_size
    requested_iters = math.ceil(int(training.training_tokens) / training.tokens_per_iter)
    training.max_steps = math.ceil(requested_iters / grad_accumulation_steps)
    forced_overfit_steps = int(
        hydra_cfg.bypass.get("single_batch_overfit_steps", 0) or 0
    )
    if forced_overfit_steps:
        training.max_steps = forced_overfit_steps
    training.max_iters = training.max_steps * grad_accumulation_steps
    training.max_token_count = training.max_iters * training.tokens_per_iter
    training.lr_decay_steps = training.max_steps
    training.min_lr = float(training.learning_rate) * float(
        training.get("min_lr_factor", 0.0) or 0.0
    )


def _teacher_dir(hydra_cfg) -> Path:
    if bool(hydra_cfg.bypass.get("elastic", False)):
        sorted_teacher = Path(hydra_cfg.puzzle_dir) / "ckpts" / "sorted_teacher"
        if not (sorted_teacher / "config.json").exists():
            raise FileNotFoundError(
                "AutoModel elastic local KD requires the sorted teacher at "
                f"{sorted_teacher}; run the pruning/sorting stage first"
            )
        return sorted_teacher
    return Path(hydra_cfg.teacher_dir)


def _model_config(
    hydra_cfg,
    model_path: Path,
    base_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_cfg = _as_dict(hydra_cfg.get("model", None))
    configured = dict(base_model or {})
    data_cfg = _as_dict(hydra_cfg.get("data", None))
    model_target = (
        "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
        if data_cfg.get("modality") == "multimodal"
        else "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
    )
    configured.update({
        "_target_": model_target,
        "pretrained_model_name_or_path": str(model_path),
        "anymodel_descriptor": str(hydra_cfg.descriptor),
        "force_hf": bool(model_cfg.get("force_hf", True)),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": str(
            hydra_cfg.bypass.get(
                "dtype",
                model_cfg.get("torch_dtype", configured.get("torch_dtype", "bf16")),
            )
        ),
    })
    if model_cfg.get("attn_implementation") is not None:
        configured["attn_implementation"] = model_cfg["attn_implementation"]
    return configured


def _logical_dp_size(hydra_cfg, recipe_distributed: dict[str, Any]) -> int:
    """Return sample-parallel lanes after removing the EP overlay from AutoModel DP."""

    del hydra_cfg  # The generated recipe is the canonical topology source.
    automodel_dp = max(_int_or_default(recipe_distributed.get("dp_size"), 1), 1)
    ep_size = max(_int_or_default(recipe_distributed.get("ep_size"), 1), 1)
    logical_dp, remainder = divmod(automodel_dp, ep_size)
    if remainder:
        raise ValueError(
            "AutoModel dp_size must be divisible by ep_size; "
            f"got dp_size={automodel_dp}, ep_size={ep_size}"
        )
    return logical_dp


def build_local_kd_recipe_config(hydra_cfg) -> dict[str, Any]:
    """Translate Puzzletron bypass configuration to an AutoModel recipe dictionary.

    The bypass stage owns its parallel mesh. Puzzletron generates the stable recipe
    boilerplate and then injects model identity, training budget, and checkpoint location.
    """
    automodel_cfg = hydra_cfg.bypass.get("automodel", None)
    recipe = build_stage_recipe_config(automodel_cfg)

    teacher_dir = _teacher_dir(hydra_cfg)
    base_model = dict(recipe.get("model", {}))
    teacher_model = _model_config(hydra_cfg, teacher_dir, base_model)
    # Resume is restored through AutoModel's distributed checkpointer after the
    # student is tracked. Always construct the student from the canonical
    # teacher config; an AutoModel checkpoint directory is not necessarily a
    # Hugging Face from_pretrained source.
    student_model = _model_config(hydra_cfg, teacher_dir, base_model)

    recipe["model"] = teacher_model
    recipe["student_model"] = student_model
    for model_key in ("model", "student_model"):
        inject_descriptor_model_kwargs(
            recipe,
            model_path=teacher_dir,
            descriptor_name=str(hydra_cfg.descriptor),
            trust_remote_code=bool(recipe[model_key]["trust_remote_code"]),
            model_key=model_key,
        )
    _inject_canonical_data(recipe, hydra_cfg)
    distributed = recipe.setdefault("distributed", {})
    explicit_dp_size = distributed.get("dp_size")
    dp_size = _logical_dp_size(hydra_cfg, distributed)
    scheduler_dp_size = _int_or_default(
        explicit_dp_size
        if explicit_dp_size not in (None, "none", "None", "")
        else distributed.get("ep_size"),
        1,
    )
    global_microbatch_size = int(hydra_cfg.bypass.training.micro_batch_size)
    grad_accumulation_steps = int(
        hydra_cfg.bypass.training.grad_accumulation_steps
    )
    if global_microbatch_size % dp_size:
        raise ValueError(
            "bypass.training.micro_batch_size is a global microbatch and must be "
            f"divisible by dp_size; got {global_microbatch_size} and {dp_size}"
        )
    per_rank_microbatch_size = global_microbatch_size // dp_size
    recipe["step_scheduler"] = {
        **dict(recipe.get("step_scheduler", {})),
        # Puzzletron's micro_batch_size is global across data-parallel ranks,
        # while AutoModel's local_batch_size is per DP rank.  The scheduler's
        # global batch includes the explicitly managed accumulation window.
        # AutoModel includes EP in its setup-time data mesh when explicit DP is
        # disabled.  EP ranks still consume the same Puzzletron batch, so this
        # factor satisfies scheduler accounting without changing batch slicing.
        "global_batch_size": (
            per_rank_microbatch_size * scheduler_dp_size * grad_accumulation_steps
        ),
        "local_batch_size": per_rank_microbatch_size,
        "max_steps": int(hydra_cfg.bypass.training.max_steps),
    }
    optimizer_name = str(hydra_cfg.bypass.training.get("optimizer", "adamw")).lower()
    if optimizer_name not in {"adamw", "sgd"}:
        raise ValueError(f"unsupported bypass.training.optimizer={optimizer_name!r}")
    recipe["optimizer"] = {
        "_target_": "torch.optim.AdamW" if optimizer_name == "adamw" else "torch.optim.SGD",
        "lr": float(hydra_cfg.bypass.training.learning_rate),
        "weight_decay": float(hydra_cfg.bypass.training.weight_decay),
    }
    if recipe["optimizer"]["_target_"].endswith("AdamW"):
        recipe["optimizer"]["betas"] = [
            float(hydra_cfg.bypass.training.beta1),
            float(hydra_cfg.bypass.training.beta2),
        ]
    else:
        recipe["optimizer"]["momentum"] = float(
            hydra_cfg.bypass.training.get("momentum", 0.0)
        )

    checkpoint_dir = Path(hydra_cfg.bypass.experiment_dir) / "automodel_checkpoints"
    teacher_setup_dir = Path(hydra_cfg.bypass.experiment_dir) / ".automodel_teacher_setup"
    recipe["local_kd_checkpoint_dir"] = str(checkpoint_dir)
    recipe["checkpoint"] = {
        **dict(recipe.get("checkpoint", {})),
        # The base forward-only recipe builds the teacher first and otherwise
        # auto-restores LATEST from this directory. Keep it disabled and isolated
        # until LocalDistillationRecipe swaps the tracked model to the student.
        "enabled": False,
        "checkpoint_dir": str(teacher_setup_dir),
        "model_save_format": "safetensors",
        # Overfit probes publish metrics only. Avoid exporting a redundant
        # ~230 GiB HF checkpoint after every smoke probe.
        "save_consolidated": not bool(
            hydra_cfg.bypass.get("single_batch_overfit", False)
        ),
        # The teacher is built by the base forward-only recipe. Student resume is
        # loaded explicitly after the student becomes the tracked model.
        "restore_from": None,
    }
    recipe.setdefault("loss_fn", {
        "_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    })
    recipe.setdefault("packed_sequence", {"packed_sequence_size": 0})
    recipe.setdefault("dist_env", {"backend": "nccl"})
    if hydra_cfg.get("nccl_timeout_minutes", None) is not None:
        recipe["dist_env"]["timeout_minutes"] = int(hydra_cfg.nccl_timeout_minutes)

    pp_size = int(distributed.get("pp_size", 1) or 1)
    pipeline = distributed.get("pipeline")
    if pp_size > 1:
        pipeline = dict(pipeline or {})
        pipeline.setdefault("pp_schedule", "1f1b")
        pipeline.setdefault("scale_grads_in_schedule", False)
        pipeline.setdefault("round_virtual_stages_to_pp_multiple", "up")
        pipeline.setdefault("dtype", str(hydra_cfg.bypass.get("dtype", "bf16")))
        # Local KD executes each Puzzletron batch as one forward-only pipeline
        # microbatch. Its static stage metadata must therefore use the real
        # bypass microbatch size, not the scoring recipe's placeholder value.
        training_microbatch_size = global_microbatch_size
        pp_microbatch_size = training_microbatch_size // dp_size
        # NeMo validates a training schedule before the forward-only recipe
        # replaces it. Supply enough synthetic schedule batch slots for every
        # PP stage; the actual Puzzletron batch is still executed as one forward
        # microbatch by ActivationScoringRecipe.
        setup_local_batch_size = max(
            training_microbatch_size,
            pp_size * pp_microbatch_size,
        )
        recipe["step_scheduler"]["local_batch_size"] = setup_local_batch_size
        recipe["step_scheduler"]["global_batch_size"] = (
            setup_local_batch_size * scheduler_dp_size
        )
        pipeline["pp_microbatch_size"] = pp_microbatch_size
        # Local KD forwards the teacher once and replays student blocks
        # locally, so its PP schedule must have exactly one forward
        # microbatch.  Leaving the base scoring recipe's larger pp_batch_size
        # here makes each PP rank publish the same shard identity and drops all
        # but one stage during HF consolidation.
        pipeline["pp_batch_size"] = pp_microbatch_size
        cp_size = int(distributed.get("cp_size", 1) or 1)
        block_size = int(hydra_cfg.bypass.data.block_size)
        if block_size % cp_size:
            raise ValueError(
                f"bypass.data.block_size={block_size} must be divisible by cp_size={cp_size}"
            )
        pipeline["pp_seq_len"] = block_size // cp_size
        distributed["pipeline"] = pipeline

    recipe = inject_descriptor_pipeline_config(
        recipe,
        model_path=teacher_dir,
        descriptor_name=str(hydra_cfg.descriptor),
        trust_remote_code=bool(teacher_model["trust_remote_code"]),
    )
    return OmegaConf.to_container(OmegaConf.create(recipe), resolve=True)
