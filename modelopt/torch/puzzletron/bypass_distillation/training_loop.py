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

"""Bypass distillation training loop for per-block knowledge distillation.

This module implements the blockwise local distillation (BLD) stage of the PUZZLE framework.
It trains alternative transformer block configurations using per-block knowledge distillation
from a teacher model, producing a library of "puzzle pieces" with different efficiency/performance
trade-offs.
"""

import logging
import math
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
import torch.distributed
import transformers
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

import datasets
import datasets.utils.logging
import modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory as stitched_model_factory_module
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch.puzzletron.sewing_kit import InputArgs, StitchedModule
from modelopt.torch.puzzletron.sewing_kit.utils import fake_tensor
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config
from modelopt.torch.puzzletron.tools.logger import aprint, mprint
from modelopt.torch.puzzletron.tools.robust_json import json_load
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import load_and_shard_model
from modelopt.torch.puzzletron.utils.parsing import format_global_config, format_stitched_losses

from .bypass_checkpoint_utils import find_latest_run_dir, load_local_state, save_bypass_checkpoint
from .bypass_utils import get_distributed_modules_ownership, set_experiment_dir, set_experiment_id
from .data_classes import GlobalRank, IterNum, IterStatistics, LocalTrainingStats, TimeToSaveSignal
from .stitched_model_factory import StitchedModuleDescriptor, StitchedModulesProcessOwnership

time_start = time.time()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def launch_bypass_distillation(hydra_cfg: DictConfig) -> None:
    """Top-level entry point for bypass distillation stage.

    Runs sewing-kit pipeline-parallel per-block knowledge distillation.

    Supports multiple bypass configurations via ``bypass.configs`` list.
    Each entry overrides ``bypass.model.model_config_overrides`` and optionally
    ``bypass.model_factory.keys_to_learn``, then runs a full bypass training.

    If ``bypass.configs`` is absent or empty, runs a single bypass training
    with the settings already in ``bypass``.

    Args:
        hydra_cfg: The full Hydra configuration with a 'bypass' section.
    """
    configs_list = hydra_cfg.bypass.get("configs", None)

    if not configs_list:
        # Single config mode — run once with whatever is in bypass already
        mprint("Starting bypass distillation (single config)")
        run_bypassed_training(hydra_cfg)
        mprint("Bypass distillation completed")
        return

    mprint(f"Starting bypass distillation sweep ({len(configs_list)} configs)")
    for i, override in enumerate(configs_list):
        mprint(f"Bypass config {i + 1}/{len(configs_list)}: {override}")

        # Apply overrides for this run
        if "model_config_overrides" in override:
            hydra_cfg.bypass.model.model_config_overrides = override.model_config_overrides
        if "keys_to_learn" in override:
            hydra_cfg.bypass.model_factory.keys_to_learn = override.keys_to_learn

        # Reset per-run state so each config starts fresh
        hydra_cfg.bypass.experiment_id = None
        hydra_cfg.bypass.iter_num = 1
        hydra_cfg.bypass.step_num = 1
        hydra_cfg.bypass.token_count = 0
        hydra_cfg.bypass.best_val_loss = 1e9
        hydra_cfg.bypass.training.clipping_count = 0

        run_bypassed_training(hydra_cfg)
        mprint(f"Bypass config {i + 1}/{len(configs_list)} completed")

    mprint("Bypass distillation sweep completed")


def train(
    cfg: DictConfig,
    descriptor: ModelDescriptor,
    student_model: torch.nn.Module,
    student_stitched_model: StitchedModule,
    teacher_stitched_model: StitchedModule,
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    stitched_modules_process_ownership: StitchedModulesProcessOwnership,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    student_model_config: PretrainedConfig,
    skip_first_batches: int = 0,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> None:
    """Inner training loop for bypass distillation."""
    device = torch.device(f"cuda:{dist.local_rank()}")

    dist.barrier()

    time_last_save = time_start
    iter_t0 = time.time()

    resumed_iter_num = cfg.bypass.iter_num
    mprint(f"resumed_iter_num: {resumed_iter_num}")

    # Number of total stitched modules
    global_stitched_modules_count = len(stitched_modules_process_ownership)
    # Number of stitched modules per process
    num_stitched_modules_per_process = [
        sum(1 for x in stitched_modules_process_ownership if x == owner_rank)
        for owner_rank in range(dist.size())
    ]
    # Indices of stitched modules owned by the current process
    owned_stitched_module_indices = [
        i for i, owner in enumerate(stitched_modules_process_ownership) if owner == dist.rank()
    ]
    mprint(f"{global_stitched_modules_count=}")
    mprint(f"{num_stitched_modules_per_process=}")
    dist.barrier()

    if dist.is_master():
        # {iter_num: {stitched_module_name: loss}}
        stitched_losses_history = dict[IterNum, dict[str, float]]()
    else:
        stitched_losses_history = None

    # Save checkpoint before training starts
    if cfg.bypass.save_checkpoint_before_training and not cfg.bypass.disable_checkpoint_save:
        subdir_name = f"start-iter-{cfg.bypass.iter_num:06d}-ckpt"
        save_bypass_checkpoint(
            cfg=cfg,
            descriptor=descriptor,
            model=student_model,
            stitched_module_descriptors=stitched_module_descriptors,
            checkpoint_dir=Path(cfg.bypass.experiment_dir) / subdir_name,
            reference_checkpoint_dir=cfg.teacher_dir,
        )

    # Track statistics for each iteration
    iter_stats_history: dict[IterNum, IterStatistics] = {}

    # Create fake input ids for the teacher model
    fake_input_ids = fake_tensor(
        torch.ones(
            size=(cfg.bypass.training.micro_batch_size, cfg.bypass.data.block_size),
            dtype=torch.long,
            device=device,
        )
    )

    # Get pipeline neighbor ranks
    min_owned_index = min(owned_stitched_module_indices)
    max_owned_index = max(owned_stitched_module_indices)
    prev_rank: Optional[int] = (
        None if min_owned_index - 1 < 0 else stitched_modules_process_ownership[min_owned_index - 1]
    )
    next_rank: Optional[int] = (
        None
        if max_owned_index + 1 >= global_stitched_modules_count
        else stitched_modules_process_ownership[max_owned_index + 1]
    )

    torch.cuda.synchronize()

    mprint(
        f"Grad scaling status: {'enabled' if cfg.bypass.training.use_grad_scaling else 'disabled'}"
    )

    train_iterator = iter(train_dataloader)

    mprint("Waiting for everyone before training starts")
    dist.barrier()

    step_to_save = None
    # Track best loss value for each block
    best_losses_by_name = dict[str, float]()
    best_steps_by_name = dict[str, int]()
    # Buffer variables
    input_ids = torch.zeros(1, 1, dtype=torch.int64)

    aprint(
        f"previous rank: {str(prev_rank):<5} next rank: {str(next_rank):<5} {owned_stitched_module_indices=}"
    )

    # Train loop start
    while True:
        time_now = time.time()
        # Check if we've reached the maximum number of steps
        if cfg.bypass.step_num >= cfg.bypass.training.max_steps:
            if (
                cfg.bypass.model.model_overrides.save_checkpoint_when_done
                and not cfg.bypass.disable_checkpoint_save
            ):
                mprint("Saving final checkpoint before training completion")
                subdir_name = f"final-iter-{cfg.bypass.iter_num:06d}-ckpt"
                save_bypass_checkpoint(
                    cfg=cfg,
                    descriptor=descriptor,
                    model=student_model,
                    stitched_module_descriptors=stitched_module_descriptors,
                    checkpoint_dir=Path(cfg.bypass.experiment_dir) / subdir_name,
                    reference_checkpoint_dir=cfg.teacher_dir,
                )

                if cfg.bypass.model.model_overrides.delete_old_checkpoints and dist.is_master():
                    existing_ckpt_paths = list(Path(cfg.bypass.experiment_dir).glob("iter-*"))
                    for old_ckpt_path in existing_ckpt_paths:
                        if old_ckpt_path.name != subdir_name:
                            shutil.rmtree(str(old_ckpt_path))
            break

        is_accumulating = cfg.bypass.iter_num % cfg.bypass.training.grad_accumulation_steps != 0
        # Determine and set the learning rate for this iteration
        lr = (
            _get_lr(cfg, cfg.bypass.step_num)
            if cfg.bypass.training.decay_lr
            else cfg.bypass.training.learning_rate
        )
        for stitched_module_descriptor in stitched_module_descriptors.values():
            optimizer = stitched_module_descriptor.optimizer
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        if dist.is_master():
            train_data = next(train_iterator)
            input_ids = train_data["input_ids"]
            input_ids = input_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            teacher_input_ids = input_ids if prev_rank is None else fake_input_ids
            teacher_output = teacher_stitched_model({}, {}, teacher_input_ids)

            input_overrides = teacher_output.captured_inputs
            output_overrides = teacher_output.captured_outputs

            del teacher_output

        input_overrides["teacher_inputs"] = InputArgs(fake_input_ids)

        iter_stitched_module_losses: dict[str, float] = {}

        for local_stitched_module_index, (
            stitched_module_name,
            stitched_module_descriptor,
        ) in enumerate(stitched_module_descriptors.items()):
            stitched_module = stitched_module_descriptor.stitched_module
            optimizer = stitched_module_descriptor.optimizer
            grad_scaler = stitched_module_descriptor.grad_scaler

            if optimizer is not None:
                assert grad_scaler is not None

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    stitched_module_output = stitched_module(
                        input_overrides=input_overrides,
                        output_overrides=output_overrides,
                    )
                stitched_module_loss = stitched_module_output.captured_outputs["loss"]
                del stitched_module_output
                grad_scaler.scale(stitched_module_loss).backward()
            else:
                stitched_module_loss = torch.full([1], fill_value=torch.nan, dtype=torch.float32)

            iter_stitched_module_losses[stitched_module_name] = stitched_module_loss.to(
                "cpu"
            ).item()

            del stitched_module_loss

            if not is_accumulating:
                if optimizer is not None:
                    grad_clip = cfg.bypass.training.grad_clip
                    if grad_clip is not None:
                        if cfg.bypass.training.grad_clip_type == "norm":
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                parameters=stitched_module.parameters(),
                                max_norm=grad_clip,
                            )
                            if grad_norm > grad_clip:
                                cfg.bypass.training.clipping_count += 1
                        elif cfg.bypass.training.grad_clip_type == "value":
                            max_abs_grad_per_param = [
                                p.grad.abs().max().item()
                                for p in stitched_module.parameters()
                                if p.grad is not None
                            ]
                            max_abs_grad = (
                                max(max_abs_grad_per_param)
                                if len(max_abs_grad_per_param) > 0
                                else 0.0
                            )
                            if max_abs_grad > grad_clip:
                                cfg.bypass.training.clipping_count += 1
                                torch.nn.utils.clip_grad_value_(
                                    parameters=stitched_module.parameters(),
                                    clip_value=grad_clip,
                                )
                        else:
                            raise RuntimeError(f"Invalid {cfg.bypass.training.grad_clip_type}")

                    assert grad_scaler is not None
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        # Collect losses from all ranks using all_gather_object
        local_training_stats = LocalTrainingStats(
            iter_num=cfg.bypass.iter_num,
            stitched_module_losses=iter_stitched_module_losses,
        )
        all_training_stats = [None] * dist.size()
        torch.distributed.all_gather_object(all_training_stats, local_training_stats)

        if dist.is_master():
            if cfg.bypass.iter_num == resumed_iter_num:
                mprint(f"Starting from iter {cfg.bypass.iter_num}")

            # Merge all stats into the losses history
            assert stitched_losses_history is not None
            merged_losses: dict[str, float] = {}
            for stats in all_training_stats:
                if stats is not None:
                    merged_losses.update(stats.stitched_module_losses)
            stitched_losses_history[cfg.bypass.iter_num] = merged_losses

        cfg.bypass.token_count += cfg.bypass.training.tokens_per_iter
        iter_t1 = time.time()
        iter_duration = iter_t1 - iter_t0
        iter_stats_history[cfg.bypass.iter_num] = IterStatistics(
            token_count=cfg.bypass.token_count,
            iter_duration=iter_duration,
            step_num=cfg.bypass.step_num,
            lr=lr,
            clipping_count=cfg.bypass.training.clipping_count,
        )
        iter_t0 = iter_t1

        # Time-based save signal (broadcast from master)
        save_signal = [step_to_save]
        if dist.is_master():
            if cfg.bypass.model.model_overrides.save_interval_seconds is not None:
                time_now = time.time()
                if (
                    time_now - time_last_save
                    >= cfg.bypass.model.model_overrides.save_interval_seconds
                ):
                    mprint(
                        f"Time to save! {cfg.bypass.model.model_overrides.save_interval_seconds=}, "
                        f"{time_last_save=}, {time_now=}"
                    )
                    step_to_save = cfg.bypass.step_num + 5
                    save_signal = [step_to_save]
                    time_last_save = time_now

        torch.distributed.broadcast_object_list(save_signal, src=0)
        step_to_save = save_signal[0]

        # Logging
        if dist.is_master():
            assert stitched_losses_history is not None
            while len(stitched_losses_history) >= cfg.bypass.training.log_interval:
                lowest_iter = next(iter(stitched_losses_history.keys()))

                log_chunk = {
                    it: losses
                    for it, losses in stitched_losses_history.items()
                    if it - lowest_iter < cfg.bypass.training.log_interval
                }
                if len(log_chunk) < cfg.bypass.training.log_interval:
                    break

                highest_iter = list(log_chunk.keys())[-1]
                highest_iter_stats = iter_stats_history[highest_iter]

                losses_by_name = defaultdict[str, list[float]](list)
                for losses in log_chunk.values():
                    for name, loss in losses.items():
                        losses_by_name[name].append(loss)

                losses_by_name_avg = {name: mean(losses) for name, losses in losses_by_name.items()}

                # Update best losses tracking
                for name, current_loss in losses_by_name_avg.items():
                    if name not in best_losses_by_name or current_loss < best_losses_by_name[name]:
                        best_losses_by_name[name] = current_loss
                        best_steps_by_name[name] = highest_iter

                chunk_iter_durations = [
                    iter_stats_history[it].iter_duration for it in log_chunk.keys()
                ]
                avg_chunk_iter_duration = mean(chunk_iter_durations)
                avg_token_speed = cfg.bypass.training.tokens_per_iter / avg_chunk_iter_duration
                mprint(
                    # `highest_iter` is in micro-batch units (iter_num); compare against
                    # max_iters (= max_steps × grad_accumulation_steps), not max_steps,
                    # so the progress fraction is in consistent units.
                    f"iter {highest_iter}/{cfg.bypass.training.max_iters:,}"
                    f" (step {highest_iter_stats.step_num}/{cfg.bypass.training.max_steps:,}):"
                    f" avg_iter_time={avg_chunk_iter_duration * 1000:.2f}ms"
                    f" avg_token_speed={avg_token_speed:,.0f}[tok/s]"
                )
                mprint(
                    format_stitched_losses(
                        losses_dict=losses_by_name_avg,
                        best_steps_dict=best_steps_by_name,
                        best_values_dict=best_losses_by_name,
                        step_number=highest_iter,
                        title="Stitched Module Losses",
                    )
                )

                if cfg.bypass.wandb_log:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "iter": highest_iter,
                                "step": highest_iter_stats.step_num,
                                "token_count": highest_iter_stats.token_count,
                                "token_speed": avg_token_speed,
                                "lr": highest_iter_stats.lr,
                                "grad_clipping": highest_iter_stats.clipping_count,
                            },
                            step=highest_iter,
                        )
                    except ImportError:
                        pass

                for it in log_chunk.keys():
                    del iter_stats_history[it]
                    del stitched_losses_history[it]

        # Validation
        if (
            not is_accumulating
            and (cfg.bypass.step_num % cfg.bypass.training.eval_interval) == 0
            and val_dataloader is not None
        ):
            from modelopt.torch.puzzletron.utils.validate_runtime_pipeline import (
                calculate_losses_pipeline,
            )

            losses, _ = calculate_losses_pipeline(
                stitched_model=student_stitched_model,
                dataloader=val_dataloader,
                descriptor=descriptor,
            )

            val_loss = float("inf")
            if losses is not None and "lm_loss" in losses:
                val_loss = losses["lm_loss"]["avg"]
                mprint(f"Validation loss at iter {cfg.bypass.iter_num}: {val_loss:.4f}")

            # Broadcast val_loss so all ranks agree on checkpoint decisions
            val_loss_tensor = torch.tensor([val_loss], device=device)
            torch.distributed.broadcast(val_loss_tensor, src=dist.size() - 1)
            val_loss = val_loss_tensor.item()

            if val_loss < cfg.bypass.best_val_loss:
                cfg.bypass.best_val_loss = val_loss
                if not cfg.bypass.disable_checkpoint_save and cfg.bypass.save_best_ckpt:
                    subdir_name = f"best-iter-{cfg.bypass.iter_num:06d}-ckpt"
                    save_bypass_checkpoint(
                        cfg=cfg,
                        descriptor=descriptor,
                        model=student_model,
                        stitched_module_descriptors=stitched_module_descriptors,
                        checkpoint_dir=Path(cfg.bypass.experiment_dir) / subdir_name,
                        reference_checkpoint_dir=cfg.teacher_dir,
                    )
                    # Keep only the most recent best-checkpoint when delete_old_checkpoints
                    # is enabled — mirrors the iter-*/final-iter-* cleanup elsewhere so a
                    # long run with many validation improvements doesn't fill the disk.
                    if cfg.bypass.model.model_overrides.delete_old_checkpoints and dist.is_master():
                        for old_ckpt_path in Path(cfg.bypass.experiment_dir).glob("best-iter-*"):
                            if old_ckpt_path.name != subdir_name:
                                shutil.rmtree(str(old_ckpt_path))
                    if cfg.bypass.kill_after_first_save:
                        raise RuntimeError("Done saving checkpoint, kill_after_first_save=True")

        # Checkpoint saving (step-based or time-based)
        if not is_accumulating and (
            (cfg.bypass.step_num % cfg.bypass.model.model_overrides.save_interval) == 0
            or step_to_save == cfg.bypass.step_num
            or (
                cfg.bypass.model.model_overrides.save_checkpoint_when_done
                and cfg.bypass.step_num >= cfg.bypass.training.max_steps
            )
        ):
            if not cfg.bypass.disable_checkpoint_save:
                if (cfg.bypass.step_num % cfg.bypass.model.model_overrides.save_interval) == 0:
                    mprint("Saving step-interval checkpoint")
                elif step_to_save == cfg.bypass.step_num:
                    mprint("Saving time-based checkpoint")
                elif (
                    cfg.bypass.model.model_overrides.save_checkpoint_when_done
                    and cfg.bypass.step_num >= cfg.bypass.training.max_steps
                ):
                    mprint("Saving final checkpoint")

                subdir_name = f"iter-{cfg.bypass.iter_num:06d}-ckpt"
                save_bypass_checkpoint(
                    cfg=cfg,
                    descriptor=descriptor,
                    model=student_model,
                    stitched_module_descriptors=stitched_module_descriptors,
                    checkpoint_dir=Path(cfg.bypass.experiment_dir) / subdir_name,
                    reference_checkpoint_dir=cfg.teacher_dir,
                )

                if cfg.bypass.kill_after_first_save:
                    dist.barrier()
                    raise RuntimeError("Done saving checkpoint, kill_after_first_save=True")

                if cfg.bypass.model.model_overrides.delete_old_checkpoints and dist.is_master():
                    existing_ckpt_paths = list(Path(cfg.bypass.experiment_dir).glob("iter-*"))
                    for old_ckpt_path in existing_ckpt_paths:
                        if old_ckpt_path.name != subdir_name:
                            shutil.rmtree(str(old_ckpt_path))

        cfg.bypass.iter_num += 1
        if not is_accumulating:
            cfg.bypass.step_num += 1

    mprint("Finished successfully!")


# Learning rate decay scheduler (cosine with warmup)
def _get_lr(cfg: DictConfig, step: int) -> float:
    warmup_steps = cfg.bypass.training.warmup_steps
    lr_decay_steps = cfg.bypass.training.lr_decay_steps
    # Degenerate budget (e.g. tiny `training_tokens` in tests): no room for cosine decay.
    # Skip warmup/decay entirely and return base LR — avoids ZeroDivisionError on
    # `lr_decay_steps - warmup_steps` and `step / warmup_steps`.
    if lr_decay_steps <= warmup_steps:
        return cfg.bypass.training.learning_rate

    # 1) linear warmup for warmup_steps steps
    if step <= warmup_steps:
        lr = cfg.bypass.training.learning_rate * step / warmup_steps
    # 2) if step > lr_decay_steps, return min learning rate
    elif step > lr_decay_steps:
        lr = cfg.bypass.training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    else:
        decay_ratio = (step - warmup_steps - 1) / (lr_decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        lr = cfg.bypass.training.min_lr + coeff * (
            cfg.bypass.training.learning_rate - cfg.bypass.training.min_lr
        )

    return lr


def run_bypassed_training(cfg: DictConfig):
    """Setup and orchestrate bypass distillation training."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARN
    )

    # Suppress debug messages from HuggingFace libraries
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    device = torch.device(f"cuda:{dist.local_rank()}")

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    trust_remote_code = descriptor.requires_trust_remote_code()
    teacher_model_config = load_model_config(cfg.teacher_dir, trust_remote_code=trust_remote_code)

    try:
        mprint("Waiting for distributed setup...")
        dist.barrier()

        if cfg.bypass.disable_initial_validate:
            cfg.bypass.validate_teacher_model = False
            cfg.bypass.validate_student_model = False

        if cfg.bypass.teacher_model_load_on_cpu:
            assert not cfg.bypass.validate_teacher_model, (
                "Teacher model validation is too slow on CPU"
            )

        num_hidden_layers = descriptor.get_language_model_config(
            teacher_model_config
        ).num_hidden_layers

        model_blocks_process_ownership = get_distributed_modules_ownership(
            module_count=num_hidden_layers,
            world_size=dist.size(),
        )

        owned_block_indexes = set(
            block_index
            for block_index, owner_rank in enumerate(model_blocks_process_ownership)
            if owner_rank == dist.rank()
        )

        cfg.teacher_dir = str(Path(cfg.teacher_dir).expanduser())
        teacher_model_config = load_model_config(
            cfg.teacher_dir,
            trust_remote_code=trust_remote_code,
        )
        # Disable KV cache during bypass forward passes. Set the attribute directly rather
        # than passing it as an AutoConfig override — some custom configs (GptOss, Qwen3-VL, etc.)
        # don't accept it as a known kwarg and would raise via the strict unused-kwargs check.
        if hasattr(teacher_model_config, "use_cache"):
            teacher_model_config.use_cache = False
        if hasattr(teacher_model_config, "text_config") and hasattr(
            teacher_model_config.text_config, "use_cache"
        ):
            teacher_model_config.text_config.use_cache = False

        student_model = None
        if cfg.bypass.init_checkpoint_path is not None:
            mprint(f"Loading student model from {cfg.bypass.init_checkpoint_path}")
            student_model = load_and_shard_model(
                descriptor=descriptor,
                checkpoint_path=cfg.bypass.init_checkpoint_path,
                owned_block_indexes=owned_block_indexes,
            )

        cfg.bypass.training.min_lr = (
            cfg.bypass.training.learning_rate * cfg.bypass.training.min_lr_factor
        )
        cfg.bypass.training.batch_size_per_iter = cfg.bypass.training.micro_batch_size
        cfg.bypass.training.tokens_per_iter = (
            cfg.bypass.data.block_size * cfg.bypass.training.batch_size_per_iter
        )
        cfg.bypass.training.max_steps = math.ceil(
            cfg.bypass.training.training_tokens / cfg.bypass.training.tokens_per_iter
        )
        cfg.bypass.training.max_iters = (
            cfg.bypass.training.max_steps * cfg.bypass.training.grad_accumulation_steps
        )
        cfg.bypass.training.max_token_count = (
            cfg.bypass.training.max_iters * cfg.bypass.training.tokens_per_iter
        )
        cfg.bypass.training.lr_decay_steps = cfg.bypass.training.max_steps

        if cfg.bypass.training.val_micro_batch_size is None:
            cfg.bypass.training.val_micro_batch_size = cfg.bypass.training.micro_batch_size

        if cfg.bypass.training.warmup_steps is None:
            cfg.bypass.training.warmup_steps = 0

        mprint(f"\n{format_global_config(cfg.bypass, 'Bypass Configurations')}")
        mprint(f"Max token count:  {cfg.bypass.training.max_token_count:,}")

        seed = cfg.bypass.seed
        torch.manual_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.teacher_dir,
            trust_remote_code=True,
            token=True,
        )

        assert teacher_model_config is not None

        mprint(f"Load and shard model with: {owned_block_indexes=}, {cfg.teacher_dir=}")
        teacher_model = load_and_shard_model(
            descriptor=descriptor,
            checkpoint_path=cfg.teacher_dir,
            owned_block_indexes=owned_block_indexes,
            model_config=teacher_model_config,
        )

        teacher_model.requires_grad_(False)

        # Create dataloaders
        from modelopt.torch.puzzletron.utils.data.dataloaders import (
            create_train_dataloader,
            create_validation_dataloader,
            load_from_disk_fn,
            load_streaming_fn,
        )

        if cfg.bypass.data.eval_samples_per_process is not None:
            max_eval_samples = cfg.bypass.data.eval_samples_per_process * dist.size()
        else:
            max_eval_samples = cfg.bypass.data.max_eval_samples

        load_dataset_fn = (
            load_streaming_fn if not cfg.bypass.data.load_from_disk else load_from_disk_fn
        )

        train_dataloader = create_train_dataloader(
            seed=seed,
            tokenizer=tokenizer,
            block_size=cfg.bypass.data.block_size,
            dataset_path=cfg.dataset_path,
            content_field=cfg.bypass.data.data_column,
            fim_rate=cfg.bypass.data.fim_rate,
            fim_spm_rate=cfg.bypass.data.fim_spm_rate,
            micro_batch_size=cfg.bypass.training.micro_batch_size,
            load_dataset_fn=load_dataset_fn,
            keep_in_memory=cfg.bypass.data.keep_in_memory,
            source_datasets_to_discard=cfg.bypass.get("source_datasets_to_discard", tuple()),
            bos_rate=cfg.bypass.data.bos_rate,
            shuffle_seed=cfg.bypass.data.shuffle_train_data_seed,
        )

        val_dataloader = None
        if not cfg.bypass.disable_validation:
            val_dataloader = create_validation_dataloader(
                accelerator=None,
                seed=seed,
                tokenizer=tokenizer,
                block_size=cfg.bypass.data.block_size,
                dataset=cfg.dataset_path,
                content_field=cfg.bypass.data.data_column,
                fim_rate=cfg.bypass.data.fim_rate,
                fim_spm_rate=cfg.bypass.data.fim_spm_rate,
                micro_batch_size=cfg.bypass.training.val_micro_batch_size,
                eval_samples=max_eval_samples,
                load_dataset_fn=load_dataset_fn,
                dataset_name=cfg.bypass.data.val_dataset_name,
                keep_in_memory=cfg.bypass.data.keep_in_memory,
                source_datasets_to_discard=cfg.bypass.get("source_datasets_to_discard", tuple()),
                bos_rate=cfg.bypass.data.bos_rate,
            )

        # Set ID from experiment configuration
        set_experiment_id(cfg)
        # Set directory for experiment ID
        set_experiment_dir(cfg)

        dist.barrier()

        with torch.device(device):
            stitched_model_factory_fn = getattr(
                stitched_model_factory_module, cfg.bypass.model_factory.factory
            )
            (
                student_model,
                teacher_stitched_model,
                teacher_val_stitched_module,
                student_val_stitched_model,
                stitched_module_descriptors,
                student_model_config,
            ) = stitched_model_factory_fn(
                teacher_model=teacher_model,
                descriptor=descriptor,
                cfg=cfg.bypass,
                model_blocks_process_ownership=model_blocks_process_ownership,
                student_model=student_model,
            )

        # Check whether to resume from checkpoint
        resume_checkpoint_path = None
        if cfg.bypass.resume_checkpoint_path is not None:
            resume_checkpoint_path = cfg.bypass.resume_checkpoint_path
        elif cfg.bypass.find_last_ckpt_for_resume:
            _ckpt_dir = find_latest_run_dir(run_parent_dir=cfg.bypass.experiment_dir)
            if _ckpt_dir is None:
                mprint("Couldn't find any run dir for resume, assuming this is the first job")
            else:
                mprint(
                    f"`cfg.bypass.find_last_ckpt_for_resume` is True. "
                    f"Auto-found a checkpoint to resume: `{_ckpt_dir}`"
                )
                resume_checkpoint_path = _ckpt_dir

        if resume_checkpoint_path:
            load_local_state(
                stitched_module_descriptors=stitched_module_descriptors,
                checkpoint_path=resume_checkpoint_path,
            )

            # Load resume ckpt bypass configs and extract resume iter_num
            resume_cfg = DictConfig(json_load(Path(resume_checkpoint_path) / "args.json"))

            # Resume stats
            cfg.bypass.iter_num = resume_cfg.iter_num
            cfg.bypass.token_count = resume_cfg.token_count
            cfg.bypass.step_num = resume_cfg.step_num
            cfg.bypass.best_val_loss = resume_cfg.best_val_loss
            cfg.bypass.training.clipping_count = resume_cfg.training.clipping_count
            mprint(f"Resume from iter_num: {cfg.bypass.iter_num}")

            # Only copy wandb.run_id if it exists in resume config
            if hasattr(resume_cfg, "wandb") and hasattr(resume_cfg.wandb, "run_id"):
                cfg.bypass.wandb.run_id = resume_cfg.wandb.run_id

            cfg.bypass.save_checkpoint_before_training = False
            cfg.bypass.validate_teacher_model = False
            cfg.bypass.validate_student_model = False

            cfg.bypass.resume_checkpoint_path = resume_checkpoint_path

        # Initialize Weights and Biases
        if cfg.bypass.wandb_log:
            try:
                import wandb

                wandb.init(
                    project=cfg.bypass.wandb.project,
                    entity=cfg.bypass.wandb.entity,
                    config=dict(cfg.bypass),
                )
            except ImportError:
                mprint("wandb not installed, disabling wandb logging")
                cfg.bypass.wandb_log = False
        else:
            mprint("Weights & Biases logging disabled (wandb_log=False)")

        if cfg.bypass.validate_teacher_model and val_dataloader is not None:
            from modelopt.torch.puzzletron.utils.validate_runtime_pipeline import (
                calculate_losses_pipeline,
            )

            mprint("Evaluating teacher model:")
            losses, _ = calculate_losses_pipeline(
                stitched_model=teacher_val_stitched_module,
                dataloader=val_dataloader,
                descriptor=descriptor,
            )
            if losses is not None:
                mprint(f"Teacher validation losses: {losses}")
            mprint("Evaluated teacher model")

        torch.cuda.empty_cache()
        dist.barrier()

        parameter_count = sum(p.numel() for p in student_model.parameters())
        aprint(f"Model parameter count: {parameter_count:,}")
        cfg.bypass.parameter_count = parameter_count

        dist.barrier()
        mprint("Performing dummy runs on stitched modules:")
        torch.cuda.synchronize()
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            torch.device(device),
        ):
            input_ids = torch.ones(
                (cfg.bypass.training.micro_batch_size, cfg.bypass.data.block_size),
                dtype=torch.long,
            )
            dummy_fake_input_ids = fake_tensor(input_ids)
            mprint(f"Dummy runs on stitched modules with shape: {dummy_fake_input_ids.shape=}")
            teacher_output = teacher_stitched_model({}, {}, input_ids)
            for stitched_module_descriptor in stitched_module_descriptors.values():
                stitched_module = stitched_module_descriptor.stitched_module
                stitched_module(
                    input_overrides={
                        **teacher_output.captured_inputs,
                        "teacher_inputs": InputArgs(dummy_fake_input_ids),
                    },
                    output_overrides=teacher_output.captured_outputs,
                )
                for name, param in stitched_module.named_parameters(recurse=True):
                    if "iter_num" in name:
                        param.data = torch.zeros_like(param.data)
                    del name, param
            del input_ids, dummy_fake_input_ids, teacher_output
        torch.cuda.synchronize()
        dist.barrier()

        del teacher_model

        if cfg.bypass.validate_student_model and val_dataloader is not None:
            from modelopt.torch.puzzletron.utils.validate_runtime_pipeline import (
                calculate_losses_pipeline,
            )

            mprint("Validating model before training:")
            losses, _ = calculate_losses_pipeline(
                stitched_model=student_val_stitched_model,
                dataloader=val_dataloader,
                descriptor=descriptor,
            )
            if losses is not None:
                mprint(f"Student validation losses: {losses}")

        dist.barrier()
        torch.cuda.empty_cache()
        dist.barrier()

        train(
            cfg=cfg,
            descriptor=descriptor,
            student_model=student_model,
            student_stitched_model=student_val_stitched_model,
            teacher_stitched_model=teacher_stitched_model,
            stitched_module_descriptors=stitched_module_descriptors,
            stitched_modules_process_ownership=model_blocks_process_ownership,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            student_model_config=student_model_config,
            skip_first_batches=cfg.bypass.training.skip_first_batches,
            tokenizer=tokenizer,
        )

        aprint("Finished training successfully!")
        dist.barrier()

    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        if isinstance(e, SystemExit):
            raise e
        else:
            sys.exit(1)

    dist.barrier()
    if dist.is_master():
        mprint("Realizing bypass checkpoints")
        realize_bypass_checkpoints(cfg)


def realize_bypass_checkpoints(cfg: DictConfig):
    """Create symlinks from bypass checkpoint directories to the ckpts directory."""
    checkpoint_dir = Path(cfg.bypass.experiment_dir) / "latest"
    if not checkpoint_dir.exists():
        mprint(f"Could not find checkpoint directory: {checkpoint_dir}")
        return

    ckpts_dir = Path(cfg.puzzle_dir) / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    symlink_name = ckpts_dir / cfg.bypass.experiment_id
    if symlink_name.exists() or symlink_name.is_symlink():
        symlink_name.unlink()

    symlink_name.symlink_to(checkpoint_dir, target_is_directory=True)
    mprint(f"Created symlink: {symlink_name} -> {checkpoint_dir}")
