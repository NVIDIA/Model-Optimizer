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
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Optional

import datasets
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

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
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import load_and_shard_model
from modelopt.torch.puzzletron.utils.parsing import format_global_config, format_stitched_losses
from modelopt.torch.utils.logging import print_rank_0
from modelopt.torch.utils.robust_json import json_load

from .bypass_checkpoint_utils import find_latest_run_dir, load_local_state, save_bypass_checkpoint
from .bypass_utils import (
    bypass_run_is_complete,
    get_distributed_modules_ownership,
    get_pipeline_ownership_context,
    load_bypass_state,
    mark_bypass_run_completed,
    set_experiment_dir,
    set_experiment_id,
)
from .data_classes import GlobalRank, IterNum, IterStatistics, TimeToSaveSignal
from .stitched_model_factory import StitchedModuleDescriptor, StitchedModulesProcessOwnership

__all__ = [
    "GlobalRank",
    "IterNum",
    "IterStatistics",
    "StitchedModuleDescriptor",
    "StitchedModulesProcessOwnership",
    "TimeToSaveSignal",
    "bypass_run_is_complete",
    "find_latest_run_dir",
    "get_distributed_modules_ownership",
    "get_pipeline_ownership_context",
    "launch_bypass_distillation",
    "load_bypass_state",
    "load_local_state",
    "mark_bypass_run_completed",
    "realize_bypass_checkpoints",
    "run_bypassed_training",
    "save_bypass_checkpoint",
    "set_experiment_dir",
    "set_experiment_id",
    "train",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _autocast_context(descriptor: ModelDescriptor):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if descriptor.uses_autocast()
        else nullcontext()
    )


def _resolve_trust_remote_code(cfg: DictConfig, descriptor: ModelDescriptor) -> bool:
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    if descriptor.requires_trust_remote_code() and not trust_remote_code:
        descriptor_name = getattr(descriptor, "__name__", descriptor.__class__.__name__)
        mprint(
            f"WARNING: descriptor {descriptor_name} usually requires trust_remote_code=True, "
            "but cfg.trust_remote_code is false; loading will proceed without executing "
            "custom checkpoint code."
        )
    return trust_remote_code


def _get_resume_state_path(cfg: DictConfig, resume_checkpoint_path: Optional[str]) -> Optional[str]:
    if cfg.bypass.init_checkpoint_path is not None:
        if resume_checkpoint_path is not None:
            mprint(
                f"Ignoring resume checkpoint state from {resume_checkpoint_path} because "
                f"bypass.init_checkpoint_path={cfg.bypass.init_checkpoint_path} is set"
            )
        return None
    return resume_checkpoint_path


def _get_resume_skip_first_batches(saved_skip: int, resume_iter_num: int) -> int:
    return saved_skip + max(0, resume_iter_num)


def _finalize_bypass_run(cfg: DictConfig) -> None:
    """Realize and mark a completed bypass run when a checkpoint exists."""
    if cfg.bypass.get("disable_checkpoint_save", False):
        mprint(
            "Bypass checkpoint saving is disabled; skipping checkpoint realization "
            "and completion marker"
        )
        return

    if not dist.is_master():
        return

    mprint("Realizing bypass checkpoints")
    try:
        realized_checkpoint, ckpts_symlink = realize_bypass_checkpoints(cfg)
    except FileNotFoundError as err:
        mprint(f"{err}; skipping bypass completion marker")
        return
    mark_bypass_run_completed(cfg, realized_checkpoint, ckpts_symlink)


def _clip_stitched_module_grads(
    stitched_module: StitchedModule, grad_clip: float, grad_clip_type: str
) -> int:
    params_with_grads = [p for p in stitched_module.parameters() if p.grad is not None]
    if not params_with_grads:
        return 0

    device = params_with_grads[0].device
    clipped_count = torch.zeros((), dtype=torch.int64, device=device)
    if grad_clip_type == "norm":
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=params_with_grads,
            max_norm=grad_clip,
        )
        grad_norm = torch.as_tensor(grad_norm, device=device)
        clipped_count += (grad_norm > grad_clip).to(torch.int64)
    elif grad_clip_type == "value":
        max_abs_grad = torch.stack([p.grad.detach().abs().max() for p in params_with_grads]).max()
        clipped_count += (max_abs_grad > grad_clip).to(torch.int64)
        torch.nn.utils.clip_grad_value_(
            parameters=params_with_grads,
            clip_value=grad_clip,
        )
    else:
        raise RuntimeError(f"Invalid {grad_clip_type}")

    return int(clipped_count.item())


def _step_stitched_module_optimizer(
    stitched_module: StitchedModule,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    grad_clip: Optional[float],
    grad_clip_type: str,
) -> int:
    clipped_count = 0
    if grad_clip is not None:
        grad_scaler.unscale_(optimizer)
        clipped_count = _clip_stitched_module_grads(
            stitched_module=stitched_module,
            grad_clip=grad_clip,
            grad_clip_type=grad_clip_type,
        )

    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return clipped_count


def launch_bypass_distillation(
    hydra_cfg: DictConfig, num_nodes: int = 1, node_index: int = 0
) -> None:
    """Top-level entry point for bypass distillation stage.

    Runs sewing-kit pipeline-parallel per-block knowledge distillation.

    Supports multiple bypass configurations via ``bypass.configs`` list.
    Each entry overrides ``bypass.model.model_config_overrides`` and optionally
    ``bypass.model_factory.keys_to_learn``, then runs a full bypass training.

    If ``bypass.configs`` is absent or empty, runs a single bypass training
    with the settings already in ``bypass``.

    Args:
        hydra_cfg: The full Hydra configuration with a 'bypass' section.
        num_nodes: total number of nodes splitting the sweep (for multi-node runs).
        node_index: this node's index in ``[0, num_nodes)``. Each node trains an
            interleaved subset of ``bypass.configs``; coordination across nodes is
            implicit via the shared ``bypass_runs`` dir + skip-if-complete.
    """
    if not (0 <= node_index < num_nodes):
        raise ValueError(f"node_index {node_index} must be in [0, {num_nodes})")

    configs_list = hydra_cfg.bypass.get("configs", None)

    if not configs_list:
        # Single config mode — only node 0 runs it (nothing to split).
        if node_index != 0:
            mprint(f"Node {node_index}/{num_nodes}: single bypass config handled by node 0, skipping")
            return
        # Single config mode — run once with whatever is in bypass already
        set_experiment_id(hydra_cfg)
        set_experiment_dir(hydra_cfg)
        dist.barrier()
        bypass_complete = bypass_run_is_complete(hydra_cfg) if dist.is_master() else None
        bypass_complete = dist.broadcast(bypass_complete, src=0)
        if bypass_complete:
            mprint(
                f"Bypass distillation already completed for {hydra_cfg.bypass.experiment_id}, skipping"
            )
            return
        mprint("Starting bypass distillation (single config)")
        run_bypassed_training(hydra_cfg)
        mprint("Bypass distillation completed")
        return

    base_model_config_overrides = OmegaConf.to_container(
        hydra_cfg.bypass.model.model_config_overrides, resolve=True
    )
    base_keys_to_learn = hydra_cfg.bypass.model_factory.keys_to_learn

    # Each node trains an interleaved subset of the sweep; original indices are
    # preserved for logging and experiment-id derivation. Coordination across
    # nodes is implicit (shared bypass_runs dir + skip-if-complete below).
    sweep = list(enumerate(configs_list))
    if num_nodes > 1:
        sweep = [(i, override) for (i, override) in sweep if i % num_nodes == node_index]
        mprint(
            f"Node {node_index}/{num_nodes} handling bypass configs "
            f"{[i for i, _ in sweep]} of {len(configs_list)}"
        )

    mprint(f"Starting bypass distillation sweep ({len(sweep)} of {len(configs_list)} configs)")
    for i, override in sweep:
        mprint(f"Bypass config {i + 1}/{len(configs_list)}: {override}")

        hydra_cfg.bypass.model.model_config_overrides = OmegaConf.create(
            base_model_config_overrides
        )
        hydra_cfg.bypass.model_factory.keys_to_learn = base_keys_to_learn

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
        # Per-block bookkeeping for the Stitched-Module-Losses table. Mirrored
        # into cfg.bypass on every log chunk so save_bypass_checkpoint's
        # args.json snapshot carries them, and resume can restore the columns
        # instead of trivially re-anchoring to the first post-resume chunk.
        hydra_cfg.bypass.best_losses_by_name = {}
        hydra_cfg.bypass.best_steps_by_name = {}
        hydra_cfg.bypass.initial_losses_by_name = {}

        set_experiment_id(hydra_cfg)
        set_experiment_dir(hydra_cfg)
        dist.barrier()
        bypass_complete = bypass_run_is_complete(hydra_cfg) if dist.is_master() else None
        bypass_complete = dist.broadcast(bypass_complete, src=0)
        if bypass_complete:
            mprint(
                f"Bypass config {i + 1}/{len(configs_list)} "
                f"({hydra_cfg.bypass.experiment_id}) already completed, skipping"
            )
        else:
            run_bypassed_training(hydra_cfg)
            mprint(f"Bypass config {i + 1}/{len(configs_list)} completed")

    mprint("Bypass distillation sweep completed")


def _flush_loss_buffer(
    local_buffer: dict[int, dict[str, float]],
    stitched_losses_history: Optional[dict[int, dict[str, float]]],
) -> None:
    """All-gather buffered per-iter losses and merge into master's history.

    Pickle-based ``all_gather_object`` was previously called on every micro-batch;
    batching to log-chunk boundaries reduces that cost ~``iters_per_log_chunk``×.
    All ranks must call this so the collective doesn't deadlock; only master
    actually accumulates into ``stitched_losses_history``.
    """
    if not local_buffer:
        return
    gathered = dist.allgather(local_buffer)
    if dist.is_master():
        assert stitched_losses_history is not None
        for rank_buf in gathered:
            for it, losses in rank_buf.items():
                stitched_losses_history.setdefault(it, {}).update(losses)


def _delete_old_checkpoints(
    experiment_dir: Path,
    glob_pattern: str,
    keep_name: str,
) -> None:
    if not dist.is_master():
        return
    for old_ckpt_path in experiment_dir.glob(glob_pattern):
        if old_ckpt_path.name != keep_name:
            shutil.rmtree(str(old_ckpt_path))


def _save_training_checkpoint(
    *,
    cfg: DictConfig,
    descriptor: ModelDescriptor,
    model: torch.nn.Module,
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    subdir_name: str,
    checkpoint_role: str,
    cleanup_glob: str | None = None,
) -> None:
    save_bypass_checkpoint(
        cfg=cfg,
        descriptor=descriptor,
        model=model,
        stitched_module_descriptors=stitched_module_descriptors,
        checkpoint_dir=Path(cfg.bypass.experiment_dir) / subdir_name,
        reference_checkpoint_dir=cfg.teacher_dir,
        checkpoint_role=checkpoint_role,
    )
    if cleanup_glob and cfg.bypass.model.model_overrides.delete_old_checkpoints:
        _delete_old_checkpoints(Path(cfg.bypass.experiment_dir), cleanup_glob, subdir_name)


def train(
    cfg: DictConfig,
    descriptor: ModelDescriptor,
    student_model: torch.nn.Module,
    student_stitched_model: StitchedModule,
    teacher_stitched_model: StitchedModule,
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    stitched_modules_process_ownership: StitchedModulesProcessOwnership,
    train_dataloader: Optional[DataLoader],
    val_dataloader: Optional[DataLoader],
    student_model_config: PretrainedConfig,
    skip_first_batches: int = 0,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> None:
    """Inner training loop for bypass distillation."""
    device = torch.device(f"cuda:{dist.local_rank()}")

    dist.barrier()

    # Anchor the time-based save interval at training start, not module import.
    # Earlier this was a module-level `time_start = time.time()`, which made
    # the first time-based save fire immediately if the module was imported
    # well before train() actually ran (e.g. via test collection or Hydra config
    # resolution).
    time_last_save = time.time()
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
    ownership_context = get_pipeline_ownership_context(stitched_modules_process_ownership)
    owned_stitched_module_indices = ownership_context["owned_indices"]
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
        subdir_name = f"start-step-{cfg.bypass.step_num:06d}-ckpt"
        _save_training_checkpoint(
            cfg=cfg,
            descriptor=descriptor,
            model=student_model,
            stitched_module_descriptors=stitched_module_descriptors,
            subdir_name=subdir_name,
            checkpoint_role="start",
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

    prev_rank: Optional[int] = ownership_context["prev_rank"]
    next_rank: Optional[int] = ownership_context["next_rank"]

    torch.cuda.synchronize()

    mprint(
        f"Grad scaling status: {'enabled' if cfg.bypass.training.use_grad_scaling else 'disabled'}"
    )

    # Only master consumes the dataloader — `next(train_iterator)` is gated by
    # `if dist.is_master()` further down. Building the iterator (or running
    # skip_first_batches against it) on non-master ranks wastes startup time
    # and memory proportional to the dataset, since each tokenizes the full
    # corpus only to throw it away.
    train_iterator = None
    if dist.is_master():
        assert train_dataloader is not None
        train_iterator = iter(train_dataloader)

    # Advance past the first `skip_first_batches` batches before the training loop
    # starts. Used either to skip a known-bad batch range during debugging, or to
    # roll the data iterator forward when resuming a run (model + optimizer state
    # are restored from the checkpoint, but the dataloader itself starts fresh).
    if dist.is_master() and skip_first_batches > 0:
        assert train_iterator is not None
        mprint(f"Skipping first {skip_first_batches} batches before training")
        for _ in range(skip_first_batches):
            next(train_iterator)

    mprint("Waiting for everyone before training starts")
    dist.barrier()

    step_to_save = None
    # Track best loss value for each block. Seeded from cfg.bypass so resume
    # picks up where the previous run left off (run_bypassed_training restores
    # these from args.json before train_pipeline_parallel runs).
    best_losses_by_name: dict[str, float] = dict(cfg.bypass.get("best_losses_by_name", {}))
    best_steps_by_name: dict[str, int] = dict(cfg.bypass.get("best_steps_by_name", {}))
    # Anchor for the "Δ from initial" column: per-block loss from the first log chunk.
    initial_losses_by_name: dict[str, float] = dict(cfg.bypass.get("initial_losses_by_name", {}))
    non_trainable_stitched_module_names = {
        name
        for name, descriptor in stitched_module_descriptors.items()
        if descriptor.optimizer is None
    }

    # log_interval is in optimizer-step units; multiply by grad_accum to land in
    # micro-batch units, which is what the per-iter loss collection counts.
    iters_per_log_chunk = (
        cfg.bypass.training.log_interval * cfg.bypass.training.grad_accumulation_steps
    )
    # Per-rank local buffer of {iter_num: {block_name: loss}}. We accumulate
    # losses locally on every rank and only collide them via all_gather_object
    # at log-chunk boundaries — the object collective is pickle-based and
    # was previously the per-iter sync cost. See `_flush_loss_buffer` below.
    local_losses_buffer: dict[int, dict[str, float]] = {}
    # Buffer variables. Initialise on the active device so non-master ranks
    # never hand a CPU tensor to a downstream GPU op if the master-only-fetch
    # invariant is ever relaxed (today only master replaces this in the loop).
    input_ids = torch.zeros(1, 1, dtype=torch.int64, device=device)

    aprint(
        f"previous rank: {str(prev_rank):<5} next rank: {str(next_rank):<5} {owned_stitched_module_indices=}"
    )

    # Train loop start
    while True:
        time_now = time.time()
        # Check if we've reached the maximum number of steps. `step_num` is 1-based
        # and incremented at the END of each iteration, so we must use `>` (not `>=`)
        # to ensure step `max_steps` itself runs before exiting.
        if cfg.bypass.step_num > cfg.bypass.training.max_steps:
            # Drain any residual buffered losses (< log-chunk boundary) so the
            # final partial chunk's stats reach master and can be logged before
            # the function returns. Must run on every rank — collective op.
            _flush_loss_buffer(local_losses_buffer, stitched_losses_history)
            local_losses_buffer.clear()
            if (
                cfg.bypass.model.model_overrides.save_checkpoint_when_done
                and not cfg.bypass.disable_checkpoint_save
            ):
                mprint("Saving final checkpoint before training completion")
                subdir_name = f"final-step-{cfg.bypass.step_num:06d}-ckpt"
                _save_training_checkpoint(
                    cfg=cfg,
                    descriptor=descriptor,
                    model=student_model,
                    stitched_module_descriptors=stitched_module_descriptors,
                    checkpoint_role="final",
                    subdir_name=subdir_name,
                    cleanup_glob="step-*",
                )
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
            assert train_iterator is not None
            train_data = next(train_iterator)
            input_ids = train_data["input_ids"]
            input_ids = input_ids.to(device)

        with _autocast_context(descriptor), torch.no_grad():
            teacher_input_ids = input_ids if prev_rank is None else fake_input_ids
            teacher_output = teacher_stitched_model({}, {}, teacher_input_ids)

            input_overrides = teacher_output.captured_inputs
            output_overrides = teacher_output.captured_outputs

            del teacher_output

        input_overrides["teacher_inputs"] = InputArgs(fake_input_ids)

        # Collect per-block loss tensors and batch the GPU→CPU copy to a
        # single sync point at the end of the per-block loop. Doing
        # ``.to("cpu").item()`` per block forced one CUDA synchronization per
        # block per iter, serialising the GPU pipeline across N blocks.
        iter_loss_tensors: dict[str, torch.Tensor] = {}

        for local_stitched_module_index, (
            stitched_module_name,
            stitched_module_descriptor,
        ) in enumerate(stitched_module_descriptors.items()):
            stitched_module = stitched_module_descriptor.stitched_module
            optimizer = stitched_module_descriptor.optimizer
            grad_scaler = stitched_module_descriptor.grad_scaler

            if optimizer is not None:
                assert grad_scaler is not None

                with _autocast_context(descriptor):
                    stitched_module_output = stitched_module(
                        input_overrides=input_overrides,
                        output_overrides=output_overrides,
                    )
                stitched_module_loss = stitched_module_output.captured_outputs["loss"]
                del stitched_module_output
                scaled_stitched_module_loss = (
                    stitched_module_loss / cfg.bypass.training.grad_accumulation_steps
                )
                grad_scaler.scale(scaled_stitched_module_loss).backward()
                iter_loss_tensors[stitched_module_name] = stitched_module_loss.detach()
                del scaled_stitched_module_loss
            else:
                # No real trainable parameters on this rank/block. Keep this out
                # of the numeric loss stream so genuine non-finite losses from
                # trainable blocks remain visible instead of being conflated with
                # an intentional "not trainable" sentinel.
                stitched_module_loss = None

            del stitched_module_loss

            if not is_accumulating:
                if optimizer is not None:
                    assert grad_scaler is not None
                    cfg.bypass.training.clipping_count += _step_stitched_module_optimizer(
                        stitched_module=stitched_module,
                        optimizer=optimizer,
                        grad_scaler=grad_scaler,
                        grad_clip=cfg.bypass.training.grad_clip,
                        grad_clip_type=cfg.bypass.training.grad_clip_type,
                    )

        # Single GPU→CPU sync for all per-block losses collected above. Stacking
        # into a 1-D tensor lets us issue exactly one ``.to("cpu")`` instead of
        # one per block.
        if iter_loss_tensors:
            loss_stack = torch.stack([t.flatten()[0] for t in iter_loss_tensors.values()])
            iter_stitched_module_losses: dict[str, float] = dict(
                zip(iter_loss_tensors.keys(), loss_stack.to("cpu").tolist())
            )
        else:
            iter_stitched_module_losses = {}

        if dist.is_master() and cfg.bypass.iter_num == resumed_iter_num:
            mprint(f"Starting from iter {cfg.bypass.iter_num}")

        # Buffer this rank's per-block losses locally. The collide-across-ranks
        # gather happens only at log-chunk boundaries (`_flush_loss_buffer`),
        # which cuts the per-iter pickle-based all_gather_object cost down to
        # one gather per `iters_per_log_chunk` micro-batches.
        local_losses_buffer[cfg.bypass.iter_num] = iter_stitched_module_losses
        if len(local_losses_buffer) >= iters_per_log_chunk:
            _flush_loss_buffer(local_losses_buffer, stitched_losses_history)
            local_losses_buffer.clear()

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

        step_to_save = dist.broadcast(save_signal[0], src=0)

        # Logging
        if dist.is_master():
            assert stitched_losses_history is not None
            # `iters_per_log_chunk` is computed once before the loop (in
            # micro-batch units = log_interval × grad_accum) and reused for
            # both the gather-batching threshold and this log drain.
            while len(stitched_losses_history) >= iters_per_log_chunk:
                lowest_iter = next(iter(stitched_losses_history.keys()))

                log_chunk = {
                    it: losses
                    for it, losses in stitched_losses_history.items()
                    if it - lowest_iter < iters_per_log_chunk
                }
                if len(log_chunk) < iters_per_log_chunk:
                    break

                highest_iter = list(log_chunk.keys())[-1]
                highest_iter_stats = iter_stats_history[highest_iter]

                losses_by_name = defaultdict[str, list[float]](list)
                for losses in log_chunk.values():
                    for name, loss in losses.items():
                        losses_by_name[name].append(loss)

                losses_by_name_avg = {name: mean(losses) for name, losses in losses_by_name.items()}
                non_finite_losses_by_name = {
                    name: loss
                    for name, loss in losses_by_name_avg.items()
                    if not math.isfinite(loss)
                }
                if non_finite_losses_by_name:
                    cfg.bypass.non_finite_losses_by_name = dict(non_finite_losses_by_name)
                    mprint(f"Non-finite stitched losses detected: {non_finite_losses_by_name}")

                # Anchor "Δ from initial" at the very first iter's per-block losses
                # (lowest_iter — typically iter 1 on a fresh run, the resumed iter
                # otherwise). Using the first chunk's *average* would tautologically
                # make Δ == 0 on the first row, since "Loss Value" is that same average.
                if not initial_losses_by_name:
                    initial_losses_by_name.update(stitched_losses_history[lowest_iter])

                # Update best losses tracking. Record the optimizer-step number
                # so the "Best Step" column matches the header's "step N/max" units.
                for name, current_loss in losses_by_name_avg.items():
                    if not math.isfinite(current_loss):
                        continue
                    if name not in best_losses_by_name or current_loss < best_losses_by_name[name]:
                        best_losses_by_name[name] = current_loss
                        best_steps_by_name[name] = highest_iter_stats.step_num

                # Mirror to cfg.bypass so save_bypass_checkpoint's args.json snapshot
                # carries these forward across resumes.
                cfg.bypass.best_losses_by_name = dict(best_losses_by_name)
                cfg.bypass.best_steps_by_name = dict(best_steps_by_name)
                cfg.bypass.initial_losses_by_name = dict(initial_losses_by_name)

                chunk_iter_durations = [
                    iter_stats_history[it].iter_duration for it in log_chunk.keys()
                ]
                avg_chunk_iter_duration = mean(chunk_iter_durations)
                # Report time in step units (= grad_accum × per-iter), since one
                # step is one optimizer update — what the user actually thinks of
                # as "a training step." Tokens/sec is invariant to that framing.
                avg_step_time = (
                    avg_chunk_iter_duration * cfg.bypass.training.grad_accumulation_steps
                )
                avg_token_speed = cfg.bypass.training.tokens_per_iter / avg_chunk_iter_duration
                mprint(
                    f"step {highest_iter_stats.step_num}/{cfg.bypass.training.max_steps:,}:"
                    f" avg_step_time={avg_step_time * 1000:.2f}ms"
                    f" avg_token_speed={avg_token_speed:,.0f}[tok/s]"
                )
                mprint(
                    format_stitched_losses(
                        losses_dict=losses_by_name_avg,
                        best_steps_dict=best_steps_by_name,
                        best_values_dict=best_losses_by_name,
                        initial_values_dict=initial_losses_by_name,
                        not_trainable_names=non_trainable_stitched_module_names,
                        step_number=highest_iter_stats.step_num,
                        title="Stitched Module Losses",
                    )
                )

                if cfg.bypass.wandb_log:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "step": highest_iter_stats.step_num,
                                "token_count": highest_iter_stats.token_count,
                                "token_speed": avg_token_speed,
                                "lr": highest_iter_stats.lr,
                                "grad_clipping": highest_iter_stats.clipping_count,
                            },
                            step=highest_iter_stats.step_num,
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
            val_loss = dist.broadcast(val_loss, src=dist.size() - 1)

            if val_loss < cfg.bypass.best_val_loss:
                cfg.bypass.best_val_loss = val_loss
                if not cfg.bypass.disable_checkpoint_save and cfg.bypass.save_best_ckpt:
                    subdir_name = f"best-step-{cfg.bypass.step_num:06d}-ckpt"
                    _save_training_checkpoint(
                        cfg=cfg,
                        descriptor=descriptor,
                        model=student_model,
                        stitched_module_descriptors=stitched_module_descriptors,
                        checkpoint_role="best",
                        subdir_name=subdir_name,
                        cleanup_glob="best-step-*",
                    )
                    if cfg.bypass.kill_after_first_save:
                        raise RuntimeError("Done saving checkpoint, kill_after_first_save=True")

        # Checkpoint saving (step-based or time-based)
        if not is_accumulating and (
            (cfg.bypass.step_num % cfg.bypass.model.model_overrides.save_interval) == 0
            or step_to_save == cfg.bypass.step_num
        ):
            if not cfg.bypass.disable_checkpoint_save:
                if (cfg.bypass.step_num % cfg.bypass.model.model_overrides.save_interval) == 0:
                    mprint("Saving step-interval checkpoint")
                elif step_to_save == cfg.bypass.step_num:
                    mprint("Saving time-based checkpoint")

                subdir_name = f"step-{cfg.bypass.step_num:06d}-ckpt"
                _save_training_checkpoint(
                    cfg=cfg,
                    descriptor=descriptor,
                    model=student_model,
                    stitched_module_descriptors=stitched_module_descriptors,
                    checkpoint_role="resume",
                    subdir_name=subdir_name,
                    cleanup_glob="step-*",
                )

                if cfg.bypass.kill_after_first_save:
                    dist.barrier()
                    raise RuntimeError("Done saving checkpoint, kill_after_first_save=True")

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
        if warmup_steps == 0:
            # Defensive: training loop's step starts at 1 so this branch is
            # unreachable today, but a future caller passing step=0 would hit
            # a ZeroDivisionError on `step / warmup_steps` below.
            return cfg.bypass.training.learning_rate
        lr = cfg.bypass.training.learning_rate * step / warmup_steps
    # 2) if step > lr_decay_steps, return min learning rate
    elif step > lr_decay_steps:
        lr = cfg.bypass.training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    else:
        decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
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

    set_experiment_id(cfg)
    set_experiment_dir(cfg)
    dist.barrier()
    bypass_complete = bypass_run_is_complete(cfg) if dist.is_master() else None
    bypass_complete = dist.broadcast(bypass_complete, src=0)
    if bypass_complete:
        print_rank_0(f"Bypass run {cfg.bypass.experiment_id} is already complete, skipping")
        return

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    trust_remote_code = _resolve_trust_remote_code(cfg, descriptor)
    OmegaConf.update(cfg, "bypass.trust_remote_code", trust_remote_code, force_add=True)
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

        # Resume detection has to run BEFORE the weight-loading branch below
        # so a resume can route through ``load_and_shard_model`` (the HF
        # checkpoint at ``resume_checkpoint_path`` is now the single source
        # of truth for weights — see _save_local_state docstring).
        # set_experiment_id / set_experiment_dir are idempotent and only
        # depend on cfg.bypass.model.model_config_overrides + cfg.puzzle_dir,
        # so it's safe to call them this early.
        resume_checkpoint_path: Optional[str] = None
        resume_cfg: Optional[DictConfig] = None
        resume_skip_first_batches = cfg.bypass.training.skip_first_batches
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

        resume_state_path = _get_resume_state_path(cfg, resume_checkpoint_path)
        if resume_state_path:
            resume_cfg = DictConfig(json_load(Path(resume_state_path) / "args.json"))
            saved_skip = resume_cfg.training.get(
                "skip_first_batches", cfg.bypass.training.skip_first_batches
            )
            resume_skip_first_batches = _get_resume_skip_first_batches(
                saved_skip, resume_cfg.iter_num
            )
            if "data" in resume_cfg and "shuffle_train_data_seed" in resume_cfg.data:
                cfg.bypass.data.shuffle_train_data_seed = resume_cfg.data.shuffle_train_data_seed
            if "seed" in resume_cfg:
                cfg.bypass.seed = resume_cfg.seed

        # Both ``init_checkpoint_path`` and ``resume_checkpoint_path`` point at
        # an HF-format directory; share the same loader. ``init_checkpoint_path``
        # wins if both are set (explicit user override beats auto-detect).
        weight_load_path = cfg.bypass.init_checkpoint_path or resume_state_path
        student_model = None
        if weight_load_path is not None:
            mprint(f"Loading student model from {weight_load_path}")
            student_model = load_and_shard_model(
                descriptor=descriptor,
                checkpoint_path=weight_load_path,
                owned_block_indexes=owned_block_indexes,
                trust_remote_code=trust_remote_code,
            )

        cfg.bypass.training.min_lr = (
            cfg.bypass.training.learning_rate * cfg.bypass.training.min_lr_factor
        )
        cfg.bypass.training.batch_size_per_iter = cfg.bypass.training.micro_batch_size
        cfg.bypass.training.tokens_per_iter = (
            cfg.bypass.data.block_size * cfg.bypass.training.batch_size_per_iter
        )
        requested_iters = math.ceil(
            cfg.bypass.training.training_tokens / cfg.bypass.training.tokens_per_iter
        )
        # The loop steps optimizers only after a full grad-accum window, so round
        # the requested token budget up to complete optimizer-step units and report
        # that actual budget back to the user.
        cfg.bypass.training.max_steps = math.ceil(
            requested_iters / cfg.bypass.training.grad_accumulation_steps
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
            trust_remote_code=trust_remote_code,
            token=True,
        )

        assert teacher_model_config is not None

        mprint(f"Load and shard model with: {owned_block_indexes=}, {cfg.teacher_dir=}")
        teacher_model = load_and_shard_model(
            descriptor=descriptor,
            checkpoint_path=cfg.teacher_dir,
            owned_block_indexes=owned_block_indexes,
            model_config=teacher_model_config,
            trust_remote_code=trust_remote_code,
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

        # Only master ever fetches from the train dataloader (training_loop.train
        # gates `next(train_iterator)` on `dist.is_master()`), so skip the
        # potentially-large HF dataset load + tokenisation on non-master ranks.
        if dist.is_master():
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
                source_datasets_to_discard=cfg.bypass.data.get(
                    "source_datasets_to_discard", tuple()
                ),
                bos_rate=cfg.bypass.data.bos_rate,
                shuffle_seed=cfg.bypass.data.shuffle_train_data_seed,
            )
        else:
            train_dataloader = None

        val_dataloader = None
        # Note: val_dataloader is kept constructed on every rank even though only
        # master reads from it inside calculate_losses_pipeline. The validation
        # block uses `val_dataloader is not None` as a "validation enabled" gate
        # that must agree across ranks — and calculate_losses_pipeline itself is
        # pipeline-parallel and requires every rank to enter it. Skipping
        # construction on non-master ranks would break those invariants.
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
                source_datasets_to_discard=cfg.bypass.data.get(
                    "source_datasets_to_discard", tuple()
                ),
                bos_rate=cfg.bypass.data.bos_rate,
                realized_cache_dir=cfg.bypass.data.get("realized_cache_dir", None),
            )

        # set_experiment_id / set_experiment_dir already ran above (before
        # weight loading) so the resume detection could use experiment_dir.

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

        # ``resume_state_path`` was determined earlier (before weight
        # loading); the student weights are already in place via
        # ``load_and_shard_model``. Only the optimizer/scaler state needs to
        # be restored from the per-block ``stitched/`` files.
        if resume_state_path:
            load_local_state(
                stitched_module_descriptors=stitched_module_descriptors,
                checkpoint_path=resume_state_path,
            )

            assert resume_cfg is not None

            # Periodic checkpoints are saved before the loop increments counters,
            # so their args.json is inclusive and needs a +1 bump. Final
            # checkpoints are saved after the loop already advanced beyond the
            # last completed step, so their counters are already the next values.
            resume_from_final = Path(resume_state_path).name.startswith("final-step-")
            counter_bump = 0 if resume_from_final else 1
            cfg.bypass.iter_num = resume_cfg.iter_num + counter_bump
            cfg.bypass.token_count = resume_cfg.token_count
            cfg.bypass.step_num = resume_cfg.step_num + counter_bump
            cfg.bypass.best_val_loss = resume_cfg.best_val_loss
            cfg.bypass.training.clipping_count = resume_cfg.training.clipping_count
            # Per-block bookkeeping. .get() defaults handle resume from older ckpts
            # that predate these fields.
            cfg.bypass.best_losses_by_name = resume_cfg.get("best_losses_by_name", {})
            cfg.bypass.best_steps_by_name = resume_cfg.get("best_steps_by_name", {})
            cfg.bypass.initial_losses_by_name = resume_cfg.get("initial_losses_by_name", {})
            mprint(f"Resume from iter_num: {cfg.bypass.iter_num}")

            # Only copy wandb.run_id if it exists in resume config
            if hasattr(resume_cfg, "wandb") and hasattr(resume_cfg.wandb, "run_id"):
                cfg.bypass.wandb.run_id = resume_cfg.wandb.run_id

            cfg.bypass.save_checkpoint_before_training = False
            cfg.bypass.validate_teacher_model = False
            cfg.bypass.validate_student_model = False

            cfg.bypass.resume_checkpoint_path = resume_state_path

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
            _autocast_context(descriptor),
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
            skip_first_batches=resume_skip_first_batches,
            tokenizer=tokenizer,
        )

        aprint("Finished training successfully!")
        dist.barrier()

    except Exception:
        # Print the traceback explicitly so distributed runs surface it on every
        # rank's stderr (workers under torchrun otherwise lose ordering), then
        # re-raise so test frameworks see the real exception instead of a
        # generic SystemExit(1).
        print(traceback.format_exc(), file=sys.stderr)
        raise

    dist.barrier()
    _finalize_bypass_run(cfg)
    dist.barrier()


def realize_bypass_checkpoints(cfg: DictConfig) -> tuple[Path, Path]:
    """Create symlinks from bypass checkpoint directories to the ckpts directory."""
    state = load_bypass_state(cfg.bypass.experiment_dir) or {}
    checkpoints = state.get("checkpoints", {})
    realize_mode = cfg.bypass.get("realize_best_or_latest", "latest")
    if realize_mode == "best":
        role_preference = ("best", "final", "resume")
    elif realize_mode == "latest":
        role_preference = ("final", "resume", "best")
    else:
        raise ValueError(f"Invalid bypass.realize_best_or_latest={realize_mode!r}")

    checkpoint_dir = None
    for role in role_preference:
        candidate = checkpoints.get(role)
        if candidate and Path(candidate).exists():
            checkpoint_dir = Path(candidate).resolve()
            break

    if checkpoint_dir is None:
        fallback = Path(cfg.bypass.experiment_dir) / "latest"
        if fallback.exists():
            checkpoint_dir = fallback.resolve()
        else:
            raise FileNotFoundError(
                f"Could not find a bypass checkpoint to realize in {cfg.bypass.experiment_dir}"
            )

    ckpts_dir = Path(cfg.puzzle_dir) / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    symlink_name = ckpts_dir / cfg.bypass.experiment_id
    if symlink_name.exists() or symlink_name.is_symlink():
        symlink_name.unlink()

    symlink_name.symlink_to(checkpoint_dir.resolve(), target_is_directory=True)
    mprint(f"Created symlink: {symlink_name} -> {checkpoint_dir}")
    return checkpoint_dir, symlink_name
