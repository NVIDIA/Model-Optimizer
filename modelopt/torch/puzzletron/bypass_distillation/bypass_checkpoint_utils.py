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

"""Checkpoint utilities for bypass distillation."""

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from tqdm import tqdm

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import save_checkpoint_from_shards
from modelopt.torch.puzzletron.tools.logger import aprint, mprint
from modelopt.torch.utils.robust_json import json_dump

from .bypass_utils import load_bypass_state, update_bypass_checkpoint_state
from .stitched_model_factory import StitchedModuleDescriptor

__all__ = ["find_latest_run_dir", "load_local_state", "save_bypass_checkpoint"]


def find_latest_run_dir(run_parent_dir: Union[str, Path]) -> str | None:
    """Find the latest plain-step checkpoint directory within a run parent directory.

    Resume prefers the manifest's final checkpoint, then the latest plain step
    checkpoint. It must not pick ``best-step-*`` because validation-best snapshots
    can be stale relative to the latest optimizer state, nor ``start-step-*``.
    """
    run_parent_dir = Path(run_parent_dir)

    state = load_bypass_state(run_parent_dir)
    if state is not None:
        checkpoints = state.get("checkpoints", {})
        for role in ("final", "resume"):
            candidate = checkpoints.get(role)
            if candidate and (Path(candidate) / "saving_completed").exists():
                return str(candidate)

    # Check for the "latest" symlink. Current checkpoints only update it for
    # plain periodic resume checkpoints, but older runs may have pointed it at a
    # best/start/final checkpoint. Validate the target name before accepting it.
    latest_dir = run_parent_dir / "latest"
    if latest_dir.exists():
        latest_resolved = latest_dir.resolve()
        if (
            re.match(r"^step-\d+-ckpt$", latest_resolved.name)
            and (latest_resolved / "saving_completed").exists()
        ):
            return str(latest_resolved)

    # Fallback: scan plain ``step-NNNNNN-ckpt`` directories only.
    # Treat a missing parent dir as "no previous runs" rather than fatal — this
    # handles two cases cleanly: a freshly-wiped bypass dir, and the race where
    # non-master ranks reach this function before master finishes the
    # ``set_experiment_dir`` mkdir on a shared filesystem.
    if not run_parent_dir.exists():
        return None
    step_re = re.compile(r"^step-(\d+)-ckpt$")
    candidate_dirs: list[tuple[int, Path]] = []
    for d in run_parent_dir.iterdir():
        if not d.is_dir():
            continue
        match = step_re.match(d.name)
        if match:
            candidate_dirs.append((int(match.group(1)), d))

    if not candidate_dirs:
        return None

    candidate_dirs.sort(key=lambda x: x[0], reverse=True)
    for _, ckpt_dir in candidate_dirs:
        if (ckpt_dir / "saving_completed").exists():
            return str(ckpt_dir)
    return None


def load_local_state(
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_path: str | Path,
) -> None:
    """Load optimizer and grad-scaler state for each stitched module.

    Weights are NOT loaded here — they live in the HF checkpoint at
    ``checkpoint_path`` and must be loaded into the student model via
    ``load_and_shard_model`` before this function runs (typically by setting
    ``init_checkpoint_path`` to the resume directory). This avoids
    persisting the same parameters twice (once in ``stitched/*.pth`` and
    once in the HF state dict).

    Modifies ``stitched_module_descriptors`` in place.
    """
    device = torch.device(f"cuda:{dist.local_rank()}")
    load_dir = Path(checkpoint_path)

    if not load_dir.exists():
        raise RuntimeError(f'Can\'t load local state. "{load_dir}" does not exist.')

    for stitched_module_name, stitched_module_descriptor in stitched_module_descriptors.items():
        optimizer = stitched_module_descriptor.optimizer
        grad_scaler = stitched_module_descriptor.grad_scaler

        if optimizer is not None:
            optimizer_state_path = (
                load_dir / "stitched" / f"{stitched_module_name}.optimizer_state.pth"
            )
            mprint(
                f"Loading optimizer state for module {stitched_module_name} from {optimizer_state_path}"
            )
            loaded_optimizer_state = torch.load(
                optimizer_state_path, map_location=device, weights_only=True
            )
            optimizer.load_state_dict(loaded_optimizer_state)
            del loaded_optimizer_state

        # Restore GradScaler state (only relevant when use_grad_scaling=True; for the
        # default bf16 / use_grad_scaling=False path the scaler is disabled and its
        # state is a no-op, but we still load it if present for forward-compatibility).
        # Older checkpoints predating this save path won't have the file — skip silently.
        if grad_scaler is not None:
            grad_scaler_state_path = (
                load_dir / "stitched" / f"{stitched_module_name}.grad_scaler.pth"
            )
            if grad_scaler_state_path.exists():
                mprint(
                    f"Loading grad_scaler state for module {stitched_module_name} "
                    f"from {grad_scaler_state_path}"
                )
                loaded_scaler_state = torch.load(
                    grad_scaler_state_path, map_location=device, weights_only=True
                )
                grad_scaler.load_state_dict(loaded_scaler_state)
                del loaded_scaler_state


def _save_local_file(obj, save_path: Path | str, overwrite=True):
    save_path = Path(save_path)
    if save_path.exists():
        if not overwrite:
            mprint(f'WARNING: Local save path "{save_path}" already exists. Skipping')
            return
    torch.save(obj, save_path)


def _save_local_state(
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_dir: Path | str,
    overwrite=True,
) -> None:
    """Persist optimizer and grad-scaler state for each stitched module.

    Weights are intentionally NOT saved here. The same trainable parameters
    would otherwise land on disk twice — once as ``stitched/{block}.state_dict.pth``
    and once as part of the HF checkpoint that ``save_bypass_checkpoint``
    writes at the top level via ``save_checkpoint(model, ...)``. The HF
    checkpoint is the single source of truth for weights; this directory
    only carries the optimizer/scaler state that the HF format doesn't
    cover.
    """
    save_dir = Path(checkpoint_dir) / "stitched"

    if dist.is_master():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Main process creates the directory, so we must wait for it to finish
    dist.barrier()

    for stitched_module_name, stitched_module_descriptor in tqdm(
        stitched_module_descriptors.items(), disable=not dist.is_master()
    ):
        optimizer = stitched_module_descriptor.optimizer
        grad_scaler = stitched_module_descriptor.grad_scaler

        if optimizer is not None:
            optimizer_state_path = save_dir / f"{stitched_module_name}.optimizer_state.pth"
            aprint(
                f"Saving optimizer state for module {stitched_module_name} to {optimizer_state_path}"
            )
            _save_local_file(optimizer.state_dict(), optimizer_state_path, overwrite=overwrite)

        # Persist GradScaler state. Required for correct resume when
        # use_grad_scaling=True (state dict carries running scale + growth tracker).
        # For the default bf16 / use_grad_scaling=False path the state dict is trivial
        # but cheap, so save unconditionally whenever a scaler exists — keeps the
        # save/load paths symmetric with the optimizer.
        if grad_scaler is not None:
            grad_scaler_state_path = save_dir / f"{stitched_module_name}.grad_scaler.pth"
            mprint(
                f"Saving grad_scaler state for module {stitched_module_name} "
                f"to {grad_scaler_state_path}"
            )
            _save_local_file(grad_scaler.state_dict(), grad_scaler_state_path, overwrite=overwrite)

    dist.barrier()


def save_bypass_checkpoint(
    cfg: DictConfig,
    descriptor: ModelDescriptor,
    model: torch.nn.Module,
    stitched_module_descriptors: OrderedDict[str, StitchedModuleDescriptor],
    checkpoint_dir: Path | str,
    reference_checkpoint_dir: Optional[Path] = None,
    checkpoint_role: str = "resume",
) -> None:
    """Save a bypass distillation checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    mprint("Starting checkpoint save")
    mprint(f"Saving checkpoint to {checkpoint_dir}")

    # Save stitched module states
    _save_local_state(
        stitched_module_descriptors=stitched_module_descriptors,
        checkpoint_dir=checkpoint_dir,
        overwrite=cfg.bypass.model.model_overrides.delete_old_checkpoints,
    )
    # Save as HF checkpoint. Must use the gather-aware variant: bypass training is
    # pipeline-parallel so each rank's `model.state_dict()` only carries its own
    # owned blocks. The unsharded `save_checkpoint` would have every rank write a
    # partial `model.safetensors.index.json` to the same path (last writer wins),
    # producing an index that omits most ranks' weights — resume then leaves params
    # on the meta device.
    save_checkpoint_from_shards(model=model, checkpoint_dir=checkpoint_dir, descriptor=descriptor)

    if dist.is_master():
        if checkpoint_role == "resume":
            # Create 'latest' symlink via tmp-symlink + atomic rename so concurrent
            # readers on a shared filesystem never observe a missing `latest`. The
            # plain unlink + symlink_to pair leaves a brief window where the link
            # doesn't exist; Path.replace (== os.replace) is atomic on POSIX.
            latest_symlink = Path(cfg.bypass.experiment_dir) / "latest"
            tmp_symlink = latest_symlink.with_name(f".latest_tmp_{os.getpid()}")
            tmp_symlink.unlink(missing_ok=True)
            tmp_symlink.symlink_to(checkpoint_dir.name)
            tmp_symlink.replace(latest_symlink)
        # Save config args json
        json_dump(cfg.bypass, checkpoint_dir / "args.json")
        model_factory_cfg = cfg.bypass.get("model_factory", {})
        json_dump(
            {"keys_to_learn": model_factory_cfg.get("keys_to_learn", "entire_block")},
            checkpoint_dir / "bypass_config.json",
        )
        # Save completed file
        completed_file = checkpoint_dir / "saving_completed"
        completed_file.touch()
        update_bypass_checkpoint_state(cfg, checkpoint_dir, checkpoint_role)

    dist.barrier()
    mprint("Checkpoint save done")
