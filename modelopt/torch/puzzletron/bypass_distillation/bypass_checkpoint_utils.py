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

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from tqdm import tqdm

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import save_checkpoint
from modelopt.torch.puzzletron.tools.logger import aprint, mprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump

from .stitched_model_factory import StitchedModuleDescriptor


def find_latest_run_dir(run_parent_dir: Union[str, Path]) -> str | None:
    """Find the latest plain-iter checkpoint directory within a run parent directory.

    Resume must pick a directory created by the step-interval / time-based / final save
    paths (named ``iter-NNNNNN-ckpt``) — not ``best-iter-*`` (which corresponds to a
    validation-best snapshot whose optimizer state may be stale relative to the latest
    iter), nor ``start-iter-*`` / ``final-iter-*`` (markers, not resume points).
    """
    run_parent_dir = Path(run_parent_dir)

    # Check for the "latest" symlink — set only by save_bypass_checkpoint, always
    # points at a plain ``iter-*`` directory. Fast path.
    latest_dir = run_parent_dir / "latest"
    if latest_dir.exists() and (latest_dir / "saving_completed").exists():
        return str(latest_dir)

    # Fallback: scan plain ``iter-NNNNNN-ckpt`` directories only.
    iter_re = re.compile(r"^iter-(\d+)-ckpt$")
    candidate_dirs: list[tuple[int, Path]] = []
    for d in run_parent_dir.iterdir():
        if not d.is_dir():
            continue
        match = iter_re.match(d.name)
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
    """Load local state from a checkpoint.

    Loads both optimizer and state dicts into stitched module descriptors.
    Modifies stitched_module_descriptors in place.
    """
    device = torch.device(f"cuda:{dist.local_rank()}")
    load_dir = Path(checkpoint_path)

    if not load_dir.exists():
        raise RuntimeError(f'Can\'t load local state. "{load_dir}" does not exist.')

    for stitched_module_name, stitched_module_descriptor in stitched_module_descriptors.items():
        stitched_module = stitched_module_descriptor.stitched_module
        optimizer = stitched_module_descriptor.optimizer
        grad_scaler = stitched_module_descriptor.grad_scaler

        state_dict_path = load_dir / "stitched" / f"{stitched_module_name}.state_dict.pth"
        mprint(f"Loading state dict for module {stitched_module_name} from {state_dict_path}")
        loaded_state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        loaded_state_dict = {**stitched_module.state_dict(), **loaded_state_dict}

        stitched_module.load_state_dict(loaded_state_dict)
        del loaded_state_dict

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
    save_dir = Path(checkpoint_dir) / "stitched"

    if dist.is_master():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Main process creates the directory, so we must wait for it to finish
    dist.barrier()

    for stitched_module_name, stitched_module_descriptor in tqdm(
        stitched_module_descriptors.items()
    ):
        optimizer = stitched_module_descriptor.optimizer
        grad_scaler = stitched_module_descriptor.grad_scaler

        state_dict_path = save_dir / f"{stitched_module_name}.state_dict.pth"
        aprint(f"Saving state dict for module {stitched_module_name} to {state_dict_path}")
        state_dict = {
            **stitched_module_descriptor.owned_parameters,
            **stitched_module_descriptor.owned_buffers,
        }
        _save_local_file(state_dict, state_dict_path, overwrite=overwrite)

        if optimizer is not None:
            optimizer_state_path = save_dir / f"{stitched_module_name}.optimizer_state.pth"
            mprint(
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
    # Save as HF checkpoint
    save_checkpoint(model=model, checkpoint_dir=checkpoint_dir, descriptor=descriptor)

    if dist.is_master():
        # Create 'latest' symlink
        latest_symlink = Path(cfg.bypass.experiment_dir) / "latest"
        latest_symlink.unlink(missing_ok=True)
        latest_symlink.symlink_to(checkpoint_dir.name)
        # Save config args json
        json_dump(cfg.bypass, checkpoint_dir / "args.json")
        # Save completed file
        completed_file = checkpoint_dir / "saving_completed"
        completed_file.touch()

    dist.barrier()
    mprint("Checkpoint save done")
