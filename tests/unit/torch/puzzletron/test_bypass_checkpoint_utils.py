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

"""CPU unit tests for ``bypass_checkpoint_utils``.

The save/resume contract here is the most important regression surface in the
bypass feature: a wrong checkpoint pick or a missing ``saving_completed``
marker silently restarts training from the wrong iteration.

What's covered here (CPU-only, codecov-visible):
    * ``find_latest_run_dir`` — every branch of the regex/scan/symlink logic.
    * ``_save_local_state`` — same three save-path assertions as the GPU file
      (state_dict / optimizer / grad_scaler), but on CPU so codecov picks them
      up. The GPU file's ``test_load_local_state_*`` cases stay there because
      ``load_local_state`` constructs ``torch.device(f"cuda:{rank}")`` directly.
    * ``save_bypass_checkpoint`` — orchestration: ``latest`` symlink update,
      ``args.json`` dump, ``saving_completed`` marker, master-only gating.
"""

import os
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp.grad_scaler import GradScaler

from modelopt.torch.puzzletron.bypass_distillation import bypass_checkpoint_utils as bcu
from modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory import (
    StitchedModuleDescriptor,
)

# ---------------------------------------------------------------------------
# Shared fixture: silence the dist helpers so these run single-process / CPU.
# Mirrors tests/gpu/torch/puzzletron/test_bypass_checkpoint_utils.py:56-62.
# ---------------------------------------------------------------------------


@pytest.fixture
def bcu_no_dist(monkeypatch):
    monkeypatch.setattr(bcu.dist, "local_rank", lambda: 0)
    monkeypatch.setattr(bcu.dist, "is_master", lambda: True)
    monkeypatch.setattr(bcu.dist, "barrier", lambda: None)
    return bcu


def _make_descriptor(*, with_optimizer: bool = True, with_scaler: bool = True):
    """Build a CPU-only StitchedModuleDescriptor — the GPU file's helper minus
    the configurable init_scale (we don't round-trip the scaler here)."""
    module = nn.Linear(4, 4, bias=False)
    owned_parameters = dict(module.named_parameters())
    optimizer = torch.optim.AdamW(list(module.parameters()), lr=1e-3) if with_optimizer else None
    scaler = GradScaler(device="cpu", enabled=True, init_scale=2.0**16) if with_scaler else None
    return StitchedModuleDescriptor(
        stitched_module=module,
        owned_parameters=owned_parameters,
        owned_buffers={},
        optimizer=optimizer,
        grad_scaler=scaler,
    )


def _make_checkpoint_dir(parent: Path, name: str, *, completed: bool = True) -> Path:
    checkpoint_dir = parent / name
    checkpoint_dir.mkdir(parents=True)
    if completed:
        (checkpoint_dir / "saving_completed").touch()
    return checkpoint_dir


# ---------------------------------------------------------------------------
# find_latest_run_dir
# ---------------------------------------------------------------------------


def test_find_latest_run_dir_scans_highest_completed_plain_step(tmp_path: Path):
    """The scan branch picks the highest completed plain step checkpoint only."""
    scan_dir = tmp_path / "scan"
    _make_checkpoint_dir(scan_dir, "step-000005-ckpt")
    expected = _make_checkpoint_dir(scan_dir, "step-000020-ckpt")
    _make_checkpoint_dir(scan_dir, "step-000099-ckpt", completed=False)
    for name in ("best-step-000099-ckpt", "start-step-000001-ckpt", "final-step-000050-ckpt"):
        _make_checkpoint_dir(scan_dir, name)

    assert bcu.find_latest_run_dir(scan_dir) == str(expected)

    no_completed_plain_steps = tmp_path / "no_completed_plain_steps"
    _make_checkpoint_dir(no_completed_plain_steps, "step-000010-ckpt", completed=False)
    _make_checkpoint_dir(no_completed_plain_steps, "best-step-000020-ckpt")
    assert bcu.find_latest_run_dir(no_completed_plain_steps) is None


def test_find_latest_run_dir_handles_latest_symlink_fast_path_and_fallbacks(tmp_path: Path):
    """The ``latest`` symlink, when present and complete, short-circuits the
    scan — even when a numerically higher step dir also has a marker. This
    matters because the scan branch can be slow on filesystems with many
    step dirs (NFS, lustre)."""
    complete_latest = tmp_path / "complete_latest"
    target = _make_checkpoint_dir(complete_latest, "step-000010-ckpt")
    _make_checkpoint_dir(complete_latest, "step-000020-ckpt")
    (complete_latest / "latest").symlink_to(target.name)
    assert bcu.find_latest_run_dir(complete_latest) == str(target.resolve())

    incomplete_latest = tmp_path / "incomplete_latest"
    incomplete = _make_checkpoint_dir(incomplete_latest, "step-000020-ckpt", completed=False)
    completed = _make_checkpoint_dir(incomplete_latest, "step-000010-ckpt")
    (incomplete_latest / "latest").symlink_to(incomplete.name)
    assert bcu.find_latest_run_dir(incomplete_latest) == str(completed)

    latest_to_best = tmp_path / "latest_to_best"
    best = _make_checkpoint_dir(latest_to_best, "best-step-000020-ckpt")
    completed = _make_checkpoint_dir(latest_to_best, "step-000010-ckpt")
    (latest_to_best / "latest").symlink_to(best.name)
    assert bcu.find_latest_run_dir(latest_to_best) == str(completed)


# ---------------------------------------------------------------------------
# _save_local_state: optimizer + grad_scaler only.
# Weights deliberately do NOT land here — the HF checkpoint at the same
# directory carries the full student state dict via ``save_checkpoint``.
# Saving the per-block weights again would just double the disk footprint.
# ---------------------------------------------------------------------------


def test_save_local_state_writes_only_optimizer_and_grad_scaler_state(tmp_path: Path, bcu_no_dist):
    descriptors = OrderedDict([("block_0", _make_descriptor())])
    bcu_no_dist._save_local_state(descriptors, tmp_path)
    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.optimizer_state.pth").exists()
    assert (stitched / "block_0.grad_scaler.pth").exists()
    assert not (stitched / "block_0.state_dict.pth").exists()

    stale_optimizer_state = {"stale": torch.tensor([1])}
    stale_scaler_state = {"stale": torch.tensor([1])}
    torch.save(stale_optimizer_state, stitched / "block_0.optimizer_state.pth")
    torch.save(stale_scaler_state, stitched / "block_0.grad_scaler.pth")

    bcu_no_dist._save_local_state(descriptors, tmp_path)

    optimizer_state = torch.load(stitched / "block_0.optimizer_state.pth", weights_only=True)
    grad_scaler_state = torch.load(stitched / "block_0.grad_scaler.pth", weights_only=True)
    assert "stale" not in optimizer_state
    assert "stale" not in grad_scaler_state


def test_save_local_state_respects_optional_optimizer_and_grad_scaler(tmp_path: Path, bcu_no_dist):
    for name, descriptor, expected_files in [
        ("full", _make_descriptor(), {"block_0.optimizer_state.pth", "block_0.grad_scaler.pth"}),
        (
            "no_scaler",
            _make_descriptor(with_scaler=False),
            {"block_0.optimizer_state.pth"},
        ),
        ("no_optimizer", _make_descriptor(with_optimizer=False, with_scaler=False), set()),
    ]:
        checkpoint_dir = tmp_path / name
        descriptors = OrderedDict([("block_0", descriptor)])
        bcu_no_dist._save_local_state(descriptors, checkpoint_dir)
        stitched = checkpoint_dir / "stitched"
        assert {path.name for path in stitched.glob("*")} == expected_files


# ---------------------------------------------------------------------------
# save_bypass_checkpoint — orchestration: symlink, args.json, marker
# ---------------------------------------------------------------------------


def _make_save_cfg(experiment_dir: Path, *, delete_old: bool = True):
    """Minimal cfg shape used by ``save_bypass_checkpoint``.

    ``cfg.bypass`` is the object that gets dumped to ``args.json``, so it must
    be JSON-serialisable (or DictConfig-with-primitives, which json_dump handles).
    """
    return OmegaConf.create(
        {
            "bypass": {
                "experiment_dir": str(experiment_dir),
                "model": {"model_overrides": {"delete_old_checkpoints": delete_old}},
                "iter_num": 7,
            }
        }
    )


@pytest.fixture
def patched_save(monkeypatch, bcu_no_dist):
    """Stub out the heavy callees so the test only exercises the orchestration
    logic in ``save_bypass_checkpoint``."""
    monkeypatch.setattr(bcu_no_dist, "_save_local_state", lambda **kwargs: None)
    monkeypatch.setattr(bcu_no_dist, "save_checkpoint_from_shards", lambda **kwargs: None)
    return bcu_no_dist


def test_save_bypass_checkpoint_updates_latest_symlink_and_marker(tmp_path: Path, patched_save):
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    old_target = experiment_dir / "step-000003-ckpt"
    old_target.mkdir()
    checkpoint_dir = experiment_dir / "step-000007-ckpt"
    checkpoint_dir.mkdir()
    (experiment_dir / "latest").symlink_to(old_target.name)

    cfg = _make_save_cfg(experiment_dir)
    patched_save.save_bypass_checkpoint(
        cfg=cfg,
        descriptor=None,
        model=None,
        stitched_module_descriptors=OrderedDict(),
        checkpoint_dir=checkpoint_dir,
    )

    latest = experiment_dir / "latest"
    assert latest.is_symlink()
    # Symlink target is relative — just the dir name, so it resolves under experiment_dir.
    assert os.readlink(latest) == "step-000007-ckpt"
    assert latest.resolve() == checkpoint_dir.resolve()
    assert (checkpoint_dir / "args.json").exists()
    assert (checkpoint_dir / "saving_completed").exists()


def test_save_bypass_checkpoint_best_does_not_replace_latest(tmp_path: Path, patched_save):
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    resume_target = experiment_dir / "step-000003-ckpt"
    resume_target.mkdir()
    best_target = experiment_dir / "best-step-000007-ckpt"
    best_target.mkdir()
    (experiment_dir / "latest").symlink_to(resume_target.name)

    cfg = _make_save_cfg(experiment_dir)
    patched_save.save_bypass_checkpoint(
        cfg=cfg,
        descriptor=None,
        model=None,
        stitched_module_descriptors=OrderedDict(),
        checkpoint_dir=best_target,
        checkpoint_role="best",
    )

    assert os.readlink(experiment_dir / "latest") == "step-000003-ckpt"
    assert (best_target / "saving_completed").exists()
    assert (best_target / "bypass_config.json").exists()


def test_save_bypass_checkpoint_master_only_skips_symlink_on_non_master(
    tmp_path: Path, monkeypatch, patched_save
):
    """Non-master ranks must not write the symlink, args.json, or marker —
    only rank 0 owns those files. The other ranks still call _save_local_state
    (their owned blocks) but stop short of the per-experiment metadata."""
    monkeypatch.setattr(patched_save.dist, "is_master", lambda: False)

    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    checkpoint_dir = experiment_dir / "step-000007-ckpt"
    checkpoint_dir.mkdir()

    cfg = _make_save_cfg(experiment_dir)
    patched_save.save_bypass_checkpoint(
        cfg=cfg,
        descriptor=None,
        model=None,
        stitched_module_descriptors=OrderedDict(),
        checkpoint_dir=checkpoint_dir,
    )

    assert not (experiment_dir / "latest").exists()
    assert not (checkpoint_dir / "args.json").exists()
    assert not (checkpoint_dir / "saving_completed").exists()
