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
    * ``_save_local_file`` — overwrite/skip semantics.
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


# ---------------------------------------------------------------------------
# find_latest_run_dir
# ---------------------------------------------------------------------------


def test_find_latest_run_dir_returns_none_for_empty_dir(tmp_path: Path):
    assert bcu.find_latest_run_dir(tmp_path) is None


def test_find_latest_run_dir_picks_only_iter_with_marker(tmp_path: Path):
    iter_dir = tmp_path / "iter-000010-ckpt"
    iter_dir.mkdir()
    (iter_dir / "saving_completed").touch()
    assert bcu.find_latest_run_dir(tmp_path) == str(iter_dir)


def test_find_latest_run_dir_picks_highest_iter_number(tmp_path: Path):
    """When several plain iter checkpoints have completed markers, the highest
    integer wins — not lexicographic order, not insertion order."""
    for i in (5, 10, 20):
        d = tmp_path / f"iter-{i:06d}-ckpt"
        d.mkdir()
        (d / "saving_completed").touch()
    assert bcu.find_latest_run_dir(tmp_path) == str(tmp_path / "iter-000020-ckpt")


def test_find_latest_run_dir_skips_iter_without_marker(tmp_path: Path):
    """A partially-written checkpoint (no ``saving_completed``) must be skipped
    even when it has a higher iter number — otherwise resume would crash on a
    truncated state dict."""
    high = tmp_path / "iter-000099-ckpt"
    high.mkdir()
    # No saving_completed → must be ignored.
    low = tmp_path / "iter-000050-ckpt"
    low.mkdir()
    (low / "saving_completed").touch()
    assert bcu.find_latest_run_dir(tmp_path) == str(low)


def test_find_latest_run_dir_returns_none_when_no_iter_has_marker(tmp_path: Path):
    (tmp_path / "iter-000010-ckpt").mkdir()
    (tmp_path / "iter-000020-ckpt").mkdir()
    # No saving_completed anywhere.
    assert bcu.find_latest_run_dir(tmp_path) is None


def test_find_latest_run_dir_excludes_non_plain_iter_names(tmp_path: Path):
    """``best-iter-*`` / ``start-iter-*`` / ``final-iter-*`` aren't valid resume
    targets — pinned by the docstring on lines 39-42."""
    for name in ("best-iter-000099-ckpt", "start-iter-000001-ckpt", "final-iter-000050-ckpt"):
        d = tmp_path / name
        d.mkdir()
        (d / "saving_completed").touch()
    # No plain iter-*-ckpt at all.
    assert bcu.find_latest_run_dir(tmp_path) is None


def test_find_latest_run_dir_uses_latest_symlink_fast_path(tmp_path: Path):
    """The ``latest`` symlink, when present and complete, short-circuits the
    scan — even when a numerically higher iter dir also has a marker. This
    matters because the scan branch can be slow on filesystems with many
    iter dirs (NFS, lustre)."""
    target = tmp_path / "iter-000010-ckpt"
    target.mkdir()
    (target / "saving_completed").touch()
    (tmp_path / "latest").symlink_to(target.name)

    higher = tmp_path / "iter-000020-ckpt"
    higher.mkdir()
    (higher / "saving_completed").touch()

    # Symlink wins despite higher iter existing.
    assert bcu.find_latest_run_dir(tmp_path) == str(tmp_path / "latest")


def test_find_latest_run_dir_falls_through_when_latest_lacks_marker(tmp_path: Path):
    """A ``latest`` symlink whose target lacks ``saving_completed`` (interrupted
    save) must be ignored, falling through to the highest completed iter."""
    incomplete = tmp_path / "iter-000020-ckpt"
    incomplete.mkdir()
    # No saving_completed.
    (tmp_path / "latest").symlink_to(incomplete.name)

    completed = tmp_path / "iter-000010-ckpt"
    completed.mkdir()
    (completed / "saving_completed").touch()

    assert bcu.find_latest_run_dir(tmp_path) == str(completed)


# ---------------------------------------------------------------------------
# _save_local_file
# ---------------------------------------------------------------------------


def test_save_local_file_writes_object_to_disk(tmp_path: Path):
    target = tmp_path / "blob.pth"
    bcu._save_local_file({"a": torch.tensor([1, 2, 3])}, target)
    assert target.exists()
    loaded = torch.load(target, weights_only=True)
    assert torch.equal(loaded["a"], torch.tensor([1, 2, 3]))


def test_save_local_file_overwrite_true_replaces_contents(tmp_path: Path):
    target = tmp_path / "blob.pth"
    bcu._save_local_file({"v": torch.tensor([1])}, target)
    bcu._save_local_file({"v": torch.tensor([99])}, target, overwrite=True)
    loaded = torch.load(target, weights_only=True)
    assert torch.equal(loaded["v"], torch.tensor([99]))


def test_save_local_file_overwrite_false_skips_existing(tmp_path: Path):
    target = tmp_path / "blob.pth"
    bcu._save_local_file({"v": torch.tensor([1])}, target)
    # Second save should be a no-op.
    bcu._save_local_file({"v": torch.tensor([99])}, target, overwrite=False)
    loaded = torch.load(target, weights_only=True)
    assert torch.equal(loaded["v"], torch.tensor([1]))


# ---------------------------------------------------------------------------
# _save_local_state: optimizer + grad_scaler only.
# Weights deliberately do NOT land here — the HF checkpoint at the same
# directory carries the full student state dict via ``save_checkpoint``.
# Saving the per-block weights again would just double the disk footprint.
# ---------------------------------------------------------------------------


def test_save_local_state_writes_optimizer_and_grad_scaler(tmp_path: Path, bcu_no_dist):
    descriptors = OrderedDict([("block_0", _make_descriptor())])
    bcu_no_dist._save_local_state(descriptors, tmp_path)
    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.optimizer_state.pth").exists()
    assert (stitched / "block_0.grad_scaler.pth").exists()


def test_save_local_state_does_not_write_weights_state_dict(tmp_path: Path, bcu_no_dist):
    """Pin the de-duplication: weights live in the HF checkpoint, not here."""
    descriptors = OrderedDict([("block_0", _make_descriptor())])
    bcu_no_dist._save_local_state(descriptors, tmp_path)
    assert not (tmp_path / "stitched" / "block_0.state_dict.pth").exists()


def test_save_local_state_skips_grad_scaler_when_descriptor_has_none(tmp_path: Path, bcu_no_dist):
    descriptors = OrderedDict([("block_0", _make_descriptor(with_scaler=False))])
    bcu_no_dist._save_local_state(descriptors, tmp_path)
    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.optimizer_state.pth").exists()
    assert not (stitched / "block_0.grad_scaler.pth").exists()


def test_save_local_state_skips_optimizer_when_descriptor_has_none(tmp_path: Path, bcu_no_dist):
    descriptors = OrderedDict(
        [("block_0", _make_descriptor(with_optimizer=False, with_scaler=False))]
    )
    bcu_no_dist._save_local_state(descriptors, tmp_path)
    stitched = tmp_path / "stitched"
    assert not (stitched / "block_0.optimizer_state.pth").exists()
    assert not (stitched / "block_0.grad_scaler.pth").exists()


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
    monkeypatch.setattr(bcu_no_dist, "save_checkpoint", lambda **kwargs: None)
    return bcu_no_dist


def test_save_bypass_checkpoint_creates_latest_symlink_and_marker(tmp_path: Path, patched_save):
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    checkpoint_dir = experiment_dir / "iter-000007-ckpt"
    checkpoint_dir.mkdir()

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
    assert os.readlink(latest) == "iter-000007-ckpt"
    assert latest.resolve() == checkpoint_dir.resolve()
    assert (checkpoint_dir / "args.json").exists()
    assert (checkpoint_dir / "saving_completed").exists()


def test_save_bypass_checkpoint_replaces_existing_latest_symlink(tmp_path: Path, patched_save):
    """A stale ``latest`` from a prior save must be replaced, not appended to.
    Without ``unlink(missing_ok=True)`` the symlink_to() call would raise
    FileExistsError mid-save and leave the run unable to checkpoint."""
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    old_target = experiment_dir / "iter-000003-ckpt"
    old_target.mkdir()
    new_target = experiment_dir / "iter-000007-ckpt"
    new_target.mkdir()
    (experiment_dir / "latest").symlink_to(old_target.name)

    cfg = _make_save_cfg(experiment_dir)
    patched_save.save_bypass_checkpoint(
        cfg=cfg,
        descriptor=None,
        model=None,
        stitched_module_descriptors=OrderedDict(),
        checkpoint_dir=new_target,
    )

    assert os.readlink(experiment_dir / "latest") == "iter-000007-ckpt"


def test_save_bypass_checkpoint_master_only_skips_symlink_on_non_master(
    tmp_path: Path, monkeypatch, patched_save
):
    """Non-master ranks must not write the symlink, args.json, or marker —
    only rank 0 owns those files. The other ranks still call _save_local_state
    (their owned blocks) but stop short of the per-experiment metadata."""
    monkeypatch.setattr(patched_save.dist, "is_master", lambda: False)

    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    checkpoint_dir = experiment_dir / "iter-000007-ckpt"
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
