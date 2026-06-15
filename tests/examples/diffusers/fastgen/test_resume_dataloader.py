# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression test for the DMD2 mid-run resume data-position fix.

On resume the ``StatefulDataLoader``'s restored state does not advance past the resume
point, so the previous recipe re-served the same data slice every window (data
under-coverage). The fix, ``DMD2DiffusionRecipe._rebuild_dataloader_for_resume``,
rebuilds a fresh loader and skips the deterministic sampler to the position implied by
``global_step``.

This drives that method through the REAL ``SequentialBucketSampler`` +
``StatefulDataLoader`` on a tiny in-memory dataset (no GPU, no SLURM, no data cache,
milliseconds) and asserts the first sample served after a resume equals what a clean
no-resume run serves at that global step -- including across epoch boundaries, where the
per-epoch reshuffle matters. The previous loader-state resume re-served a stale sample,
so this assertion fails on the old code (and the method did not exist at all).

Dependency-guarded with ``importorskip`` so it skips where torch / nemo_automodel /
torchdata are absent (e.g. a CPU login node) and runs in the training container.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest

# Put the example dir on sys.path so ``dmd2_recipe`` imports as a top-level module,
# exactly as dmd2_finetune.py does (mirrors test_vendored_migration.py).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

_N = 20  # samples (== batches, batch_size 1) per epoch in the synthetic dataset


class _Dataset:
    """Minimal map-style dataset matching SequentialBucketSampler's expectations (one bucket)."""

    def __init__(self, n: int):
        self.bucket_groups = {(64, 64): {"indices": list(range(n)), "resolution": (64, 64)}}
        self.sorted_bucket_keys = [(64, 64)]
        self.calculator = None
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return int(i)  # identity: the served value IS the global sample index


def _build(n, sampler_cls, loader_cls):
    """A real sampler + StatefulDataLoader over one shared synthetic dataset."""
    ds = _Dataset(n)
    sampler = sampler_cls(
        ds,
        base_batch_size=1,
        base_resolution=(64, 64),
        drop_last=False,
        shuffle_buckets=True,
        shuffle_within_bucket=True,
        dynamic_batch_size=False,
        seed=42,
        num_replicas=1,
        rank=0,
    )
    loader = loader_cls(ds, batch_sampler=sampler, collate_fn=lambda b: b, num_workers=0)
    return sampler, loader


def test_resume_rebuild_serves_clean_run_position(monkeypatch):
    """The recipe's resume reset serves the SAME sample a clean no-resume run serves at that step.

    Drives ``DMD2DiffusionRecipe._rebuild_dataloader_for_resume`` over a real sampler/loader.
    Fails on the previous code: the method did not exist, and its loader-state resume
    re-served a stale sample instead of the ``global_step``-correct one.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nemo_automodel")
    pytest.importorskip("torchdata")
    from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    dmd2_recipe = pytest.importorskip("dmd2_recipe")
    # The reset logs only on the main process; force True off the distributed path.
    monkeypatch.setattr(dmd2_recipe, "is_main_process", lambda: True, raising=False)

    n = _N

    # Reference: the deterministic sample order of a clean, no-resume run over 3 epochs.
    ref_sampler, ref_loader = _build(n, SequentialBucketSampler, StatefulDataLoader)
    clean = []
    for epoch in range(3):
        ref_sampler.set_epoch(epoch)
        clean.extend(batch[0] for batch in ref_loader)
    assert len(clean) == 3 * n, "synthetic epoch_len mismatch"
    assert len(set(clean)) == n, "each epoch must fully cover the dataset"

    # Mid-epoch, epoch-boundary, and cross-epoch resume points.
    for global_step in (1, 5, n - 1, n, n + 3, 2 * n, 2 * n + 7):
        sampler, loader = _build(n, SequentialBucketSampler, StatefulDataLoader)
        stub = SimpleNamespace(
            dataloader=loader,
            sampler=sampler,
            step_scheduler=SimpleNamespace(epoch_len=n, grad_acc_steps=1, epoch=0),
        )

        dmd2_recipe.DMD2DiffusionRecipe._rebuild_dataloader_for_resume(stub, global_step)

        # Position was derived from global_step (also drives the epoch label + progress bar).
        assert stub.step_scheduler.epoch == global_step // n
        assert stub.sampler._batches_to_skip == global_step % n

        first = next(iter(stub.dataloader))[0]
        assert first == clean[global_step], (
            f"resume@{global_step}: served {first}, a clean run serves {clean[global_step]} "
            "(re-serving / wrong-position bug)"
        )


def test_resume_reset_is_noop_on_fresh_start(monkeypatch):
    """global_step == 0 (fresh start) must NOT rebuild/skip -- the first window is the clean run."""
    pytest.importorskip("torch")
    pytest.importorskip("nemo_automodel")
    pytest.importorskip("torchdata")
    from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    dmd2_recipe = pytest.importorskip("dmd2_recipe")
    monkeypatch.setattr(dmd2_recipe, "is_main_process", lambda: True, raising=False)

    sampler, loader = _build(_N, SequentialBucketSampler, StatefulDataLoader)
    stub = SimpleNamespace(
        dataloader=loader,
        sampler=sampler,
        step_scheduler=SimpleNamespace(epoch_len=_N, grad_acc_steps=1, epoch=0),
    )
    dmd2_recipe.DMD2DiffusionRecipe._rebuild_dataloader_for_resume(stub, 0)

    assert stub.dataloader is loader, "fresh start must not rebuild the dataloader"
    assert getattr(stub.sampler, "_batches_to_skip", 0) == 0
    assert stub.step_scheduler.epoch == 0
