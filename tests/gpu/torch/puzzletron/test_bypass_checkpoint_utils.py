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

"""Save/load round-trip tests for ``bypass_checkpoint_utils``.

These tests pin two correctness-critical pieces of ``bypass_checkpoint_utils``:

1. ``_save_local_state`` persists the GradScaler state alongside the optimizer
   state (regression coverage for the recent CodeRabbit-driven fix — without
   it, fp16 + use_grad_scaling=True runs silently lost the running scale +
   growth tracker on resume).
2. ``load_local_state`` restores it from disk.

Lives under ``tests/gpu/`` because the production ``load_local_state`` builds
``torch.device(f"cuda:{rank}")`` for ``map_location``, so a real CUDA device
is required to round-trip ``torch.load`` without monkeypatching the device
machinery. The full bypass GPU integration test cannot cover this path
because the test infrastructure ships bf16 and ``GradScaler.step()`` is
fp16-only (raises ``NotImplementedError:
_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'``).
These tests sidestep that by hitting the save/load functions directly,
without ever invoking ``.step()``.
"""

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler

from modelopt.torch.puzzletron.bypass_distillation import bypass_checkpoint_utils as bcu
from modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory import (
    StitchedModuleDescriptor,
)


# ---------------------------------------------------------------------------
# Fixture: silence the dist helpers so the save/load functions run on a
# single GPU process without `torchrun` / NCCL setup.
# ---------------------------------------------------------------------------


@pytest.fixture
def bcu_no_dist(monkeypatch):
    """Mock the dist helpers so ``bypass_checkpoint_utils`` runs without distributed init."""
    monkeypatch.setattr(bcu.dist, "local_rank", lambda: 0)
    monkeypatch.setattr(bcu.dist, "is_master", lambda: True)
    monkeypatch.setattr(bcu.dist, "barrier", lambda: None)
    return bcu


def _make_descriptor(
    *,
    with_optimizer: bool = True,
    with_scaler: bool = True,
    grad_scaler_init_scale: float = 2.0**16,
):
    """Build a minimal StitchedModuleDescriptor on CPU.

    ``stitched_module`` is a real ``nn.Linear`` so ``state_dict()`` /
    ``load_state_dict()`` work without needing the actual ``StitchedModule``
    machinery (which depends on the sewing-kit graph, distributed init, etc.).

    The GradScaler is created with ``enabled=True`` so that ``state_dict()``
    actually contains content (a disabled scaler returns ``{}``, making
    round-trip tests vacuous). We never call ``.scale()`` / ``.step()`` so
    none of the fp16-only kernels run — only the bookkeeping fields
    (``scale``, ``growth_factor``, ``backoff_factor``, ``growth_interval``,
    ``_growth_tracker``) go through save/load.
    """
    module = nn.Linear(4, 4, bias=False)
    owned_parameters = dict(module.named_parameters())
    owned_buffers: dict[str, torch.Tensor] = {}
    optimizer = (
        torch.optim.AdamW(list(module.parameters()), lr=1e-3) if with_optimizer else None
    )
    scaler = (
        GradScaler(device="cpu", enabled=True, init_scale=grad_scaler_init_scale)
        if with_scaler
        else None
    )
    return StitchedModuleDescriptor(
        stitched_module=module,
        owned_parameters=owned_parameters,
        owned_buffers=owned_buffers,
        optimizer=optimizer,
        grad_scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Save: every relevant artifact lands on disk
# ---------------------------------------------------------------------------


def test_save_local_state_writes_state_dict_optimizer_and_grad_scaler(
    tmp_path: Path, bcu_no_dist
):
    bcu = bcu_no_dist
    descriptor = _make_descriptor()
    descriptors = OrderedDict([("block_0", descriptor)])

    bcu._save_local_state(descriptors, tmp_path)

    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.state_dict.pth").exists()
    assert (stitched / "block_0.optimizer_state.pth").exists()
    # The CodeRabbit-driven fix added this third file. Without it, resuming
    # an fp16 + grad-scaling run would default-init the scaler.
    assert (stitched / "block_0.grad_scaler.pth").exists()


def test_save_local_state_skips_grad_scaler_when_descriptor_has_none(
    tmp_path: Path, bcu_no_dist
):
    bcu = bcu_no_dist
    descriptor = _make_descriptor(with_scaler=False)
    descriptors = OrderedDict([("block_0", descriptor)])

    bcu._save_local_state(descriptors, tmp_path)

    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.state_dict.pth").exists()
    # No scaler in the descriptor → no .grad_scaler.pth file written.
    assert not (stitched / "block_0.grad_scaler.pth").exists()


def test_save_local_state_skips_optimizer_when_descriptor_has_none(
    tmp_path: Path, bcu_no_dist
):
    """Pipeline-parallel idle ranks pass optimizer=None; no file should appear."""
    bcu = bcu_no_dist
    descriptor = _make_descriptor(with_optimizer=False, with_scaler=False)
    descriptors = OrderedDict([("block_0", descriptor)])

    bcu._save_local_state(descriptors, tmp_path)

    stitched = tmp_path / "stitched"
    assert (stitched / "block_0.state_dict.pth").exists()
    assert not (stitched / "block_0.optimizer_state.pth").exists()


# ---------------------------------------------------------------------------
# Load: state survives the round-trip and lands back on the live scaler
# ---------------------------------------------------------------------------


def test_load_local_state_restores_grad_scaler_state(tmp_path: Path, bcu_no_dist):
    """Round-trip: scaler with non-default init_scale → save → load into fresh scaler → state matches.

    This is the regression test for the CodeRabbit-flagged bug: prior to the
    fix, ``load_local_state`` skipped the scaler entirely, so a resumed run
    would silently start with a default scale (typically 65536.0) regardless
    of where the previous run had grown the scale to.

    We compare via ``state_dict()`` rather than poking at private attributes
    because the canonical save/load contract is ``state_dict()`` <->
    ``load_state_dict()``; ``state_dict()['scale']`` is the field a real
    bypass run would have grown over time.
    """
    bcu = bcu_no_dist

    # 1. Save phase: scaler with a non-default init scale.
    save_descriptor = _make_descriptor(grad_scaler_init_scale=12345.0)
    saved_state = save_descriptor.grad_scaler.state_dict()
    assert saved_state["scale"] == 12345.0  # sanity: state actually carries the value
    descriptors_save = OrderedDict([("block_0", save_descriptor)])
    bcu._save_local_state(descriptors_save, tmp_path)

    # 2. Load phase: a fresh descriptor with a different init scale; the load
    #    must overwrite it with the saved value.
    load_descriptor = _make_descriptor(grad_scaler_init_scale=999.0)
    pre_load_state = load_descriptor.grad_scaler.state_dict()
    assert pre_load_state != saved_state  # sanity: starts in a distinct state
    descriptors_load = OrderedDict([("block_0", load_descriptor)])
    bcu.load_local_state(descriptors_load, tmp_path)

    assert load_descriptor.grad_scaler.state_dict() == saved_state


def test_load_local_state_handles_legacy_checkpoint_without_grad_scaler(
    tmp_path: Path, bcu_no_dist
):
    """Backward compat: a checkpoint saved before the GradScaler-fix must still load.

    Older bypass runs predating the GradScaler save did not write
    ``block_0.grad_scaler.pth``. The current ``load_local_state`` must skip
    silently in that case rather than raising — our deployed users have
    legacy checkpoints they want to resume from.
    """
    bcu = bcu_no_dist

    # First save with a scaler so we have a normal "complete" save…
    save_descriptor = _make_descriptor()
    descriptors_save = OrderedDict([("block_0", save_descriptor)])
    bcu._save_local_state(descriptors_save, tmp_path)
    # …then delete the grad_scaler artifact to mimic a legacy checkpoint.
    (tmp_path / "stitched" / "block_0.grad_scaler.pth").unlink()

    # Loading must not raise.
    load_descriptor = _make_descriptor()
    descriptors_load = OrderedDict([("block_0", load_descriptor)])
    bcu.load_local_state(descriptors_load, tmp_path)


def test_load_local_state_restores_optimizer_state(tmp_path: Path, bcu_no_dist):
    """End-to-end optimizer round-trip — covers the resume path's main job."""
    bcu = bcu_no_dist

    save_descriptor = _make_descriptor()
    # Take an optimizer step so AdamW has non-default ``state`` (exp_avg etc).
    for p in save_descriptor.stitched_module.parameters():
        p.grad = torch.ones_like(p)
    save_descriptor.optimizer.step()
    saved_state = save_descriptor.optimizer.state_dict()
    descriptors_save = OrderedDict([("block_0", save_descriptor)])
    bcu._save_local_state(descriptors_save, tmp_path)

    load_descriptor = _make_descriptor()
    # Fresh optimizer's state dict should differ from `saved_state` until load.
    assert load_descriptor.optimizer.state_dict() != saved_state
    descriptors_load = OrderedDict([("block_0", load_descriptor)])
    bcu.load_local_state(descriptors_load, tmp_path)

    # After load, AdamW step counter and exp_avg buffers must match.
    # Production runs co-locate model + state on cuda:0, but this fixture has the
    # model on CPU so the loaded state ends up split: exp_avg / exp_avg_sq follow
    # the param device (CPU), while AdamW's `step` tensor is loaded via
    # ``map_location='cuda:0'`` and stays there. Move both to CPU for the
    # comparison — we're verifying value equality, not device placement.
    loaded_state = load_descriptor.optimizer.state_dict()
    assert loaded_state["state"].keys() == saved_state["state"].keys()
    for param_id in loaded_state["state"]:
        for key, val in saved_state["state"][param_id].items():
            loaded_val = loaded_state["state"][param_id][key]
            if torch.is_tensor(val):
                assert torch.equal(loaded_val.to("cpu"), val.to("cpu")), (
                    f"optimizer.state[{param_id}][{key}] not restored"
                )
            else:
                assert loaded_val == val
