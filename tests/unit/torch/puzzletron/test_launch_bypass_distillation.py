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

"""Unit tests for ``launch_bypass_distillation`` (sweep dispatcher).

The dispatcher's job is to iterate over ``bypass.configs``, apply each override
to the live ``hydra_cfg``, reset the per-run state machine, and invoke
``run_bypassed_training``. Reordering or dropping a reset would silently make
the second sweep entry resume from the first entry's iter counter — a bug
that would only surface as wasted compute and confused checkpoint dirs.

We patch ``run_bypassed_training`` to a recorder so this stays a pure-Python
test (no GPU, no real training).
"""

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.amp.grad_scaler import GradScaler

import modelopt.torch.puzzletron.bypass_distillation.training_loop as tl


def _base_cfg(tmp_path, configs=None):
    """Build a minimal cfg shape that ``launch_bypass_distillation`` reads.

    Includes only the keys touched by the dispatcher itself; ``run_bypassed_training``
    is mocked so its richer requirements are irrelevant here.
    """
    cfg = {
        "puzzle_dir": str(tmp_path / "puzzletron_bypass_unit"),
        "descriptor": "test_descriptor",
        "bypass": {
            "model": {"model_config_overrides": {"intermediate_size": 1024}},
            "model_factory": {"keys_to_learn": "subblock_ffn"},
            "experiment_id": "stale-id",
            "iter_num": 999,
            "step_num": 999,
            "token_count": 999_999,
            "best_val_loss": 0.0,
            "training": {"clipping_count": 42},
        },
    }
    if configs is not None:
        cfg["bypass"]["configs"] = configs
    return OmegaConf.create(cfg)


def _record_calls(monkeypatch):
    """Patch ``run_bypassed_training`` to capture deep-copied cfg snapshots."""
    snapshots = []

    def _recorder(cfg):
        # Deep-copy via container conversion; the live cfg is mutated between calls.
        snapshots.append(OmegaConf.to_container(cfg, resolve=True))

    monkeypatch.setattr(tl, "run_bypassed_training", _recorder)
    return snapshots


def test_single_config_modes_run_once_without_reset(monkeypatch, tmp_path):
    """Absent and empty ``bypass.configs`` both use the single-config path."""
    for configs in (None, []):
        snapshots = _record_calls(monkeypatch)
        cfg = _base_cfg(tmp_path, configs=configs)
        tl.launch_bypass_distillation(cfg)
        assert len(snapshots) == 1
        # Single-config path doesn't touch the state machine.
        assert snapshots[0]["bypass"]["iter_num"] == 999
        assert snapshots[0]["bypass"]["training"]["clipping_count"] == 42


def test_sweep_configs_apply_overrides_reset_state_and_restore_base_keys(monkeypatch, tmp_path):
    """Each sweep entry gets its override, reset counters, and base keys fallback."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        tmp_path,
        configs=[
            {
                "model_config_overrides": {"intermediate_size": 256},
                "keys_to_learn": "subblock_attention",
            },
            {"model_config_overrides": {"intermediate_size": 128}},
        ],
    )
    tl.launch_bypass_distillation(cfg)

    assert len(snapshots) == 2
    assert snapshots[0]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 256}
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_attention"
    assert snapshots[1]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 128}
    assert snapshots[1]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_ffn"

    for snap, expected_prefix in zip(snapshots, ["bypass_attention_", "bypass_ffn_"], strict=True):
        assert snap["bypass"]["experiment_id"].startswith(expected_prefix)
        assert snap["bypass"]["iter_num"] == 1
        assert snap["bypass"]["step_num"] == 1
        assert snap["bypass"]["token_count"] == 0
        assert snap["bypass"]["best_val_loss"] == 1e9
        assert snap["bypass"]["training"]["clipping_count"] == 0


def test_resolve_trust_remote_code_requires_explicit_cfg_opt_in(monkeypatch):
    class DescriptorRequiringTrust:
        @staticmethod
        def requires_trust_remote_code():
            return True

    messages = []

    def capture_message(*args):
        messages.append(" ".join(map(str, args)))

    monkeypatch.setattr(tl, "mprint", capture_message)

    assert tl._resolve_trust_remote_code(OmegaConf.create({}), DescriptorRequiringTrust) is False
    assert any("trust_remote_code" in message for message in messages)
    messages.clear()

    assert (
        tl._resolve_trust_remote_code(
            OmegaConf.create({"trust_remote_code": True}), DescriptorRequiringTrust
        )
        is True
    )
    assert messages == []


def test_resume_state_path_prefers_explicit_init_checkpoint(monkeypatch):
    messages = []

    def capture_message(*args):
        messages.append(" ".join(map(str, args)))

    monkeypatch.setattr(tl, "mprint", capture_message)
    cfg = OmegaConf.create({"bypass": {"init_checkpoint_path": "/tmp/init-ckpt"}})

    assert tl._get_resume_state_path(cfg, "/tmp/resume-ckpt") is None
    assert any("init_checkpoint_path" in message for message in messages)

    cfg.bypass.init_checkpoint_path = None
    assert tl._get_resume_state_path(cfg, "/tmp/resume-ckpt") == "/tmp/resume-ckpt"


def test_flush_loss_buffer_single_rank_without_process_group():
    local_buffer = {1: {"block_0": 0.25}}
    stitched_losses_history = {}

    tl._flush_loss_buffer(local_buffer, stitched_losses_history)

    assert stitched_losses_history == local_buffer


def test_run_bypassed_training_skips_completed_runs_on_all_ranks(monkeypatch, tmp_path):
    for is_master in (True, False):
        cfg = _base_cfg(tmp_path)
        cfg.bypass.experiment_id = None
        checks = []
        broadcasts = []
        messages = []

        def fail(*args, **kwargs):
            raise AssertionError("training setup should not run after completed bypass check")

        def check_complete(cfg_arg):
            checks.append(cfg_arg)
            return True

        def broadcast(value, src):
            broadcasts.append((value, src))
            return True

        monkeypatch.setattr(tl.dist, "local_rank", lambda: 0)
        monkeypatch.setattr(tl.dist, "barrier", lambda: None)
        monkeypatch.setattr(tl.dist, "is_master", lambda: is_master)
        monkeypatch.setattr(tl.dist, "broadcast", broadcast)
        monkeypatch.setattr(tl, "bypass_run_is_complete", check_complete)
        monkeypatch.setattr(tl, "print_rank_0", lambda *args, **kwargs: messages.append(args[0]))
        monkeypatch.setattr(tl.ModelDescriptorFactory, "get", fail)

        tl.run_bypassed_training(cfg)

        if is_master:
            assert checks == [cfg]
            assert broadcasts == [(True, 0)]
        else:
            assert checks == []
            assert broadcasts == [(None, 0)]
        assert messages == [f"Bypass run {cfg.bypass.experiment_id} is already complete, skipping"]


def test_clip_stitched_module_grads_counts_only_clipped_blocks():
    for grad, grad_clip, grad_clip_type, expected_count, validate_grad in [
        (
            torch.full((1, 2), 10.0),
            0.1,
            "norm",
            1,
            lambda module: torch.linalg.vector_norm(module.weight.grad) <= 0.1 + 1e-6,
        ),
        (
            torch.tensor([[0.05, 2.0]]),
            0.5,
            "value",
            1,
            lambda module: module.weight.grad.abs().max() <= 0.5,
        ),
        (
            torch.full((1, 2), 0.01),
            1.0,
            "value",
            0,
            lambda module: torch.equal(module.weight.grad, torch.full_like(module.weight, 0.01)),
        ),
    ]:
        module = torch.nn.Linear(2, 1, bias=False)
        module.weight.grad = grad
        assert tl._clip_stitched_module_grads(module, grad_clip, grad_clip_type) == expected_count
        assert validate_grad(module)


def test_step_stitched_module_optimizer_unscales_before_clipping(monkeypatch):
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
    grad_scaler = GradScaler(device="cpu", enabled=True, init_scale=16.0)
    grad_scaler.scale(module.weight.sum() * 2.0).backward()
    assert module.weight.grad is not None
    assert torch.equal(module.weight.grad, torch.full_like(module.weight, 32.0))
    observed = {}

    def capture_clip(stitched_module, grad_clip, grad_clip_type):
        observed["stitched_module"] = stitched_module
        observed["grad_clip"] = grad_clip
        observed["grad_clip_type"] = grad_clip_type
        observed["grad"] = module.weight.grad.detach().clone()
        return 1

    monkeypatch.setattr(tl, "_clip_stitched_module_grads", capture_clip)

    clipped_count = tl._step_stitched_module_optimizer(
        stitched_module=module,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        grad_clip=1.0,
        grad_clip_type="norm",
    )

    assert clipped_count == 1
    assert observed["stitched_module"] is module
    assert observed["grad_clip"] == 1.0
    assert observed["grad_clip_type"] == "norm"
    assert torch.equal(observed["grad"], torch.full_like(module.weight, 2.0))
    assert module.weight.grad is None


def test_finalize_bypass_run_marks_completion_only_after_realization(monkeypatch):
    monkeypatch.setattr(tl.dist, "is_master", lambda: True)

    cfg = OmegaConf.create({"bypass": {"disable_checkpoint_save": True}})

    def fail(*args, **kwargs):
        raise AssertionError("checkpoint realization should be skipped")

    monkeypatch.setattr(tl, "realize_bypass_checkpoints", fail)
    monkeypatch.setattr(tl, "mark_bypass_run_completed", fail)
    tl._finalize_bypass_run(cfg)

    completed = {}
    cfg = OmegaConf.create({"bypass": {"disable_checkpoint_save": False}})
    monkeypatch.setattr(
        tl, "realize_bypass_checkpoints", lambda _cfg: (_ for _ in ()).throw(FileNotFoundError)
    )
    monkeypatch.setattr(tl, "mark_bypass_run_completed", lambda *args: completed.update(hit=True))

    tl._finalize_bypass_run(cfg)

    assert completed == {}

    realized = Path("/tmp/realized")
    symlink = Path("/tmp/ckpts/run_0")
    monkeypatch.setattr(tl, "realize_bypass_checkpoints", lambda _cfg: (realized, symlink))
    monkeypatch.setattr(
        tl,
        "mark_bypass_run_completed",
        lambda cfg_arg, realized_arg, symlink_arg: completed.update(
            cfg=cfg_arg, realized=realized_arg, symlink=symlink_arg
        ),
    )

    tl._finalize_bypass_run(cfg)

    assert completed == {"cfg": cfg, "realized": realized, "symlink": symlink}


def test_realize_bypass_checkpoints_uses_resolved_symlink_target(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    experiment_dir = Path("puzzle/bypass/bypass_runs/run_0")
    checkpoint_dir = experiment_dir / "final-step-000002-ckpt"
    checkpoint_dir.mkdir(parents=True)
    (experiment_dir / "bypass_state.json").write_text(
        json.dumps({"checkpoints": {"final": str(checkpoint_dir)}})
    )
    cfg = OmegaConf.create(
        {
            "puzzle_dir": "puzzle",
            "bypass": {
                "experiment_dir": str(experiment_dir),
                "experiment_id": "run_0",
                "realize_best_or_latest": "latest",
            },
        }
    )

    realized_checkpoint, ckpts_symlink = tl.realize_bypass_checkpoints(cfg)

    assert realized_checkpoint == checkpoint_dir.resolve()
    assert ckpts_symlink.readlink() == checkpoint_dir.resolve()
    assert ckpts_symlink.exists()
