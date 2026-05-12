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

from omegaconf import OmegaConf

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


def test_no_configs_key_runs_once(monkeypatch, tmp_path):
    """Absent ``bypass.configs`` is the single-config path — one call, no resets."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(tmp_path, configs=None)
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 1
    # Single-config path doesn't touch the state machine — values remain as supplied.
    assert snapshots[0]["bypass"]["iter_num"] == 999
    assert snapshots[0]["bypass"]["training"]["clipping_count"] == 42


def test_empty_configs_list_runs_once(monkeypatch, tmp_path):
    """``configs: []`` must hit the same branch as missing — the truthiness
    check on ``bypass.configs`` treats both as 'no sweep'."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(tmp_path, configs=[])
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 1


def test_two_configs_run_twice_with_distinct_overrides(monkeypatch, tmp_path):
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        tmp_path,
        configs=[
            {"model_config_overrides": {"intermediate_size": 256}},
            {"model_config_overrides": {"intermediate_size": 128}},
        ],
    )
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 2
    assert snapshots[0]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 256}
    assert snapshots[1]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 128}


def test_keys_to_learn_override_applied(monkeypatch, tmp_path):
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(tmp_path, configs=[{"keys_to_learn": "subblock_attention"}])
    tl.launch_bypass_distillation(cfg)
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_attention"


def test_per_run_state_reset_before_each_call(monkeypatch, tmp_path):
    """Every sweep entry must see iter_num=1, step_num=1, token_count=0,
    best_val_loss=1e9, clipping_count=0, and a fresh experiment_id even when the
    previous entry left the cfg in some other state."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        tmp_path,
        configs=[
            {"model_config_overrides": {"intermediate_size": 256}},
            {"model_config_overrides": {"intermediate_size": 128}},
        ],
    )
    tl.launch_bypass_distillation(cfg)
    for snap in snapshots:
        assert snap["bypass"]["experiment_id"].startswith("bypass_ffn_")
        assert snap["bypass"]["iter_num"] == 1
        assert snap["bypass"]["step_num"] == 1
        assert snap["bypass"]["token_count"] == 0
        assert snap["bypass"]["best_val_loss"] == 1e9
        assert snap["bypass"]["training"]["clipping_count"] == 0


def test_override_without_keys_to_learn_leaves_cfg_value_untouched(monkeypatch, tmp_path):
    """A sweep entry that only sets ``model_config_overrides`` must not clobber
    the inherited ``keys_to_learn`` (the dispatcher's `if "keys_to_learn" in override`
    guard)."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(tmp_path, configs=[{"model_config_overrides": {"intermediate_size": 256}}])
    tl.launch_bypass_distillation(cfg)
    # keys_to_learn was set to "subblock_ffn" in _base_cfg — must survive.
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_ffn"


def test_sweep_entry_without_keys_to_learn_uses_base_not_previous_override(monkeypatch, tmp_path):
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        tmp_path,
        configs=[
            {"keys_to_learn": "subblock_attention"},
            {"model_config_overrides": {"intermediate_size": 256}},
        ],
    )
    tl.launch_bypass_distillation(cfg)
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_attention"
    assert snapshots[1]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_ffn"


def test_trust_remote_code_defaults_to_false_even_when_descriptor_requires_it(monkeypatch):
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


def test_trust_remote_code_uses_explicit_cfg_opt_in(monkeypatch):
    class DescriptorRequiringTrust:
        @staticmethod
        def requires_trust_remote_code():
            return True

    messages = []

    def capture_message(*args):
        messages.append(" ".join(map(str, args)))

    monkeypatch.setattr(tl, "mprint", capture_message)

    cfg = OmegaConf.create({"trust_remote_code": True})
    assert tl._resolve_trust_remote_code(cfg, DescriptorRequiringTrust) is True
    assert messages == []


def test_resume_state_ignored_when_init_checkpoint_path_wins(monkeypatch):
    messages = []

    def capture_message(*args):
        messages.append(" ".join(map(str, args)))

    monkeypatch.setattr(tl, "mprint", capture_message)
    cfg = OmegaConf.create({"bypass": {"init_checkpoint_path": "/tmp/init-ckpt"}})

    assert tl._get_resume_state_path(cfg, "/tmp/resume-ckpt") is None
    assert any("init_checkpoint_path" in message for message in messages)


def test_resume_state_used_when_no_init_checkpoint_path():
    cfg = OmegaConf.create({"bypass": {"init_checkpoint_path": None}})

    assert tl._get_resume_state_path(cfg, "/tmp/resume-ckpt") == "/tmp/resume-ckpt"


def test_flush_loss_buffer_single_rank_without_process_group():
    local_buffer = {1: {"block_0": 0.25}}
    stitched_losses_history = {}

    tl._flush_loss_buffer(local_buffer, stitched_losses_history)

    assert stitched_losses_history == local_buffer


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
