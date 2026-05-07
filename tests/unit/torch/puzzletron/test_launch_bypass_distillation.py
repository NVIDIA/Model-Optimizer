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

from omegaconf import OmegaConf

import modelopt.torch.puzzletron.bypass_distillation.training_loop as tl


def _base_cfg(configs=None):
    """Build a minimal cfg shape that ``launch_bypass_distillation`` reads.

    Includes only the keys touched by the dispatcher itself; ``run_bypassed_training``
    is mocked so its richer requirements are irrelevant here.
    """
    cfg = {
        "bypass": {
            "model": {"model_config_overrides": {"intermediate_size": 1024}},
            "model_factory": {"keys_to_learn": "subblock_ffn"},
            "experiment_id": "stale-id",
            "iter_num": 999,
            "step_num": 999,
            "token_count": 999_999,
            "best_val_loss": 0.0,
            "training": {"clipping_count": 42},
        }
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


def test_no_configs_key_runs_once(monkeypatch):
    """Absent ``bypass.configs`` is the single-config path — one call, no resets."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(configs=None)
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 1
    # Single-config path doesn't touch the state machine — values remain as supplied.
    assert snapshots[0]["bypass"]["iter_num"] == 999
    assert snapshots[0]["bypass"]["training"]["clipping_count"] == 42


def test_empty_configs_list_runs_once(monkeypatch):
    """``configs: []`` must hit the same branch as missing — the truthiness
    check on line 85 of training_loop.py treats both as 'no sweep'."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(configs=[])
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 1


def test_two_configs_run_twice_with_distinct_overrides(monkeypatch):
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        configs=[
            {"model_config_overrides": {"intermediate_size": 256}},
            {"model_config_overrides": {"intermediate_size": 128}},
        ]
    )
    tl.launch_bypass_distillation(cfg)
    assert len(snapshots) == 2
    assert snapshots[0]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 256}
    assert snapshots[1]["bypass"]["model"]["model_config_overrides"] == {"intermediate_size": 128}


def test_keys_to_learn_override_applied(monkeypatch):
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(configs=[{"keys_to_learn": "subblock_attention"}])
    tl.launch_bypass_distillation(cfg)
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_attention"


def test_per_run_state_reset_before_each_call(monkeypatch):
    """Every sweep entry must see iter_num=1, step_num=1, token_count=0,
    best_val_loss=1e9, clipping_count=0, experiment_id=None — even when the
    previous entry left the cfg in some other state."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(
        configs=[
            {"model_config_overrides": {"intermediate_size": 256}},
            {"model_config_overrides": {"intermediate_size": 128}},
        ]
    )
    tl.launch_bypass_distillation(cfg)
    for snap in snapshots:
        assert snap["bypass"]["experiment_id"] is None
        assert snap["bypass"]["iter_num"] == 1
        assert snap["bypass"]["step_num"] == 1
        assert snap["bypass"]["token_count"] == 0
        assert snap["bypass"]["best_val_loss"] == 1e9
        assert snap["bypass"]["training"]["clipping_count"] == 0


def test_override_without_keys_to_learn_leaves_cfg_value_untouched(monkeypatch):
    """A sweep entry that only sets ``model_config_overrides`` must not clobber
    the inherited ``keys_to_learn`` (the dispatcher's `if "keys_to_learn" in override`
    guard, line 99)."""
    snapshots = _record_calls(monkeypatch)
    cfg = _base_cfg(configs=[{"model_config_overrides": {"intermediate_size": 256}}])
    tl.launch_bypass_distillation(cfg)
    # keys_to_learn was set to "subblock_ffn" in _base_cfg — must survive.
    assert snapshots[0]["bypass"]["model_factory"]["keys_to_learn"] == "subblock_ffn"
