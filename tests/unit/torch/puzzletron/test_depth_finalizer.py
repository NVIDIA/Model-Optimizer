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

"""Tests for distributed depth-importance finalization."""

import json

import pytest

import examples.puzzletron.finalize_depth_importance as depth_finalizer


def test_depth_finalizer_publishes_terminal_manifest(tmp_path, monkeypatch):
    config_path = tmp_path / "experiment.yaml"
    config_path.touch()
    output_dir = tmp_path / "depth" / "iterative"
    output_dir.mkdir(parents=True)
    selected = [
        {"kind": "attention", "layer_idx": 1},
        {"kind": "mamba", "layer_idx": 2},
    ]
    (output_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "max_removals": 2,
                "selected": selected,
                "scenarios": [{}, {}, {}],
            }
        )
    )
    config = {
        "model": {"path": "tiny-model"},
        "depth_importance": {"enabled": True, "max_removals": 2},
    }
    loaded_overrides = []

    def load_config(path, *, overrides=None):
        assert path == config_path
        loaded_overrides.extend(overrides or ())
        return config

    monkeypatch.setattr(depth_finalizer, "pipeline_config_from_path", load_config)

    outputs = depth_finalizer.finalize_depth_importance(
        config_path,
        tmp_path,
        output_dir,
        overrides=["++depth_importance.micro_batch_size=1"],
    )

    manifest = json.loads((tmp_path / "manifests" / "depth_importance.json").read_text())
    assert loaded_overrides == ["++depth_importance.micro_batch_size=1"]
    assert outputs == {
        "trajectory_path": str(output_dir / "trajectory.json"),
        "scenario_count": 3,
        "selected": selected,
    }
    assert manifest["stage"] == "depth_importance"
    assert manifest["status"] == "success"
    assert manifest["outputs"] == outputs


def test_depth_finalizer_rejects_incomplete_trajectory(tmp_path, monkeypatch):
    config_path = tmp_path / "experiment.yaml"
    output_dir = tmp_path / "depth"
    output_dir.mkdir()
    (output_dir / "trajectory.json").write_text(
        json.dumps({"status": "complete", "max_removals": 2, "selected": [{}]})
    )
    monkeypatch.setattr(
        depth_finalizer,
        "pipeline_config_from_path",
        lambda _path, *, overrides=None: {"depth_importance": {"max_removals": 2}},
    )

    with pytest.raises(RuntimeError, match="depth trajectory is incomplete"):
        depth_finalizer.finalize_depth_importance(config_path, tmp_path, output_dir)
