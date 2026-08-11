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

"""Tests for tokenize_data cache resolution and stage execution."""

import json
from pathlib import Path

import pytest

from examples.puzzletron.main import _validate_worker_result
from examples.puzzletron.tokenize_data import tokenize_data_stage
from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.orchestration.config import load_experiment_config
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from puzzletron_orchestrator.token_caches import resolve_tokenize_caches


def test_resolve_tokenize_caches_uses_explicit_entries():
    caches = resolve_tokenize_caches(
        {
            "tokenize_data": {
                "caches": [
                    {
                        "output": "/tmp/train.tokens",
                        "split": "train",
                        "num_samples": 8,
                        "seq_length": 16,
                        "shuffle_seed": 1,
                    }
                ]
            },
            "train_token_cache_path": "/tmp/ignored.tokens",
        }
    )
    assert len(caches) == 1
    assert caches[0]["output"] == "/tmp/train.tokens"


def test_resolve_tokenize_caches_defaults_from_campaign_paths():
    caches = resolve_tokenize_caches(
        {
            "train_token_cache_path": "/tmp/train.tokens",
            "validation_token_cache_path": "/tmp/validation.tokens",
            "data": {
                "max_sample_length": 4096,
                "calibration": {"num_samples": 32768, "seq_len": 4096},
                "replacement_scoring": {"num_samples": 128},
            },
            "replacement_scoring": {"eval_samples": 192},
            "depth_importance": {"eval_samples": 256},
            "sort_sanity": {"enabled": False, "eval_samples": 4096},
            "width_sanity": {"eval_samples": 512},
            "pruning": {"shuffle_seed": 444},
            "tokenize_data": {"enabled": True, "caches": []},
        }
    )
    assert [cache["split"] for cache in caches] == ["train", "validation"]
    assert caches[0]["num_samples"] == 32768
    assert caches[0]["seq_length"] == 4096
    assert caches[1]["num_samples"] == 512


def test_resolve_tokenize_caches_preserves_zero_shuffle_seed():
    caches = resolve_tokenize_caches(
        {
            "train_token_cache_path": "/tmp/train.tokens",
            "validation_token_cache_path": "/tmp/validation.tokens",
            "pruning": {"shuffle_seed": 0},
        }
    )

    assert [cache["shuffle_seed"] for cache in caches] == [0, 1]


def test_tokenize_data_stage_dispatches_derived_train_and_validation_caches(
    tmp_path, monkeypatch, write_token_cache
):
    commands = []
    config = _tokenize_stage_config(tmp_path)
    resolved = resolve_tokenize_caches(config)

    def _run(command, *, check):
        assert check is True
        commands.append(tuple(command))
        output = Path(command[command.index("--output") + 1])
        cache = next(cache for cache in resolved if Path(cache["output"]) == output)
        write_token_cache(config, cache)

    monkeypatch.setattr("examples.puzzletron.tokenize_data.subprocess.run", _run)
    result = tokenize_data_stage(config)

    assert result.status == "success"
    assert [command[command.index("--split") + 1] for command in commands] == [
        "train",
        "validation",
    ]
    assert [command[command.index("--shuffle-seed") + 1] for command in commands] == [
        "0",
        "1",
    ]
    assert all("--trust-remote-code" not in command for command in commands)
    assert stage_is_complete(config, "tokenize_data")


def test_tokenize_data_manifest_accepts_equivalent_controller_and_worker_configs(
    tmp_path, monkeypatch, write_token_cache
):
    output = tmp_path / "dataset_cache" / "train.tokens"
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""\
defaults: [_self_]
puzzle_dir: {tmp_path}
dataset_path: {tmp_path / "dataset"}
model:
  source: {tmp_path / "model"}
  trust_remote_code: false
convert:
  teacher_dir: {tmp_path / "teacher"}
data:
  modality: text
  layout: fixed
  max_sample_length: 8
search_space:
  axes:
    hidden_width:
      enabled: true
      values: [256]
sort_sanity:
  enabled: false
width_sanity:
  enabled: false
tokenize_data:
  enabled: true
  workers: 1
  caches:
    - output: {output}
      split: train
      num_samples: 1
      seq_length: 8
      shuffle_seed: 1
"""
    )
    controller_config = load_experiment_config(experiment)
    worker_config = pipeline_config_from_path(experiment)

    assert worker_config["sort_sanity"]["include_reverse"] is True
    assert worker_config["width_sanity"]["target_values"] == {"hidden_width": 256}
    assert "include_reverse" not in controller_config["sort_sanity"]
    assert "target_values" not in controller_config["width_sanity"]

    caches = resolve_tokenize_caches(worker_config)

    def _run(command, *, check):
        assert check is True
        cache_output = Path(command[command.index("--output") + 1])
        cache = next(cache for cache in caches if Path(cache["output"]) == cache_output)
        write_token_cache(worker_config, cache)

    monkeypatch.setattr("examples.puzzletron.tokenize_data.subprocess.run", _run)
    result = tokenize_data_stage(worker_config)

    manifest = json.loads((tmp_path / "manifests" / "tokenize_data.json").read_text())
    assert manifest["semantic_config"]["sort_sanity"] == {"enabled": False}
    assert manifest["semantic_config"]["width_sanity"] == {"enabled": False}
    resolved_path = tmp_path / manifest["execution_record"]["resolved_config_path"]
    resolved_config = json.loads(resolved_path.read_text())["resolved_stage_config"]
    assert resolved_config["sort_sanity"]["include_reverse"] is True
    assert resolved_config["width_sanity"]["target_values"] == {"hidden_width": 256}
    _validate_worker_result(worker_config, result, expected_stage="tokenize_data")
    assert stage_is_complete(worker_config, "tokenize_data")
    assert stage_is_complete(controller_config, "tokenize_data")
    controller_config["sort_sanity"]["enabled"] = True
    assert not stage_is_complete(controller_config, "tokenize_data")


def test_tokenize_data_stage_passes_trust_remote_code_only_when_enabled(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        "examples.puzzletron.tokenize_data.subprocess.run",
        lambda command, *, check: commands.append(tuple(command)),
    )
    config = _tokenize_stage_config(tmp_path)
    config["model"] = {"trust_remote_code": True}
    config.pop("validation_token_cache_path")

    tokenize_data_stage(config)

    assert len(commands) == 1
    assert "--trust-remote-code" in commands[0]


def test_tokenize_data_stage_rejects_enabled_stage_without_resolvable_cache(tmp_path):
    config = {
        "puzzle_dir": str(tmp_path),
        "tokenize_data": {"enabled": True, "caches": []},
    }

    with pytest.raises(ValueError, match="no caches are configured"):
        tokenize_data_stage(config)


def _tokenize_stage_config(tmp_path):
    return {
        "puzzle_dir": str(tmp_path),
        "dataset_path": str(tmp_path / "dataset"),
        "train_token_cache_path": str(tmp_path / "train.tokens"),
        "validation_token_cache_path": str(tmp_path / "validation.tokens"),
        "data": {
            "calibration": {"num_samples": 2, "seq_len": 8},
            "replacement_scoring": {"num_samples": 1},
        },
        "pruning": {"shuffle_seed": 0},
        "convert": {"teacher_dir": str(tmp_path / "teacher")},
        "tokenize_data": {"enabled": True, "workers": 2},
        "width_sanity": {"eval_samples": 2},
    }
