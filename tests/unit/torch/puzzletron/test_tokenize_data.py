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

import pytest

from examples.puzzletron.tokenize_data import resolve_tokenize_caches, tokenize_data_stage


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
            "pruning": {"shuffle_seed": 444},
            "tokenize_data": {"enabled": True, "caches": []},
        }
    )
    assert [cache["split"] for cache in caches] == ["train", "validation"]
    assert caches[0]["num_samples"] == 32768
    assert caches[0]["seq_length"] == 4096
    assert caches[1]["num_samples"] == 128


def test_resolve_tokenize_caches_preserves_zero_shuffle_seed():
    caches = resolve_tokenize_caches(
        {
            "train_token_cache_path": "/tmp/train.tokens",
            "validation_token_cache_path": "/tmp/validation.tokens",
            "pruning": {"shuffle_seed": 0},
        }
    )

    assert [cache["shuffle_seed"] for cache in caches] == [0, 1]


def test_tokenize_data_stage_dispatches_derived_train_and_validation_caches(tmp_path, monkeypatch):
    commands = []

    def _run(command, *, check):
        assert check is True
        commands.append(tuple(command))

    monkeypatch.setattr("examples.puzzletron.tokenize_data.subprocess.run", _run)
    result = tokenize_data_stage(_tokenize_stage_config(tmp_path))

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
        "pruning": {"shuffle_seed": 0},
        "convert": {"teacher_dir": str(tmp_path / "teacher")},
        "tokenize_data": {"enabled": True, "workers": 2},
    }
