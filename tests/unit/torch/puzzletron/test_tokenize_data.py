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

"""Tests for tokenize_data cache resolution."""

from examples.puzzletron.tokenize_data import resolve_tokenize_caches


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
