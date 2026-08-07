# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
