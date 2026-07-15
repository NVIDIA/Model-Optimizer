# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.stages.pipeline import _bypass_sanity_overfit_config


def test_public_bypass_sanity_config_drives_legacy_overfit_probe():
    config = {
        "bypass_sanity": {
            "steps": 1,
            "fixed_smallest": True,
            "diverse_nested": True,
        },
        "bypass": {
            "overfit": {
                "repetitions": 32,
                "modes": ["smallest_fixed"],
                "learning_rate": 3.0e-4,
            }
        },
    }

    assert _bypass_sanity_overfit_config(config) == {
        "repetitions": 1,
        "modes": ["smallest_fixed", "diverse_resampled"],
        "learning_rate": 3.0e-4,
        "decay_lr": False,
        "weight_decay": 0.0,
        "minimum_relative_decrease": 0.05,
    }


def test_legacy_overfit_probe_config_is_preserved_without_public_controls():
    config = {
        "bypass_sanity": {"enabled": True},
        "bypass": {"overfit": {"repetitions": 8, "modes": ["smallest_fixed"]}},
    }

    assert _bypass_sanity_overfit_config(config) == {
        "repetitions": 8,
        "modes": ["smallest_fixed"],
    }


def test_public_bypass_sanity_uses_overfit_specific_optimizer_defaults():
    config = {
        "bypass_sanity": {
            "steps": 32,
            "fixed_smallest": True,
            "diverse_nested": True,
        },
        "bypass": {},
    }

    assert _bypass_sanity_overfit_config(config) == {
        "repetitions": 32,
        "modes": ["smallest_fixed", "diverse_resampled"],
        "learning_rate": 3.0e-4,
        "decay_lr": False,
        "weight_decay": 0.0,
        "minimum_relative_decrease": 0.05,
    }
