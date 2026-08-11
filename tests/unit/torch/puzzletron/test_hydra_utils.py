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

"""Tests for Hydra configuration cloning and resolver helpers."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedValueType

import modelopt.torch.puzzletron.stages.pipeline as pipeline_stages
import modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats as calc_subblock_stats
from modelopt.torch.puzzletron.tools.hydra_utils import (
    _warmup_steps_resolver,
    clone_hydra_config,
    warmup_steps,
)


class _ToyPruningMixin:
    """Stand-in for resolved Hydra ``_target_`` objects such as pruning mixins."""


def test_clone_hydra_config_preserves_resolved_python_objects():
    cfg = OmegaConf.create(
        {"pruning": {"activation_passes": [{"name": "ffn"}]}},
        flags={"allow_objects": True},
    )
    pruning_mixin = _ToyPruningMixin()
    cfg.pruning.activation_passes[0].pruning_mixin = pruning_mixin

    cloned = clone_hydra_config(cfg)

    assert cloned.pruning.activation_passes[0].pruning_mixin is pruning_mixin
    container = OmegaConf.to_container(cfg, resolve=True)
    with pytest.raises(UnsupportedValueType, match="supported primitive type"):
        OmegaConf.create(container)


def test_static_workload_stats_preserves_resolved_python_objects(monkeypatch):
    pruning_mixin = _ToyPruningMixin()
    hydra_cfg = OmegaConf.create(
        {
            "calc_subblock_stats": {
                "batch_sizes": [1],
                "prefill_seq_len": 128,
                "generation_seq_len": 32,
                "runtime_stats": {"enabled": True},
                "merge_with_existing_stats": False,
            },
            "pruning": {"activation_passes": [{"name": "ffn"}]},
        },
        flags={"allow_objects": True},
    )
    hydra_cfg.pruning.activation_passes[0].pruning_mixin = pruning_mixin
    launched = []
    monkeypatch.setattr(calc_subblock_stats, "launch_calc_subblock_stats", launched.append)

    pipeline_stages._calculate_static_workload_stats(
        {"mip": {"workloads": {"interactive": {"isl": 256, "osl": 64, "batch_size": 2}}}},
        hydra_cfg,
    )

    assert len(launched) == 1
    selected = launched[0]
    assert selected.pruning.activation_passes[0].pruning_mixin is pruning_mixin
    assert list(selected.calc_subblock_stats.batch_sizes) == [2]
    assert selected.calc_subblock_stats.prefill_seq_len == 256
    assert selected.calc_subblock_stats.generation_seq_len == 64
    assert selected.calc_subblock_stats.runtime_stats.enabled is False
    assert selected.calc_subblock_stats.merge_with_existing_stats is True


def test_warmup_steps_casts_inputs_before_computing():
    assert warmup_steps("100", "10", "2", "5", "0.5") == 1


def test_warmup_steps_preserves_legacy_defaults():
    assert warmup_steps("1000", "10", "2") == 2
    assert _warmup_steps_resolver("1000", "10", "2") == 2
    assert _warmup_steps_resolver("1000", "10", "2", "0.5") == 25
    assert _warmup_steps_resolver("1000", "10", "2", "5", "0.5") == 5


def test_warmup_steps_resolver_rejects_unknown_arity():
    with pytest.raises(ValueError, match="expects 3, 4, or 5 arguments"):
        _warmup_steps_resolver("1000", "10")


def test_warmup_steps_rejects_non_castable_inputs():
    with pytest.raises(ValueError, match="castable to int"):
        warmup_steps("not-int", "10", "2")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokens": -1, "block": 1, "mbs": 1, "grad_accum": 1, "pct": 0.1}, "tokens"),
        ({"tokens": 1, "block": 0, "mbs": 1, "grad_accum": 1, "pct": 0.1}, "block"),
        ({"tokens": 1, "block": 1, "mbs": 0, "grad_accum": 1, "pct": 0.1}, "mbs"),
        ({"tokens": 1, "block": 1, "mbs": 1, "grad_accum": 0, "pct": 0.1}, "grad_accum"),
        ({"tokens": 1, "block": 1, "mbs": 1, "grad_accum": 1, "pct": 1.1}, "pct"),
    ],
)
def test_warmup_steps_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        warmup_steps(**kwargs)
