# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic logical-ID PDD held-out oracle tests."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))
if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pdd_test_utils import build_toy_lifecycle, make_batch
from pdd_training import (
    build_pdd_validation_assignments,
    pdd_validation_noise,
    pdd_validation_support,
    run_pdd_validation,
)

from modelopt.torch.fastgen import PDDConfig


def test_canonical_2k_assignment_covers_all_1568_pairs_32_starts_and_128_heads() -> None:
    config = PDDConfig(
        grid_size=128,
        flow_shift=5.0,
        block_size_min=4,
        block_size_max=64,
        inference_blocks=[32, 32, 32, 32],
        student_sample_steps=4,
    )
    sample_ids = [f"heldout-{index:04d}" for index in range(2000)]
    assignments = build_pdd_validation_assignments(
        list(reversed(sample_ids)),
        config,
        validation_seed=2026,
    )

    assert len(pdd_validation_support(config)) == 1568
    assert len(assignments) == 2000
    assert len({(assignment.n, assignment.k) for assignment in assignments}) == 1568
    assert len({assignment.n for assignment in assignments}) == 32
    assert len({assignment.k for assignment in assignments}) == 128
    assert assignments == build_pdd_validation_assignments(
        sample_ids,
        config,
        validation_seed=2026,
    )


def test_assignment_rejects_duplicate_missing_coverage_and_noise_is_per_id_stable() -> None:
    lifecycle = build_toy_lifecycle()
    with pytest.raises(ValueError, match="unique"):
        build_pdd_validation_assignments(
            ["duplicate", "duplicate"],
            lifecycle.config,
            validation_seed=1,
            require_full_coverage=False,
        )
    with pytest.raises(ValueError, match="at least"):
        build_pdd_validation_assignments(
            ["too-small"],
            lifecycle.config,
            validation_seed=1,
        )

    before = torch.get_rng_state().clone()
    first = pdd_validation_noise(
        "logical-id",
        (3,),
        validation_seed=7,
        device=torch.device("cpu"),
    )
    second = pdd_validation_noise(
        "logical-id",
        (3,),
        validation_seed=7,
        device=torch.device("cpu"),
    )
    different = pdd_validation_noise(
        "different-id",
        (3,),
        validation_seed=7,
        device=torch.device("cpu"),
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_repeated_validation_is_exact_and_does_not_change_training_state_or_rng() -> None:
    lifecycle = build_toy_lifecycle()
    sample_ids = tuple(f"validation-{index:02d}" for index in range(12))
    assignments = build_pdd_validation_assignments(
        sample_ids,
        lifecycle.config,
        validation_seed=44,
        require_full_coverage=False,
    )
    batches = [
        make_batch((assignment.sample_id,), offset=assignment.ordinal / 100)
        for assignment in assignments
    ]
    parameter_before = {
        name: parameter.detach().clone() for name, parameter in lifecycle.student.named_parameters()
    }
    optimizer_before = lifecycle.optimizer.state_dict()
    scheduler_before = lifecycle.scheduler.state_dict()
    rng_before = torch.get_rng_state().clone()
    student_mode_before = lifecycle.student.training
    teacher_mode_before = lifecycle.teacher.training

    first = run_pdd_validation(
        lifecycle.pipeline,
        (batch for batch in batches),
        assignments,
        validation_seed=44,
    )
    second = run_pdd_validation(
        lifecycle.pipeline,
        list(reversed(batches)),
        assignments,
        validation_seed=44,
    )

    assert first == second
    assert first.mean_loss == second.mean_loss
    assert first.pair_count == len({(item.n, item.k) for item in assignments})
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert lifecycle.student.training is student_mode_before
    assert lifecycle.teacher.training is teacher_mode_before
    assert lifecycle.optimizer.state_dict() == optimizer_before
    assert lifecycle.scheduler.state_dict() == scheduler_before
    for name, parameter in lifecycle.student.named_parameters():
        torch.testing.assert_close(parameter, parameter_before[name], rtol=0, atol=0)
