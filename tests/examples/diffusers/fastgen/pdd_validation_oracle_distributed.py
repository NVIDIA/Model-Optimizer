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

"""Two-rank CPU/Gloo equivalence harness for the deterministic PDD validation oracle."""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import torch
import torch.distributed as dist

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))
if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pdd.training import build_pdd_validation_assignments, run_pdd_validation
from pdd_test_utils import build_toy_lifecycle, make_batch


def _expect_failure(error_type, callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def main() -> None:
    lifecycle = build_toy_lifecycle()
    sample_ids = tuple(f"distributed-validation-{index:02d}" for index in range(13))
    assignments = build_pdd_validation_assignments(
        sample_ids,
        lifecycle.config,
        validation_seed=91,
        require_full_coverage=False,
    )
    all_batches = [
        make_batch((assignment.sample_id,), offset=assignment.ordinal / 100)
        for assignment in assignments
    ]
    baseline = run_pdd_validation(
        lifecycle.pipeline,
        all_batches,
        assignments,
        validation_seed=91,
    )

    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 2
        local_batches = all_batches[rank::world_size]
        padded_batch_count = max(
            len(all_batches[candidate_rank::world_size]) for candidate_rank in range(world_size)
        )
        while len(local_batches) < padded_batch_count:
            local_batches.append(
                dataclasses.replace(
                    make_batch((f"dummy-rank-{rank}",)),
                    valid_mask=(False,),
                )
            )
        distributed = run_pdd_validation(
            lifecycle.pipeline,
            local_batches,
            assignments,
            validation_seed=91,
        )
        assert distributed.records == baseline.records
        assert abs(distributed.mean_loss - baseline.mean_loss) <= 1e-12
        assert distributed.ordered_id_sha256 == baseline.ordered_id_sha256

        invalid_mask = list(local_batches)
        if rank == 0:
            invalid_mask[0] = dataclasses.replace(invalid_mask[0], valid_mask=())
        _expect_failure(
            RuntimeError,
            lambda: run_pdd_validation(
                lifecycle.pipeline,
                invalid_mask,
                assignments,
                validation_seed=91,
            ),
        )
        dist.barrier()

        unassigned = list(local_batches)
        if rank == 0:
            unassigned[0] = dataclasses.replace(
                unassigned[0],
                sample_ids=("not-in-heldout-assignments",),
                valid_mask=(True,),
            )
        _expect_failure(
            RuntimeError,
            lambda: run_pdd_validation(
                lifecycle.pipeline,
                unassigned,
                assignments,
                validation_seed=91,
            ),
        )
        dist.barrier()

        nonfinite = list(local_batches)
        if rank == 0:
            nonfinite[0] = dataclasses.replace(
                nonfinite[0],
                data=torch.full_like(nonfinite[0].data, float("nan")),
            )
        _expect_failure(
            FloatingPointError,
            lambda: run_pdd_validation(
                lifecycle.pipeline,
                nonfinite,
                assignments,
                validation_seed=91,
            ),
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
