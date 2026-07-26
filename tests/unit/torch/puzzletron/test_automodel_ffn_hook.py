# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the AutoModel FFN independent-channel scorer.

Validates two contracts (no GPU required, gloo/CPU):
  * **Equivalent decisions vs legacy** — same channel ranking (and exact values
    for a single forward) as ``IndependentChannelContributionHook``.
  * **Node invariance** — under a 2x2 (dp x tp) DTensor layout, the finalized
    score equals the single-process reference.
"""

import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from modelopt.torch.prune.importance_hooks.base_hooks import (
    IndependentChannelContributionHook,
    IterativeChannelContributionHook,
)
from modelopt.torch.puzzletron.plugins.automodel.hooks import (
    FFNIndependentScorer,
    FFNIterativeScorer,
)
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups


class _WeightHolder:
    """Minimal stand-in for a down_proj module exposing only ``.weight`` (picklable)."""

    def __init__(self, weight):
        self.weight = weight


def test_ffn_independent_matches_legacy_single_forward():
    torch.manual_seed(0)
    intermediate, hidden, n_tokens = 6, 8, 20
    down_proj = nn.Linear(intermediate, hidden, bias=False)
    act = torch.randn(n_tokens, intermediate)

    legacy = IndependentChannelContributionHook(down_proj)
    legacy(down_proj, (act,), None)
    legacy_score = legacy.to_dict()["score"]

    scorer = FFNIndependentScorer(down_proj, MeshGroups())
    scorer.set_batch_metadata(
        sequence_ids=torch.zeros((1, n_tokens), dtype=torch.long),
        num_samples=1,
    )
    scorer(down_proj, (act,), None)
    out = scorer.finalize()

    # A single forward => global mean == legacy's per-batch mean: values match exactly.
    assert torch.allclose(out["score"], legacy_score, rtol=1e-4, atol=1e-6)
    assert torch.equal(out["score"].argsort(descending=True), legacy_score.argsort(descending=True))


def test_ffn_independent_multibatch_ranking_matches_legacy():
    torch.manual_seed(1)
    intermediate, hidden = 6, 8
    down_proj = nn.Linear(intermediate, hidden, bias=False)
    batches = [torch.randn(10, intermediate) for _ in range(3)]  # uniform token counts

    legacy = IndependentChannelContributionHook(down_proj)
    scorer = FFNIndependentScorer(down_proj, MeshGroups())
    for batch in batches:
        legacy(down_proj, (batch,), None)
        scorer(down_proj, (batch,), None)

    legacy_score = legacy.to_dict()["score"]
    my_score = scorer.finalize()["score"]
    # Global-mean vs sum-of-per-batch-means differ by a positive scalar -> same ranking.
    assert torch.equal(my_score.argsort(descending=True), legacy_score.argsort(descending=True))


def test_ffn_independent_excludes_canonical_padding():
    down_proj = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        down_proj.weight.fill_(1)
    activations = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]]])

    scorer = FFNIndependentScorer(down_proj, MeshGroups())
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0, 0, -1]]),
        num_samples=1,
    )
    scorer(down_proj, (activations,), None)

    expected = torch.tensor([2.0, 3.0]) * torch.sqrt(torch.tensor(1.0))
    torch.testing.assert_close(scorer.finalize()["score"], expected)


def test_ffn_padding_metadata_tracks_pipeline_microbatches():
    down_proj = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        down_proj.weight.fill_(1)
    scorer = FFNIndependentScorer(down_proj, MeshGroups())
    scorer.set_batch_metadata(
        sequence_ids=torch.tensor([[0, 0, -1], [1, -1, -1]]),
        num_samples=2,
    )

    scorer(down_proj, (torch.tensor([[[1.0], [3.0], [1000.0]]]),), None)
    scorer(down_proj, (torch.tensor([[[5.0], [1000.0], [1000.0]]]),), None)

    torch.testing.assert_close(scorer.finalize()["score"], torch.tensor([3.0]))


def _job_ffn_node_invariance(rank, size):
    assert size == 4
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    groups = MeshGroups.from_device_mesh(mesh)

    torch.manual_seed(0)
    intermediate, hidden, n_tokens = 8, 4, 16
    weight = torch.randn(hidden, intermediate, dtype=torch.float64)
    act = torch.randn(n_tokens, intermediate, dtype=torch.float64)

    # Single-process reference (legacy single-forward formula).
    weight_norm = torch.linalg.vector_norm(weight.float(), dim=0)
    ref = weight_norm * act.abs().float().mean(dim=0)

    # Distributed: weight TP-sharded on the intermediate dim; activation sharded on
    # both tokens (dp) and channels (tp), as under TP + data parallel.
    holder = _WeightHolder(distribute_tensor(weight, mesh, [Replicate(), Shard(1)]))
    act_dtensor = distribute_tensor(act, mesh, [Shard(0), Shard(1)])

    scorer = FFNIndependentScorer(holder, groups)
    scorer(holder, (act_dtensor,), None)
    out = scorer.finalize()["score"]

    assert out.shape == (intermediate,), out.shape
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6), (out, ref)


def test_ffn_independent_node_invariance():
    spawn_multiprocess_job(size=4, job=_job_ffn_node_invariance, backend="gloo")


def test_ffn_iterative_matches_legacy_single_process():
    torch.manual_seed(0)
    intermediate, hidden, n_iters = 8, 6, 4  # schedule prunes 2 channels/iter
    down_proj = nn.Linear(intermediate, hidden, bias=False)
    batches = [torch.randn(1, 5, intermediate) for _ in range(n_iters)]

    legacy = IterativeChannelContributionHook(down_proj, {"validation_full_iters": n_iters})
    scorer = FFNIterativeScorer(down_proj, MeshGroups(), validation_full_iters=n_iters)
    for batch in batches:
        out = down_proj(batch)
        legacy(down_proj, (batch,), out)  # legacy advances inside __call__
        scorer(down_proj, (batch,), out)  # ours accumulates ...
        scorer.step_iteration()  # ... then advances here (single process: no reduce)

    legacy_out = legacy.to_dict()
    out = scorer.finalize()
    # Greedy iterative pruning is deterministic -> identical ranking and order.
    assert torch.equal(
        out["channels_importance_ascending"], legacy_out["channels_importance_ascending"]
    )
    assert torch.equal(out["score"], legacy_out["score"])


def test_ffn_iterative_exact_checkpoint_finalizes_without_replay():
    torch.manual_seed(4)
    down_proj = nn.Linear(8, 6, bias=False)
    scorer = FFNIterativeScorer(
        down_proj,
        MeshGroups(),
        validation_full_iters=2,
    )
    for _ in range(2):
        batch = torch.randn(1, 3, 8)
        scorer(down_proj, (batch,), down_proj(batch))
        scorer.step_iteration()
    expected = scorer.finalize()

    restored = FFNIterativeScorer(
        down_proj,
        MeshGroups(),
        validation_full_iters=2,
    )
    restored.load_checkpoint_state(scorer.checkpoint_state())
    actual = restored.finalize()

    torch.testing.assert_close(actual["score"], expected["score"])
    torch.testing.assert_close(
        actual["channels_importance_ascending"],
        expected["channels_importance_ascending"],
    )


# --- iterative scorer under data parallelism (dp=2) ---------------------------------
_ITER_INTER, _ITER_HID, _ITER_ITERS = 8, 6, 4


def _iter_down_proj():
    torch.manual_seed(123)
    down = nn.Linear(_ITER_INTER, _ITER_HID, bias=False)
    return down


def _iter_batches():
    """Per iteration, a distinct batch for rank 0 and rank 1 (deterministic, fp32)."""
    torch.manual_seed(0)
    r0 = [torch.randn(1, 5, _ITER_INTER) for _ in range(_ITER_ITERS)]
    r1 = [torch.randn(1, 5, _ITER_INTER) for _ in range(_ITER_ITERS)]
    return r0, r1


def _iter_reference():
    """Single process: per iteration, accumulate BOTH ranks' batches, then step once.

    For a 2-rank SUM-reduce this is bit-identical to the dp=2 path (pending = ts0 + ts1).
    """
    down = _iter_down_proj()
    r0, r1 = _iter_batches()
    scorer = FFNIterativeScorer(down, MeshGroups(), validation_full_iters=_ITER_ITERS)
    for i in range(_ITER_ITERS):
        scorer(down, (r0[i],), down(r0[i]))
        scorer(down, (r1[i],), down(r1[i]))
        scorer.step_iteration()
    return scorer.finalize()


def _job_ffn_iterative_dp(rank, size):
    assert size == 2
    down = _iter_down_proj()
    r0, r1 = _iter_batches()
    my_batches = r0 if rank == 0 else r1

    groups = MeshGroups(token_group=dist.group.WORLD)  # 2-rank data-parallel token group
    scorer = FFNIterativeScorer(down, groups, validation_full_iters=_ITER_ITERS)
    for i in range(_ITER_ITERS):
        scorer(down, (my_batches[i],), down(my_batches[i]))
        scorer.step_iteration()  # reduces this iteration's contribution across the 2 ranks
    out = scorer.finalize()

    ref = _iter_reference()
    assert torch.equal(
        out["channels_importance_ascending"], ref["channels_importance_ascending"]
    ), (rank, out["channels_importance_ascending"], ref["channels_importance_ascending"])


def test_ffn_iterative_dp_equivalence():
    spawn_multiprocess_job(size=2, job=_job_ffn_iterative_dp, backend="gloo")
