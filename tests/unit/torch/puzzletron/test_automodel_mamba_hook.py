# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from modelopt.torch.puzzletron.plugins.automodel.hooks.mamba import MambaInProjContributionScorer
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups


def _projected(x_values: torch.Tensor) -> torch.Tensor:
    """Pack x into Nemotron's [gate, x, B, C, dt] projection layout."""
    shape = (*x_values.shape[:-1], 12)
    projected = torch.zeros(shape, dtype=x_values.dtype)
    projected[..., 4:8] = x_values
    return projected


def _scorer() -> MambaInProjContributionScorer:
    return MambaInProjContributionScorer(
        nn.Identity(),
        MeshGroups(),
        num_heads=2,
        head_dim=2,
        num_groups=1,
        state_dim=1,
        name="layers.0.mixer.in_proj",
    )


def test_packed_thd_and_batched_mamba_scores_match() -> None:
    values = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
         [[5.0, 6.0, 7.0, 8.0], [7.0, 8.0, 9.0, 10.0]]]
    )
    batched = _scorer()
    batched.set_batch_metadata(sequence_ids=torch.tensor([[0, 0], [1, 1]]), num_samples=2)
    batched(None, (), _projected(values))

    packed = _scorer()
    packed.set_batch_metadata(sequence_ids=torch.tensor([[0, 0, 1, 1]]), num_samples=2)
    packed(None, (), _projected(values.reshape(4, 4)))

    torch.testing.assert_close(packed.finalize()["x_scores"], batched.finalize()["x_scores"])
