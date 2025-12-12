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

"""Unit tests for megatron hooks."""

import torch
import torch.nn as nn
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron()

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from megatron.core.parallel_state import initialize_model_parallel

from modelopt.torch.nas.plugins.megatron_hooks import (
    IterativeChannelContributionHook,
    MegatronL2NormHook,
)


def _test_iterative_channel_contribution_hook_with_shape(dim1: int, dim2: int):
    """Helper function to test IterativeChannelContributionHook with given activation shape.

    Args:
        dim1: First dimension of activation tensor (before in_features).
        dim2: Second dimension of activation tensor (before in_features).
    """
    torch.manual_seed(42)

    linear_layer = nn.Linear(in_features=6, out_features=4, bias=False)
    activation_hooks_kwargs = {
        "validation_full_iters": 3,
        "clear_gpu_memory": False,
        "calibration_method": None,
    }
    hook = IterativeChannelContributionHook(linear_layer, activation_hooks_kwargs)
    linear_layer.register_forward_hook(hook)

    for _ in range(activation_hooks_kwargs["validation_full_iters"]):
        activations = torch.randn(dim1, dim2, linear_layer.in_features)
        _ = linear_layer(activations)

    results = hook.to_dict()

    #
    # Assertions
    #
    assert results["score"].shape == (6,)
    assert results["channels_importance_ascending"].shape == (6,)

    expected_scores = torch.tensor([5, 1, 3, 2, 4, 0])
    assert torch.equal(results["score"], expected_scores)

    expected_channels_asc = torch.tensor([5, 1, 3, 2, 4, 0])
    assert torch.equal(results["channels_importance_ascending"], expected_channels_asc)

    # Test that accumulate() returns the same scores as to_dict()["score"]
    scores_from_accumulate = hook.accumulate()
    assert torch.equal(scores_from_accumulate, expected_scores)


def test_iterative_channel_contribution_hook_sbi():
    """Test IterativeChannelContributionHook returns correct scores for input [seq_len, batch_size, in_features]."""
    _test_iterative_channel_contribution_hook_with_shape(dim1=32, dim2=8)


def test_iterative_channel_contribution_hook_bsi():
    """Test IterativeChannelContributionHook returns correct scores for input [batch_size, seq_len, in_features]."""
    _test_iterative_channel_contribution_hook_with_shape(dim1=8, dim2=32)


def _test_l2_norm_hook(rank, size):
    """Internal test function that runs in spawned process with distributed setup."""
    # Initialize Megatron parallel state (distributed is already initialized by spawn_multiprocess_job)
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    torch.manual_seed(42)

    linear_layer = nn.Linear(in_features=6, out_features=4, bias=False)
    hook = MegatronL2NormHook(max_size=None)
    linear_layer.register_forward_hook(hook)

    num_iterations = 3
    for _ in range(num_iterations):
        activations = torch.randn(2, 3, linear_layer.in_features)
        _ = linear_layer(activations)

    scores = hook.accumulate()

    #
    # Assertions
    #
    assert scores.shape == (6,)

    expected_scores = torch.tensor(
        [3.2030, 2.5018, 2.5272, 1.9222, 2.6204, 2.2623], dtype=torch.float32
    )
    assert torch.allclose(scores, expected_scores, atol=1e-4), (
        f"Expected scores {expected_scores}, got {scores}"
    )


def test_l2_norm_hook():
    """Test MegatronL2NormHook returns correct scores after accumulating activations."""
    spawn_multiprocess_job(
        size=1,
        job=_test_l2_norm_hook,
        backend="gloo",
    )
