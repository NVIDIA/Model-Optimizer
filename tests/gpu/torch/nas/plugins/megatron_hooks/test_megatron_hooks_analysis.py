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

"""Unit tests for megatron hooks analysis tools."""

import pytest
import torch
import torch.nn as nn
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron()

from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from megatron.core.parallel_state import initialize_model_parallel

from modelopt.torch.nas.plugins.megatron_hooks import (
    IndependentChannelContributionHook,
    IterativeChannelContributionHook,
    MegatronL2NormHook,
    evaluate_importance_scores,
)


def test_evaluate_importance_scores_basic():
    """Test basic functionality of importance score evaluation with synthetic scores."""
    torch.manual_seed(42)

    # Create a simple linear layer (same dimensions as other tests for comparability)
    layer = nn.Linear(in_features=50, out_features=30, bias=False)

    # Create synthetic hook that generates sequential importance scores
    hook = SyntheticImportanceHook(num_features=50)

    # Use shared helper to run evaluation
    metrics = _run_hook_and_evaluate(layer, hook, num_iterations=1000, prune_ratio=0.4)

    print(f"[SyntheticImportanceHook] Metrics: {metrics}")

    # Check values with deterministic seed
    assert metrics["num_pruned"] == 20  # 40% of 50 = 20
    assert metrics["rmse"] == pytest.approx(0.3689444, rel=1e-5)
    assert metrics["cosine_similarity"] == pytest.approx(0.77117118, rel=1e-5)


def _test_evaluate_importance_scores_with_l2_norm_hook(rank, size):
    """Test evaluate_importance_scores with MegatronL2NormHook."""
    # Initialize Megatron parallel state
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    torch.manual_seed(42)

    # Create layer and hook
    layer = nn.Linear(in_features=50, out_features=30, bias=False)
    hook = MegatronL2NormHook(max_size=None)

    # Run evaluation
    metrics = _run_hook_and_evaluate(layer, hook, num_iterations=1000, prune_ratio=0.4)

    print(f"[L2NormHook] Metrics: {metrics}")

    # L2NormHook specific assertions
    assert metrics["num_pruned"] == 20  # 40% of 50 = 20
    assert metrics["rmse"] == pytest.approx(0.3616334, rel=1e-5)
    assert metrics["cosine_similarity"] == pytest.approx(0.7814186, rel=1e-5)


def _test_evaluate_importance_scores_with_iterative_channel_contribution_hook(rank, size):
    """Test evaluate_importance_scores with IterativeChannelContributionHook."""
    # Initialize Megatron parallel state
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    torch.manual_seed(42)

    # Create layer and hook
    layer = nn.Linear(in_features=50, out_features=30, bias=False)
    activation_hooks_kwargs = {
        "validation_full_iters": 1000,
        "clear_gpu_memory": False,
        "calibration_method": None,
    }
    hook = IterativeChannelContributionHook(layer, activation_hooks_kwargs)

    # Run evaluation
    metrics = _run_hook_and_evaluate(layer, hook, num_iterations=1000, prune_ratio=0.4)

    print(f"[IterativeChannelContributionHook] Metrics: {metrics}")

    # Iterative channel contribution hook specific assertions
    assert metrics["num_pruned"] == 20  # 40% of 50 = 20
    assert metrics["rmse"] == pytest.approx(0.339014, rel=1e-5)
    assert metrics["cosine_similarity"] == pytest.approx(0.8110392, rel=1e-5)


def _test_evaluate_importance_scores_with_independent_channel_contribution_hook(rank, size):
    """Test evaluate_importance_scores with IndependentChannelContributionHook."""
    # Initialize Megatron parallel state
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    torch.manual_seed(42)

    # Create layer and hook
    layer = nn.Linear(in_features=50, out_features=30, bias=False)
    hook = IndependentChannelContributionHook(layer)

    # Run evaluation
    metrics = _run_hook_and_evaluate(layer, hook, num_iterations=1000, prune_ratio=0.4)

    print(f"[IndependentChannelContributionHook] Metrics: {metrics}")

    # Independent channel contribution hook specific assertions
    assert metrics["num_pruned"] == 20  # 40% of 50 = 20
    assert metrics["rmse"] == pytest.approx(0.3385471, rel=1e-5)
    assert metrics["cosine_similarity"] == pytest.approx(0.8116209, rel=1e-5)


def test_evaluate_importance_scores_with_l2_norm_hook():
    """Test evaluate_importance_scores using MegatronL2NormHook."""
    spawn_multiprocess_job(
        size=1,
        job=_test_evaluate_importance_scores_with_l2_norm_hook,
        backend="gloo",
    )


def test_evaluate_importance_scores_with_iterative_channel_contribution_hook():
    """Test evaluate_importance_scores using IterativeChannelContributionHook."""
    spawn_multiprocess_job(
        size=1,
        job=_test_evaluate_importance_scores_with_iterative_channel_contribution_hook,
        backend="gloo",
    )


def test_evaluate_importance_scores_with_independent_channel_contribution_hook():
    """Test evaluate_importance_scores using IndependentChannelContributionHook."""
    spawn_multiprocess_job(
        size=1,
        job=_test_evaluate_importance_scores_with_independent_channel_contribution_hook,
        backend="gloo",
    )


def _run_hook_and_evaluate(
    layer: nn.Linear,
    hook,
    num_iterations: int,
    prune_ratio: float,
) -> dict:
    """Shared helper to run hook, collect scores, and evaluate.

    Args:
        layer: Linear layer to test
        hook: Hook instance (already created)
        num_iterations: Number of forward passes
        prune_ratio: Fraction of channels to prune

    Returns:
        Dictionary with evaluation metrics
    """
    handle = layer.register_forward_hook(hook)  # Store the handle

    # Run forward passes
    all_activations = []
    for _ in range(num_iterations):
        activations = torch.randn(
            16, 8, layer.in_features
        )  # seq=16, batch=8, in_features=50 (Megatron format)
        all_activations.append(activations)
        _ = layer(activations)

    # Get importance scores from hook
    importance_scores = hook.accumulate()

    # Remove the hook before evaluation to avoid triggering it again
    handle.remove()

    # Evaluate the importance scores by simulating pruning on all collected activations
    # Pass the list of activations to compute averaged metrics across batches
    metrics = evaluate_importance_scores(
        layer,
        all_activations,  # List of activation batches
        importance_scores,
        prune_ratio=prune_ratio,
    )

    return metrics


class SyntheticImportanceHook:
    """Synthetic hook that generates sequential importance scores for testing.

    This is a simple mock hook that doesn't compute real importance,
    just returns torch.arange(num_features) to test the evaluation pipeline.
    """

    def __init__(self, num_features: int):
        """Initialize with the number of features."""
        self.num_features = num_features

    def __call__(self, module, args, output):
        """Hook callback - does nothing for synthetic hook."""

    def accumulate(self) -> torch.Tensor:
        """Return synthetic importance scores: [0, 1, 2, ..., num_features-1]."""
        return torch.arange(self.num_features, dtype=torch.float32)
