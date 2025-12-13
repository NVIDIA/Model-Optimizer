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
"""Analysis tools for evaluating importance scores from hooks."""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["evaluate_importance_scores"]


def evaluate_importance_scores(
    linear_layer: nn.Linear,
    activations_batches: list[torch.Tensor],
    importance_scores: torch.Tensor,
    prune_ratio: float = 0.2,
) -> dict[str, float]:
    """Compute reconstruction error after pruning input channels of a linear layer.

    This function simulates channel pruning by zeroing out input channels identified as
    least important, then measures how much the layer's output changes.

    Args:
        linear_layer: The linear layer to analyze with shape (out_features, in_features).
                     For example: nn.Linear(in_features=1024, out_features=4096)
        activations_batches: List of input activation tensors.
                            Each tensor has shape [seq_len, batch_size, in_features].
                            The last dimension must match linear_layer.in_features.
                            Example: List of [16, 8, 1024] tensors
        importance_scores: Importance score for each input channel (feature).
                          Shape: [in_features]. Lower scores = less important.
                          Example: [1024] tensor with one score per input feature
        prune_ratio: Fraction of input channels to prune (default: 0.2 means prune 20%).

    Returns:
        Dictionary containing averaged metrics across all activation batches:
            - rmse: Root mean squared error between original and pruned output
            - cosine_similarity: Cosine similarity between original and pruned output
            - num_pruned: Number of input channels pruned

    Example:
        >>> layer = nn.Linear(in_features=1024, out_features=4096)
        >>> # Collect multiple batches for robust evaluation
        >>> activations_list = [torch.randn(16, 8, 1024) for _ in range(100)]
        >>> scores = torch.randn(1024)  # one score per input feature
        >>> metrics = evaluate_importance_scores(layer, activations_list, scores, 0.2)
        >>> print(f"RMSE: {metrics['rmse']:.4f}, Pruned: {metrics['num_pruned']} channels")

    Note:
        - This simulates pruning (zeros out inputs) without modifying layer weights
        - "Channels" refers to INPUT features, not output features

    """
    num_channels = importance_scores.shape[0]
    num_to_prune = int(num_channels * prune_ratio)

    # Identify channels to prune (lowest scoring = least important)
    _, channels_to_prune = torch.topk(importance_scores, num_to_prune, largest=False)

    # Compute metrics for each batch and average
    rmse_values = []
    cosine_values = []

    for activations in activations_batches:
        # Get original output
        original_output = linear_layer(activations)

        # Prune by zeroing out identified channels
        pruned_activations = activations.clone()
        pruned_activations[..., channels_to_prune] = 0

        # Get pruned output
        pruned_output = linear_layer(pruned_activations)

        # Compute metrics for this batch
        rmse = torch.sqrt(F.mse_loss(pruned_output, original_output)).item()
        rmse_values.append(rmse)

        # Cosine similarity (flatten to vectors)
        original_flat = original_output.reshape(-1)
        pruned_flat = pruned_output.reshape(-1)
        cosine = F.cosine_similarity(
            original_flat.unsqueeze(0), pruned_flat.unsqueeze(0), dim=1
        ).item()
        cosine_values.append(cosine)

    # Return averaged metrics
    return {
        "rmse": sum(rmse_values) / len(rmse_values),
        "cosine_similarity": sum(cosine_values) / len(cosine_values),
        "num_pruned": num_to_prune,
    }
