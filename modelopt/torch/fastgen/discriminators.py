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

"""Discriminator modules for the DMD2 GAN branch.

Ports FastGen's image-DiT discriminator from
``source/FastGen/fastgen/networks/discriminators.py`` so that ModelOpt's
:class:`~modelopt.torch.fastgen.methods.dmd.DMDPipeline` can run the GAN branch
without a FastGen dependency. The discriminator is **model-agnostic**: it takes
a list of spatial feature tensors ``[B, C, H, W]`` and returns concatenated
logits ``[B, num_heads]``. The model-specific work of producing those tensors
(installing forward hooks, reshaping packed-token streams into spatial maps)
lives in the per-model plugins (``plugins/qwen_image.py``).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Discriminator", "Discriminator_ImageDiT"]


def _get_optimal_groups(num_channels: int) -> int:
    """Return a GroupNorm group count that divides ``num_channels`` evenly.

    Matches the heuristic in FastGen's discriminator: prefer 32 groups when
    possible, fall back to the largest divisor below 32, and use
    ``num_channels // 4`` for very small channel counts.
    """
    if num_channels <= 32:
        groups = max(1, num_channels // 4)
    else:
        groups = 32
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
    assert num_channels % groups == 0, f"{num_channels} not divisible by {groups}"
    return groups


class Discriminator(nn.Module):
    """Base class for DMD2 discriminators."""

    def __init__(self, feature_indices: set[int] | None = None) -> None:
        """Store the teacher block indices whose features feed the discriminator."""
        super().__init__()
        self.feature_indices = feature_indices

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Map captured teacher features to discriminator logits (overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement forward()")


# Class name kept verbatim from the FastGen reference implementation.
class Discriminator_ImageDiT(Discriminator):  # noqa: N801
    """Image-DiT discriminator with one lightweight conv head per captured block.

    Input: list of feature tensors with shape ``[B, inner_dim, H, W]``, one per
    block index in :attr:`feature_indices`.

    Output: concatenated logits ``[B, num_heads]`` (one column per head). The
    DMD2 generator/discriminator losses read this as a 2D tensor.

    Per-head parameter count is ~``inner_dim * (inner_dim // 2) * 16 + ...``;
    for ``inner_dim=3072`` (Flux / Qwen-Image) that's ~75 M params per head, so
    keep ``len(feature_indices)`` small (≤3 heads is typical).
    """

    def __init__(
        self,
        feature_indices: set[int] | None = None,
        num_blocks: int = 57,
        inner_dim: int = 3072,
    ) -> None:
        """Build one lightweight conv classification head per captured block."""
        super().__init__(feature_indices=feature_indices)

        if self.feature_indices is None:
            self.feature_indices = {int(num_blocks // 2)}
        self.feature_indices = {i for i in self.feature_indices if i < num_blocks}
        self.num_features = len(self.feature_indices)
        self.inner_dim = inner_dim

        hidden_channels = inner_dim // 2
        self.cls_pred_heads = nn.ModuleList()
        for _ in range(self.num_features):
            head = nn.Sequential(
                nn.Conv2d(
                    in_channels=inner_dim,
                    out_channels=hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(
                    num_groups=_get_optimal_groups(hidden_channels),
                    num_channels=hidden_channels,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.cls_pred_heads.append(head)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Run each per-block conv head and concatenate their logits to ``[B, num_heads]``."""
        if not isinstance(feats, list) or len(feats) != self.num_features:
            raise ValueError(
                f"Expected list of {self.num_features} feature tensors, "
                f"got {type(feats).__name__} with length "
                f"{len(feats) if isinstance(feats, list) else 'N/A'}"
            )
        all_logits = []
        for head, feat in zip(self.cls_pred_heads, feats):
            logits = head(feat)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=1)
