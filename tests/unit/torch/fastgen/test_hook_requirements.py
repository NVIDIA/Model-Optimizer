# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the "did you attach hooks?" / "is this FSDP-wrapped?" runtime guards.

Covers R2.1 (GAN branches must raise a clear ``RuntimeError`` when
``teacher._fastgen_captured`` is missing) and R2.5 (``create_fake_score`` must
reject FSDP-wrapped teachers when ``deep_copy=True``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from modelopt.torch.fastgen import DMDConfig, DMDPipeline, create_fake_score


class _ToyTransformer(nn.Module):
    """Linear-on-hidden-states transformer that matches the pipeline's call convention."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        return self.linear(hidden_states)


class _ToyDiscriminator(nn.Module):
    """Consumes ``list[Tensor]`` and returns 2D logits ``(B, 1)``."""

    def forward(self, features):
        x = features[0]
        return x.flatten(start_dim=1).mean(dim=-1, keepdim=True)


def _build_gan_pipeline(d: int) -> DMDPipeline:
    torch.manual_seed(0)
    cfg = DMDConfig(pred_type="flow", gan_loss_weight_gen=0.03)
    return DMDPipeline(
        _ToyTransformer(d),
        _ToyTransformer(d),
        _ToyTransformer(d),
        cfg,
        discriminator=_ToyDiscriminator(),
    )


def test_compute_student_loss_raises_when_hooks_missing():
    """GAN-enabled ``compute_student_loss`` without ``attach_feature_capture``
    must raise a ``RuntimeError`` naming the attach helper, not strip under
    ``-O`` like the previous ``assert``."""
    d, b = 4, 2
    pipeline = _build_gan_pipeline(d)
    latents = torch.randn(b, d)
    noise = torch.randn(b, d)

    with pytest.raises(RuntimeError, match="attach_feature_capture"):
        pipeline.compute_student_loss(latents, noise)


def test_compute_discriminator_loss_raises_when_hooks_missing():
    """``compute_discriminator_loss`` without ``attach_feature_capture`` must
    raise ``RuntimeError`` at the first drain, before the discriminator is called."""
    d, b = 4, 2
    pipeline = _build_gan_pipeline(d)
    latents = torch.randn(b, d)
    noise = torch.randn(b, d)

    with pytest.raises(RuntimeError, match="attach_feature_capture"):
        pipeline.compute_discriminator_loss(latents, noise)


def test_create_fake_score_raises_on_fsdp2_wrapped():
    """``create_fake_score(deep_copy=True)`` on a module whose first parameter
    looks like a DTensor (``full_tensor`` attribute) must raise with a message
    pointing at the meta-init recipe in the docstring."""
    m = nn.Linear(4, 4)
    # Monkey-patch the first parameter to look like a DTensor.
    first = next(m.parameters())
    first.full_tensor = lambda: first  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="meta-init"):
        create_fake_score(m, deep_copy=True)


def test_create_fake_score_no_copy_skips_fsdp_check():
    """``deep_copy=False`` reuses the teacher directly — the FSDP-wrap check
    is skipped because there is no ``copy.deepcopy`` to protect against."""
    m = nn.Linear(4, 4)
    first = next(m.parameters())
    first.full_tensor = lambda: first  # type: ignore[attr-defined]

    fake_score = create_fake_score(m, deep_copy=False)
    assert fake_score is m
    assert fake_score.training  # create_fake_score must set .train() regardless
    assert all(p.requires_grad for p in fake_score.parameters())
