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

"""Targeted unit tests for distillation losses not exercised elsewhere.

tests/unit/torch/distill/test_distill.py wires losses through ``mtd.convert``
but never runs MGDLoss's forward path nor MFTLoss's ``apply_threshold_to_all=False``
branch; these tests cover that logic directly with hand-computed expectations.
"""

import pytest
import torch
import torch.nn as nn

from modelopt.torch.distill.losses import MFTLoss, MGDLoss


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(1234)


class TestMFTLoss:
    def test_threshold_not_applied_to_correct_argmax_rows(self):
        # apply_threshold_to_all=False leaves correct-argmax rows untouched while
        # still correcting incorrect-argmax rows.
        # Row 0: p = [0.7, 0.2, 0.1], label 0 (argmax correct) -> unchanged.
        # Row 1: p = [0.6, 0.3, 0.1], label 1 (argmax 0 is wrong):
        #   mixin = (p_argmax - p_label + thr) / (1 + p_argmax - p_label)
        #         = (0.6 - 0.3 + 0.2) / (1 + 0.3) = 5/13
        #   adjusted = p * (8/13) + one_hot(1) * (5/13) = [4.8, 7.4, 0.8] / 13
        mft = MFTLoss(threshold=0.2)
        p = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        corrected = mft._prepare_corrected_distributions(
            p.log(), torch.tensor([0, 1]), 0.2, apply_threshold_to_all=False
        )
        expected = torch.tensor([[0.7, 0.2, 0.1], [4.8 / 13.0, 7.4 / 13.0, 0.8 / 13.0]])
        assert torch.allclose(corrected, expected, atol=1e-5)


class TestMGDLoss:
    def test_zeroed_generation_gives_mse_against_zero(self):
        # Matching channel counts must make align an identity (no 1x1 conv).
        # With lambda_mgd=0 the random mask is all ones (rand() > 1 never holds),
        # and with all generation weights/biases zeroed the reconstructed feature
        # map is exactly zero, so the loss reduces to mse(0, out_t) = mean(out_t^2).
        mgd = MGDLoss(4, 4, lambda_mgd=0.0)
        assert isinstance(mgd.align, nn.Identity)
        for param in mgd.generation.parameters():
            param.data.zero_()
        out_s = torch.randn(2, 4, 5, 5)
        out_t = torch.randn(2, 4, 5, 5)
        loss = mgd(out_s, out_t)
        assert torch.isclose(loss, (out_t**2).mean(), atol=1e-6)

    def test_alpha_mgd_scales_loss_exactly(self):
        # lambda_mgd=0 makes the forward deterministic (mask is all ones), and
        # sharing weights isolates the alpha factor: loss(alpha=3) == 3 * loss(alpha=1).
        mgd_a = MGDLoss(4, 4, alpha_mgd=1.0, lambda_mgd=0.0)
        mgd_b = MGDLoss(4, 4, alpha_mgd=3.0, lambda_mgd=0.0)
        mgd_b.load_state_dict(mgd_a.state_dict())
        out_s = torch.randn(2, 4, 6, 6)
        out_t = torch.randn(2, 4, 6, 6)
        assert torch.isclose(mgd_b(out_s, out_t), 3.0 * mgd_a(out_s, out_t), atol=1e-6)

    def test_full_mask_ignores_student_features(self):
        # lambda_mgd=1.0 zeroes the entire mask (rand() > 0 holds a.s.), so the
        # student features are fully masked out and the loss only depends on
        # generation(0) vs the teacher features.
        mgd = MGDLoss(4, 4, lambda_mgd=1.0)
        out_t = torch.randn(1, 4, 3, 3)
        loss_a = mgd(torch.randn(1, 4, 3, 3), out_t)
        loss_b = mgd(torch.full((1, 4, 3, 3), 7.0), out_t)
        assert torch.isclose(loss_a, loss_b, atol=1e-7)

    def test_channel_alignment_and_gradient_flow(self):
        # Mismatched channels route the student features through the 1x1 align
        # conv, and gradients must reach the student input and every loss-module
        # parameter so DistillationModel can train them.
        mgd = MGDLoss(3, 5, lambda_mgd=0.0)
        out_s = torch.randn(1, 3, 4, 4, requires_grad=True)
        out_t = torch.randn(1, 5, 4, 4)
        loss = mgd(out_s, out_t)
        assert loss.ndim == 0
        assert loss.item() >= 0.0  # MSE is non-negative
        loss.backward()
        assert out_s.grad is not None
        assert mgd.align.weight.grad is not None
        assert all(p.grad is not None for p in mgd.generation.parameters())

    def test_spatial_mismatch_raises(self):
        mgd = MGDLoss(4, 4)
        with pytest.raises(AssertionError):
            mgd(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 5, 5))

    def test_teacher_features_not_detached(self):
        # NOTE: documents current behavior; arguably a bug because MGDLoss does
        # not detach the teacher feature map (unlike LogitsDistillationLoss and
        # MFTLoss, which detach their targets), so gradients propagate into the
        # teacher whenever its features require grad.
        mgd = MGDLoss(4, 4, lambda_mgd=0.0)
        out_s = torch.randn(1, 4, 3, 3, requires_grad=True)
        out_t = torch.randn(1, 4, 3, 3, requires_grad=True)
        mgd(out_s, out_t).backward()
        assert out_t.grad is not None
