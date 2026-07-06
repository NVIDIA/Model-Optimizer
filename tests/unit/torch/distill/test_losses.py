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

"""Unit tests for distillation losses and loss balancers using plain tensors.

These tests exercise the loss/balancer math directly (hand-computed expectations),
complementing tests/unit/torch/distill/test_distill.py which only wires the losses
through ``mtd.convert`` end-to-end without checking any numeric values.
"""

import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.distill.loss_balancers import (
    STUDENT_LOSS_KEY,
    DistillationLossBalancer,
    StaticLossBalancer,
)
from modelopt.torch.distill.losses import LogitsDistillationLoss, MFTLoss, MGDLoss


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(1234)


class TestLogitsDistillationLoss:
    def test_identical_logits_zero_loss(self):
        logits = torch.randn(4, 10)
        loss = LogitsDistillationLoss()(logits, logits.clone())
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_hand_computed_value_mean_reduction(self):
        # Student logits [0, ln3] -> softmax [1/4, 3/4].
        # Teacher logits [0, 0]   -> softmax [1/2, 1/2].
        # Pointwise KL = t * (log t - log s):
        #   0.5 * ln(0.5/0.25) + 0.5 * ln(0.5/0.75) = 0.5 * ln2 + 0.5 * ln(2/3)
        #   = 0.5 * ln(4/3)
        # reduction="mean" divides by the total element count (2), not batch size:
        #   loss = 0.25 * ln(4/3)
        logits_s = torch.tensor([[0.0, math.log(3.0)]])
        logits_t = torch.tensor([[0.0, 0.0]])
        loss = LogitsDistillationLoss()(logits_s, logits_t)
        expected = 0.25 * math.log(4.0 / 3.0)
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_hand_computed_value_batchmean_reduction(self):
        # Same distributions as above but two rows (mirrored); each row's KL sum
        # is 0.5 * ln(4/3), and "batchmean" divides the total by batch size 2:
        #   loss = 0.5 * ln(4/3)
        logits_s = torch.tensor([[0.0, math.log(3.0)], [math.log(3.0), 0.0]])
        logits_t = torch.zeros(2, 2)
        loss = LogitsDistillationLoss(reduction="batchmean")(logits_s, logits_t)
        expected = 0.5 * math.log(4.0 / 3.0)
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_temperature_softening_and_rescaling(self):
        # With T=2, logits are divided by 2 before softmax, so student logits
        # [0, 2*ln3] soften to the same [1/4, 3/4] distribution as T=1 with
        # [0, ln3]. The pointwise loss is then multiplied by T^2 = 4:
        #   loss = 4 * 0.25 * ln(4/3) = ln(4/3)
        logits_s = torch.tensor([[0.0, 2.0 * math.log(3.0)]])
        logits_t = torch.tensor([[0.0, 0.0]])
        loss = LogitsDistillationLoss(temperature=2.0)(logits_s, logits_t)
        expected = math.log(4.0 / 3.0)
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_temperature_squared_factor_exact_ratio(self):
        # Temperature-invariant distributions: uniform teacher and uniform student
        # stay uniform under any T, so the un-scaled KL term is constant and the
        # returned loss must scale exactly as T^2.
        logits_s = torch.tensor([[0.0, math.log(3.0)]])
        logits_t = torch.tensor([[0.0, 0.0]])
        loss_t1 = LogitsDistillationLoss(temperature=1.0)(logits_s, logits_t)
        loss_t3 = LogitsDistillationLoss(temperature=3.0)(logits_s * 3.0, logits_t * 3.0)
        assert torch.isclose(loss_t3, 9.0 * loss_t1, atol=1e-6)

    def test_none_reduction_shape_and_values(self):
        # reduction="none" returns the pointwise loss summed over the class dim,
        # i.e. per-sample KL values. Both rows are mirrored so each equals
        # 0.5 * ln(4/3).
        logits_s = torch.tensor([[0.0, math.log(3.0)], [math.log(3.0), 0.0]])
        logits_t = torch.zeros(2, 2)
        loss = LogitsDistillationLoss(reduction="none")(logits_s, logits_t)
        expected = 0.5 * math.log(4.0 / 3.0)
        assert loss.shape == (2,)
        assert torch.allclose(loss, torch.full((2,), expected), atol=1e-6)

    def test_none_reduction_3d_keeps_batch_and_seq_dims(self):
        logits_s = torch.randn(2, 5, 7)
        logits_t = torch.randn(2, 5, 7)
        loss = LogitsDistillationLoss(reduction="none")(logits_s, logits_t)
        assert loss.shape == (2, 5)
        assert (loss >= 0).all()  # KL summed over full class dim is non-negative

    def test_none_reduction_supports_masking(self):
        # The documented use-case for reduction="none": apply a loss mask then
        # reduce manually. Masked-out positions must not affect the result.
        logits_s = torch.randn(2, 4, 6)
        logits_t = torch.randn(2, 4, 6)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        per_token = LogitsDistillationLoss(reduction="none")(logits_s, logits_t)
        masked_mean = (per_token * mask).sum() / mask.sum()

        expected = per_token[0, :2].sum() + per_token[1, :1].sum()
        assert torch.isclose(masked_mean, expected / 3.0, atol=1e-6)

    def test_mean_equals_sum_over_numel(self):
        logits_s = torch.randn(3, 8)
        logits_t = torch.randn(3, 8)
        loss_mean = LogitsDistillationLoss(reduction="mean")(logits_s, logits_t)
        loss_sum = LogitsDistillationLoss(reduction="sum")(logits_s, logits_t)
        # NOTE: documents current behavior; arguably a bug because F.kl_div's
        # "mean" divides by numel (batch * classes) rather than batch size, and
        # PyTorch warns it will change "mean" to behave like "batchmean" in a
        # future major release -- this default would then silently change scale.
        assert torch.isclose(loss_mean, loss_sum / logits_s.numel(), atol=1e-6)

    def test_teacher_logits_detached_from_graph(self):
        logits_s = torch.randn(2, 5, requires_grad=True)
        logits_t = torch.randn(2, 5, requires_grad=True)
        loss = LogitsDistillationLoss()(logits_s, logits_t)
        loss.backward()
        assert logits_s.grad is not None
        assert logits_t.grad is None  # soft targets are detached

    def test_float64_dtype_preserved(self):
        logits_s = torch.randn(2, 5, dtype=torch.float64)
        logits_t = torch.randn(2, 5, dtype=torch.float64)
        loss = LogitsDistillationLoss()(logits_s, logits_t)
        assert loss.dtype == torch.float64

    def test_loss_positive_for_differing_distributions(self):
        logits_s = torch.randn(4, 10)
        logits_t = logits_s + torch.randn(4, 10)
        loss = LogitsDistillationLoss(reduction="batchmean")(logits_s, logits_t)
        assert loss.item() > 0.0


class TestMFTLoss:
    def test_corrected_distribution_correct_argmax_hand_computed(self):
        # Teacher distribution p = [0.7, 0.2, 0.1], label = 0 (argmax correct).
        # p_label = 0.7 <= 1 - threshold(0.2) = 0.8, so capped = 0.7 + 0.2 = 0.9.
        # mixin = (0.9 - 0.7) / (1 - 0.7) = 2/3.
        # adjusted = p * (1/3) + one_hot(0) * (2/3)
        #          = [0.7/3 + 2/3, 0.2/3, 0.1/3] = [0.9, 0.0666.., 0.0333..]
        mft = MFTLoss(threshold=0.2)
        p = torch.tensor([[0.7, 0.2, 0.1]])
        corrected = mft._prepare_corrected_distributions(p.log(), torch.tensor([0]), 0.2)
        expected = torch.tensor([[0.9, 0.2 / 3.0, 0.1 / 3.0]])
        assert torch.allclose(corrected, expected, atol=1e-5)

    def test_corrected_distribution_incorrect_argmax_hand_computed(self):
        # Teacher distribution p = [0.6, 0.3, 0.1], label = 1 (argmax 0 is wrong).
        # mixin = (p_argmax - p_label + thr) / (1 + p_argmax - p_label)
        #       = (0.6 - 0.3 + 0.2) / (1 + 0.3) = 0.5 / 1.3 = 5/13.
        # adjusted = p * (8/13) + one_hot(1) * (5/13)
        #          = [4.8/13, 2.4/13 + 5/13, 0.8/13] = [4.8, 7.4, 0.8] / 13
        mft = MFTLoss(threshold=0.2)
        p = torch.tensor([[0.6, 0.3, 0.1]])
        corrected = mft._prepare_corrected_distributions(p.log(), torch.tensor([1]), 0.2)
        expected = torch.tensor([[4.8 / 13.0, 7.4 / 13.0, 0.8 / 13.0]])
        assert torch.allclose(corrected, expected, atol=1e-5)

    def test_corrected_distribution_enforces_threshold_separation(self):
        # After correcting an incorrect-argmax token, the separation between the
        # label prob and the former argmax prob should be exactly the threshold.
        mft = MFTLoss(threshold=0.2)
        p = torch.tensor([[0.6, 0.3, 0.1]])
        corrected = mft._prepare_corrected_distributions(p.log(), torch.tensor([1]), 0.2)
        separation = corrected[0, 1] - corrected[0, 0]
        assert torch.isclose(separation, torch.tensor(0.2), atol=1e-5)

    def test_corrected_distribution_confident_correct_becomes_one_hot(self):
        # p_label = 0.85 > 1 - threshold = 0.8, so the capped target is 1.0 and
        # mixin = (1 - 0.85) / (1 - 0.85) = 1: the distribution collapses to a
        # one-hot on the label.
        mft = MFTLoss(threshold=0.2)
        p = torch.tensor([[0.85, 0.1, 0.05]])
        corrected = mft._prepare_corrected_distributions(p.log(), torch.tensor([0]), 0.2)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(corrected, expected, atol=1e-5)

    def test_corrected_distributions_remain_normalized(self):
        mft = MFTLoss(threshold=0.3)
        logits = torch.randn(16, 11)
        labels = torch.randint(0, 11, (16,))
        corrected = mft._prepare_corrected_distributions(logits, labels, 0.3)
        assert torch.allclose(corrected.sum(dim=-1), torch.ones(16), atol=1e-5)
        assert (corrected >= 0).all()

    def test_zero_threshold_matched_logits_zero_loss(self):
        # With threshold 0 and labels equal to the teacher argmax, the correction
        # is the identity (mixin = 0), so KL(student || teacher) with identical
        # logits is exactly zero.
        logits = torch.randn(4, 6)
        labels = logits.argmax(dim=-1)
        loss = MFTLoss(threshold=0.0)(logits.clone(), logits.clone(), labels)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_forward_matches_manual_kl_on_corrected_targets(self):
        # forward == kl_div(log_softmax(student), corrected_teacher, batchmean) * T^2
        temperature = 2.0
        mft = MFTLoss(temperature=temperature, threshold=0.2)
        logits_s = torch.randn(5, 7)
        logits_t = torch.randn(5, 7)
        labels = torch.randint(0, 7, (5,))

        loss = mft(logits_s, logits_t, labels)

        corrected = mft._prepare_corrected_distributions(logits_t / temperature, labels, 0.2)
        expected = (
            F.kl_div(
                F.log_softmax(logits_s / temperature, dim=-1), corrected, reduction="batchmean"
            )
            * temperature**2
        )
        assert torch.isclose(loss, expected, atol=1e-6)

    def test_forward_flattens_3d_logits(self):
        # (B, S, C) logits are flattened to (B*S, C); labels must already be 1D.
        logits_s = torch.randn(2, 3, 5)
        logits_t = torch.randn(2, 3, 5)
        labels = torch.randint(0, 5, (6,))
        loss = MFTLoss(threshold=0.2)(logits_s, logits_t, labels)
        assert loss.ndim == 0

        flat_loss = MFTLoss(threshold=0.2)(logits_s.reshape(6, 5), logits_t.reshape(6, 5), labels)
        assert torch.isclose(loss, flat_loss, atol=1e-6)

    def test_forward_rejects_2d_labels(self):
        mft = MFTLoss()
        logits = torch.randn(4, 5)
        bad_labels = torch.randint(0, 5, (4, 1))
        with pytest.raises(
            ValueError, match=r"Logits must be a 2D tensor and labels must be a 1D tensor\."
        ):
            mft(logits, logits.clone(), bad_labels)

    def test_prepare_rejects_non_2d_logits(self):
        mft = MFTLoss()
        with pytest.raises(
            ValueError, match=r"Logits must be a 2D tensor and labels must be a 1D tensor\."
        ):
            mft._prepare_corrected_distributions(
                torch.randn(2, 3, 4), torch.randint(0, 4, (6,)), 0.2
            )

    def test_teacher_logits_detached_from_graph(self):
        logits_s = torch.randn(3, 5, requires_grad=True)
        logits_t = torch.randn(3, 5, requires_grad=True)
        labels = torch.randint(0, 5, (3,))
        loss = MFTLoss(threshold=0.2)(logits_s, logits_t, labels)
        loss.backward()
        assert logits_s.grad is not None
        assert logits_t.grad is None  # corrected targets are detached

    def test_temperature_squared_factor(self):
        # Scaling logits by T and passing temperature=T yields identical softened
        # distributions, so the loss differs exactly by the T^2 factor.
        logits_s = torch.randn(4, 6)
        logits_t = torch.randn(4, 6)
        labels = torch.randint(0, 6, (4,))
        loss_t1 = MFTLoss(temperature=1.0, threshold=0.2)(logits_s, logits_t, labels)
        loss_t2 = MFTLoss(temperature=2.0, threshold=0.2)(logits_s * 2.0, logits_t * 2.0, labels)
        assert torch.isclose(loss_t2, 4.0 * loss_t1, atol=1e-5)


class TestMGDLoss:
    def test_align_is_identity_for_matching_channels(self):
        mgd = MGDLoss(num_student_channels=8, num_teacher_channels=8)
        assert isinstance(mgd.align, nn.Identity)

    def test_align_is_1x1_conv_for_mismatched_channels(self):
        mgd = MGDLoss(num_student_channels=3, num_teacher_channels=5)
        assert isinstance(mgd.align, nn.Conv2d)
        assert mgd.align.in_channels == 3
        assert mgd.align.out_channels == 5
        assert mgd.align.kernel_size == (1, 1)

    def test_zeroed_generation_gives_mse_against_zero(self):
        # With lambda_mgd=0 the random mask is all ones (rand() > 1 never holds),
        # and with all generation weights/biases zeroed the reconstructed feature
        # map is exactly zero, so the loss reduces to mse(0, out_t) = mean(out_t^2).
        mgd = MGDLoss(4, 4, lambda_mgd=0.0)
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

    def test_channel_alignment_forward(self):
        mgd = MGDLoss(3, 5)
        out_s = torch.randn(2, 3, 4, 4)
        out_t = torch.randn(2, 5, 4, 4)
        loss = mgd(out_s, out_t)
        assert loss.ndim == 0
        assert loss.item() >= 0.0  # MSE is non-negative

    def test_spatial_mismatch_raises(self):
        mgd = MGDLoss(4, 4)
        with pytest.raises(AssertionError):
            mgd(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 5, 5))

    def test_gradients_flow_to_loss_module_and_student(self):
        mgd = MGDLoss(3, 5, lambda_mgd=0.0)
        out_s = torch.randn(1, 3, 4, 4, requires_grad=True)
        out_t = torch.randn(1, 5, 4, 4)
        loss = mgd(out_s, out_t)
        loss.backward()
        assert out_s.grad is not None
        assert mgd.align.weight.grad is not None
        assert all(p.grad is not None for p in mgd.generation.parameters())

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

    def test_loss_module_parameters_registered(self):
        # The align conv and both generation convs must be registered submodules
        # so DistillationModel can train/convert them with the rest of the model.
        mgd = MGDLoss(3, 5)
        num_params = sum(1 for _ in mgd.parameters())
        # align (w, b) + 2 generation convs (w, b each) = 6 tensors
        assert num_params == 6


class TestStaticLossBalancer:
    def test_default_weight_warns_about_sum_below_one(self):
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            StaticLossBalancer()

    def test_sum_above_one_raises(self):
        with pytest.raises(ValueError, match=r"should be \[0., 1.\]"):
            StaticLossBalancer([0.8, 0.7])

    def test_float_above_one_raises(self):
        with pytest.raises(ValueError, match=r"actual sum 1.5"):
            StaticLossBalancer(1.5)

    def test_negative_sum_raises(self):
        with pytest.raises(ValueError, match=r"should be \[0., 1.\]"):
            StaticLossBalancer([-0.2, -0.3])

    def test_weights_summing_to_one_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            StaticLossBalancer([0.6, 0.4])

    def test_negative_individual_weight_accepted_when_sum_valid(self):
        # NOTE: documents current behavior; arguably a bug because only the sum
        # is validated, so an individually negative KD weight (which subtracts a
        # loss term) passes silently as long as the total lands in [0, 1].
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            balancer = StaticLossBalancer([-0.5, 1.0])
        loss = balancer({"a": torch.tensor(2.0), "b": torch.tensor(3.0)})
        assert torch.isclose(loss, torch.tensor(-0.5 * 2.0 + 1.0 * 3.0))

    def test_int_weight_raises_type_error(self):
        # NOTE: documents current behavior; arguably a bug because the isinstance
        # check only accepts `float`, so an integer weight (e.g. 1) falls through
        # to the list branch and crashes with an unhelpful TypeError from sum().
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            StaticLossBalancer(1)

    def test_hand_computed_aggregate_with_student_loss(self):
        # weights [0.3, 0.4]; kd losses 2.0 and 3.0; student loss 10.0.
        # aggregate = 0.3*2 + 0.4*3 + (1 - 0.7)*10 = 0.6 + 1.2 + 3.0 = 4.8
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            balancer = StaticLossBalancer([0.3, 0.4])
        loss_dict = {
            STUDENT_LOSS_KEY: torch.tensor(10.0),
            "MSELoss_0": torch.tensor(2.0),
            "MSELoss_1": torch.tensor(3.0),
        }
        assert torch.isclose(balancer(loss_dict), torch.tensor(4.8))

    def test_hand_computed_aggregate_scalar_weight(self):
        # float weight 0.25 applied to the single kd loss; student gets 0.75.
        # aggregate = 0.25*8 + 0.75*4 = 2 + 3 = 5
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            balancer = StaticLossBalancer(0.25)
        loss_dict = {
            STUDENT_LOSS_KEY: torch.tensor(4.0),
            "LogitsDistillationLoss_0": torch.tensor(8.0),
        }
        assert torch.isclose(balancer(loss_dict), torch.tensor(5.0))

    def test_kd_only_aggregate_without_student_loss(self):
        balancer = StaticLossBalancer([0.6, 0.4])
        loss_dict = {"L_0": torch.tensor(5.0), "L_1": torch.tensor(10.0)}
        # aggregate = 0.6*5 + 0.4*10 = 7.0 (no student term added)
        assert torch.isclose(balancer(loss_dict), torch.tensor(7.0))

    def test_weights_summing_to_one_zero_out_student_loss(self):
        # With weights summing to exactly 1.0, the student loss contributes
        # (1 - 1) * student = 0 even when provided.
        balancer = StaticLossBalancer([1.0])
        loss_dict = {STUDENT_LOSS_KEY: torch.tensor(100.0), "L_0": torch.tensor(3.0)}
        assert torch.isclose(balancer(loss_dict), torch.tensor(3.0))

    def test_student_loss_reduction_fn_applied(self):
        # An unreduced student loss is reduced by the configured fn before
        # balancing: mean([2, 4]) = 3, aggregate = 0.5*6 + 0.5*3 = 4.5
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            balancer = StaticLossBalancer(0.5)
        balancer.set_student_loss_reduction_fn(lambda x: x.mean())
        loss_dict = {
            STUDENT_LOSS_KEY: torch.tensor([2.0, 4.0]),
            "L_0": torch.tensor(6.0),
        }
        assert torch.isclose(balancer(loss_dict), torch.tensor(4.5))

    def test_weight_count_mismatch_asserts(self):
        balancer = StaticLossBalancer([0.5, 0.5])
        with pytest.raises(AssertionError, match="does not correspond to number of kd losses"):
            balancer({"L_0": torch.tensor(1.0)})

    def test_weights_applied_in_dict_insertion_order(self):
        # The docstring promises weights map to kd losses "in order specified";
        # verify the zip is over dict insertion order.
        balancer = StaticLossBalancer([1.0, 0.0])
        loss_a = {"first": torch.tensor(2.0), "second": torch.tensor(30.0)}
        loss_b = {"second": torch.tensor(30.0), "first": torch.tensor(2.0)}
        assert torch.isclose(balancer(loss_a), torch.tensor(2.0))
        assert torch.isclose(balancer(loss_b), torch.tensor(30.0))

    def test_aggregate_preserves_grad_graph(self):
        with pytest.warns(UserWarning, match="weights do not sum to 1.0"):
            balancer = StaticLossBalancer(0.5)
        kd = torch.tensor(2.0, requires_grad=True)
        student = torch.tensor(3.0, requires_grad=True)
        aggregate = balancer({STUDENT_LOSS_KEY: student, "L_0": kd})
        assert aggregate.requires_grad
        aggregate.backward()
        assert torch.isclose(kd.grad, torch.tensor(0.5))
        assert torch.isclose(student.grad, torch.tensor(0.5))

    def test_input_dict_not_mutated(self):
        balancer = StaticLossBalancer([1.0])
        loss_dict = {STUDENT_LOSS_KEY: torch.tensor(1.0), "L_0": torch.tensor(2.0)}
        balancer(loss_dict)
        assert set(loss_dict) == {STUDENT_LOSS_KEY, "L_0"}


class TestDistillationLossBalancerBase:
    def test_student_loss_key_constant(self):
        assert STUDENT_LOSS_KEY == "student_loss"

    def test_base_class_forward_not_implemented(self):
        # NOTE: documents current behavior; arguably a bug because forward is
        # decorated with abc.abstractmethod but the class lacks an ABC metaclass
        # (nn.Module's metaclass is plain `type`), so instantiation succeeds and
        # abstractness is only enforced at call time via NotImplementedError.
        balancer = DistillationLossBalancer()
        with pytest.raises(NotImplementedError):
            balancer({"L_0": torch.tensor(1.0)})

    def test_set_student_loss_reduction_fn_stores_callable(self):
        balancer = StaticLossBalancer([1.0])
        assert balancer._student_loss_reduction_fn is None
        fn = lambda x: x.sum()  # noqa: E731
        balancer.set_student_loss_reduction_fn(fn)
        assert balancer._student_loss_reduction_fn is fn

    def test_balancer_is_nn_module(self):
        # Balancers must be nn.Modules so DistillationModel can register them.
        assert isinstance(StaticLossBalancer([1.0]), nn.Module)
