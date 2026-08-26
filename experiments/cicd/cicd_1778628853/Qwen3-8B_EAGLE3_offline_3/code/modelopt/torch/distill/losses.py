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

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Different types of distillation losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss

__all__ = ["LogitsDistillationLoss", "MFTLoss", "MGDLoss"]


class LogitsDistillationLoss(Loss):
    """KL-Divergence loss on output logits.

    This function implements the distillation loss found in the paper: https://arxiv.org/abs/1503.02531.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        """Constructor.

        Args:
            temperature: A value used to soften the logits_t and logits_s before computing loss on them.
            reduction: How to reduce the final pointwise loss before returning. Pass ``"none"`` to
                use your own reduction function afterwards, i.e. with loss masks.
        """
        super().__init__()
        self._temperature: float = temperature
        self._reduction: str = reduction

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        """Compute KD loss on student and teacher logits.

        Args:
            logits_s: Student's logits, treated as prediction.
            logits_t: Teacher's logits, treated as label.

        .. note::

            Assumes class logits dimension is last.
        """
        soft_log_probs = F.log_softmax(logits_s / self._temperature, dim=-1)
        soft_targets = F.softmax(logits_t / self._temperature, dim=-1)

        kd_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction=self._reduction)

        if self._reduction == "none":
            # Remove vocab dimension
            kd_loss = kd_loss.sum(dim=-1)

        # Since the magnitudes of the gradients produced by the soft logits scale as 1/(T^2),
        # multiplying them by T^2 ensures that the relative contributions of the logits
        # remain roughly unchanged while experimenting with meta-parameters.
        kd_loss *= self._temperature**2

        return kd_loss


class MFTLoss(Loss):
    """KL-divergence loss with Minifinetuning threshold modification.

    This function implements the distillation loss found in the paper: https://arxiv.org/abs/2506.15702.
    """

    def __init__(
        self, temperature: float = 1.0, threshold: float = 0.2, reduction: str = "batchmean"
    ):
        """Constructor.

        Args:
            temperature: A value used to soften the logits_t and logits_s before computing the MFT loss on them.
            reduction: How to reduce the final pointwise loss before returning. Pass ``"none"`` to
                use your own reduction function afterwards, i.e. with loss masks.
            threshold: A value used to correct the teacher's distribution. It is used to ensure that
                the separation between the correct and incorrect argmax tokens is large enough.
                The value should be in the range [0, 1]. Defaults to 0.2.
        """
        super().__init__()
        self._temperature: float = temperature
        self._reduction: str = reduction
        self._threshold: float = threshold

    def forward(
        self, logits_s: torch.Tensor, logits_t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute KD loss on student and teacher logits.

        Args:
            logits_s: Student's logits, treated as prediction.
            logits_t: Teacher's logits, treated as training target.
            labels: Labels for the ground truth, used to prepare the corrected teacher distributions.

        .. note::

            Assumes class logits dimension is last.
        """
        soft_log_probs = F.log_softmax(logits_s / self._temperature, dim=-1)  # (B, ..., C)
        soft_log_probs = soft_log_probs.view(-1, soft_log_probs.size(-1))  # (new B, C)

        target_logits: torch.Tensor = logits_t / self._temperature  # (B, ..., C)
        target_logits = target_logits.view(-1, target_logits.size(-1))  # (new B, C)
        soft_targets = self._prepare_corrected_distributions(
            target_logits, labels, self._threshold, apply_threshold_to_all=True
        )

        kd_loss = F.kl_div(
            soft_log_probs, soft_targets.detach(), reduction=self._reduction
        )  # shape depends on reduction; "batchmean" would result in a scalar (1,)

        # Since the magnitudes of the gradients produced by the soft logits scale as 1/(T^2),
        # multiplying them by T^2 ensures that the relative contributions of the logits
        # remain roughly unchanged while experimenting with meta-parameters.
        kd_loss *= self._temperature**2

        return kd_loss

    def _prepare_corrected_distributions(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
        apply_threshold_to_all: bool = True,
    ) -> torch.Tensor:
        """Prepare the corrected distributions for MFT loss.

        Args:
            logits: The logits from the teacher model, shape (batch, channels) # e.g. (batch_size * seq_len, vocab_size)
                in case of LMs
            labels: The ground truth labels, shape (batch) # e.g. (batch_size * seq_len) in case of LMs
            threshold: The threshold value for the MFT correction.
            apply_threshold_to_all: If True, apply the threshold correction to all tokens,
                not just the incorrect argmax tokens. Defaults to True.

        Returns:
            A tensor containing the corrected distributions, shape (batch_size * seq_len, vocab_size).
        """
        # Ensure logits is a 2D tensor and labels is a 1D tensor
        if logits.dim() != 2 or labels.dim() != 1:
            raise ValueError("Logits must be a 2D tensor and labels must be a 1D tensor.")
        # logits: (batch, channels)
        # labels: (batch)
        distribution = F.softmax(logits, dim=-1)  # (batch, channels)

        argmax = distribution.argmax(dim=-1)  # (batch,)
        incorrect_argmax = argmax != labels  # (batch,)

        p_argmax = torch.gather(distribution, 1, argmax.unsqueeze(1)).squeeze(1)  # (batch,)
        p_label = torch.gather(distribution, 1, labels.unsqueeze(1)).squeeze(1)  # (batch,)

        # correction of the distribution at the tokens where the argmax is incorrect
        mixin_factor = (p_argmax - p_label + threshold) / (
            1 + p_argmax - p_label + 1e-7
        )  # (batch,)
        adjusted_incorrect_distribution = distribution * (
            1 - mixin_factor.unsqueeze(1)
        )  # (batch, channels)
        _ = adjusted_incorrect_distribution.scatter_add_(
            1, labels.unsqueeze(1), mixin_factor.unsqueeze(1)
        )  # (batch, channels)

        if apply_threshold_to_all:
            # correction of the distribution at the tokens where the argmax is correct but
            #  the separation may not be large enough
            capped_targets = torch.where(
                p_label > 1 - threshold, 1, p_label + threshold
            )  # (batch,)
            mixin_factor = (capped_targets - p_argmax) / (1 - p_argmax + 1e-7)  # (batch,)
            adjusted_correct_distribution = distribution * (
                1 - mixin_factor.unsqueeze(1)
            )  # (batch, channels)
            _ = adjusted_correct_distribution.scatter_add_(
                1, labels.unsqueeze(1), mixin_factor.unsqueeze(1)
            )
        else:
            adjusted_correct_distribution = distribution

        return torch.where(
            incorrect_argmax.unsqueeze(1),
            adjusted_incorrect_distribution,
            adjusted_correct_distribution,
        )  # (batch, channels)


class MGDLoss(Loss):
    """PyTorch version of Masked Generative Distillation.

    This function implements the distillation loss found in the paper: https://arxiv.org/abs/2205.01529.
    """

    def __init__(
        self,
        num_student_channels: int,
        num_teacher_channels: int,
        alpha_mgd: float = 1.0,
        lambda_mgd: float = 0.65,
    ):
        """Constructor.

        Args:
            num_student_channels: Number of channels in the student's feature map.
            num_teacher_channels: Number of channels in the teacher's feature map.
            alpha_mgd: Scalar final loss is multiplied by. Defaults to 1.0.
            lambda_mgd: Masked ratio. Defaults to 0.65.
        """
        super().__init__()
        self._alpha_mgd: float = alpha_mgd
        self._lambda_mgd: float = lambda_mgd

        if num_student_channels != num_teacher_channels:
            self.align = nn.Conv2d(
                num_student_channels,
                num_teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.align = nn.Identity()

        self.generation = nn.Sequential(
            nn.Conv2d(
                num_teacher_channels,
                num_teacher_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_teacher_channels,
                num_teacher_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def _loss_fn(self, out_s: torch.Tensor, out_t: torch.Tensor):
        n, _, h, w = out_t.shape

        mat = torch.rand((n, 1, h, w), device=out_s.device)
        mat = torch.where(mat > 1 - self._lambda_mgd, 0, 1)

        masked_feats = torch.mul(out_s, mat)
        new_feats = self.generation(masked_feats)

        kd_loss = F.mse_loss(new_feats, out_t)

        return kd_loss

    def forward(self, out_s: torch.Tensor, out_t: torch.Tensor):
        """Forward function.

        Args:
            out_s: Student's feature map (shape BxCxHxW).
            out_t: Teacher's feature map (shape BxCxHxW).
        """
        assert out_s.shape[-2:] == out_t.shape[-2:]

        out_s = self.align(out_s)
        loss = self._loss_fn(out_s, out_t) * self._alpha_mgd

        return loss
