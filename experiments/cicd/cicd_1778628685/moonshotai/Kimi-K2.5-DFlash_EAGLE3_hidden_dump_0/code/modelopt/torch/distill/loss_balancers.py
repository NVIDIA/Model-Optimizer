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

"""Basic loss balancers for Distillation task."""

import abc
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

__all__ = ["DistillationLossBalancer", "StaticLossBalancer"]

STUDENT_LOSS_KEY = "student_loss"


class DistillationLossBalancer(nn.Module):
    """Interface for loss balancers."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._student_loss_reduction_fn = None

    def set_student_loss_reduction_fn(
        self, student_loss_reduction_fn: Callable[[Any], torch.Tensor]
    ):
        """Set student loss reduction function value.

        Needed in special case of loss-reducing the student loss prior to balancing.
        """
        # NOTE: Cannot use setter decorator since ``nn.Module`` intercepts
        # ``__setattr__`` for objects which are also instances of ``nn.Module``.
        self._student_loss_reduction_fn = student_loss_reduction_fn

    @abc.abstractmethod
    def forward(self, loss: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute aggregate loss.

        Args:
            loss: The loss dict to aggregate.
                The keys will be the class name of the loss function applied to obtain the loss,
                suffixed by `_{idx}` for uniqueness. And if a student loss is provided to
                ``mtd.DistillationModel.compute_kd_loss`` then it will have the key
                ``mtd.loss_balancers.STUDENT_LOSS_KEY``. For example, if the ``criterion`` argument
                to ``mtd.convert`` is
                ``{("mod1_s", "mod1_t"): torch.nn.MSELoss(), ("mod2_s", "mod2_t"): torch.nn.MSELoss()}``
                and the student_loss provided to ``mtd.DistillationModel.compute_kd_loss`` is not
                None, then the loss dict here will look like
                ``{"student_loss": torch.tensor(...), "MSELoss_0": torch.tensor(...), "MSELoss_1": torch.tensor(...)}``.

        Returns:
            The total loss after balancing student and kd loss loss components.
        """
        raise NotImplementedError


class StaticLossBalancer(DistillationLossBalancer):
    """Static weights-based loss aggregation of KD losses."""

    def __init__(self, kd_loss_weight: float | list[float] = 0.5):
        """Constructor.

        Args:
            kd_loss_weight: The static weight to be applied to balance the knowledge distillation
                loss and original student loss.
                If it is a float, it would be applied to the sum(KD losses).
                If it is a list, the keys are the KD loss keys, in order specified to the
                ``criterion`` argument, and the weight corresponding to each key is applied to
                the corresponding loss value.
                If the weights do not sum to 1.0, a ``student_loss`` should be passed into
                ``mtd.DistillationModel.compute_kd_loss``, and the weight difference will be applied
                to this loss value.

        Raises:
            ValueError if kd_loss_weight is out of bounds.
        """
        super().__init__()
        if isinstance(kd_loss_weight, float):
            kd_loss_weight = [kd_loss_weight]

        sum_kd_loss_weight = sum(kd_loss_weight)
        if sum_kd_loss_weight < 0.0 or sum_kd_loss_weight > 1.0:
            raise ValueError(
                "The sum of values of kd_loss_weight should be [0., 1.],"
                f" actual sum {sum_kd_loss_weight}"
            )
        elif sum_kd_loss_weight < 1.0:
            warnings.warn(
                "`StaticLossBalancer` weights do not sum to 1.0."
                " Argument `student_loss` should be passed into `DistillationModel.compute_kd_loss`"
            )

        self._kd_loss_weight = kd_loss_weight

    def forward(self, loss: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute aggregate loss.

        Args:
            loss: The loss dict to aggregate.

        Returns:
            The total loss after balancing student and kd loss loss components.
        """
        kd_loss_dict = loss.copy()

        if STUDENT_LOSS_KEY in loss:
            output_loss = kd_loss_dict.pop(STUDENT_LOSS_KEY)
            if self._student_loss_reduction_fn is not None:
                output_loss = self._student_loss_reduction_fn(output_loss)
        else:
            output_loss = None

        assert len(kd_loss_dict) == len(self._kd_loss_weight), (
            "Number of `kd_loss_weight` does not correspond to number of kd losses computed."
        )

        aggregate_loss = sum(
            loss * weight for loss, weight in zip(kd_loss_dict.values(), self._kd_loss_weight)
        )
        if output_loss is not None:
            aggregate_loss += (1.0 - sum(self._kd_loss_weight)) * output_loss

        return aggregate_loss
