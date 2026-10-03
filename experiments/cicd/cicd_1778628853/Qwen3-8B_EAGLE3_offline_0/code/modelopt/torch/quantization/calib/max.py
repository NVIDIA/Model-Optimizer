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

"""Calibrator that returns the absolute max of all collected tensors."""

import torch

from .. import utils as quant_utils
from .calibrator import _Calibrator

__all__ = ["MaxCalibrator"]


class MaxCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum value globally.

    Args:
        calib_desc: A MaxCalibDescriptor.
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see :class:`QuantizerAttributeConfig <..config.QuantizerAttributeConfig>`.
        unsigned: A boolean. using unsigned quantization.

    Readonly Properties:
        amaxs: A list of amax. Numpy array is saved as it is likely to be used for some plot.
    """

    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False):
        """Initialize."""
        super().__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []  # shall we have a better name?
        self._calib_amax = None

    @property
    def amaxs(self):
        """Returns the list of amax`s collected so far."""
        return self._amaxs

    @torch.no_grad()
    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        # meta device is used for initialization
        if x.device.type == "meta":
            self._calib_amax = local_amax
            return
        assert not torch.any(torch.isnan(local_amax)), (
            f"detected nan values in amax. nan in original tensor: {torch.any(torch.isnan(x))}"
        )
        assert torch.all(local_amax >= 0), (
            "detected negative values after abs, could be torch or cuda bug"
        )
        assert not torch.any(torch.isinf(local_amax)), (
            f"detected inf values in amax. inf in original tensor: {torch.any(torch.isinf(x))}"
        )
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax = torch.max(self._calib_amax, local_amax)

        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())

    def reset(self):
        """Reset the collected absolute max."""
        self._calib_amax = None

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        return self._calib_amax

    def __str__(self):
        s = "MaxCalibrator("
        s += "track_amax={_track_amax}"
        s += ")"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "MaxCalibrator("
        s += super().__repr__()
        s += " calib_amax={_calib_amax}"
        s += " track_amax={_track_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)
