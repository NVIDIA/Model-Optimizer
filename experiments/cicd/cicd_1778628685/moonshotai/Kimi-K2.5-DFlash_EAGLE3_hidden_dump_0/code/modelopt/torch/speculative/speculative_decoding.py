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

"""User-facing API for converting a model into a `modelopt.torch.speculative.MedusaModel`."""

from typing import Any

import torch.nn as nn

from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.opt.mode import ModeLike

from .mode import SpeculativeDecodingModeRegistry

__all__ = ["convert"]


def convert(model: nn.Module, mode: ModeLike | dict[str, Any]) -> nn.Module:
    """Main conversion function to turn a base model into a speculative decoding model.

    Args:
        model: The base model to be used.
        mode: A (list of) string(s) or Mode(s) or a list of tuples containing the mode and its
            config indicating the desired mode(s) (and configurations) for the convert
            process. Modes set up the model for different algorithms for model optimization. The
            following modes are available:

            *   :class:`"medusa"<modelopt.torch.speculative.mode.MedusaModeDescriptor>`: The
                ``model`` will be converted into a medusa model with added medusa head.
                The mode's config is described in
                :class:`MedusaConfig<modelopt.torch.speculative.config.MedusaConfig>`.
            *   :class:`"eagle"<modelopt.torch.speculative.mode.EagleModeDescriptor>`: The
                ``model`` will be converted into a eagle model with added eagle weights.
                The mode's config is described in
                :class:`EagleConfig<modelopt.torch.speculative.config.EagleConfig>`.

            If the mode argument is specified as a dictionary, the keys should indicate the mode and
            the values specify the per-mode configuration.

    Returns:
        An instance of :class:`MedusaModel <modelopt.torch.distill.MedusaModel` or
        :class:`EagleModel <modelopt.torch.distill.EagleModel` its subclass.

    """
    if isinstance(mode, dict):
        mode = [(mode["algorithm"], mode["config"])]
    return apply_mode(model, mode=mode, registry=SpeculativeDecodingModeRegistry)
