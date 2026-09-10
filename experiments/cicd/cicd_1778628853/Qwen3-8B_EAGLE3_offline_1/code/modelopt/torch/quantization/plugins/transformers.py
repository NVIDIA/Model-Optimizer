# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Support quantization for Transformers."""

import torch.nn as nn

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from .custom import CUSTOM_POST_CONVERSION_PLUGINS


def make_deepspeed_compatible(model: nn.Module):
    """Make the model compatible with DeepSpeed."""
    try:
        from deepspeed.runtime.zero.parameter_offload import ZeROOrderedDict
    except ImportError:
        return
    is_deepspeed_zero3_enabled = any(
        hasattr(module, "_parameters") and isinstance(module._parameters, ZeROOrderedDict)
        for module in model.modules()
    )

    if is_deepspeed_zero3_enabled:
        # For zero stage 3, the _parameters is a ZeROOrderedDict, tensor_quantizer._parameters
        # is usually a dict, so we need to check if it is a ZeROOrderedDict if the model is wrapped
        # by deepspeed.
        def _make_deepspeed_compatible(module):
            """Make a module's _parameters DeepSpeed compatible."""
            if isinstance(module, TensorQuantizer) and not isinstance(
                module._parameters, ZeROOrderedDict
            ):
                module._parameters = ZeROOrderedDict(module._parameters)

        # Make all modules DeepSpeed compatible
        for module in model.modules():
            _make_deepspeed_compatible(module)


CUSTOM_POST_CONVERSION_PLUGINS.add(make_deepspeed_compatible)
