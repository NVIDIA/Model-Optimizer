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

"""Falcon PTQ modeling (HF model type ``falcon``)."""

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear
from modelopt.torch.quantization.plugins.custom import CUSTOM_MODEL_PLUGINS

__all__: list[str] = []


try:
    from transformers.models.falcon.modeling_falcon import FalconLinear

    if FalconLinear not in QuantModuleRegistry:
        QuantModuleRegistry.register({FalconLinear: "hf.FalconLinear"})(_QuantLinear)
except ImportError:
    pass


def register_falcon_linears_on_the_fly(model):
    """Register Falcon linear modules as a QUANT_MODULE.

    Certain falcon models (for example, falcon 40b) use remote code, which are loaded dynamically, to build their model.
    Therefore, we need to register the linear on the fly before quantization.
    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        linear_type = type(model.transformer.h[0].self_attention.dense)
        # Create a QuantFalconLinear class on the fly
        if QuantModuleRegistry.get(linear_type) is None:
            QuantModuleRegistry.register({linear_type: linear_type.__name__})(_QuantLinear)


CUSTOM_MODEL_PLUGINS.add(register_falcon_linears_on_the_fly)
