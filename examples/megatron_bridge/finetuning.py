# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Parameter selection for the Megatron-Bridge distillation example."""

import fnmatch

from torch import nn

from modelopt.torch.quantization.lora import _QuantLoRALinear, enable_quant_input_gradients
from modelopt.torch.quantization.qtensor import QTensorWrapper

__all__ = []


def _configure_trainable_parameters(model: nn.Module, target_parameters: list[str]) -> nn.Module:
    """Train only parameters matching shell-style patterns, retaining fake quantizers.

    Match names relative to the student root, before DDP/optimizer construction.
    Every pattern must match locally; packed weights and unmerged adapters are rejected.
    Other parameters are frozen, including any learned quantization scales.
    """
    if not target_parameters:
        raise ValueError("target_parameters must contain at least one pattern.")
    if any(isinstance(module, _QuantLoRALinear) for module in model.modules()):
        raise ValueError("Full-rank training requires an adapter-free student checkpoint.")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    selected = set()
    for pattern in target_parameters:
        matches = {name for name in parameters if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"No student parameters match {pattern!r} on this model chunk.")
        selected.update(matches)
    selected_ids = {id(parameters[name]) for name in selected}
    for name, parameter in parameters.items():
        if id(parameter) not in selected_ids:
            continue
        if name not in selected:
            raise ValueError(f"Selected parameter is shared with unselected parameter {name!r}.")
        if isinstance(parameter, QTensorWrapper) or not parameter.is_floating_point():
            raise ValueError(
                f"Full-rank training requires an uncompressed floating weight: {name}."
            )
    model.requires_grad_(False)
    for name in selected:
        parameters[name].requires_grad_(True)
    return enable_quant_input_gradients(model)
