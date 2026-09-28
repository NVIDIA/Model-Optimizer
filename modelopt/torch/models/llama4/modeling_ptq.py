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

"""Llama4 PTQ modeling (HF model type ``llama4``)."""

import torch

from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    _transposed_quantize,
    _TransposedExpertsCalibMixin,
)

__all__: list[str] = []


class _QuantLlama4TextExperts(_TransposedExpertsCalibMixin, QuantModule):
    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(
            self.gate_up_proj_input_quantizer(hidden_states),
            _transposed_quantize(self.gate_up_proj, self.gate_up_proj_weight_quantizer),
        )
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm(
            self.down_proj_input_quantizer(up * self.act_fn(gate)),
            _transposed_quantize(self.down_proj, self.down_proj_weight_quantizer),
        )
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


try:
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

    if Llama4TextExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({Llama4TextExperts: "hf.Llama4TextExperts"})(
            _QuantLlama4TextExperts
        )
except ImportError:
    pass
