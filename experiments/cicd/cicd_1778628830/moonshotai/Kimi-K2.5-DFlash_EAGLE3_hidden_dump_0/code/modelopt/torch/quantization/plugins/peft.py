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

"""Support quantization for peft LoRA linear layers."""

from contextlib import contextmanager

import torch.nn.functional as F
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.layer import ParamWrapper

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper

from ..nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from .huggingface import _transposed_quantize

__all__ = []


@QuantModuleRegistry.register({LoraLinear: "LoraLinear"})
class _QuantLoraLinear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = TensorQuantizer()
        self.output_quantizer = TensorQuantizer()

    def _is_compressed_weight(self, weight):
        return isinstance(weight, QTensorWrapper)

    def forward(self, x, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters or adapter_names is not None or self.merged:
            return super().forward(x, *args, **kwargs)

        x = self.input_quantizer(x)
        weight = self.base_layer.weight

        if not self._is_compressed_weight(weight):
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():  # noqa: SIM118
                    continue
                lora_a = self.lora_A[active_adapter]
                lora_b = self.lora_B[active_adapter]
                scaling = self.scaling[active_adapter]

                if not self.use_dora[active_adapter]:
                    weight = weight + ((scaling * lora_b.weight) @ lora_a.weight).to(
                        weight.device, weight.dtype
                    )
                else:
                    raise NotImplementedError("dora not implemented")
            weight = self.weight_quantizer(weight)
            output = F.linear(x, weight, self.base_layer.bias)
        else:
            # TODO: Move this to RealQuantLoraLinear
            # For compressed weights, compute base output and LoRA outputs separately
            base_output = self.base_layer(x)

            # Only compute LoRA outputs if there are active adapters
            if self.active_adapters:
                # Start with zero LoRA output
                lora_output = None

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():  # noqa: SIM118
                        continue
                    lora_a = self.lora_A[active_adapter]
                    lora_b = self.lora_B[active_adapter]
                    scaling = self.scaling[active_adapter]

                    if not self.use_dora[active_adapter]:
                        # Compute LoRA output step by step to maintain gradient flow
                        lora_a_output = F.linear(x, lora_a.weight)
                        lora_b_output = F.linear(lora_a_output, lora_b.weight)
                        adapter_output = scaling * lora_b_output

                        lora_output = (
                            adapter_output if lora_output is None else lora_output + adapter_output
                        )

                output = base_output + lora_output
            else:
                output = base_output

        output = self.output_quantizer(output)
        return output

    def merge(self, *args, **kwargs):
        assert not self._is_compressed_weight(self.base_layer.weight), (
            "We dont support merging for QLoRA yet!"
        )
        super().merge(*args, **kwargs)
        base_layer = self.get_base_layer()
        base_layer.input_quantizer = self.input_quantizer
        base_layer.weight_quantizer = self.weight_quantizer
        base_layer.output_quantizer = self.output_quantizer


@QuantModuleRegistry.register({ParamWrapper: "ParamWrapper"})
class _QuantParamWrapper(QuantModule):
    def _setup(self):
        pass

    @contextmanager
    def _activate_lora(self, active_adapters: list[str]):
        base_layer = self.get_base_layer()
        if not isinstance(base_layer, DynamicModule) or not hasattr(
            base_layer, self.parameter_name + "_weight_quantizer"
        ):
            with super()._activate_lora(active_adapters):
                yield
            return

        delta_weight = None
        for active_adapter in active_adapters:
            if active_adapter not in self.lora_A:
                continue
            delta_weight = (
                self.get_delta_weight(active_adapter)
                if delta_weight is None
                else delta_weight + self.get_delta_weight(active_adapter)
            )

        quantizer = getattr(base_layer, self.parameter_name + "_weight_quantizer")
        with base_layer.reset_dynamic_attributes():
            base_param = getattr(base_layer, self.parameter_name)
            quantized_val = _transposed_quantize(
                base_param if delta_weight is None else base_param + delta_weight, quantizer
            )
            delattr(base_layer, self.parameter_name)
            setattr(base_layer, self.parameter_name, quantized_val)
            try:
                yield
            finally:
                setattr(base_layer, self.parameter_name, base_param)
