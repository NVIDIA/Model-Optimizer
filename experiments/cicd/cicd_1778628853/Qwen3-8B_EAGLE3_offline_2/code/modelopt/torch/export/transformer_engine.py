# Adapted from https://github.com/huggingface/accelerate/blob/c2f193a25c5449c0e34408bbbffa61ce420b00f6/
# src/accelerate/utils/transformer_engine.py

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""Convert the Model Optimizer quantized model to the transformer_engine."""

import torch
from torch import nn

__all__ = ["convert_to_transformer_engine"]


def _convert_model(model):
    """Recursively converts the linear layers to their `transformers_engine` counterpart."""
    import transformer_engine.pytorch as te

    with te.fp8_model_init(enabled=True):
        for name, module in model.named_children():
            if type(module).__name__ == "QuantLinear":
                # Return early if the linear layer weights are not multiples of 16
                if any(p % 16 != 0 for p in module.weight.shape):
                    return
                has_bias = module.bias is not None
                te_module = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=has_bias,
                    params_dtype=module.weight.dtype,
                )
                te_module.weight.copy_(module.weight)
                if has_bias:
                    te_module.bias.copy_(module.bias)

                input_quantizer = module.input_quantizer
                quant_max = input_quantizer.maxbound
                input_scale_inv = input_quantizer.export_amax().float()[0] / quant_max
                input_scale = 1 / input_scale_inv

                weight_quantizer = module.weight_quantizer
                quant_max = weight_quantizer.maxbound
                weight_scale_inv = weight_quantizer.export_amax().float()[0] / quant_max
                weight_scale = 1 / weight_scale_inv

                # Update the TE layers with the Model Optimizer scaling factors
                te_module.fp8_meta["scaling_fwd"].scale_inv[0].copy_(input_scale_inv)
                te_module.fp8_meta["scaling_fwd"].scale_inv[1].copy_(weight_scale_inv)
                te_module.fp8_meta["scaling_fwd"].scale[0].copy_(input_scale)
                te_module.fp8_meta["scaling_fwd"].scale[1].copy_(weight_scale)

                setattr(model, name, te_module)
            else:
                _convert_model(module)


def convert_to_transformer_engine(model: nn.Module):
    """Converts the `Model Optimizer` quantized model to the `transformers_engine`."""
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast

    with torch.no_grad():
        _convert_model(model)

    fp8_recipe = te_recipe.DelayedScaling()
    model.forward = fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)(model.forward)

    return model
