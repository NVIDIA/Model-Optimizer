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

"""A registry of model-specific patches to apply post conversion of speculative decoding."""

from collections.abc import Callable

from transformers.utils.quantization_config import CompressedTensorsConfig

all = ["apply_model_patch"]

# Registry to manage model-specific patch functions for easy extensibility
_MODEL_PATCH_REGISTRY: dict[str, Callable] = {}


def register_model_patch(model_name: str):
    """Decorator to register a patch function for a specific model type."""

    def decorator(func: Callable):
        _MODEL_PATCH_REGISTRY[model_name] = func
        return func

    return decorator


def apply_model_patch(module):
    """Apply a registered patch to the given module based on model_type."""
    model_type = module.config.model_type
    if model_type in _MODEL_PATCH_REGISTRY:
        _MODEL_PATCH_REGISTRY[model_type](module)


@register_model_patch("kimi_k2")
def patch_for_kimi_k2(module):
    """Patch for Kimi-K2-Thinking model.

    - Avoid quantizing drafter by updating quantization_config
    - Customizes _compute_ttt_attention_mask behavior
    """
    quant_config = getattr(module.config, "quantization_config", None)
    if isinstance(quant_config, CompressedTensorsConfig):
        quant_config.ignore.append("re:.*eagle_module.*")

    original_func = module._compute_ttt_attention_mask

    def _patched_compute_ttt_attention_mask(self, batch_size, seq_length, ttt_step):
        tensor_mask = original_func(batch_size, seq_length, ttt_step)
        return tensor_mask.repeat(batch_size, 1, 1, 1)

    # Replace the method on the specific instance for Kimi-K2 compatibility
    module._compute_ttt_attention_mask = _patched_compute_ttt_attention_mask.__get__(
        module, type(module)
    )
