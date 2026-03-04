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

"""A registry of target model-specific patches on class HFEagleModel.

The patches are indexed by model type and will be called by HFEagleModel object at the end of modify().
"""

import types
from collections.abc import Callable

import transformers
from packaging.version import Version
from transformers.utils.quantization_config import CompressedTensorsConfig

all = ["apply_model_patch"]

_MODEL_PATCH_REGISTRY: dict[str, Callable] = {}


def register_model_patch(model_type: str):
    """Decorator to register a patch function for a specific model type."""

    def decorator(func: Callable):
        _MODEL_PATCH_REGISTRY[model_type] = func
        return func

    return decorator


def apply_model_patch(module):
    """Apply a registered patch to the given module based on model_type."""
    model_type = getattr(module.config, "model_type", None)
    if model_type in _MODEL_PATCH_REGISTRY:
        _MODEL_PATCH_REGISTRY[model_type](module)


@register_model_patch("kimi_k2")
def patch_for_kimi_k2(module):
    """Patch for Kimi-K2-Thinking as target model.

    - Version check for transformers < 5.0
    - Avoid quantizing drafter by updating quantization_config
    - Repeat attention mask at batch dimension
    """
    if Version(transformers.__version__) >= Version("5.0"):
        raise RuntimeError(
            "Kimi K2 is not supported for transformers >= 5.0. \
            Please install transformers >=4.57, <5.0"
        )

    # Avoid quantizing drafter by updating quantization_config
    quant_config = getattr(module.config, "quantization_config", None)
    if isinstance(quant_config, CompressedTensorsConfig):
        quant_config.quantization_config.ignore.append("re:.*eagle_module.*")

    # Kimi K2 assert attention mask shape as (bsz, 1, qlen, kvlen)
    # https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/main/modeling_deepseek.py#L829
    # So we repeat the attention mask at batch dimension
    original_func = module._compute_ttt_attention_mask

    def _patched_compute_ttt_attention_mask(self, batch_size, seq_length, ttt_step):
        tensor_mask = original_func(batch_size, seq_length, ttt_step)
        return tensor_mask.repeat(batch_size, 1, 1, 1)

    module._compute_ttt_attention_mask = types.MethodType(
        _patched_compute_ttt_attention_mask, module
    )
