# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom mapping from Qwen3-VL Hugging Face models to Megatron Core models.

Qwen3-VL differs from Qwen3 in one structural way: language-model weights live
under ``model.language_model.`` instead of ``model.``, while ``lm_head.weight``
remains at the root level.  The mappings below are derived automatically from
the Qwen3 mappings by inserting ``language_model.`` after ``model.`` for every
prefix that starts with ``model.``.

Note: the visual encoder (``model.visual.*``) is intentionally excluded — this
mapping covers only the language-model decoder used for quantization and export.

Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/model.safetensors.index.json
"""

from .mcore_custom import CustomModuleMapping
from .mcore_qwen import qwen3_causal_lm_export, qwen3_causal_lm_import


def _with_language_model_prefix(
    mapping: dict[str, CustomModuleMapping],
) -> dict[str, CustomModuleMapping]:
    """Derive a VL mapping from a base Qwen3 mapping.

    Rewrites every ``target_name_or_prefix`` that starts with ``model.`` to
    ``model.language_model.<rest>``.  Prefixes that do not start with
    ``model.`` (e.g. ``lm_head.``) are left unchanged.
    """
    result = {}
    for key, m in mapping.items():
        prefix = m.target_name_or_prefix
        if prefix.startswith("model."):
            prefix = "model.language_model." + prefix[len("model.") :]
        result[key] = type(m)(target_name_or_prefix=prefix, func_kwargs=m.func_kwargs)
    return result


qwen3vl_causal_lm_import = _with_language_model_prefix(qwen3_causal_lm_import)
qwen3vl_causal_lm_export = _with_language_model_prefix(qwen3_causal_lm_export)
