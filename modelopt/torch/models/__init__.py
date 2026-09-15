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

"""Per-model descriptors, one package per HF model type.

Directory names are the HF ``config.model_type`` they describe; a model whose code
ships with the checkpoint uses its remote-code ``model_type`` and records that as
``ModelSpec.modeling_source``, while a model defined in transformers records the release
its definitions come from as ``ModelSpec.min_transformers_version``. Each package's
``specs.py`` registers one global ``ModelSpec`` (built from the section classes in this
package's own ``specs.py``) at import time. Importing this package registers them all.
Consumers resolve a spec via the registry lookups (``get_spec`` / ``match_moe_block``)
and read its sections.

The per-model file is named for what it holds, not for who reads it: a model's spec is
general model data, and export is only its first consumer.
"""

from .moe import is_moe
from .specs import (
    ExportSpec,
    ModelSpec,
    MoESpec,
    SpecSection,
    get_spec,
    get_specs,
    hf_model_type,
    list_all_possible,
    match_class_names,
    match_moe_block,
    match_moe_model,
    register,
)

# Named rather than star-imported so a star-import of this package does not also re-export
# the per-model subpackages imported below for their registration side effect. The list
# cannot drift silently: an entry that no longer exists fails at import, and one that is
# no longer used fails lint.
__all__ = [
    "ExportSpec",
    "MoESpec",
    "ModelSpec",
    "SpecSection",
    "get_spec",
    "get_specs",
    "hf_model_type",
    "is_moe",
    "list_all_possible",
    "match_class_names",
    "match_moe_block",
    "match_moe_model",
    "register",
]

# Importing the model packages registers every spec as a side effect.
from . import (  # isort: skip
    arctic,
    dbrx,
    deepseek,
    deepseek_v3,
    deepseek_v4,
    gemma,
    gemma2,
    gemma3,
    gemma4,
    gemma4_text,
    gpt_oss,
    llama,
    minimax,
    mixtral,
    phimoe,
    nemotron,
    nemotron_h,
    qwen2_moe,
    qwen3,
    qwen3_5_moe,
    qwen3_moe,
    qwen3_next,
)
