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

Directory names are the HF ``config.model_type`` they describe, in snake case
(``NemotronH_Nano_VL_V2`` is ``nemotron_h_nano_vl_v2``); a model whose code ships with
the checkpoint uses its remote-code ``model_type`` and records that as
``ModelSpec.modeling_source``, while a model defined in transformers records the release
its definitions come from as ``ModelSpec.min_transformers_version``. Each package's
``specs.py`` registers one global ``ModelSpec`` (built from the section classes in this
package's own ``specs.py``) at import time. Importing this package registers them all.
Consumers resolve a spec via the registry lookups (``get_spec`` / ``match_moe_block``)
and read its sections.

The per-model file is named for what it holds, not for who reads it: a model's spec is
general model data, and export is only its first consumer.

A package may also hold ``modeling_ptq.py`` and ``modeling_ptq_vllm.py``, the model-specific PTQ
wrappers for transformers and vLLM. They are not imported here: quantization imports a model's
own with ``import_model_modules`` when it first needs them, to convert the model or find its
decoder layers (see ``README.md``).
"""

import importlib
import os
import re

from .moe import is_moe
from .specs import (
    EPSpec,
    ExportSpec,
    ModelSpec,
    MoESpec,
    PTQSpec,
    SpecSection,
    get_ptq_spec,
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
    "EPSpec",
    "ExportSpec",
    "MoESpec",
    "ModelSpec",
    "PTQSpec",
    "SpecSection",
    "get_ptq_spec",
    "get_spec",
    "get_specs",
    "hf_model_type",
    "import_model_modules",
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
    bart,
    deci,
    deepseek,
    deepseek_v3,
    deepseek_v32,
    deepseek_v4,
    diffusion_gemma,
    falcon,
    gemma,
    gemma2,
    gemma3,
    gemma4,
    gemma4_text,
    gpt_oss,
    kimi_k2,
    llama,
    llava_llama,
    llama4,
    llama_nemotron_nano_vl,
    minimax,
    mixtral,
    phimoe,
    nemotron,
    nemotron_h,
    nemotron_h_nano_vl_v2,
    nemotron_nas,
    nemotron_parse,
    minimax_m3_vl,
    llama4_text,
    kimi_vl,
    kimi_k25,
    nemotron_h_omni,
    nemotron_h_nano_omni_reasoning_v3,
    qwen2_moe,
    qwen3,
    qwen3_5_moe,
    qwen3_moe,
    qwen3_next,
    qwen3_vl_moe,
    qwen4_exp,
    step3p5,
    step3p7,
    t5,
    whisper,
)


def import_model_modules(model, module_name: str) -> None:
    """Import ``<package>/<module_name>.py`` for each of ``model``'s model types that has one.

    The model types are the root ``model.config.model_type``, then those of the configs nested in
    it, so a composite model (a VLM, say) also gets its language model's modules. Quantization
    loads a model's PTQ wrappers this way before converting it or finding its decoder layers
    (``modeling_ptq`` for transformers, ``modeling_ptq_vllm`` for vLLM): such a module registers
    them on import, so no model's wrappers are loaded until a model of its type is used.
    """
    models_dir = os.path.dirname(__file__)
    for model_type in _model_types(model):
        package = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", model_type).lower().replace("-", "_")
        if package.isidentifier() and os.path.isfile(
            os.path.join(models_dir, package, f"{module_name}.py")
        ):
            importlib.import_module(f"{__name__}.{package}.{module_name}")


def _model_types(model) -> list[str]:
    """Return the root ``model.config.model_type``, then those of the configs nested in it."""
    model_types: list[str] = []
    configs = [getattr(model, "config", None)]
    seen: set[int] = set()
    while configs:
        config = configs.pop(0)
        model_type = getattr(config, "model_type", None)
        if not isinstance(model_type, str) or id(config) in seen:
            continue
        seen.add(id(config))
        if model_type and model_type not in model_types:
            model_types.append(model_type)
        # A nested config (``text_config``, ``llm_config``, ...) is an attribute of its parent.
        configs.extend(getattr(config, "__dict__", {}).values())
    return model_types
