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

A package may also hold ``modeling_ptq.py`` and ``modeling_ptq_vllm.py``, the model-specific PTQ
wrappers for transformers and vLLM. They are not imported here; the HF and vLLM quantization
plugins import them with ``import_model_modules`` (see ``README.md``).
"""

import importlib
import importlib.util
import pkgutil

from modelopt.torch.utils import import_plugin

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


def import_model_modules(module_name: str) -> None:
    """Import ``<model_type>/<module_name>.py`` from every model package that has one.

    The quantization plugins load the model-specific PTQ wrappers this way (``modeling_ptq`` for
    transformers, ``modeling_ptq_vllm`` for vLLM), so they name no model: such a module registers
    its wrappers on import, and adding one needs no list entry. Packages are visited in name
    order, so the registration order is the same on every run, and each import is guarded on its
    own, so a module that fails to import leaves the others registered. Packages and modules are
    found through the import system rather than as files on disk, so a zipped or bytecode-only
    install finds them too.
    """
    for package in sorted(name for _, name, is_pkg in pkgutil.iter_modules(__path__) if is_pkg):
        module = f"{__name__}.{package}.{module_name}"
        with import_plugin(f"{package} {module_name}"):
            if importlib.util.find_spec(module) is not None:
                importlib.import_module(module)
