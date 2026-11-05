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

"""The per-model facts of the Hugging Face PTQ flow, and the accessor that reads them."""

from dataclasses import fields

import pytest

from modelopt.torch.models import PTQSpec, get_ptq_spec, get_specs

# Model type -> the PTQSpec fields it turns on. A model's special-casing in the PTQ flow is a
# registry entry, so adding one fails here until the table names it.
EXPECTED_PTQ_FIELDS = {
    "diffusion_gemma": {"sequential_device_map", "generate_includes_prompt"},
    "llava_llama": {"fsdp2_unsupported"},  # VILA, NVILA and NVILA-Lite
    "bart": {"no_device_map"},
    "t5": {"no_device_map"},
    "deci": {"dtype_kwarg"},
    "nemotron-nas": {"dtype_kwarg"},  # the DeciLM architecture under another model type
    "deepseek": {"preview_unsupported"},
    "deepseek_v3": {"preview_unsupported"},
    "deepseek_v32": {"preview_unsupported"},
    "deepseek_v4": {"preview_unsupported"},
    "kimi_k2": {"preview_unsupported"},  # a DeepSeek-V3 architecture under its own model type
    "kimi_k25": {"preview_unsupported"},  # a VLM wrapping a kimi_k2 language model
    "nemotron_h": {"preview_unsupported"},
    "llama4": {"post_ptq_generation_unsupported"},
    "llama4_text": {"post_ptq_generation_unsupported"},
}


def _turned_on(section: PTQSpec) -> set[str]:
    return {f.name for f in fields(section) if getattr(section, f.name) != f.default}


def test_ptq_sections_match_the_table():
    registered = {s.model_type: _turned_on(s.ptq_spec) for s in get_specs() if s.ptq_spec}
    assert registered == EXPECTED_PTQ_FIELDS


@pytest.mark.parametrize("model_type", [None, "llama", "not_a_registered_model_type"])
def test_a_model_without_a_ptq_section_gets_the_generic_flow(model_type):
    assert get_ptq_spec(model_type) == PTQSpec()
