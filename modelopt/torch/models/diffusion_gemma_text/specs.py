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

"""DiffusionGemma text-only specs (HF model type ``diffusion_gemma_text``).

The text model ``diffusion_gemma`` nests as its ``text_config``, which is the root type when the
text model is loaded on its own; it shares ``diffusion_gemma``'s PTQ facts.
"""

from ..diffusion_gemma.specs import GENERATE_INCLUDES_PROMPT, SEQUENTIAL_DEVICE_MAP
from ..specs import ModelSpec, PTQSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="diffusion_gemma_text",
        min_transformers_version="5.11",
        ptq_spec=PTQSpec(
            sequential_device_map=SEQUENTIAL_DEVICE_MAP,
            generate_includes_prompt=GENERATE_INCLUDES_PROMPT,
        ),
    )
)
