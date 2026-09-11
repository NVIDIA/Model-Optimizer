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

"""Gemma4 text-only specs (HF model type ``gemma4_text``).

A text-only Gemma4 checkpoint's root ``model_type`` is ``gemma4_text`` (following the
gemma3 precedent), while the VLM's is ``gemma4``. The MoE block lives in the text
model either way, so the layout is imported from ``gemma4`` rather than restated --
the two must not drift apart.
"""

from ..gemma4.specs import _GEMMA4_MOE_SPEC
from ..specs import ExportSpec, ModelSpec, register

__all__: list[str] = []

register(
    ModelSpec(
        model_type="gemma4_text",
        min_transformers_version="5.5",
        export_spec=ExportSpec(grouped_expert_export=True),
        moe_spec=_GEMMA4_MOE_SPEC,
    )
)
